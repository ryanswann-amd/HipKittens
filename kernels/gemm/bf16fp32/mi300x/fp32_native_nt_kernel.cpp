// Native FP32 NT GEMM — best version: 65 TF, occ 7, KS=16
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
constexpr int BS=128, KS=16, NW=8, WS=64, NT=NW*WS, T=16, DK=4, RB=BS/4, KI=KS/DK;
__device__ inline void mfma(float (&D)[4], float A, float B, const float (&C)[4]) {
    typedef __attribute__((__vector_size__(4*sizeof(float)))) float f4;
    *(f4*)D = __builtin_amdgcn_mfma_f32_16x16x4f32(A, B, *(f4*)C, 0, 0, 0);
}
__global__ __launch_bounds__(NT, 4)
void fp32_gemm_nt(float* __restrict__ C, const float* __restrict__ A,
                  const float* __restrict__ B, int M, int N, int K) {
    __shared__ float As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4, mr=ln%16, mk=ln/16;
    float c0[2][2][4]={}, c1[2][2][4]={};
    for(int i=threadIdx.x; i<BS*KS; i+=NT) { int r=i/KS,c=i%KS; As[r][c]=A[(tm*BS+r)*K+c]; Bs[r][c]=B[(tn*BS+r)*K+c]; }
    __syncthreads();
    for(int kt=0; kt<K/KS-1; kt++) {
        float4 ab, bb; int idx=threadIdx.x;
        if(idx<BS*KS/4) { int r=idx/(KS/4),c4=(idx%(KS/4))*4; ab=*(const float4*)&A[(tm*BS+r)*K+(kt+1)*KS+c4]; bb=*(const float4*)&B[(tn*BS+r)*K+(kt+1)*KS+c4]; }
        #pragma unroll
        for(int ki=0;ki<KI;ki++) { int koff=ki*DK+mk;
            float a0=As[wr*RB+0*T+mr][koff],a1=As[wr*RB+1*T+mr][koff],a2=As[(wr+2)*RB+0*T+mr][koff],a3=As[(wr+2)*RB+1*T+mr][koff];
            float b0=Bs[wc*RB+0*T+mr][koff],b1=Bs[wc*RB+1*T+mr][koff];
            mfma(c0[0][0],a0,b0,c0[0][0]); mfma(c0[0][1],a0,b1,c0[0][1]); mfma(c0[1][0],a1,b0,c0[1][0]); mfma(c0[1][1],a1,b1,c0[1][1]);
            mfma(c1[0][0],a2,b0,c1[0][0]); mfma(c1[0][1],a2,b1,c1[0][1]); mfma(c1[1][0],a3,b0,c1[1][0]); mfma(c1[1][1],a3,b1,c1[1][1]);
        }
        asm volatile("s_waitcnt vmcnt(0)"); __syncthreads();
        if(idx<BS*KS/4) { int r=idx/(KS/4),c4=(idx%(KS/4))*4; *(float4*)&As[r][c4]=ab; *(float4*)&Bs[r][c4]=bb; }
        __syncthreads();
    }
    #pragma unroll
    for(int ki=0;ki<KI;ki++) { int koff=ki*DK+mk;
        float a0=As[wr*RB+0*T+mr][koff],a1=As[wr*RB+1*T+mr][koff],a2=As[(wr+2)*RB+0*T+mr][koff],a3=As[(wr+2)*RB+1*T+mr][koff];
        float b0=Bs[wc*RB+0*T+mr][koff],b1=Bs[wc*RB+1*T+mr][koff];
        mfma(c0[0][0],a0,b0,c0[0][0]); mfma(c0[0][1],a0,b1,c0[0][1]); mfma(c0[1][0],a1,b0,c0[1][0]); mfma(c0[1][1],a1,b1,c0[1][1]);
        mfma(c1[0][0],a2,b0,c1[0][0]); mfma(c1[0][1],a2,b1,c1[0][1]); mfma(c1[1][0],a3,b0,c1[1][0]); mfma(c1[1][1],a3,b1,c1[1][1]);
    }
    for(int bi=0;bi<2;bi++) for(int bj=0;bj<2;bj++) { int col=tn*BS+wc*RB+bj*T+mr;
        for(int k=0;k<4;k++) { C[(tm*BS+wr*RB+bi*T+mk*4+k)*N+col]=c0[bi][bj][k]; C[(tm*BS+(wr+2)*RB+bi*T+mk*4+k)*N+col]=c1[bi][bj][k]; } }
}
PYBIND11_MODULE(hk_fp32_native_nt, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int M=sa[0].cast<int>(), K=sa[1].cast<int>(), N=sb[0].cast<int>();
        fp32_gemm_nt<<<dim3((M/BS)*(N/BS)), dim3(NT), 0, (hipStream_t)0>>>((float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const float*)A.attr("data_ptr")().cast<uint64_t>(), (const float*)B.attr("data_ptr")().cast<uint64_t>(), M, N, K);
    });
}
