// Native FP32 NT GEMM — inline ASM inner loop for precise waitcnt control
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

constexpr int BS=128, KS=16, NW=8, WS=64, NT=NW*WS, T=16, DK=4, RB=BS/4, KI=KS/DK;

__global__ __launch_bounds__(NT, 4)
void fp32_gemm_nt(float* __restrict__ C, const float* __restrict__ A,
                  const float* __restrict__ B, int M, int N, int K) {
    __shared__ float As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4, mr=ln%16, mk=ln/16;

    // Accumulators in VGPRs
    float c0[2][2][4]={}, c1[2][2][4]={};
    const int nk=K/KS;

    // Compute shared memory offsets for A and B subtiles
    // A subtile row offsets (4 variants: wr*RB+{0,1}*T, (wr+2)*RB+{0,1}*T)
    int a_off[4];
    a_off[0] = (wr*RB + 0*T + mr) * KS;    // As[wr*RB+0*T+mr][koff]
    a_off[1] = (wr*RB + 1*T + mr) * KS;
    a_off[2] = ((wr+2)*RB + 0*T + mr) * KS;
    a_off[3] = ((wr+2)*RB + 1*T + mr) * KS;
    int b_off[2];
    b_off[0] = (wc*RB + 0*T + mr) * KS;
    b_off[1] = (wc*RB + 1*T + mr) * KS;

    // Prologue
    for(int i=threadIdx.x; i<BS*KS; i+=NT) { int r=i/KS,c=i%KS; As[r][c]=A[(tm*BS+r)*K+c]; Bs[r][c]=B[(tn*BS+r)*K+c]; }
    __syncthreads();

    float* as_base = &As[0][0];
    float* bs_base = &Bs[0][0];

    for(int kt=0; kt<nk-1; kt++) {
        // Prefetch
        float4 ab, bb; int idx=threadIdx.x;
        if(idx<BS*KS/4) { int r=idx/(KS/4),c4=(idx%(KS/4))*4;
            ab=*(const float4*)&A[(tm*BS+r)*K+(kt+1)*KS+c4];
            bb=*(const float4*)&B[(tn*BS+r)*K+(kt+1)*KS+c4]; }

        // Inner loop with controlled waitcnt
        // Read all values for ki=0, then fire MFMAs overlapped with reads for ki=1
        #pragma unroll
        for(int ki=0; ki<KI; ki++) {
            int koff = ki*DK + mk;
            // Issue 6 ds_read for this ki (A: 4 reads, B: 2 reads)
            float a0 = as_base[a_off[0] + koff];
            float a1 = as_base[a_off[1] + koff];
            float a2 = as_base[a_off[2] + koff];
            float a3 = as_base[a_off[3] + koff];
            float bv0 = bs_base[b_off[0] + koff];
            float bv1 = bs_base[b_off[1] + koff];

            // Wait only for the reads needed for the FIRST 2 MFMAs
            // (a0 and bv0 must be ready, a2 can still be loading)
            asm volatile("s_waitcnt lgkmcnt(2)");

            // Fire MFMAs interleaved — the later MFMAs can use data
            // that arrives while earlier MFMAs execute (8-cycle latency)
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c0[0][0][0]))
                : "v"(a0), "v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c0[0][1][0]))
                : "v"(a0), "v"(bv1));

            asm volatile("s_waitcnt lgkmcnt(0)");  // now wait for a1, a2, a3

            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c0[1][0][0]))
                : "v"(a1), "v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c0[1][1][0]))
                : "v"(a1), "v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c1[0][0][0]))
                : "v"(a2), "v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c1[0][1][0]))
                : "v"(a2), "v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c1[1][0][0]))
                : "v"(a3), "v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0"
                : "+v"(*((__attribute__((__vector_size__(16))) float*)&c1[1][1][0]))
                : "v"(a3), "v"(bv1));
        }

        asm volatile("s_waitcnt vmcnt(0)"); __syncthreads();
        if(idx<BS*KS/4) { int r=idx/(KS/4),c4=(idx%(KS/4))*4; *(float4*)&As[r][c4]=ab; *(float4*)&Bs[r][c4]=bb; }
        __syncthreads();
    }

    // Last tile (same but no prefetch)
    #pragma unroll
    for(int ki=0; ki<KI; ki++) {
        int koff = ki*DK + mk;
        float a0=as_base[a_off[0]+koff], a1=as_base[a_off[1]+koff];
        float a2=as_base[a_off[2]+koff], a3=as_base[a_off[3]+koff];
        float bv0=bs_base[b_off[0]+koff], bv1=bs_base[b_off[1]+koff];
        typedef __attribute__((__vector_size__(16))) float f4v;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c0[0][0][0]) : "v"(a0), "v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c0[0][1][0]) : "v"(a0), "v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c0[1][0][0]) : "v"(a1), "v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c0[1][1][0]) : "v"(a1), "v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c1[0][0][0]) : "v"(a2), "v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c1[0][1][0]) : "v"(a2), "v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c1[1][0][0]) : "v"(a3), "v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0, %1, %2, %0" : "+v"(*(f4v*)&c1[1][1][0]) : "v"(a3), "v"(bv1));
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
