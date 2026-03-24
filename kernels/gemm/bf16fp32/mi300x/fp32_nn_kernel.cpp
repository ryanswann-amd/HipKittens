// FP32 NN GEMM: C = A @ B where A is MxK, B is KxN
// In-kernel transpose: loads B from KxN global, writes TRANSPOSED to shared.
// FP32 elements are 4 bytes → use float4 (ds_write_b128) for efficient writes.

#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 128
#endif
#include "kittens.cuh"
#include <pybind11/pybind11.h>
using namespace kittens;

constexpr int BS=BLOCK_SIZE_VAL, KS=16, NW=8, WS=64, NT=NW*WS, T=16, DK=4, RB=BS/4, KI=KS/DK;
typedef __attribute__((__vector_size__(16))) float f4v;

// Transpose load B: B_global[k][n] → Bs[n][k] in shared
// Groups: K_GROUP=4 K-rows × N_GROUP=4 N-elements → write float4 per N position
__device__ void load_B_nn_fp32(
    float Bs[][KS],
    const float* __restrict__ B_base,
    int N_stride,  // = N (floats between K-rows)
    int k_offset)
{
    constexpr int N_GROUP = 4;
    constexpr int K_GROUP = 4;
    constexpr int N_BATCHES = BS / N_GROUP;     // 32
    constexpr int K_BATCHES = KS / K_GROUP;     // 4
    constexpr int TOTAL = K_BATCHES * N_BATCHES; // 128

    #pragma unroll 1
    for (int batch = threadIdx.x; batch < TOTAL; batch += NT) {
        int kb = batch / N_BATCHES;
        int nb = batch % N_BATCHES;
        int k = kb * K_GROUP;
        int n = nb * N_GROUP;

        // Load 4 K-rows × 4 N-elements (float4 each, coalesced along N)
        const float* base = B_base + (k_offset + k) * N_stride + n;
        float4 r0 = *(const float4*)(base);
        float4 r1 = *(const float4*)(base + N_stride);
        float4 r2 = *(const float4*)(base + 2*N_stride);
        float4 r3 = *(const float4*)(base + 3*N_stride);

        // Pack 4 K-values per N position and write float4 (ds_write_b128)
        // Bs[n+i][k:k+4] is contiguous in shared → float4 write
        float4 col0 = {r0.x, r1.x, r2.x, r3.x};
        float4 col1 = {r0.y, r1.y, r2.y, r3.y};
        float4 col2 = {r0.z, r1.z, r2.z, r3.z};
        float4 col3 = {r0.w, r1.w, r2.w, r3.w};
        *(float4*)&Bs[n+0][k] = col0;
        *(float4*)&Bs[n+1][k] = col1;
        *(float4*)&Bs[n+2][k] = col2;
        *(float4*)&Bs[n+3][k] = col3;
    }
}

__global__ __launch_bounds__(NT, 4)
void fp32_gemm_nn(float* __restrict__ C, const float* __restrict__ A,
                  const float* __restrict__ B, int M, int N, int K) {
    __shared__ float As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4, mr=ln%16, mk=ln/16;

    float c0[2][2][4]={}, c1[2][2][4]={};
    float* as=&As[0][0]; float* bs=&Bs[0][0];
    int ao[4]={(wr*RB+0*T+mr)*KS,(wr*RB+1*T+mr)*KS,((wr+2)*RB+0*T+mr)*KS,((wr+2)*RB+1*T+mr)*KS};
    int bo[2]={(wc*RB+0*T+mr)*KS,(wc*RB+1*T+mr)*KS};

    // A: MxK — buffer descriptor same as NT
    const float* A_base = A + tm * BS * K;
    i32x4 a_sr = make_srsrc(A_base, BS*K*4, K*4);
    // B: KxN — use raw pointer for transposed loads
    const float* B_ptr = B;
    int B_n_off = tn * BS;

    // Prologue: load A normally, load B with transpose
    {constexpr int LPR=KS/4, TOTAL=BS*LPR;
    for(int idx=threadIdx.x;idx<TOTAL;idx+=NT){int r=idx/LPR,c4=(idx%LPR)*4;int bo2=(r*K+c4)*4;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,bo2,0,0);
        *(float4*)&As[r][c4]=*(float4*)&ra;}}
    load_B_nn_fp32(Bs, B_ptr + B_n_off, N, 0);
    __syncthreads();

    for(int kt=0; kt<K/KS-1; kt++) {
        // Prefetch A
        float4 ab;
        int idx=threadIdx.x;
        if(idx<BS*KS/4){int r=idx/(KS/4),c4=(idx%(KS/4))*4;int bo2=(r*K+(kt+1)*KS+c4)*4;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,bo2,0,0);
            ab=*(float4*)&ra;}

        // Compute (same as NT)
        #pragma unroll
        for(int ki=0;ki<KI;ki++){int koff=ki*DK+mk;
            float a0=as[ao[0]+koff],a1=as[ao[1]+koff],a2=as[ao[2]+koff],a3=as[ao[3]+koff];
            float bv0=bs[bo[0]+koff],bv1=bs[bo[1]+koff];
            asm volatile("s_waitcnt lgkmcnt(2)");
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][0]):"v"(a0),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][1]):"v"(a0),"v"(bv1));
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][0]):"v"(a1),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][1]):"v"(a1),"v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][0]):"v"(a2),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][1]):"v"(a2),"v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][0]):"v"(a3),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][1]):"v"(a3),"v"(bv1));}

        asm volatile("s_waitcnt vmcnt(0)");
        __syncthreads();
        if(idx<BS*KS/4){int r=idx/(KS/4),c4=(idx%(KS/4))*4;*(float4*)&As[r][c4]=ab;}
        load_B_nn_fp32(Bs, B_ptr + B_n_off, N, (kt+1)*KS);
        __syncthreads();
    }

    // Last tile
    #pragma unroll
    for(int ki=0;ki<KI;ki++){int koff=ki*DK+mk;
        float a0=as[ao[0]+koff],a1=as[ao[1]+koff],a2=as[ao[2]+koff],a3=as[ao[3]+koff];
        float bv0=bs[bo[0]+koff],bv1=bs[bo[1]+koff];
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][0]):"v"(a0),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][1]):"v"(a0),"v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][0]):"v"(a1),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][1]):"v"(a1),"v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][0]):"v"(a2),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][1]):"v"(a2),"v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][0]):"v"(a3),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][1]):"v"(a3),"v"(bv1));}

    // Store (same as NT)
    for(int bi=0;bi<2;bi++) for(int bj=0;bj<2;bj++){int col=tn*BS+wc*RB+bj*T+mr;
        for(int k=0;k<4;k++){C[(tm*BS+wr*RB+bi*T+mk*4+k)*N+col]=c0[bi][bj][k];C[(tm*BS+(wr+2)*RB+bi*T+mk*4+k)*N+col]=c1[bi][bj][k];}}
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp32_nn
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int Msz=sa[0].cast<int>(), Ksz=sa[1].cast<int>(), Nsz=sb[1].cast<int>();
        fp32_gemm_nn<<<dim3((Nsz/BS)*(Msz/BS)),dim3(NT),0,(hipStream_t)0>>>(
            (float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const float*)A.attr("data_ptr")().cast<uint64_t>(),
            (const float*)B.attr("data_ptr")().cast<uint64_t>(), Msz, Nsz, Ksz);
    }, "FP32 NN GEMM: C = A @ B (in-kernel transpose)");
}
