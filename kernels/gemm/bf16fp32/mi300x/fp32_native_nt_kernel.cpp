#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 128
#endif
// Native FP32 NT GEMM — buffer_load + inline ASM MFMA scheduling
// Uses make_srsrc + llvm_amdgcn_raw_buffer_load_b128 for global reads
// (same path as BF16 HK kernel) and inline ASM for MFMA waitcnt control.

#include "kittens.cuh"  // For make_srsrc, llvm_amdgcn_raw_buffer_load_b128
#include <pybind11/pybind11.h>
using namespace kittens;

constexpr int BS=BLOCK_SIZE_VAL, KS=16, NW=8, WS=64, NT=NW*WS, T=16, DK=4, RB=BS/4, KI=KS/DK;
typedef __attribute__((__vector_size__(16))) float f4v;

__global__ __launch_bounds__(NT, 4)
void fp32_gemm_nt(float* __restrict__ C, const float* __restrict__ A,
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

    // Buffer descriptors for A and B
    const float* A_base = A + tm * BS * K;
    const float* B_base = B + tn * BS * K;
    int A_bytes = BS * K * sizeof(float);
    int B_bytes = BS * K * sizeof(float);
    int A_stride = K * sizeof(float);
    int B_stride = K * sizeof(float);
    i32x4 a_srsrc = make_srsrc(A_base, A_bytes, A_stride);
    i32x4 b_srsrc = make_srsrc(B_base, B_bytes, B_stride);

    // Prologue: load first tile via buffer_load
    {
        constexpr int EPL = 4;  // elements per float4 load
        constexpr int LPR = KS / EPL;  // loads per row = 4
        constexpr int TOTAL = BS * LPR;
        for(int idx=threadIdx.x; idx<TOTAL; idx+=NT) {
            int r = idx / LPR, c4 = (idx % LPR) * EPL;
            int byte_off = (r * K + c4) * sizeof(float);
            __uint128_t ra = llvm_amdgcn_raw_buffer_load_b128(a_srsrc, byte_off, 0, 0);
            __uint128_t rb = llvm_amdgcn_raw_buffer_load_b128(b_srsrc, byte_off, 0, 0);
            *(float4*)&As[r][c4] = *reinterpret_cast<float4*>(&ra);
            *(float4*)&Bs[r][c4] = *reinterpret_cast<float4*>(&rb);
        }
    }
    __syncthreads();

    for(int kt=0; kt<K/KS-1; kt++) {
        // Prefetch next tile via buffer_load
        float4 ab, bb;
        int idx=threadIdx.x;
        if(idx<BS*KS/4) {
            int r=idx/(KS/4), c4=(idx%(KS/4))*4;
            int byte_off = (r * K + (kt+1)*KS + c4) * sizeof(float);
            __uint128_t ra = llvm_amdgcn_raw_buffer_load_b128(a_srsrc, byte_off, 0, 0);
            __uint128_t rb = llvm_amdgcn_raw_buffer_load_b128(b_srsrc, byte_off, 0, 0);
            ab = *reinterpret_cast<float4*>(&ra);
            bb = *reinterpret_cast<float4*>(&rb);
        }

        // Compute with inline ASM MFMA + controlled waitcnt
        #pragma unroll
        for(int ki=0; ki<KI; ki++) {
            int koff = ki*DK + mk;
            float a0=as[ao[0]+koff], a1=as[ao[1]+koff], a2=as[ao[2]+koff], a3=as[ao[3]+koff];
            float bv0=bs[bo[0]+koff], bv1=bs[bo[1]+koff];
            asm volatile("s_waitcnt lgkmcnt(2)");
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][0]):"v"(a0),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][1]):"v"(a0),"v"(bv1));
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][0]):"v"(a1),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][1]):"v"(a1),"v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][0]):"v"(a2),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][1]):"v"(a2),"v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][0]):"v"(a3),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][1]):"v"(a3),"v"(bv1));
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __syncthreads();
        if(idx<BS*KS/4) { int r=idx/(KS/4),c4=(idx%(KS/4))*4; *(float4*)&As[r][c4]=ab; *(float4*)&Bs[r][c4]=bb; }
        __syncthreads();
    }

    // Last tile
    #pragma unroll
    for(int ki=0;ki<KI;ki++) { int koff=ki*DK+mk;
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
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][1]):"v"(a3),"v"(bv1));
    }

    // Store
    for(int bi=0;bi<2;bi++) for(int bj=0;bj<2;bj++) { int col=tn*BS+wc*RB+bj*T+mr;
        for(int k=0;k<4;k++) { C[(tm*BS+wr*RB+bi*T+mk*4+k)*N+col]=c0[bi][bj][k]; C[(tm*BS+(wr+2)*RB+bi*T+mk*4+k)*N+col]=c1[bi][bj][k]; } }
}


#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp32_native_nt
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int M=sa[0].cast<int>(), K=sa[1].cast<int>(), N=sb[0].cast<int>();
        fp32_gemm_nt<<<dim3((M/BS)*(N/BS)), dim3(NT), 0, (hipStream_t)0>>>((float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const float*)A.attr("data_ptr")().cast<uint64_t>(), (const float*)B.attr("data_ptr")().cast<uint64_t>(), M, N, K);
    });
}
