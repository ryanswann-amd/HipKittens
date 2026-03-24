#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 128
#endif
// Native FP8 NT GEMM — optimized with buffer_load + sched_group_barrier
// v_mfma_f32_32x32x16_fp8_fp8: 32768 FLOPs per instruction, K=16
// Each thread provides 8 packed fp8 (long = 64 bits) for A and B.

#include "kittens.cuh"
#include <pybind11/pybind11.h>
using namespace kittens;

constexpr int BS=BLOCK_SIZE_VAL, KS=32, NW=8, WS=64, NT=NW*WS;
constexpr int T=32, DK=16, RB=T, KI=KS/DK;  // KI=2

__device__ inline void mfma_fp8(float (&D)[16], long A, long B, const float (&C)[16]) {
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16;
    *(f16*)D = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(A, B, *(f16*)C, 0, 0, 0);
}

__global__ __launch_bounds__(NT, 2)
void fp8_gemm_nt(float* __restrict__ C, const char* __restrict__ A,
                 const char* __restrict__ B, int M, int N, int K) {
    __shared__ char As[BS][KS], Bs[BS][KS];  // FP8: 1 byte each
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4;

    float c0[16]={}, c1[16]={};  // 32x32 accum: 16 fp32 per thread × 2 blocks
    
    // Buffer descriptors for FP8 data (1 byte per element)
    const char* A_base=A+tm*BS*K; const char* B_base=B+tn*BS*K;
    i32x4 a_sr=make_srsrc(A_base,BS*K,K), b_sr=make_srsrc(B_base,BS*K,K);
    
    // Prologue: load via buffer_load (16 bytes = 16 fp8 per float4)
    for(int idx=threadIdx.x; idx<BS*KS/16; idx+=NT) {
        int r=idx/(KS/16), c16=(idx%(KS/16))*16;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,(r*K+c16),0,0);
        __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,(r*K+c16),0,0);
        *(float4*)&As[r][c16]=*(float4*)&ra; *(float4*)&Bs[r][c16]=*(float4*)&rb;
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for(int kt=0;kt<K/KS-1;kt++) {
        // Prefetch next FP8 tile
        float4 ab,bb; int idx=threadIdx.x;
        if(idx<BS*KS/16) { int r=idx/(KS/16),c16=(idx%(KS/16))*16;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,(r*K+(kt+1)*KS+c16),0,0);
            __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,(r*K+(kt+1)*KS+c16),0,0);
            ab=*(float4*)&ra; bb=*(float4*)&rb; }

        // Compute: KI=2 MFMA calls per K_STEP
        #pragma unroll
        for(int ki=0;ki<KI;ki++) {
            // Each thread reads 8 bytes (long) from shared: row + K offset
            int a_row0 = wr * RB + (ln % 32);
            int a_row1 = (wr + 2) * RB + (ln % 32);
            int b_row = wc * RB + (ln % 32);
            int k_base = ki * DK + (ln / 32) * 8;  // lane/32 selects which 8 of the 16 K values

            asm volatile("s_waitcnt lgkmcnt(0)"); long a0 = *(const long*)&As[a_row0][k_base];
            long a1 = *(const long*)&As[a_row1][k_base];
            long bv = *(const long*)&Bs[b_row][k_base];

            __builtin_amdgcn_sched_group_barrier(0x2, 3, 0);  // 3 DS reads
            __builtin_amdgcn_s_setprio(1); mfma_fp8(c0, a0, bv, c0);
            mfma_fp8(c1, a1, bv, c1); __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_group_barrier(0x8, 2, 0);  // 2 MFMAs
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        if(idx<BS*KS/16) { int r=idx/(KS/16),c16=(idx%(KS/16))*16;
            *(float4*)&As[r][c16]=ab; *(float4*)&Bs[r][c16]=bb; }
        __builtin_amdgcn_s_barrier();
    }

    // Last tile
    #pragma unroll
    for(int ki=0;ki<KI;ki++) {
        int a_row0=wr*RB+(ln%32), a_row1=(wr+2)*RB+(ln%32), b_row=wc*RB+(ln%32);
        int k_base=ki*DK+(ln/32)*8;
        long a0=*(const long*)&As[a_row0][k_base], a1=*(const long*)&As[a_row1][k_base];
        long bv=*(const long*)&Bs[b_row][k_base];
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1); mfma_fp8(c0, a0, bv, c0);
        mfma_fp8(c1, a1, bv, c1); __builtin_amdgcn_s_setprio(0);
    }

    // Store: 32x32 output, 16 values per thread
    int col=tn*BS+wc*RB+(ln%32);
    for(int g=0;g<4;g++) for(int j=0;j<4;j++) {
        int lr=g*8+(ln/32)*4+j;
        C[(tm*BS+wr*RB+lr)*N+col]=c0[g*4+j];
        C[(tm*BS+(wr+2)*RB+lr)*N+col]=c1[g*4+j];
    }
}

PYBIND11_MODULE(hk_fp8_native_nt, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int M=sa[0].cast<int>(), K=sa[1].cast<int>(), N=sb[0].cast<int>();
        fp8_gemm_nt<<<dim3((M/BS)*(N/BS)),dim3(NT),0,(hipStream_t)0>>>((float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const char*)A.attr("data_ptr")().cast<uint64_t>(),(const char*)B.attr("data_ptr")().cast<uint64_t>(),M,N,K);
    });
}
