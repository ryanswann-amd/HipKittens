#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 128
#endif
#ifndef K_STEP_VAL
#define K_STEP_VAL 32
#endif

#include "kittens.cuh"
#include <pybind11/pybind11.h>
using namespace kittens;

constexpr int BS=BLOCK_SIZE_VAL, KS=K_STEP_VAL, NW=8, WS=64, NT=NW*WS;
constexpr int T=32, DK=16, RB=T, KI=KS/DK;
typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16v;

__device__ inline void mfma_fp8(float (&D)[16], long A, long B, const float (&C)[16]) {
    *(f16v*)D = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(A, B, *(f16v*)C, 0, 0, 0);
}

__global__ __launch_bounds__(NT, 4)
void fp8_gemm_nt(float* __restrict__ C, const char* __restrict__ A,
                 const char* __restrict__ B, int M, int N, int K) {
    __shared__ char As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4;
    float c0[16]={}, c1[16]={};
    const char* Ab=A+tm*BS*K; const char* Bb=B+tn*BS*K;
    i32x4 a_sr=make_srsrc(Ab,BS*K,K), b_sr=make_srsrc(Bb,BS*K,K);

    // Prologue
    for(int idx=threadIdx.x;idx<BS*KS/16;idx+=NT) {
        int r=idx/(KS/16),c16=(idx%(KS/16))*16;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+c16,0,0);
        __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+c16,0,0);
        *(float4*)&As[r][c16]=*(float4*)&ra; *(float4*)&Bs[r][c16]=*(float4*)&rb;
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for(int kt=0;kt<K/KS-1;kt++) {
        // Prefetch
        float4 ab,bb; int idx=threadIdx.x;
        if(idx<BS*KS/16) { int r=idx/(KS/16),c16=(idx%(KS/16))*16;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+(kt+1)*KS+c16,0,0);
            __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+(kt+1)*KS+c16,0,0);
            ab=*(float4*)&ra; bb=*(float4*)&rb; }

        // Pipelined inner loop: reads for ki+1 during ki's MFMAs
        int a_row0=wr*RB+(ln%32), a_row1=(wr+2)*RB+(ln%32), b_row=wc*RB+(ln%32);

        // Pre-read ki=0
        int k0=(ln/32)*8;
        long a0_0=*(const long*)&As[a_row0][k0];
        long a1_0=*(const long*)&As[a_row1][k0];
        long b_0=*(const long*)&Bs[b_row][k0];

        #pragma unroll
        for(int ki=0; ki<KI-1; ki++) {
            int k_next=(ki+1)*DK+(ln/32)*8;
            // Wait for current ki reads, then fire MFMA + issue next reads
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mfma_fp8(c0, a0_0, b_0, c0);
            // Issue reads for ki+1 during the 8-cycle MFMA latency
            long a0_n=*(const long*)&As[a_row0][k_next];
            long a1_n=*(const long*)&As[a_row1][k_next];
            long b_n=*(const long*)&Bs[b_row][k_next];
            mfma_fp8(c1, a1_0, b_0, c1);
            __builtin_amdgcn_s_setprio(0);
            // Rotate
            a0_0=a0_n; a1_0=a1_n; b_0=b_n;
        }
        // Last ki
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma_fp8(c0, a0_0, b_0, c0);
        mfma_fp8(c1, a1_0, b_0, c1);
        __builtin_amdgcn_s_setprio(0);

        // Store prefetch
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        if(idx<BS*KS/16) { int r=idx/(KS/16),c16=(idx%(KS/16))*16;
            *(float4*)&As[r][c16]=ab; *(float4*)&Bs[r][c16]=bb; }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Last K tile
    int a_row0=wr*RB+(ln%32), a_row1=(wr+2)*RB+(ln%32), b_row=wc*RB+(ln%32);
    #pragma unroll
    for(int ki=0;ki<KI;ki++) {
        int k_base=ki*DK+(ln/32)*8;
        long a0=*(const long*)&As[a_row0][k_base];
        long a1=*(const long*)&As[a_row1][k_base];
        long bv=*(const long*)&Bs[b_row][k_base];
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma_fp8(c0, a0, bv, c0);
        mfma_fp8(c1, a1, bv, c1);
        __builtin_amdgcn_s_setprio(0);
    }

    int col=tn*BS+wc*RB+(ln%32);
    for(int g=0;g<4;g++) for(int j=0;j<4;j++) {
        int lr=g*8+(ln/32)*4+j;
        C[(tm*BS+wr*RB+lr)*N+col]=c0[g*4+j];
        C[(tm*BS+(wr+2)*RB+lr)*N+col]=c1[g*4+j]; }
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp8_native_nt
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int M=sa[0].cast<int>(), K=sa[1].cast<int>(), N=sb[0].cast<int>();
        fp8_gemm_nt<<<dim3((M/BS)*(N/BS)),dim3(NT),0,(hipStream_t)0>>>((float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const char*)A.attr("data_ptr")().cast<uint64_t>(),(const char*)B.attr("data_ptr")().cast<uint64_t>(),M,N,K);
    });
}
