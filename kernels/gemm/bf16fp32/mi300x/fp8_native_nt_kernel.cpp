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

__global__ __launch_bounds__(NT, 4)
void fp8_gemm_nt(float* __restrict__ C, const char* __restrict__ A,
                 const char* __restrict__ B, int M, int N, int K) {
    __shared__ char As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4;
    f16v c0={}, c1={};
    const char* Ab=A+tm*BS*K; const char* Bb=B+tn*BS*K;
    i32x4 a_sr=make_srsrc(Ab,BS*K,K), b_sr=make_srsrc(Bb,BS*K,K);

    int a_row0=wr*RB+(ln%32), a_row1=(wr+2)*RB+(ln%32), b_row=wc*RB+(ln%32);
    int lk=(ln/32)*8;

    // Compute LDS byte addresses for each subtile position
    uint32_t lds_base_a = (uint32_t)(uintptr_t)&As[0][0];
    uint32_t lds_base_b = (uint32_t)(uintptr_t)&Bs[0][0];
    uint32_t addr_a0_0 = lds_base_a + a_row0*KS + 0*DK + lk;
    uint32_t addr_a1_0 = lds_base_a + a_row1*KS + 0*DK + lk;
    uint32_t addr_b_0  = lds_base_b + b_row*KS + 0*DK + lk;
    uint32_t addr_a0_1 = lds_base_a + a_row0*KS + 1*DK + lk;
    uint32_t addr_a1_1 = lds_base_a + a_row1*KS + 1*DK + lk;
    uint32_t addr_b_1  = lds_base_b + b_row*KS + 1*DK + lk;

    for(int idx=threadIdx.x;idx<BS*KS/16;idx+=NT) {
        int r=idx/(KS/16),c16=(idx%(KS/16))*16;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+c16,0,0);
        __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+c16,0,0);
        *(float4*)&As[r][c16]=*(float4*)&ra; *(float4*)&Bs[r][c16]=*(float4*)&rb;
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    for(int kt=0;kt<K/KS-1;kt++) {
        float4 ab,bb; int idx=threadIdx.x;
        if(idx<BS*KS/16) { int r=idx/(KS/16),c16=(idx%(KS/16))*16;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+(kt+1)*KS+c16,0,0);
            __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+(kt+1)*KS+c16,0,0);
            ab=*(float4*)&ra; bb=*(float4*)&rb; }

        // FULL ASM INNER LOOP: all reads + MFMAs in one block
        long r0, r1, r2, r3, r4, r5;
        asm volatile(
            // Issue ALL 6 reads at once
            "ds_read_b64 %2, %8\n"      // r0 = a0_ki0
            "ds_read_b64 %3, %9\n"      // r1 = a1_ki0
            "ds_read_b64 %4, %10\n"     // r2 = b_ki0
            "ds_read_b64 %5, %11\n"     // r3 = a0_ki1
            "ds_read_b64 %6, %12\n"     // r4 = a1_ki1
            "ds_read_b64 %7, %13\n"     // r5 = b_ki1
            // Wait for first 3 reads (ki=0), 3 still pending
            "s_waitcnt lgkmcnt(3)\n"
            "s_setprio 1\n"
            // ki=0 MFMAs (ki=1 reads complete during 8-cycle latency)
            "v_mfma_f32_32x32x16_fp8_fp8 %0, %2, %4, %0\n"   // c0 += a0_ki0 * b_ki0
            "v_mfma_f32_32x32x16_fp8_fp8 %1, %3, %4, %1\n"   // c1 += a1_ki0 * b_ki0
            // Wait for ki=1 reads
            "s_waitcnt lgkmcnt(0)\n"
            // ki=1 MFMAs
            "v_mfma_f32_32x32x16_fp8_fp8 %0, %5, %7, %0\n"   // c0 += a0_ki1 * b_ki1
            "v_mfma_f32_32x32x16_fp8_fp8 %1, %6, %7, %1\n"   // c1 += a1_ki1 * b_ki1
            "s_setprio 0\n"
            : "+v"(c0), "+v"(c1),
              "=v"(r0), "=v"(r1), "=v"(r2), "=v"(r3), "=v"(r4), "=v"(r5)
            : "v"(addr_a0_0), "v"(addr_a1_0), "v"(addr_b_0),
              "v"(addr_a0_1), "v"(addr_a1_1), "v"(addr_b_1)
            : "memory");

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        if(idx<BS*KS/16) { int r=idx/(KS/16),c16=(idx%(KS/16))*16;
            *(float4*)&As[r][c16]=ab; *(float4*)&Bs[r][c16]=bb; }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Last tile
    {
        long r0,r1,r2,r3,r4,r5;
        asm volatile(
            "ds_read_b64 %2, %8\n"
            "ds_read_b64 %3, %9\n"
            "ds_read_b64 %4, %10\n"
            "ds_read_b64 %5, %11\n"
            "ds_read_b64 %6, %12\n"
            "ds_read_b64 %7, %13\n"
            "s_waitcnt lgkmcnt(3)\n"
            "s_setprio 1\n"
            "v_mfma_f32_32x32x16_fp8_fp8 %0, %2, %4, %0\n"
            "v_mfma_f32_32x32x16_fp8_fp8 %1, %3, %4, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_f32_32x32x16_fp8_fp8 %0, %5, %7, %0\n"
            "v_mfma_f32_32x32x16_fp8_fp8 %1, %6, %7, %1\n"
            "s_setprio 0\n"
            : "+v"(c0), "+v"(c1),
              "=v"(r0),"=v"(r1),"=v"(r2),"=v"(r3),"=v"(r4),"=v"(r5)
            : "v"(addr_a0_0),"v"(addr_a1_0),"v"(addr_b_0),
              "v"(addr_a0_1),"v"(addr_a1_1),"v"(addr_b_1)
            : "memory");
    }

    float* cp = (float*)&c0;
    float* cp1 = (float*)&c1;
    int col=tn*BS+wc*RB+(ln%32);
    for(int g=0;g<4;g++) for(int j=0;j<4;j++) {
        int lr=g*8+(ln/32)*4+j;
        C[(tm*BS+wr*RB+lr)*N+col]=cp[g*4+j];
        C[(tm*BS+(wr+2)*RB+lr)*N+col]=cp1[g*4+j]; }
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
