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
constexpr int T=32, DK=16, KI=KS/DK;
constexpr int NROW = BS / (T * 2);
constexpr int NCOL = BS / (T * 4);
constexpr int NACC = NROW * NCOL;
static_assert(NROW * 2 * T == BS, "BS must be divisible by 2*T for row tiling");
static_assert(NCOL * 4 * T == BS, "BS must be divisible by 4*T for col tiling");
typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16v;

__global__ __launch_bounds__(NT, 2)
void fp8_gemm_nt(float* __restrict__ C, const char* __restrict__ A,
                 const char* __restrict__ B, int M, int N, int K) {
    __shared__ char As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4;
    int lk=(ln/32)*8;

    int ar[NROW], br[NCOL];
    #pragma unroll
    for(int i=0;i<NROW;i++) ar[i]=(wr+i*2)*T+(ln%32);
    #pragma unroll
    for(int i=0;i<NCOL;i++) br[i]=(wc+i*4)*T+(ln%32);

    f16v acc[NACC];
    #pragma unroll
    for(int i=0;i<NACC;i++) acc[i]={};

    const char* Ab=A+tm*BS*K; const char* Bb=B+tn*BS*K;
    i32x4 a_sr=make_srsrc(Ab,BS*K,K), b_sr=make_srsrc(Bb,BS*K,K);
    const int total_loads = (BS * KS) / 16;

    // Prologue: load first K tile
    for(int i=threadIdx.x; i<total_loads; i+=NT) {
        int r=i/(KS/16), c16=(i%(KS/16))*16;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+c16,0,0);
        __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+c16,0,0);
        *(float4*)&As[r][c16]=*(float4*)&ra;
        *(float4*)&Bs[r][c16]=*(float4*)&rb;
    }
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    const int num_tiles = K/KS;
    for(int kt=0; kt<num_tiles-1; kt++) {
        // 1. Issue async global loads for NEXT tile → register buffer
        float4 a_buf, b_buf;
        {
            int i = threadIdx.x;
            if(i < total_loads) {
                int r=i/(KS/16), c16=(i%(KS/16))*16;
                __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+(kt+1)*KS+c16,0,0);
                __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+(kt+1)*KS+c16,0,0);
                a_buf=*(float4*)&ra; b_buf=*(float4*)&rb;
            }
        }

        // 2. Compute on current shared tile (overlapped with global loads in flight)
        #pragma unroll
        for(int ki=0; ki<KI; ki++) {
            int koff=ki*DK+lk;
            long av[NROW], bv[NCOL];
            #pragma unroll
            for(int r=0;r<NROW;r++) av[r]=*(const long*)&As[ar[r]][koff];
            #pragma unroll
            for(int c=0;c<NCOL;c++) bv[c]=*(const long*)&Bs[br[c]][koff];
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            #pragma unroll
            for(int c=0;c<NCOL;c++) {
                #pragma unroll
                for(int r=0;r<NROW;r++) {
                    acc[r*NCOL+c]=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av[r],bv[c],acc[r*NCOL+c],0,0,0);
                }
            }
            __builtin_amdgcn_s_setprio(0);
        }

        // 3. Wait for loads, store to shared, barrier
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        {
            int i = threadIdx.x;
            if(i < total_loads) {
                int r=i/(KS/16), c16=(i%(KS/16))*16;
                *(float4*)&As[r][c16]=a_buf;
                *(float4*)&Bs[r][c16]=b_buf;
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Last tile: compute only (no prefetch needed)
    #pragma unroll
    for(int ki=0; ki<KI; ki++) {
        int koff=ki*DK+lk;
        long av[NROW], bv[NCOL];
        #pragma unroll
        for(int r=0;r<NROW;r++) av[r]=*(const long*)&As[ar[r]][koff];
        #pragma unroll
        for(int c=0;c<NCOL;c++) bv[c]=*(const long*)&Bs[br[c]][koff];
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for(int c=0;c<NCOL;c++) {
            #pragma unroll
            for(int r=0;r<NROW;r++) {
                acc[r*NCOL+c]=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(av[r],bv[c],acc[r*NCOL+c],0,0,0);
            }
        }
        __builtin_amdgcn_s_setprio(0);
    }

    // Store results
    #pragma unroll
    for(int c=0;c<NCOL;c++) {
        int col_base = tn*BS + (wc+c*4)*T + (ln%32);
        #pragma unroll
        for(int r=0;r<NROW;r++) {
            float* cp = (float*)&acc[r*NCOL+c];
            int row_base = tm*BS + (wr+r*2)*T;
            #pragma unroll
            for(int g=0;g<4;g++) {
                #pragma unroll
                for(int j=0;j<4;j++) {
                    int lr = g*8 + (ln/32)*4 + j;
                    C[(row_base+lr)*N + col_base] = cp[g*4+j];
                }
            }
        }
    }
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
