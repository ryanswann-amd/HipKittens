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
static_assert(KI == 2, "This kernel is optimized for KI=2 (KS=32, DK=16)");
typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16v;

#define MFMA(acc, a, b) acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, acc, 0, 0, 0)

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

    // Prologue
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
        // 1. Issue async global loads for NEXT tile
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

        // 2. INTERLEAVED compute: ki=0 reads → ki=0 MFMAs with ki=1 reads → ki=1 MFMAs
        {
            int k0 = 0*DK + lk;
            int k1 = 1*DK + lk;

            // Phase A: Issue all ki=0 shared reads
            long av0[NROW], bv0[NCOL];
            #pragma unroll
            for(int r=0;r<NROW;r++) av0[r]=*(const long*)&As[ar[r]][k0];
            #pragma unroll
            for(int c=0;c<NCOL;c++) bv0[c]=*(const long*)&Bs[br[c]][k0];
            __builtin_amdgcn_sched_barrier(0);

            // Phase B: Wait for ki=0 reads, then ki=0 MFMAs interleaved with ki=1 reads
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);

            // ki=0 col 0 MFMAs + start ki=1 A reads
            MFMA(acc[0*NCOL+0], av0[0], bv0[0]);
            long av1_0 = *(const long*)&As[ar[0]][k1];  // ki=1 read
            if constexpr (NROW > 1) {
                MFMA(acc[1*NCOL+0], av0[1], bv0[0]);
                long av1_tmp = *(const long*)&As[ar[1]][k1];
                if constexpr (NROW > 2) {
                    MFMA(acc[2*NCOL+0], av0[2], bv0[0]);
                    long av1_2 = *(const long*)&As[ar[2]][k1];
                    if constexpr (NROW > 3) {
                        MFMA(acc[3*NCOL+0], av0[3], bv0[0]);
                        long av1_3 = *(const long*)&As[ar[3]][k1];
                        // ki=0 col 1 MFMAs + ki=1 B reads
                        if constexpr (NCOL > 1) {
                            long bv1_0 = *(const long*)&Bs[br[0]][k1];
                            MFMA(acc[0*NCOL+1], av0[0], bv0[1]);
                            long bv1_1 = *(const long*)&Bs[br[1]][k1];
                            MFMA(acc[1*NCOL+1], av0[1], bv0[1]);
                            MFMA(acc[2*NCOL+1], av0[2], bv0[1]);
                            MFMA(acc[3*NCOL+1], av0[3], bv0[1]);
                            __builtin_amdgcn_s_setprio(0);
                            __builtin_amdgcn_sched_barrier(0);

                            // Phase C: ki=1 MFMAs (reads already in flight)
                            asm volatile("s_waitcnt lgkmcnt(0)");
                            __builtin_amdgcn_s_setprio(1);
                            MFMA(acc[0*NCOL+0], av1_0, bv1_0);
                            MFMA(acc[1*NCOL+0], av1_tmp, bv1_0);
                            MFMA(acc[2*NCOL+0], av1_2, bv1_0);
                            MFMA(acc[3*NCOL+0], av1_3, bv1_0);
                            MFMA(acc[0*NCOL+1], av1_0, bv1_1);
                            MFMA(acc[1*NCOL+1], av1_tmp, bv1_1);
                            MFMA(acc[2*NCOL+1], av1_2, bv1_1);
                            MFMA(acc[3*NCOL+1], av1_3, bv1_1);
                        } else {
                            __builtin_amdgcn_s_setprio(0);
                            __builtin_amdgcn_sched_barrier(0);
                            long bv1_0 = *(const long*)&Bs[br[0]][k1];
                            asm volatile("s_waitcnt lgkmcnt(0)");
                            __builtin_amdgcn_s_setprio(1);
                            MFMA(acc[0*NCOL+0], av1_0, bv1_0);
                            MFMA(acc[1*NCOL+0], av1_tmp, bv1_0);
                            MFMA(acc[2*NCOL+0], av1_2, bv1_0);
                            MFMA(acc[3*NCOL+0], av1_3, bv1_0);
                        }
                    } else {
                        // NROW==3 path (not used currently)
                        __builtin_amdgcn_s_setprio(0);
                        long bv1_0 = *(const long*)&Bs[br[0]][k1];
                        asm volatile("s_waitcnt lgkmcnt(0)");
                        __builtin_amdgcn_s_setprio(1);
                        MFMA(acc[0], av1_0, bv1_0);
                        MFMA(acc[1], av1_tmp, bv1_0);
                        MFMA(acc[2], av1_2, bv1_0);
                    }
                } else {
                    // NROW==2 path (BS=128)
                    if constexpr (NCOL > 1) {
                        long bv1_0 = *(const long*)&Bs[br[0]][k1];
                        MFMA(acc[0*NCOL+1], av0[0], bv0[1]);
                        long bv1_1 = *(const long*)&Bs[br[1]][k1];
                        MFMA(acc[1*NCOL+1], av0[1], bv0[1]);
                        __builtin_amdgcn_s_setprio(0);
                        __builtin_amdgcn_sched_barrier(0);
                        asm volatile("s_waitcnt lgkmcnt(0)");
                        __builtin_amdgcn_s_setprio(1);
                        MFMA(acc[0*NCOL+0], av1_0, bv1_0);
                        MFMA(acc[1*NCOL+0], av1_tmp, bv1_0);
                        MFMA(acc[0*NCOL+1], av1_0, bv1_1);
                        MFMA(acc[1*NCOL+1], av1_tmp, bv1_1);
                    } else {
                        // NROW==2, NCOL==1 (BS=128, simplest case)
                        __builtin_amdgcn_s_setprio(0);
                        __builtin_amdgcn_sched_barrier(0);
                        long bv1_0 = *(const long*)&Bs[br[0]][k1];
                        asm volatile("s_waitcnt lgkmcnt(0)");
                        __builtin_amdgcn_s_setprio(1);
                        MFMA(acc[0], av1_0, bv1_0);
                        MFMA(acc[1], av1_tmp, bv1_0);
                    }
                }
            } else {
                // NROW==1 (BS=64, not used)
                __builtin_amdgcn_s_setprio(0);
                long bv1_0 = *(const long*)&Bs[br[0]][k1];
                asm volatile("s_waitcnt lgkmcnt(0)");
                MFMA(acc[0], av1_0, bv1_0);
            }
            __builtin_amdgcn_s_setprio(0);
        }

        // 3. Wait for globals, store to shared, barrier
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

    // Last tile: same interleaved pattern but no prefetch
    {
        int k0 = 0*DK + lk;
        int k1 = 1*DK + lk;

        long av0[NROW], bv0[NCOL];
        #pragma unroll
        for(int r=0;r<NROW;r++) av0[r]=*(const long*)&As[ar[r]][k0];
        #pragma unroll
        for(int c=0;c<NCOL;c++) bv0[c]=*(const long*)&Bs[br[c]][k0];
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);

        // Same interleaved MFMA pattern
        #pragma unroll
        for(int c=0;c<NCOL;c++) {
            #pragma unroll
            for(int r=0;r<NROW;r++) {
                MFMA(acc[r*NCOL+c], av0[r], bv0[c]);
            }
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);

        long av1[NROW], bv1[NCOL];
        #pragma unroll
        for(int r=0;r<NROW;r++) av1[r]=*(const long*)&As[ar[r]][k1];
        #pragma unroll
        for(int c=0;c<NCOL;c++) bv1[c]=*(const long*)&Bs[br[c]][k1];

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for(int c=0;c<NCOL;c++) {
            #pragma unroll
            for(int r=0;r<NROW;r++) {
                MFMA(acc[r*NCOL+c], av1[r], bv1[c]);
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

#undef MFMA

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
