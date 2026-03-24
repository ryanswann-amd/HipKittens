// FP8 TN GEMM via fused transpose+NT: C = A^T @ B where B is KxN
//
// Phase 1: Transpose B from KxN → NxK (byte-level, works for any FP8 format)
// Phase 2: FP8 NT GEMM kernel on transposed data
//
// Same approach as BF16 fused kernels but with 1-byte element transpose.

#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 256
#endif
#ifndef K_STEP_VAL
#define K_STEP_VAL 32
#endif
#include "kittens.cuh"
#include <pybind11/pybind11.h>
using namespace kittens;

constexpr int BS=BLOCK_SIZE_VAL, KS=K_STEP_VAL, NW=8, WS=64, NT_THR=NW*WS;
constexpr int T=32, DK=16, KI=KS/DK;
constexpr int NROW = BS / (T * 2);
constexpr int NCOL = BS / (T * 4);
constexpr int NACC = NROW * NCOL;
static_assert(NROW * 2 * T == BS, "BS must be divisible by 2*T");
static_assert(NCOL * 4 * T == BS, "BS must be divisible by 4*T");
static_assert(KI == 2, "Optimized for KI=2");
typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16v;

// ===================== Phase 1: Byte Transpose Kernel =====================
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ __launch_bounds__(TILE_DIM * BLOCK_ROWS)
void transpose_fp8(
    char* __restrict__ dst,
    const char* __restrict__ src,
    int rows, int cols)
{
    // Transpose rows×cols → cols×rows (byte-level)
    __shared__ char tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols)
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows)
            dst[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ===================== Phase 2: FP8 NT GEMM Kernel =====================
// (Self-contained copy of fp8_native_nt_kernel.cpp's compute kernel)

#define MFMA(acc, a, b) acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, acc, 0, 0, 0)

__global__ __launch_bounds__(NT_THR, 2)
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

    for(int i=threadIdx.x; i<total_loads; i+=NT_THR) {
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
        float4 a_buf, b_buf;
        {int i=threadIdx.x; if(i<total_loads) {
            int r=i/(KS/16),c16=(i%(KS/16))*16;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+(kt+1)*KS+c16,0,0);
            __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,r*K+(kt+1)*KS+c16,0,0);
            a_buf=*(float4*)&ra; b_buf=*(float4*)&rb;}}

        #pragma unroll
        for(int ki=0;ki<KI;ki++) {
            int koff=ki*DK+lk;
            long av[NROW], bv[NCOL];
            #pragma unroll
            for(int r=0;r<NROW;r++) av[r]=*(const long*)&As[ar[r]][koff];
            #pragma unroll
            for(int c=0;c<NCOL;c++) bv[c]=*(const long*)&Bs[br[c]][koff];
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            #pragma unroll
            for(int c=0;c<NCOL;c++)
                #pragma unroll
                for(int r=0;r<NROW;r++)
                    MFMA(acc[r*NCOL+c], av[r], bv[c]);
            __builtin_amdgcn_s_setprio(0);
        }

        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        {int i=threadIdx.x; if(i<total_loads) {
            int r=i/(KS/16),c16=(i%(KS/16))*16;
            *(float4*)&As[r][c16]=a_buf; *(float4*)&Bs[r][c16]=b_buf;}}
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for(int ki=0;ki<KI;ki++) {
        int koff=ki*DK+lk;
        long av[NROW], bv[NCOL];
        #pragma unroll
        for(int r=0;r<NROW;r++) av[r]=*(const long*)&As[ar[r]][koff];
        #pragma unroll
        for(int c=0;c<NCOL;c++) bv[c]=*(const long*)&Bs[br[c]][koff];
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for(int c=0;c<NCOL;c++)
            #pragma unroll
            for(int r=0;r<NROW;r++)
                MFMA(acc[r*NCOL+c], av[r], bv[c]);
        __builtin_amdgcn_s_setprio(0);
    }

    #pragma unroll
    for(int c=0;c<NCOL;c++) {
        int col_base = tn*BS + (wc+c*4)*T + (ln%32);
        #pragma unroll
        for(int r=0;r<NROW;r++) {
            float* cp = (float*)&acc[r*NCOL+c];
            int row_base = tm*BS + (wr+r*2)*T;
            #pragma unroll
            for(int g=0;g<4;g++)
                #pragma unroll
                for(int j=0;j<4;j++)
                    C[(row_base+g*8+(ln/32)*4+j)*N + col_base] = cp[g*4+j];
        }
    }
}

#undef MFMA

// ===================== Combined Dispatch =====================
#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp8_tn_fused
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        int Ksz = sa[0].cast<int>();  // A is KxM
        int Msz = sa[1].cast<int>();
        int Nsz = sb[1].cast<int>();  // B is KxN

        uint64_t a_ptr = A.attr("data_ptr")().cast<uint64_t>();
        uint64_t b_ptr = B.attr("data_ptr")().cast<uint64_t>();
        uint64_t c_ptr = C.attr("data_ptr")().cast<uint64_t>();

        // Static workspaces for transposed A (MxK) and B (NxK)
        static char* ws_a = nullptr;
        static char* ws_b = nullptr;
        static size_t ws_a_size = 0, ws_b_size = 0;
        size_t need_a = (size_t)Msz * Ksz;
        size_t need_b = (size_t)Nsz * Ksz;
        if (need_a > ws_a_size) { if (ws_a) hipFree(ws_a); hipMalloc(&ws_a, need_a); ws_a_size = need_a; }
        if (need_b > ws_b_size) { if (ws_b) hipFree(ws_b); hipMalloc(&ws_b, need_b); ws_b_size = need_b; }

        dim3 tb(TILE_DIM, BLOCK_ROWS);
        // Phase 1a: Transpose A from KxM to MxK
        dim3 tg_a((Msz + TILE_DIM - 1) / TILE_DIM, (Ksz + TILE_DIM - 1) / TILE_DIM);
        transpose_fp8<<<tg_a, tb>>>(ws_a, (const char*)a_ptr, Ksz, Msz);
        // Phase 1b: Transpose B from KxN to NxK
        dim3 tg_b((Nsz + TILE_DIM - 1) / TILE_DIM, (Ksz + TILE_DIM - 1) / TILE_DIM);
        transpose_fp8<<<tg_b, tb>>>(ws_b, (const char*)b_ptr, Ksz, Nsz);

        // Phase 2: FP8 NT GEMM with transposed A (MxK) and transposed B (NxK)
        fp8_gemm_nt<<<dim3((Nsz/BS)*(Msz/BS)), dim3(NT_THR), 0, (hipStream_t)0>>>(
            (float*)c_ptr, (const char*)ws_a, (const char*)ws_b, Msz, Nsz, Ksz);
    }, "FP8 TN GEMM via transpose+NT: C = A^T @ B");
}
