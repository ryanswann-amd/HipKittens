// FP8 NN GEMM: C = A @ B where A is MxK, B is KxN
//
// In-kernel transpose: loads B from KxN global memory, writes TRANSPOSED
// to shared memory (NxK layout = Bs[n][k]), then uses same NT compute body.
// No workspace buffers needed.
//
// B transpose technique: load 4 consecutive K-rows of 16 N-elements each
// (float4 = 16 FP8 bytes), interleave into 4-byte column vectors, write
// with ds_write_b32 to shared memory in transposed layout.

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
static_assert(NROW * 2 * T == BS, "BS must be divisible by 2*T");
static_assert(NCOL * 4 * T == BS, "BS must be divisible by 4*T");
static_assert(KI >= 2, "KS must be at least 2*DK=32");
typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16v;

#define MFMA(acc, a, b) acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, acc, 0, 0, 0)

// Load B from KxN global and write TRANSPOSED (NxK) into shared Bs.
// Groups: 4 K-rows × 16 N-elements per batch. Writes 4 bytes per N position (ds_write_b32).
template<int KS_STRIDE>
__device__ void load_B_nn_fp8(
    char Bs[][KS_STRIDE],
    const char* __restrict__ B_base,  // B_KxN base for this tile
    int N_stride,                      // stride between K-rows in global (= N)
    int k_offset)                      // k_tile * KS
{
    constexpr int N_GROUP = 16;  // N-elements per float4 load
    constexpr int K_GROUP = 4;   // K-rows loaded together
    constexpr int N_BATCHES = BS / N_GROUP;
    constexpr int K_BATCHES = KS / K_GROUP;
    constexpr int TOTAL = K_BATCHES * N_BATCHES;

    #pragma unroll 1
    for (int batch = threadIdx.x; batch < TOTAL; batch += NT) {
        int kb = batch / N_BATCHES;
        int nb = batch % N_BATCHES;
        int k = kb * K_GROUP;
        int n = nb * N_GROUP;

        const char* base = B_base + (k_offset + k) * N_stride + n;
        float4 r0 = *(const float4*)(base);
        float4 r1 = *(const float4*)(base + N_stride);
        float4 r2 = *(const float4*)(base + 2*N_stride);
        float4 r3 = *(const float4*)(base + 3*N_stride);

        const char *b0=(const char*)&r0, *b1=(const char*)&r1,
                   *b2=(const char*)&r2, *b3=(const char*)&r3;

        #pragma unroll
        for (int i = 0; i < N_GROUP; i++) {
            union { char c[4]; int w; } col;
            col.c[0]=b0[i]; col.c[1]=b1[i]; col.c[2]=b2[i]; col.c[3]=b3[i];
            *(int*)&Bs[n + i][k] = col.w;
        }
    }
}

__global__ __launch_bounds__(NT, 2)
void fp8_gemm_nn(float* __restrict__ C, const char* __restrict__ A,
                 const char* __restrict__ B, int M, int N, int K) {
    // Bs has +4 padding to eliminate LDS bank conflicts on transposed writes
    // bank = (byte_addr/4) % 32. With stride KS, bank repeats every 8 rows.
    // Padding changes stride to KS+4, making gcd(stride/4, 32) = 1 → no conflicts.
    constexpr int BS_PAD = KS + 4;
    __shared__ char As[BS][KS], Bs[BS][BS_PAD];
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

    // A: MxK — same buffer descriptor as NT
    const char* Ab = A + tm*BS*K;
    i32x4 a_sr = make_srsrc(Ab, BS*K, K);
    // B: KxN — different layout! Use raw pointer for transposed loads
    const char* Bb = B;  // full B matrix base
    int B_n_offset = tn * BS;  // column offset for this tile
    const int total_a_loads = (BS * KS) / 16;

    // Prologue: load A normally, load B with transpose
    for(int i=threadIdx.x; i<total_a_loads; i+=NT) {
        int r=i/(KS/16), c16=(i%(KS/16))*16;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+c16,0,0);
        *(float4*)&As[r][c16]=*(float4*)&ra;
    }
    load_B_nn_fp8(Bs, Bb + B_n_offset, N, 0);
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    const int num_tiles = K/KS;
    for(int kt=0; kt<num_tiles-1; kt++) {
        // 1. Prefetch A for next tile
        float4 a_buf;
        {int i=threadIdx.x; if(i<total_a_loads) {
            int r=i/(KS/16),c16=(i%(KS/16))*16;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,r*K+(kt+1)*KS+c16,0,0);
            a_buf=*(float4*)&ra;}}

        // 2. Compute on current shared tiles
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
            for(int c=0;c<NCOL;c++)
                #pragma unroll
                for(int r=0;r<NROW;r++)
                    MFMA(acc[r*NCOL+c], av[r], bv[c]);
            __builtin_amdgcn_s_setprio(0);
        }

        // 3. Store A + transpose-load B
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        {int i=threadIdx.x; if(i<total_a_loads) {
            int r=i/(KS/16),c16=(i%(KS/16))*16;
            *(float4*)&As[r][c16]=a_buf;}}
        load_B_nn_fp8(Bs, Bb + B_n_offset, N, (kt+1)*KS);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    // Last tile
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
        for(int c=0;c<NCOL;c++)
            #pragma unroll
            for(int r=0;r<NROW;r++)
                MFMA(acc[r*NCOL+c], av[r], bv[c]);
        __builtin_amdgcn_s_setprio(0);
    }

    // Store results (same as NT)
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

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp8_nn
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        int Msz = sa[0].cast<int>(), Ksz = sa[1].cast<int>(), Nsz = sb[1].cast<int>();
        fp8_gemm_nn<<<dim3((Nsz/BS)*(Msz/BS)), dim3(NT), 0, (hipStream_t)0>>>(
            (float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const char*)A.attr("data_ptr")().cast<uint64_t>(),
            (const char*)B.attr("data_ptr")().cast<uint64_t>(),
            Msz, Nsz, Ksz);
    }, "FP8 NN GEMM: C = A @ B (in-kernel transpose)");
}
