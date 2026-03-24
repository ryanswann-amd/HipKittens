/**
 * HipKittens BF16 GEMM — MI300X (gfx942, CDNA3)
 *
 * Double-buffered shared memory for true compute/memory overlap:
 *   - Compute MFMAs on buffer[tic] while loading next tile to buffer[toc]
 *   - No separate load phase — loads are issued before compute and complete during it
 *   - v_mfma_f32_32x32x8_bf16 — 32x32 output, K=8
 */

#include "kittens.cuh"
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
using namespace kittens;

#ifndef BLOCK_M
#define BLOCK_M 256
#endif
#ifndef BLOCK_N
#define BLOCK_N 256
#endif
#ifndef K_STEP_SIZE
#define K_STEP_SIZE 64
#endif
#ifndef WARPS_M
#define WARPS_M 2
#endif
#ifndef WARPS_N
#define WARPS_N 4
#endif

constexpr int BM = BLOCK_M;
constexpr int BN = BLOCK_N;
constexpr int BK = K_STEP_SIZE;
constexpr int WM = WARPS_M;
constexpr int WN = WARPS_N;
constexpr int NWARPS = WM * WN;
constexpr int NTHREADS = kittens::WARP_THREADS * NWARPS;

constexpr int REG_M = BM / WM;
constexpr int REG_N = BN / WN;
constexpr int DOT_SLICE = 8;
constexpr int K_SLICES = BK / DOT_SLICE;
constexpr int WGM = 4;

using rt_c_s = ducks::rt_shape::rt_32x32;
using rt_ab_s = ducks::rt_shape::rt_32x8;

constexpr int MFMA_M = REG_M / 32;
constexpr int MFMA_N = REG_N / 32;
constexpr int a_f4pr = BK / 8;
constexpr int b_f4pr = BK / 8;

// Check double buffer fits in LDS
static_assert((BM * BK + BN * BK) * 2 * sizeof(bf16) <= 65536,
              "Double-buffered shared memory must fit in 64KB");

__global__ __launch_bounds__(NTHREADS, 2)
void gemm_kernel(const bf16* __restrict__ A,
                 const bf16* __restrict__ B,
                 bf16* __restrict__ C,
                 int M, int N, int K) {

    // Double-buffered shared memory
    __shared__ bf16 smem_A[2][BM * BK];
    __shared__ bf16 smem_B[2][BN * BK];

    rt_fl<REG_M, REG_N, col_l, rt_c_s> C_accum;
    zero(C_accum);
    rt_bf<REG_M, DOT_SLICE, row_l, rt_ab_s> a_frag;
    rt_bf<REG_N, DOT_SLICE, row_l, rt_ab_s> b_frag;

    int wgid = blockIdx.x;
    const int nb_m = M / BM, nb_n = N / BN;
    wgid = chiplet_transform_chunked(wgid, nb_m * nb_n, NUM_XCDS, WGM * WGM);
    int nwig = WGM * nb_m, gid = wgid / nwig;
    int first = gid * WGM, gsm = min(nb_m - first, WGM);
    int pid_m = first + ((wgid % nwig) % gsm);
    int pid_n = (wgid % nwig) / gsm;
    const int row0 = pid_m * BM, col0 = pid_n * BN;

    const int tid = threadIdx.x;
    const int warp_row = kittens::warpid() / WN;
    const int warp_col = kittens::warpid() % WN;
    const int lane = tid % 64;
    const int num_k = K / BK;
    const int lr32 = lane % 32, lc32 = (lane / 32) * 4;

    // Pre-compute per-warp LDS base offsets (invariant across K-tiles)
    const int a_warp_off = warp_row * REG_M * BK + lr32 * BK + lc32;
    const int b_warp_off = warp_col * REG_N * BK + lr32 * BK + lc32;

    int tic = 0, toc = 1;

    // Buffer resource descriptors for A and B (hardware address calculation)
    const int a_stride_bytes = K * sizeof(bf16);
    const int b_stride_bytes = K * sizeof(bf16);
    i32x4 a_srsrc = make_srsrc(A + row0 * K, BM * a_stride_bytes, a_stride_bytes);
    i32x4 b_srsrc = make_srsrc(B + col0 * K, BN * b_stride_bytes, b_stride_bytes);

    // Per-thread load parameters (computed once, reused every iteration)
    constexpr int A_PER_T = (BM * a_f4pr + NTHREADS - 1) / NTHREADS;
    constexpr int B_PER_T = (BN * b_f4pr + NTHREADS - 1) / NTHREADS;

    // Pre-compute byte offsets for each thread's loads (invariant across K-tiles)
    int a_byte_off[A_PER_T], b_byte_off[B_PER_T];
    #pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
        int idx = tid + i * NTHREADS;
        int r = idx / a_f4pr, c8 = (idx % a_f4pr) * 8;
        a_byte_off[i] = (r * K + c8) * sizeof(bf16);
    }
    #pragma unroll
    for (int i = 0; i < B_PER_T; ++i) {
        int idx = tid + i * NTHREADS;
        int r = idx / b_f4pr, c8 = (idx % b_f4pr) * 8;
        b_byte_off[i] = (r * K + c8) * sizeof(bf16);
    }

    // LDS base pointers for writing
    uint32_t a_lds = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_A[0])));
    uint32_t b_lds = __builtin_amdgcn_readfirstlane(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_B[0])));
    constexpr int SMEM_TILE = BM * BK * sizeof(bf16);  // bytes per A or B buffer

    // === PROLOGUE: Load first tile via buffer_load → register → shared ===
    {
        float4 a_reg[A_PER_T], b_reg[B_PER_T];
        #pragma unroll
        for (int i = 0; i < A_PER_T; ++i) {
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(a_srsrc, a_byte_off[i], 0, 0);
            a_reg[i] = *reinterpret_cast<float4*>(&raw);
        }
        #pragma unroll
        for (int i = 0; i < B_PER_T; ++i) {
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(b_srsrc, b_byte_off[i], 0, 0);
            b_reg[i] = *reinterpret_cast<float4*>(&raw);
        }
        asm volatile("s_waitcnt vmcnt(0)");
        // Write to shared[tic]
        #pragma unroll
        for (int i = 0; i < A_PER_T; ++i) {
            int idx = tid + i * NTHREADS;
            uint32_t off = a_lds + tic * SMEM_TILE + idx * 16;
            store_shared_vec(off, {a_reg[i].x, a_reg[i].y});
            store_shared_vec(off + 8, {a_reg[i].z, a_reg[i].w});
        }
        #pragma unroll
        for (int i = 0; i < B_PER_T; ++i) {
            int idx = tid + i * NTHREADS;
            uint32_t off = b_lds + tic * SMEM_TILE + idx * 16;
            store_shared_vec(off, {b_reg[i].x, b_reg[i].y});
            store_shared_vec(off + 8, {b_reg[i].z, b_reg[i].w});
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    // === MAIN LOOP: compute on tic, prefetch next to toc ===
    #pragma unroll 1
    for (int kt = 0; kt < num_k - 1; ++kt) {
        // Issue buffer_load for next K-tile (async VMEM — overlaps with MFMA)
        const int k_byte_off = (kt + 1) * BK * sizeof(bf16);
        float4 a_reg[A_PER_T], b_reg[B_PER_T];
        #pragma unroll
        for (int i = 0; i < A_PER_T; ++i) {
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(
                a_srsrc, a_byte_off[i] + k_byte_off, 0, 0);
            a_reg[i] = *reinterpret_cast<float4*>(&raw);
        }
        #pragma unroll
        for (int i = 0; i < B_PER_T; ++i) {
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(
                b_srsrc, b_byte_off[i] + k_byte_off, 0, 0);
            b_reg[i] = *reinterpret_cast<float4*>(&raw);
        }

        // Compute all K_SLICES on tic buffer (MFMA overlaps with VMEM loads above)
        {
            const bf16* a_base = smem_A[tic] + a_warp_off;
            const bf16* b_base = smem_B[tic] + b_warp_off;

            #pragma unroll
            for (int ks = 0; ks < K_SLICES; ++ks) {
                const int k_off = ks * DOT_SLICE;
                #pragma unroll
                for (int bm = 0; bm < MFMA_M; ++bm) {
                    bf16_2* fp = &a_frag.tiles[bm][0].data[0];
                    const bf16* tp = a_base + bm * 32 * BK + k_off;
                    fp[0] = *reinterpret_cast<const bf16_2*>(tp);
                    fp[1] = *reinterpret_cast<const bf16_2*>(tp + 2);
                }
                #pragma unroll
                for (int bn = 0; bn < MFMA_N; ++bn) {
                    bf16_2* fp = &b_frag.tiles[bn][0].data[0];
                    const bf16* tp = b_base + bn * 32 * BK + k_off;
                    fp[0] = *reinterpret_cast<const bf16_2*>(tp);
                    fp[1] = *reinterpret_cast<const bf16_2*>(tp + 2);
                }
                mma_ABt(C_accum, a_frag, b_frag, C_accum);
            }
        }

        // Wait for buffer_loads, write to shared[toc] via ds_write
        asm volatile("s_waitcnt vmcnt(0)");
        #pragma unroll
        for (int i = 0; i < A_PER_T; ++i) {
            int idx = tid + i * NTHREADS;
            uint32_t off = a_lds + toc * SMEM_TILE + idx * 16;
            store_shared_vec(off, {a_reg[i].x, a_reg[i].y});
            store_shared_vec(off + 8, {a_reg[i].z, a_reg[i].w});
        }
        #pragma unroll
        for (int i = 0; i < B_PER_T; ++i) {
            int idx = tid + i * NTHREADS;
            uint32_t off = b_lds + toc * SMEM_TILE + idx * 16;
            store_shared_vec(off, {b_reg[i].x, b_reg[i].y});
            store_shared_vec(off + 8, {b_reg[i].z, b_reg[i].w});
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        tic ^= 1;
        toc ^= 1;
    }

    // === LAST TILE: compute only ===
    {
        const bf16* a_base = smem_A[tic] + a_warp_off;
        const bf16* b_base = smem_B[tic] + b_warp_off;

        #pragma unroll
        for (int ks = 0; ks < K_SLICES; ++ks) {
            const int k_off = ks * DOT_SLICE;
            #pragma unroll
            for (int bm = 0; bm < MFMA_M; ++bm) {
                bf16_2* fp = &a_frag.tiles[bm][0].data[0];
                const bf16* tp = a_base + bm * 32 * BK + k_off;
                fp[0] = *reinterpret_cast<const bf16_2*>(tp);
                fp[1] = *reinterpret_cast<const bf16_2*>(tp + 2);
            }
            #pragma unroll
            for (int bn = 0; bn < MFMA_N; ++bn) {
                bf16_2* fp = &b_frag.tiles[bn][0].data[0];
                const bf16* tp = b_base + bn * 32 * BK + k_off;
                fp[0] = *reinterpret_cast<const bf16_2*>(tp);
                fp[1] = *reinterpret_cast<const bf16_2*>(tp + 2);
            }
            mma_ABt(C_accum, a_frag, b_frag, C_accum);
        }
    }

    // === EPILOGUE ===
    #pragma unroll
    for (int bm = 0; bm < MFMA_M; ++bm) {
        #pragma unroll
        for (int bn = 0; bn < MFMA_N; ++bn) {
            int c_col = lane % 32;
            float2* acc = &C_accum.tiles[bm][bn].data[0];
            int gc = col0 + warp_col * REG_N + bn * 32 + c_col;
            if (gc < N) {
                bf16* out = C + gc;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int c_row = (lane / 32) * 4 + b * 8;
                    int gr = row0 + warp_row * REG_M + bm * 32 + c_row;
                    if (gr     < M) out[(gr)     * N] = __float2bfloat16(acc[b*2].x);
                    if (gr + 1 < M) out[(gr + 1) * N] = __float2bfloat16(acc[b*2].y);
                    if (gr + 2 < M) out[(gr + 2) * N] = __float2bfloat16(acc[b*2+1].x);
                    if (gr + 3 < M) out[(gr + 3) * N] = __float2bfloat16(acc[b*2+1].y);
                }
            }
        }
    }
}

void dispatch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(0);
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    gemm_kernel<<<dim3((M / BM) * (N / BN)), dim3(NTHREADS), 0, stream>>>(
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(B.data_ptr()),
        reinterpret_cast<bf16*>(C.data_ptr()), M, N, K);
}

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#ifndef MODULE_NAME
#define MODULE_NAME tk_gemm
#endif

PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = "HipKittens BF16 GEMM — " STRINGIFY(BLOCK_M) "x" STRINGIFY(BLOCK_N);
    m.def("dispatch", &dispatch_gemm, "C = A @ B^T");
    m.attr("BLOCK_M") = py::int_(BM);
    m.attr("BLOCK_N") = py::int_(BN);
    m.attr("K_STEP")  = py::int_(BK);
    m.attr("WARPS_M") = py::int_(WM);
    m.attr("WARPS_N") = py::int_(WN);
}
