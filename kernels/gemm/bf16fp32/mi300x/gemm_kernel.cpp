/**
 * HipKittens BF16 GEMM Kernel — Parameterized for MI300X (gfx942, CDNA3)
 *
 * Uses HipKittens types and MMA, but manual global→shared memory transfers
 * (since the HK group::load uses buffer_load_lds which is CDNA4-only).
 *
 * Tile sizes via -D flags: BLOCK_M, BLOCK_N, K_STEP_SIZE, WARPS_M, WARPS_N
 * MFMA: v_mfma_f32_16x16x16bf16_1k (CDNA3) — 16x16 output, K=16
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

// Per-warp tile (multiples of 16 for 16x16 MFMA)
constexpr int REG_M = BM / WM;
constexpr int REG_N = BN / WN;
constexpr int DOT_SLICE = 16;
constexpr int K_SLICES = BK / DOT_SLICE;

constexpr int WGM = 4;

// HipKittens types — only for shared→register loads and MMA
using rt_s = ducks::rt_shape::rt_16x16;

// No gl types needed — kernel takes raw pointers, pybind11 takes torch::Tensor

// Manual global→shared copy using vectorized loads
__device__ void copy_global_to_shared(bf16* __restrict__ dst,
                                      const bf16* __restrict__ src,
                                      int rows, int cols, int src_stride,
                                      int num_threads) {
    // Copy rows x cols tile from global (with stride) to shared (packed)
    const int tid = threadIdx.x;
    const int total = rows * cols;
    // Use float4 (128-bit) loads when possible
    if (cols % 8 == 0 && ((uintptr_t)src % 16) == 0) {
        const int total4 = total / 8;  // 8 bf16 = 16 bytes = 1 float4
        for (int i = tid; i < total4; i += num_threads) {
            int elem_idx = i * 8;
            int r = elem_idx / cols;
            int c = elem_idx % cols;
            ((float4*)dst)[i] = *((const float4*)(src + r * src_stride + c));
        }
    } else {
        for (int i = tid; i < total; i += num_threads) {
            int r = i / cols;
            int c = i % cols;
            dst[i] = src[r * src_stride + c];
        }
    }
}

__global__ __launch_bounds__(NTHREADS, 2)
void gemm_kernel(const bf16* __restrict__ A,
                 const bf16* __restrict__ B,
                 bf16* __restrict__ C,
                 int M, int N, int K) {

    // Single-buffered shared memory (fits in 64KB LDS for all tile sizes)
    __shared__ bf16 As_buf[BM * BK];
    __shared__ bf16 Bs_buf[BN * BK];

    // Per-warp FP32 accumulator
    rt_fl<REG_M, REG_N, col_l, rt_s> C_accum;
    zero(C_accum);

    rt_bf<REG_M, DOT_SLICE, row_l, rt_s> a_frag;
    rt_bf<REG_N, DOT_SLICE, row_l, rt_s> b_frag;

    // --- WG ID swizzling for L2 locality ---
    int wgid = blockIdx.x;
    const int num_blocks_m = M / BM;
    const int num_blocks_n = N / BN;
    const int NUM_WGS = num_blocks_m * num_blocks_n;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);

    int num_wgid_in_group = WGM * num_blocks_m;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_blocks_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    const int row_start = pid_m * BM;
    const int col_start = pid_n * BN;

    const int warp_id  = kittens::warpid();
    const int warp_row = warp_id / WN;
    const int warp_col = warp_id % WN;
    const int num_k_tiles = K / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load current tile
        int cur_k = kt * BK;
        copy_global_to_shared(As_buf, A + row_start * K + cur_k, BM, BK, K, NTHREADS);
        copy_global_to_shared(Bs_buf, B + col_start * K + cur_k, BN, BK, K, NTHREADS);
        __syncthreads();

        // Compute MMA on current tile
        #pragma unroll
        for (int ks = 0; ks < K_SLICES; ++ks) {
            const bf16* a_ptr = As_buf + warp_row * REG_M * BK + ks * DOT_SLICE;
            const bf16* b_ptr = Bs_buf + warp_col * REG_N * BK + ks * DOT_SLICE;

            // Manual load into register tiles
            // Each thread in the warp loads its portion of the fragment
            const int lane = threadIdx.x % 64;  // lane within warp

            // For rt_16x16 shape with bf16: each base tile is 16x16
            // The register tile has height = REG_M/16 base tiles, width = DOT_SLICE/16 = 1
            // Each base tile distributes 256 bf16 values across 64 threads = 4 values/thread = bf16_2[2]
            #pragma unroll
            for (int bm = 0; bm < REG_M / 16; ++bm) {
                // In a 16x16 bf16 tile with rt_16x16 shape (stride=4):
                // Thread mapping: each thread owns 4 elements
                // row = lane / 16, within the 16-row block
                // For row-layout register tile:
                //   threads 0-15 own row 0 elements, threads 16-31 own row 1, etc.
                //   Actually the exact mapping depends on the MFMA instruction's data layout
                // Use __builtin to load 4 bf16 values per thread for the MFMA input format
                int base_row = bm * 16;
                // v_mfma_f32_16x16x16bf16_1k input layout:
                //   lane_id = 0..63
                //   A element [row][col] where row = lane_id % 16, col = (lane_id / 16) * 4 + {0,1,2,3}
                int a_row = lane % 16;
                int a_col_base = (lane / 16) * 4;
                bf16_2* frag_ptr = &a_frag.tiles[bm][0].data[0];
                const bf16* tile_ptr = a_ptr + (base_row + a_row) * BK;
                frag_ptr[0] = *(bf16_2*)(tile_ptr + a_col_base);
                frag_ptr[1] = *(bf16_2*)(tile_ptr + a_col_base + 2);
            }

            #pragma unroll
            for (int bn = 0; bn < REG_N / 16; ++bn) {
                int base_row = bn * 16;
                int b_row = lane % 16;
                int b_col_base = (lane / 16) * 4;
                bf16_2* frag_ptr = &b_frag.tiles[bn][0].data[0];
                const bf16* tile_ptr = b_ptr + (base_row + b_row) * BK;
                frag_ptr[0] = *(bf16_2*)(tile_ptr + b_col_base);
                frag_ptr[1] = *(bf16_2*)(tile_ptr + b_col_base + 2);
            }

            mma_ABt(C_accum, a_frag, b_frag, C_accum);
        }

        __syncthreads();
    }

    // --- Epilogue: store FP32 accumulators to BF16 output ---
    // C_accum is rt_fl<REG_M, REG_N, col_l, rt_16x16>
    // Each base tile is 16x16, with height = REG_M/16, width = REG_N/16
    const int lane = threadIdx.x % 64;
    #pragma unroll
    for (int bm = 0; bm < REG_M / 16; ++bm) {
        #pragma unroll
        for (int bn = 0; bn < REG_N / 16; ++bn) {
            // For col_l accumulator with rt_16x16 shape:
            // D layout: col = lane % 16, row_base = (lane / 16) * 4
            int c_col = lane % 16;
            int c_row_base = (lane / 16) * 4;

            float2* acc_ptr = &C_accum.tiles[bm][bn].data[0];
            int global_row_base = row_start + warp_row * REG_M + bm * 16 + c_row_base;
            int global_col = col_start + warp_col * REG_N + bn * 16 + c_col;

            // Each thread has 4 floats (float2[2]) for 4 consecutive rows
            for (int dr = 0; dr < 4; ++dr) {
                float val;
                if (dr < 2) {
                    val = (dr == 0) ? acc_ptr[0].x : acc_ptr[0].y;
                } else {
                    val = (dr == 2) ? acc_ptr[1].x : acc_ptr[1].y;
                }
                int r = global_row_base + dr;
                if (r < M && global_col < N) {
                    C[r * N + global_col] = __float2bfloat16(val);
                }
            }
        }
    }
}

// Pybind11 wrapper
void dispatch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // B is NxK (transposed)

    dim3 grid((M / BM) * (N / BN));
    dim3 block(NTHREADS);

    // Double-buffered shared memory
    size_t smem = 2 * (BM * BK + BN * BK) * sizeof(bf16);
    hipFuncSetAttribute((void*)gemm_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, 0);  // using static shared

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    gemm_kernel<<<grid, block, 0, stream>>>(
        (const bf16*)A.data_ptr(),
        (const bf16*)B.data_ptr(),
        (bf16*)C.data_ptr(),
        M, N, K);
}

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

#ifndef MODULE_NAME
#define MODULE_NAME tk_gemm
#endif

PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = "HipKittens BF16 GEMM — " STRINGIFY(BLOCK_M) "x" STRINGIFY(BLOCK_N) "x" STRINGIFY(K_STEP_SIZE);
    m.def("dispatch", &dispatch_gemm, "Run GEMM: C = A @ B^T (A is MxK, B is NxK, C is MxN)");
    m.attr("BLOCK_M") = py::int_(BM);
    m.attr("BLOCK_N") = py::int_(BN);
    m.attr("K_STEP")  = py::int_(BK);
    m.attr("WARPS_M") = py::int_(WM);
    m.attr("WARPS_N") = py::int_(WN);
}
