/**
 * HipKittens BF16 GEMM — MI300X (gfx942, CDNA3)
 *
 * Simple single-buffered design: load→compute→load.
 * No register prefetch buffers (avoids spills for large tiles).
 * v_mfma_f32_16x16x16bf16_1k — 16x16 output, K=16
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
// CDNA3: v_mfma_f32_32x32x8_bf16 — 32x32 output, K=8 (same as Triton uses)
constexpr int DOT_SLICE = 8;
constexpr int K_SLICES = BK / DOT_SLICE;
constexpr int WGM = 4;

using rt_c_s = ducks::rt_shape::rt_32x32;  // Accumulator: 32x32 output
using rt_ab_s = ducks::rt_shape::rt_32x8;  // A/B fragment: 32 rows × 8 K cols
constexpr int a_f4pr = BK / 8;
constexpr int b_f4pr = BK / 8;

__global__ __launch_bounds__(NTHREADS, 2)
void gemm_kernel(const bf16* __restrict__ A,
                 const bf16* __restrict__ B,
                 bf16* __restrict__ C,
                 int M, int N, int K) {

    __shared__ bf16 smem_A[BM * BK];
    __shared__ bf16 smem_B[BN * BK];

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
    const int lr = lane % 16, lc = (lane / 16) * 4;

    // === MAIN LOOP ===
    #pragma unroll 1
    for (int kt = 0; kt < num_k; ++kt) {
        // --- Load tile to shared (issue all globals before waits) ---
        {
            const int cur_k = kt * BK;
            float4* da = reinterpret_cast<float4*>(smem_A);
            float4* db = reinterpret_cast<float4*>(smem_B);

            // Each thread loads one float4 from A and one from B
            // Issue BOTH global loads before writing to shared
            const int a_total = BM * a_f4pr;
            const int b_total = BN * b_f4pr;

            // This loop handles cases where total > NTHREADS
            for (int base = 0; base < max(a_total, b_total); base += NTHREADS) {
                float4 a_val, b_val;
                bool has_a = (base + tid) < a_total;
                bool has_b = (base + tid) < b_total;
                int a_idx = base + tid;
                int b_idx = base + tid;

                // Issue global loads (both non-blocking)
                if (has_a) {
                    int r = a_idx / a_f4pr, c4 = a_idx % a_f4pr;
                    a_val = reinterpret_cast<const float4*>(A + (row0 + r) * K + cur_k)[c4];
                }
                if (has_b) {
                    int r = b_idx / b_f4pr, c4 = b_idx % b_f4pr;
                    b_val = reinterpret_cast<const float4*>(B + (col0 + r) * K + cur_k)[c4];
                }

                // Write to shared (compiler inserts vmcnt wait before use)
                if (has_a) da[a_idx] = a_val;
                if (has_b) db[b_idx] = b_val;
            }
        }
        __syncthreads();

        // --- Compute all K_SLICES ---
        #pragma unroll
        for (int ks = 0; ks < K_SLICES; ++ks) {
            const bf16* ap = smem_A + warp_row * REG_M * BK + ks * DOT_SLICE;
            const bf16* bp = smem_B + warp_col * REG_N * BK + ks * DOT_SLICE;
            // v_mfma_f32_32x32x8_bf16: 32x32 output, K=8
            // Data layout: row = lane % 32, col_base = (lane / 32) * 4
            // Each thread holds 4 bf16 = bf16_2[2] for the K=8 input
            const int lr32 = lane % 32, lc32 = (lane / 32) * 4;
            #pragma unroll
            for (int bm = 0; bm < REG_M / 32; ++bm) {
                bf16_2* fp = &a_frag.tiles[bm][0].data[0];
                const bf16* tp = ap + (bm * 32 + lr32) * BK + lc32;
                fp[0] = *reinterpret_cast<const bf16_2*>(tp);
                fp[1] = *reinterpret_cast<const bf16_2*>(tp + 2);
            }
            #pragma unroll
            for (int bn = 0; bn < REG_N / 32; ++bn) {
                bf16_2* fp = &b_frag.tiles[bn][0].data[0];
                const bf16* tp = bp + (bn * 32 + lr32) * BK + lc32;
                fp[0] = *reinterpret_cast<const bf16_2*>(tp);
                fp[1] = *reinterpret_cast<const bf16_2*>(tp + 2);
            }
            mma_ABt(C_accum, a_frag, b_frag, C_accum);
        }

        __syncthreads();
    }

    // === EPILOGUE: 32x32 accumulator → BF16 output ===
    // v_mfma_f32_32x32x8 output layout (gfx942):
    //   Each thread holds 16 floats = float2[8]
    //   4 blocks of 4 consecutive rows:
    //   block b (0-3): row = (lane/32)*4 + b*8 + {0,1,2,3}, col = lane%32
    #pragma unroll
    for (int bm = 0; bm < REG_M / 32; ++bm) {
        #pragma unroll
        for (int bn = 0; bn < REG_N / 32; ++bn) {
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
