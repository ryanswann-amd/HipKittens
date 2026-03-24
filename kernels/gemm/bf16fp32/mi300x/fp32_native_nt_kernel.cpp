// Native FP32 NT GEMM with 8-cluster pipeline schedule
// Uses v_mfma_f32_16x16x4f32 with register buffer pipelining.
//
// K_STEP=16 (FP32: 4 bytes/elem → 128×16×4 = 8KB per shared tile = 16KB total)
// DOT_K=4 → 4 MFMA calls per K_STEP (same as BF16's 4 calls)
// Shared tiles: raw float arrays with manual addressing

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

constexpr int BS = 128;
constexpr int KS = 16;
constexpr int NUM_WARPS = 8;
constexpr int WARP_SIZE = 64;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
constexpr int TILE = 16;
constexpr int DOT_K = 4;
constexpr int RB = BS / 4;  // 32 (2 base tiles)
constexpr int K_ITERS = KS / DOT_K;  // 4

// Chiplet-aware WG ID transform (same as BF16 kernel)
constexpr int NUM_XCDS = 8;
__device__ int chiplet_wgid(int wgid, int num_wgs) {
    constexpr int WGM = 4;
    int chunk = WGM * WGM;
    int num_chunks = (num_wgs + chunk - 1) / chunk;
    int chunk_id = wgid % num_chunks;
    int local_id = wgid / num_chunks;
    return chunk_id * chunk + local_id;
}

__device__ inline void mfma_f32(float (&D)[4], float A, float B, const float (&C)[4]) {
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f4;
    *(f4*)D = __builtin_amdgcn_mfma_f32_16x16x4f32(A, B, *(f4*)C, 0, 0, 0);
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp32_gemm_nt(float* __restrict__ C,
                  const float* __restrict__ A,
                  const float* __restrict__ B,
                  int M, int N, int K)
{
    __shared__ float As[BS][KS];
    __shared__ float Bs[BS][KS];

    const int num_n = N / BS;
    int wgid = blockIdx.x;
    wgid = chiplet_wgid(wgid, gridDim.x);
    int tile_m = wgid / num_n;
    int tile_n = wgid % num_n;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int mfma_r = lane % 16;
    const int mfma_k = lane / 16;
    const int num_k = K / KS;

    // Accumulators: 2 row blocks × 2 col blocks × 4 outputs
    float c0[2][2][4] = {};  // warp_row rows
    float c1[2][2][4] = {};  // warp_row+2 rows

    // Helper: read A subtile value for MFMA
    #define A_VAL(row_block, ki) As[(row_block)*RB + mfma_r][(ki)*DOT_K + mfma_k]
    #define A_VAL2(row_block, ki) As[((row_block)+2)*RB + mfma_r][(ki)*DOT_K + mfma_k]
    #define B_VAL(col_block, ki) Bs[(col_block)*RB + mfma_r][(ki)*DOT_K + mfma_k]

    // Prologue: load first tiles
    for (int i = threadIdx.x; i < BS * KS; i += NUM_THREADS) {
        int r = i / KS, c = i % KS;
        As[r][c] = A[(tile_m * BS + r) * K + c];
        Bs[r][c] = B[(tile_n * BS + r) * K + c];
    }
    __syncthreads();

    for (int kt = 0; kt < num_k - 1; kt++) {
        // Register buffer: prefetch next A and B tiles
        // FP32: float4 = 4 elements, BS*KS/NUM_THREADS = 4 elements per thread = 1 float4
        float4 a_buf, b_buf;
        {
            int idx = threadIdx.x;
            int r = idx / (KS/4), c4 = idx % (KS/4);
            if (idx < BS * KS / 4) {
                a_buf = *(float4*)&A[(tile_m * BS + r) * K + (kt+1)*KS + c4*4];
                b_buf = *(float4*)&B[(tile_n * BS + r) * K + (kt+1)*KS + c4*4];
            }
        }

        // 8-cluster schedule: load subtiles + MFMA interleaved
        // Cluster 0-1: K=0
        {
            float a0 = A_VAL(warp_row, 0), a1 = A_VAL2(warp_row, 0);
            for (int bj = 0; bj < 2; bj++) {
                float b = B_VAL(warp_col*2+bj, 0);  // Wait, warp_col range is 0-3, *2+bj would exceed BS
                // Fix: warp_col already indexes 4 blocks of 32, bj indexes 2 base tiles of 16
                // So: col_block = warp_col, bi=0/1 for 2 base tiles within warp_row's 32-row block
            }
        }

        // Actually, let me use a simpler but correct interleaved schedule:
        // Each K_ITER does: load from shared + MFMA
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            #pragma unroll
            for (int bi = 0; bi < 2; bi++) {
                float a0 = A_VAL(warp_row*2+bi, ki);     // warp_row block, bi-th base tile
                float a1 = A_VAL2(warp_row*2+bi, ki);    // warp_row+2 block
                // Wait, A_VAL macro uses row_block*RB. warp_row=0/1, so warp_row*2 = 0/2.
                // But RB=32 and TILE=16, so we need bi to index within the 32-row block.
                // row_block should give: warp_row*RB + bi*TILE + mfma_r
                // My macro: A_VAL(row_block, ki) = As[row_block*RB + mfma_r][ki*DOT_K + mfma_k]
                // But this doesn't have the bi*TILE offset!
                // Fix: use direct indexing
            }
        }

        // OK let me just write it directly without macros
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int koff = ki * DOT_K + mfma_k;
            #pragma unroll
            for (int bi = 0; bi < 2; bi++) {
                int ar0 = warp_row * RB + bi * TILE + mfma_r;
                int ar1 = (warp_row + 2) * RB + bi * TILE + mfma_r;
                float a0 = As[ar0][koff];
                float a1 = As[ar1][koff];

                #pragma unroll
                for (int bj = 0; bj < 2; bj++) {
                    int br = warp_col * RB + bj * TILE + mfma_r;
                    float b = Bs[br][koff];
                    mfma_f32(c0[bi][bj], a0, b, c0[bi][bj]);
                    mfma_f32(c1[bi][bj], a1, b, c1[bi][bj]);
                }
            }
        }
        __syncthreads();

        // Store prefetched data to shared
        {
            int idx = threadIdx.x;
            int r = idx / (KS/4), c4 = idx % (KS/4);
            if (idx < BS * KS / 4) {
                *(float4*)&As[r][c4*4] = a_buf;
                *(float4*)&Bs[r][c4*4] = b_buf;
            }
        }
        __syncthreads();
    }

    // Last K tile (no prefetch)
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int koff = ki * DOT_K + mfma_k;
        #pragma unroll
        for (int bi = 0; bi < 2; bi++) {
            float a0 = As[warp_row * RB + bi * TILE + mfma_r][koff];
            float a1 = As[(warp_row+2) * RB + bi * TILE + mfma_r][koff];
            #pragma unroll
            for (int bj = 0; bj < 2; bj++) {
                float b = Bs[warp_col * RB + bj * TILE + mfma_r][koff];
                mfma_f32(c0[bi][bj], a0, b, c0[bi][bj]);
                mfma_f32(c1[bi][bj], a1, b, c1[bi][bj]);
            }
        }
    }

    // Store
    #pragma unroll
    for (int bi = 0; bi < 2; bi++) {
        #pragma unroll
        for (int bj = 0; bj < 2; bj++) {
            int col = tile_n * BS + warp_col * RB + bj * TILE + (lane % 16);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int r0 = tile_m * BS + warp_row * RB + bi * TILE + (lane/16)*4 + k;
                int r1 = tile_m * BS + (warp_row+2) * RB + bi * TILE + (lane/16)*4 + k;
                C[r0 * N + col] = c0[bi][bj][k];
                C[r1 * N + col] = c1[bi][bj][k];
            }
        }
    }
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp32_native_nt
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        int M = sa[0].cast<int>(), K = sa[1].cast<int>(), N = sb[0].cast<int>();
        fp32_gemm_nt<<<dim3((M/BS)*(N/BS)), dim3(NUM_THREADS), 0, (hipStream_t)0>>>(
            (float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const float*)A.attr("data_ptr")().cast<uint64_t>(),
            (const float*)B.attr("data_ptr")().cast<uint64_t>(), M, N, K);
    }, "Native FP32 GEMM with pipelined schedule");
}
