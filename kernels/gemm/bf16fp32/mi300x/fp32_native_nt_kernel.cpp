// Native FP32 NT GEMM using v_mfma_f32_16x16x4f32
// Full FP32 precision, no type conversion.

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

constexpr int BS = 128;
constexpr int KS = 16;
constexpr int NUM_WARPS = 8;
constexpr int WARP_SIZE = 64;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
constexpr int TILE = 16;
constexpr int DOT_K = 4;
constexpr int REG_BLOCK = BS / 4;  // 32
constexpr int K_ITERS = KS / DOT_K;  // 4

__device__ inline void mfma_f32_16x16x4(float (&D)[4], float A, float B, const float (&C)[4]) {
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;
    *(float4_t*)D = __builtin_amdgcn_mfma_f32_16x16x4f32(A, B, *(float4_t*)C, 0, 0, 0);
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp32_gemm_nt(float* __restrict__ C,
                  const float* __restrict__ A,
                  const float* __restrict__ B,
                  int M, int N, int K)
{
    __shared__ float As[BS][KS];
    __shared__ float Bs[BS][KS];

    const int num_n_tiles = N / BS;
    int tile_idx = blockIdx.x;
    int tile_m = tile_idx / num_n_tiles;
    int tile_n = tile_idx % num_n_tiles;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_row = warp_id / 4;  // 0 or 1
    int warp_col = warp_id % 4;  // 0-3

    // Two accumulator sets: one for warp_row, one for warp_row+2
    // This covers all 4 row blocks (0,1,2,3 × REG_BLOCK=32 = 128 rows)
    float acc0[2][2][4];  // warp_row block (rows warp_row*32 .. +31)
    float acc1[2][2][4];  // warp_row+2 block (rows (warp_row+2)*32 .. +31)
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 4; k++) {
                acc0[i][j][k] = 0.0f;
                acc1[i][j][k] = 0.0f;
            }

    const int num_k_tiles = K / KS;
    int mfma_row = lane % 16;
    int mfma_k = lane / 16;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // Load A and B tiles to shared
        for (int idx = threadIdx.x; idx < BS * KS; idx += NUM_THREADS) {
            int r = idx / KS, c = idx % KS;
            As[r][c] = A[(tile_m * BS + r) * K + kt * KS + c];
            Bs[r][c] = B[(tile_n * BS + r) * K + kt * KS + c];
        }
        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int koff = ki * DOT_K;

            #pragma unroll
            for (int bi = 0; bi < 2; bi++) {
                // A from warp_row block
                float a0 = As[warp_row * REG_BLOCK + bi * TILE + mfma_row][koff + mfma_k];
                // A from warp_row+2 block
                float a1 = As[(warp_row + 2) * REG_BLOCK + bi * TILE + mfma_row][koff + mfma_k];

                #pragma unroll
                for (int bj = 0; bj < 2; bj++) {
                    float b_val = Bs[warp_col * REG_BLOCK + bj * TILE + mfma_row][koff + mfma_k];
                    mfma_f32_16x16x4(acc0[bi][bj], a0, b_val, acc0[bi][bj]);
                    mfma_f32_16x16x4(acc1[bi][bj], a1, b_val, acc1[bi][bj]);
                }
            }
        }
        __syncthreads();
    }

    // Store both accumulator sets
    #pragma unroll
    for (int bi = 0; bi < 2; bi++) {
        #pragma unroll
        for (int bj = 0; bj < 2; bj++) {
            int out_col = tile_n * BS + warp_col * REG_BLOCK + bj * TILE + (lane % 16);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int row0 = tile_m * BS + warp_row * REG_BLOCK + bi * TILE + (lane / 16) * 4 + k;
                int row1 = tile_m * BS + (warp_row + 2) * REG_BLOCK + bi * TILE + (lane / 16) * 4 + k;
                C[row0 * N + out_col] = acc0[bi][bj][k];
                C[row1 * N + out_col] = acc1[bi][bj][k];
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
    }, "Native FP32 GEMM using v_mfma_f32_16x16x4f32");
}
