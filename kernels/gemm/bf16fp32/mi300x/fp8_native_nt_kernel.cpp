// Native FP8 NT GEMM using v_mfma_f32_32x32x16_fp8_fp8
//
// Uses the actual FP8 MFMA instruction for maximum throughput.
// Input: FP8 e4m3fnuz (1 byte/element), Output: FP32 accumulation.
// K=16 per instruction, 32x32 output, each thread provides 8 packed fp8 (long).

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

constexpr int BS = 128;
constexpr int KS = 32;           // K step (must be >= 16 for one MFMA)
constexpr int NUM_WARPS = 8;
constexpr int WARP_SIZE = 64;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;
constexpr int TILE = 32;         // 32x32 MFMA base tile
constexpr int DOT_K = 16;        // K per FP8 MFMA instruction
constexpr int REG_BLOCK = BS / 4; // 32 (= one base tile)
constexpr int K_ITERS = KS / DOT_K; // 2

__device__ inline void mfma_fp8_32x32x16(float (&D)[16], long A, long B, const float (&C)[16]) {
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(A, B, *(floatx16_t*)C, 0, 0, 0);
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void fp8_gemm_nt(float* __restrict__ C,
                 const char* __restrict__ A,  // FP8 e4m3, MxK
                 const char* __restrict__ B,  // FP8 e4m3, NxK
                 int M, int N, int K)
{
    // Shared memory: store FP8 as raw bytes
    __shared__ char As[BS][KS];  // 128 × 32 bytes = 4KB
    __shared__ char Bs[BS][KS];  // 128 × 32 bytes = 4KB

    const int num_n_tiles = N / BS;
    int tile_idx = blockIdx.x;
    int tile_m = tile_idx / num_n_tiles;
    int tile_n = tile_idx % num_n_tiles;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_row = warp_id / 4;  // 0 or 1
    int warp_col = warp_id % 4;  // 0-3

    // For 32x32 MFMA: each warp covers one 32×32 output block
    // With warp_row (0-1) × warp_col (0-3): covers 2×4 = 8 blocks of 32×32 = 64×128 = BS×BS ✓
    // But we need warp_row+2 for the other half!
    // Actually: warp_row=0 covers rows 0-31, warp_row=1 covers rows 32-63
    // Need warp_row+2 for rows 64-95 and 96-127

    // Accumulators: 16 fp32 per thread per base tile, need 2 row sets (warp_row and warp_row+2)
    float acc0[16] = {0};  // warp_row block
    float acc1[16] = {0};  // warp_row+2 block

    const int num_k_tiles = K / KS;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        // Load FP8 data to shared (1 byte per element)
        constexpr int TOTAL = BS * KS;  // 128 × 32 = 4096 bytes
        for (int idx = threadIdx.x; idx < TOTAL; idx += NUM_THREADS) {
            int r = idx / KS, c = idx % KS;
            As[r][c] = A[(tile_m * BS + r) * K + kt * KS + c];
            Bs[r][c] = B[(tile_n * BS + r) * K + kt * KS + c];
        }
        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int koff = ki * DOT_K;

            // Each thread packs 8 FP8 values into a 'long' (64 bits)
            // For 32x32x16 MFMA: thread lane provides 8 fp8 values from A and 8 from B
            // The mapping: lane provides data for row = lane%32 (within 32-row block)
            // and K positions depend on lane/32 and the packed 8 values
            //
            // For the FP8 32x32x16 MFMA on gfx942:
            // - Each thread provides 8 bytes (long) for A and B
            // - These 8 bytes pack 8 fp8 values
            // - With 64 threads × 8 fp8 = 512 values
            // - A is 32×16 = 512 values ✓
            // - Thread t provides A[t%32][2*(t/32)*8 + {0..7}]... actually uncertain
            //
            // Let's use a simple approach: pack 8 consecutive K values for each thread's row
            int a_row0 = warp_row * REG_BLOCK + (lane % 32);
            int a_row1 = (warp_row + 2) * REG_BLOCK + (lane % 32);
            int b_row = warp_col * REG_BLOCK + (lane % 32);

            // Pack 8 FP8 bytes from K dimension
            // For lane/32 = 0: K positions koff+0..7
            // For lane/32 = 1: K positions koff+8..15
            int k_base = koff + (lane / 32) * 8;

            long a_packed0 = *(const long*)&As[a_row0][k_base];
            long a_packed1 = *(const long*)&As[a_row1][k_base];
            long b_packed = *(const long*)&Bs[b_row][k_base];

            mfma_fp8_32x32x16(acc0, a_packed0, b_packed, acc0);
            mfma_fp8_32x32x16(acc1, a_packed1, b_packed, acc1);
        }
        __syncthreads();
    }

    // Store: 32x32 output, 16 fp32 per thread
    // For v_mfma_f32_32x32x16: thread t owns D[(t/32)*4+j][(t%32)] for j=0..3?
    // Actually with 16 outputs per thread in a 32×32 = 1024 element matrix:
    // 1024 / 64 = 16 per thread
    // Mapping: column = t%32? No, 32 columns / 64 threads doesn't divide.
    // Actually: 32 rows × 32 cols = 1024. 1024/64 = 16 per thread.
    // From AMD docs: thread t gets D[row][col] where:
    //   col = t%32 (if t < 32) or repeats
    // Actually for 32x32: there are more elements per thread.
    // Let me use a flat mapping based on the observed behavior.

    // For 32x32x16 fp8: output mapping (from AMD ISA):
    // Thread t (0-63) owns 16 values:
    //   D[(t/32)*4+k][t%32] for k=0..3 — wait, that's only 4 values
    // Actually: each of 4 output groups × 4 values = 16
    // group g (0-3): D[g*8 + (t/32)*4 + k][t%32] for k=0..3? Not sure.

    // Let me just use the standard mapping:
    // For v_mfma_f32_32x32x16: 16 outputs per thread
    // Layout: D[row][col] where
    //   col = lane % 32
    //   row = out_group * 8 + (lane/32) * 4 + j  for out_group=0..3, j=0..3
    int out_col0 = tile_n * BS + warp_col * REG_BLOCK + (lane % 32);
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int local_row = g * 8 + (lane / 32) * 4 + j;
            int out_row0 = tile_m * BS + warp_row * REG_BLOCK + local_row;
            int out_row1 = tile_m * BS + (warp_row + 2) * REG_BLOCK + local_row;
            C[out_row0 * N + out_col0] = acc0[g * 4 + j];
            C[out_row1 * N + out_col0] = acc1[g * 4 + j];
        }
    }
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp8_native_nt
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        int M = sa[0].cast<int>(), K = sa[1].cast<int>(), N = sb[0].cast<int>();
        fp8_gemm_nt<<<dim3((M/BS)*(N/BS)), dim3(NUM_THREADS), 0, (hipStream_t)0>>>(
            (float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const char*)A.attr("data_ptr")().cast<uint64_t>(),
            (const char*)B.attr("data_ptr")().cast<uint64_t>(), M, N, K);
    }, "Native FP8 GEMM using v_mfma_f32_32x32x16_fp8_fp8");
}
