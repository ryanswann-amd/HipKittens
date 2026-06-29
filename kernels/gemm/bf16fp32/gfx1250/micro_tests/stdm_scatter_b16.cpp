/**
 * @file stdm_scatter_b16.cpp
 * @brief Micro-test: row-indexed scatter with 16-bit indices (idx_width::b16).
 *
 * Same as stdm_scatter but exercises the narrow-index packing: b16 fits 16
 * indices per descriptor (vs 8 for b32). Fills LDS with a known pattern and
 * scatters 16 rows to non-contiguous rows of a 2D matrix; verifies the target
 * rows and that the rest stays zero.
 *
 *   LDS tile: 16 x 32 (pattern)         dst: 256 x 64 (rest stays 0)
 *   +-------------+                      dst row 5   <- LDS row 0
 *   | row 0 ------\------- scatter ----> dst row 20  <- LDS row 1
 *   | row 1 -------\----------------\--> dst row 35  <- LDS row 2
 *   |  ...                           \       ...
 *   +-------------+                      dst row 230 <- LDS row 15
 *   rows[] (16 entries, b16) -> LDS row i written to dst row rows[i].
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int SCATTER_ROWS = 16;        // b16 cap is 16 indices per descriptor
constexpr int SCATTER_COLS = 32;
constexpr int TILE_ELEMS   = SCATTER_ROWS * SCATTER_COLS;
constexpr int NUM_THREADS  = 128;
constexpr int DST_ROWS = 256;
constexpr int DST_COLS = 64;
using NoPad = ducks::st_shape::st_16x16;

__global__ __launch_bounds__(NUM_THREADS, 1)
void stdm_scatter_b16_kernel(gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, SCATTER_ROWS, SCATTER_COLS, NoPad>*>(&buf[0]);

    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        int r = i / SCATTER_COLS, c = i % SCATTER_COLS;
        using lds_u16 = uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        __hip_bfloat16 v(float(r * 100 + c + 1));
        lds_p[i] = *reinterpret_cast<const uint16_t*>(&v);
    }
    sync::sync();

    // 16 sorted, unique destination rows -> one b16 descriptor.
    uint32_t rows[16] = {5, 20, 35, 50, 65, 80, 95, 110,
                         125, 140, 155, 170, 185, 200, 215, 230};

    if (warpid() == 0) {
        store_tdm(dst, tile, rows, 16, {.idx_w = tdm::idx_width::b16});
        tdm::store_async_wait();
    }
    sync::sync();
}

int main()
{
    std::printf("stdm_scatter_b16: %d rows x %d cols (b16) into [%d x %d]\n",
                SCATTER_ROWS, SCATTER_COLS, DST_ROWS, DST_COLS);

    constexpr int dst_total = DST_ROWS * DST_COLS;
    std::vector<__hip_bfloat16> h_dst(dst_total, __hip_bfloat16(0.f));

    __hip_bfloat16 *d_dst;
    hipMalloc(&d_dst, dst_total * sizeof(__hip_bfloat16));
    hipMemset(d_dst, 0, dst_total * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(DST_ROWS), size_t(DST_COLS));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(stdm_scatter_b16_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    stdm_scatter_b16_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, dst_total * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int target_rows[16] = {5, 20, 35, 50, 65, 80, 95, 110,
                           125, 140, 155, 170, 185, 200, 215, 230};
    int errors = 0;
    for (int gi = 0; gi < SCATTER_ROWS; ++gi) {
        int dst_row = target_rows[gi];
        for (int c = 0; c < SCATTER_COLS; ++c) {
            float got = static_cast<float>(h_dst[dst_row * DST_COLS + c]);
            float exp = static_cast<float>(__hip_bfloat16(float(gi * 100 + c + 1)));
            if (std::fabs(got - exp) > 0.5f) {
                if (errors < 10)
                    std::printf("  MISMATCH row=%d col=%d: got=%.1f exp=%.1f\n", dst_row, c, got, exp);
                ++errors;
            }
        }
    }
    for (int r = 0; r < DST_ROWS; ++r) {
        bool is_target = false;
        for (int gi = 0; gi < SCATTER_ROWS; ++gi) if (target_rows[gi] == r) is_target = true;
        if (is_target) continue;
        for (int c = 0; c < SCATTER_COLS; ++c) {
            float v = static_cast<float>(h_dst[r * DST_COLS + c]);
            if (v != 0.f) {
                if (errors < 10)
                    std::printf("  CORRUPTION row=%d col=%d: got=%.1f (expected 0)\n", r, c, v);
                ++errors;
            }
        }
    }

    std::printf("  errors: %d\n", errors);
    hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
