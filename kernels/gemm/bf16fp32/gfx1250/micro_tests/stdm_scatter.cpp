/**
 * @file stdm_scatter.cpp
 * @brief Micro-test: row-indexed scatter via store_tdm(gl, st, rows, n).
 *
 * Fills LDS with a known pattern and scatters 8 rows (uint32 index mode) to
 * non-contiguous rows of a 2D matrix; verifies the target rows and that the
 * rest stays zero.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int SCATTER_ROWS = 8;
constexpr int SCATTER_COLS = 32;
constexpr int TILE_ELEMS   = SCATTER_ROWS * SCATTER_COLS;
constexpr int NUM_THREADS  = 128;
constexpr int DST_ROWS = 128;
constexpr int DST_COLS = 64;
using NoPad = ducks::st_shape::st_8x32;

__global__ __launch_bounds__(NUM_THREADS, 1)
void stdm_scatter_kernel(gl<bf16, -1, -1, -1, -1> dst)
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

    uint32_t rows[8] = {10, 20, 30, 40, 50, 60, 70, 80};

    if (warpid() == 0) {
        store_tdm(dst, tile, rows, 8, {.idx_w = tdm::idx_width::b32});
        tdm::store_async_wait();
    }
    sync::sync();
}

int main()
{
    std::printf("stdm_scatter: %d rows x %d cols into [%d x %d]\n",
                SCATTER_ROWS, SCATTER_COLS, DST_ROWS, DST_COLS);

    constexpr int dst_total = DST_ROWS * DST_COLS;
    std::vector<__hip_bfloat16> h_dst(dst_total, __hip_bfloat16(0.f));

    __hip_bfloat16 *d_dst;
    hipMalloc(&d_dst, dst_total * sizeof(__hip_bfloat16));
    hipMemset(d_dst, 0, dst_total * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(DST_ROWS), size_t(DST_COLS));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(stdm_scatter_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    stdm_scatter_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, dst_total * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int target_rows[8] = {10, 20, 30, 40, 50, 60, 70, 80};
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
