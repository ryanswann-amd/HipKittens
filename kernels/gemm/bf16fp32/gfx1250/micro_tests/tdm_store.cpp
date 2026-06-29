/**
 * @file tdm_store.cpp
 * @brief Micro-test: 2D dense TDM store (LDS -> global) via store_tdm.
 *
 * Fills LDS with a float pattern, stores a ROWSxCOLS tile into the top-left
 * of a larger destination matrix with `kittens::store_tdm(gl, st, coord)`,
 * and checks the destination region (bf16-rounded reference).
 *
 *   LDS tile: 32 x 32              dst: 64 x 64 (rest must stay 0)
 *   +-------------+                +--------------------------------+
 *   |   32 x 32   |    store       | 32x32 |                        |
 *   |  (pattern)  | =========>     | tile  |        0 ...           |
 *   +-------------+                |-------+                        |
 *                                  |              0 ...             |
 *                                  +--------------------------------+
 *   coord {0,0,0,0} -> only the top-left 32x32 window is written.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int ROWS = 32;
constexpr int COLS = 32;
constexpr int TILE_ELEMS = ROWS * COLS;
constexpr int NUM_THREADS = 128;
constexpr int DST_ROWS = 64;
constexpr int DST_COLS = 64;
using NoPad = ducks::st_shape::st_16x16;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_store_kernel(gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    // Flat LDS buffer viewed as a non-padded `st`; this is the store *source*.
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, ROWS, COLS, NoPad>*>(&buf[0]);

    // All threads seed LDS with a per-element pattern (r*100+c+1) before the store.
    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        int r = i / COLS, c = i % COLS;
        using lds_u16 = uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        __hip_bfloat16 v(float(r * 100 + c + 1));
        lds_p[i] = *reinterpret_cast<const uint16_t*>(&v);
    }
    sync::sync();   // all rows written before the engine reads LDS

    // One warp stores the tile to the top-left of `dst` (coord {b,d,row,col}=0).
    // store_tdm mirrors load_tdm with the gl/st argument order flipped.
    if (warpid() == 0) {
        store_tdm(dst, tile, {0, 0, 0, 0});
        tdm::store_async_wait();
    }
    sync::sync();
}

int main()
{
    std::printf("tdm_store: tile<%d,%d> into [%d x %d]\n", ROWS, COLS, DST_ROWS, DST_COLS);

    constexpr int dst_total = DST_ROWS * DST_COLS;
    std::vector<__hip_bfloat16> h_dst(dst_total, __hip_bfloat16(0.f));

    __hip_bfloat16 *d_dst;
    hipMalloc(&d_dst, dst_total * sizeof(__hip_bfloat16));
    hipMemset(d_dst, 0, dst_total * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(DST_ROWS), size_t(DST_COLS));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_store_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_store_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, dst_total * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    // Inside the tile region expect the seeded pattern; outside it expect zero
    // (the store must not touch the rest of the destination matrix).
    int errors = 0;
    for (int r = 0; r < DST_ROWS; ++r)
        for (int c = 0; c < DST_COLS; ++c) {
            float got = static_cast<float>(h_dst[r * DST_COLS + c]);
            float exp = (r < ROWS && c < COLS)
                      ? static_cast<float>(__hip_bfloat16(float(r * 100 + c + 1))) : 0.f;
            if (std::fabs(got - exp) > 0.5f) {
                if (errors < 10)
                    std::printf("  MISMATCH [%d,%d]: got=%.1f exp=%.1f\n", r, c, got, exp);
                ++errors;
            }
        }

    std::printf("  errors: %d/%d\n", errors, dst_total);
    hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
