/**
 * @file tdm_store.cpp
 * @brief Micro-test: 2D dense TDM store (LDS -> global) via store_tdm.
 *
 * Fills LDS with a float pattern, stores a ROWSxCOLS tile into the top-left
 * of a larger destination matrix with `kittens::store_tdm(gl, st, coord)`,
 * and checks the destination region (bf16-rounded reference).
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
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, ROWS, COLS, NoPad>*>(&buf[0]);

    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        int r = i / COLS, c = i % COLS;
        using lds_u16 = uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        __hip_bfloat16 v(float(r * 100 + c + 1));
        lds_p[i] = *reinterpret_cast<const uint16_t*>(&v);
    }
    sync::sync();

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
