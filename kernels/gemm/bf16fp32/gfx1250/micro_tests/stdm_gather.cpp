/**
 * @file stdm_gather.cpp
 * @brief Micro-test: row-indexed gather via load_tdm(st, gl, rows, n).
 *
 * Gathers 16 non-contiguous rows (uint16 index mode) of a 2D matrix into a
 * contiguous LDS tile and verifies each gathered row matches the source.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int GATHER_ROWS = 16;
constexpr int GATHER_COLS = 32;
constexpr int TILE_ELEMS  = GATHER_ROWS * GATHER_COLS;
constexpr int NUM_THREADS = 128;
constexpr int SRC_ROWS = 256;
constexpr int SRC_COLS = 64;
using NoPad = ducks::st_shape::st_8x32;

__global__ __launch_bounds__(NUM_THREADS, 1)
void stdm_gather_kernel(const gl<bf16, -1, -1, -1, -1> src,
                        gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, GATHER_ROWS, GATHER_COLS, NoPad>*>(&buf[0]);

    uint32_t rows[16] = {0, 15, 30, 45, 60, 75, 90, 105,
                         120, 135, 150, 165, 180, 195, 210, 225};

    if (warpid() == 0) {
        load_tdm(tile, src, rows, 16, {.idx_w = tdm::idx_width::b16});
        tdm::load_async_wait();
    }
    sync::sync();

    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        using lds_u16 = const uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        reinterpret_cast<uint16_t*>(dst.raw_ptr)[i] = lds_p[i];
    }
}

int main()
{
    std::printf("stdm_gather: %d rows x %d cols from [%d x %d]\n",
                GATHER_ROWS, GATHER_COLS, SRC_ROWS, SRC_COLS);

    constexpr int src_total = SRC_ROWS * SRC_COLS;
    std::vector<__hip_bfloat16> h_src(src_total), h_dst(TILE_ELEMS, __hip_bfloat16(0.f));
    for (int r = 0; r < SRC_ROWS; ++r)
        for (int c = 0; c < SRC_COLS; ++c)
            h_src[r * SRC_COLS + c] = __hip_bfloat16(float(r * 100 + c));

    __hip_bfloat16 *d_src, *d_dst;
    hipMalloc(&d_src, src_total * sizeof(__hip_bfloat16));
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(__hip_bfloat16));
    hipMemcpy(d_src, h_src.data(), src_total * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> src_gl(d_src, 1, 1, size_t(SRC_ROWS), size_t(SRC_COLS));
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, 1, size_t(TILE_ELEMS));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(stdm_gather_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    stdm_gather_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, TILE_ELEMS * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int rows_ref[16] = {0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225};
    int errors = 0;
    for (int gi = 0; gi < GATHER_ROWS; ++gi) {
        int src_row = rows_ref[gi];
        for (int c = 0; c < GATHER_COLS; ++c) {
            float got = static_cast<float>(h_dst[gi * GATHER_COLS + c]);
            float exp = static_cast<float>(h_src[src_row * SRC_COLS + c]);
            if (std::fabs(got - exp) > 0.5f) {
                if (errors < 10)
                    std::printf("  MISMATCH row[%d] (src=%d) col=%d: got=%.1f exp=%.1f\n",
                                gi, src_row, c, got, exp);
                ++errors;
            }
        }
    }

    std::printf("  errors: %d/%d\n", errors, TILE_ELEMS);
    hipFree(d_src); hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
