/**
 * @file tdm_padded.cpp
 * @brief Micro-test: TDM load into a padded LDS tile (pad bits derived from st).
 *
 * Uses a padded shared-tile shape (`st_16x32_padded<>`), so the verb derives
 * pad_enable/pad_interval/pad_amount from the `st` and the TDM unit inserts
 * padding in LDS. Reads back through the same padding map and compares.
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
constexpr int SRC_ROWS = 64;
constexpr int SRC_COLS = 64;

using PadShape = ducks::st_shape::st_16x32_padded<>;  // <128, 8>
constexpr int PAD_INTERVAL = 128, PAD_AMOUNT = 8;
constexpr int PAD_ELEMS = TILE_ELEMS + (TILE_ELEMS / PAD_INTERVAL) * PAD_AMOUNT;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_padded_kernel(const gl<bf16, -1, -1, -1, -1> src,
                       gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    bf16(&buf)[PAD_ELEMS] = al.allocate_in<segment<0>, bf16, PAD_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, ROWS, COLS, PadShape>*>(&buf[0]);

    if (warpid() == 0) {
        load_tdm(tile, src, {0, 0, 0, 0});
        tdm::load_async_wait();
    }
    sync::sync();

    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        int padded_idx = i + (i / PAD_INTERVAL) * PAD_AMOUNT;
        using lds_u16 = const uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        reinterpret_cast<uint16_t*>(dst.raw_ptr)[i] = lds_p[padded_idx];
    }
}

int main()
{
    std::printf("tdm_padded: tile<%d,%d> padded<%d,%d>\n",
                ROWS, COLS, PAD_INTERVAL, PAD_AMOUNT);

    constexpr int src_total = SRC_ROWS * SRC_COLS;
    std::vector<__hip_bfloat16> h_src(src_total), h_dst(TILE_ELEMS, __hip_bfloat16(0.f));
    for (int i = 0; i < src_total; ++i) h_src[i] = __hip_bfloat16(float(i));

    __hip_bfloat16 *d_src, *d_dst;
    hipMalloc(&d_src, src_total * sizeof(__hip_bfloat16));
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(__hip_bfloat16));
    hipMemcpy(d_src, h_src.data(), src_total * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> src_gl(d_src, 1, 1, size_t(SRC_ROWS), size_t(SRC_COLS));
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, 1, size_t(TILE_ELEMS));

    size_t shm = PAD_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_padded_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_padded_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, TILE_ELEMS * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            float got = static_cast<float>(h_dst[r * COLS + c]);
            float exp = static_cast<float>(h_src[r * SRC_COLS + c]);
            if (std::fabs(got - exp) > 0.5f) {
                if (errors < 10)
                    std::printf("  MISMATCH [%d,%d]: got=%.1f exp=%.1f\n", r, c, got, exp);
                ++errors;
            }
        }

    std::printf("  errors: %d/%d\n", errors, TILE_ELEMS);
    hipFree(d_src); hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
