/**
 * @file tdm_dense.cpp
 * @brief Foundation micro-test: dense TDM load on the derived-by-default surface.
 *
 * Exercises milestone 2 of tdm-implementation-design.md: the 3-arg
 * `load_tdm(dst, src, idx)` verb whose tensor extents, dense row stride, dtype,
 * tile dims, LDS padding, LDS address and global base are ALL derived from
 * `dst` + `src` + `idx` (no explicit M/K/K, no `load_tdm_arrive`).
 *
 * The kernel TDM-loads a ROWS x COLS tile straight into a padded `st_bf` LDS
 * tile, drains with `wait_tdm()`, then copies the tile back to a dense global
 * buffer through the tile's own `lds_offset()` map. Host then checks the
 * round-trip is element-exact against the source region.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int ROWS = 64;   // tile rows  (A_tile-shaped: 64 x 32)
constexpr int COLS = 32;   // tile cols
constexpr int TILE_ELEMS = ROWS * COLS;
constexpr int NUM_THREADS = 128;

constexpr int SRC_ROWS = 128; // source tensor is larger than the tile
constexpr int SRC_COLS = 32;

// Non-padded shape: the dense load + row-stride core is verified element-exact
// under the functional model. (The padded shape's TDM pad-field encoding is unverified --
// see the [UNVERIFIED] note in tdm_descriptor.cuh.)
using A_tile = st_bf<ROWS, COLS, ducks::st_shape::st_16x32>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_dense_kernel(const gl<bf16, -1, -1, -1, -1> src,
                      gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    A_tile& A_st = al.allocate<A_tile>();

    // Derived-by-default load: extents (SRC_ROWS x SRC_COLS) and the dense row
    // stride come from `src`; the tile dims / dtype / padding come from `A_st`.
    if (warpid() == 0) {
        load_tdm(A_st, src, {0, 0, 0, 0});
        sync::wait_tdm();
    }
    sync::sync();

    // Copy LDS tile -> dense global through the tile's padded address map.
    using lds_u16 = const uint16_t __attribute__((address_space(3)));
    auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&A_st.data[0]));
    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        reinterpret_cast<uint16_t*>(dst.raw_ptr)[i] = lds_p[A_tile::lds_offset(i)];
    }
}

int main()
{
    std::printf("tdm_dense: derived load_tdm(dst, src, idx), tile %dx%d from %dx%d source\n",
                ROWS, COLS, SRC_ROWS, SRC_COLS);

    const int src_total = SRC_ROWS * SRC_COLS;
    std::vector<__hip_bfloat16> h_src(src_total), h_dst(TILE_ELEMS, __hip_bfloat16(0.f));
    for (int i = 0; i < src_total; ++i)
        h_src[i] = __hip_bfloat16(float(i % 257));   // deterministic pattern

    __hip_bfloat16 *d_src, *d_dst;
    hipMalloc(&d_src, src_total * sizeof(__hip_bfloat16));
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(__hip_bfloat16));
    hipMemcpy(d_src, h_src.data(), src_total * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> src_gl(d_src, 1, 1, size_t(SRC_ROWS), size_t(SRC_COLS));
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, 1, size_t(TILE_ELEMS));

    size_t shm = 64 * 1024;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_dense_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_dense_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::printf("  kernel error: %s\n", hipGetErrorString(err));
        return 2;
    }

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
