/**
 * @file tdm_store.cpp
 * @brief Foundation micro-test: dense TDM store on the derived-by-default surface.
 *
 * Exercises the `store_tdm(dst, src, idx)` verb (the store-direction mirror of
 * `load_tdm`) as a LOAD -> STORE round trip: TDM-load a tile from a global
 * source into an `st_bf` LDS tile, then TDM-store that tile straight back out to
 * a fresh dense global buffer. Both verbs derive tensor extents, dense row
 * stride, dtype, tile dims, LDS address and global base from `src`/`dst` + `idx`
 * -- nothing redundant at the call site. A correct load and a correct store
 * compose to the identity, so the destination must equal the source region
 * element-exact.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int ROWS = 64;   // tile rows (A_tile-shaped: 64 x 32)
constexpr int COLS = 32;   // tile cols
constexpr int TILE_ELEMS = ROWS * COLS;
constexpr int NUM_THREADS = 128;

constexpr int SRC_ROWS = 128;  // source tensor is larger than the tile
constexpr int SRC_COLS = 32;

// Non-padded shape: the dense load + store core is verified element-exact under
// the functional model. (Padded TDM pad-field encoding is unverified -- see the
// [UNVERIFIED] note in tdm_descriptor.cuh; the store path ignores pad anyway.)
using A_tile = st_bf<ROWS, COLS, ducks::st_shape::st_16x32>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_store_kernel(const gl<bf16, -1, -1, -1, -1> src,
                      gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    A_tile& A_st = al.allocate<A_tile>();

    if (warpid() == 0) {
        // Derived load: extents/stride from `src`, tile dims/dtype from `A_st`.
        load_tdm(A_st, src, {0, 0, 0, 0});
        sync::wait_tdm();
        // Derived store: extents/stride from `dst`. Same descriptor lowering,
        // store builtin. Writes the tile back out at tile coord {0,0,0,0}.
        store_tdm(dst, A_st, {0, 0, 0, 0});
        sync::wait_tdm();
    }
    sync::sync();
}

int main()
{
    std::printf("tdm_store: load_tdm + store_tdm round trip, tile %dx%d from %dx%d source\n",
                ROWS, COLS, SRC_ROWS, SRC_COLS);

    const int src_total = SRC_ROWS * SRC_COLS;
    std::vector<__hip_bfloat16> h_src(src_total), h_dst(TILE_ELEMS, __hip_bfloat16(0.f));
    for (int i = 0; i < src_total; ++i)
        h_src[i] = __hip_bfloat16(float(i % 257));   // deterministic, bf16-exact

    __hip_bfloat16 *d_src, *d_dst;
    hipMalloc(&d_src, src_total * sizeof(__hip_bfloat16));
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(__hip_bfloat16));
    hipMemcpy(d_src, h_src.data(), src_total * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> src_gl(d_src, 1, 1, size_t(SRC_ROWS), size_t(SRC_COLS));
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(ROWS), size_t(COLS));

    size_t shm = 64 * 1024;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_store_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_store_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
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
