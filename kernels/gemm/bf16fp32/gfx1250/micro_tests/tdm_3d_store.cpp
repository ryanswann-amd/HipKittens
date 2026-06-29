/**
 * @file tdm_3d_store.cpp
 * @brief Micro-test: 3D affine TDM store (LDS -> global) via tdm::affine.
 *
 * The store path mirrors the affine load: the innermost plane (rows x cols)
 * is derived from the `st`; the outer batch axis rides in a `tdm::affine`
 * value. Fills LDS with BATCH stacked planes and scatters them back to the
 * first BATCH planes of a larger zeroed tensor.
 *
 *   LDS: BATCH=2 stacked planes        dst tensor [TB=4][TR=32][TC=64]
 *   +-----------+                       plane0   plane1   plane2  plane3
 *   | plane 0   | (16x32)   store      +------+ +------+ +----+ +----+
 *   +-----------+  =========>          |16x32 | |16x32 | | 0  | | 0  |
 *   | plane 1   | (16x32)              | <-L0 | | <-L1 | |    | |    |
 *   +-----------+                      +------+ +------+ +----+ +----+
 *   affine axis2 = (extent=TB, stride2=TR*TC elems, tile2=BATCH).
 *   Within each written plane only the top-left 16x32 is touched; the rest of
 *   the tensor (plane interiors and planes >= BATCH) must stay zero.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

constexpr int BATCH = 2;
constexpr int ROWS  = 16;
constexpr int COLS  = 32;
constexpr int TILE_ELEMS = BATCH * ROWS * COLS;
constexpr int NUM_THREADS = 128;
using NoPad = ducks::st_shape::st_16x16;

// Destination tensor [batch, rows, cols] carried in gl's (depth, rows, cols).
constexpr int TB = 4, TR = 32, TC = 64;
constexpr int DST_TOTAL = TB * TR * TC;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_3d_store_kernel(gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    // The `st` describes one ROWSxCOLS plane, but we hold BATCH planes back to
    // back in a flat buffer (the store source).
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, ROWS, COLS, NoPad>*>(&buf[0]);

    // Seed all BATCH planes with a per-element pattern (plane*10000+r*100+c+1)
    // so a mismatch reveals which plane/row/col was written wrong.
    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        int b = i / (ROWS * COLS), rem = i % (ROWS * COLS);
        int r = rem / COLS, c = rem % COLS;
        using lds_u16 = uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        __hip_bfloat16 v(float(b * 10000 + r * 100 + c + 1));
        lds_p[i] = *reinterpret_cast<const uint16_t*>(&v);
    }
    sync::sync();

    // One outer (batch) axis: (extent=TB for OOB, stride=one plane, tile=BATCH).
    auto a = tdm::affine::make(/*dim2=*/TB, /*stride2=*/TR * TC, /*tile2=*/BATCH);

    if (warpid() == 0) {
        store_tdm(dst, tile, {0, 0, 0, 0}, a);
        tdm::store_async_wait();
    }
    sync::sync();
}

int main()
{
    std::printf("tdm_3d_store: affine tile [%d,%d,%d] into tensor[%d,%d,%d]\n",
                BATCH, ROWS, COLS, TB, TR, TC);

    std::vector<__hip_bfloat16> h_dst(DST_TOTAL, __hip_bfloat16(0.f));

    __hip_bfloat16 *d_dst;
    hipMalloc(&d_dst, DST_TOTAL * sizeof(__hip_bfloat16));
    hipMemset(d_dst, 0, DST_TOTAL * sizeof(__hip_bfloat16));

    // gl carries the innermost plane (rows=TR, cols=TC); the batch axis comes
    // from the affine value, base at the tensor origin.
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(TR), size_t(TC));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_3d_store_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_3d_store_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, DST_TOTAL * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int b = 0; b < TB; ++b)
        for (int r = 0; r < TR; ++r)
            for (int c = 0; c < TC; ++c) {
                float got = static_cast<float>(h_dst[b * TR * TC + r * TC + c]);
                // Inside the written window expect the pattern; everywhere else 0.
                float exp = (b < BATCH && r < ROWS && c < COLS)
                          ? static_cast<float>(__hip_bfloat16(float(b * 10000 + r * 100 + c + 1)))
                          : 0.f;
                if (std::fabs(got - exp) > 0.5f) {
                    if (errors < 10)
                        std::printf("  MISMATCH [%d,%d,%d]: got=%.1f exp=%.1f\n", b, r, c, got, exp);
                    ++errors;
                }
            }

    std::printf("  errors: %d/%d\n", errors, DST_TOTAL);
    hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
