/**
 * @file tdm_5d_store.cpp
 * @brief Micro-test: full 5D affine TDM store (groups 0-3) via tdm::affine.
 *
 * Store counterpart of tdm_5d -- exercises all 20 descriptor DWords on the
 * store path. Innermost plane (D1 x D0) derived from the `st`; the three outer
 * axes ride in a `tdm::affine` value. Fills LDS with the D4*D3*D2 inner planes
 * and scatters them back to the matching strided slots of a zeroed tensor.
 *
 *   tensor  T4 x T3 x T2 x T1 x T0 = 4 x 4 x 4 x 32 x 32
 *   window  D4 x D3 x D2 x D1 x D0 = 2 x 2 x 2 x 16 x 16  (8 inner planes)
 *
 *   LDS: [plane 0]...[plane 7]  ->  tensor slot (i4,i3,i2): top-left 16x16
 *   strides: stride2=T1*T0, stride3=T2*T1*T0, stride4=T3*T2*T1*T0.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

// Window: D4 x D3 x D2 x D1 x D0 (innermost D1 x D0 = 16 x 16).
constexpr int D4 = 2, D3 = 2, D2 = 2, D1 = 16, D0 = 16;
constexpr int TILE_ELEMS = D4 * D3 * D2 * D1 * D0;
constexpr int NUM_THREADS = 128;
// Tensor: T4 x T3 x T2 x T1 x T0.
constexpr int T4 = 4, T3 = 4, T2 = 4, T1 = 32, T0 = 32;
constexpr int DST_TOTAL = T4 * T3 * T2 * T1 * T0;
using NoPad = ducks::st_shape::st_16x16;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_5d_store_kernel(gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, D1, D0, NoPad>*>(&buf[0]);

    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        using lds_u16 = uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        __hip_bfloat16 v(float((i % 4000) + 1));
        lds_p[i] = *reinterpret_cast<const uint16_t*>(&v);
    }
    sync::sync();

    // Three outer axes: strides are cumulative products of inner extents.
    auto a = tdm::affine::make(
        /*dim2=*/T2, /*stride2=*/T1 * T0,           /*tile2=*/D2,
        /*dim3=*/T3, /*stride3=*/T2 * T1 * T0,      /*tile3=*/D3,
        /*dim4=*/T4, /*stride4=*/T3 * T2 * T1 * T0, /*tile4=*/D4);

    if (warpid() == 0) {
        store_tdm(dst, tile, {0, 0, 0, 0}, a);
        tdm::store_async_wait();
    }
    sync::sync();
}

int main()
{
    std::printf("tdm_5d_store: tile<%d,%d,%d,%d,%d> into tensor[%d,%d,%d,%d,%d]\n",
                D4, D3, D2, D1, D0, T4, T3, T2, T1, T0);

    std::vector<__hip_bfloat16> h_dst(DST_TOTAL, __hip_bfloat16(0.f));

    __hip_bfloat16 *d_dst;
    hipMalloc(&d_dst, DST_TOTAL * sizeof(__hip_bfloat16));
    hipMemset(d_dst, 0, DST_TOTAL * sizeof(__hip_bfloat16));

    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(T1), size_t(T0));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_5d_store_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_5d_store_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, DST_TOTAL * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    std::vector<__hip_bfloat16> h_ref(DST_TOTAL, __hip_bfloat16(0.f));
    for (int i4 = 0; i4 < D4; ++i4)
      for (int i3 = 0; i3 < D3; ++i3)
        for (int i2 = 0; i2 < D2; ++i2)
          for (int i1 = 0; i1 < D1; ++i1)
            for (int i0 = 0; i0 < D0; ++i0) {
                int dense = i4*(D3*D2*D1*D0) + i3*(D2*D1*D0) + i2*(D1*D0) + i1*D0 + i0;
                int tens  = i4*(T3*T2*T1*T0) + i3*(T2*T1*T0) + i2*(T1*T0) + i1*T0 + i0;
                h_ref[tens] = __hip_bfloat16(float((dense % 4000) + 1));
            }

    int errors = 0;
    for (int i = 0; i < DST_TOTAL; ++i) {
        float got = static_cast<float>(h_dst[i]);
        float exp = static_cast<float>(h_ref[i]);
        if (std::fabs(got - exp) > 0.5f) {
            if (errors < 10)
                std::printf("  MISMATCH idx=%d: got=%.1f exp=%.1f\n", i, got, exp);
            ++errors;
        }
    }

    std::printf("  errors: %d/%d\n", errors, DST_TOTAL);
    hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
