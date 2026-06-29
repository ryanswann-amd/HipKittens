/**
 * @file tdm_5d.cpp
 * @brief Micro-test: full 5D affine TDM load (groups 0-3) via tdm::affine.
 *
 * Innermost plane (D1 x D0) derived from the `st`; the three outer axes ride
 * in a `tdm::affine` value. Exercises all 20 descriptor DWords.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

// Tile: D4 x D3 x D2 x D1 x D0 (innermost D1 x D0 = 16 x 16).
constexpr int D4 = 2, D3 = 2, D2 = 2, D1 = 16, D0 = 16;
constexpr int TILE_ELEMS = D4 * D3 * D2 * D1 * D0;
constexpr int NUM_THREADS = 128;
// Tensor: T4 x T3 x T2 x T1 x T0.
constexpr int T4 = 4, T3 = 4, T2 = 4, T1 = 32, T0 = 32;
constexpr int TENSOR_TOTAL = T4 * T3 * T2 * T1 * T0;
using NoPad = ducks::st_shape::st_16x16;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_5d_kernel(const gl<bf16, -1, -1, -1, -1> src,
                   gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, D1, D0, NoPad>*>(&buf[0]);

    // Three outer axes (innermost-first: axis2, axis3, axis4).
    auto a = tdm::affine::make(
        /*dim2=*/T2, /*stride2=*/T1 * T0,           /*tile2=*/D2,
        /*dim3=*/T3, /*stride3=*/T2 * T1 * T0,      /*tile3=*/D3,
        /*dim4=*/T4, /*stride4=*/T3 * T2 * T1 * T0, /*tile4=*/D4);

    if (warpid() == 0) {
        load_tdm(tile, src, {0, 0, 0, 0}, a);
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
    std::printf("tdm_5d: tile<%d,%d,%d,%d,%d> from tensor[%d,%d,%d,%d,%d]\n",
                D4, D3, D2, D1, D0, T4, T3, T2, T1, T0);

    std::vector<__hip_bfloat16> h_src(TENSOR_TOTAL), h_dst(TILE_ELEMS, __hip_bfloat16(0.f));
    for (int i = 0; i < TENSOR_TOTAL; ++i) h_src[i] = __hip_bfloat16(float(i % 2000) * 0.005f);

    __hip_bfloat16 *d_src, *d_dst;
    hipMalloc(&d_src, TENSOR_TOTAL * sizeof(__hip_bfloat16));
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(__hip_bfloat16));
    hipMemcpy(d_src, h_src.data(), TENSOR_TOTAL * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(__hip_bfloat16));

    // gl carries the innermost plane extents (rows=T1, cols=T0); outer axes
    // come from the affine value, base at the tensor origin.
    gl<bf16, -1, -1, -1, -1> src_gl(d_src, 1, 1, size_t(T1), size_t(T0));
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, 1, size_t(TILE_ELEMS));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_5d_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_5d_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, TILE_ELEMS * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i4 = 0; i4 < D4; ++i4)
      for (int i3 = 0; i3 < D3; ++i3)
        for (int i2 = 0; i2 < D2; ++i2)
          for (int i1 = 0; i1 < D1; ++i1)
            for (int i0 = 0; i0 < D0; ++i0) {
                int src_idx = i4*(T3*T2*T1*T0) + i3*(T2*T1*T0) + i2*(T1*T0) + i1*T0 + i0;
                int dst_idx = i4*(D3*D2*D1*D0) + i3*(D2*D1*D0) + i2*(D1*D0) + i1*D0 + i0;
                float got = static_cast<float>(h_dst[dst_idx]);
                float exp = static_cast<float>(h_src[src_idx]);
                if (std::fabs(got - exp) > 0.001f) {
                    if (errors < 10)
                        std::printf("  MISMATCH [%d,%d,%d,%d,%d]: got=%.4f exp=%.4f\n",
                                    i4, i3, i2, i1, i0, got, exp);
                    ++errors;
                }
            }

    std::printf("  errors: %d/%d\n", errors, TILE_ELEMS);
    hipFree(d_src); hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
