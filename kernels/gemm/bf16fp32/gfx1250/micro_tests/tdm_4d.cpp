/**
 * @file tdm_4d.cpp
 * @brief Micro-test: 4D affine TDM load via tdm::affine (two outer axes).
 *
 * Fills the dimensionality gap between the 3D and 5D affine tests. The
 * innermost plane (D1 x D0) is derived from the `st`; axes 2 and 3 ride in a
 * `tdm::affine` value as (extent, element-stride, tile) triples.
 *
 *   tensor  T3 x T2 x T1 x T0 = 4 x 4 x 32 x 32
 *   window  D3 x D2 x D1 x D0 = 2 x 2 x 16 x 16
 *
 *   axis0/1 (inner 16x16 plane)  <- derived from `st` + `gl`
 *   axis2/3 (outer nesting)      <- ride in tdm::affine
 *
 *   The window is D3*D2 = 4 inner 16x16 planes; the engine writes them
 *   contiguously into LDS in row-major (axis3 outer ... axis2 inner) order:
 *
 *     LDS: [plane 0][plane 1][plane 2][plane 3]   (each 16x16)
 *
 *   Strides are cumulative products of the tensor's inner extents:
 *     stride2 = T1*T0,  stride3 = T2*T1*T0.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace kittens;

// Tile: D3 x D2 x D1 x D0 (innermost D1 x D0 = 16 x 16).
constexpr int D3 = 2, D2 = 2, D1 = 16, D0 = 16;
constexpr int TILE_ELEMS = D3 * D2 * D1 * D0;
constexpr int NUM_THREADS = 128;
// Tensor: T3 x T2 x T1 x T0 (larger than the window along every axis).
constexpr int T3 = 4, T2 = 4, T1 = 32, T0 = 32;
constexpr int TENSOR_TOTAL = T3 * T2 * T1 * T0;
using NoPad = ducks::st_shape::st_16x16;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_4d_kernel(const gl<bf16, -1, -1, -1, -1> src,
                   gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    // Innermost D1xD0 plane lives in `tile`; the 4 (D3*D2) planes the engine
    // writes are backed by this larger flat buffer and read back flat.
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, D1, D0, NoPad>*>(&buf[0]);

    // Two outer axes, each a (extent, element-stride, tile) triple. axis2 steps
    // over a full T1xT0 plane, axis3 over T2 such planes.
    auto a = tdm::affine::make(
        /*dim2=*/T2, /*stride2=*/T1 * T0,      /*tile2=*/D2,
        /*dim3=*/T3, /*stride3=*/T2 * T1 * T0, /*tile3=*/D3);

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
    std::printf("tdm_4d: tile<%d,%d,%d,%d> from tensor[%d,%d,%d,%d]\n",
                D3, D2, D1, D0, T3, T2, T1, T0);

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
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_4d_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_4d_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, TILE_ELEMS * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    // Map each window element back to its source index via the tensor strides
    // and the (denser) destination strides.
    int errors = 0;
    for (int i3 = 0; i3 < D3; ++i3)
      for (int i2 = 0; i2 < D2; ++i2)
        for (int i1 = 0; i1 < D1; ++i1)
          for (int i0 = 0; i0 < D0; ++i0) {
              int src_idx = i3*(T2*T1*T0) + i2*(T1*T0) + i1*T0 + i0;
              int dst_idx = i3*(D2*D1*D0) + i2*(D1*D0) + i1*D0 + i0;
              float got = static_cast<float>(h_dst[dst_idx]);
              float exp = static_cast<float>(h_src[src_idx]);
              if (std::fabs(got - exp) > 0.001f) {
                  if (errors < 10)
                      std::printf("  MISMATCH [%d,%d,%d,%d]: got=%.4f exp=%.4f\n",
                                  i3, i2, i1, i0, got, exp);
                  ++errors;
              }
          }

    std::printf("  errors: %d/%d\n", errors, TILE_ELEMS);
    hipFree(d_src); hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
