/**
 * @file tdm_3d.cpp
 * @brief Micro-test: 3D affine TDM load (batch x rows x cols) via tdm::affine.
 *
 * The innermost plane (rows x cols) is derived from the `st`; the outer batch
 * axis rides in a `tdm::affine` value (extent, element-stride, tile count).
 * The engine writes BATCH planes contiguously into LDS; we back the `st` with
 * an oversized flat buffer and read it back flat.
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

// Source tensor [batch, rows, cols] carried in gl's (depth, rows, cols).
constexpr int TB = 4, TR = 32, TC = 64;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_3d_kernel(const gl<bf16, -1, -1, -1, -1> src,
                   gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    // The `st` describes only the innermost ROWSxCOLS plane, but the engine
    // writes BATCH such planes back-to-back, so back the tile with a buffer
    // sized for all of them and read it flat.
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, ROWS, COLS, NoPad>*>(&buf[0]);

    // One extra (outer) axis = the batch dimension. Its descriptor triple is
    // (extent, element-stride, tile count): extent TB is the tensor's batch
    // size (for OOB clamping), the stride is one batch plane = TR*TC elements,
    // and we load BATCH planes. The innermost rows/cols + row stride are
    // derived from `tile` and `src`, so they are not repeated here.
    auto a = tdm::affine::make(/*dim2=*/TB, /*stride2=*/TR * TC, /*tile2=*/BATCH);

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
    std::printf("tdm_3d: affine tile [%d,%d,%d] from tensor[%d,%d,%d]\n",
                BATCH, ROWS, COLS, TB, TR, TC);

    constexpr int total = TB * TR * TC;
    std::vector<__hip_bfloat16> h_src(total), h_dst(TILE_ELEMS, __hip_bfloat16(0.f));
    for (int i = 0; i < total; ++i) h_src[i] = __hip_bfloat16(float(i % 1000) * 0.01f);

    __hip_bfloat16 *d_src, *d_dst;
    hipMalloc(&d_src, total * sizeof(__hip_bfloat16));
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(__hip_bfloat16));
    hipMemcpy(d_src, h_src.data(), total * sizeof(__hip_bfloat16), hipMemcpyHostToDevice);
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(__hip_bfloat16));

    // gl: depth=TB carries the batch axis, rows=TR, cols=TC.
    gl<bf16, -1, -1, -1, -1> src_gl(d_src, 1, size_t(TB), size_t(TR), size_t(TC));
    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, 1, size_t(TILE_ELEMS));

    size_t shm = TILE_ELEMS * sizeof(__hip_bfloat16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_3d_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_3d_kernel<<<1, NUM_THREADS, shm, 0>>>(src_gl, dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, TILE_ELEMS * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int b = 0; b < BATCH; ++b)
        for (int r = 0; r < ROWS; ++r)
            for (int c = 0; c < COLS; ++c) {
                int src_idx = b * TR * TC + r * TC + c;
                int dst_idx = b * ROWS * COLS + r * COLS + c;
                float got = static_cast<float>(h_dst[dst_idx]);
                float exp = static_cast<float>(h_src[src_idx]);
                if (std::fabs(got - exp) > 0.001f) {
                    if (errors < 10)
                        std::printf("  MISMATCH [%d,%d,%d]: got=%.4f exp=%.4f\n", b, r, c, got, exp);
                    ++errors;
                }
            }

    std::printf("  errors: %d/%d\n", errors, TILE_ELEMS);
    hipFree(d_src); hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
