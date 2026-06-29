/**
 * @file tdm_store_bits.cpp
 * @brief Bitwise diagnostic for the derive-by-default TDM store.
 *
 * Fills LDS with a distinct 16-bit pattern per element (the linear index),
 * stores via `store_tdm`, then compares the raw uint16 bits read back from
 * global. Immune to bf16 rounding: any permutation/granularity bug shows up
 * as a bit mismatch.
 *
 *   LDS (raw bits = linear index)        dst (top-left 32x64 window)
 *   [0x0000][0x0001][0x0002]...   store  [0x0000][0x0001][0x0002]...
 *   [0x0040][0x0041]...          ======> [0x0040][0x0041]...
 *        ... element i = (u16)i ...           ... must equal i ...
 *   Compared AS BITS, not as bf16 numbers: a reorder or wrong transfer
 *   granularity surfaces as an exact 0xXXXX vs 0xYYYY mismatch.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace kittens;

constexpr int ROWS = 32;
constexpr int COLS = 64;
constexpr int TILE_ELEMS = ROWS * COLS;
constexpr int NUM_THREADS = 128;
constexpr int DST_ROWS = 64;
constexpr int DST_COLS = 128;
using NoPad = ducks::st_shape::st_16x16;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_store_bits_kernel(gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    // Flat LDS buffer viewed as a non-padded `st`; the store source.
    bf16(&buf)[TILE_ELEMS] = al.allocate_in<segment<0>, bf16, TILE_ELEMS>();
    auto& tile = *reinterpret_cast<st<bf16, ROWS, COLS, NoPad>*>(&buf[0]);

    // Write the linear index as the raw 16-bit value of each element. Treating it
    // as bits (not a bf16 number) makes the later compare immune to bf16 rounding:
    // any reorder/granularity bug shows up as a distinct, exact bit mismatch.
    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS) {
        using lds_u16 = uint16_t __attribute__((address_space(3)));
        auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&buf[0]));
        lds_p[i] = static_cast<uint16_t>(i);
    }
    sync::sync();

    // One warp stores the tile to the top-left of `dst`; drain before host readback.
    if (warpid() == 0) {
        store_tdm(dst, tile, {0, 0, 0, 0});
        tdm::store_async_wait();
    }
    sync::sync();
}

int main()
{
    std::printf("tdm_store_bits: tile<%d,%d> into [%d x %d] (bitwise)\n",
                ROWS, COLS, DST_ROWS, DST_COLS);

    constexpr int dst_total = DST_ROWS * DST_COLS;
    std::vector<uint16_t> h_dst(dst_total, 0);

    bf16 *d_dst;
    hipMalloc(&d_dst, dst_total * sizeof(bf16));
    hipMemset(d_dst, 0, dst_total * sizeof(bf16));

    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(DST_ROWS), size_t(DST_COLS));

    size_t shm = TILE_ELEMS * sizeof(bf16) + 256;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_store_bits_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_store_bits_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipDeviceSynchronize();

    hipMemcpy(h_dst.data(), d_dst, dst_total * sizeof(uint16_t), hipMemcpyDeviceToHost);

    // Exact-bit check inside the tile, plus a zero check outside it (the store
    // must leave the surrounding destination untouched).
    int errors = 0;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            uint16_t got = h_dst[r * DST_COLS + c];
            uint16_t exp = static_cast<uint16_t>(r * COLS + c);
            if (got != exp) {
                if (errors < 16)
                    std::printf("  BIT MISMATCH [%d,%d]: got=0x%04x exp=0x%04x\n", r, c, got, exp);
                ++errors;
            }
        }
    for (int r = 0; r < DST_ROWS; ++r)
        for (int c = 0; c < DST_COLS; ++c) {
            if (r < ROWS && c < COLS) continue;
            if (h_dst[r * DST_COLS + c] != 0) {
                if (errors < 16)
                    std::printf("  CORRUPTION [%d,%d]: got=0x%04x (expected 0)\n",
                                r, c, h_dst[r * DST_COLS + c]);
                ++errors;
            }
        }

    std::printf("  errors: %d\n", errors);
    hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
