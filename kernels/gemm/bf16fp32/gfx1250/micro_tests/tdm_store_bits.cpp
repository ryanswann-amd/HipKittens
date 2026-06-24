/**
 * @file tdm_store_bits.cpp
 * @brief Foundation micro-test: isolated, bitwise-exact dense TDM store.
 *
 * Unlike tdm_store.cpp (a load->store round trip), this tests `store_tdm` ALONE
 * and at bit granularity. The LDS tile is filled directly with a DISTINCT 16-bit
 * pattern per element (the linear index), then stored to global; the raw uint16
 * bits read back must match exactly. This is immune to bf16 float-rounding and
 * cannot be passed by a compensating load+store bug -- any row/col permutation,
 * pair-swizzle, or granularity error in the store path surfaces as a bit
 * mismatch. The store reads the tile contiguously from `src.data` (tile_dim0 =
 * COLS, tile_dim1 = ROWS, no pad), so LDS slot i corresponds to tile element i.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace kittens;

constexpr int ROWS = 64;
constexpr int COLS = 32;
constexpr int TILE_ELEMS = ROWS * COLS;
constexpr int NUM_THREADS = 128;

using A_tile = st_bf<ROWS, COLS, ducks::st_shape::st_16x32>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tdm_store_bits_kernel(gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    A_tile& A_st = al.allocate<A_tile>();

    // Fill the LDS tile contiguously with a distinct 16-bit pattern per element.
    using lds_u16 = uint16_t __attribute__((address_space(3)));
    auto* lds_p = (lds_u16*)(reinterpret_cast<uintptr_t>(&A_st.data[0]));
    for (int i = threadIdx.x; i < TILE_ELEMS; i += NUM_THREADS)
        lds_p[i] = static_cast<uint16_t>(i);
    sync::sync();

    if (warpid() == 0) {
        store_tdm(dst, A_st, {0, 0, 0, 0});
        sync::wait_tdm();
    }
    sync::sync();
}

int main()
{
    std::printf("tdm_store_bits: isolated store_tdm, tile %dx%d (bitwise)\n", ROWS, COLS);

    std::vector<uint16_t> h_dst(TILE_ELEMS, 0);

    bf16 *d_dst;
    hipMalloc(&d_dst, TILE_ELEMS * sizeof(bf16));
    hipMemset(d_dst, 0, TILE_ELEMS * sizeof(bf16));

    gl<bf16, -1, -1, -1, -1> dst_gl(d_dst, 1, 1, size_t(ROWS), size_t(COLS));

    size_t shm = 64 * 1024;
    hipFuncSetAttribute(reinterpret_cast<const void*>(tdm_store_bits_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shm);
    tdm_store_bits_kernel<<<1, NUM_THREADS, shm, 0>>>(dst_gl);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::printf("  kernel error: %s\n", hipGetErrorString(err));
        return 2;
    }

    hipMemcpy(h_dst.data(), d_dst, TILE_ELEMS * sizeof(uint16_t), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < TILE_ELEMS; ++i) {
        uint16_t exp = static_cast<uint16_t>(i);
        if (h_dst[i] != exp) {
            if (errors < 16)
                std::printf("  BIT MISMATCH [%d]: got=0x%04x exp=0x%04x\n",
                            i, h_dst[i], exp);
            ++errors;
        }
    }

    std::printf("  errors: %d/%d\n", errors, TILE_ELEMS);
    hipFree(d_dst);
    return errors > 0 ? 1 : 0;
}
