/**
 * @file tdm_iterate_hardstop.cpp
 * @brief Negative compile-test: iterate mode is hard-stopped (D9).
 *
 * Passing a `tdm::iterate` value to `load_tdm` MUST fail to compile (the
 * iterate sub-field offsets are unverified; the functional model tags
 * iterate as untested). This file is expected to NOT compile -- the test
 * runner asserts a compile *failure* and treats a successful build as a
 * regression. See `run_tdm_tests.sh`.
 */

#include "kittens.cuh"
#include <hip/hip_runtime.h>

using namespace kittens;

__global__ void iterate_should_not_compile(const gl<bf16, -1, -1, -1, -1> src,
                                           gl<bf16, -1, -1, -1, -1> dst)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    bf16(&buf)[512] = al.allocate_in<segment<0>, bf16, 512>();
    auto& tile = *reinterpret_cast<st<bf16, 16, 32, ducks::st_shape::st_16x16>*>(&buf[0]);

    auto it = tdm::iterate::make(/*lds_inc=*/16, /*gbl_inc=*/4, /*count=*/3);
    // EXPECTED: static_assert failure -- iterate is hard-stopped (D9).
    load_tdm(tile, src, {0, 0, 0, 0}, it);
}

int main() { return 0; }
