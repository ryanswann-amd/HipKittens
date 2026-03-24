// FP8 NT GEMM: C = A @ B^T where A(fp8, MxK), B(fp8, NxK) → C(bf16, MxN)
//
// Loads FP8 (e4m3fnuz) from global, converts to BF16 during global→shared
// store, then runs BF16 MFMA. FP8 loads are 2x more bandwidth-efficient
// than BF16 (16 elements per float4 vs 8).

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <hip/hip_fp8.h>
using namespace kittens;

#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 256
#endif
#ifndef K_STEP_VAL
#define K_STEP_VAL 64
#endif

constexpr int BLOCK_SIZE = BLOCK_SIZE_VAL;
constexpr int K_STEP = K_STEP_VAL;
constexpr int REG_BLOCK = BLOCK_SIZE / 4;
constexpr int DOT_SLICE = 16;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_C = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    const __hip_fp8_e4m3_fnuz* a;  // FP8 input A: MxK
    const __hip_fp8_e4m3_fnuz* b;  // FP8 input B: NxK
    _gl_C c;
    int M_dim, N_dim, K_dim;
    hipStream_t stream;
    dim3 grid()  { return dim3((N_dim / BLOCK_SIZE) * (M_dim / BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

// Load FP8 from global, convert to BF16, store to shared
template<int N_THR>
__device__ void load_fp8_to_shared(
    st_bf<BLOCK_SIZE, K_STEP>& dst,
    const __hip_fp8_e4m3_fnuz* src,
    int stride, int rows, int cols)
{
    // FP8: 1 byte per element. float4 = 16 bytes = 16 fp8 elements
    constexpr int FP8_PER_LOAD = 16;
    constexpr int LOADS_PER_ROW = K_STEP / FP8_PER_LOAD;  // 64/16 = 4
    constexpr int TOTAL = BLOCK_SIZE * LOADS_PER_ROW;

    for (int idx = threadIdx.x; idx < TOTAL; idx += N_THR) {
        int r = idx / LOADS_PER_ROW;
        int c_base = (idx % LOADS_PER_ROW) * FP8_PER_LOAD;

        // Load 16 fp8 values (16 bytes = float4)
        float4 raw = *(const float4*)&src[r * stride + c_base];
        const __hip_fp8_e4m3_fnuz* fp8_vals = (const __hip_fp8_e4m3_fnuz*)&raw;

        // Convert 16 fp8 → 16 bf16, store in 8 pairs (ds_write_b32 each)
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            bf16 b0 = __float2bfloat16((float)fp8_vals[i]);
            bf16 b1 = __float2bfloat16((float)fp8_vals[i + 1]);
            bf16 pair[2] = {b0, b1};
            bf16* d = dst.idx(&dst.data[0], {r, c_base + i});
            *(uint32_t*)d = *(uint32_t*)pair;
        }
    }
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    rt_bf<REG_BLOCK, DOT_SLICE> tiles[8];
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    const int num_pid_m = ceil_div(g.M_dim, BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N_dim, BLOCK_SIZE);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int row = pid_m;
    const int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int num_tiles = g.K_dim / K_STEP;

    // Prologue
    load_fp8_to_shared<NUM_THREADS>(As, g.a + row * BLOCK_SIZE * g.K_dim, g.K_dim, BLOCK_SIZE, K_STEP);
    load_fp8_to_shared<NUM_THREADS>(Bs, g.b + col * BLOCK_SIZE * g.K_dim, g.K_dim, BLOCK_SIZE, K_STEP);
    __builtin_amdgcn_s_barrier();
    if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }

    // Main loop (no register buffer pipeline — FP8 loads need conversion)
    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(tiles[6], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
        load(tiles[7], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        // Load next tiles with FP8→BF16 conversion
        asm volatile("s_waitcnt lgkmcnt(0)");
        load_fp8_to_shared<NUM_THREADS>(As, g.a + row * BLOCK_SIZE * g.K_dim + (tile+1) * K_STEP, g.K_dim, BLOCK_SIZE, K_STEP);
        load_fp8_to_shared<NUM_THREADS>(Bs, g.b + col * BLOCK_SIZE * g.K_dim + (tile+1) * K_STEP, g.K_dim, BLOCK_SIZE, K_STEP);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[7], tiles[6], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[6], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    __builtin_amdgcn_sched_barrier(0);
    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }
    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp8_nt
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        int Msz = sa[0].cast<int>(), Ksz = sa[1].cast<int>(), Nsz = sb[0].cast<int>();
        auto gc = kittens::make_gl<_gl_C>(C.attr("data_ptr")().cast<uint64_t>(), 1, 1, Msz, Nsz);
        micro_globals g{
            (const __hip_fp8_e4m3_fnuz*)A.attr("data_ptr")().cast<uint64_t>(),
            (const __hip_fp8_e4m3_fnuz*)B.attr("data_ptr")().cast<uint64_t>(),
            gc, Msz, Nsz, Ksz, (hipStream_t)0};
        unsigned long mem = 65536;
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
        micro_tk<<<dim3((Nsz/BLOCK_SIZE)*(Msz/BLOCK_SIZE)), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
    }, "FP8 NT GEMM: loads FP8 e4m3, computes in BF16 MFMA, outputs BF16");
}
