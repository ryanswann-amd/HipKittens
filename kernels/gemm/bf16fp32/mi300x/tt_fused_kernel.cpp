// BF16 TT GEMM via fused transpose+NT: C = A^T @ B^T where A is KxM, B is NxK
//
// Phase 1: Transpose A_KxM → A_MxK (bandwidth-limited)
// Phase 2: NT kernel with A_MxK × B_NxK^T = MxN (580 TF, 0 spills)
// B is already NxK — same layout as NT, no transpose needed.

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
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

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

// ===================== Phase 1: Transpose Kernel =====================
// Simple, bandwidth-optimal transpose using shared memory tile.
// Each block transposes a 32×32 tile of bf16 values.
constexpr int TILE_DIM = 32;

__global__ __launch_bounds__(256)
void transpose_kernel(
    bf16* __restrict__ dst,       // NxK output
    const bf16* __restrict__ src, // KxN input
    int K, int N)
{
    __shared__ bf16 tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load from src (KxN) — coalesced along N
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += 8) {
        if ((y + j) < K && x < N)
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * N + x];
    }
    __syncthreads();

    // Write to dst (NxK) — coalesced along K
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += 8) {
        if ((y + j) < N && x < K)
            dst[(y + j) * K + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ===================== Phase 2: NT GEMM Kernel =====================
// Exact copy of cdna3_kernel.cpp's micro_tk — the proven 580 TF kernel.

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    int M_dim, N_dim, K_dim;
    hipStream_t stream;
    dim3 grid()  { return dim3((N_dim / BLOCK_SIZE) * (M_dim / BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

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

    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        constexpr int BUFFER_SIZE = (BLOCK_SIZE * K_STEP) / NUM_THREADS;
        float4 a_buffer_next[BUFFER_SIZE * sizeof(bf16) / sizeof(float4)];
        float4 b_buffer_next[BUFFER_SIZE * sizeof(bf16) / sizeof(float4)];

        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, tile + 1}, As);
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

        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, tile + 1}, Bs);
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

        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
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
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

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
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}

// ===================== Combined Dispatch =====================

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_tt_fused
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        // TT: A is KxM, B is NxK → M=A.cols, N=B.rows, K=A.rows
        int Ksz = sa[0].cast<int>();
        int Msz = sa[1].cast<int>();
        int Nsz = sb[0].cast<int>();

        uint64_t a_ptr = A.attr("data_ptr")().cast<uint64_t>();
        uint64_t b_ptr = B.attr("data_ptr")().cast<uint64_t>();
        uint64_t c_ptr = C.attr("data_ptr")().cast<uint64_t>();

        // Static workspace for transposed A
        static bf16* workspace = nullptr;
        static size_t workspace_size = 0;
        size_t needed = (size_t)Msz * Ksz * sizeof(bf16);
        if (needed > workspace_size) {
            if (workspace) hipFree(workspace);
            hipMalloc(&workspace, needed);
            workspace_size = needed;
        }

        // Phase 1: Transpose A from KxM to MxK
        dim3 trans_grid((Msz + TILE_DIM - 1) / TILE_DIM, (Ksz + TILE_DIM - 1) / TILE_DIM);
        dim3 trans_block(TILE_DIM, 8);
        transpose_kernel<<<trans_grid, trans_block>>>(
            workspace, (const bf16*)a_ptr, Ksz, Msz);

        // Phase 2: NT GEMM with A_MxK (transposed) × B_NxK (original)
        auto ga = kittens::make_gl<_gl_A>((uint64_t)workspace, 1, 1, Msz, Ksz);
        auto gb = kittens::make_gl<_gl_B>(b_ptr, 1, 1, Nsz, Ksz);  // B is NxK, same as NT
        auto gc = kittens::make_gl<_gl_C>(c_ptr, 1, 1, Msz, Nsz);

        micro_globals g{ga, gb, gc, Msz, Nsz, Ksz, (hipStream_t)0};
        unsigned long mem = 65536;
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
        micro_tk<<<dim3((Nsz/BLOCK_SIZE)*(Msz/BLOCK_SIZE)), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
    }, "BF16 TT GEMM via transpose+NT: C = A^T @ B^T");
}
