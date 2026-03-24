// BF16 NN GEMM: C = A @ B (A is MxK row-major, B is KxN row-major)
// Based on the CDNA3 branch NT kernel with modifications for NN transpose.
// Changes from NT:
//   1. Bs shared tile is st_bf<K_STEP, BLOCK_SIZE> (transposed: KS rows × BS cols)
//   2. B gl is (rows=K, cols=N), load coords: {0, 0, tile, col}
//   3. B fragments use col_l layout (separate btiles array)
//   4. mma_AB instead of mma_ABt
//   5. Dispatch: N = B.shape[1] (cols)

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
    // NN: B shared tile is KS×BS (transposed from NT's BS×KS)
    st_bf<K_STEP, BLOCK_SIZE> (&Bs) = al.allocate<st_bf<K_STEP, BLOCK_SIZE>>();

    // A fragments (row_l) — same as NT
    rt_bf<REG_BLOCK, DOT_SLICE> atiles[8];
    // B fragments (col_l) — different from NT's row_l
    rt_bf<DOT_SLICE, REG_BLOCK, ducks::rt_layout::col> btiles[4];
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

    // Load first tile
    G::load(As, g.a, {0, 0, row, 0});
    // NN: load B[0:KS, col*BS:(col+1)*BS] from B_KxN
    G::load(Bs, g.b, {0, 0, 0, col});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        constexpr int BUFFER_SIZE = (BLOCK_SIZE * K_STEP) / NUM_THREADS;
        float4 a_buffer_next[BUFFER_SIZE * sizeof(bf16) / sizeof(float4)];
        float4 b_buffer_next[BUFFER_SIZE * sizeof(bf16) / sizeof(float4)];

        // Cluster 0: Load A next + fragments for K_SLICE 0
        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, tile + 1}, As);
        load(atiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        load(atiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        // NN: B subtile from st_bf<KS, BS>: subtile<DOT_SLICE, REG_BLOCK> at {ks, warp_col}
        load(btiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {0, warp_col}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: MMA K_SLICE 0 — mma_AB(D, A_row, B_col, C)
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0], atiles[1], btiles[0], C_accum[0]);
        mma_AB(C_accum[1], atiles[2], btiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: Fragments for K_SLICE 1
        load(btiles[1], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {1, warp_col}));
        load(atiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(atiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(btiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {2, warp_col}));
        load(atiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3: MMA K_SLICE 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0], atiles[4], btiles[1], C_accum[0]);
        mma_AB(C_accum[1], atiles[5], btiles[1], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: Load B next + fragments for K_SLICE 2-3
        // NN: B load from {0, 0, tile+1, col} (tile selects K block, col selects N block)
        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE, g.b, {0, 0, tile + 1, col}, Bs);
        load(atiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(btiles[2], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {3, warp_col}));
        load(atiles[7], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(atiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5: MMA K_SLICE 2
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0], atiles[1], btiles[0], C_accum[0]);
        mma_AB(C_accum[1], atiles[2], btiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: Store next tile to shared
        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7: MMA K_SLICE 3
        __builtin_amdgcn_s_setprio(1);
        mma_AB(C_accum[0], atiles[7], btiles[2], C_accum[0]);
        mma_AB(C_accum[1], atiles[5], btiles[2], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue (last tile — same structure without prefetch)
    load(btiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {0, warp_col}));
    load(atiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load(atiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_AB(C_accum[0], atiles[1], btiles[0], C_accum[0]);
    mma_AB(C_accum[1], atiles[2], btiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(btiles[1], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {1, warp_col}));
    load(atiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(atiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_AB(C_accum[0], atiles[4], btiles[1], C_accum[0]);
    mma_AB(C_accum[1], atiles[5], btiles[1], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(btiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {2, warp_col}));
    load(atiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load(atiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(btiles[1], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {3, warp_col}));
    load(atiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load(atiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_AB(C_accum[0], atiles[1], btiles[0], C_accum[0]);
    mma_AB(C_accum[1], atiles[2], btiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_AB(C_accum[0], atiles[4], btiles[1], C_accum[0]);
    mma_AB(C_accum[1], atiles[5], btiles[1], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Torch dispatch for NN: C = A @ B (A: MxK, B: KxN)
void dispatch_torch(uint64_t a_ptr, uint64_t b_ptr, uint64_t c_ptr,
                    int Msz, int Nsz, int Ksz) {
    // NN: A is MxK, B is KxN
    auto ga = kittens::make_gl<_gl_A>(a_ptr, 1, 1, Msz, Ksz);
    // NN: B gl has rows=K, cols=N
    auto gb = kittens::make_gl<_gl_B>(b_ptr, 1, 1, Ksz, Nsz);
    auto gc = kittens::make_gl<_gl_C>(c_ptr, 1, 1, Msz, Nsz);

    micro_globals g{ga, gb, gc, Msz, Nsz, Ksz, (hipStream_t)0};
    int grid = (Nsz / BLOCK_SIZE) * (Msz / BLOCK_SIZE);
    unsigned long mem = 65536;
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<<<dim3(grid), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_nn
#endif

PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        dispatch_torch(
            A.attr("data_ptr")().cast<uint64_t>(),
            B.attr("data_ptr")().cast<uint64_t>(),
            C.attr("data_ptr")().cast<uint64_t>(),
            sa[0].cast<int>(), sb[1].cast<int>(), sa[1].cast<int>());
    }, "BF16 NN GEMM: C = A @ B");
}
