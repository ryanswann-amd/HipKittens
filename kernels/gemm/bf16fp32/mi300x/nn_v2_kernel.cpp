// BF16 NN GEMM v2: C = A @ B where B is KxN
//
// Key insight: load B_KxN NATURALLY into st_bf<K_STEP, BLOCK_SIZE> (vectorized),
// read B subtiles as row_layout from shared (vectorized ds_read_b64),
// swap_layout_inplace to col_layout (lane shuffles, ~16 cycles),
// then mma_AB(C, A_row, B_col, C).
//
// This keeps the FULL register buffer pipeline for both A and B — same as NT!
// The only overhead vs NT is the swap_layout (~16 lane shuffles per B subtile).
// NO scalar shared writes. NO separate transpose kernel.

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
    _gl_A a;   // A: MxK
    _gl_B b;   // B: KxN (stored naturally, NOT transposed)
    _gl_C c;   // C: MxN
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
    // A: BLOCK_SIZE × K_STEP (MxK layout, same as NT)
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    // B: K_STEP × BLOCK_SIZE (KxN layout — NATURAL for B_KxN, vectorized loads!)
    st_bf<K_STEP, BLOCK_SIZE> (&Bs) = al.allocate<st_bf<K_STEP, BLOCK_SIZE>>();

    // A tiles: same as NT (REG_BLOCK × DOT_SLICE, row_layout)
    rt_bf<REG_BLOCK, DOT_SLICE> a_tiles[5];
    // B tiles: DOT_SLICE × REG_BLOCK, loaded as row then swapped to col for mma_AB
    rt_bf<DOT_SLICE, REG_BLOCK> b_tiles[3];

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

    // Prologue: Load first tiles
    G::load(As, g.a, {0, 0, row, 0});         // A: MxK, same as NT
    G::load(Bs, g.b, {0, 0, 0, col});          // B: KxN → st_bf<K_STEP, BLOCK_SIZE>
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    // Dummy template for B's register buffer load (matches KxN gl dimensions)
    // load_global_to_register_buffer needs ST template matching the source layout
    st_bf<K_STEP, BLOCK_SIZE>& Bs_tmpl = Bs;

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        constexpr int A_BUF_SIZE = (BLOCK_SIZE * K_STEP) / NUM_THREADS;
        constexpr int B_BUF_SIZE = (K_STEP * BLOCK_SIZE) / NUM_THREADS;  // same total elements
        float4 a_buffer_next[A_BUF_SIZE * sizeof(bf16) / sizeof(float4)];
        float4 b_buffer_next[B_BUF_SIZE * sizeof(bf16) / sizeof(float4)];

        // Cluster 0: Prefetch next A + load current subtiles
        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_BUF_SIZE, g.a, {0, 0, row, tile + 1}, As);
        load(a_tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        load(a_tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        // B subtile: DOT_SLICE rows (K) × REG_BLOCK cols (N) from KxN shared
        load(b_tiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {0, warp_col}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: swap B layout + mma_AB for K=0
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        { auto& bc = swap_layout_inplace(b_tiles[0]); mma_AB(C_accum[0], a_tiles[0], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[1], bc, C_accum[1]); }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: Load K=1, K=2 subtiles
        load(b_tiles[1], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {1, warp_col}));
        load(a_tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(a_tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(b_tiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {2, warp_col}));
        load(a_tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3: K=1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        { auto& bc = swap_layout_inplace(b_tiles[1]); mma_AB(C_accum[0], a_tiles[2], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[3], bc, C_accum[1]); }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: Prefetch next B + load K=2,3 subtiles
        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_BUF_SIZE, g.b, {0, 0, tile + 1, col}, Bs_tmpl);
        load(a_tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(b_tiles[2], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {3, warp_col}));
        load(a_tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(a_tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5: K=2
        __builtin_amdgcn_s_setprio(1);
        { auto& bc = swap_layout_inplace(b_tiles[0]); mma_AB(C_accum[0], a_tiles[0], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[1], bc, C_accum[1]); }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: Store both buffers to shared (VECTORIZED — no transpose!)
        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7: K=3
        __builtin_amdgcn_s_setprio(1);
        { auto& bc = swap_layout_inplace(b_tiles[2]); mma_AB(C_accum[0], a_tiles[4], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[3], bc, C_accum[1]); }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    __builtin_amdgcn_sched_barrier(0);
    load(b_tiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {0, warp_col}));
    load(a_tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load(a_tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    { auto& bc = swap_layout_inplace(b_tiles[0]); mma_AB(C_accum[0], a_tiles[0], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[1], bc, C_accum[1]); }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(b_tiles[1], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {1, warp_col}));
    load(a_tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(a_tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    { auto& bc = swap_layout_inplace(b_tiles[1]); mma_AB(C_accum[0], a_tiles[2], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[3], bc, C_accum[1]); }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(b_tiles[0], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {2, warp_col}));
    load(a_tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load(a_tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(b_tiles[1], subtile_inplace<DOT_SLICE, REG_BLOCK>(Bs, {3, warp_col}));
    load(a_tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load(a_tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    { auto& bc = swap_layout_inplace(b_tiles[0]); mma_AB(C_accum[0], a_tiles[0], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[1], bc, C_accum[1]); }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    { auto& bc = swap_layout_inplace(b_tiles[1]); mma_AB(C_accum[0], a_tiles[2], bc, C_accum[0]); mma_AB(C_accum[1], a_tiles[3], bc, C_accum[1]); }
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

void dispatch_torch(uint64_t a_ptr, uint64_t b_ptr, uint64_t c_ptr, int Msz, int Nsz, int Ksz) {
    auto ga = kittens::make_gl<_gl_A>(a_ptr, 1, 1, Msz, Ksz);
    auto gb = kittens::make_gl<_gl_B>(b_ptr, 1, 1, Ksz, Nsz);  // B: KxN
    auto gc = kittens::make_gl<_gl_C>(c_ptr, 1, 1, Msz, Nsz);

    micro_globals g{ga, gb, gc, Msz, Nsz, Ksz, (hipStream_t)0};
    unsigned long mem = 65536;
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<<<dim3((Nsz/BLOCK_SIZE)*(Msz/BLOCK_SIZE)), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_nn_v2
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
    }, "BF16 NN GEMM v2: vectorized load + swap_layout + mma_AB");
}
