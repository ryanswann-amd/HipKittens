#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// Rectangular GEMM tile: separate BLOCK_M and BLOCK_N
#ifndef BLOCK_M_VAL
#define BLOCK_M_VAL 128
#endif
#ifndef BLOCK_N_VAL
#define BLOCK_N_VAL 256
#endif
constexpr int BLOCK_M = BLOCK_M_VAL;
constexpr int BLOCK_N = BLOCK_N_VAL;
#ifndef K_STEP_VAL
#define K_STEP_VAL 64
#endif
constexpr int K_STEP  = K_STEP_VAL;

// Each dimension divided into 4 register sub-tiles (2 warp_rows * 2 accumulators for M, 4 warp_cols for N)
constexpr int REG_M     = BLOCK_M / 4;   // 32 for BLOCK_M=128, 64 for BLOCK_M=256
constexpr int REG_N     = BLOCK_N / 4;   // 64 for BLOCK_N=256, 32 for BLOCK_N=128
constexpr int DOT_SLICE = 16;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

// Buffer sizes differ for A and B when tile is rectangular
constexpr int A_TOTAL_ELEMS  = BLOCK_M * K_STEP;
constexpr int B_TOTAL_ELEMS  = BLOCK_N * K_STEP;
constexpr int A_BUFFER_SIZE  = A_TOTAL_ELEMS / NUM_THREADS;  // bf16 elements per thread for A
constexpr int B_BUFFER_SIZE  = B_TOTAL_ELEMS / NUM_THREADS;  // bf16 elements per thread for B
constexpr int A_BUF_F4       = A_BUFFER_SIZE * sizeof(bf16) / sizeof(float4);  // float4 words for A
constexpr int B_BUF_F4       = B_BUFFER_SIZE * sizeof(bf16) / sizeof(float4);  // float4 words for B

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    int M_dim, N_dim, K_dim;
    hipStream_t stream;
    dim3 grid()  { return dim3((N_dim / BLOCK_N) * (M_dim / BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() {
        // A tile: BLOCK_M * K_STEP * 2 bytes, B tile: BLOCK_N * K_STEP * 2 bytes
        size_t needed = (BLOCK_M + BLOCK_N) * K_STEP * sizeof(bf16);
        return (needed > 65536) ? needed : 65536;
    }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_M, K_STEP> (&As) = al.allocate<st_bf<BLOCK_M, K_STEP>>();
    st_bf<BLOCK_N, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_N, K_STEP>>();

    // A fragments use REG_M, B fragments use REG_N
    rt_bf<REG_M, DOT_SLICE> a_frag[4];  // 4 A fragments for scheduling
    rt_bf<REG_N, DOT_SLICE> b_frag[4];  // 4 B fragments for scheduling
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    // Get original WGID.
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    // Swizzle for better L2 within the same XCD. Use separate M/N tiling sizes.
    const int num_pid_m = ceil_div(g.M_dim, BLOCK_M);
    const int num_pid_n = ceil_div(g.N_dim, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m;
    const int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;  // 0 or 1
    const int warp_col = warp_id % 4;  // 0..3

    const int num_tiles = g.K_dim / K_STEP;

    // Load first tile into shared memory
    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {

        // Register buffers for pipelining - different sizes for A and B
        float4 a_buffer_next[A_BUF_F4];
        float4 b_buffer_next[B_BUF_F4];

        // Cluster 0: Load A from global + load A,B subtiles from shared
        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_BUFFER_SIZE, g.a, {0, 0, row, tile + 1}, As);
        load(a_frag[0], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 0}));
        load(a_frag[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 0}));
        load(b_frag[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: MMA k_idx=0
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], a_frag[0], b_frag[0], C_accum[0]);
        mma_ABt(C_accum[1], a_frag[1], b_frag[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2: Load subtiles for k_idx=1,2
        load(b_frag[1], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 1}));
        load(a_frag[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 1}));
        load(a_frag[3], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(b_frag[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 2}));
        load(a_frag[0], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3: MMA k_idx=1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], a_frag[2], b_frag[1], C_accum[0]);
        mma_ABt(C_accum[1], a_frag[3], b_frag[1], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4: Load B from global + load subtiles for k_idx=3
        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_BUFFER_SIZE, g.b, {0, 0, col, tile + 1}, Bs);
        load(a_frag[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(b_frag[2], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 3}));
        load(a_frag[3], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 3}));
        load(a_frag[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5: MMA k_idx=2
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], a_frag[0], b_frag[0], C_accum[0]);
        mma_ABt(C_accum[1], a_frag[1], b_frag[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: Store register buffers to shared
        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7: MMA k_idx=3
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], a_frag[3], b_frag[2], C_accum[0]);
        mma_ABt(C_accum[1], a_frag[2], b_frag[2], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // Epilogue — last K tile
    // Cluster 0
    __builtin_amdgcn_sched_barrier(0);
    load(b_frag[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 0}));
    load(a_frag[0], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 0}));
    load(a_frag[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], a_frag[0], b_frag[0], C_accum[0]);
    mma_ABt(C_accum[1], a_frag[1], b_frag[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2
    load(b_frag[1], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 1}));
    load(a_frag[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 1}));
    load(a_frag[3], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], a_frag[2], b_frag[1], C_accum[0]);
    mma_ABt(C_accum[1], a_frag[3], b_frag[1], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4
    load(b_frag[0], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 2}));
    load(a_frag[0], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 2}));
    load(a_frag[1], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(b_frag[2], subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col, 3}));
    load(a_frag[3], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row, 3}));
    load(a_frag[2], subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], a_frag[0], b_frag[0], C_accum[0]);
    mma_ABt(C_accum[1], a_frag[1], b_frag[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], a_frag[3], b_frag[2], C_accum[0]);
    mma_ABt(C_accum[1], a_frag[2], b_frag[2], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    // Store output: coord is in units of register tile (REG_M rows x REG_N cols)
    // BLOCK_M/REG_M = 4 sub-rows per block, BLOCK_N/REG_N = 4 sub-cols per block
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
    auto gb = kittens::make_gl<_gl_B>(b_ptr, 1, 1, Nsz, Ksz);
    auto gc = kittens::make_gl<_gl_C>(c_ptr, 1, 1, Msz, Nsz);

    micro_globals g{ga, gb, gc, Msz, Nsz, Ksz, (hipStream_t)0};
    unsigned long mem = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<<<dim3((Nsz/BLOCK_N)*(Msz/BLOCK_M)), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_rect_gemm
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        dispatch_torch(
            A.attr("data_ptr")().cast<uint64_t>(),
            B.attr("data_ptr")().cast<uint64_t>(),
            C.attr("data_ptr")().cast<uint64_t>(),
            sa[0].cast<int>(), sb[0].cast<int>(), sa[1].cast<int>());
    });
}
