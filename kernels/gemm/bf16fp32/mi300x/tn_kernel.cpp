// BF16 TN GEMM: C = A^T @ B where A is KxM (row-major), B is KxN (row-major)
//
// Strategy: Same 8-cluster schedule as NT kernel (mma_ABt, same tile types).
// BOTH A and B need transposing: A_KxM → MxK, B_KxN → NxK in shared.
// Uses the same 4-way K-grouped vectorized write technique for both.
// Neither A nor B can be pipelined via register buffers (no room for two
// transposed stores), so both are loaded directly in cluster 6.

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
    _gl_A a;   // A: KxM
    _gl_B b;   // B: KxN
    _gl_C c;   // C: MxN
    int M_dim, N_dim, K_dim;
    hipStream_t stream;
    dim3 grid()  { return dim3((N_dim / BLOCK_SIZE) * (M_dim / BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

// Transposed load: reads KxBS from global, writes BSxK to shared.
// Used for both A_KxM → MxK and B_KxN → NxK.
template<int N_THR>
__device__ void load_transposed(
    st_bf<BLOCK_SIZE, K_STEP>& dst,
    const bf16* base,
    int stride)
{
    using T = bf16;
    constexpr int N_GROUP = 8;
    constexpr int K_GROUP = 4;
    constexpr int N_BATCHES = BLOCK_SIZE / N_GROUP;
    constexpr int K_BATCHES = K_STEP / K_GROUP;
    constexpr int TOTAL = K_BATCHES * N_BATCHES;

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll 1
    for (int batch = threadIdx.x; batch < TOTAL; batch += N_THR) {
        int kb = batch / N_BATCHES;
        int nb = batch % N_BATCHES;
        int k = kb * K_GROUP;
        int n = nb * N_GROUP;

        float4 r0 = *(const float4*)&base[k * stride + n];
        float4 r1 = *(const float4*)&base[(k + 1) * stride + n];
        float4 r2 = *(const float4*)&base[(k + 2) * stride + n];
        float4 r3 = *(const float4*)&base[(k + 3) * stride + n];
        const T* v0 = (const T*)&r0;
        const T* v1 = (const T*)&r1;
        const T* v2 = (const T*)&r2;
        const T* v3 = (const T*)&r3;

        #pragma unroll
        for (int i = 0; i < N_GROUP; ++i) {
            T col[4] = {v0[i], v1[i], v2[i], v3[i]};
            uint32_t addr = dst.idx(dst_ptr, {n + i, k});
            store_shared_vec(addr, *(float2*)col);
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

    // Base pointers for A_KxM and B_KxN
    const bf16* A_base = (const bf16*)&g.a[{0, 0, 0, 0}];
    const bf16* B_base = (const bf16*)&g.b[{0, 0, 0, 0}];
    const int A_stride = g.M_dim;  // stride between K-rows in A_KxM
    const int B_stride = g.N_dim;  // stride between K-rows in B_KxN

    // Prologue: both A and B transposed
    load_transposed<NUM_THREADS>(As, A_base + row * BLOCK_SIZE, A_stride);
    load_transposed<NUM_THREADS>(Bs, B_base + col * BLOCK_SIZE, B_stride);
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        // No register buffer pipelining — both A and B loaded directly in cluster 6

        // Cluster 0
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(tiles[6], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
        load(tiles[7], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6: Both A and B transposed loads
        asm volatile("s_waitcnt lgkmcnt(0)");
        load_transposed<NUM_THREADS>(As, A_base + (tile + 1) * K_STEP * A_stride + row * BLOCK_SIZE, A_stride);
        load_transposed<NUM_THREADS>(Bs, B_base + (tile + 1) * K_STEP * B_stride + col * BLOCK_SIZE, B_stride);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[7], tiles[6], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[6], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue: IDENTICAL to NT
    __builtin_amdgcn_sched_barrier(0);
    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
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
    auto ga = kittens::make_gl<_gl_A>(a_ptr, 1, 1, Ksz, Msz);  // A: KxM
    auto gb = kittens::make_gl<_gl_B>(b_ptr, 1, 1, Ksz, Nsz);  // B: KxN
    auto gc = kittens::make_gl<_gl_C>(c_ptr, 1, 1, Msz, Nsz);

    micro_globals g{ga, gb, gc, Msz, Nsz, Ksz, (hipStream_t)0};
    unsigned long mem = 65536;
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
    micro_tk<<<dim3((Nsz/BLOCK_SIZE)*(Msz/BLOCK_SIZE)), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
}

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_tn
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        // TN: A is KxM, B is KxN → M=A.cols, N=B.cols, K=A.rows
        dispatch_torch(
            A.attr("data_ptr")().cast<uint64_t>(),
            B.attr("data_ptr")().cast<uint64_t>(),
            C.attr("data_ptr")().cast<uint64_t>(),
            sa[1].cast<int>(), sb[1].cast<int>(), sa[0].cast<int>());
    }, "BF16 TN GEMM: C = A^T @ B");
}
