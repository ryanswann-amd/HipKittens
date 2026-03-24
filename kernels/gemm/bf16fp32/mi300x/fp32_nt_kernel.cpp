// FP32 NT GEMM: C = A @ B^T where A(fp32, MxK), B(fp32, NxK) → C(bf16, MxN)
//
// Loads FP32 from global, converts to BF16 during global→shared store,
// then runs the same BF16 MFMA schedule as the NT kernel.
// No separate conversion step, no temporary buffers.

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

// GL types: FP32 for input, BF16 for shared/compute/output
using _gl_A_f32 = gl<float, -1, -1, -1, -1>;
using _gl_B_f32 = gl<float, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;
using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A_f32 a;  // FP32 input
    _gl_B_f32 b;  // FP32 input
    _gl_C c;      // BF16 output
    int M_dim, N_dim, K_dim;
    hipStream_t stream;
    dim3 grid()  { return dim3((N_dim / BLOCK_SIZE) * (M_dim / BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

// Load FP32 from global, convert to BF16, store to shared tile
// Uses vectorized float4 loads (4 fp32 = 16 bytes) and pairwise bf16 conversion
template<int N_THR>
__device__ void load_f32_to_shared(
    st_bf<BLOCK_SIZE, K_STEP>& dst,
    const float* base,
    int stride,       // row stride in fp32 elements
    int rows, int cols)
{
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    constexpr int ELEMS_PER_LOAD = 4;  // float4 = 4 fp32
    constexpr int LOADS_PER_ROW = K_STEP / ELEMS_PER_LOAD;  // 64/4 = 16
    constexpr int TOTAL_LOADS = BLOCK_SIZE * LOADS_PER_ROW;

    #pragma unroll 1
    for (int idx = threadIdx.x; idx < TOTAL_LOADS; idx += N_THR) {
        int r = idx / LOADS_PER_ROW;
        int c_base = (idx % LOADS_PER_ROW) * ELEMS_PER_LOAD;

        // Vectorized FP32 global load
        float4 f4 = *(const float4*)&base[r * stride + c_base];

        // Convert 4 fp32 → 4 bf16, pack as 2 pairs
        bf16 b0 = __float2bfloat16(f4.x);
        bf16 b1 = __float2bfloat16(f4.y);
        bf16 b2 = __float2bfloat16(f4.z);
        bf16 b3 = __float2bfloat16(f4.w);

        // Store as 2 × float (2 packed bf16 each) via ds_write_b32
        bf16 pair0[2] = {b0, b1};
        bf16 pair1[2] = {b2, b3};

        bf16* d0 = dst.idx(&dst.data[0], {r, c_base});
        bf16* d1 = dst.idx(&dst.data[0], {r, c_base + 2});
        *(uint32_t*)d0 = *(uint32_t*)pair0;
        *(uint32_t*)d1 = *(uint32_t*)pair1;
    }
}

// Load FP32 global → register buffer with conversion to BF16 layout
// The buffer stores bf16 data ready for store_register_buffer_to_shared
template<int N_THR, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ void load_f32_to_register_buffer(
    float4* reg_buffer, const int buffer_size,
    const GL& src, const COORD& idx, const ST& dst_template)
{
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(bf16);  // 8 bf16 per float4
    constexpr int bf16_cols = ST::cols;  // tile width in bf16 elements
    constexpr int loads_per_row = bf16_cols / 4;  // 4 fp32 values per float4 load
    constexpr int total_f32_loads = ST::rows * loads_per_row;
    constexpr int total_calls = (total_f32_loads + N_THR - 1) / N_THR;

    const int row_stride = src.template stride<2>();  // stride in fp32 elements
    coord<> unit_coord = idx.template unit_coord<2, 3>();
    const float* base_ptr = (const float*)&src[unit_coord];

    int buf_idx = 0;
    for (int call = 0; call < total_calls && buf_idx < buffer_size; ++call) {
        int load_idx = call * N_THR + (threadIdx.x % N_THR);
        int row = load_idx / loads_per_row;
        int col = (load_idx % loads_per_row) * 4;  // col in fp32 elements = col in bf16

        if (row < ST::rows) {
            // Load 4 fp32 from global
            float4 f4 = *(const float4*)&base_ptr[row * row_stride + col];

            // Convert to 4 bf16, pack into float4 (8 bf16 slots, only first 4 used)
            bf16 b[4] = {__float2bfloat16(f4.x), __float2bfloat16(f4.y),
                         __float2bfloat16(f4.z), __float2bfloat16(f4.w)};
            // Store as half a float4 (4 bf16 = 8 bytes = float2)
            float2 packed;
            *(uint32_t*)&packed.x = *(uint32_t*)&b[0];
            *(uint32_t*)&packed.y = *(uint32_t*)&b[2];

            // We need to fill float4 buffers matching bf16 tile layout
            // Actually, the register buffer needs to match store_register_buffer_to_shared format
            // which expects float4 containing 8 bf16 values. We load 4 fp32 = 4 bf16.
            // So we need 2 fp32 loads to fill one float4 buffer entry.
            // This is complex — let's use the simpler load_f32_to_shared approach instead.
        }
        buf_idx++;
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

    // FP32 base pointers
    const float* A_base = (const float*)&g.a[{0, 0, 0, 0}];
    const float* B_base = (const float*)&g.b[{0, 0, 0, 0}];
    const int A_stride = g.K_dim;  // A is MxK, stride = K
    const int B_stride = g.K_dim;  // B is NxK, stride = K

    // Prologue: load first tiles with FP32→BF16 conversion
    load_f32_to_shared<NUM_THREADS>(As, A_base + row * BLOCK_SIZE * A_stride, A_stride, BLOCK_SIZE, K_STEP);
    load_f32_to_shared<NUM_THREADS>(Bs, B_base + col * BLOCK_SIZE * B_stride, B_stride, BLOCK_SIZE, K_STEP);
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }

    // Main loop — no register buffer pipelining (FP32 loads need conversion)
    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        // Clusters 0-5: load subtiles from shared + MFMA (same as NT)
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

        // Cluster 6: Load NEXT tiles with FP32→BF16 conversion
        asm volatile("s_waitcnt lgkmcnt(0)");
        load_f32_to_shared<NUM_THREADS>(As, A_base + row * BLOCK_SIZE * A_stride + (tile + 1) * K_STEP, A_stride, BLOCK_SIZE, K_STEP);
        load_f32_to_shared<NUM_THREADS>(Bs, B_base + col * BLOCK_SIZE * B_stride + (tile + 1) * K_STEP, B_stride, BLOCK_SIZE, K_STEP);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[7], tiles[6], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[6], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue (same as NT)
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

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp32_nt
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa = A.attr("shape").cast<pybind11::tuple>();
        auto sb = B.attr("shape").cast<pybind11::tuple>();
        int Msz = sa[0].cast<int>(), Ksz = sa[1].cast<int>(), Nsz = sb[0].cast<int>();

        auto ga = kittens::make_gl<_gl_A_f32>(A.attr("data_ptr")().cast<uint64_t>(), 1, 1, Msz, Ksz);
        auto gb = kittens::make_gl<_gl_B_f32>(B.attr("data_ptr")().cast<uint64_t>(), 1, 1, Nsz, Ksz);
        auto gc = kittens::make_gl<_gl_C>(C.attr("data_ptr")().cast<uint64_t>(), 1, 1, Msz, Nsz);

        micro_globals g{ga, gb, gc, Msz, Nsz, Ksz, (hipStream_t)0};
        unsigned long mem = 65536;
        hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem);
        micro_tk<<<dim3((Nsz/BLOCK_SIZE)*(Msz/BLOCK_SIZE)), dim3(NUM_THREADS), mem, (hipStream_t)0>>>(g);
    }, "FP32 NT GEMM: loads FP32, computes in BF16 MFMA, outputs BF16");
}
