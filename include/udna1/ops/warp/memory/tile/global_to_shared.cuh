/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "tdm_descriptor.cuh"

namespace kittens {

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */

template<int axis, bool assume_aligned, 
        ducks::st::all ST, ducks::gl::all GL, 
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;

    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = dst.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst[unit_coord]);
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

/**
 * gfx1250 raw-pointer global <-> LDS transfers
 *
 * Three hardware paths move a global tile into LDS, all landing straight in
 * LDS with no VGPR staging:
 *
 *   - `global_load_async_to_lds_*`: each active thread copies B bytes
 *     (B8/B32/B64/B128 = 1/4/8/16 B) from global to LDS, so a b128 load moves
 *     16 B x 32 threads = 512 B per wave per instruction, into this
 *     workgroup's LDS. Drained with `wait_async`.
 *   - `cluster_load_async_to_lds_*`: the same per-wave payload, except the one
 *     L2 return is broadcast into the LDS of several workgroups in a cluster at
 *     once (up to ~5x amplification; bypasses L1) -- for workgroup-cluster
 *     kernels where multiple workgroups want the same tile. Also drained with
 *     `wait_async`.
 *   - `tensor_load_to_lds` (TDM): a dedicated DMA-style engine, 
 *     moves a WHOLE tile per instruction from an SGPR descriptor 
 *     and does its own address generation. Drained with `wait_tdm`.
 *
 * These ops dispatch through the gfx1250 shared tile `st`, which owns its LDS
 * storage and address map, mirroring the canonical `load(tile, gl, idx)`
 * surface -- no separate padding descriptor. Kernels allocate an `st_bf` tile
 * (optionally via `shared_allocator::allocate_in<segment<I>>`) and pass it
 * straight in.
 *
 */

/**
 * @brief Cooperative register-mediated global -> LDS tile copy (gfx1250 baseline).
 *
 * Plain `global_load` -> VGPR -> `ds_store` path. Use this when no async
 * intrinsic is available or for correctness baselines. The destination
 * `st` tile owns the subtile-major + padding LDS address map.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                            const COORD& idx, int row_stride)
{
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    // The COORD is interpreted as tile-index coordinates `{b, d, tile_row, tile_col}`
    // -- convert to element coordinates by multiplying the trailing two by ROWS/COLS.
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        // st maps the logical (row-major) index to its subtile-major,
        // padded LDS slot.
        dst.data[dst.lds_offset(i)] = base[row * row_stride + col];
    }
}

/**
 * @brief Cooperative register-mediated LDS -> global tile copy (gfx1250).
 *
 * Inverse of the register-mediated `load(st, gl, idx, row_stride)`: reads
 * each element from the tile's subtile-major/padded slot `lds_offset(flat)`
 * and scatters it back to global memory. Pairs with `load` / `load_async` /
 * `load_tdm`, which all land data in the same LDS address map.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store(const GL& dst, const st<T, ROWS, COLS, Shape>& src,
                             const COORD& idx, int row_stride)
{
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    T* base = dst.raw_ptr
            + (((int64_t(idx.b) * dst.depth() + idx.d) * dst.rows() + gr_base)
               * dst.cols() + gc_base);

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        base[row * row_stride + col] = src.data[src.lds_offset(i)];
    }
}

/**
 * @brief Cooperative async global -> LDS tile copy on gfx1250.
 *
 * Lowers to `global_load_async_to_lds_b128` (single-WG) when `cluster_mask == 0`,
 * and to `cluster_load_async_to_lds_b128` (multicast) when non-zero. Each lane
 * issues one 16-byte transfer; the warp covers `8 * N_THREADS` elements per
 * iteration. Drain with `kittens::sync::wait_async()` before consuming.
 *
 * @tparam N_THREADS    Number of threads participating in the load.
 * @param  dst          Destination `st` tile (owns the padded LDS map).
 * @param  src          Global tile descriptor.
 * @param  idx          Tile coordinate inside `src`.
 * @param  row_stride   Element stride between rows in `src`.
 * @param  cluster_mask `M0` cluster multicast mask (0 for single-WG, non-zero for a workgroup cluster).
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    static_assert(sizeof(T) * 8 == 16, "load_async issues one b128 (16B) per lane");
    constexpr int elems_per_load = 16 / sizeof(T);
    constexpr int total_elems    = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += N_THREADS * elems_per_load)
    {
        const int row = i / COLS;
        const int col = i % COLS;

        // The gfx1250 async-to-LDS builtins want address-space-qualified
        // pointers (AS(1) global, AS(3) LDS). `reinterpret_cast` cannot add
        // an address space, so route through `uintptr_t` + a C-style cast,
        // matching the pattern used elsewhere in this file for AS(3).
        uintptr_t g_uint = reinterpret_cast<uintptr_t>(base + row * row_stride + col);
        uintptr_t l_uint = reinterpret_cast<uintptr_t>(dst.data + dst.lds_offset(i));
        auto* g_ptr = (detail::i32x4_gvec*)(g_uint);
        auto* l_ptr = (detail::i32x4_lvec*)(l_uint);

        if (cluster_mask) {
            __builtin_amdgcn_cluster_load_async_to_lds_b128(
                g_ptr, l_ptr, 0, 0, static_cast<int>(cluster_mask));
        } else {
            __builtin_amdgcn_global_load_async_to_lds_b128(g_ptr, l_ptr, 0, 0);
        }
    }
}

/**
 * @brief Hardware tile DMA (TDM) global -> LDS load on gfx1250.
 *
 * Issues a single `tensor_load_to_lds` instruction whose D# descriptor
 * encodes the 2D tile shape, source tensor extents, row stride, and optional
 * LDS padding.
 *
 * The transfer is issued once by the whole wave, not per thread: it uses no
 * vector registers (VGPRs) and ignores the active-thread mask, so
 * which threads are active makes no difference. The entire tile is described
 * by a small block of scalar registers.
 *
 * A CU has one TDM per SIMD-pair (a gfx1250 CU is four SIMDx32s grouped into two pairs). 
 * That single engine handles one request stream and is shared by the waves on its pair, so
 * extra issuers don't make the copy faster, they just contend for it and use
 * up its in-flight slots (at most 3 transfers per wave, 6 per SIMD).
 *
 * Drain with `kittens::sync::wait_tdm()`.
 *
 * @param  dst         Destination `st` tile (its shape's pad fields drive the D#).
 * @param  src         Global tile descriptor.
 * @param  idx         Tile coordinate.
 * @param  tensor_rows,tensor_cols  Source tensor extents (elements).
 * @param  row_stride  Source row stride (elements).
 * @param  cluster_mask Optional `workgroup_mask` (0 for single-WG, non-zero
 *                     to switch the load into `CLUSTER_LOAD_ASYNC` micro-ops).
 */
/**
 * @brief Optional knobs for the derived-by-default `load_tdm` / `store_tdm`.
 *
 * Everything the `st` + `gl` already encode (dtype, tile dims, padding, LDS
 * address, tensor extents, dense stride, global base) is derived, never passed.
 * Only the genuinely optional behavior lives here. Designated initializers keep
 * the common call short: `load_tdm(dst, src, idx, {.arrive = &bar})`.
 */
struct tdm_opts {
    uint64_t*         arrive  = nullptr;                  // !=0 => atomic_barrier_enable + LDS bar addr; else poll via wait_tdm
    uint32_t          cluster = 0;                        // multicast mask (load only)
    detail::idx_width idx_w   = detail::idx_width::b16;   // gather index width (reserved; gather is not yet implemented)
};

namespace detail {

/**
 * @brief Build a dense affine load descriptor and issue the TDM load builtin.
 *
 * Shared lowering used by both the derived `load_tdm` verb and the deprecated
 * explicit-extent shims. Reuses the proven groups-0/1 packing; groups 2/3 are
 * zero (dense affine). `bar_lds_addr == 0` means ordering-only (poll with
 * `wait_tdm`); non-zero sets the D# auto-arrive.
 */
template<typename Shape, int ROWS, int COLS, typename T>
__device__ __forceinline__ void issue_tdm_load(
    st<T, ROWS, COLS, Shape>& dst, const T* base,
    int tensor_rows, int tensor_cols, int row_stride,
    uint32_t cluster_mask, uint32_t bar_lds_addr)
{
    tdm_desc d;
    d.base         = reinterpret_cast<uint64_t>(base);
    d.lds_addr     = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst.data));
    d.tensor_rows  = static_cast<uint32_t>(tensor_rows);
    d.tensor_cols  = static_cast<uint32_t>(tensor_cols);
    d.row_stride_e = static_cast<uint32_t>(row_stride);
    d.cluster_mask = static_cast<uint16_t>(cluster_mask);
    d.bar_lds_addr = static_cast<uint16_t>(bar_lds_addr);
    d.mode         = tdm_mode::affine;
    d.dir          = tdm_dir::load;

    v4u32 g0, g2, g3;
    v8u32 g1;
    encode_tdm<Shape, ROWS, COLS, T>(d, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Build a dense affine store descriptor and issue the TDM store builtin.
 *
 * The store-direction mirror of `issue_tdm_load`: the descriptor is built by the
 * SAME `encode_tdm` lowering (groups 0/1 + dense affine groups 2/3 = 0); only
 * `dir` and the dispatched builtin differ (`tensor_store_from_lds`). The `st`
 * tile is the LDS SOURCE here (`d.lds_addr = src.data`); `base` is the global
 * destination top-left.
 *
 * Per SP3 the `workgroup_mask` (multicast) is load-only, so stores hard-zero the
 * cluster mask. `bar_lds_addr == 0` means ordering-only (drain via `wait_tdm`);
 * non-zero sets the D# auto-arrive on store retirement (XACK).
 */
template<typename Shape, int ROWS, int COLS, typename T>
__device__ __forceinline__ void issue_tdm_store(
    const st<T, ROWS, COLS, Shape>& src, T* base,
    int tensor_rows, int tensor_cols, int row_stride,
    uint32_t bar_lds_addr)
{
    tdm_desc d;
    d.base         = reinterpret_cast<uint64_t>(base);
    d.lds_addr     = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data));
    d.tensor_rows  = static_cast<uint32_t>(tensor_rows);
    d.tensor_cols  = static_cast<uint32_t>(tensor_cols);
    d.row_stride_e = static_cast<uint32_t>(row_stride);
    d.cluster_mask = 0;            // multicast is load-only (SP3); stores ignore it.
    d.bar_lds_addr = static_cast<uint16_t>(bar_lds_addr);
    d.mode         = tdm_mode::affine;
    d.dir          = tdm_dir::store;

    v4u32 g0, g2, g3;
    v8u32 g1;
    encode_tdm<Shape, ROWS, COLS, T>(d, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

template<typename T, int ROWS, int COLS, typename GL, typename COORD>
__device__ __forceinline__ const T* tdm_tile_base(const GL& src, const COORD& idx)
{
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    return src.raw_ptr
         + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
            * src.cols() + gc_base);
}

} // namespace detail

/**
 * @brief Hardware tile DMA (TDM) global -> LDS load (udna1, derived form).
 *
 * Mirrors the canonical `load(tile, gl, idx)`: the source tensor extents, dense
 * row stride, dtype, tile dims, LDS padding, LDS address and global base are ALL
 * derived from `dst` + `src` + `idx`. The only optional behavior (auto-arrive,
 * multicast) lives in `tdm_opts`.
 *
 * Issues a single `tensor_load_to_lds` for the whole wave (no VGPR operands,
 * ignores the active-thread mask). Drain with `kittens::sync::wait_tdm()`, or,
 * when `opts.arrive` is set, wait on the barrier's phase flip via
 * `kittens::sync::wait_barrier(opts.arrive, phase)`.
 *
 * @param dst   Destination `st` tile (its shape's pad fields drive the D#).
 * @param src   Global tile descriptor (supplies extents + dense stride).
 * @param idx   Tile coordinate.
 * @param opts  Optional: `.arrive` LDS barrier cell, `.cluster` multicast mask.
 */
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                const COORD& idx, tdm_opts opts = {})
{
    const T* base = detail::tdm_tile_base<T, ROWS, COLS>(src, idx);
    const uint32_t bar = opts.arrive
        ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(opts.arrive)) : 0u;
    detail::issue_tdm_load<Shape, ROWS, COLS, T>(
        dst, base,
        static_cast<int>(src.rows()), static_cast<int>(src.cols()),
        static_cast<int>(src.template stride<2>()),
        opts.cluster, bar);
}

/**
 * @brief Hardware tile DMA (TDM) LDS -> global store (udna1, derived form).
 *
 * The store-direction mirror of `load_tdm`: issues a single
 * `tensor_store_from_lds` for the whole wave, writing the `src` `st` tile back
 * to the `dst` global tile at `idx`. The destination tensor extents, dense row
 * stride, dtype, tile dims, LDS address and global base are ALL derived from
 * `dst` + `src` + `idx` -- nothing derivable is repeated at the call site.
 *
 * Drain with `kittens::sync::wait_tdm()` (loads and stores share `tensorcnt`),
 * or, when `opts.arrive` is set, wait on the barrier's phase flip via
 * `kittens::sync::wait_barrier(opts.arrive, phase)`.
 *
 * @note Per SP3 the multicast/`workgroup_mask` and LDS-pad fields are load-only;
 *       `opts.cluster` is ignored on the store path. Use only non-padded shapes
 *       on the store path until the LDS pad encoding is reconciled (see the
 *       [UNVERIFIED] note in `tdm_descriptor.cuh`).
 *
 * @param dst   Destination global tile descriptor (supplies extents + stride).
 * @param src   Source `st` tile (LDS source; its dims/dtype drive the D#).
 * @param idx   Tile coordinate inside `dst`.
 * @param opts  Optional: `.arrive` LDS barrier cell to auto-arrive on completion.
 */
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store_tdm(const GL& dst, const st<T, ROWS, COLS, Shape>& src,
                                 const COORD& idx, tdm_opts opts = {})
{
    T* base = const_cast<T*>(detail::tdm_tile_base<T, ROWS, COLS>(dst, idx));
    const uint32_t bar = opts.arrive
        ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(opts.arrive)) : 0u;
    detail::issue_tdm_store<Shape, ROWS, COLS, T>(
        src, base,
        static_cast<int>(dst.rows()), static_cast<int>(dst.cols()),
        static_cast<int>(dst.template stride<2>()),
        bar);
}

/**
 * @brief Deprecated: explicit-extent TDM load (pre-derived surface).
 *
 * Back-compat shim for the original 6-arg form. New code should let the extents
 * and dense stride be derived from `src`: `load_tdm(dst, src, idx)` (optionally
 * `{.arrive = &bar}` / `{.cluster = mask}`). Forwarding shim retained for one
 * release.
 */
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
[[deprecated("extents are derived from src; use load_tdm(dst, src, idx, opts)")]]
__device__ inline void load_tdm(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                const COORD& idx,
                                int tensor_rows, int tensor_cols, int row_stride,
                                uint32_t cluster_mask = 0)
{
    const T* base = detail::tdm_tile_base<T, ROWS, COLS>(src, idx);
    detail::issue_tdm_load<Shape, ROWS, COLS, T>(
        dst, base, tensor_rows, tensor_cols, row_stride, cluster_mask, /*bar=*/0);
}

/**
 * @brief Deprecated: TDM load that auto-arrives at an LDS barrier on completion.
 *
 * Folded into `tdm_opts.arrive`: prefer
 * `load_tdm(dst, src, idx, {.arrive = bar})`. The barrier must be primed via
 * `kittens::sync::init_barrier(bar, count)` before the first referencing call.
 * Forwarding shim retained for one release.
 *
 * @note See the original design note: the D# auto-arrive path
 * (`atomic_barrier_enable` w0 bit 18, `atomic_barrier_address` w1[15:0]) is for
 * runtimes that model it natively; simulators that don't should use the explicit
 * `wait_tdm()` + `async_barrier_arrive()` pattern (see `gemm_tdm_arrive.cpp`).
 */
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
[[deprecated("use load_tdm(dst, src, idx, {.arrive = bar})")]]
__device__ inline void load_tdm_arrive(
    st<T, ROWS, COLS, Shape>& dst, const GL& src, const COORD& idx,
    int tensor_rows, int tensor_cols, int row_stride,
    uint64_t* bar, uint32_t cluster_mask = 0)
{
    const T* base = detail::tdm_tile_base<T, ROWS, COLS>(src, idx);
    const uint32_t bar_lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(bar));
    detail::issue_tdm_load<Shape, ROWS, COLS, T>(
        dst, base, tensor_rows, tensor_cols, row_stride, cluster_mask, bar_lds_addr);
}

/**
 * @brief Cooperative L2 prefetch for an upcoming tile.
 *
 * Lowers to `__builtin_amdgcn_global_prefetch` issued from every participating
 * lane. The hint = 0 selects the default cache policy.
 */
template<int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void prefetch_l2(const GL& src, const COORD& idx, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    constexpr int elems_per_pf = 16 / sizeof(T);
    constexpr int total_elems  = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_pf; i < total_elems;
         i += N_THREADS * elems_per_pf)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const T* addr = base + row * row_stride + col;
        __builtin_amdgcn_global_prefetch(
            (const void __attribute__((address_space(1)))*)addr, 0);
    }
}

}
