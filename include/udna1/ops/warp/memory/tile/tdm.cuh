/**
 * @file
 * @brief Tensor Data Mover (TDM) -- derive-by-default tile transfers.
 *
 * This header implements the API described in the TDM design doc. The shape
 * of the surface:
 *
 *   - Two verbs, one per direction: `kittens::load_tdm` / `kittens::store_tdm`.
 *   - The addressing mode is selected by the *type* of the third/fourth
 *     argument:
 *       coord                       => dense N-D
 *       coord + tdm::affine         => 5D affine (extra axes 2-4)
 *       (const uint32_t* rows, n)   => gather (load) / scatter (store)
 *       coord + tdm::iterate        => hard-stopped (does not emit)
 *   - Extents, strides, dtype, tile dims, LDS address, and global base are
 *     all *derived* from `dst`, `src`, and `idx`. The only non-derivable
 *     higher-dimensional data rides in the `tdm::affine` value.
 *   - One options struct, `tdm::tdm_opts`, carries the optional knobs:
 *     `.arrive` (auto-arrive at an LDS barrier cell), `.cluster` (multicast
 *     mask, load only), `.idx_w` (gather index width).
 *   - All verbs lower into a single device-constructible POD,
 *     `tdm::detail::tdm_desc`, through one `encode(...)` point.
 *
 * The descriptor bit layout reproduced in `encode` is the one validated
 * bit-exact against the project's correctness oracle. Strides and tensor
 * extents are in **element** units; padding interval/amount are in element
 * units and encoded per SP3. Completion is a single per-descriptor
 * `TENSORcnt` tick, drained via `tdm::load_async_wait<N>()` (lowers to
 * `s_wait_tensorcnt N`).
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../../sync/barrier.cuh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kittens {
namespace tdm {

/* ============================================================ *
 *  Public addressing types and options
 * ============================================================ */

/// @brief Gather/scatter row-index width. b16 => 16 rows/op, b32 => 8 rows/op.
enum class idx_width : uint8_t { b16, b32 };

/**
 * @brief Optional knobs for a TDM transfer.
 *
 * @var arrive   When non-null, the LDS byte address of a `sync::barrier_lds`
 *               cell (`&bar.state`); sets `atomic_barrier_enable` so the TDM
 *               unit auto-arrives on completion. When null, drain via
 *               `load_async_wait<N>()`.
 * @var cluster  16-bit multicast workgroup mask (load only; ignored on store).
 * @var idx_w    Gather index width (16- or 32-bit).
 */
struct tdm_opts {
    uint64_t* arrive  = nullptr;
    uint32_t  cluster = 0;
    idx_width idx_w   = idx_width::b16;
};

/**
 * @brief Outer addressing axes (dims 2-4) for an N-D affine transfer.
 *
 * The innermost two dims (the tile's rows/cols and the tensor's row stride)
 * are always derived from `dst`/`src`. Anything beyond that -- up to three
 * extra axes -- is non-derivable and rides here. Construct with `make`.
 *
 * `ndim` is the number of extra axes (1..3), giving a total rank of 2+ndim.
 * Each axis is `(extent, stride, tile)` with stride in element units.
 */
struct affine {
    int      ndim = 0;
    uint32_t dim[3]   = {0, 0, 0};   // extents of axes 2,3,4 (outer)
    uint64_t stride[3] = {0, 0, 0};  // element strides of axes 2,3,4
    uint16_t tile[3]  = {0, 0, 0};   // tile counts along axes 2,3,4

    __host__ __device__ static affine make(int dim2, int stride2, int tile2) {
        affine a; a.ndim = 1;
        a.dim[0] = uint32_t(dim2); a.stride[0] = uint64_t(uint32_t(stride2));
        a.tile[0] = uint16_t(tile2);
        return a;
    }
    __host__ __device__ static affine make(int dim2, int stride2, int tile2,
                                           int dim3, int stride3, int tile3) {
        affine a = make(dim2, stride2, tile2); a.ndim = 2;
        a.dim[1] = uint32_t(dim3); a.stride[1] = uint64_t(uint32_t(stride3));
        a.tile[1] = uint16_t(tile3);
        return a;
    }
    __host__ __device__ static affine make(int dim2, int stride2, int tile2,
                                           int dim3, int stride3, int tile3,
                                           int dim4, int stride4, int tile4) {
        affine a = make(dim2, stride2, tile2, dim3, stride3, tile3); a.ndim = 3;
        a.dim[2] = uint32_t(dim4); a.stride[2] = uint64_t(uint32_t(stride4));
        a.tile[2] = uint16_t(tile4);
        return a;
    }
};

/**
 * @brief Iterate-mode descriptor (hard-stopped).
 *
 * Constructible so call sites compile, but passing it to a verb is a
 * compile-time hard-stop: the iterate sub-field offsets are unverified and
 * its behavior is untested by the correctness oracle. See the design doc.
 */
struct iterate {
    uint32_t lds_inc = 0, gbl_inc = 0, count = 0;
    __host__ __device__ static iterate make(int lds_inc, int gbl_inc, int count) {
        iterate it; it.lds_inc = uint32_t(lds_inc); it.gbl_inc = uint32_t(gbl_inc);
        it.count = uint32_t(count); return it;
    }
};

/* ============================================================ *
 *  Internal descriptor POD + single encode point
 * ============================================================ */

namespace detail {

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));
using v8u32 = unsigned int __attribute__((ext_vector_type(8)));

enum class tdm_mode : uint8_t { affine, gather };
enum class tdm_dir  : uint8_t { load, store };

/// @brief Detect an optional `pad_interval`/`pad_amount` on a tile shape.
template<typename S, typename = void>
struct pad_of { static constexpr int interval = 0; static constexpr int amount = 0; };
template<typename S>
struct pad_of<S, std::void_t<decltype(S::pad_interval)>> {
    static constexpr int interval = S::pad_interval;
    static constexpr int amount   = S::pad_amount;
};

/**
 * @brief The one internal descriptor. All arrays are innermost-first.
 *
 * Groups 0-1 carry the common control word, the two innermost tensor dims,
 * the two innermost tile dims and innermost stride; the tagged union carries
 * exactly one of the mutually exclusive group 2-3 layouts.
 */
struct tdm_desc {
    uint64_t base         = 0;   // global byte address (raw pointer)
    uint32_t lds_addr     = 0;   // LDS byte address (dst for load / src for store)
    uint8_t  data_size_log2 = 0; // log2(bytes per element)
    uint16_t cluster_mask = 0;   // multicast (load only)
    uint32_t bar_lds_addr = 0;   // 0 => no auto-arrive
    uint16_t pad_interval = 0;   // element units (0 => no padding)
    uint16_t pad_amount   = 0;   // element units
    uint8_t  rank         = 2;   // 2..5 for affine/dense
    tdm_mode mode         = tdm_mode::affine;
    tdm_dir  dir          = tdm_dir::load;
    union {
        struct {
            uint32_t tdim[5];      // innermost-first tensor extents
            uint64_t tstride[4];   // innermost-first element strides
            uint16_t tile[5];      // innermost-first tile dims
        } aff;
        struct {
            uint32_t tensor_cols, tensor_rows, row_stride;
            uint16_t row_len;      // columns per gathered row (tile_dim0)
            uint16_t n;            // rows gathered (tile_dim1)
            bool     idx_32bit;
            uint32_t rows_packed[8];  // groups 2-3 index words
        } gat;
    } addr;
};

/**
 * @brief The single lowering point: pack a `tdm_desc` into the four operands.
 *
 * Bit positions reproduce the oracle-validated layout (see file header). The
 * descriptor is a 640-bit value spread across four SGPR operands -- group 0
 * (4 DWords), group 1 (8 DWords), group 2 (4 DWords), group 3 (4 DWords) --
 * consumed by `tensor_load_to_lds` / `tensor_store_from_lds`. Multi-byte
 * fields are little-endian and several straddle a DWord boundary, so the
 * packing below masks/shifts each field into one or two adjacent DWords.
 *
 * Field map (innermost-first numbering; "tdimK"/"tstrideK"/"tileK" are the
 * K-th dimension counting from the contiguous/innermost axis outward):
 *
 *   g0[0]  bit0     = pred (must be 1 or the op is a no-op)
 *          bit30    = gather index size (0 = u16, 1 = u32)
 *          bit31    = gather enable (0 = dense/affine, 1 = gather/scatter)
 *   g0[1]  = LDS byte address (dst for load, src for store)
 *   g0[2]  = global address bits [31:0]
 *   g0[3]  bits[24:0]  = global address bits [56:32]
 *          bits[31:30] = type field = 0x2
 *   g1[0]  bits[15:0]  = multicast workgroup mask (load only)
 *          bits[17:16] = data_size (log2 bytes/elem: 0=1B,1=2B,2=4B,3=8B)
 *          bit18       = atomic_barrier_enable (auto-arrive)
 *          bit20       = pad_enable
 *          bits[24:22] = pad_interval (log2 dwords - 1)
 *          bits[30:25] = pad_amount   (dwords - 1)
 *   g1[1..4]  = barrier addr (low 16b) + tdim0/tdim1 + tile0/1/2
 *   g1[5..7]  = tstride0 (64b) + tstride1 (48b)
 *   g2/g3     = tdim2/3/4, tstride2/3, tile3/4  (affine)  -- OR --
 *               16 u16 / 8 u32 packed row indices         (gather)
 */
__device__ __forceinline__ void encode(
    const tdm_desc& d, v4u32& g0, v8u32& g1, v4u32& g2, v4u32& g3)
{
    // Start from all-zero so every unfilled field (higher dims on low-rank
    // tiles, the unused union half, etc.) is a defined zero, not garbage.
    g0 = v4u32{0, 0, 0, 0};
    g1 = v8u32{0, 0, 0, 0, 0, 0, 0, 0};
    g2 = v4u32{0, 0, 0, 0};
    g3 = v4u32{0, 0, 0, 0};

    const bool gather = (d.mode == tdm_mode::gather);

    // ---- group 0: pred, gather flags, lds_addr, global_addr, type ----
    // pred=1 is mandatory; the two MSBs select gather mode and index width.
    g0[0] = 1u                                                   // pred = 1
          | (gather ? (1u << 31) : 0u)                           // gather enable
          | ((gather && d.addr.gat.idx_32bit) ? (1u << 30) : 0u);// index size
    g0[1] = d.lds_addr;                                          // LDS byte addr
    g0[2] = uint32_t(d.base);                                    // global addr [31:0]
    // The global address is 57 bits: low 32 go in g0[2], the next 25 in g0[3]
    // bits[24:0]; the top two bits of g0[3] hold the constant type field (0x2).
    g0[3] = (uint32_t(d.base >> 32) & 0x01FFFFFFu) | (2u << 30); // addr[56:32] | type

    // ---- group 1 word 0: control (data_size, pad, barrier, multicast) ----
    // bit18 turns on the TDM auto-arrive (DS async-barrier) when an LDS
    // barrier address was supplied; otherwise the caller drains TENSORcnt.
    const uint32_t bar_enable = (d.bar_lds_addr != 0) ? (1u << 18) : 0u;
    uint32_t pad_enable = 0, pad_int_enc = 0, pad_amt_enc = 0;
    if (d.pad_interval > 0) {
        // The HW expresses padding in DWords, not elements, so convert via
        // the element size. interval is stored log2-1 and amount is stored
        // minus-1 (both per SP3); e.g. bf16 <128,8> -> interval 5, amount 3.
        const uint32_t bytes = 1u << d.data_size_log2;
        pad_enable  = 1u;
        pad_int_enc = __builtin_ctz(uint32_t(d.pad_interval) * bytes / 4u) - 1u;
        pad_amt_enc = uint32_t(d.pad_amount) * bytes / 4u - 1u;
    }
    g1[0] = (uint32_t(d.cluster_mask) & 0xFFFFu)   // [15:0]  multicast mask
          | (uint32_t(d.data_size_log2) << 16)     // [17:16] data_size
          |  bar_enable                            // [18]    auto-arrive
          | (pad_enable  << 20)                    // [20]    pad_enable
          | (pad_int_enc << 22)                    // [24:22] pad_interval
          | (pad_amt_enc << 25);                   // [30:25] pad_amount

    if (gather) {
        // Gather/scatter is always 2D. tile_dim0 = columns per row,
        // tile_dim1 = number of rows the HW will read indices for (== n).
        const auto& g = d.addr.gat;
        const uint32_t tdim0 = g.tensor_cols, tdim1 = g.tensor_rows;
        const uint32_t til0 = g.row_len,      til1  = g.n;
        // tdim0/tdim1 are 32-bit but their fields begin at bit 16 of g1[1]/g1[2],
        // so each splits across two DWords (low 16 here, high 16 next word).
        g1[1] = (uint32_t(d.bar_lds_addr) & 0xFFFFu) | (tdim0 << 16);
        g1[2] = (tdim0 >> 16) | (tdim1 << 16);
        g1[3] = (tdim1 >> 16) | (til0 << 16);
        g1[4] = til1;
        const uint64_t s0 = uint32_t(g.row_stride);  // stride between source rows
        g1[5] = uint32_t(s0);
        g1[6] = uint32_t(s0 >> 32);
        // groups 2-3 carry the packed row-index list instead of higher dims
        // (mutually exclusive with the affine layout below).
        g2[0] = g.rows_packed[0]; g2[1] = g.rows_packed[1];
        g2[2] = g.rows_packed[2]; g2[3] = g.rows_packed[3];
        g3[0] = g.rows_packed[4]; g3[1] = g.rows_packed[5];
        g3[2] = g.rows_packed[6]; g3[3] = g.rows_packed[7];
        return;
    }

    // ---- affine / dense: innermost-first dims & strides ----
    // `rank` controls how many dims are live; the `if constexpr`-style guards
    // here are runtime but every higher field defaults to the zero we wrote
    // above, so a rank-2 (dense) descriptor leaves g2/g3 at zero.
    const auto& a = d.addr.aff;
    const int rank = d.rank;
    const uint32_t tdim0 = a.tdim[0];                       // innermost extent (cols)
    const uint32_t tdim1 = (rank >= 2) ? a.tdim[1] : 0u;    // next extent (rows)
    const uint16_t til0  = a.tile[0];                       // innermost tile (cols)
    const uint16_t til1  = (rank >= 2) ? a.tile[1] : 0;     // next tile (rows)
    const uint16_t til2  = (rank >= 3) ? a.tile[2] : 0;     // axis-2 tile

    // Same straddling layout as the gather case: tdim0/tdim1 each span two
    // DWords starting at bit 16; tile0 rides the top half of g1[3].
    g1[1] = (uint32_t(d.bar_lds_addr) & 0xFFFFu) | (tdim0 << 16);
    g1[2] = (tdim0 >> 16) | (tdim1 << 16);
    g1[3] = (tdim1 >> 16) | (uint32_t(til0) << 16);
    g1[4] = uint32_t(til1) | (uint32_t(til2) << 16);

    if (rank >= 2) {
        // tstride0 = elements between innermost rows; a full 64-bit field.
        const uint64_t s0 = a.tstride[0];
        g1[5] = uint32_t(s0);
        g1[6] = uint32_t(s0 >> 32);
    }
    if (rank >= 3) {
        // tstride1 = elements between axis-1 planes; 48-bit, low 16 share
        // g1[6]'s high half, the remaining 32 fill g1[7].
        const uint64_t s1 = a.tstride[1];
        g1[6] |= (uint32_t(s1) & 0xFFFFu) << 16;
        g1[7]  =  uint32_t(s1 >> 16);
        g2[0]  = a.tdim[2];                                 // axis-2 extent (32-bit)
    }
    if (rank >= 4) {
        g2[1] = a.tdim[3];                                 // axis-3 extent (32-bit)
        // tstride2 = 48-bit: low 32 in g2[2], high 16 in g2[3][15:0];
        // tile3 occupies g2[3][31:16].
        const uint64_t s2 = a.tstride[2];
        g2[2] = uint32_t(s2);
        g2[3] = (uint32_t(s2 >> 32) & 0xFFFFu) | (uint32_t(a.tile[3]) << 16);
    }
    if (rank == 5) {
        // The outermost axis is the most fragmented: tstride3 is 48-bit
        // (low 32 in g3[0], high 16 in g3[1][15:0]); tdim4 is 32-bit split
        // across g3[1][31:16] and g3[2][15:0]; tile4 sits in g3[2][31:16].
        const uint64_t s3 = a.tstride[3];
        g3[0] = uint32_t(s3);
        g3[1] = (uint32_t(s3 >> 32) & 0xFFFFu) | ((a.tdim[4] & 0xFFFFu) << 16);
        g3[2] = ((a.tdim[4] >> 16) & 0xFFFFu)  | (uint32_t(a.tile[4]) << 16);
    }
}

template<typename T>
__host__ __device__ constexpr uint8_t data_size_log2() {
    return (sizeof(T) == 1) ? 0 : (sizeof(T) == 2) ? 1 : (sizeof(T) == 4) ? 2 : 3;
}

/// @brief Global element pointer at the top-left corner of tile `idx`.
template<int TILE_R, int TILE_C, typename T, ducks::gl::all GL,
         ducks::coord::tile COORD>
__device__ __forceinline__ const T* tile_base(const GL& src, const COORD& idx) {
    const int gr_base = idx.r * TILE_R;
    const int gc_base = idx.c * TILE_C;
    return src.raw_ptr
         + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
            * src.cols() + gc_base);
}

/**
 * @brief Fill the innermost-first dims/strides shared by dense and affine.
 *
 * This is where "derive by default" happens: the innermost two axes are read
 * off the tile (`ST::rows`/`ST::cols`) and the source layout (`gl`), so the
 * caller never repeats them. The outer axes (2..4), which cannot be inferred,
 * come straight from the `tdm::affine` value.
 *
 * Index mapping between the two conventions:
 *   - descriptor arrays are innermost-first: index 0 = contiguous axis.
 *   - `affine` exposes axes outward as dim/stride/tile triples, so its k-th
 *     extra axis lands at descriptor index 2+k, and its k-th stride at
 *     tstride index 1+k (tstride[0] is always the innermost row stride).
 */
template<typename ST, ducks::gl::all GL>
__device__ __forceinline__ void fill_affine_dims(
    tdm_desc& d, const GL& src, const affine& a)
{
    d.mode = tdm_mode::affine;
    d.rank = uint8_t(2 + a.ndim);           // 2 innermost + ndim outer axes
    auto& af = d.addr.aff;
    // Clear all slots first; encode() reads every index up to `rank`.
    #pragma unroll
    for (int i = 0; i < 5; ++i) { af.tdim[i] = 0; af.tile[i] = 0; }
    #pragma unroll
    for (int i = 0; i < 4; ++i) { af.tstride[i] = 0; }

    // Innermost two axes derived from the tile (LDS shape) + tensor (gl).
    af.tdim[0] = uint32_t(src.cols());                            // axis-0 extent
    af.tdim[1] = uint32_t(src.rows());                            // axis-1 extent
    af.tile[0] = uint16_t(ST::cols);                             // axis-0 tile width
    af.tile[1] = uint16_t(ST::rows);                            // axis-1 tile height
    af.tstride[0] = uint64_t(uint32_t(src.template stride<2>()));  // row stride (elems)

    // Outer axes (2..4) copied verbatim from the affine descriptor.
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        if (k < a.ndim) {
            af.tdim[2 + k]    = a.dim[k];
            af.tile[2 + k]    = a.tile[k];
            af.tstride[1 + k] = a.stride[k];
        }
    }
}

/**
 * @brief Pack up to `cap` row indices into the 8 group-2/3 index words.
 *
 * The HW row capacity is fixed by the index width: 16 rows of u16 (two per
 * DWord) or 8 rows of u32 (one per DWord), so the same 8 DWords (g2[0..3],
 * g3[0..3]) hold either packing. Slots past `n` stay zero; `encode` only
 * lets the HW read `n` of them via tile_dim1.
 */
__device__ __forceinline__ void pack_rows(
    tdm_desc& d, const uint32_t* rows, int n, idx_width w)
{
    auto& g = d.addr.gat;
    #pragma unroll
    for (int i = 0; i < 8; ++i) g.rows_packed[i] = 0;
    g.idx_32bit = (w == idx_width::b32);
    if (w == idx_width::b16) {
        // u16 mode: 2 indices per DWord (even index -> low half, odd -> high).
        for (int i = 0; i < n && i < 16; ++i) {
            const uint32_t v = uint32_t(rows[i]) & 0xFFFFu;
            g.rows_packed[i >> 1] |= v << (16 * (i & 1));
        }
    } else {
        // u32 mode: 1 index per DWord, 8 rows max.
        for (int i = 0; i < n && i < 8; ++i) g.rows_packed[i] = uint32_t(rows[i]);
    }
}

} // namespace detail

/* ============================================================ *
 *  Completion helpers (per-descriptor TENSORcnt)
 * ============================================================ */

/// @brief Drain pending TDM transfers, leaving at most N in flight.
template<int N = 0>
__device__ __forceinline__ void load_async_wait() { sync::wait_tdm<N>(); }

/// @brief Drain pending TDM transfers, leaving at most N in flight.
template<int N = 0>
__device__ __forceinline__ void store_async_wait() { sync::wait_tdm<N>(); }

/* ============================================================ *
 *  Gather streaming range
 * ============================================================ */

/// @brief One chunk of a gather stream: a span of `n` row indices.
struct gather_chunk { const uint32_t* rows; int n; };

/**
 * @brief Range over a row-index array yielding per-descriptor chunks.
 *
 * Splits `total` indices into chunks of the per-descriptor cap (16 for b16,
 * 8 for b32). Compose with `load_tdm` + a destination ring + a barrier ring;
 * the in-flight budget stays the caller's explicit concern.
 */
struct gather_stream {
    const uint32_t* rows; int total; int cap;
    __host__ __device__ gather_stream(const uint32_t* r, int n, idx_width w)
        : rows(r), total(n), cap(w == idx_width::b16 ? 16 : 8) {}
    struct iterator {
        const uint32_t* rows; int off, total, cap;
        // Current chunk: a `cap`-sized window, clamped on the final partial one.
        __host__ __device__ gather_chunk operator*() const {
            int n = total - off; if (n > cap) n = cap; return {rows + off, n};
        }
        __host__ __device__ iterator& operator++() { off += cap; return *this; }
        // The end sentinel is ignored on purpose: a range-for stops as soon as
        // `off` reaches `total`, which is all this loop needs to test.
        __host__ __device__ bool operator!=(const iterator&) const { return off < total; }
    };
    __host__ __device__ iterator begin() const { return {rows, 0, total, cap}; }
    __host__ __device__ iterator end()   const { return {rows, total, total, cap}; }
};

} // namespace tdm

/* ============================================================ *
 *  Public verbs (kittens::) -- mode dispatched by argument type
 * ============================================================ */

/* ----------  DENSE / AFFINE LOAD (G -> LDS)  ---------- */

/// @brief N-D affine TDM load. Pass `tdm::affine` for axes 2-4.
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(
    st<T, ROWS, COLS, Shape>& dst, const GL& src, const COORD& idx,
    const tdm::affine& a, const tdm::tdm_opts& opts = {})
{
    // Assemble the descriptor entirely from derived data + the opts/affine
    // values, then lower it once. `base` is the global address of this tile's
    // top-left corner; `lds_addr` is where the engine deposits the tile.
    tdm::detail::tdm_desc d;
    d.base = reinterpret_cast<uint64_t>(
        tdm::detail::tile_base<ROWS, COLS, T, GL, COORD>(src, idx));
    d.lds_addr = uint32_t(reinterpret_cast<uintptr_t>(&dst.data[0]));
    d.data_size_log2 = tdm::detail::data_size_log2<T>();        // from element type
    d.cluster_mask = uint16_t(opts.cluster);                   // multicast (load only)
    d.bar_lds_addr = uint32_t(reinterpret_cast<uintptr_t>(opts.arrive));  // auto-arrive
    d.pad_interval = uint16_t(tdm::detail::pad_of<Shape>::interval);  // from st shape
    d.pad_amount   = uint16_t(tdm::detail::pad_of<Shape>::amount);
    d.dir = tdm::detail::tdm_dir::load;
    tdm::detail::fill_affine_dims<st<T, ROWS, COLS, Shape>, GL>(d, src, a);

    tdm::detail::v4u32 g0, g2, g3; tdm::detail::v8u32 g1;
    tdm::detail::encode(d, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/// @brief Dense N-D TDM load (innermost 2D derived; no extra axes).
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(
    st<T, ROWS, COLS, Shape>& dst, const GL& src, const COORD& idx,
    const tdm::tdm_opts& opts = {})
{
    load_tdm(dst, src, idx, tdm::affine{}, opts);
}

/* ----------  DENSE / AFFINE STORE (LDS -> G)  ---------- */

/// @brief N-D affine TDM store. Mirrors `load_tdm` with flipped arg order.
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store_tdm(
    const GL& dst, st<T, ROWS, COLS, Shape>& src, const COORD& idx,
    const tdm::affine& a, const tdm::tdm_opts& opts = {})
{
    tdm::detail::tdm_desc d;
    d.base = reinterpret_cast<uint64_t>(
        tdm::detail::tile_base<ROWS, COLS, T, GL, COORD>(dst, idx));
    d.lds_addr = uint32_t(reinterpret_cast<uintptr_t>(&src.data[0]));
    d.data_size_log2 = tdm::detail::data_size_log2<T>();
    d.cluster_mask = 0;  // stores ignore the multicast mask
    d.bar_lds_addr = uint32_t(reinterpret_cast<uintptr_t>(opts.arrive));
    d.pad_interval = uint16_t(tdm::detail::pad_of<Shape>::interval);
    d.pad_amount   = uint16_t(tdm::detail::pad_of<Shape>::amount);
    d.dir = tdm::detail::tdm_dir::store;
    tdm::detail::fill_affine_dims<st<T, ROWS, COLS, Shape>, GL>(d, dst, a);

    tdm::detail::v4u32 g0, g2, g3; tdm::detail::v8u32 g1;
    tdm::detail::encode(d, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

/// @brief Dense N-D TDM store.
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store_tdm(
    const GL& dst, st<T, ROWS, COLS, Shape>& src, const COORD& idx,
    const tdm::tdm_opts& opts = {})
{
    store_tdm(dst, src, idx, tdm::affine{}, opts);
}

/* ----------  GATHER (G -> LDS) / SCATTER (LDS -> G)  ---------- */

/**
 * @brief Row-indexed gather load: pack rows `M[rows[i]]` contiguously in LDS.
 *
 * `n` rows of `COLS` columns are gathered (column origin 0). The HW caps the
 * row count at 16 (`idx_width::b16`) or 8 (`idx_width::b32`); `opts.idx_w`
 * selects the index width. Indices must be sorted, unique, wave-uniform.
 */
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL>
__device__ inline void load_tdm(
    st<T, ROWS, COLS, Shape>& dst, const GL& src,
    const uint32_t* rows, int n, const tdm::tdm_opts& opts = {})
{
    tdm::detail::tdm_desc d;
    d.mode = tdm::detail::tdm_mode::gather;
    d.dir  = tdm::detail::tdm_dir::load;
    d.base = reinterpret_cast<uint64_t>(src.raw_ptr);
    d.lds_addr = uint32_t(reinterpret_cast<uintptr_t>(&dst.data[0]));
    d.data_size_log2 = tdm::detail::data_size_log2<T>();
    d.bar_lds_addr = uint32_t(reinterpret_cast<uintptr_t>(opts.arrive));
    d.pad_interval = uint16_t(tdm::detail::pad_of<Shape>::interval);
    d.pad_amount   = uint16_t(tdm::detail::pad_of<Shape>::amount);
    auto& g = d.addr.gat;
    g.tensor_cols = uint32_t(src.cols());
    g.tensor_rows = uint32_t(src.rows());
    g.row_stride  = uint32_t(src.template stride<2>());
    g.row_len     = uint16_t(COLS);
    g.n           = uint16_t(n);
    tdm::detail::pack_rows(d, rows, n, opts.idx_w);

    tdm::detail::v4u32 g0, g2, g3; tdm::detail::v8u32 g1;
    tdm::detail::encode(d, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Row-indexed scatter store: write LDS rows back to `M[rows[i]]`.
 *
 * Same descriptor as gather with the store verb. An out-of-range
 * destination index is skipped by the HW rather than zero-filled.
 */
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL>
__device__ inline void store_tdm(
    const GL& dst, st<T, ROWS, COLS, Shape>& src,
    const uint32_t* rows, int n, const tdm::tdm_opts& opts = {})
{
    tdm::detail::tdm_desc d;
    d.mode = tdm::detail::tdm_mode::gather;
    d.dir  = tdm::detail::tdm_dir::store;
    d.base = reinterpret_cast<uint64_t>(dst.raw_ptr);
    d.lds_addr = uint32_t(reinterpret_cast<uintptr_t>(&src.data[0]));
    d.data_size_log2 = tdm::detail::data_size_log2<T>();
    d.bar_lds_addr = uint32_t(reinterpret_cast<uintptr_t>(opts.arrive));
    d.pad_interval = uint16_t(tdm::detail::pad_of<Shape>::interval);
    d.pad_amount   = uint16_t(tdm::detail::pad_of<Shape>::amount);
    auto& g = d.addr.gat;
    g.tensor_cols = uint32_t(dst.cols());
    g.tensor_rows = uint32_t(dst.rows());
    g.row_stride  = uint32_t(dst.template stride<2>());
    g.row_len     = uint16_t(COLS);
    g.n           = uint16_t(n);
    tdm::detail::pack_rows(d, rows, n, opts.idx_w);

    tdm::detail::v4u32 g0, g2, g3; tdm::detail::v8u32 g1;
    tdm::detail::encode(d, g0, g1, g2, g3);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

/* ----------  ITERATE -- HARD-STOP  ---------- */

namespace tdm { namespace detail {
template<typename> struct iterate_hardstop : std::false_type {};
} }

/// @brief Iterate mode is hard-stopped: passing `tdm::iterate` won't compile.
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(
    st<T, ROWS, COLS, Shape>&, const GL&, const COORD&,
    const tdm::iterate&, const tdm::tdm_opts& = {})
{
    static_assert(tdm::detail::iterate_hardstop<T>::value,
        "tdm::iterate is hard-stopped: the iterate "
        "sub-field offsets are unverified and its behavior is untested by "
        "the correctness oracle. Issue separate descriptors per tile instead.");
}

} // namespace kittens

#endif // KITTENS_UDNA1
