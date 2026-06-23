/**
 * @file
 * @brief Internal TDM descriptor (D#) value type and group encoders (udna1).
 *
 * This header isolates the SGRP bit-packing for the Tensor Data Mover from the
 * user-facing ops in `global_to_shared.cuh`. It is the single lowering point:
 * the public `load_tdm` / `store_tdm` verbs fill a `detail::tdm_desc`, then call
 * `detail::encode_tdm<...>()` to emit the four operand groups for the builtin.
 *
 * Scope of THIS file today (the "foundation" / milestone 2 of
 * tdm-implementation-design.md):
 *   - groups 0/1 packing, lifted VERBATIM from the proven 2D encoder that
 *     shipped as `build_tdm_descriptor_2d` (renamed `build_tdm_groups01`), and
 *   - the dense affine path (groups 2/3 = 0), which is the existing 2D / dense
 *     N-D behavior.
 *
 * Explicitly NOT in this file (blocked): the groups 2/3 layouts for true 5D
 * affine, iterate, and gather/scatter. Those depend on the authoritative
 * field map in `docs/tensor_dma/32-tdm-gather.md` / `mi400_tx_api.xlsx`, which
 * is the "reconcile first" milestone. Per the design's rule -- *no fabricated
 * bit positions* -- the advanced builders below are left as hard stops rather
 * than guessed encodings.
 */

#pragma once

#include "../../../../common/common.cuh"

namespace kittens {
namespace detail {

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));
using v8u32 = unsigned int __attribute__((ext_vector_type(8)));

/// Which groups-2/3 addressing variant the descriptor carries. Mutually
/// exclusive because the three modes physically share SGPR groups 2 and 3.
enum class tdm_mode : uint8_t { affine, iterate, gather };
/// Transfer direction -- selects the load vs store builtin at dispatch.
enum class tdm_dir  : uint8_t { load, store };
/// Gather index packing width (groups 2/3 image). Unused until gather lands.
enum class idx_width : uint8_t { b16 = 0, b32 = 1 };

/**
 * @brief Device-constructible TDM descriptor image.
 *
 * Mirrors the hardware D#: groups 0/1 are common to every mode; groups 2/3 are
 * a tagged union selected by `mode`. This is a POD (no std::variant on device);
 * the foundation only constructs/encodes the `affine` arm with dense (<=2D)
 * extents, which encodes to g2 = g3 = 0.
 */
struct tdm_desc {
    // --- common (groups 0-1) ---
    uint64_t base         = 0;   // global byte address (57-bit HW field)
    uint32_t lds_addr     = 0;   // LDS byte address (dst for load / src for store)
    uint32_t tensor_rows  = 0;   // dim1 extent (elements)
    uint32_t tensor_cols  = 0;   // dim0 extent (elements)
    uint32_t row_stride_e = 0;   // dim0 (row) stride in ELEMENTS. See note in build_tdm_groups01.
    uint16_t cluster_mask = 0;   // multicast (load only)
    uint16_t bar_lds_addr = 0;   // 0 => ordering-only; else atomic_barrier_enable + addr

    tdm_mode mode = tdm_mode::affine;
    tdm_dir  dir  = tdm_dir::load;

    // --- addressing (groups 2-3): exactly ONE per `mode` ---
    // Foundation: only the dense affine arm (dims2-4 == 0) is honored.
    union {
        struct { uint32_t tdim[5]; uint32_t tstride[4]; uint16_t tile[5]; } affine;
        struct { uint32_t tdim[3]; uint32_t tstride[2]; uint16_t tile[3];
                 uint32_t lds_inc, gbl_inc, iter_count; } iterate;
        struct { uint16_t row_len; idx_width w; uint8_t n;
                 uint32_t rows_packed[8]; } gather;
    } addr = {};
};

/// Pad-interval accessor that is well-formed for shapes without padding.
template<typename Shape>
__device__ __host__ constexpr uint32_t detail_pad_interval() {
    if constexpr (requires { Shape::pad_interval; }) return (uint32_t)Shape::pad_interval;
    else return 0u;
}
/// Pad-amount accessor that is well-formed for shapes without padding.
template<typename Shape>
__device__ __host__ constexpr uint32_t detail_pad_amount() {
    if constexpr (requires { Shape::pad_amount; }) return (uint32_t)Shape::pad_amount;
    else return 0u;
}

/**
 * @brief Pack groups 0 + 1 of the TDM D# (common to every addressing mode).
 *
 * Bit layout is identical to the proven 2D encoder (`build_tdm_descriptor_2d`),
 * just sourced from a `tdm_desc` instead of loose arguments. The compile-time
 * tile dims / dtype / LDS padding still come through the template parameters,
 * matching `st<T, ROWS, COLS, Shape>`.
 *
 * @tparam Shape  Shared-tile shape (supplies pad_interval / pad_amount).
 * @tparam ROWS   Tile rows (tile_dim1).
 * @tparam COLS   Tile cols (tile_dim0).
 * @tparam T      Element type (sets data_size).
 */
template<typename Shape, int ROWS, int COLS, typename T>
__device__ __forceinline__ void build_tdm_groups01(v4u32& g0, v8u32& g1, const tdm_desc& d)
{
    // ---- Group 0: count, lds_addr, global_addr, type ----
    g0[0] = 1u;                                                  // count
    g0[1] = d.lds_addr;
    g0[2] = static_cast<uint32_t>(d.base);
    g0[3] = (static_cast<uint32_t>(d.base >> 32) & 0x01FFFFFFu) | (2u << 30);

    // ---- Group 1: data_size, padding, dims, stride, optional barrier ----
    // data_size encoded as log2(bytes_per_element).
    constexpr uint32_t data_size_enc = (sizeof(T) == 1) ? 0
                                     : (sizeof(T) == 2) ? 1
                                     : (sizeof(T) == 4) ? 2
                                     : 3;
    // Shapes without LDS padding (e.g. st_16x32, st_32x32) don't define
    // pad_interval/pad_amount; the accessors return 0 for those.
    constexpr uint32_t pad_interval = detail_pad_interval<Shape>();
    constexpr uint32_t pad_amount   = detail_pad_amount<Shape>();
    constexpr uint32_t pad_enable   = (pad_interval > 0) ? 1u : 0u;
    // [UNVERIFIED] LDS pad encoding. This is the upstream transform, preserved
    // as-is (NOT changed) because no pad encoding has been confirmed against a
    // model: under the functional model padding is NOT applied with these values, so
    // padded tiles round-trip wrong (off by pad_amount every pad_interval). The
    // dense / non-padded path IS verified element-exact. Correct pad bits need
    // the groups-0/1 field map reconciliation (milestone 1). Do not trust padded
    // TDM tiles until then.
    constexpr uint32_t pad_int_enc  = (pad_interval > 0)
        ? ( __builtin_ctz(pad_interval * sizeof(T) / 4) ) : 0;
    constexpr uint32_t pad_amt_enc  = (pad_amount > 0)
        ? ( (pad_amount * sizeof(T) / 4) - 1 ) : 0;

    // w0 = multicast_mask[15:0], data_size[17:16], atomic_barrier_enable[18],
    //      iterate_enable[19], pad_enable[20], pad_interval[24:22], pad_amount[31:25].
    const uint32_t atomic_bar_enable = (d.bar_lds_addr != 0) ? (1u << 18) : 0u;
    const uint32_t iterate_enable    = (d.mode == tdm_mode::iterate) ? (1u << 19) : 0u;

    uint32_t w0 = (data_size_enc << 16)
                | (pad_enable    << 20)
                |  atomic_bar_enable
                |  iterate_enable
                | (pad_int_enc   << 22)
                | (pad_amt_enc   << 25)
                | (d.cluster_mask & 0xFFFFu);

    const uint32_t tdim0    = d.tensor_cols;
    const uint32_t tdim1    = d.tensor_rows;
    const uint32_t tiledim0 = static_cast<uint32_t>(COLS);
    const uint32_t tiledim1 = static_cast<uint32_t>(ROWS);

    uint32_t w1 = (d.bar_lds_addr & 0xFFFFu) | (tdim0 << 16);
    uint32_t w2 = (tdim0 >> 16) | (tdim1 << 16);
    uint32_t w3 = (tdim1 >> 16) | (tiledim0 << 16);
    uint32_t w4 = tiledim1;

    // Row (dim0) stride is encoded in ELEMENTS, not bytes. NOTE: the original
    // upstream `build_tdm_descriptor_2d` wrote `row_stride * sizeof(T)` here,
    // which the TDM functional model consumes as elements ->
    // a 2x (sizeof(bf16)) row skip. Verified element-exact under the functional model with
    // the element-count form. (gemm_tdm_arrive never exercised this on a model,
    // so the byte form went unnoticed.)
    const uint64_t stride0 = static_cast<uint64_t>(d.row_stride_e);
    uint32_t w5 = static_cast<uint32_t>(stride0);
    uint32_t w6 = static_cast<uint32_t>(stride0 >> 32);
    uint32_t w7 = 0;

    g1[0] = w0; g1[1] = w1; g1[2] = w2; g1[3] = w3;
    g1[4] = w4; g1[5] = w5; g1[6] = w6; g1[7] = w7;
}

/**
 * @brief Groups 2/3 for the affine mode.
 *
 * Foundation: only the dense (<=2D) case is supported, where the higher
 * dims/strides/tiles are absent and groups 2/3 are all-zero -- exactly what the
 * 2D path emits today. A true 5D affine descriptor (non-zero tdim[2..4]) needs
 * the verified groups-2/3 field map and is intentionally NOT encoded here.
 */
__device__ __forceinline__ void build_tdm_g23_affine(v4u32& g2, v4u32& g3, const tdm_desc& d)
{
    const bool higher_dims = (d.addr.affine.tdim[2] | d.addr.affine.tdim[3] | d.addr.affine.tdim[4]) != 0;
    if (higher_dims) {
        // [BLOCKED] 5D affine groups-2/3 layout unverified (needs
        // docs/tensor_dma/32-tdm-gather.md). Refuse rather than emit guessed bits.
        __builtin_trap();
    }
    g2 = v4u32{0, 0, 0, 0};
    g3 = v4u32{0, 0, 0, 0};
}

/**
 * @brief Single lowering point: encode a `tdm_desc` into the four operand groups.
 *
 * Only the affine (dense) arm is wired in the foundation. iterate/gather are
 * compile-time/device hard stops until the groups-2/3 field map is reconciled.
 */
template<typename Shape, int ROWS, int COLS, typename T>
__device__ __forceinline__ void encode_tdm(const tdm_desc& d,
                                           v4u32& g0, v8u32& g1, v4u32& g2, v4u32& g3)
{
    build_tdm_groups01<Shape, ROWS, COLS, T>(g0, g1, d);
    switch (d.mode) {
        case tdm_mode::affine:  build_tdm_g23_affine(g2, g3, d); break;
        // [BLOCKED] iterate / gather groups-2/3 encoders pending the verified
        // field map. The public surface does not construct these modes yet.
        default:                __builtin_trap();
    }
}

} // namespace detail
} // namespace kittens
