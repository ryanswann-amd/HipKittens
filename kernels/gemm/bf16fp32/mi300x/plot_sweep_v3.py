#!/usr/bin/env python3
"""Generate scatter plot and ratio heatmap from sweep_v3.csv."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '3'  # prevent CUDA init issues

import csv
import numpy as np

# Use Agg backend (no display needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

OUT_DIR = '/data0/ryaswann/orchestrators/hipkittens_gemm_lib/human_context/hipkittens/HIPKITTENS-0004-gemm-library'
CSV_PATH = os.path.join(OUT_DIR, 'sweep_v3.csv')

def load_data():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            M = int(r['M'])
            N = int(r['N'])
            K = int(r['K'])
            hk_tf = float(r['hk_tflops'])
            hbl_tf = float(r['hbl_tflops'])
            ratio = float(r['ratio_pct'])
            tile = int(r['tile_selected'])
            ai = float(r['arith_intensity'])
            if hk_tf > 0 and hbl_tf > 0:
                rows.append((M, N, K, hk_tf, hbl_tf, ratio, tile, ai))
    return rows


def plot_scatter(rows):
    """Plot 1: Arithmetic intensity vs TFLOPS for both HK and hBL."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ai_vals = [r[7] for r in rows]
    hk_vals = [r[3] for r in rows]
    hbl_vals = [r[4] for r in rows]

    ax.scatter(ai_vals, hbl_vals, c='red', alpha=0.25, s=8, label='hipBLASLt (torch.mm)', zorder=2)
    ax.scatter(ai_vals, hk_vals, c='blue', alpha=0.25, s=8, label='HipKittens', zorder=3)

    # Add BF16 peak line
    ax.axhline(y=1307, color='green', linestyle='--', alpha=0.5, label='MI300X BF16 Peak (1307 TF)')

    ax.set_xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    ax.set_ylabel('TFLOPS', fontsize=12)
    ax.set_title('BF16 NT GEMM: HipKittens vs hipBLASLt — Roofline View\n'
                 f'8129 shapes, M/N: 128-8192 step 128, K: max(M,N) or 4096', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=700)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'sweep_v3_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved: {path}")
    plt.close(fig)


def plot_heatmap(rows):
    """Plot 2: M vs N ratio heatmap (using K=4096 subset where available, else K=max(M,N))."""
    # Build M x N grid. Use K=4096 data where available, fall back to K=max(M,N).
    # Priority: K=4096 > K=max(M,N)
    grid = {}  # (M, N) → ratio
    for M, N, K, hk_tf, hbl_tf, ratio, tile, ai in rows:
        key = (M, N)
        if key not in grid or K == 4096:
            # Filter extreme outliers (likely GPU contention artifacts)
            if 10 < ratio < 300:
                grid[key] = ratio

    dims = sorted(set(M for M, N in grid.keys()))
    n_dims = sorted(set(N for M, N in grid.keys()))

    # Create 2D array
    ratio_map = np.full((len(dims), len(n_dims)), np.nan)
    m_idx = {m: i for i, m in enumerate(dims)}
    n_idx = {n: i for i, n in enumerate(n_dims)}

    for (M, N), ratio in grid.items():
        if M in m_idx and N in n_idx:
            ratio_map[m_idx[M], n_idx[N]] = ratio

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Diverging colormap centered at 100%
    norm = TwoSlopeNorm(vmin=30, vcenter=100, vmax=130)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(ratio_map, origin='lower', aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    # Axis labels (show every 4th)
    step = max(1, len(dims) // 16)
    ax.set_xticks(range(0, len(n_dims), step))
    ax.set_xticklabels([str(n_dims[i]) for i in range(0, len(n_dims), step)], rotation=45, fontsize=8)
    ax.set_yticks(range(0, len(dims), step))
    ax.set_yticklabels([str(dims[i]) for i in range(0, len(dims), step)], fontsize=8)

    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('M', fontsize=12)
    ax.set_title('BF16 NT GEMM Ratio: HipKittens / hipBLASLt (%)\n'
                 'Green = HK wins, Red = hBL wins', fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='HK / hBL (%)')

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'sweep_v3_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved: {path}")
    plt.close(fig)


def plot_tile_analysis(rows):
    """Plot 3: TFLOPS by tile, showing where each tile is selected."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    tiles = [128, 192, 256]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    for idx, (tile, color) in enumerate(zip(tiles, colors)):
        ax = axes[idx]
        tile_rows = [r for r in rows if r[6] == tile]

        if not tile_rows:
            ax.set_title(f'{tile}x{tile}: no data')
            continue

        ratios = [r[5] for r in tile_rows]
        flops = [2 * r[0] * r[1] * r[2] for r in tile_rows]

        ax.scatter(flops, ratios, c=color, alpha=0.3, s=12)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)

        avg = sum(ratios) / len(ratios)
        wins = sum(1 for r in ratios if r > 100)
        ax.set_title(f'{tile}x{tile} tile ({len(tile_rows)} shapes)\n'
                     f'Avg: {avg:.1f}%, Wins: {wins}/{len(tile_rows)} ({100*wins/len(tile_rows):.0f}%)',
                     fontsize=11)
        ax.set_xlabel('Total FLOPs', fontsize=10)
        ax.set_ylabel('HK / hBL (%)', fontsize=10)
        ax.set_ylim(10, 160)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    fig.suptitle('BF16 NT: Performance Ratio by Tile Size', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'sweep_v3_tiles.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Tile analysis saved: {path}")
    plt.close(fig)


if __name__ == '__main__':
    rows = load_data()
    print(f"Loaded {len(rows)} data points")

    # Filter extreme outliers for statistics
    clean = [r for r in rows if 10 < r[5] < 300]
    ratios = [r[5] for r in clean]
    print(f"\nClean data (10% < ratio < 300%): {len(clean)} points")
    print(f"Average ratio:  {sum(ratios)/len(ratios):.1f}%")
    print(f"Median ratio:   {sorted(ratios)[len(ratios)//2]:.1f}%")
    print(f"Win rate:       {sum(1 for r in ratios if r > 100)}/{len(clean)} ({100*sum(1 for r in ratios if r > 100)/len(clean):.1f}%)")
    print(f"Below 80%:      {sum(1 for r in ratios if r < 80)}/{len(clean)}")
    print(f"Below 50%:      {sum(1 for r in ratios if r < 50)}/{len(clean)}")

    plot_scatter(rows)
    plot_heatmap(rows)
    plot_tile_analysis(rows)
    print("\nAll plots generated!")
