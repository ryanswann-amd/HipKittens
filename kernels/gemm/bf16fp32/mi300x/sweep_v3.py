#!/usr/bin/env python3
"""Comprehensive BF16 NT sweep: HipKittens vs hipBLASLt across ~8K shapes.

Tests M, N from 128 to 8192 in steps of 128 with K = max(M,N) and K = 4096.
Generates scatter plot (arithmetic intensity vs TFLOPS) and ratio heatmap.
"""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '3'

import torch
import sys
import time
import csv
import math
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hipkittens_gemm as hk

# Output paths
OUT_DIR = '/data0/ryaswann/orchestrators/hipkittens_gemm_lib/human_context/hipkittens/HIPKITTENS-0004-gemm-library'
CSV_PATH = os.path.join(OUT_DIR, 'sweep_v3.csv')

WARMUP = 3
ITERS = 5
DTYPE = torch.bfloat16
STEP = 128  # M,N step size
MAX_DIM = 8192

def bench_hbl(A, B, warmup=WARMUP, iters=ITERS):
    """Benchmark hipBLASLt via torch.mm (A @ B^T)."""
    for _ in range(warmup):
        torch.mm(A, B.t())
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(A, B.t())
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed

def bench_hk(A, B, warmup=WARMUP, iters=ITERS):
    """Benchmark HipKittens gemm_nt."""
    M, K = A.shape
    N = B.shape[0]

    # Determine output dtype
    c_dtype = torch.bfloat16
    C = torch.zeros(M, N, dtype=c_dtype, device=A.device)

    for _ in range(warmup):
        C.zero_()
        hk.gemm(A, B, C, trans='nt')
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        C.zero_()
        hk.gemm(A, B, C, trans='nt')
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed

def compute_tflops(M, N, K, elapsed):
    """Compute TFLOPS from dimensions and elapsed time."""
    flops = 2.0 * M * N * K
    return flops / elapsed / 1e12

def arithmetic_intensity_bf16(M, N, K):
    """Arithmetic intensity = FLOPs / bytes_transferred.
    BF16: 2 bytes per element.
    A: M*K, B: N*K (NT), C: M*N elements.
    """
    flops = 2.0 * M * N * K
    bytes_moved = (M * K + N * K + M * N) * 2  # BF16 = 2 bytes
    return flops / bytes_moved if bytes_moved > 0 else 0

def get_tile_selected(M, N, K):
    """Get which tile HipKittens selects for this shape."""
    try:
        _, bs = hk._select_tile_best('bf16', 'nt', M, N, K)
        return bs
    except Exception:
        return 0

def main():
    torch.cuda.set_device(0)  # device 0 within HIP_VISIBLE_DEVICES=3

    # Build shape list
    dims = list(range(STEP, MAX_DIM + 1, STEP))
    shapes = []
    for M in dims:
        for N in dims:
            # K = max(M, N)
            shapes.append((M, N, max(M, N)))
            # K = 4096 (but avoid duplicate if max(M,N) == 4096)
            if max(M, N) != 4096:
                shapes.append((M, N, 4096))

    total = len(shapes)
    print(f"Total shapes to test: {total}")
    print(f"Step size: {STEP}, dimensions: {len(dims)} ({dims[0]} to {dims[-1]})")
    print(f"Output CSV: {CSV_PATH}")
    print(f"GPU: HIP_VISIBLE_DEVICES=3")
    print()

    # Quick sanity check
    print("Sanity check: 1024x1024...")
    A = torch.randn(1024, 1024, dtype=DTYPE, device='cuda') / 10
    B = torch.randn(1024, 1024, dtype=DTYPE, device='cuda') / 10
    C_hk = hk.gemm(A, B, trans='nt')
    C_ref = torch.mm(A, B.t())
    err = (C_hk.float() - C_ref.float()).abs().max().item()
    print(f"  Max error: {err:.4f} ({'PASS' if err < 1.0 else 'FAIL'})")
    del A, B, C_hk, C_ref
    torch.cuda.empty_cache()

    # Run sweep
    results = []
    start_time = time.time()

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['M', 'N', 'K', 'dtype', 'hk_tflops', 'hbl_tflops', 'ratio_pct', 'tile_selected', 'arith_intensity'])

        for i, (M, N, K) in enumerate(shapes):
            try:
                A = torch.randn(M, K, dtype=DTYPE, device='cuda') / 10
                B = torch.randn(N, K, dtype=DTYPE, device='cuda') / 10

                tile = get_tile_selected(M, N, K)

                # Bench hipBLASLt
                t_hbl = bench_hbl(A, B)
                hbl_tf = compute_tflops(M, N, K, t_hbl)

                # Bench HipKittens
                t_hk = bench_hk(A, B)
                hk_tf = compute_tflops(M, N, K, t_hk)

                ratio = (hk_tf / hbl_tf * 100) if hbl_tf > 0 else 0
                ai = arithmetic_intensity_bf16(M, N, K)

                row = [M, N, K, 'bf16', f'{hk_tf:.1f}', f'{hbl_tf:.1f}', f'{ratio:.1f}', tile, f'{ai:.1f}']
                writer.writerow(row)
                results.append((M, N, K, hk_tf, hbl_tf, ratio, tile, ai))

                del A, B

            except Exception as e:
                row = [M, N, K, 'bf16', '0', '0', '0', 0, '0']
                writer.writerow(row)
                results.append((M, N, K, 0, 0, 0, 0, 0))
                if i < 10:  # Only print first few errors
                    print(f"  ERROR at M={M} N={N} K={K}: {e}")
                    traceback.print_exc()

            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>5}/{total}] M={M:>5} N={N:>5} K={K:>5} | "
                      f"HK={results[-1][3]:>6.1f} hBL={results[-1][4]:>6.1f} "
                      f"ratio={results[-1][5]:>5.1f}% tile={results[-1][6]} | "
                      f"ETA: {eta/60:.1f}min")
                f.flush()

            # Periodic GPU memory cleanup
            if (i + 1) % 500 == 0:
                torch.cuda.empty_cache()

    # Summary
    valid = [(m, n, k, hk_t, hbl_t, r, tile, ai) for m, n, k, hk_t, hbl_t, r, tile, ai in results if hk_t > 0]
    ratios = [r for _, _, _, _, _, r, _, _ in valid]

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE — BF16 NT")
    print(f"{'='*60}")
    print(f"Total shapes tested: {len(results)}")
    print(f"Successful:          {len(valid)}")
    print(f"Failed:              {len(results) - len(valid)}")
    print(f"Average ratio:       {sum(ratios)/len(ratios):.1f}%")
    print(f"Median ratio:        {sorted(ratios)[len(ratios)//2]:.1f}%")
    print(f"Win rate (>100%):    {sum(1 for r in ratios if r > 100)}/{len(valid)} ({100*sum(1 for r in ratios if r > 100)/len(valid):.1f}%)")
    print(f"Below 80%:           {sum(1 for r in ratios if r < 80)}/{len(valid)} ({100*sum(1 for r in ratios if r < 80)/len(valid):.1f}%)")
    print(f"Below 50%:           {sum(1 for r in ratios if r < 50)}/{len(valid)} ({100*sum(1 for r in ratios if r < 50)/len(valid):.1f}%)")

    # Tile distribution
    tile_counts = {}
    for _, _, _, _, _, _, tile, _ in valid:
        tile_counts[tile] = tile_counts.get(tile, 0) + 1
    print(f"\nTile distribution:")
    for tile in sorted(tile_counts.keys()):
        avg_r = sum(r for _, _, _, _, _, r, t, _ in valid if t == tile) / tile_counts[tile]
        wins = sum(1 for _, _, _, _, _, r, t, _ in valid if t == tile and r > 100)
        print(f"  {tile}x{tile}: {tile_counts[tile]:>5} shapes, avg ratio {avg_r:.1f}%, wins {wins}")

    # Best and worst shapes
    valid_sorted = sorted(valid, key=lambda x: x[5], reverse=True)
    print(f"\nTop 10 best shapes (HK vs hBL):")
    for m, n, k, hk_t, hbl_t, r, tile, ai in valid_sorted[:10]:
        print(f"  M={m:>5} N={n:>5} K={k:>5} | HK={hk_t:.1f} hBL={hbl_t:.1f} ratio={r:.1f}% tile={tile}")
    print(f"\nTop 10 worst shapes:")
    for m, n, k, hk_t, hbl_t, r, tile, ai in valid_sorted[-10:]:
        print(f"  M={m:>5} N={n:>5} K={k:>5} | HK={hk_t:.1f} hBL={hbl_t:.1f} ratio={r:.1f}% tile={tile}")

    total_elapsed = time.time() - start_time
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {CSV_PATH}")


if __name__ == '__main__':
    main()
