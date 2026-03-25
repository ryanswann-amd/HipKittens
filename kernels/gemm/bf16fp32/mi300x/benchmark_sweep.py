"""
Comprehensive GEMM benchmark sweep: HipKittens vs hipBLASLt
MI300X (gfx942), NT transpose, all dtypes, sizes 128-8192 step 128

Each dtype runs in a subprocess to isolate GPU faults.
Results are saved incrementally to CSV.
"""
import os, sys, csv, json, subprocess, signal

OUT_DIR = "/data0/ryaswann/orchestrators/hipkittens_gemm_lib/human_context/hipkittens/HIPKITTENS-0004-gemm-library"
CSV_PATH = os.path.join(OUT_DIR, "benchmark_sweep_nt.csv")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SIZES = list(range(128, 8320, 128))
DTYPES = ['bf16', 'fp16', 'fp32', 'fp8_e4m3fnuz']


def run_dtype_sweep(dtype_name):
    """Run benchmark for one dtype in a subprocess, return results."""
    worker = os.path.join(SCRIPT_DIR, "_bench_worker.py")
    env = os.environ.copy()
    env['HIP_VISIBLE_DEVICES'] = '0'

    results = []
    for N in SIZES:
        try:
            proc = subprocess.run(
                [sys.executable, worker, dtype_name, str(N)],
                capture_output=True, text=True, timeout=120,
                cwd=SCRIPT_DIR, env=env
            )
            line = proc.stdout.strip()
            if line and ',' in line:
                parts = line.split(',')
                hk_tf = float(parts[0])
                hbl_tf = float(parts[1])
                ratio = (hk_tf / hbl_tf * 100) if hbl_tf > 0 else 0.0
                results.append([N, dtype_name, round(hk_tf, 2), round(hbl_tf, 2), round(ratio, 1)])
                marker = ">>>" if hk_tf > hbl_tf else "   "
                print(f"  {marker} N={N:5d}  HK={hk_tf:7.1f} TF  hBL={hbl_tf:7.1f} TF  ratio={ratio:6.1f}%", flush=True)
            else:
                stderr_msg = proc.stderr.strip()[-200:] if proc.stderr else "no output"
                print(f"       N={N:5d}  SKIP ({stderr_msg})", flush=True)
                results.append([N, dtype_name, 0.0, 0.0, 0.0])
        except subprocess.TimeoutExpired:
            print(f"       N={N:5d}  TIMEOUT", flush=True)
            results.append([N, dtype_name, 0.0, 0.0, 0.0])
        except Exception as e:
            print(f"       N={N:5d}  ERROR: {e}", flush=True)
            results.append([N, dtype_name, 0.0, 0.0, 0.0])

    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    for dtype_name in DTYPES:
        print(f"\n{'='*60}")
        print(f"  Benchmarking dtype: {dtype_name} (NT transpose)")
        print(f"{'='*60}", flush=True)

        results = run_dtype_sweep(dtype_name)
        all_results.extend(results)

        # Incremental save
        with open(CSV_PATH, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['size', 'dtype', 'hk_tflops', 'hbl_tflops', 'ratio_pct'])
            w.writerows(all_results)

    print(f"\nCSV saved to: {CSV_PATH}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: HipKittens vs hipBLASLt — MI300X NT GEMM")
    print(f"{'='*70}")
    print(f"{'Dtype':>15} {'Avg%':>7} {'Min%':>7} {'Max%':>7} {'HK>hBL':>8} {'Count':>6}")
    print(f"{'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")

    for dtype_name in DTYPES:
        dtype_results = [r for r in all_results if r[1] == dtype_name and r[4] > 0]
        if not dtype_results:
            print(f"{dtype_name:>15}   (no valid results)")
            continue
        ratios = [r[4] for r in dtype_results]
        wins = sum(1 for r in ratios if r > 100)
        avg_r = sum(ratios) / len(ratios)
        min_r = min(ratios)
        max_r = max(ratios)
        total_pts = len(ratios)
        print(f"{dtype_name:>15} {avg_r:6.1f}% {min_r:6.1f}% {max_r:6.1f}% {wins:>5}/{total_pts:<3}")

    # Sizes where HK wins per dtype
    print(f"\n--- Sizes where HipKittens beats hipBLASLt ---")
    for dtype_name in DTYPES:
        wins = [r for r in all_results if r[1] == dtype_name and r[4] > 100]
        if wins:
            sizes_str = ", ".join(str(r[0]) for r in wins)
            print(f"  {dtype_name}: {sizes_str}")
        else:
            print(f"  {dtype_name}: (none)")

    # Generate plots
    generate_plots(all_results)


def generate_plots(results):
    """Generate TFLOPS comparison and ratio plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors_hk = '#2196F3'
    colors_hbl = '#FF9800'

    # --- Main TFLOPS Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("HipKittens vs hipBLASLt GEMM — MI300X (gfx942)\nNT Transpose, Square MxMxM", fontsize=14, fontweight='bold')

    for idx, dtype_name in enumerate(DTYPES):
        ax = axes[idx // 2][idx % 2]
        dtype_results = [r for r in results if r[1] == dtype_name and r[2] > 0]
        if not dtype_results:
            ax.set_title(f"{dtype_name} (no data)")
            continue

        sizes = [r[0] for r in dtype_results]
        hk = [r[2] for r in dtype_results]
        hbl = [r[3] for r in dtype_results]

        ax.plot(sizes, hk, color=colors_hk, linewidth=1.5, marker='o', markersize=2, label='HipKittens')
        ax.plot(sizes, hbl, color=colors_hbl, linewidth=1.5, marker='s', markersize=2, label='hipBLASLt')
        ax.set_title(f"{dtype_name.upper()}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Matrix Size (M=N=K)")
        ax.set_ylabel("TFLOPS")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 8500)

    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "benchmark_tflops_nt.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"TFLOPS plot saved to: {path1}")

    # --- Ratio Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("HipKittens / hipBLASLt Ratio — MI300X (gfx942)\nNT Transpose, Square MxMxM", fontsize=14, fontweight='bold')

    for idx, dtype_name in enumerate(DTYPES):
        ax = axes[idx // 2][idx % 2]
        dtype_results = [r for r in results if r[1] == dtype_name and r[4] > 0]
        if not dtype_results:
            ax.set_title(f"{dtype_name} (no data)")
            continue

        sizes = [r[0] for r in dtype_results]
        ratios = [r[4] for r in dtype_results]

        colors = ['#4CAF50' if r > 100 else '#F44336' for r in ratios]
        ax.scatter(sizes, ratios, c=colors, s=15, zorder=3)
        ax.plot(sizes, ratios, color='#666666', linewidth=0.8, alpha=0.5)
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Parity (100%)')
        ax.set_title(f"{dtype_name.upper()}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Matrix Size (M=N=K)")
        ax.set_ylabel("HK / hBL (%)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 8500)

    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "benchmark_ratio_nt.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ratio plot saved to: {path2}")


if __name__ == '__main__':
    main()
