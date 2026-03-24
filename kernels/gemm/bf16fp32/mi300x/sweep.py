"""Comprehensive GEMM library characterization sweep.
Runs ALL tile sizes at hundreds of problem sizes with correctness validation.
Outputs CSV for scatter plotting.

Usage: HIP_VISIBLE_DEVICES=0 python3 sweep.py [--quick]
"""

import sys, os, time, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

def load_kernels():
    kernels = {}
    for name in ['hk_128x128x64', 'hk_192x192x64', 'hk_256x256x64']:
        try:
            kernels[name] = __import__(name)
        except ImportError:
            pass
    return kernels

def bench_kernel(mod, A, B, C, warmup=3, iters=5):
    """Benchmark a kernel, return (tflops, max_err, status)."""
    N = A.shape[0]
    try:
        mod.dispatch(A, B, C)
        torch.cuda.synchronize()

        C_ref = torch.matmul(A, B.t().contiguous()).to(torch.bfloat16)
        max_err = (C.float() - C_ref.float()).abs().max().item()

        for _ in range(warmup):
            mod.dispatch(A, B, C)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start.record()
            mod.dispatch(A, B, C)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg_ms = sum(times) / len(times)
        tflops = 2.0 * N * N * N / (avg_ms * 1e9)
        return tflops, max_err, "PASS" if max_err < 0.1 else "FAIL"
    except Exception as e:
        return 0, -1, f"ERR:{e}"

def bench_hipblaslt(A, Bt, warmup=3, iters=5):
    N = A.shape[0]
    for _ in range(warmup):
        torch.matmul(A, Bt)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        torch.matmul(A, Bt)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return 2.0 * N * N * N / (sum(times) / len(times) * 1e9)

def generate_sizes(quick=False):
    """Generate a comprehensive set of square problem sizes."""
    sizes = set()

    if quick:
        # Quick mode: ~50 sizes
        sizes.update(range(128, 1024, 128))
        sizes.update(range(1024, 4096, 256))
        sizes.update(range(4096, 8192+1, 512))
    else:
        # Full mode: ~300+ sizes
        # Fine grain at small sizes
        sizes.update(range(128, 512, 64))
        # Medium grain
        sizes.update(range(512, 2048, 64))
        # Coarser at medium sizes
        sizes.update(range(2048, 4096, 128))
        # Large sizes
        sizes.update(range(4096, 8192+1, 256))
        # Non-power-of-2 critical sizes
        sizes.update([768, 1536, 2304, 3072, 3840, 4608, 5376, 6144, 6912, 7680])
        # Multiples of 192 (for 192x192 tile)
        sizes.update(range(192, 8192, 192))

    # Filter: must be divisible by at least one tile (128, 192, or 256)
    # AND divisible by K_STEP=64
    valid = sorted(s for s in sizes if s % 64 == 0 and (s % 128 == 0 or s % 192 == 0 or s % 256 == 0))
    return valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode (~50 sizes)')
    parser.add_argument('--output', default='sweep_results.csv', help='Output CSV file')
    args = parser.parse_args()

    kernels = load_kernels()
    tile_sizes = {name: int(name.split('_')[1].split('x')[0]) for name in kernels}

    print(f"Loaded kernels: {list(kernels.keys())}")
    sizes = generate_sizes(quick=args.quick)
    print(f"Testing {len(sizes)} sizes from {min(sizes)} to {max(sizes)}")

    results = []
    total = len(sizes)

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'tile', 'tflops', 'max_err', 'status', 'hipblaslt_tflops', 'ratio_pct'])

        for idx, N in enumerate(sizes):
            torch.manual_seed(42)
            A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
            B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
            Bt = B.t().contiguous()

            # hipBLASLt baseline
            hbl = bench_hipblaslt(A, Bt)

            # Test each compatible tile
            best_tf = 0
            best_tile = ""
            for name, mod in kernels.items():
                bs = tile_sizes[name]
                if N % bs != 0:
                    continue

                C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')
                tf, err, status = bench_kernel(mod, A, B, C)
                ratio = tf / hbl * 100 if hbl > 0 else 0

                writer.writerow([N, name, f'{tf:.1f}', f'{err:.4f}', status, f'{hbl:.1f}', f'{ratio:.1f}'])

                if tf > best_tf and status == "PASS":
                    best_tf = tf
                    best_tile = name

            ratio = best_tf / hbl * 100 if hbl > 0 else 0
            marker = ">>>" if ratio > 100 else "   "
            print(f'  [{idx+1}/{total}] {N:>5}: best={best_tile.replace("hk_",""):>12} {best_tf:>6.0f} TF vs hBL {hbl:>6.0f} TF = {ratio:>5.1f}% {marker}')

            # Flush periodically
            if idx % 10 == 0:
                csvfile.flush()

    print(f"\nResults saved to {args.output}")
    print(f"Total: {len(sizes)} sizes tested")

if __name__ == '__main__':
    main()
