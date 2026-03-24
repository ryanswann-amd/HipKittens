"""Massive GEMM sweep with cache flushing and proper statistics.
Each shape: flush L2 cache → warmup → multiple timed iterations with per-iter flush.
Usage: HIP_VISIBLE_DEVICES=0 python3 sweep_massive.py [--count N] [--output FILE]
"""
import sys, os, csv, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

# L2 cache flush: write to a large buffer to evict all cache lines
# MI300X: 32MB L2 per XCD × 8 XCDs = 256MB total L2
FLUSH_SIZE = 512 * 1024 * 1024  # 512MB — 2x L2 to be safe
_flush_buf = None

def get_flush_buf():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.zeros(FLUSH_SIZE // 4, dtype=torch.float32, device='cuda')
    return _flush_buf

def flush_l2_cache():
    """Flush L2 cache by writing to a large buffer."""
    buf = get_flush_buf()
    buf.fill_(1.0)
    torch.cuda.synchronize()

def bench_kernel(mod, A, B, C, warmup=5, iters=10):
    """Benchmark with L2 flush between iterations."""
    M, K = A.shape
    N = B.shape[0]

    try:
        # Correctness check (single run, no flush needed)
        mod.dispatch(A, B, C)
        torch.cuda.synchronize()
        C_ref = torch.matmul(A, B.t().contiguous()).to(torch.bfloat16)
        max_err = (C.float() - C_ref.float()).abs().max().item()

        # Warmup (with flush before first timed run)
        for _ in range(warmup):
            mod.dispatch(A, B, C)
        torch.cuda.synchronize()

        # Timed runs with L2 flush between each
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            flush_l2_cache()
            start.record()
            mod.dispatch(A, B, C)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg_ms = sum(times) / len(times)
        median_ms = sorted(times)[len(times) // 2]
        min_ms = min(times)
        std_ms = (sum((t - avg_ms)**2 for t in times) / len(times)) ** 0.5
        tflops = 2.0 * M * N * K / (median_ms * 1e9)  # use median for stability

        return {
            'tflops': tflops,
            'median_ms': median_ms,
            'avg_ms': avg_ms,
            'min_ms': min_ms,
            'std_ms': std_ms,
            'max_err': max_err,
            'status': "PASS" if max_err < 0.1 else "FAIL",
            'iters': iters,
        }
    except Exception as e:
        return {'tflops': 0, 'max_err': -1, 'status': f"ERR:{e}",
                'median_ms': 0, 'avg_ms': 0, 'min_ms': 0, 'std_ms': 0, 'iters': 0}

def bench_hipblaslt(A, Bt, warmup=5, iters=10):
    """Benchmark hipBLASLt with L2 flush between iterations."""
    M = A.shape[0]
    N = Bt.shape[1]
    K = A.shape[1]

    for _ in range(warmup):
        torch.matmul(A, Bt)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        flush_l2_cache()
        start.record()
        torch.matmul(A, Bt)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_ms = sorted(times)[len(times) // 2]
    return 2.0 * M * N * K / (median_ms * 1e9)

def generate_shapes(count, seed=42):
    """Generate random M×N×K shapes from all valid combinations."""
    rng = random.Random(seed)
    all_shapes = set()

    # Multiples of 128
    for M in range(128, 8192+1, 128):
        for N in range(M, 8192+1, 128):
            for K in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                all_shapes.add((M, N, K))

    # Multiples of 192
    for M in range(192, 6144+1, 192):
        for N in range(M, 6144+1, 192):
            for K in [64, 192, 384, 768, 1536, 3072, 6144]:
                if K % 64 == 0:
                    all_shapes.add((M, N, K))

    # Multiples of 256
    for M in range(256, 8192+1, 256):
        for N in range(M, 8192+1, 256):
            for K in [64, 256, 512, 1024, 2048, 4096, 8192]:
                all_shapes.add((M, N, K))

    all_shapes = sorted(all_shapes)
    if count < len(all_shapes):
        return rng.sample(all_shapes, count)
    return all_shapes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=3000)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--output', default='sweep_massive.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    kernels = {}
    for name in ['hk_128x128x64', 'hk_192x192x64', 'hk_256x256x64']:
        try:
            kernels[name] = (__import__(name), int(name.split('_')[1].split('x')[0]))
        except ImportError:
            pass
    print(f"Kernels: {list(kernels.keys())}")

    shapes = sorted(generate_shapes(args.count, args.seed))
    print(f"Testing {len(shapes)} shapes, {args.warmup} warmup, {args.iters} iters, L2 flush between each")

    # Pre-allocate flush buffer
    flush_l2_cache()
    print("L2 flush buffer allocated (512MB)")

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['M', 'N', 'K', 'tile', 'tflops', 'median_ms', 'avg_ms', 'min_ms',
                     'std_ms', 'max_err', 'status', 'iters',
                     'hipblaslt_tflops', 'ratio_pct'])

        total_pass = total_fail = total_beat = 0
        t0 = time.time()

        for idx, (M, N, K) in enumerate(shapes):
            A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0
            B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0
            Bt = B.t().contiguous()

            hbl = bench_hipblaslt(A, Bt, warmup=args.warmup, iters=args.iters)

            best_tf, best_tile = 0, ''
            for name, (mod, bs) in kernels.items():
                if M % bs != 0 or N % bs != 0:
                    continue
                C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
                r = bench_kernel(mod, A, B, C, warmup=args.warmup, iters=args.iters)

                ratio = r['tflops'] / hbl * 100 if hbl > 0 else 0
                w.writerow([M, N, K, name, f"{r['tflops']:.1f}", f"{r['median_ms']:.4f}",
                           f"{r['avg_ms']:.4f}", f"{r['min_ms']:.4f}", f"{r['std_ms']:.4f}",
                           f"{r['max_err']:.4f}", r['status'], r['iters'],
                           f'{hbl:.1f}', f'{ratio:.1f}'])

                if r['status'] == 'PASS':
                    total_pass += 1
                    if r['tflops'] > best_tf:
                        best_tf, best_tile = r['tflops'], name
                else:
                    total_fail += 1

            if best_tf > hbl > 0:
                total_beat += 1

            if (idx+1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (idx+1) / elapsed
                eta = (len(shapes) - idx - 1) / rate / 60
                print(f'  [{idx+1}/{len(shapes)}] pass={total_pass} fail={total_fail} '
                      f'beat={total_beat} ({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}min)')
                f.flush()

    elapsed = time.time() - t0
    print(f"\n=== COMPLETE ===")
    print(f"Shapes: {len(shapes)}, Points: {total_pass+total_fail}")
    print(f"Pass: {total_pass}, Fail: {total_fail}")
    print(f"Beat hipBLASLt: {total_beat}/{len(shapes)} ({total_beat/max(len(shapes),1)*100:.0f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Method: {args.iters} iters per point, L2 flush (512MB) between each, median TFLOPS")

if __name__ == '__main__':
    main()
