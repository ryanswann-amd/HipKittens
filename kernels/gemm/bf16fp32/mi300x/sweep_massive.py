"""Massive GEMM sweep: thousands of M×N×K shapes with validation.
Usage: HIP_VISIBLE_DEVICES=0 python3 sweep_massive.py [--count N] [--output FILE]
"""
import sys, os, csv, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=3000, help='Number of shapes to test')
    parser.add_argument('--output', default='sweep_massive.csv', help='Output CSV')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load kernels
    kernels = {}
    for name in ['hk_128x128x64', 'hk_192x192x64', 'hk_256x256x64']:
        try:
            kernels[name] = (__import__(name), int(name.split('_')[1].split('x')[0]))
        except ImportError:
            pass
    print(f"Kernels: {list(kernels.keys())}")

    # Generate ALL valid shapes, then sample
    all_shapes = set()

    # Multiples of 128
    for M in range(128, 8192+1, 128):
        for N in range(M, 8192+1, 128):  # N >= M to avoid duplicates
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
    print(f"Total possible shapes: {len(all_shapes)}")

    # Sample
    if args.count < len(all_shapes):
        shapes = random.sample(all_shapes, args.count)
        shapes.sort()
    else:
        shapes = all_shapes

    print(f"Testing {len(shapes)} shapes")

    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['M', 'N', 'K', 'tile', 'tflops', 'max_err', 'status',
                     'hipblaslt_tflops', 'ratio_pct', 'best_tile', 'best_tflops'])

        total_pass = 0
        total_fail = 0
        total_beat = 0
        t0 = time.time()

        for idx, (M, N, K) in enumerate(shapes):
            A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0
            B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0
            Bt = B.t().contiguous()

            # hipBLASLt
            try:
                for _ in range(2): torch.matmul(A, Bt)
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                ts = []
                for _ in range(3):
                    s.record(); torch.matmul(A, Bt); e.record()
                    torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
                hbl = 2*M*N*K / (sum(ts)/len(ts) * 1e9)
            except:
                hbl = 0

            best_tf, best_tile = 0, ''
            for name, (mod, bs) in kernels.items():
                if M % bs != 0 or N % bs != 0:
                    continue
                C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
                try:
                    mod.dispatch(A, B, C)
                    torch.cuda.synchronize()
                    C_ref = torch.matmul(A, Bt).to(torch.bfloat16)
                    err = (C.float() - C_ref.float()).abs().max().item()

                    for _ in range(2): mod.dispatch(A, B, C)
                    torch.cuda.synchronize()
                    ts2 = []
                    for _ in range(3):
                        s.record(); mod.dispatch(A, B, C); e.record()
                        torch.cuda.synchronize(); ts2.append(s.elapsed_time(e))
                    tf = 2*M*N*K / (sum(ts2)/len(ts2) * 1e9)

                    status = "PASS" if err < 0.1 else "FAIL"
                    ratio = tf/hbl*100 if hbl > 0 else 0
                    w.writerow([M, N, K, name, f'{tf:.1f}', f'{err:.4f}', status,
                               f'{hbl:.1f}', f'{ratio:.1f}', '', ''])

                    if status == "PASS" and tf > best_tf:
                        best_tf, best_tile = tf, name
                    if status == "PASS":
                        total_pass += 1
                    else:
                        total_fail += 1
                except:
                    total_fail += 1

            if best_tf > hbl and best_tf > 0:
                total_beat += 1

            if (idx+1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (idx+1) / elapsed
                eta = (len(shapes) - idx - 1) / rate / 60
                print(f'  [{idx+1}/{len(shapes)}] pass={total_pass} fail={total_fail} '
                      f'beat={total_beat} ({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}min)')
                f.flush()

    elapsed = time.time() - t0
    unique_shapes = len(shapes)
    print(f"\n=== COMPLETE ===")
    print(f"Shapes: {unique_shapes}, Points: {total_pass+total_fail}")
    print(f"Pass: {total_pass}, Fail: {total_fail}")
    print(f"Beat hipBLASLt: {total_beat}/{unique_shapes} ({total_beat/max(unique_shapes,1)*100:.0f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
