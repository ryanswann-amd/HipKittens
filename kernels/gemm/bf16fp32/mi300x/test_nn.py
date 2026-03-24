"""Test NN GEMM kernel: C = A @ B (no transpose)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

def test_nn(mod_name, bs, sizes=None):
    mod = __import__(mod_name)
    if sizes is None:
        sizes = [bs, bs*2, bs*3, bs*4, 1024, 2048, 4096]
    sizes = [s for s in sizes if s % bs == 0]

    print(f"\n=== {mod_name} (BS={bs}) ===")
    all_pass = True
    for N in sizes:
        M, K = N, N
        torch.manual_seed(42)
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0
        B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda') / 10.0  # B is KxN for NN
        C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

        # Run kernel: C = A @ B
        mod.dispatch(A, B, C)
        torch.cuda.synchronize()

        # Reference: C_ref = A @ B
        C_ref = torch.matmul(A, B).to(torch.bfloat16)

        err = (C.float() - C_ref.float()).abs().max().item()
        rel = err / (C_ref.float().abs().max().item() + 1e-8)
        status = "PASS" if err < 0.5 else "FAIL"
        if status == "FAIL":
            all_pass = False

        # Benchmark
        for _ in range(3):
            mod.dispatch(A, B, C)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            mod.dispatch(A, B, C)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 10
        tflops = 2.0 * M * N * K / (ms * 1e9)

        print(f"  {M:>5}x{N:>5}x{K:>5}: err={err:.4f} rel={rel:.4f} {status:>4} | {tflops:>6.1f} TF  {ms:.3f} ms")

    return all_pass

if __name__ == '__main__':
    passed = True
    for mod_name, bs in [('hk_nn_256x256x64', 256)]:
        try:
            if not test_nn(mod_name, bs):
                passed = False
        except ImportError as e:
            print(f"  {mod_name}: not found ({e})")

    # Also test non-square shapes
    try:
        mod = __import__('hk_nn_256x256x64')
        print("\n=== Non-square shapes (256x256x64) ===")
        for M, N, K in [(256, 512, 256), (512, 256, 512), (1024, 2048, 512), (2048, 1024, 1024)]:
            A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0
            B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda') / 10.0
            C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            mod.dispatch(A, B, C)
            torch.cuda.synchronize()
            C_ref = torch.matmul(A, B).to(torch.bfloat16)
            err = (C.float() - C_ref.float()).abs().max().item()
            status = "PASS" if err < 0.5 else "FAIL"
            if status == "FAIL":
                passed = False
            print(f"  {M:>5}x{N:>5}x{K:>5}: err={err:.4f} {status}")
    except ImportError:
        pass

    print(f"\n{'ALL PASSED' if passed else 'SOME FAILED'}")
