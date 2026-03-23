"""Test HipKittens BF16 GEMM library — correctness and performance on MI300X."""

import torch
import sys
import os
import time
import importlib

# Add this directory to path so we can import the tile modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Available tile configurations
TILES = {
    '64x64':   'tile_64x64',
    '128x128': 'tile_128x128',
    '128x256': 'tile_128x256',
    '256x128': 'tile_256x128',
    '256x256': 'tile_256x256',
}

def load_tiles():
    """Load all compiled tile kernel modules."""
    modules = {}
    for name, mod_name in TILES.items():
        try:
            modules[name] = importlib.import_module(mod_name)
            print(f"  Loaded {name}: BLOCK_M={modules[name].BLOCK_M}, BLOCK_N={modules[name].BLOCK_N}, K_STEP={modules[name].K_STEP}")
        except ImportError as e:
            print(f"  SKIP {name}: {e}")
    return modules

def test_correctness(modules, sizes=None):
    """Test each tile against PyTorch reference (C = A @ B^T)."""
    if sizes is None:
        sizes = [256, 512, 1024, 2048, 4096]

    print("\n=== Correctness Tests ===")
    results = {}

    for name, mod in modules.items():
        bm, bn = mod.BLOCK_M, mod.BLOCK_N
        print(f"\nTile {name} (BLOCK_M={bm}, BLOCK_N={bn}):")

        for N in sizes:
            # Skip if size not divisible by tile
            if N % bm != 0 or N % bn != 0:
                print(f"  {N}x{N}: SKIP (not divisible)")
                continue

            torch.manual_seed(42)
            A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
            B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
            C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')

            # Reference: C_ref = A @ B^T (B is passed as NxK to kernel)
            C_ref = torch.matmul(A, B.t()).to(torch.bfloat16)

            # Kernel
            mod.dispatch(A, B, C)
            torch.cuda.synchronize()

            # Compare
            diff = (C.float() - C_ref.float()).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            rel_err = (diff / (C_ref.float().abs() + 1e-6)).mean().item()

            status = "PASS" if max_err < 5.0 else "FAIL"
            print(f"  {N}x{N}: {status} max_err={max_err:.4f} mean_err={mean_err:.6f} rel_err={rel_err:.6f}")
            results[(name, N)] = (status, max_err, mean_err)

    return results

def benchmark(modules, sizes=None, num_warmup=10, num_iters=20):
    """Benchmark each tile across problem sizes."""
    if sizes is None:
        sizes = [512, 1024, 2048, 4096, 8192]

    print("\n=== Performance Benchmarks ===")
    print(f"{'Size':>6} | ", end="")
    for name in modules:
        print(f"{name:>10} | ", end="")
    print(f"{'PyTorch':>10}")
    print("-" * (10 + 14 * (len(modules) + 1)))

    all_results = {}

    for N in sizes:
        torch.manual_seed(42)
        A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
        B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
        flops = 2.0 * N * N * N

        print(f"{N:>6} | ", end="")

        for name, mod in modules.items():
            bm, bn = mod.BLOCK_M, mod.BLOCK_N
            if N % bm != 0 or N % bn != 0:
                print(f"{'N/A':>10} | ", end="")
                continue

            C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')

            # Warmup
            for _ in range(num_warmup):
                mod.dispatch(A, B, C)
            torch.cuda.synchronize()

            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(num_iters):
                start.record()
                mod.dispatch(A, B, C)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            avg_ms = sum(times) / len(times)
            tflops = flops / (avg_ms * 1e9)
            print(f"{tflops:>8.1f} TF | ", end="")
            all_results[(name, N)] = tflops

        # PyTorch reference
        Bt = B.t().contiguous()
        for _ in range(num_warmup):
            torch.matmul(A, Bt)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(num_iters):
            start.record()
            torch.matmul(A, Bt)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg_ms = sum(times) / len(times)
        tflops = flops / (avg_ms * 1e9)
        print(f"{tflops:>8.1f} TF")
        all_results[('pytorch', N)] = tflops

    return all_results

if __name__ == '__main__':
    print("HipKittens BF16 GEMM Library — MI300X (gfx942)")
    print("=" * 60)

    print("\nLoading tile kernels:")
    modules = load_tiles()

    if not modules:
        print("No tile kernels found! Run 'make all' first.")
        sys.exit(1)

    # Correctness
    correctness = test_correctness(modules)

    # Performance
    perf = benchmark(modules)
