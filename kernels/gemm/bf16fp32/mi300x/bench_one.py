"""Benchmark a single tile kernel. Run as: python bench_one.py <module_name> <size>"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mod_name = sys.argv[1]
N = int(sys.argv[2])
mod = __import__(mod_name)

bm, bn, bk = mod.BLOCK_M, mod.BLOCK_N, mod.K_STEP
if N % bm != 0 or N % bn != 0 or N % bk != 0:
    print(f"SKIP {N} not divisible by {bm}x{bn}x{bk}")
    sys.exit(0)

torch.manual_seed(42)
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')
flops = 2.0 * N * N * N

# Correctness check
C_ref = torch.matmul(A, B.t()).to(torch.bfloat16)
mod.dispatch(A, B, C)
torch.cuda.synchronize()
diff = (C.float() - C_ref.float()).abs()
max_err = diff.max().item()

# Benchmark
warmup, iters = 10, 20
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
tf = flops / (avg_ms * 1e9)
status = "PASS" if max_err < 5.0 else "FAIL"
print(f"{mod_name},{N},{tf:.2f},{avg_ms:.4f},{max_err:.4f},{status}")
