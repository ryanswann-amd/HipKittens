"""Benchmark Triton GEMM at various tile sizes to find perf ceiling."""
import triton, triton.language as tl, torch

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K, sa0, sa1, sb0, sb1, sc0, sc1,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    nig = GM * npn; gid = pid // nig
    fm = gid * GM; gsm = min(npm - fm, GM)
    pm = fm + ((pid % nig) % gsm); pn = (pid % nig) // gsm
    om = pm*BM + tl.arange(0, BM); on = pn*BN + tl.arange(0, BN); ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None]*sa0 + ok[None, :]*sa1
    bp = b_ptr + ok[:, None]*sb0 + on[None, :]*sb1
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(ap); b = tl.load(bp); acc += tl.dot(a, b)
        ap += BK*sa1; bp += BK*sb0
    c = acc.to(tl.bfloat16)
    cp = c_ptr + om[:, None]*sc0 + on[None, :]*sc1
    tl.store(cp, c, mask=(om[:, None] < M) & (on[None, :] < N))

N = 4096
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')

configs = [(128,128,64,4), (128,128,32,4), (256,256,64,8), (128,256,64,8), (256,128,64,8)]
for bm, bn, bk, nw in configs:
    if N % bm != 0 or N % bn != 0:
        continue
    g = lambda m: (triton.cdiv(N, m['BM']) * triton.cdiv(N, m['BN']),)
    try:
        for _ in range(10):
            _mm[g](A, B, C, N, N, N, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                   C.stride(0), C.stride(1), BM=bm, BN=bn, BK=bk, GM=4, num_warps=nw)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        ts = []
        for _ in range(20):
            s.record()
            _mm[g](A, B, C, N, N, N, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                   C.stride(0), C.stride(1), BM=bm, BN=bn, BK=bk, GM=4, num_warps=nw)
            e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        avg = sum(ts)/len(ts)
        tf = 2*N**3 / (avg*1e9)
        print(f'Triton {bm}x{bn}x{bk} w={nw}: {tf:.1f} TF ({avg:.4f} ms)')
    except Exception as ex:
        print(f'Triton {bm}x{bn}x{bk} w={nw}: FAIL ({ex})')
