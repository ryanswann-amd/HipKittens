import torch
N = 4096
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
Bt = B.t().contiguous()
for _ in range(5):
    torch.matmul(A, Bt)
torch.cuda.synchronize()
torch.matmul(A, Bt)
torch.cuda.synchronize()
