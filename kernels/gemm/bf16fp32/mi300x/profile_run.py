import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import tile_256x256 as mod

N = 4096
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0
C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')

for _ in range(5):
    mod.dispatch(A, B, C)
torch.cuda.synchronize()
mod.dispatch(A, B, C)
torch.cuda.synchronize()
