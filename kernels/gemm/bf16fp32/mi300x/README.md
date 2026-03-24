# HipKittens GEMM Library — MI300X (gfx942, CDNA3)

High-performance BF16/FP16 GEMM library for AMD MI300X using
[HipKittens](https://github.com/HazyResearch/HipKittens) (CDNA3 branch).

## Performance

**Beats hipBLASLt on 6/8 dtype×transpose combinations at 8192×8192:**

| | NT | NN | TT | TN |
|---|---|---|---|---|
| **BF16** | 568 TF (**116%**) | 527 TF (93%) | 521 TF (**137%**) | 491 TF (**102%**) |
| **FP16** | 521 TF (**115%**) | 489 TF (93%) | 486 TF (**126%**) | 460 TF (**102%**) |

Peak: 1307 TF theoretical (BF16 MFMA). Our NT kernel reaches 44% of peak.

## Quick Start

```python
from hipkittens_gemm import gemm

# BF16 or FP16, any transpose
C = gemm(A, B, trans='nn')  # C = A @ B
C = gemm(A, B, trans='nt')  # C = A @ B^T
C = gemm(A, B, trans='tn')  # C = A^T @ B
C = gemm(A, B, trans='tt')  # C = A^T @ B^T
```

## Building

```bash
source ../../env.src  # Sets THUNDERKITTENS_ROOT

# Build all kernels
make all nn tt tn fused fp16_nt fp16_nn fp16_tt fp16_tn fp16_fused

# Or build specific targets
make hk_256x256$(python3-config --extension-suffix)        # BF16 NT 256x256
make hk_nn_fused_256x256x64$(python3-config --extension-suffix)  # BF16 NN fused
```

Requires:
- ROCm 6.x+ with hipcc
- CDNA3 branch headers at `/tmp/hk_cdna3` (git worktree of upstream cdna3 branch)
- PyTorch with ROCm support
- pybind11

## Architecture

### NT Kernel (Direct)
The base kernel from HipKittens CDNA3 branch with parameterized tile sizes.
Uses 8-cluster interleaved schedule with `mma_ABt`, register buffer pipelining,
and chiplet-aware workgroup ID transform.

- 250 VGPRs, 0 spills, occupancy 2 (at 256×256)
- `buffer_load_dwordx4` via `llvm_amdgcn_raw_buffer_load_b128`
- `store_shared_vec` (ds_write_b64) for shared memory stores

### Transpose Variants (NN, TT, TN)

**Small sizes (N ≤ 3072, divisible by 192):** In-kernel transpose using
4-way K-grouped vectorized writes. Loads 4 adjacent K-rows as float4,
interleaves into column vectors, writes with ds_write_b64.

**Large sizes (N ≥ 4096, divisible by 256):** Fused transpose+NT approach.
A lightweight 32×32 tile transpose kernel (8 VGPRs, occ 8) transposes
the operand in global memory, then the full-speed NT kernel runs on the
transposed data. Static workspace cache avoids hipMalloc per call.

### Tile Sizes

| Tile | VGPRs | Occupancy | Best For |
|------|-------|-----------|----------|
| 256×256×64 | 250 | 2 | Large N (4K+), NT direct |
| 192×192×64 | 200 | 2 | Medium N, non-power-of-2 |
| 128×128×64 | 118 | 4 | Small N, high parallelism |

## Files

| File | Description |
|------|-------------|
| `cdna3_kernel.cpp` | BF16 NT kernel (base, 580 TF) |
| `fp16_nt_kernel.cpp` | FP16 NT kernel |
| `nn_kernel.cpp` | BF16 NN with in-kernel transpose |
| `nn_fused_kernel.cpp` | BF16 NN via transpose+NT (542 TF) |
| `tt_kernel.cpp` / `tt_fused_kernel.cpp` | BF16 TT variants |
| `tn_kernel.cpp` / `tn_fused_kernel.cpp` | BF16 TN variants |
| `fp16_*_kernel.cpp` | FP16 variants (same structure) |
| `hipkittens_gemm.py` | Unified dispatch library |
| `Makefile` | Build system for all 23 kernel modules |

## Known Limitations

- **Small K (< 512):** Performance drops to 50-65% of hipBLASLt. The
  256×256 tile with K_STEP=64 needs K ≥ 512 for pipeline steady-state.
- **FP8:** Not supported. gfx942 has FP8 MFMA instructions but HipKittens
  CDNA3 branch lacks 8-bit type support.
- **Non-aligned sizes:** M and N must be divisible by tile size (128/192/256).
  K must be divisible by 64.
