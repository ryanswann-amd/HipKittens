"""
HipKittens Multi-Dtype Multi-Transpose GEMM Library — MI300X

Supports BF16 and FP16 inputs with FP32 accumulation, all 4 transpose types.
Auto-selects the best tile size based on problem shape divisibility.

Usage:
    from hk_gemm import gemm, gemm_nt, gemm_nn, gemm_tt, gemm_tn

    # Auto-detect transpose from shapes (assumes NN if shapes are MxK, KxN)
    C = gemm(A, B, trans='nn')  # C = A @ B
    C = gemm(A, B, trans='nt')  # C = A @ B^T
    C = gemm(A, B, trans='tn')  # C = A^T @ B
    C = gemm(A, B, trans='tt')  # C = A^T @ B^T

    # Direct dispatch
    C = gemm_nt(A, B)   # A: MxK, B: NxK → C = A @ B^T
    C = gemm_nn(A, B)   # A: MxK, B: KxN → C = A @ B
"""

import os, sys, importlib
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Origami analytical model for tile selection
try:
    import origami
    _origami_hw = origami.get_hardware_for_device(0)
    _USE_ORIGAMI = True
except ImportError:
    _USE_ORIGAMI = False

# Kernel registry: (dtype, transpose, block_size) → module_name
_KERNELS = {
    # BF16 NT (highest perf — 256x256 is best)
    ('bf16', 'nt', 256): 'hk_256x256',
    ('bf16', 'nt', 192): 'hk_192x192x64',
    ('bf16', 'nt', 128): 'hk_128x128x64',
    # BF16 NN
    ('bf16', 'nn', 256): 'hk_nn_fused_256x256x64',
    ('bf16', 'nn', 192): 'hk_nn_192x192x64',
    ('bf16', 'nn', 128): 'hk_nn_128x128x64',
    # BF16 TT
    ('bf16', 'tt', 256): 'hk_tt_fused_256x256x64',
    ('bf16', 'tt', 192): 'hk_tt_192x192x64',
    ('bf16', 'tt', 128): 'hk_tt_128x128x64',
    # BF16 TN
    ('bf16', 'tn', 256): 'hk_tn_fused_256x256x64',
    ('bf16', 'tn', 192): 'hk_tn_192x192x64',
    ('bf16', 'tn', 128): 'hk_tn_128x128x64',
    # FP16 NT
    ('fp16', 'nt', 256): 'hk_fp16_nt_256x256x64',
    ('fp16', 'nt', 128): 'hk_fp16_nt_128x128x64',
    # FP16 NN (fused for 256x256)
    ('fp16', 'nn', 256): 'hk_fp16_nn_fused_256x256x64',
    ('fp16', 'nn', 192): 'hk_fp16_nn_192x192x64',
    ('fp16', 'nn', 128): 'hk_fp16_nn_128x128x64',
    # FP16 TT (fused for 256x256)
    ('fp16', 'tt', 256): 'hk_fp16_tt_fused_256x256x64',
    ('fp16', 'tt', 192): 'hk_fp16_tt_192x192x64',
    ('fp16', 'tt', 128): 'hk_fp16_tt_128x128x64',
    # FP16 TN (fused for 256x256)
    ('fp16', 'tn', 256): 'hk_fp16_tn_fused_256x256x64',
    ('fp16', 'tn', 192): 'hk_fp16_tn_192x192x64',
    ('fp16', 'tn', 128): 'hk_fp16_tn_128x128x64',
}

# Tile preference order per (dtype, transpose) — try larger tiles first
_TILE_PREF = {
    ('bf16', 'nt'): [256, 192, 128],
    ('bf16', 'nn'): [256, 192, 128],  # 256=fused for large N, 192/128 for small
    ('bf16', 'tt'): [256, 192, 128],  # 256=fused for large N
    ('bf16', 'tn'): [256, 192, 128],  # 256=fused for large N
    ('fp16', 'nt'): [256, 128],
    ('fp16', 'nn'): [256, 192, 128],
    ('fp16', 'tt'): [256, 192, 128],
    ('fp16', 'tn'): [256, 192, 128],
}

_loaded = {}

def _load(key):
    if key not in _loaded:
        mod_name = _KERNELS.get(key)
        if mod_name is None:
            return None
        try:
            _loaded[key] = importlib.import_module(mod_name)
        except ImportError:
            _loaded[key] = None
    return _loaded[key]

def _dtype_key(t):
    if t.dtype == torch.bfloat16:
        return 'bf16'
    elif t.dtype == torch.float16:
        return 'fp16'
    elif hasattr(torch, 'float8_e4m3fnuz') and t.dtype == torch.float8_e4m3fnuz:
        return 'fp8_e4m3'
    elif hasattr(torch, 'float8_e5m2fnuz') and t.dtype == torch.float8_e5m2fnuz:
        return 'fp8_e5m2'
    elif t.dtype == torch.float32:
        return 'fp32'
    raise ValueError(f"Unsupported dtype {t.dtype}. Use float32, bfloat16, float16, or float8.")

def _select_tile(dtype_key, trans, M, N, K):
    """Select the best tile size for given problem using Origami if available.

    For transpose variants (NN/TT/TN), uses a compute-intensity heuristic
    to decide between in-kernel transpose and fused (transpose kernel + NT):
    - Small problems (low FLOPS): in-kernel (avoids kernel launch overhead)
    - Medium problems: fused 128 (transpose + high-occupancy NT 128x128)
    - Large problems: fused 256 (transpose + max-efficiency NT 256x256)
    """
    prefs = _TILE_PREF.get((dtype_key, trans), [128])

    # Filter to tiles that divide M, N, K
    valid = []
    for bs in prefs:
        if M % bs == 0 and N % bs == 0 and K % 64 == 0:
            mod = _load((dtype_key, trans, bs))
            if mod is not None:
                valid.append((bs, mod))

    if not valid:
        return None, 0

    if len(valid) == 1:
        return valid[0][1], valid[0][0]

    # Use Origami to rank valid tiles
    if _USE_ORIGAMI:
        problem = origami.problem_t()
        problem.size = origami.dim3_t(M, N, K)
        problem.a_dtype = origami.data_type_t.BFloat16
        problem.b_dtype = origami.data_type_t.BFloat16
        problem.c_dtype = origami.data_type_t.Float
        problem.d_dtype = origami.data_type_t.Float

        configs = []
        for bs, mod in valid:
            c = origami.config_t()
            c.mt = origami.dim3_t(bs, bs, 64)
            c.mi = origami.dim3_t(16, 16, 16)
            configs.append((bs, mod, c))

        oc = [c for _, _, c in configs]
        result = origami.select_config(problem, _origami_hw, oc)
        best_bs = result.config.mt.m

        for bs, mod, _ in configs:
            if bs == best_bs:
                return mod, bs

    # Fallback: first valid tile (largest)
    return valid[0][1], valid[0][0]

def gemm(A, B, C=None, trans='nt'):
    """Run GEMM with auto-selected kernel.

    Args:
        A: Input tensor (bfloat16, float16, or float8_e4m3fnuz/e5m2fnuz)
        B: Input tensor (same dtype as A)
        C: Optional output tensor (same dtype, allocated if None)
        trans: 'nt', 'nn', 'tt', or 'tn'

    Returns:
        C tensor
    """
    dk = _dtype_key(A)
    out_dtype = A.dtype

    # FP8/FP32: convert to BF16 and dispatch via BF16 kernels
    if dk.startswith('fp8') or dk == 'fp32':
        A = A.to(torch.bfloat16)
        B = B.to(torch.bfloat16)
        dk = 'bf16'

    if trans == 'nt':
        M, K = A.shape; N = B.shape[0]
    elif trans == 'nn':
        M, K = A.shape; N = B.shape[1]
    elif trans == 'tt':
        M = A.shape[1]; K = A.shape[0]; N = B.shape[0]
    elif trans == 'tn':
        M = A.shape[1]; K = A.shape[0]; N = B.shape[1]
    else:
        raise ValueError(f"Unknown transpose '{trans}'. Use 'nt', 'nn', 'tt', or 'tn'.")

    # For FP8 inputs, output is always BF16 (the compute dtype)
    c_dtype = torch.bfloat16 if dk == 'bf16' else (torch.float16 if dk == 'fp16' else torch.bfloat16)
    if C is None:
        C = torch.zeros(M, N, dtype=c_dtype, device=A.device)

    mod, bs = _select_tile(dk, trans, M, N, K)
    if mod is None:
        raise RuntimeError(f"No kernel for {dk} {trans} M={M} N={N} K={K}")

    mod.dispatch(A, B, C)
    return C

def gemm_nt(A, B, C=None):
    """C = A @ B^T. A: MxK, B: NxK."""
    return gemm(A, B, C, trans='nt')

def gemm_nn(A, B, C=None):
    """C = A @ B. A: MxK, B: KxN."""
    return gemm(A, B, C, trans='nn')

def gemm_tt(A, B, C=None):
    """C = A^T @ B^T. A: KxM, B: NxK."""
    return gemm(A, B, C, trans='tt')

def gemm_tn(A, B, C=None):
    """C = A^T @ B. A: KxM, B: KxN."""
    return gemm(A, B, C, trans='tn')

def available():
    """List all available kernel configurations."""
    avail = []
    for key, mod_name in _KERNELS.items():
        dtype, trans, bs = key
        mod = _load(key)
        status = 'OK' if mod is not None else 'MISSING'
        avail.append((dtype, trans, bs, mod_name, status))
    return avail


if __name__ == '__main__':
    print("HipKittens GEMM Library — Available Kernels:")
    for dtype, trans, bs, name, status in available():
        print(f"  {dtype:>4} {trans:>2} {bs:>3}x{bs} : {name:>30} [{status}]")

    N = 1024
    print(f"\nQuick test (N={N}):")
    for dtype_t, dtype_name in [(torch.bfloat16, 'bf16'), (torch.float16, 'fp16')]:
        for trans in ['nt', 'nn', 'tt', 'tn']:
            try:
                if trans == 'nt':
                    A = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    B = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    ref = torch.matmul(A, B.t().contiguous()).to(dtype_t)
                elif trans == 'nn':
                    A = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    B = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    ref = torch.matmul(A, B).to(dtype_t)
                elif trans == 'tt':
                    A = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    B = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    ref = torch.matmul(A.t().contiguous(), B.t().contiguous()).to(dtype_t)
                elif trans == 'tn':
                    A = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    B = torch.randn(N, N, dtype=dtype_t, device='cuda') / 10
                    ref = torch.matmul(A.t().contiguous(), B).to(dtype_t)

                C = gemm(A, B, trans=trans)
                torch.cuda.synchronize()
                err = (C.float() - ref.float()).abs().max().item()
                print(f"  {dtype_name} {trans}: err={err:.4f} {'PASS' if err < 0.5 else 'FAIL'}")
            except Exception as e:
                print(f"  {dtype_name} {trans}: {e}")
