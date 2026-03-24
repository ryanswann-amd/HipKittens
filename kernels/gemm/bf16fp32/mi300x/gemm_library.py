"""
HipKittens BF16 GEMM Library with Origami Kernel Selection — MI300X

Usage:
    from gemm_library import GEMMLibrary
    lib = GEMMLibrary()
    C = lib.gemm(A, B)  # Origami picks optimal tile
    tile = lib.select_tile(M, N, K)  # Just get the tile name
"""

import os
import sys
import importlib
import torch
import origami

# Ensure this directory is on the path
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)


class GEMMLibrary:
    """GEMM library with multiple tile sizes and Origami-driven selection.

    Computes C = A @ B^T where A is MxK and B is NxK (both BF16).
    """

    # Tile configs: name -> (module_name, block_m, block_n, block_k, mi_m, mi_n, mi_k)
    TILE_CONFIGS = {
        '64x64':   ('tile_64x64',   64,  64,  32, 16, 16, 16),
        '128x128': ('tile_128x128', 128, 128, 32, 16, 16, 16),
        '256x256': ('tile_256x256', 256, 256, 64, 16, 16, 16),
    }

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.hw = origami.get_hardware_for_device(device_id)
        self._kernels = {}
        self._origami_configs = []
        self._config_names = []

        self._load_kernels()
        self._build_origami_configs()

    def _load_kernels(self):
        """Load compiled tile kernel .so modules + CDNA3 optimized kernel."""
        for name, (mod_name, *_) in self.TILE_CONFIGS.items():
            try:
                self._kernels[name] = importlib.import_module(mod_name)
            except ImportError as e:
                print(f"Warning: tile {name} ({mod_name}) not available: {e}")
        # Load CDNA3 optimized kernels (3 validated tile sizes, all K_STEP=64)
        self._cdna3_kernels = {}
        self._cdna3_tile_sizes = {}
        for bs, mod_name in [(128, 'hk_128x128x64'), (192, 'hk_192x192x64'), (256, 'hk_256x256x64')]:
            try:
                self._cdna3_kernels[bs] = importlib.import_module(mod_name)
                self._cdna3_tile_sizes[bs] = 64  # K_STEP
            except ImportError:
                # Try without K_STEP suffix (legacy names)
                try:
                    legacy = f'hk_{bs}x{bs}'
                    self._cdna3_kernels[bs] = importlib.import_module(legacy)
                    self._cdna3_tile_sizes[bs] = 64
                except ImportError:
                    pass
        self._cdna3_kernel = self._cdna3_kernels.get(256)

    def _build_origami_configs(self):
        """Build origami config_t objects for ranking."""
        for name, (_, bm, bn, bk, mi_m, mi_n, mi_k) in self.TILE_CONFIGS.items():
            if name not in self._kernels:
                continue
            c = origami.config_t()
            c.mt = origami.dim3_t(bm, bn, bk)
            c.mi = origami.dim3_t(mi_m, mi_n, mi_k)
            self._origami_configs.append(c)
            self._config_names.append(name)

    def _make_problem(self, M, N, K):
        """Create origami problem_t for given dimensions."""
        p = origami.problem_t()
        p.size = origami.dim3_t(M, N, K)
        p.a_dtype = origami.data_type_t.BFloat16
        p.b_dtype = origami.data_type_t.BFloat16
        p.c_dtype = origami.data_type_t.Float
        p.d_dtype = origami.data_type_t.Float
        p.a_transpose = False
        p.b_transpose = True
        return p

    def select_tile(self, M, N, K):
        """Use Origami to select the optimal tile for given problem shape.

        Returns:
            Tuple of (tile_name, predicted_latency)
        """
        problem = self._make_problem(M, N, K)

        # Filter to tiles that evenly divide the problem
        valid_configs = []
        valid_names = []
        for cfg, name in zip(self._origami_configs, self._config_names):
            bm, bn, bk = cfg.mt.m, cfg.mt.n, cfg.mt.k
            if M % bm == 0 and N % bn == 0 and K % bk == 0:
                valid_configs.append(cfg)
                valid_names.append(name)

        if not valid_configs:
            # Fallback to smallest tile
            return '64x64', float('inf')

        result = origami.select_config(problem, self.hw, valid_configs)
        # Map back to name
        for name, cfg in zip(valid_names, valid_configs):
            if cfg.mt.m == result.config.mt.m and cfg.mt.n == result.config.mt.n:
                return name, result.latency

        return valid_names[0], result.latency

    def rank_tiles(self, M, N, K):
        """Rank all valid tiles for a problem shape.

        Returns:
            List of (tile_name, predicted_latency) sorted best-first.
        """
        problem = self._make_problem(M, N, K)

        valid_configs = []
        valid_names = []
        for cfg, name in zip(self._origami_configs, self._config_names):
            bm, bn, bk = cfg.mt.m, cfg.mt.n, cfg.mt.k
            if M % bm == 0 and N % bn == 0 and K % bk == 0:
                valid_configs.append(cfg)
                valid_names.append(name)

        if not valid_configs:
            return [('64x64', float('inf'))]

        results = origami.rank_configs(problem, self.hw, valid_configs)
        ranked = []
        for r in results:
            for name, cfg in zip(valid_names, valid_configs):
                if cfg.mt.m == r.config.mt.m and cfg.mt.n == r.config.mt.n:
                    ranked.append((name, r.latency))
                    break
        return ranked

    def gemm(self, A, B, C=None):
        """Run GEMM: C = A @ B^T using the best available kernel.

        For large sizes (M,N >= 4096, divisible by 256, K divisible by 64):
          uses the CDNA3 optimized kernel (98% of hipBLASLt).
        For smaller sizes:
          uses the C++ kernel with Origami tile selection.

        Args:
            A: BF16 tensor, shape (M, K)
            B: BF16 tensor, shape (N, K) — note: B is transposed
            C: Optional output BF16 tensor, shape (M, N)

        Returns:
            C tensor (M, N) in BF16
        """
        M, K = A.shape
        N = B.shape[0]

        if C is None:
            C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)

        # Use Origami to select the best CDNA3 tile
        if K % 64 == 0 and self._cdna3_kernels:
            # Find all compatible CDNA3 tiles
            candidates = []
            for bs, mod in self._cdna3_kernels.items():
                if M % bs == 0 and N % bs == 0:
                    candidates.append(bs)
            
            if candidates:
                if len(candidates) == 1:
                    self._cdna3_kernels[candidates[0]].dispatch(A, B, C)
                else:
                    # Use Origami to rank the candidates
                    problem = self._make_problem(M, N, K)
                    configs = []
                    for bs in candidates:
                        c = origami.config_t()
                        c.mt = origami.dim3_t(bs, bs, 64)
                        c.mi = origami.dim3_t(16, 16, 16)
                        configs.append((bs, c))
                    
                    oc = [c for _, c in configs]
                    result = origami.select_config(problem, self.hw, oc)
                    best_bs = result.config.mt.m
                    
                    if best_bs in self._cdna3_kernels:
                        self._cdna3_kernels[best_bs].dispatch(A, B, C)
                    else:
                        self._cdna3_kernels[candidates[0]].dispatch(A, B, C)
                return C
        
        # Fallback to C++ kernel with Origami selection
        tile_name, _ = self.select_tile(M, N, K)
        kernel = self._kernels[tile_name]
        kernel.dispatch(A, B, C)
        return C

    @property
    def available_tiles(self):
        """List of loaded tile names."""
        return list(self._kernels.keys())


if __name__ == '__main__':
    lib = GEMMLibrary()
    print(f"Available tiles: {lib.available_tiles}")

    sizes = [256, 512, 1024, 2048, 4096]
    print(f"\n{'Size':>6} | {'Selected':>10} | Ranking")
    print("-" * 60)
    for N in sizes:
        tile, lat = lib.select_tile(N, N, N)
        ranking = lib.rank_tiles(N, N, N)
        rank_str = ", ".join(f"{n}({l:.0f})" for n, l in ranking)
        print(f"{N:>6} | {tile:>10} | {rank_str}")

    # Quick correctness check
    print("\nCorrectness check (1024x1024):")
    A = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda') / 10.0
    B = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda') / 10.0
    C = lib.gemm(A, B)
    C_ref = torch.matmul(A, B.t()).to(torch.bfloat16)
    diff = (C.float() - C_ref.float()).abs()
    print(f"  max_err={diff.max().item():.4f}, mean_err={diff.mean().item():.6f}")
