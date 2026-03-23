"""Validate Origami kernel selection accuracy against measured performance."""

import subprocess, sys, os, json

_DIR = os.path.dirname(os.path.abspath(__file__))

def bench_tile(tile_mod, size):
    """Benchmark a single tile in a subprocess (avoids multi-module crashes)."""
    result = subprocess.run(
        [sys.executable, os.path.join(_DIR, 'bench_one.py'), tile_mod, str(size)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, 'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', '0')}
    )
    if result.returncode != 0:
        return None
    line = result.stdout.strip()
    if not line or line.startswith('SKIP'):
        return None
    parts = line.split(',')
    return {'tile': parts[0], 'size': int(parts[1]), 'tflops': float(parts[2]),
            'ms': float(parts[3]), 'max_err': float(parts[4]), 'status': parts[5]}

def main():
    sys.path.insert(0, _DIR)
    from gemm_library import GEMMLibrary

    lib = GEMMLibrary()
    tiles = ['tile_64x64', 'tile_128x128', 'tile_256x256']
    tile_names = {'tile_64x64': '64x64', 'tile_128x128': '128x128', 'tile_256x256': '256x256'}
    sizes = [256, 512, 1024, 2048, 4096]

    print("HipKittens GEMM Library — Origami Selection Validation")
    print("=" * 80)

    # Benchmark all tiles at all sizes
    perf_data = {}  # (tile_name, size) -> tflops
    for tile in tiles:
        for size in sizes:
            result = bench_tile(tile, size)
            name = tile_names[tile]
            if result and result['status'] == 'PASS':
                perf_data[(name, size)] = result['tflops']
                print(f"  {name:>10} @ {size:>5}: {result['tflops']:>8.2f} TFLOPS ({result['status']})")
            else:
                print(f"  {name:>10} @ {size:>5}: SKIP/FAIL")

    # Compare Origami selection vs actual best
    print(f"\n{'Size':>6} | {'Origami Pick':>12} | {'Pick TFLOPS':>11} | {'Best Tile':>10} | {'Best TFLOPS':>11} | {'Efficiency':>10}")
    print("-" * 80)

    efficiencies = []
    for size in sizes:
        # Origami's pick
        origami_tile, origami_lat = lib.select_tile(size, size, size)

        # Find actual best from measurements
        best_tile = None
        best_tflops = 0
        for name in tile_names.values():
            tf = perf_data.get((name, size))
            if tf and tf > best_tflops:
                best_tflops = tf
                best_tile = name

        # Origami pick's measured performance
        origami_tflops = perf_data.get((origami_tile, size), 0)

        if best_tflops > 0 and origami_tflops > 0:
            efficiency = origami_tflops / best_tflops * 100
            efficiencies.append(efficiency)
        else:
            efficiency = 0

        print(f"{size:>6} | {origami_tile:>12} | {origami_tflops:>9.2f} TF | {best_tile:>10} | {best_tflops:>9.2f} TF | {efficiency:>8.1f}%")

    if efficiencies:
        avg_eff = sum(efficiencies) / len(efficiencies)
        print(f"\nAverage heuristic efficiency: {avg_eff:.1f}%")
        print(f"Target: >= 90%")
        print(f"Result: {'PASS' if avg_eff >= 90 else 'NEEDS IMPROVEMENT'}")

    # Summary table for ticket
    print("\n## Results Summary (for ticket)")
    print("| Size | Origami Pick | Pick TFLOPS | Best Tile | Best TFLOPS | Efficiency |")
    print("|------|-------------|-------------|-----------|-------------|------------|")
    for size in sizes:
        origami_tile, _ = lib.select_tile(size, size, size)
        origami_tflops = perf_data.get((origami_tile, size), 0)
        best_tile = max(((n, perf_data.get((n, size), 0)) for n in tile_names.values()), key=lambda x: x[1])
        eff = origami_tflops / best_tile[1] * 100 if best_tile[1] > 0 else 0
        print(f"| {size} | {origami_tile} | {origami_tflops:.1f} | {best_tile[0]} | {best_tile[1]:.1f} | {eff:.1f}% |")

if __name__ == '__main__':
    main()
