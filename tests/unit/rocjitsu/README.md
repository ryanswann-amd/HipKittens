# Unit tests on rocjitsu (no GPU)

[rocjitsu](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu) runs unmodified
HIP binaries against a simulated AMD GPU on the CPU, which lets `.github/workflows/rocjitsu.yml`
run the unit tests on free GitHub-hosted runners. It is functional only: it checks the bytes,
never timing, and real hardware stays the authority on correctness.

| suite | simulated GPU |
|---|---|
| `tests/unit/cdna4` | gfx950 (MI355X) |
| `tests/unit/cdna5` | gfx1250 (MI455X) |

The toolchain and emulator come from Triton's public AMD CI image (TheRock ROCm, with rocjitsu
packaged as `mirage`), so nothing is built from source.

## Run it locally

```bash
IMAGE=tritonci/triton-amd-ci:rocm10.2.0a20261012-pytorch2.16.0a0-gfx942-gfx950-gfx1250-r1
docker run --rm --init --cpus 4 --memory 14g --user $(id -u):$(id -g) -e HOME=/tmp/home \
  -v $PWD:/work -w /work $IMAGE \
  tests/unit/rocjitsu/run.sh cdna4 results TEST_WARP_REGISTER_TILE_MAPS
```

`run.sh <suite> <outdir> <TEST_ define>...` builds that selection, runs it under `rjsim`, and
writes `build.log`, `run.log` and `failed.txt` to `<outdir>`. Use any of the `TEST_` macros from
`tests/unit/common/testing_commons/testing_flags.cuh`. Each CI shard builds its own binary.

## Known failures

`known_failures_gfx950.txt` lists tests that fail under the emulator but have not been confirmed
on hardware. CI fails only on a failure that is not in that list, and prints a notice when a listed
test starts passing so it can be removed. If a new failure is an emulator problem rather than a
HipKittens bug, check it on a real GPU before adding it to the list.
