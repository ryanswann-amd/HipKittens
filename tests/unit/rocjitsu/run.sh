#!/usr/bin/env bash
# Build one tests/unit/<suite> configuration and run it on a rocjitsu-simulated GPU.
#
#   tests/unit/rocjitsu/run.sh <cdna4|cdna5> <outdir> <TEST_ define>...
#   e.g. tests/unit/rocjitsu/run.sh cdna4 results TEST_WARP_MEMORY_TILE
#
# Runs inside the Triton AMD CI image (TheRock ROCm + rocjitsu packaged as `mirage`), from the
# repo root. Fails on any failed test not listed in known_failures_<arch>.txt, on a build error,
# or if the binary dies before printing its summary. Known failures that now pass are reported
# so the list can be trimmed.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
suite=$1 out=$2; shift 2
case $suite in
  cdna4) arch=gfx950 ;;
  cdna5) arch=gfx1250 ;;
  *) echo "run.sh: unknown suite '$suite'" >&2; exit 2 ;;
esac
defines=$(printf -- '-D%s ' "$@")
mkdir -p "$out"
rocm-sdk init >/dev/null
export ROCM_PATH=$(rocm-sdk path --root)

echo "::group::build $suite ($defines)"
SECONDS=0
make -C "tests/unit/$suite" clean >/dev/null
if ! make -C "tests/unit/$suite" -j"$(nproc)" HIPCXX="$ROCM_PATH/bin/hipcc" ROCM_PATH="$ROCM_PATH" \
       TEST_DEFINES="$defines" >"$out/build.log" 2>&1; then
  grep -m20 -E "error" "$out/build.log"; echo "::endgroup::"
  echo "::error::build failed for $suite $defines"
  exit 1
fi
echo "build ${SECONDS}s"; echo "::endgroup::"

SECONDS=0
(cd "tests/unit/$suite" && "$here/rjsim" "$arch" -- ./unit_tests) 2>&1 \
  | grep --line-buffered -v 'amdgpu.ids' | tee "$out/run.log" | grep --line-buffered -E 'Starting|FAILED'
echo "run ${SECONDS}s"

if ! grep -qE '^[0-9]+ tests failed' "$out/run.log"; then
  echo "::error::unit_tests exited before printing its summary (crash, hang or emulator fault)"
  tail -20 "$out/run.log"
  exit 1
fi
tail -3 "$out/run.log"

# Failed test labels: the summary list (between "Failed tests:" and the next blank line), plus
# every per-test ALERT line, in case a test validates without recording its result.
{ sed -n '/^Failed tests:/,/^$/p' "$out/run.log" | sed '1d;/^$/d;/ALL TESTS PASSED/d'
  sed -n 's/.*ALERT! FAILED test `\(.*\)` -----.*/\1/p' "$out/run.log"; } | sort -u >"$out/failed.txt"
known="$here/known_failures_$arch.txt"
grep -vE '^\s*(#|$)' "$known" 2>/dev/null | sort -u >"$out/known.txt"
comm -23 "$out/failed.txt" "$out/known.txt" >"$out/new_failures.txt"
# A known failure counts as fixed only if this run executed it and it passed.
while read -r t; do
  grep -qF "test \`$t\`  -- PASSED" "$out/run.log" && echo "$t"
done <"$out/known.txt" >"$out/fixed.txt"

if [ -s "$out/fixed.txt" ]; then
  echo "::notice::$(wc -l <"$out/fixed.txt") known failure(s) now pass; remove them from $(basename "$known"):"
  cat "$out/fixed.txt"
fi
if [ -s "$out/new_failures.txt" ]; then
  echo "::error::$(wc -l <"$out/new_failures.txt") unexpected failure(s):"
  cat "$out/new_failures.txt"
  exit 1
fi
echo "OK: $(wc -l <"$out/failed.txt") failure(s), all known"
