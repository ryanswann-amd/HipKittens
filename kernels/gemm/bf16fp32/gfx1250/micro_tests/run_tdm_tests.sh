#!/usr/bin/env bash
# Compile + run the TDM/sTDM micro-tests for the derive-by-default tdm:: API.
# Compile in the cached ROCm Docker image; run against a software functional
# model (no real GPU). All paths are env-overridable so no site-specific
# install path is baked into the repo.
#
# Env overrides:
#   REPO       repo root (default: current git toplevel)
#   OUT        output dir for binaries (default: /tmp/tdm-out)
#   IMG        compile image (required: a ROCm image with hipcc for the target)
#   MODEL_ENV  path to the functional-model env script to `source` before running
#   MOUNT      host path to bind into Docker (default: REPO)
set -uo pipefail
REPO=${REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo .)}
OUT=${OUT:-/tmp/tdm-out}
IMG=${IMG:?set IMG to a ROCm compile image with hipcc}
MOUNT=${MOUNT:-$REPO}
ARCH=${ARCH:?set ARCH to the target offload arch (e.g. the udna1 target)}
TESTS="tdm_dense tdm_store tdm_store_bits tdm_3d tdm_5d tdm_padded stdm_gather stdm_scatter"
MT="$REPO/kernels/gemm/bf16fp32/$ARCH/micro_tests"

echo "=== node: $(hostname)  repo: $REPO ==="
mkdir -p "$OUT"; rm -f "$OUT"/*.out

echo "=== [1/3] compile micro-tests in Docker ==="
docker run --rm -v "$MOUNT":"$MOUNT" -w "$MT" "$IMG" bash -c '
  for t in '"$TESTS"'; do
    if hipcc -DKITTENS_UDNA1 --offload-arch='"$ARCH"' -std=c++20 -O3 -w \
         -I'"$REPO"'/include $t.cpp -o '"$OUT"'/$t.out 2>/tmp/$t.err; then
      echo "  BUILD $t OK"
    else
      echo "  BUILD $t FAIL"; grep -m2 error: /tmp/$t.err | sed "s/^/      /"
    fi
  done'

echo "=== [2/3] iterate hard-stop must NOT compile (D9) ==="
if docker run --rm -v "$MOUNT":"$MOUNT" -w "$MT" "$IMG" bash -c \
     'hipcc -DKITTENS_UDNA1 --offload-arch='"$ARCH"' -std=c++20 -O3 -w \
        -I'"$REPO"'/include tdm_iterate_hardstop.cpp -o '"$OUT"'/iterate.out' \
     2>/tmp/iterate.err; then
  echo "  FAIL  iterate_hardstop compiled (expected a hard-stop)"; hs_fail=1
else
  echo "  PASS  iterate_hardstop rejected at compile time"; hs_fail=0
fi

echo "=== [3/3] run on the functional model ==="
[ -n "${MODEL_ENV:-}" ] && source "$MODEL_ENV"
pass=0; fail=0
for t in $TESTS; do
  b="$OUT/$t.out"
  if [ ! -f "$b" ]; then echo "  FAIL  $t  (no binary)"; fail=$((fail+1)); continue; fi
  timeout 300 "$b" >/tmp/$t.log 2>&1; rc=$?
  res=$(grep -iE "errors:" /tmp/$t.log | tail -1)
  if [ $rc -eq 0 ] && echo "$res" | grep -qE "errors: 0(/|$)"; then
    echo "  PASS  $t  ($res)"; pass=$((pass+1))
  else
    echo "  FAIL  $t  (rc=$rc) ${res:-<no result>}"; tail -3 /tmp/$t.log | sed "s/^/        /"; fail=$((fail+1))
  fi
done
echo "=== summary: PASS=$pass FAIL=$fail  iterate_hardstop=$([ $hs_fail -eq 0 ] && echo OK || echo REGRESSED) ==="
[ $fail -eq 0 ] && [ $hs_fail -eq 0 ]
