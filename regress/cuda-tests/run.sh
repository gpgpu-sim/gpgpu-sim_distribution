#!/bin/bash
# Small self-checking CUDA programs for simulator bugs that the benchmark
# suites only show indirectly. Each one is built for compute_70 PTX and run
# under GPGPU-Sim (QV100 config, functional mode); it must exit 0 and print
# PASSED. Expects GPGPU-Sim's setup to be sourced and nvcc on PATH.
#
#   regress/cuda-tests/run.sh [test.cu ...]   default: every .cu here

set -u
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/../.." && pwd)
WORK=$(mktemp -d)
cp "$ROOT"/configs/tested-cfgs/SM7_QV100/* "$WORK"/
[ $# -gt 0 ] || set -- "$DIR"/*.cu
rc=0
for cu in "$@"; do
  t=$(basename "$cu" .cu)
  if ! nvcc -O2 --cudart shared -gencode arch=compute_70,code=compute_70 \
      -o "$WORK/$t" "$cu" > "$WORK/$t.build" 2>&1; then
    echo "::error::$t does not compile"; cat "$WORK/$t.build"; rc=1; continue
  fi
  (cd "$WORK" && PTX_SIM_MODE_FUNC=1 timeout 1200 "./$t" > "$t.out" 2>&1)
  r=$?
  if [ $r -eq 0 ] && grep -qx PASSED "$WORK/$t.out"; then
    echo "$t: PASSED"
  else
    echo "::error::$t failed (exit $r); its last lines:"
    tail -20 "$WORK/$t.out"; rc=1
  fi
done
rm -rf "$WORK"
exit $rc
