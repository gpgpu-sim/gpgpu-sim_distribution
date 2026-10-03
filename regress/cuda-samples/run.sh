#!/bin/bash
# Build NVIDIA cuda-samples (scaled down by scale.sh) and run the ones listed
# in samples.txt in GPGPU-Sim. Each sample checks its own answer against the
# CPU and exits 0 when it passes (2 when waived). Run from the repository root
# after `source setup`; exits 1 if any sample fails.
#
#   regress/cuda-samples/run.sh [sample ...]   (default: every sample listed)
#
# Environment: CS (checkout location, default /tmp/cuda-samples), SIM_T
# (seconds per sample, default 900), NJ (parallel samples, default nproc,
# at most 8), CONFIG_DIR (default configs/tested-cfgs/SM7_QV100).
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$ROOT/regress/cuda-samples
CS=${CS:-/tmp/cuda-samples}
SIM_T=${SIM_T:-900}
NJ=${NJ:-$(nproc)}; [ "$NJ" -gt 8 ] && NJ=8
CONFIG_DIR=${CONFIG_DIR:-$ROOT/configs/tested-cfgs/SM7_QV100}
TAG=v12.8
[ -n "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ] || { echo "source setup first"; exit 1; }

# name [args] per line; '#' starts a comment
list=$(sed 's/#.*//; s/[[:space:]]*$//' $HERE/samples.txt | awk 'NF')
[ $# -gt 0 ] && list=$(printf '%s\n' "$list" | awk -v want=" $* " 'index(want, " " $1 " ")')
names=$(printf '%s\n' "$list" | awk '{print $1}')

if [ ! -d $CS/.git ]; then
  git -c advice.detachedHead=false clone -q --depth 1 -b $TAG https://github.com/NVIDIA/cuda-samples.git $CS || exit 1
  $HERE/scale.sh $CS || exit 1
fi
( cd $CS && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared > /tmp/cuda-samples-cmake.log 2>&1 &&
  cmake --build build -j$(nproc) --target $names -- -k > /tmp/cuda-samples-build.log 2>&1 ) ||
  { grep -E 'error|Error' /tmp/cuda-samples-build.log | head -20; }

one() { # name args...
  n=$1; shift
  exe=$(find $CS/build/Samples -type f -executable -name $n ! -path '*CMakeFiles*' | head -1)
  [ -n "$exe" ] || { printf "%-28s %-8s %5s  %s\n" $n NOBUILD - "not built"; return; }
  d=$OUT/$n; rm -rf $d; mkdir -p $d
  src=$CS/$(dirname ${exe#$CS/build/}); cp -as $src/. $d/ 2>/dev/null
  cp $CONFIG_DIR/* $d/
  s=$(date +%s)
  ( cd $d && PTX_SIM_MODE_FUNC=1 timeout $SIM_T $exe "$@" > out.txt 2>&1 < /dev/null ); rc=$?
  t=$(( $(date +%s) - s ))
  if [ $rc = 2 ]; then v=WAIVED
  elif [ $rc = 124 ]; then v=TIMEOUT
  elif [ $rc = 0 ] && ! grep -qE 'FAIL|[Tt]est failed' $d/out.txt; then v=PASS
  else v=FAIL; fi
  [ $v = TIMEOUT ] || grep -q "GPGPU-Sim" $d/out.txt || v=NOSIM # ran on the GPU, not the simulator
  why=$(grep -m1 -E 'ERROR|Assertion|not implemented|Segmentation|CRASH|CUDA error|FAIL' $d/out.txt | cut -c1-100)
  printf "%-28s %-8s %5s  %s\n" $n $v $t "$why"
}
export -f one; export CS SIM_T CONFIG_DIR OUT=${OUT:-/tmp/cuda-samples-runs}
printf '%s\n' "$list" | xargs -P $NJ -L 1 bash -c 'one "$@"' _ | sort > /tmp/cuda-samples.txt
printf "%-28s %-8s %5s  %s\n" sample result secs note; cat /tmp/cuda-samples.txt
for f in $(awk '$2 != "PASS" && $2 != "WAIVED" {print $1}' /tmp/cuda-samples.txt); do
  echo "::group::$f output (last 30 lines)"; tail -30 $OUT/$f/out.txt 2>/dev/null; echo "::endgroup::"
done
want=$(printf '%s\n' "$list" | wc -l); got=$(wc -l < /tmp/cuda-samples.txt)
[ "$got" = "$want" ] || { echo "only $got of $want samples reported a result"; exit 1; }
! awk '$2 != "PASS" && $2 != "WAIVED"' /tmp/cuda-samples.txt | grep -q .
