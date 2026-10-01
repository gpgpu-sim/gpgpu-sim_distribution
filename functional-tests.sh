#!/bin/bash
# PR-tier functional regressions: build GPGPU-Sim and the gpu-app-collection
# suites the app list needs, fetch their input data, and run every app in
# regress/define-functional-pr.yml in PTX mode. Exits non-zero if any app
# crashes, deadlocks, asserts or fails its own check.
#
#   ./functional-tests.sh        everything
#   ./functional-tests.sh data   only fetch the input data into $APPDATA
#
# Runs in the Accel-Sim regression image (CI), or locally with CUDA_INSTALL_PATH
# and GPUAPPS_ROOT set. Environment:
#   CONFIG     simulator config (default QV100)
#   APPS       app definition file (default regress/define-functional-pr.yml)
#   CORES      parallel simulations (default: all cores, at most 8)
#   HOURS      monitor time limit (default 2)
#   APPDATA    directory holding the extracted input data; reused if present
#   ACCELSIM_REF  Accel-Sim tools commit

set -eu
: "${CUDA_INSTALL_PATH:?set CUDA_INSTALL_PATH}"
: "${GPUAPPS_ROOT:?set GPUAPPS_ROOT to a gpu-app-collection checkout}"
ROOT=$(cd "$(dirname "$0")" && pwd)
CONFIG=${CONFIG:-QV100}
APPS=$(realpath "${APPS:-$ROOT/regress/define-functional-pr.yml}")
CORES=${CORES:-$(nproc)}; [ "$CORES" -gt 8 ] && CORES=8
HOURS=${HOURS:-2}
APPDATA=${APPDATA:-$HOME/gpgpu-sim-appdata}
ACCELSIM_REF=${ACCELSIM_REF:-d930ad6d02c09bb56867132583735aba0389cff4}
GAC_REF=dad09cb0487845edc7524ded814c6cde9f0ef6a1
DATA_URL=https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/all.gpgpu-sim-app-data.tgz
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

# Fields of the app definition file: suite names, make targets, data
# subdirectories and executables.
field() {
  python3 -c "
import sys, yaml
d = yaml.safe_load(open('$APPS'))
f = '$1'
for s, v in d.items():
    if f == 'suite': print(s)
    elif f == 'exe': print('\n'.join(list(e)[0] for e in v['execs']))
    elif v.get(f): print(v[f])
" | sort -u
}
SUITES=$(field suite | paste -sd, -)
DATA_SUBDIRS=$(field data_subdir)

fetch_data() {
  # Only the data directories these suites use are extracted (the full
  # tarball is 2.5 GB and Purdue's server is slow); a cache or an earlier run
  # can pre-populate $APPDATA.
  local want="" d
  for d in $DATA_SUBDIRS; do [ -d "$APPDATA/$d" ] || want="$want *$d/*"; done
  if [ -n "$want" ]; then
    mkdir -p "$APPDATA"
    echo "downloading $DATA_URL for:$want"
    # the regression image has wget but not curl
    wget -q -O- --tries=3 "$DATA_URL" | tar xz -C "$APPDATA" --wildcards $want
  fi
  for d in $DATA_SUBDIRS; do
    [ -d "$APPDATA/$d" ] || { echo "missing $APPDATA/$d after extraction"; exit 1; }
  done
  du -sh "$APPDATA"
}

if [ "${1:-}" = data ]; then fetch_data; exit 0; fi

echo "config=$CONFIG cores=$CORES hours=$HOURS suites=$SUITES"
git config --global --add safe.directory '*' 2>/dev/null || true

echo "::group::Build GPGPU-Sim"
cd "$ROOT"
cmake -B build && cmake --build build -j && cmake --install build
set +u; source setup > /dev/null; set -u
echo "::endgroup::"

echo "::group::Build benchmark suites"
# The minimal image carries only rodinia_2.0-ft; fetch the sources if missing.
if [ ! -f "$GPUAPPS_ROOT/src/Makefile" ]; then
  git clone -q https://github.com/accel-sim/gpu-app-collection.git /tmp/gac
  git -C /tmp/gac checkout -q $GAC_REF
  cp -a /tmp/gac/. "$GPUAPPS_ROOT/"
fi
(
  cd "$GPUAPPS_ROOT/src"
  set +u; source ./setup_environment > /dev/null; set -u
  for t in $(field make_target); do
    # -i: a suite target also builds apps that are not in the list and may
    # not compile under this CUDA; the binary check below is what counts.
    echo "make $t"; make -i -j"$CORES" "$t" > /tmp/make-$t.log 2>&1 || true
  done
)
BIN=$GPUAPPS_ROOT/bin/$(nvcc --version | sed -nre 's/.*release ([0-9]+\.[0-9]+).*/\1/p')/release
for exe in $(field exe); do
  [ -x "$BIN/$exe" ] || { echo "binary $BIN/$exe was not built"; grep -ih error /tmp/make-*.log | head -30; exit 1; }
done
echo "::endgroup::"

echo "::group::Input data"
fetch_data
for d in $DATA_SUBDIRS; do
  mkdir -p "$GPUAPPS_ROOT/$(dirname $d)"
  rm -rf "$GPUAPPS_ROOT/$d"; ln -s "$APPDATA/$d" "$GPUAPPS_ROOT/$d"
done
echo "::endgroup::"

echo "::group::Accel-Sim tools"
[ -d accel-sim-framework ] || git clone -q https://github.com/accel-sim/accel-sim-framework.git
git -C accel-sim-framework checkout -q "$ACCELSIM_REF"
JL=./accel-sim-framework/util/job_launching
cp "$APPS" $JL/apps/define-functional-pr.yml
echo "::endgroup::"

$JL/run_simulations.py -C "$CONFIG" -B "$SUITES" -N functional -l local -c "$CORES"
set +e
$JL/monitor_func_test.py -v -N functional -j procman -T "$HOURS" -S 60
rc=$?
set -e

echo "::group::Per-app time and instructions"
for o in $(find accel-sim-framework/sim_run_* -name '*.o[0-9]*' | sort); do
  t=$(grep 'gpgpu_simulation_time' "$o" | tail -1 | sed 's/.*= *//')
  i=$(grep 'gpu_tot_sim_insn' "$o" | tail -1 | sed 's/.*= *//')
  echo "$(echo $o | sed 's|.*sim_run_[^/]*/||; s|/[^/]*$||') | $t | insn=$i"
done
echo "::endgroup::"
exit $rc
