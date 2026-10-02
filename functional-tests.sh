#!/bin/bash
# Functional regressions: build GPGPU-Sim and the gpu-app-collection suites
# the app list needs, fetch their input data, and run every app in an app
# definition file in PTX mode (regress/define-functional-pr.yml for the PR
# tier, regress/define-functional-long.yml for the long tier), after the small
# CUDA tests in regress/cuda-tests. Exits non-zero if any test or app crashes,
# deadlocks, asserts or fails its own check.
#
#   ./functional-tests.sh        everything
#   ./functional-tests.sh data   only fetch the input data into $APPDATA
#   ./functional-tests.sh refs   run the apps natively on a GPU instead of the
#                                simulator and write their results to $REFS
#
# Runs in the Accel-Sim regression image (CI), or locally with CUDA_INSTALL_PATH
# and GPUAPPS_ROOT set. Environment:
#   CONFIG     simulator config (default QV100)
#   APPS       app definition file (default regress/define-functional-pr.yml)
#   SUITES     comma-separated suites of APPS to run (default: all of them)
#   CORES      parallel simulations (default: all cores, at most 8)
#   HOURS      monitor time limit (default 2)
#   APPDATA    directory holding the extracted input data; reused if present
#   REFS       GPU reference results (regress/functional-refs.py); if set, each
#              run's results must match them
#   PTX_SIM_MODE_FUNC  1 for pure functional simulation (no timing model)
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
only = '${SUITES:-}'.split(',') if '${SUITES:-}' else list(d)
for s, v in d.items():
    if s not in only: continue
    if f == 'suite': print(s)
    elif f == 'exe': print('\n'.join(list(e)[0] for e in v['execs']))
    elif v.get(f): print(v[f])
" | sort -u
}
SUITES=$(field suite | paste -sd, -)
[ -n "$SUITES" ] || { echo "no suites selected from $APPS"; exit 1; }
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

MODE=${1:-sim}
echo "config=$CONFIG cores=$CORES hours=$HOURS suites=$SUITES mode=$MODE"
git config --global --add safe.directory '*' 2>/dev/null || true
cd "$ROOT"

if [ "$MODE" != refs ]; then
  echo "::group::Build GPGPU-Sim"
  cmake -B build && cmake --build build -j && cmake --install build
  set +u; source setup > /dev/null; set -u
  echo "::endgroup::"
  echo "::group::CUDA tests (regress/cuda-tests)"
  tests_rc=0
  "$ROOT/regress/cuda-tests/run.sh" || tests_rc=1
  echo "::endgroup::"
fi

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

if [ "$MODE" = refs ]; then
  : "${REFS:?set REFS to the reference file to write}"
  nvidia-smi -L
  export CUDA_VERSION=$(basename "$(dirname "$BIN")")
  export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:${LD_LIBRARY_PATH:-}
  SUITES=$SUITES python3 "$ROOT/regress/functional-refs.py" capture "$APPS" "$REFS"
  exit 0
fi

echo "::group::Accel-Sim tools"
[ -d accel-sim-framework ] || git clone -q https://github.com/accel-sim/accel-sim-framework.git
git -C accel-sim-framework checkout -q "$ACCELSIM_REF"
JL=./accel-sim-framework/util/job_launching
cp "$APPS" $JL/apps/$(basename "$APPS")
echo "::endgroup::"

$JL/run_simulations.py -C "$CONFIG" -B "$SUITES" -N functional -l local -c "$CORES"
set +e
$JL/monitor_func_test.py -v -N functional -j procman -T "$HOURS" -S 60
rc=$?
set -e

echo "::group::Per-app time and instructions"
# The monitor counts a run that exits cleanly as passing even if it never
# launched a kernel; a run with no simulated instructions fails here.
nosim=""
for o in $(find accel-sim-framework/sim_run_* -name '*.o[0-9]*' | sort); do
  t=$(grep 'gpgpu_simulation_time' "$o" | tail -1 | sed 's/.*= *//')
  i=$(grep 'gpu_tot_sim_insn' "$o" | tail -1 | sed 's/.*= *//')
  if [ -z "$i" ] && [ -n "$t" ]; then
    # functional mode (PTX_SIM_MODE_FUNC=1) has no gpu_tot_sim_insn; estimate
    # thread instructions from its simulation rate and time
    r=$(grep 'gpgpu_simulation_rate' "$o" | tail -1 | sed 's/.*= *//; s/ .*//')
    s=$(echo "$t" | sed 's/.*(\([0-9]*\) sec).*/\1/')
    i="~$((${r:-0} * ${s:-0}))"
  fi
  run=$(echo $o | sed 's|.*sim_run_[^/]*/||; s|/[^/]*$||')
  echo "$run | $t | insn=$i"
  [ -n "$i" ] && [ "$i" != 0 ] && [ "$i" != "~0" ] || nosim="$nosim $o"
done
echo "::endgroup::"
for o in $nosim; do
  echo "::error::no simulated instructions in $o; its last lines:"
  tail -20 "$o"
  rc=1
done
if [ -n "${REFS:-}" ]; then
  echo "::group::Results against the GPU"
  python3 "$ROOT/regress/functional-refs.py" compare "$REFS" accel-sim-framework/sim_run_* || rc=1
  echo "::endgroup::"
fi
if [ "${tests_rc:-0}" != 0 ]; then
  echo "::error::a CUDA test in regress/cuda-tests failed (see its group above)"
  rc=1
fi
exit $rc
