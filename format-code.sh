# This bash script formats GPGPU-Sim using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

# Require specific clang-format version for consistency
REQUIRED_VERSION="16.0.6"

# Find clang-format from Python environment (venv or user install)
PY_SCRIPTS_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('scripts'))" 2>/dev/null)
PY_USER_SCRIPTS_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('scripts', sysconfig.get_preferred_scheme('user')))" 2>/dev/null)

if [ -n "$PY_SCRIPTS_DIR" ] && [ -x "$PY_SCRIPTS_DIR/clang-format" ]; then
    CLANG_FORMAT="$PY_SCRIPTS_DIR/clang-format"
elif [ -n "$PY_USER_SCRIPTS_DIR" ] && [ -x "$PY_USER_SCRIPTS_DIR/clang-format" ]; then
    CLANG_FORMAT="$PY_USER_SCRIPTS_DIR/clang-format"
else
    echo "Error: clang-format not found in Python environment"
    echo "  Checked: ${PY_SCRIPTS_DIR:-?}/clang-format"
    echo "  Checked: ${PY_USER_SCRIPTS_DIR:-?}/clang-format"
    echo "Please install it with: pip install clang-format==$REQUIRED_VERSION"
    echo "  On newer Ubuntu (23.04+), use: pipx install clang-format==$REQUIRED_VERSION"
    echo "  Or create a venv and install there"
    exit 1
fi

CURRENT_VERSION=$("$CLANG_FORMAT" --version | grep -oP '\d+\.\d+\.\d+')
if [ "$CURRENT_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "Error: clang-format version mismatch"
    echo "  Required: $REQUIRED_VERSION"
    echo "  Found:    $CURRENT_VERSION ($(command -v "$CLANG_FORMAT" || echo "$CLANG_FORMAT"))"
    echo "Please install the correct version with: pip install clang-format==$REQUIRED_VERSION"
    exit 1
fi

echo "Running clang-format $REQUIRED_VERSION on $THIS_DIR"
"$CLANG_FORMAT" -i ${THIS_DIR}/libcuda/*.h --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/libcuda/*.cc --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/*.h --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/*.cc --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/gpgpu-sim/*.h --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/gpgpu-sim/*.cc --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/cuda-sim/*.h --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/cuda-sim/*.cc --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/accelwattch/*.h --style=file:${THIS_DIR}/.clang-format
"$CLANG_FORMAT" -i ${THIS_DIR}/src/accelwattch/*.cc --style=file:${THIS_DIR}/.clang-format
