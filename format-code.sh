# This bash script formats GPGPU-Sim using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
echo "Running clang-format on $THIS_DIR"
clang-format -i ${THIS_DIR}/libcuda/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/libcuda/*.cc --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/*.cc --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/gpgpu-sim/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/gpgpu-sim/*.cc --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/cuda-sim/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/cuda-sim/*.cc --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/accelwattch/*.h --style=file:${THIS_DIR}/.clang-format
clang-format -i ${THIS_DIR}/src/accelwattch/*.cc --style=file:${THIS_DIR}/.clang-format
