# This bash script formats GPGPU-Sim using clang-format
THIS_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
clang-format -i ${THIS_DIR}/libcuda/*.h
clang-format -i ${THIS_DIR}/libcuda/*.cc
clang-format -i ${THIS_DIR}/src/*.h
clang-format -i ${THIS_DIR}/src/*.cc
clang-format -i ${THIS_DIR}/src/gpgpu-sim/*.h
clang-format -i ${THIS_DIR}/src/gpgpu-sim/*.cc
clang-format -i ${THIS_DIR}/src/cuda-sim/*.h
clang-format -i ${THIS_DIR}/src/cuda-sim/*.cc
clang-format -i ${THIS_DIR}/src/gpuwattch/*.h
clang-format -i ${THIS_DIR}/src/gpuwattch/*.cc
