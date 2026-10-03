#!/bin/bash
# Shrink cuda-samples v12.8 problem sizes and repeat counts so the samples
# finish in GPGPU-Sim in a few minutes, and build them the way the simulator
# needs: compute_70 PTX, shared cudart (set at configure time), and no
# separable compilation (with -rdc the executables carry no PTX).
#
#   scale.sh <cuda-samples checkout>
CS=${1:?usage: scale.sh <cuda-samples checkout>}
S=$CS/Samples
bad=0
# edit <file> <sed expression> <text that must be present afterwards>
edit() {
  sed -i "$2" "$S/$1"
  grep -qF -- "$3" "$S/$1" || { echo "scale.sh: no match in $1 for: $2"; bad=1; }
}
for f in $(find $S -name CMakeLists.txt); do
  sed -i 's/^set(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES 70)/' $f
  case $f in */cdp*) ;; *) sed -i 's/CUDA_SEPARABLE_COMPILATION ON/CUDA_SEPARABLE_COMPILATION OFF/' $f ;; esac
done

edit 5_Domain_Specific/BlackScholes/BlackScholes.cu \
  's/^const int OPT_N = 4000000;/const int OPT_N = 65536;/; s/^const int NUM_ITERATIONS = 512;/const int NUM_ITERATIONS = 1;/' \
  'const int NUM_ITERATIONS = 1;'
edit 2_Concepts_and_Techniques/convolutionSeparable/main.cpp \
  's/const int imageW = 3072;/const int imageW = 768;/; s/const int imageH = 3072;/const int imageH = 768;/; s/const int iterations = 16;/const int iterations = 1;/' \
  'const int iterations = 1;'
edit 5_Domain_Specific/fastWalshTransform/fastWalshTransform.cu \
  's/^const int log2Data = 23;/const int log2Data = 18;/' 'const int log2Data = 18;'
edit 2_Concepts_and_Techniques/histogram/main.cpp \
  's/^const int numRuns = 16;/const int numRuns = 1;/; s/uint byteCount = 64 \* 1048576;/uint byteCount = 1048576;/' \
  'uint byteCount = 1048576;'
edit 0_Introduction/mergeSort/main.cpp \
  's/const uint N = 4 \* 1048576;/const uint N = 65536;/' 'const uint N = 65536;'
edit 2_Concepts_and_Techniques/scan/main.cpp \
  's/const uint N = 13 \* 1048576 \/ 2;/const uint N = 262144;/; s/const int iCycles = 100;/const int iCycles = 1;/' \
  'const int iCycles = 1;'
edit 2_Concepts_and_Techniques/sortingNetworks/main.cpp \
  's/const uint N = 1048576;/const uint N = 65536;/' 'const uint N = 65536;'
edit 6_Performance/alignedTypes/alignedTypes.cu \
  's/^const int MEM_SIZE = 50000000;/const int MEM_SIZE = 1000000;/; s/^const int NUM_ITERATIONS = 32;/const int NUM_ITERATIONS = 1;/' \
  'const int NUM_ITERATIONS = 1;'
edit 2_Concepts_and_Techniques/eigenvalues/main.cu \
  's/unsigned int mat_size = 2048;/unsigned int mat_size = 1024;/; s/unsigned int iters_timing = 100;/unsigned int iters_timing = 1;/' \
  'unsigned int iters_timing = 1;'
edit 5_Domain_Specific/MonteCarloMultiGPU/MonteCarloMultiGPU.cpp \
  's/int nOptions = 8 \* 1024;/int nOptions = 64;/; s/int PATH_N = 262144;/int PATH_N = 8192;/' 'int PATH_N = 8192;'
edit 0_Introduction/simpleHyperQ/simpleHyperQ.cu \
  's/int nstreams = 32; /int nstreams = 8; /; s/float kernel_time = 10; /float kernel_time = 0.01; /' \
  'float kernel_time = 0.01;'
edit 0_Introduction/simpleMultiCopy/simpleMultiCopy.cu \
  's/^int N = 1 << 22;/int N = 1 << 16;/; s/^int nreps = 10; /int nreps = 1; /' 'int N = 1 << 16;'
edit 6_Performance/transpose/transpose.cu \
  's/^int MATRIX_SIZE_X = 1024;/int MATRIX_SIZE_X = 256;/; s/^int MATRIX_SIZE_Y = 1024;/int MATRIX_SIZE_Y = 256;/; s/^#define NUM_REPS 100/#define NUM_REPS 1/' \
  '#define NUM_REPS 1'
exit $bad
