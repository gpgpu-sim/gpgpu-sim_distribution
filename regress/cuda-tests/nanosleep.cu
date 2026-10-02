// nanosleep, with an immediate and a register operand (__nanosleep), which
// CUB emits in its decoupled look-back scans. The kernel must run.
//
//   nanosleep   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 64;

__global__ void sleep_then_write(unsigned *out, unsigned ns) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  asm volatile("nanosleep.u32 100;");
  __nanosleep(ns);
  out[i] = i + 1;
}

int main() {
  static unsigned out[N];
  unsigned *dout;
  cudaMalloc(&dout, sizeof(out));
  cudaMemset(dout, 0, sizeof(out));
  sleep_then_write<<<1, N>>>(dout, 450);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < N; i++)
    if (out[i] != (unsigned)i + 1) bad++;
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
