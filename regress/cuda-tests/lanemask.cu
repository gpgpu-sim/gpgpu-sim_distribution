// The %lanemask_eq, _le, _lt, _ge and _gt special registers, read by every
// thread of several warps. Results must match the host.
//
//   lanemask   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 128;
const int OPS = 5;

__global__ void masks(unsigned *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned *o = out + OPS * i;
  asm("mov.u32 %0, %%lanemask_eq;" : "=r"(o[0]));
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(o[1]));
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(o[2]));
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(o[3]));
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(o[4]));
}

int main() {
  static unsigned out[OPS * N];
  unsigned *dout;
  cudaMalloc(&dout, sizeof(out));
  masks<<<N / 64, 64>>>(dout);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < N; i++) {
    unsigned lane = i % 32, eq = 1u << lane, lt = eq - 1, le = lt | eq;
    unsigned want[OPS] = {eq, le, lt, ~lt, ~le};
    for (int k = 0; k < OPS; k++)
      if (out[OPS * i + k] != want[k]) {
        if (bad < 10)
          printf("thread %d op %d: got %08x, expected %08x\n", i, k,
                 out[OPS * i + k], want[k]);
        bad++;
      }
  }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
