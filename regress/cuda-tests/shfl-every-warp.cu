// A warp shuffle in every warp of a block, not just the first. In functional
// simulation the warp-wide instructions (shfl, wmma) must find the threads of
// the warp that executes them.
//
//   shfl-every-warp   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int THREADS = 128;

__global__ void reverse(unsigned *out) {
  unsigned t = threadIdx.x, lane = t % 32;
  unsigned v = t * 3 + 1, r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 31, -1;"
               : "=r"(r)
               : "r"(v), "r"(31 - lane));
  out[t] = r;
}

int main() {
  unsigned *d, h[THREADS];
  cudaMalloc(&d, sizeof(h));
  reverse<<<1, THREADS>>>(d);
  cudaError_t e = cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (unsigned t = 0; t < THREADS; t++) {
    unsigned want = (t - t % 32 + 31 - t % 32) * 3 + 1;
    if (h[t] != want) {
      if (bad < 10) printf("thread %u: got %u, expected %u\n", t, h[t], want);
      bad++;
    }
  }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
