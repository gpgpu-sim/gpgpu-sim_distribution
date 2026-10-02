// Warp shuffles in the form nvcc emits for __shfl_*_sync, "shfl d|p", where
// p says whether the source lane was in range; and shuffles within segments
// narrower than a warp (the width argument).
//
//   shfl-predicate   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int TESTS = 6;

__global__ void shuffles(unsigned *out) {
  unsigned lane = threadIdx.x % 32, x = threadIdx.x * 7 + 3, v, valid;
  unsigned *o = out + TESTS * threadIdx.x;
  asm volatile(
      "{ .reg .pred q; shfl.sync.up.b32 %0|q, %2, 3, 0, -1;"
      " selp.u32 %1, 1, 0, q; }"
      : "=r"(v), "=r"(valid)
      : "r"(x));
  o[0] = v;
  o[1] = valid;
  o[2] = __shfl_sync(0xffffffff, x, 31 - lane);
  o[3] = __shfl_down_sync(0xffffffff, x, 2, 16);
  o[4] = __shfl_xor_sync(0xffffffff, x, 5, 8);
  o[5] = __shfl_sync(0xffffffff, x, 3, 16);
}

static unsigned expected(unsigned t, int k) {
  unsigned lane = t % 32, base = t - lane;
  auto x = [&](unsigned l) { return (base + l) * 7 + 3; };
  switch (k) {
    case 0: return lane >= 3 ? x(lane - 3) : x(lane);
    case 1: return lane >= 3;
    case 2: return x(31 - lane);
    case 3: return lane % 16 + 2 < 16 ? x(lane + 2) : x(lane);
    case 4: return x(lane ^ 5);
    default: return x((lane & 16) + 3);
  }
}

int main() {
  const int THREADS = 64;
  unsigned *d, h[TESTS * THREADS];
  cudaMalloc(&d, sizeof(h));
  shuffles<<<1, THREADS>>>(d);
  cudaError_t e = cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (int t = 0; t < THREADS; t++)
    for (int k = 0; k < TESTS; k++) {
      unsigned want = expected(t, k), got = h[TESTS * t + k];
      if (got != want) {
        if (bad < 10)
          printf("thread %d test %d: got %u, expected %u\n", t, k, got, want);
        bad++;
      }
    }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
