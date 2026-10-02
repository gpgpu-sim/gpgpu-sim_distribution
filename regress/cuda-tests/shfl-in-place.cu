// Warp shuffles whose destination is also their source register, in full,
// divergent and predicated warps. Every lane must read the other lane's value
// from before the shuffle, and each lane must write its own destination.
//
//   shfl-in-place   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int TESTS = 7;

__device__ unsigned up(unsigned x, unsigned d) {
  asm volatile("shfl.sync.up.b32 %0, %0, %1, 0, -1;" : "+r"(x) : "r"(d));
  return x;
}
__device__ unsigned down(unsigned x, unsigned d) {
  asm volatile("shfl.sync.down.b32 %0, %0, %1, 31, -1;" : "+r"(x) : "r"(d));
  return x;
}
__device__ unsigned bfly(unsigned x, unsigned d) {
  asm volatile("shfl.sync.bfly.b32 %0, %0, %1, 31, -1;" : "+r"(x) : "r"(d));
  return x;
}
__device__ unsigned idx(unsigned x, unsigned s, unsigned mask) {
  asm volatile("shfl.sync.idx.b32 %0, %0, %1, 31, %2;"
               : "+r"(x)
               : "r"(s), "r"(mask));
  return x;
}

__global__ void shuffles(unsigned *out) {
  unsigned lane = threadIdx.x % 32;
  unsigned *o = out + TESTS * threadIdx.x;
  // inclusive prefix sum of lane + 1
  unsigned x = lane + 1;
  for (unsigned d = 1; d < 32; d *= 2) {
    unsigned t = up(x, d);
    if (lane >= d) x += t;
  }
  o[0] = x;
  // sum over the warp by halving
  x = lane * lane;
  for (unsigned d = 16; d > 0; d /= 2) x += down(x, d);
  o[1] = x;
  // butterfly exchange
  o[2] = bfly(lane * 3 + 1, 5);
  // rotate by one lane
  o[3] = idx(lane * 5 + 2, (lane + 31) % 32, 0xffffffff);
  // divergent: lanes 4k+1, 4k+2, 4k+3 take the value of lane 4k+3
  o[4] = 0;
  if (lane % 4 != 0) o[4] = idx(lane + 100, lane | 3, 0xeeeeeeee);
  // predicated: only even lanes shuffle, odd lanes keep 7
  unsigned p = lane % 2 == 0, v = lane + 200, r = 7;
  asm volatile(
      "{ .reg .pred p; setp.ne.u32 p, %2, 0;"
      " @p shfl.sync.idx.b32 %0, %1, %3, 31, 0x55555555; }"
      : "+r"(r)
      : "r"(v), "r"(p), "r"((lane + 2) % 32));
  o[5] = r;
  // a whole-warp shuffle after the predicated one
  o[6] = idx(lane * 7, 31 - lane, 0xffffffff);
}

static unsigned expected(unsigned lane, int k) {
  switch (k) {
    case 0: return (lane + 1) * (lane + 2) / 2;
    case 1: {
      unsigned s = 0;
      for (unsigned l = 0; l < 32; l++) s += l * l;
      return lane == 0 ? s : 0xdeadbeef;  // only lane 0 has the full sum
    }
    case 2: return (lane ^ 5) * 3 + 1;
    case 3: return ((lane + 31) % 32) * 5 + 2;
    case 4: return lane % 4 ? (lane | 3) + 100 : 0;
    case 5: return lane % 2 ? 7 : (lane + 2) % 32 + 200;
    default: return (31 - lane) * 7;
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
      unsigned want = expected(t % 32, k), got = h[TESTS * t + k];
      if (want == 0xdeadbeef) continue;
      if (got != want) {
        if (bad < 10)
          printf("thread %d test %d: got %u, expected %u\n", t, k, got, want);
        bad++;
      }
    }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
