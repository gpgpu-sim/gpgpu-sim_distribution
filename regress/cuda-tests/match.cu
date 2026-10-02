// match.any.sync and match.all.sync (__match_any_sync, __match_all_sync) on
// 32- and 64-bit values, with all lanes, with only the lanes of a divergent
// branch, and with the destination register as the source. Results must
// match the host.
//
//   match   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 128;
const int OPS = 7;
const unsigned FULL = 0xffffffff;

__global__ void match(const unsigned *in, unsigned *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = in[i], *o = out + OPS * i;
  int pred;
  o[0] = __match_any_sync(FULL, v);
  o[1] = __match_all_sync(FULL, v, &pred);
  o[2] = pred;
  // the high halves differ between even and odd lanes
  unsigned long long w = (unsigned long long)(i & 1) << 40 | v;
  o[3] = __match_any_sync(FULL, w);
  o[4] = 0;
  if (v & 1) o[4] = __match_any_sync(__activemask(), v >> 1);
  unsigned x = v;
  asm volatile("match.any.sync.b32 %0, %0, %1;" : "+r"(x) : "r"(FULL));
  o[5] = x;
  o[6] = __match_all_sync(FULL, w >> 40 << 40, &pred) | (pred ? 0 : 1);
}

int main() {
  static unsigned in[N], out[OPS * N];
  unsigned x = 0x2545f491;
  for (int i = 0; i < N; i++) {
    x ^= x << 13, x ^= x >> 17, x ^= x << 5;  // xorshift32
    in[i] = x % 6;
  }
  for (int i = 32; i < 64; i++) in[i] = 5;  // warp 1: all the same
  unsigned *din, *dout;
  cudaMalloc(&din, sizeof(in));
  cudaMalloc(&dout, sizeof(out));
  cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
  match<<<N / 64, 64>>>(din, dout);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < N; i++) {
    int w0 = i / 32 * 32;
    unsigned v = in[i], any = 0, any64 = 0, odd = 0, all = 1;
    for (int l = 0; l < 32; l++) {
      unsigned u = in[w0 + l];
      if (u == v) any |= 1u << l;
      if (u == v && (l & 1) == (i & 1)) any64 |= 1u << l;
      if ((u & 1) && (u >> 1) == (v >> 1)) odd |= 1u << l;
      if (u != in[w0]) all = 0;
    }
    unsigned want[OPS] = {any,   all ? FULL : 0, all, any64,
                          v & 1 ? odd : 0, any, 1};
    for (int k = 0; k < OPS; k++)
      if (out[OPS * i + k] != want[k]) {
        if (bad < 10)
          printf("thread %d op %d (value %u): got %08x, expected %08x\n", i, k,
                 v, out[OPS * i + k], want[k]);
        bad++;
      }
  }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
