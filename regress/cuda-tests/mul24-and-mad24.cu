// mul24 and mad24 (.lo, .hi, .hi.sat), signed and unsigned, on operands
// whose bits above 23 are set. The product is of the low 24 bits of each
// source, sign-extended for .s32. Results must match the host.
//
//   mul24-and-mad24   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 256;
const int OPS = 9;

__global__ void multiply(const unsigned *in, unsigned *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2], r;
  unsigned *o = out + OPS * i;
  asm("mul24.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  o[0] = r;
  asm("mul24.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  o[1] = r;
  asm("mul24.lo.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  o[2] = r;
  asm("mul24.hi.s32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  o[3] = r;
  asm("mad24.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[4] = r;
  asm("mad24.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[5] = r;
  asm("mad24.lo.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[6] = r;
  asm("mad24.hi.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[7] = r;
  asm("mad24.hi.sat.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[8] = r;
}

static long long s24(unsigned x) { return (int)(x << 8) >> 8; }

static unsigned reference(unsigned a, unsigned b, unsigned c, int op) {
  unsigned long long u = (unsigned long long)(a & 0xffffff) * (b & 0xffffff);
  long long s = s24(a) * s24(b);
  long long sat;
  switch (op) {
    case 0: return (unsigned)u;
    case 1: return (unsigned)(u >> 16);
    case 2: return (unsigned)s;
    case 3: return (unsigned)(s >> 16);
    case 4: return (unsigned)u + c;
    case 5: return (unsigned)(u >> 16) + c;
    case 6: return (unsigned)s + c;
    case 7: return (unsigned)(s >> 16) + c;
    default:
      sat = (s >> 16) + (int)c;
      if (sat > 0x7fffffffLL) sat = 0x7fffffffLL;
      if (sat < -0x80000000LL) sat = -0x80000000LL;
      return (unsigned)sat;
  }
}

int main() {
  static unsigned in[3 * N], out[OPS * N];
  unsigned x = 0x2545f491;
  for (int i = 0; i < 3 * N; i++) {
    x ^= x << 13, x ^= x >> 17, x ^= x << 5;  // xorshift32
    in[i] = x;
  }
  // largest magnitudes, so .hi.sat saturates both ways
  in[0] = 0x00800000, in[1] = 0x00800000, in[2] = 0x7fffffff;
  in[3] = 0x00800000, in[4] = 0x007fffff, in[5] = 0x80000000;
  unsigned *din, *dout;
  cudaMalloc(&din, sizeof(in));
  cudaMalloc(&dout, sizeof(out));
  cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
  multiply<<<N / 64, 64>>>(din, dout);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < N; i++)
    for (int k = 0; k < OPS; k++) {
      unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
      unsigned want = reference(a, b, c, k), got = out[OPS * i + k];
      if (got != want) {
        if (bad < 10)
          printf("op %d on %08x %08x %08x: got %08x, expected %08x\n", k, a, b,
                 c, got, want);
        bad++;
      }
    }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
