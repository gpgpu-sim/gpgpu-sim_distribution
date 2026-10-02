// Integer mul and mad in their .hi and .wide forms with operands large enough
// that the full product needs twice the operand width, and float fma and
// mad, which round once (not after the multiply and again after the add).
// Results must match the host bit for bit.
//
//   mul-mad-and-fma   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

const int N = 256;
const int OPS = 16;

struct In {
  unsigned long long a, b, c;
};

__global__ void ops(const In *in, unsigned long long *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long a = in[i].a, b = in[i].b, c = in[i].c;
  unsigned a32 = a, b32 = b, c32 = c;
  unsigned short a16 = a, b16 = b, c16 = c;
  unsigned long long *o = out + OPS * i;
  unsigned r32;
  unsigned long long r64;
  unsigned short r16;
  asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r32) : "r"(a32), "r"(b32), "r"(c32));
  o[0] = r32;
  asm("mad.hi.s32 %0, %1, %2, %3;" : "=r"(r32) : "r"(a32), "r"(b32), "r"(c32));
  o[1] = r32;
  asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r64) : "r"(a32), "r"(b32), "l"(c));
  o[2] = r64;
  asm("mad.wide.s32 %0, %1, %2, %3;" : "=l"(r64) : "r"(a32), "r"(b32), "l"(c));
  o[3] = r64;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r32) : "r"(a32), "r"(b32), "r"(c32));
  o[4] = r32;
  asm("mad.hi.u16 %0, %1, %2, %3;" : "=h"(r16) : "h"(a16), "h"(b16), "h"(c16));
  o[5] = r16;
  asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(r64) : "l"(a), "l"(b), "l"(c));
  o[6] = r64;
  asm("mad.hi.s64 %0, %1, %2, %3;" : "=l"(r64) : "l"(a), "l"(b), "l"(c));
  o[7] = r64;
  o[8] = __umul64hi(a, b);
  o[9] = __mul64hi(a, b);
  o[10] = __umulhi(a32, b32);
  o[11] = (unsigned long long)a32 * b32;
  // floats in [1, 2) with full random mantissas
  float fa = __int_as_float(a32 >> 9 | 0x3f800000);
  float fb = __int_as_float(b32 >> 9 | 0x3f800000);
  float fc = -fa * fb;  // rounded product: fma leaves the rounding error
  double da = __longlong_as_double(a >> 12 | 0x3ff0000000000000ULL);
  double db = __longlong_as_double(b >> 12 | 0x3ff0000000000000ULL);
  double dc = -da * db;
  float f;
  double g;
  f = fmaf(fa, fb, fc);
  memcpy(&r32, &f, 4), o[12] = r32;
  asm("mad.rn.f32 %0, %1, %2, %3;" : "=f"(f) : "f"(fa), "f"(fb), "f"(fc));
  memcpy(&r32, &f, 4), o[13] = r32;
  g = fma(da, db, dc);
  memcpy(&r64, &g, 8), o[14] = r64;
  asm("mad.rn.f64 %0, %1, %2, %3;" : "=d"(g) : "d"(da), "d"(db), "d"(dc));
  memcpy(&r64, &g, 8), o[15] = r64;
}

static void reference(const In &in, unsigned long long *o) {
  unsigned long long a = in.a, b = in.b, c = in.c;
  unsigned a32 = a, b32 = b, c32 = c;
  int s32a = a32, s32b = b32, s32c = c32;
  unsigned short a16 = a, b16 = b, c16 = c;
  o[0] = (unsigned)((((unsigned long long)a32 * b32) >> 32) + c32);
  o[1] = (unsigned)(int)((((long long)s32a * s32b) >> 32) + s32c);
  o[2] = (unsigned long long)a32 * b32 + c;
  o[3] = (unsigned long long)((long long)s32a * s32b + (long long)c);
  o[4] = (unsigned)(a32 * b32 + c32);
  o[5] = (unsigned short)((((unsigned)a16 * b16) >> 16) + c16);
  o[6] = (unsigned long long)(((unsigned __int128)a * b) >> 64) + c;
  o[7] = (unsigned long long)((long long)(((__int128)(long long)a *
                                           (long long)b) >> 64) +
                              (long long)c);
  o[8] = (unsigned long long)(((unsigned __int128)a * b) >> 64);
  o[9] = (unsigned long long)(((__int128)(long long)a * (long long)b) >> 64);
  o[10] = (unsigned)(((unsigned long long)a32 * b32) >> 32);
  o[11] = (unsigned long long)a32 * b32;
  unsigned fa_bits = a32 >> 9 | 0x3f800000, fb_bits = b32 >> 9 | 0x3f800000;
  unsigned long long da_bits = a >> 12 | 0x3ff0000000000000ULL;
  unsigned long long db_bits = b >> 12 | 0x3ff0000000000000ULL;
  float fa, fb;
  double da, db;
  memcpy(&fa, &fa_bits, 4), memcpy(&fb, &fb_bits, 4);
  memcpy(&da, &da_bits, 8), memcpy(&db, &db_bits, 8);
  volatile float vp = fa * fb;
  float fc = -vp;
  volatile double vq = da * db;
  double dc = -vq;
  float f = std::fma(fa, fb, fc);
  double g = std::fma(da, db, dc);
  unsigned r32;
  unsigned long long r64;
  memcpy(&r32, &f, 4), o[12] = o[13] = r32;
  memcpy(&r64, &g, 8), o[14] = o[15] = r64;
}

int main() {
  static In in[N];
  static unsigned long long out[OPS * N], want[OPS];
  unsigned long long x = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < N; i++) {
    unsigned long long *v = &in[i].a;
    for (int k = 0; k < 3; k++) {
      x ^= x << 13, x ^= x >> 7, x ^= x << 17;  // xorshift64
      v[k] = x;
    }
  }
  In *din;
  unsigned long long *dout;
  cudaMalloc(&din, sizeof(in));
  cudaMalloc(&dout, sizeof(out));
  cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
  ops<<<N / 64, 64>>>(din, dout);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0, nonzero = 0;
  for (int i = 0; i < N; i++) {
    reference(in[i], want);
    if (want[12] != 0 && want[14] != 0) nonzero++;
    for (int k = 0; k < OPS; k++)
      if (out[OPS * i + k] != want[k]) {
        if (bad < 10)
          printf("op %d on %#llx %#llx %#llx: got %#llx, expected %#llx\n", k,
                 in[i].a, in[i].b, in[i].c, out[OPS * i + k], want[k]);
        bad++;
      }
  }
  // fma(a, b, -round(a * b)) is the product's rounding error: zero only when
  // the product is exact, so most inputs must give a nonzero result
  if (nonzero < N / 2) {
    printf("only %d of %d fma results are nonzero\n", nonzero, N);
    bad++;
  }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
