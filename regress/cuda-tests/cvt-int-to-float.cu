// cvt from 64-bit integers (signed and unsigned) to float and double, and
// from 32-bit integers to float, in all four rounding modes, with values that
// need rounding. Results must match a host reference bit for bit.
//
//   cvt-int-to-float   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

const int N = 256;
// {s64, u64} x {f32, f64} x {rn, rz, rm, rp}, then {s32, u32} x f32 x modes
const int OPS = 24;

__global__ void convert(const unsigned long long *in, unsigned long long *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  long long s = in[i];
  unsigned long long u = in[i], *o = out + OPS * i;
  float f[8];
  double d[8];
  f[0] = __ll2float_rn(s), f[1] = __ll2float_rz(s);
  f[2] = __ll2float_rd(s), f[3] = __ll2float_ru(s);
  f[4] = __ull2float_rn(u), f[5] = __ull2float_rz(u);
  f[6] = __ull2float_rd(u), f[7] = __ull2float_ru(u);
  d[0] = __ll2double_rn(s), d[1] = __ll2double_rz(s);
  d[2] = __ll2double_rd(s), d[3] = __ll2double_ru(s);
  d[4] = __ull2double_rn(u), d[5] = __ull2double_rz(u);
  d[6] = __ull2double_rd(u), d[7] = __ull2double_ru(u);
  int s32 = (int)(u >> 32);
  unsigned u32 = (unsigned)(u >> 32);
  float g[8] = {__int2float_rn(s32),  __int2float_rz(s32),
                __int2float_rd(s32),  __int2float_ru(s32),
                __uint2float_rn(u32), __uint2float_rz(u32),
                __uint2float_rd(u32), __uint2float_ru(u32)};
  for (int k = 0; k < 8; k++) {
    unsigned b;
    memcpy(&b, &f[k], 4);
    o[k] = b;
    memcpy(&o[8 + k], &d[k], 8);
    memcpy(&b, &g[k], 4);
    o[16 + k] = b;
  }
}

// Rounds |value| to p significant bits in integer arithmetic (so the host
// compiler's floating-point code motion can't affect the reference), for
// mode 0 = to nearest even, 1 = toward zero, 2 = down, 3 = up.
static double round_int(unsigned long long mag, bool neg, int p, int mode) {
  if (mag == 0) return 0.0;
  int bits = 64 - __builtin_clzll(mag);
  int shift = bits > p ? bits - p : 0;
  unsigned long long m = mag >> shift;
  unsigned long long rem = shift ? mag & ((1ULL << shift) - 1) : 0;
  if (rem) {
    unsigned long long half = 1ULL << (shift - 1);
    bool up = mode == 0   ? rem > half || (rem == half && (m & 1))
              : mode == 1 ? false
              : mode == 2 ? neg
                          : !neg;
    if (up) m++;
  }
  double r = ldexp((double)m, shift);
  return neg ? -r : r;
}

static void reference(unsigned long long v, unsigned long long *o) {
  long long s = (long long)v;
  bool neg = s < 0;
  unsigned long long smag = neg ? 0 - v : v;
  for (int m = 0; m < 4; m++) {
    float f[2] = {(float)round_int(smag, neg, 24, m),
                  (float)round_int(v, false, 24, m)};
    double d[2] = {round_int(smag, neg, 53, m), round_int(v, false, 53, m)};
    long long s32 = (int)(v >> 32);
    unsigned long long u32 = (unsigned)(v >> 32);
    float g[2] = {(float)round_int(s32 < 0 ? -s32 : s32, s32 < 0, 24, m),
                  (float)round_int(u32, false, 24, m)};
    for (int t = 0; t < 2; t++) {
      unsigned b;
      memcpy(&b, &f[t], 4);
      o[4 * t + m] = b;
      memcpy(&o[8 + 4 * t + m], &d[t], 8);
      memcpy(&b, &g[t], 4);
      o[16 + 4 * t + m] = b;
    }
  }
}

int main() {
  static unsigned long long in[N], out[OPS * N], want[OPS];
  unsigned long long x = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < N; i++) {
    x ^= x << 13, x ^= x >> 7, x ^= x << 17;  // xorshift64
    in[i] = x >> (i % 40);  // magnitudes from 2^24 up to 2^64
  }
  unsigned long long *din, *dout;
  cudaMalloc(&din, sizeof(in));
  cudaMalloc(&dout, sizeof(out));
  cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
  convert<<<N / 64, 64>>>(din, dout);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0, rounded = 0;
  for (int i = 0; i < N; i++) {
    reference(in[i], want);
    if (want[10] != want[11]) rounded++;  // s64 to f64 rm != rp
    for (int k = 0; k < OPS; k++)
      if (out[OPS * i + k] != want[k]) {
        if (bad < 10)
          printf("op %d on %#llx: got %#llx, expected %#llx\n", k, in[i],
                 out[OPS * i + k], want[k]);
        bad++;
      }
  }
  // most inputs have more than 53 significant bits, so double conversions
  // must round; if none did, the test is not testing rounding
  if (rounded < N / 4) {
    printf("only %d of %d conversions to double needed rounding\n", rounded,
           N);
    bad++;
  }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
