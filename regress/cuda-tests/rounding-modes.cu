// Directed rounding (.rm and .rp) in add, sub, mul, fma, div and sqrt, in
// single and double precision, as used by interval arithmetic (__fadd_rd,
// __dmul_ru, ...). Results must match the host computing under the same
// rounding mode, bit for bit.
//
//   rounding-modes   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>

const int N = 512;
const int OPS = 6;  // add sub mul fma div sqrt

__global__ void single(const float *a, const float *b, const float *c,
                       float *dn, float *up) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[i], y = b[i], z = c[i];
  float *d = dn + OPS * i, *u = up + OPS * i;
  d[0] = __fadd_rd(x, y), u[0] = __fadd_ru(x, y);
  d[1] = __fsub_rd(x, y), u[1] = __fsub_ru(x, y);
  d[2] = __fmul_rd(x, y), u[2] = __fmul_ru(x, y);
  d[3] = __fmaf_rd(x, y, z), u[3] = __fmaf_ru(x, y, z);
  d[4] = __fdiv_rd(x, y), u[4] = __fdiv_ru(x, y);
  d[5] = __fsqrt_rd(fabsf(x)), u[5] = __fsqrt_ru(fabsf(x));
}

__global__ void dbl(const double *a, const double *b, const double *c,
                    double *dn, double *up) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double x = a[i], y = b[i], z = c[i];
  double *d = dn + OPS * i, *u = up + OPS * i;
  d[0] = __dadd_rd(x, y), u[0] = __dadd_ru(x, y);
  d[1] = __dsub_rd(x, y), u[1] = __dsub_ru(x, y);
  d[2] = __dmul_rd(x, y), u[2] = __dmul_ru(x, y);
  d[3] = __fma_rd(x, y, z), u[3] = __fma_ru(x, y, z);
  d[4] = __ddiv_rd(x, y), u[4] = __ddiv_ru(x, y);
  d[5] = __dsqrt_rd(fabs(x)), u[5] = __dsqrt_ru(fabs(x));
}

template <typename T>
static void host_ops(T x, T y, T z, T *r) {
  volatile T vx = x, vy = y, vz = z;
  r[0] = vx + vy;
  r[1] = vx - vy;
  r[2] = vx * vy;
  r[3] = std::fma((T)vx, (T)vy, (T)vz);
  r[4] = vx / vy;
  r[5] = std::sqrt(std::fabs((T)vx));
}

template <typename T>
static int check(const char *name, const T *a, const T *b, const T *c,
                 const T *dn, const T *up, int *differ) {
  int bad = 0;
  for (int i = 0; i < N; i++) {
    T rd[OPS], ru[OPS];
    fesetround(FE_DOWNWARD);
    host_ops(a[i], b[i], c[i], rd);
    fesetround(FE_UPWARD);
    host_ops(a[i], b[i], c[i], ru);
    fesetround(FE_TONEAREST);
    for (int k = 0; k < OPS; k++) {
      const T *gd = dn + OPS * i + k, *gu = up + OPS * i + k;
      if (memcmp(gd, &rd[k], sizeof(T)) || memcmp(gu, &ru[k], sizeof(T))) {
        if (bad < 10)
          printf("%s op %d on %a %a %a: got [%a, %a], expected [%a, %a]\n",
                 name, k, (double)a[i], (double)b[i], (double)c[i], (double)*gd,
                 (double)*gu, (double)rd[k], (double)ru[k]);
        bad++;
      }
      if (*gd != *gu) (*differ)++;
    }
  }
  return bad;
}

int main() {
  static float fa[N], fb[N], fc[N], fdn[OPS * N], fup[OPS * N];
  static double da[N], db[N], dc[N], ddn[OPS * N], dup[OPS * N];
  unsigned long long s = 12345;
  auto rnd = [&]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(s >> 11) / (double)(1ULL << 53);
  };
  for (int i = 0; i < N; i++) {
    da[i] = (rnd() - 0.5) * 1e3;
    db[i] = (rnd() + 0.01) * 7.0;
    dc[i] = (rnd() - 0.5) * 1e-2;
    fa[i] = (float)da[i], fb[i] = (float)db[i], fc[i] = (float)dc[i];
  }
  float *f[5];
  double *d[5];
  for (int k = 0; k < 3; k++) cudaMalloc(&f[k], N * sizeof(float));
  for (int k = 3; k < 5; k++) cudaMalloc(&f[k], OPS * N * sizeof(float));
  for (int k = 0; k < 3; k++) cudaMalloc(&d[k], N * sizeof(double));
  for (int k = 3; k < 5; k++) cudaMalloc(&d[k], OPS * N * sizeof(double));
  cudaMemcpy(f[0], fa, sizeof(fa), cudaMemcpyHostToDevice);
  cudaMemcpy(f[1], fb, sizeof(fb), cudaMemcpyHostToDevice);
  cudaMemcpy(f[2], fc, sizeof(fc), cudaMemcpyHostToDevice);
  cudaMemcpy(d[0], da, sizeof(da), cudaMemcpyHostToDevice);
  cudaMemcpy(d[1], db, sizeof(db), cudaMemcpyHostToDevice);
  cudaMemcpy(d[2], dc, sizeof(dc), cudaMemcpyHostToDevice);
  single<<<N / 128, 128>>>(f[0], f[1], f[2], f[3], f[4]);
  dbl<<<N / 128, 128>>>(d[0], d[1], d[2], d[3], d[4]);
  cudaMemcpy(fdn, f[3], sizeof(fdn), cudaMemcpyDeviceToHost);
  cudaMemcpy(fup, f[4], sizeof(fup), cudaMemcpyDeviceToHost);
  cudaMemcpy(ddn, d[3], sizeof(ddn), cudaMemcpyDeviceToHost);
  cudaMemcpy(dup, d[4], sizeof(dup), cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();

  int differ = 0;
  int bad = check("f32", fa, fb, fc, fdn, fup, &differ) +
            check("f64", da, db, dc, ddn, dup, &differ);
  // inexact results round apart; if none did, the modes were ignored
  if (differ < N) {
    printf("only %d results differ between round-down and round-up\n", differ);
    bad++;
  }
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    bad++;
  }
  printf("%d wrong\n%s\n", bad, bad ? "FAILED" : "PASSED");
  return bad ? 1 : 0;
}
