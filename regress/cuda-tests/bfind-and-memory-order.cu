// PTX that nvcc 12 emits and GPGPU-Sim once rejected: bfind.shiftamt
// (__ffs, __ffsll) and memory-order/scope qualifiers on atom, ld and st
// (.relaxed, .acquire, .release, .acq_rel, .gpu). Also checks plain bfind,
// including a zero input (no bit set) and negative signed inputs.
//
//   bfind-and-memory-order   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 256;

__device__ unsigned bfind_u32(unsigned a) {
  unsigned d;
  asm("bfind.u32 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}
__device__ unsigned bfind_s32(int a) {
  unsigned d;
  asm("bfind.s32 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}
__device__ unsigned bfind_u64(unsigned long long a) {
  unsigned d;
  asm("bfind.u64 %0, %1;" : "=r"(d) : "l"(a));
  return d;
}
__device__ unsigned bfind_shiftamt_u32(unsigned a) {
  unsigned d;
  asm("bfind.shiftamt.u32 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}

__global__ void bits(const unsigned long long *in, unsigned *out) {
  int i = threadIdx.x;
  unsigned long long v = in[i];
  unsigned *o = out + 6 * i;
  o[0] = bfind_u32((unsigned)v);
  o[1] = bfind_s32((int)v);
  o[2] = bfind_u64(v);
  o[3] = bfind_shiftamt_u32((unsigned)v);
  o[4] = __ffs((int)v);
  o[5] = __ffsll((long long)v);
}

// every thread sets its bit with acq_rel and adds with relaxed ordering;
// thread 0 then publishes a flag with a release store, read back with acquire
__global__ void ordered(unsigned *mask, unsigned *sum, unsigned *flag,
                        unsigned *seen) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.or.b32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(mask + threadIdx.x / 32), "r"(1u << (threadIdx.x % 32))
               : "memory");
  asm volatile("atom.relaxed.cta.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(sum), "r"(threadIdx.x)
               : "memory");
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(flag), "r"(7u)
                 : "memory");
  __syncthreads();
  unsigned f;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(f)
               : "l"(flag)
               : "memory");
  seen[threadIdx.x] = f;
}

static unsigned ref_bfind(unsigned long long a, int msb, bool is_signed) {
  if (is_signed && ((a >> msb) & 1)) a = ~a;
  for (int i = msb; i >= 0; i--)
    if ((a >> i) & 1) return i;
  return 0xffffffff;
}

int main() {
  unsigned long long h_in[N];
  unsigned long long x = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < N; i++) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    h_in[i] = x >> (i % 64);  // values of every width
  }
  h_in[0] = 0;
  h_in[1] = 1;
  h_in[2] = 0xffffffffULL;          // -1 as s32
  h_in[3] = 0x80000000ULL;          // most negative s32
  h_in[4] = 0x8000000000000000ULL;  // only bit 63
  h_in[5] = 0x00000000fffffffeULL;  // -2 as s32

  unsigned long long *d_in;
  unsigned *d_out, *d_mask, *d_sum, *d_flag, *d_seen;
  cudaMalloc(&d_in, sizeof(h_in));
  cudaMalloc(&d_out, 6 * N * sizeof(unsigned));
  cudaMalloc(&d_mask, N / 32 * sizeof(unsigned));
  cudaMalloc(&d_sum, sizeof(unsigned));
  cudaMalloc(&d_flag, sizeof(unsigned));
  cudaMalloc(&d_seen, N * sizeof(unsigned));
  cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
  cudaMemset(d_mask, 0, N / 32 * sizeof(unsigned));
  cudaMemset(d_sum, 0, sizeof(unsigned));
  cudaMemset(d_flag, 0, sizeof(unsigned));
  bits<<<1, N>>>(d_in, d_out);
  ordered<<<1, N>>>(d_mask, d_sum, d_flag, d_seen);

  static unsigned out[6 * N], mask[N / 32], seen[N];
  unsigned sum = 0;
  cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  cudaMemcpy(mask, d_mask, sizeof(mask), cudaMemcpyDeviceToHost);
  cudaMemcpy(&sum, d_sum, sizeof(sum), cudaMemcpyDeviceToHost);
  cudaMemcpy(seen, d_seen, sizeof(seen), cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();

  int bad = 0;
  for (int i = 0; i < N; i++) {
    unsigned long long v = h_in[i];
    unsigned lo = (unsigned)v;
    unsigned b32 = ref_bfind(lo, 31, false);
    unsigned want[6] = {b32,
                        ref_bfind(lo, 31, true),
                        ref_bfind(v, 63, false),
                        b32 == 0xffffffff ? b32 : 31 - b32,
                        lo ? (unsigned)__builtin_ctz(lo) + 1 : 0,
                        v ? (unsigned)__builtin_ctzll(v) + 1 : 0};
    for (int k = 0; k < 6; k++)
      if (out[6 * i + k] != want[k]) {
        if (bad < 10)
          printf("input %#llx, result %d: got %#x, expected %#x\n", v, k,
                 out[6 * i + k], want[k]);
        bad++;
      }
  }
  for (int w = 0; w < N / 32; w++)
    if (mask[w] != 0xffffffff) {
      printf("mask word %d = %#x\n", w, mask[w]);
      bad++;
    }
  if (sum != N * (N - 1) / 2) {
    printf("sum %u, expected %u\n", sum, N * (N - 1) / 2);
    bad++;
  }
  for (int i = 0; i < N; i++)
    if (seen[i] != 7) {
      if (bad < 10) printf("thread %d read flag %u\n", i, seen[i]);
      bad++;
    }
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    bad++;
  }
  printf("%d wrong\n%s\n", bad, bad ? "FAILED" : "PASSED");
  return bad ? 1 : 0;
}
