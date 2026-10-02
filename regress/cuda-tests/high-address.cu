// Device memory above 4 GB of address space. The simulated heap starts at
// 0xC0000000, so a buffer allocated after a 4 GB pad lies above 4 GB. Plain and
// vector loads and stores must reach it (they once kept only the
// low 32 bits of the address while atomics kept all 64), and shared memory
// must keep working next to them.
//
//   high-address [pad MB]   default 4096; prints PASSED or FAILED, exits 1 on
//                           failure
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

const int N = 1 << 16;
const int BLOCK = 256;

__global__ void fill(unsigned *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  b[i] = i * 3 + 1;
}

// vector load and store
__global__ void copy4(const uint4 *src, uint4 *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N / 4) {
    uint4 v = src[i];
    v.x += 1, v.y += 1, v.z += 1, v.w += 1;
    dst[i] = v;
  }
}

// shared memory: sum each block, then add the block sum atomically
__global__ void sum(const unsigned *b, unsigned long long *total) {
  __shared__ unsigned s[BLOCK];
  s[threadIdx.x] = b[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();
  for (int k = BLOCK / 2; k > 0; k /= 2) {
    if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(total, (unsigned long long)s[0]);
}

int main(int argc, char **argv) {
  size_t pad_mb = argc > 1 ? atoll(argv[1]) : 4096;
  void *pad = 0;
  if (pad_mb && cudaMalloc(&pad, pad_mb << 20) != cudaSuccess) {
    printf("cannot allocate the %zu MB pad\nFAILED\n", pad_mb);
    return 1;
  }
  unsigned *a, *b;
  unsigned long long *total;
  cudaMalloc(&a, N * sizeof(unsigned));
  cudaMalloc(&b, N * sizeof(unsigned));
  cudaMalloc(&total, sizeof(*total));
  printf("pad %zu MB at %p; buffers at %p and %p\n", pad_mb, pad, (void *)a,
         (void *)b);
  cudaMemset(total, 0, sizeof(*total));
  fill<<<N / BLOCK, BLOCK>>>(a);
  copy4<<<N / 4 / BLOCK, BLOCK>>>((const uint4 *)a, (uint4 *)b);
  sum<<<N / BLOCK, BLOCK>>>(b, total);

  static unsigned h[N];
  unsigned long long t = 0, want = 0;
  cudaMemcpy(h, b, sizeof(h), cudaMemcpyDeviceToHost);
  cudaMemcpy(&t, total, sizeof(t), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < N; i++) {
    bad += h[i] != (unsigned)(i * 3 + 2);
    want += i * 3 + 2;
  }
  bool ok = bad == 0 && t == want;
  printf("%d of %d values wrong; sum %llu (expected %llu)\n%s\n", bad, N, t,
         want, ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
