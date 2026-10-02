// Kernel launches, async copies and memsets, events and synchronization on
// cudaStreamLegacy and cudaStreamPerThread, the handles for the default
// stream that Thrust and CUB launch on. Results must match the host.
//
//   default-stream-handles   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 256;

__global__ void add(unsigned *x, unsigned v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] += v;
}

static int check(const char *what, cudaError_t e) {
  if (e == cudaSuccess) return 0;
  printf("%s: %s\n", what, cudaGetErrorString(e));
  return 1;
}

int main() {
  static unsigned in[N], out[N];
  for (int i = 0; i < N; i++) in[i] = i;
  unsigned *d;
  int bad = 0;
  cudaEvent_t ev;
  cudaMalloc(&d, sizeof(in));
  cudaEventCreate(&ev);
  cudaStream_t handles[2] = {cudaStreamLegacy, cudaStreamPerThread};
  for (int h = 0; h < 2; h++) {
    cudaStream_t s = handles[h];
    bad += check("memsetAsync", cudaMemsetAsync(d, 0, sizeof(in), s));
    bad += check("memcpyAsync",
                 cudaMemcpyAsync(d, in, sizeof(in), cudaMemcpyHostToDevice, s));
    add<<<N / 64, 64, 0, s>>>(d, 1000 * (h + 1));
    bad += check("launch", cudaGetLastError());
    bad += check("eventRecord", cudaEventRecord(ev, s));
    bad += check("streamWaitEvent", cudaStreamWaitEvent(s, ev, 0));
    bad += check("memcpyAsync back", cudaMemcpyAsync(out, d, sizeof(out),
                                                     cudaMemcpyDeviceToHost, s));
    bad += check("streamSynchronize", cudaStreamSynchronize(s));
    bad += check("streamQuery", cudaStreamQuery(s));
    for (int i = 0; i < N; i++)
      if (out[i] != (unsigned)i + 1000 * (h + 1)) {
        if (bad < 10)
          printf("handle %d, element %d: got %u, expected %u\n", h, i, out[i],
                 i + 1000 * (h + 1));
        bad++;
      }
  }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
