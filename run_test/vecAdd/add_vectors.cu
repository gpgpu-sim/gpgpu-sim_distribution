// Courtesy of https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/
// REMOVE ME: Uncommnet the code only upon full implementation or get seg-fault
 
#include <iostream>
#include <math.h>
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(void)
{
  int N = 1<<20;
  float *x, *y;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

#ifdef PREF
  // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

  cudaStream_t stream3;
  cudaStreamCreate(&stream3);

  cudaMemPrefetchAsync(x, N*sizeof(float), device, stream1);
  cudaMemPrefetchAsync(y, N*sizeof(float), device, stream2);
#endif
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

#ifdef PREF
  add<<<numBlocks, blockSize, 0, stream3>>>(N, x, y);
#else
  add<<<numBlocks, blockSize>>>(N, x, y);
#endif
 
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}

