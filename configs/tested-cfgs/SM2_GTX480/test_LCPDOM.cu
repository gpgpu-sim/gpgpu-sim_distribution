/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 */
__global__ void vectorAdd(int *data, int *result) {
  
  int idx = threadIdx.x;
  int i = 0, K = 32;
  while (i < K) {
    int X = data[idx];
    if (X % (2 + i) == 0) {
      result[idx] += X;
    } else if (X == 31) {
      result[idx] += 2 * X;
      break;
    }
    i++;
    data[idx]++;
  }
  result[idx] *= 2;
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 32;
  size_t size = numElements * sizeof(int);
  printf("[Vector addition of %d elements]\n", numElements);

  int *h_data = (int *)malloc(size);
  int *h_result = (int *)malloc(size);

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_data[i] = i;
  }

  int *d_data = NULL;
  err = cudaMalloc((void **)&d_data, size);
  int *d_result = NULL;
  err = cudaMalloc((void **)&d_result, size);

  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 32;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_result);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Check Result
  printf("[ZSY_APP] Check Result\n");
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
  printf("[ZSY_APP] data: ");
  for (int i = 0; i < numElements; ++i) {
    printf("%d ", h_data[i]);
  }
  printf("\n");

  printf("[ZSY_APP] result: ");
  for (int i = 0; i < numElements; ++i) {
    printf("%d ", h_result[i]);
  }
  printf("\n");

  // Free device global memory
  err = cudaFree(d_data);
  err = cudaFree(d_result);

  // Free host memory
  free(h_data);
  free(h_result);

  printf("Done\n");
  return 0;
}
