/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
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

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}




#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M (16)
#define MATRIX_N (16)
#define MATRIX_K (16)



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   unsigned int start_time=0,end_time=0;
   // Leading dimensions. Packed with no transpositions.
   start_time=clock();
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid/
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(c_frag, 0.0f);
   
   int i=0;
   int aRow = warpM * WMMA_M;
   int bCol = warpN * WMMA_N;
   int aCol = i;
   int bRow = i;


   // Bounds checking
   if (aRow < M && aCol < K && bRow < K && bCol < N) {
      wmma::load_matrix_sync(a_frag, a+aRow+aCol*lda, lda);
      wmma::load_matrix_sync(b_frag, b+bRow*ldb+bCol, ldb);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      //wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
   }
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;
   wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   end_time=clock();
   printf("clock=%d",end_time-start_time);
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;

   float *c_host_cublas;
   float *c_host_wmma;
   float *a_host_wmma;
   float *b_host_wmma;
   float *c_init_host_wmma;
   
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   
   
   
   // Use tensor cores
  
 
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_wmma      = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_init_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   a_host_wmma      = (float*)malloc(MATRIX_M * MATRIX_K * sizeof(float));
   b_host_wmma      = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));


   
//   printf("a_fp32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_K;n++){
		a_host_wmma[m*MATRIX_K+n]=(m*MATRIX_K+n)%10;
	}
	//printf(";\n");
   }
  // printf("b_fp32\n");
   for(int m=0;m<MATRIX_K;m++){
	for(int n=0;n<MATRIX_N;n++){
		b_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n)%10;
//		printf("%f ",b_host_wmma[m*MATRIX_N+n]);
	}
//	printf(";\n");
   }
   cudaErrCheck(cudaMemcpy(a_fp32,a_host_wmma,  MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp32,b_host_wmma,  MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		c_init_host_wmma[m*MATRIX_N+n]=(m*MATRIX_M+n)%10;
	}
   }
   cudaErrCheck(cudaMemcpy(c, c_init_host_wmma,  MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

   float alpha = 1.0f;
   float beta = 1.0f;


   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< 1, 32>>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  // wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   cudaErrCheck(cudaEventRecord(stopWMMA));


   

   // Error checking
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  // printf("c_host\n");
  // for(int m=0;m<MATRIX_M;m++){
//	for(int n=0;n<MATRIX_N;n++){
//	printf("%f ",c_host_wmma[m*MATRIX_N+n]);
//	}
//	printf(";\n");
  // }
   
      float wmmaTime;
      cudaErrCheck(cudaEventSynchronize(stopWMMA));
      cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
      printf("wmma took %fms\n", wmmaTime);
      //printf("Clock=%d",stopWMMA-startWMMA);
      printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
   
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));

   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_wmma));
   
   free(c_host_wmma);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}


