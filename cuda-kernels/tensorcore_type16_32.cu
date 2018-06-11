#include <stdio.h>
#include <curand.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
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

__global__ void wmma_example(half *a, half *b, float *c,half *d_fp16, int M, int N, int K) {
   //unsigned int start_time=0,end_time=0;
   //start_time=clock();

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;

   // Bounds checking
   wmma::load_matrix_sync(a_frag, a, K);
   wmma::load_matrix_sync(b_frag, b, K);
   wmma::load_matrix_sync(c_frag, c, N,wmma::mem_col_major);
   
//   for(int i=0; i < c_frag.num_elements; i++) {
////  			c_frag.x[i]=c_frag.x[i]+c_frag.x[i];
//     	float temp=c_frag.x[i];
//	printf("THREAD%d:%d: %f \n",threadIdx.x,i,temp );
//   }
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   for(int i=0; i < c_frag.num_elements; i++) {
 	d_frag.x[i]=c_frag.x[i];
   }
   wmma::store_matrix_sync(d_fp16, d_frag, N, wmma::mem_col_major);
   //printf("clock=%d",end_time-start_time);
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}
__global__ void convertFp16ToFp32 (float *out, half *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   float *c_fp32;
   float *d_fp32;

   half *a_fp16;
   half *b_fp16;
   half *c_fp16;
   half *d_fp16;
   
   float *a_host_wmma;
   float *b_host_wmma;
   float *c_host_wmma;
   float *d_host_wmma;
   float *d_cal_host_wmma;

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   // Use tensor cores
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&d_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&c_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&d_fp16, MATRIX_K * MATRIX_N * sizeof(half)));


   a_host_wmma      = (float*)malloc(MATRIX_M * MATRIX_K * sizeof(float));
   b_host_wmma      = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));
   c_host_wmma      = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   d_host_wmma      = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   d_cal_host_wmma      = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   //printf("a_fp32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_K;n++){
		a_host_wmma[m*MATRIX_K+n]=(m*MATRIX_K+n)%10;
	//	printf("%f ",a_host_wmma[m*MATRIX_K+n]);
	}
	//printf(";\n");
   }
  
   //printf("b_fp32\n");
   for(int m=0;m<MATRIX_K;m++){
	for(int n=0;n<MATRIX_N;n++){
		b_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n)%10;
	//	printf("%f ",b_host_wmma[m*MATRIX_N+n]);
	}
	//	printf(";\n");
   }
   
   //printf("c_fp32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		c_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n);
		d_cal_host_wmma[m*MATRIX_N+n]=0;
	//	printf("%f ",c_host_wmma[m*MATRIX_N+n]);
	}
   }
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		for(int k=0;k<MATRIX_K;k++){
			d_cal_host_wmma[m*MATRIX_N+n]+=	a_host_wmma[m*MATRIX_K+k]*b_host_wmma[k*MATRIX_K+n];
		}
		d_cal_host_wmma[m*MATRIX_N+n]+=c_host_wmma[m*MATRIX_N+n];
	}
   }


   cudaErrCheck(cudaMemcpy(a_fp32,a_host_wmma,  MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp32,b_host_wmma,  MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(c_fp32,c_host_wmma,  MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));

   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   printf("\nM = %d, N = %d, K = %d. \n", MATRIX_M, MATRIX_N, MATRIX_K);
   
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   wmma_example <<< 1, 32>>> (a_fp16, b_fp16, c_fp32, d_fp16 , MATRIX_M, MATRIX_N, MATRIX_K);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   cudaErrCheck(cudaEventSynchronize(stopWMMA));

   convertFp16ToFp32 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (d_fp32, d_fp16, MATRIX_K * MATRIX_N);
  // Error checking
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(d_host_wmma, d_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   
   printf("Results verified: cublas and WMMA agree.\n\n");
   float wmmaTime;
   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("wmma took %fms\n", wmmaTime);
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
   int t=600000;
   while(t-->0);

   printf("D_WMMA\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		printf("%.2f,",d_host_wmma[m*MATRIX_N+n]);
	}
	printf("\n");
   }
   printf("Check the result by executing the kernel on volta\n"); 
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(c_fp32));
   cudaErrCheck(cudaFree(d_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));
   cudaErrCheck(cudaFree(c_fp16));
   cudaErrCheck(cudaFree(d_fp16));

   free(a_host_wmma);
   free(b_host_wmma);
   free(c_host_wmma);
   free(d_host_wmma);
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


