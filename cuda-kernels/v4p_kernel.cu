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

__global__ void wmma_example(half *a, half *b, float *c,float *d_fp16, int M, int N, int K) {
   //unsigned int start_time=0,end_time=0;
   //start_time=clock();

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   // Bounds checking
   wmma::load_matrix_sync(a_frag, a, K);
   wmma::load_matrix_sync(b_frag, b, K);
   wmma::load_matrix_sync(c_frag, c, N,wmma::mem_col_major);
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   wmma::store_matrix_sync(d_fp16, c_frag, N, wmma::mem_col_major);
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

__global__ void convertInt32ToInt8 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n/4) {
      		out[idx] =(in[4*idx]&0xff)|(in[4*idx+1]&0xff)<<8|(in[4*idx+2]&0xff)<<16|(in[4*idx+3]&0xff)<<24;
   }
}

__global__ void convertInt8ToInt32 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   int shft_amt=8*(idx%4);
   int shft_mask=0xff<<shft_amt;
   if (idx < n) {
      	out[idx]= (in[idx/4]&shft_mask)>>shft_amt;
   }
}

int main(int argc, char* argv[]) {
   int *a_int32;
   int *b_int32;
   int *c_int32;
   int *d_int32;

   int *a_int8;
   int *b_int8;
   
   int  *a_host_wmma;
   int  *b_host_wmma;
   int  *c_host_wmma;
   int  *d_host_wmma;
   int  *d_cal_host_wmma;

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   // Use tensor cores
   cudaErrCheck(cudaMalloc((void**)&a_int32, MATRIX_M * MATRIX_K * sizeof(int)));
   cudaErrCheck(cudaMalloc((void**)&b_int32, MATRIX_K * MATRIX_N * sizeof(int)));
   cudaErrCheck(cudaMalloc((void**)&c_int32, MATRIX_K * MATRIX_N * sizeof(int)));
   cudaErrCheck(cudaMalloc((void**)&d_int32, MATRIX_K * MATRIX_N * sizeof(int)));
   cudaErrCheck(cudaMalloc((void**)&a_int8, MATRIX_M * MATRIX_K * sizeof(int)/4));
   cudaErrCheck(cudaMalloc((void**)&b_int8, MATRIX_K * MATRIX_N * sizeof(int)/4));


   a_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_K * sizeof(int));
   b_host_wmma      = (int *)malloc(MATRIX_K * MATRIX_N * sizeof(int));
   c_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_N * sizeof(int));
   d_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_N * sizeof(int));
   d_cal_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_N * sizeof(int));

   printf("a_int32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_K;n++){
		a_host_wmma[m*MATRIX_K+n]=(m*MATRIX_K+n)%16;
		printf("%d ",a_host_wmma[m*MATRIX_K+n]);
	}
	printf(";\n");
   }
  
   printf("b_int32\n");
   for(int m=0;m<MATRIX_K;m++){
	for(int n=0;n<MATRIX_N;n++){
		b_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n)%2;
		printf("%d ",b_host_wmma[m*MATRIX_N+n]);
	}
		printf(";\n");
   }
   
   printf("c_int32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		c_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n)%2;
		d_cal_host_wmma[m*MATRIX_N+n]=0;
		printf("%d ",c_host_wmma[m*MATRIX_N+n]);
	}
		printf(";\n");
   }
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		for(int k=0;k<MATRIX_K;k++){
			d_cal_host_wmma[m*MATRIX_N+n]+=	a_host_wmma[m*MATRIX_K+k]*b_host_wmma[k*MATRIX_K+n];
		}
		d_cal_host_wmma[m*MATRIX_N+n]+=c_host_wmma[m*MATRIX_N+n];
	}
   }


   cudaErrCheck(cudaMemcpy(a_int32,a_host_wmma,  MATRIX_M * MATRIX_K * sizeof(int), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_int32,b_host_wmma,  MATRIX_K * MATRIX_N * sizeof(int), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(c_int32,c_host_wmma,  MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyHostToDevice));

   convertInt32ToInt8 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_int8, a_int32, MATRIX_M * MATRIX_K);
   convertInt8ToInt32 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_int32, a_int8, MATRIX_M * MATRIX_K);
   //convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);
   //convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_fp16, c_fp32, MATRIX_K * MATRIX_N);
   cudaErrCheck(cudaMemcpy(d_host_wmma, d_int32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));


//AAMIR   printf("\nM = %d, N = %d, K = %d. \n", MATRIX_M, MATRIX_N, MATRIX_K);
//AAMIR   
//AAMIR   printf("Running with wmma...\n");
//AAMIR   cudaErrCheck(cudaEventRecord(startWMMA));
//AAMIR   wmma_example <<< 1, 32>>> (a_fp16, b_fp16, c_fp32, d_fp32 , MATRIX_M, MATRIX_N, MATRIX_K);
//AAMIR   cudaErrCheck(cudaEventRecord(stopWMMA));
//AAMIR   cudaErrCheck(cudaEventSynchronize(stopWMMA));
//AAMIR
//AAMIR   // Error checking
//AAMIR   printf("\nChecking results...\n");
//AAMIR   cudaErrCheck(cudaMemcpy(d_host_wmma, d_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
//AAMIR   
//AAMIR   printf("Results verified: cublas and WMMA agree.\n\n");
//AAMIR   float wmmaTime;
//AAMIR   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
//AAMIR   printf("wmma took %fms\n", wmmaTime);
//AAMIR   
//AAMIR   cudaErrCheck(cudaEventDestroy(startWMMA));
//AAMIR   cudaErrCheck(cudaEventDestroy(stopWMMA));
//AAMIR
//AAMIR   int t=200000;
//AAMIR   while(t-->0);
//AAMIR   printf("D_CALCULATED\n");
//AAMIR
//AAMIR   for(int m=0;m<MATRIX_M;m++){
//AAMIR	for(int n=0;n<MATRIX_N;n++){
//AAMIR		printf("%.2f,",d_cal_host_wmma[m*MATRIX_N+n]);
//AAMIR	}
//AAMIR	printf("\n");
//AAMIR   }
   printf("D_WMMA\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		printf("%d,",d_host_wmma[m*MATRIX_N+n]);
	}
	printf("\n");
    }
//AAMIR   int suc=1;
//AAMIR   for(int m=0;m<MATRIX_M;m++){
//AAMIR	for(int n=0;n<MATRIX_N;n++){
//AAMIR 		if(abs(d_cal_host_wmma[m*MATRIX_N+n]-d_host_wmma[m*MATRIX_N+n])>1) 
//AAMIR		{	
//AAMIR			printf("ERROR:\n");
//AAMIR			suc=0;
//AAMIR		}
//AAMIR	}
//AAMIR   }
//AAMIR   if(suc==1)
//AAMIR	printf("COMPLETED_SUCCESSFULLY\n");
//AAMIR   
   
   cudaErrCheck(cudaFree(a_int32));
   cudaErrCheck(cudaFree(b_int32));
   cudaErrCheck(cudaFree(c_int32));
   cudaErrCheck(cudaFree(d_int32));
   cudaErrCheck(cudaFree(a_int8));
   cudaErrCheck(cudaFree(b_int8));

   free(a_host_wmma);
   free(b_host_wmma);
   free(c_host_wmma);
   free(d_host_wmma);
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


