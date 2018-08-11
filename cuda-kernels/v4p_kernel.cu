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

__global__ void v4p_example(int *a_int32, int *b_int4, int *c,int *d_int32, int M, int N, int K) {
	
	int registers_a[8];
	int registers_b[8];
	int registers_c[8];
	int registers_d[8];
	int register_b;		//contains 8 4bit b elements
   	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.load.a.sync.row.m16n16k16.s32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8],%9;" : 
	"=r"(registers_a[0]), "=r"(registers_a[1]),"=r"(registers_a[2]),"=r"(registers_a[3]),
	"=r"(registers_a[4]),"=r"(registers_a[5]),"=r"(registers_a[6]),"=r"(registers_a[7]):
	"l"(a_int32),"r"(M)
	);
	asm("CPTX_END");
	asm("*/");
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.load.b4.sync.row.m16n16k16.s32 {%0},[%1],%2;" : 
	"=r"(registers_b[0]):
	"l"(b_int4),"r"(M)
	);
	asm("CPTX_END");
	asm("*/");
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.load.c.sync.row.m16n16k16.s32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8],%9;" : 
	"=r"(registers_c[0]), "=r"(registers_c[1]),"=r"(registers_c[2]),"=r"(registers_c[3]),
	"=r"(registers_c[4]),"=r"(registers_c[5]),"=r"(registers_c[6]),"=r"(registers_c[7]):
	"l"(c),"r"(M)
	);
	asm("CPTX_END");
	asm("*/");
	//B4
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.mma.sync.row.row.m16n16k16.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16}, {%17, %18, %19, %20, %21, %22, %23, %24};" : 
	"=r"(registers_d[0]), "=r"(registers_d[1]),"=r"(registers_d[2]),"=r"(registers_d[3]),
	"=r"(registers_d[4]),"=r"(registers_d[5]),"=r"(registers_d[6]),"=r"(registers_d[7]):
	"r"(registers_a[0]),"r"(registers_a[1]),"r"(registers_a[2]),"r"(registers_a[3]),
	"r"(registers_a[4]),"r"(registers_a[5]),"r"(registers_a[6]),"r"(registers_a[7]),
	"r"(registers_b[0]),
	"r"(registers_c[0]),"r"(registers_c[1]),"r"(registers_c[2]),"r"(registers_c[3]),
	"r"(registers_c[4]),"r"(registers_c[5]),"r"(registers_c[6]),"r"(registers_c[7])
	);
	asm("CPTX_END");
	asm("*/");

	//B8
	//asm("CPTX_END");
	//asm("*/");
	//asm("/*");
	//asm("CPTX_BEGIN");
	//asm("vp.mma.sync.row.row.m16n16k16.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17}, {%18, %19, %20, %21, %22, %23, %24, %25};" : 
	//"=r"(registers_d[0]), "=r"(registers_d[1]),"=r"(registers_d[2]),"=r"(registers_d[3]),
	//"=r"(registers_d[4]),"=r"(registers_d[5]),"=r"(registers_d[6]),"=r"(registers_d[7]):
	//"r"(registers_a[0]),"r"(registers_a[1]),"r"(registers_a[2]),"r"(registers_a[3]),
	//"r"(registers_a[4]),"r"(registers_a[5]),"r"(registers_a[6]),"r"(registers_a[7]),
	//"r"(registers_b[0]),"r"(registers_b[1]),
	//"r"(registers_c[0]),"r"(registers_c[1]),"r"(registers_c[2]),"r"(registers_c[3]),
	//"r"(registers_c[4]),"r"(registers_c[5]),"r"(registers_c[6]),"r"(registers_c[7])
	//);
	//asm("CPTX_END");
	//asm("*/");

	//B16
	//asm("CPTX_END");
	//asm("*/");
	//asm("/*");
	//asm("CPTX_BEGIN");
	//asm("vp.mma.sync.row.row.m16n16k16.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, { %20, %21, %22, %23, %24, %25, %26, %27};" : 
	//"=r"(registers_d[0]), "=r"(registers_d[1]),"=r"(registers_d[2]),"=r"(registers_d[3]),
	//"=r"(registers_d[4]),"=r"(registers_d[5]),"=r"(registers_d[6]),"=r"(registers_d[7]):
	//"r"(registers_a[0]),"r"(registers_a[1]),"r"(registers_a[2]),"r"(registers_a[3]),
	//"r"(registers_a[4]),"r"(registers_a[5]),"r"(registers_a[6]),"r"(registers_a[7]),
	//"r"(registers_b[0]),"r"(registers_b[1]),"r"(registers_b[2]),"r"(registers_b[3]),
	//"r"(registers_c[0]),"r"(registers_c[1]),"r"(registers_c[2]),"r"(registers_c[3]),
	//"r"(registers_c[4]),"r"(registers_c[5]),"r"(registers_c[6]),"r"(registers_c[7])
	//);
	//asm("CPTX_END");
	//asm("*/");


	d_int32[0]=registers_d[0];
	d_int32[1]=registers_d[1];
	d_int32[2]=registers_d[2];
	d_int32[3]=registers_d[3];
	d_int32[4]=registers_d[4];
	d_int32[5]=registers_d[5];
	d_int32[6]=registers_d[6];
	d_int32[7]=registers_d[7];
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

__global__ void convertInt32ToInt4 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n/8) {
      		out[idx] =(in[8*idx]&0xf)|(in[8*idx+1]&0xf)<<4|(in[8*idx+2]&0xf)<<8|(in[8*idx+3]&0xf)<<12|
      			  (in[8*idx+4]&0xf)<<16|(in[8*idx+5]&0xf)<<20|(in[8*idx+6]&0xf)<<24|(in[8*idx+7]&0xf)<<28;
   }
}
__global__ void convertInt32ToInt8 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n/4) {
      		out[idx] =(in[4*idx]&0xff)|(in[4*idx+1]&0xff)<<8|(in[4*idx+2]&0xff)<<16|(in[4*idx+3]&0xff)<<24;
   }
}
__global__ void convertInt32ToInt16 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n/2) {
      		out[idx] =(in[2*idx]&0xffff)|(in[2*idx+1]&0xffff)<<16;
   }
}

__global__ void convertInt4ToInt32 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   int shft_amt=4*(idx%8);
   int shft_mask=0xf<<shft_amt;
   if (idx < n) {
      	out[idx]= (in[idx/8]&shft_mask)>>shft_amt;
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
__global__ void convertInt16ToInt32 (int *out, int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   int shft_amt=16*(idx%2);
   int shft_mask=0xffff<<shft_amt;
   if (idx < n) {
      	out[idx]= (in[idx/2]&shft_mask)>>shft_amt;
   }
}

int main(int argc, char* argv[]) {
   int *a_int32;
   int *b_int32;
   int *c_int32;
   int *d_int32;

   int *a_int4;
   int *b_int4;
   int *a_int8;
   int *b_int8;
   int *a_int16;
   int *b_int16;
   
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
   cudaErrCheck(cudaMalloc((void**)&a_int4, MATRIX_M * MATRIX_K * sizeof(int)/8));
   cudaErrCheck(cudaMalloc((void**)&b_int4, MATRIX_K * MATRIX_N * sizeof(int)/8));
   cudaErrCheck(cudaMalloc((void**)&a_int8, MATRIX_M * MATRIX_K * sizeof(int)/4));
   cudaErrCheck(cudaMalloc((void**)&b_int8, MATRIX_K * MATRIX_N * sizeof(int)/4));
   cudaErrCheck(cudaMalloc((void**)&a_int16, MATRIX_M * MATRIX_K * sizeof(int)/2));
   cudaErrCheck(cudaMalloc((void**)&b_int16, MATRIX_K * MATRIX_N * sizeof(int)/2));


   a_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_K * sizeof(int));
   b_host_wmma      = (int *)malloc(MATRIX_K * MATRIX_N * sizeof(int));
   c_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_N * sizeof(int));
   d_host_wmma      = (int *)malloc(MATRIX_M * MATRIX_N * sizeof(int));
   d_cal_host_wmma  = (int *)malloc(MATRIX_M * MATRIX_N * sizeof(int));

   printf("a_int32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_K;n++){
		a_host_wmma[m*MATRIX_K+n]=(m*MATRIX_K+n)%4;
		printf("%d ",a_host_wmma[m*MATRIX_K+n]);
	}
	printf(";\n");
   }
  
   printf("b_int32\n");
   for(int m=0;m<MATRIX_K;m++){
	for(int n=0;n<MATRIX_N;n++){
		b_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n)%4;
		printf("%d ",b_host_wmma[m*MATRIX_N+n]);
	}
		printf(";\n");
   }
   
   printf("c_int32\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		c_host_wmma[m*MATRIX_N+n]=(m*MATRIX_N+n)%4;
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

   #ifdef TEST16
   	convertInt32ToInt16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_int16, a_int32, MATRIX_M * MATRIX_K);
   	convertInt16ToInt32 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_int32, a_int16, MATRIX_M * MATRIX_K);
   	cudaErrCheck(cudaMemcpy(d_host_wmma, d_int32, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyDeviceToHost));
   #endif
   #ifdef TEST8
   	convertInt32ToInt8 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_int8, a_int32, MATRIX_M * MATRIX_K);
  	convertInt8ToInt32 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_int32, a_int8, MATRIX_M * MATRIX_K);
 	cudaErrCheck(cudaMemcpy(d_host_wmma, d_int32, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyDeviceToHost));
   #endif
   #ifdef TEST4
   	convertInt32ToInt4 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_int4, a_int32, MATRIX_M * MATRIX_K);
  	convertInt4ToInt32 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_int32, a_int4, MATRIX_M * MATRIX_K);
 	cudaErrCheck(cudaMemcpy(d_host_wmma, d_int32, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyDeviceToHost));
   #endif
   convertInt32ToInt4 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (b_int4, b_int32, MATRIX_M * MATRIX_K);
   //convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);
   //convertFp32ToFp16 <<< (MATRIX_M * MATRIX_N + 255) / 256, 256 >>> (c_fp16, c_fp32, MATRIX_K * MATRIX_N);


//AAMIR   printf("\nM = %d, N = %d, K = %d. \n", MATRIX_M, MATRIX_N, MATRIX_K);
//AAMIR   
//AAMIR   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   v4p_example <<< 1, 32>>> (a_int32, b_int4, c_int32, d_int32, MATRIX_M, MATRIX_N, MATRIX_K);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   cudaErrCheck(cudaEventSynchronize(stopWMMA));

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
		printf("%x,",d_host_wmma[m*MATRIX_N+n]);
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


