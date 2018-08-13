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
#define MATRIX_M (256)
#define MATRIX_N (256)
#define MATRIX_K (256)


// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;



__global__ void vp_example(int *a, int *b, int *c, int M, int N, int K ) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   int a_frag[8];
   int b_frag[8];
   int c_frag[8];
   int acc_frag[8];
   
   acc_frag[0]=0;
   acc_frag[1]=0;
   acc_frag[2]=0;
   acc_frag[3]=0;
   acc_frag[4]=0;
   acc_frag[5]=0;
   acc_frag[6]=0;
   acc_frag[7]=0;

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
        // vp::load_matrix_sync(a_frag, a + aRow * lda+ aCol , lda);
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.load.a.sync.row.m16n16k16.s32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8],%9;" : 
	"=r"(a_frag[0]), "=r"(a_frag[1]),"=r"(a_frag[2]),"=r"(a_frag[3]),
	"=r"(a_frag[4]),"=r"(a_frag[5]),"=r"(a_frag[6]),"=r"(a_frag[7]):
	"l"(a+aRow*lda+aCol),"r"(lda)
	);
	asm("CPTX_END");
	asm("*/");
        //vp::load_matrix_sync(b_frag, b + bRow * ldb+ bCol , ldb);
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.load.b16.sync.row.m16n16k16.s32 {%0,%1,%2,%3},[%4],%5;" : 
	"=r"(b_frag[0]),"=r"(b_frag[1]),"=r"(b_frag[2]),"=r"(b_frag[3]):
	"l"(b+bRow*ldb/2+bCol),"r"(ldb/2)
	);
	asm("CPTX_END");
	asm("*/");
 
         // Perform the matrix multiplication
         //vp::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.mma.sync.row.row.m16n16k16.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, { %20, %21, %22, %23, %24, %25, %26,%27};" : 
	"=r"(acc_frag[0]), "=r"(acc_frag[1]),"=r"(acc_frag[2]),"=r"(acc_frag[3]),
	"=r"(acc_frag[4]),"=r"(acc_frag[5]),"=r"(acc_frag[6]),"=r"(acc_frag[7]):
	"r"(a_frag[0]),"r"(a_frag[1]),"r"(a_frag[2]),"r"(a_frag[3]),
	"r"(a_frag[4]),"r"(a_frag[5]),"r"(a_frag[6]),"r"(a_frag[7]),
	"r"(b_frag[0]),"r"(b_frag[1]),"r"(b_frag[2]),"r"(b_frag[3]),
	"r"(acc_frag[0]),"r"(acc_frag[1]),"r"(acc_frag[2]),"r"(acc_frag[3]),
	"r"(acc_frag[4]),"r"(acc_frag[5]),"r"(acc_frag[6]),"r"(acc_frag[7])
	);
	asm("CPTX_END");
	asm("*/");

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
        //vp::load_matrix_sync(c_frag, c + cRow*ldc + cCol , ldc, wmma::mem_row_major);
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.load.c.sync.row.m16n16k16.s32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8],%9;" : 
	"=r"(c_frag[0]), "=r"(c_frag[1]),"=r"(c_frag[2]),"=r"(c_frag[3]),
	"=r"(c_frag[4]),"=r"(c_frag[5]),"=r"(c_frag[6]),"=r"(c_frag[7]):
	"l"(c+cRow*ldc),"r"(ldc)
	);
	asm("CPTX_END");
	asm("*/");


      for(int i=0; i < 8; i++) {
         c_frag[i] =  acc_frag[i] + c_frag[i];
      }

      // Store the output
      //vp::store_matrix_sync(c + cRow *ldc + cCol , c_frag, ldc, wmma::mem_row_major);
	asm("/*");
	asm("CPTX_BEGIN");
	asm("vp.store.d.sync.row.m16n16k16.s32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8},%9;" : 
	:"l"(c+cRow*ldc+cCol),
	"r"(c_frag[0]), "r"(c_frag[1]),"r"(c_frag[2]),"r"(c_frag[3]),
	"r"(c_frag[4]),"r"(c_frag[5]),"r"(c_frag[6]),"r"(c_frag[7]),
	"r"(ldc)
	);
	asm("CPTX_END");
	asm("*/");
   }
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
		a_host_wmma[m*MATRIX_K+n]=(m*MATRIX_K+n)%10;
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
   	convertInt32ToInt4 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (b_int4, b_int32, MATRIX_M * MATRIX_K);
  	convertInt4ToInt32 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_int32, b_int4, MATRIX_M * MATRIX_K);
 	cudaErrCheck(cudaMemcpy(d_host_wmma, d_int32, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyDeviceToHost));
   #endif
   convertInt32ToInt16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (b_int16, b_int32, MATRIX_M * MATRIX_K);

   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 64;
   blockDim.y = 2;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   printf("GRID:X=%d,Y=%d\n",gridDim.x,gridDim.y);
   printf("BLOCK:X=%d,Y=%d\n",blockDim.x,blockDim.y);
   
  
   printf("Running with wmma...\n");
   cudaErrCheck(cudaEventRecord(startWMMA));
   vp_example <<< gridDim, blockDim >>> (a_int32, b_int16, c_int32, MATRIX_M, MATRIX_N, MATRIX_K);
   cudaErrCheck(cudaEventRecord(stopWMMA));
   cudaErrCheck(cudaEventSynchronize(stopWMMA));

   // Error checking
   printf("\nChecking results...\n");
   cudaErrCheck(cudaMemcpy(d_host_wmma, c_int32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   
   float wmmaTime;
   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("wmma took %fms\n", wmmaTime);
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));

   int t=200000;
   while(t-->0);
   printf("D_CALCULATED\n");

   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		printf("%d,",d_cal_host_wmma[m*MATRIX_N+n]);
	}
	printf("\n");
   }
   printf("D_WMMA\n");
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
		printf("%d,",d_host_wmma[m*MATRIX_N+n]);
	}
	printf("\n");
    }
   int suc=1;
   for(int m=0;m<MATRIX_M;m++){
	for(int n=0;n<MATRIX_N;n++){
 		if(abs(d_cal_host_wmma[m*MATRIX_N+n]-d_host_wmma[m*MATRIX_N+n])) 
		{	
			printf("ERROR:\n");
			suc=0;
		}
	}
   }
   if(suc==1)
	printf("COMPLETED_SUCCESSFULLY\n");
   
   
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


