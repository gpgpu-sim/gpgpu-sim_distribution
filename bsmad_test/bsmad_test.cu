#include <stdio.h>
#define SIZE 1024
#define THREADS_PER_BLOCK 32
#define PART_THREADS 1
#define NUM_BLOCKS 1
#define I_PREC 4 
#define O_PREC 4

__global__ void vector_add(int* A, int* B, int* res)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	res[tid] = A[tid] + B[tid];
}	

__global__ void digit_serial_mad(unsigned* i_buffer, unsigned* i_synapse, unsigned* result, unsigned* accum)
{
	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned buffer;
	unsigned synapse;
	if (tid < PART_THREADS)
	{
		buffer = i_buffer[tid];
		synapse = i_synapse[tid];
	}
	
	asm("/*");
	asm("CPTX_BEGIN");
	asm("bsmad.s32 %0, %1, %2, %3, %4, %5, %6, %7, %8;" : "=r"(result[tid]) :
		   	"r"(I_PREC), "r"(O_PREC), "r"(buffer), "r"(0), "r"(0), "r"(0), "r"(synapse), "r"(accum[tid]));
	asm("CPTX_END");
	asm("*/");
}

int main()
{
	// host values
	unsigned *buffer = (unsigned*)malloc(sizeof(unsigned));
	unsigned *synapse = (unsigned*)malloc(sizeof(unsigned));
	unsigned *result = (unsigned*)calloc(THREADS_PER_BLOCK, sizeof(unsigned));
	unsigned *accum = (unsigned*)calloc(THREADS_PER_BLOCK, sizeof(unsigned));
	// assign host values
	*buffer = 0x5000003F; 
	*synapse = 0x00000002;
	*accum = 0;
	// device pointers
	unsigned *d_buffer;
	unsigned *d_synapse;
	unsigned *d_result;
	unsigned *d_accum;
	// allocate device memory
	cudaMalloc(&d_buffer, sizeof(unsigned));
	cudaMalloc(&d_synapse, sizeof(unsigned));
	cudaMalloc(&d_result, sizeof(unsigned));
	cudaMalloc(&d_accum, sizeof(unsigned));
	// copy data to device
	cudaMemcpy(d_buffer, buffer, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(d_synapse, synapse, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, sizeof(unsigned) * THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accum, accum, sizeof(unsigned) * THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
	// call kernel
	digit_serial_mad<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_buffer, d_synapse, d_result, d_accum);
	// copy data back to host
	cudaMemcpy(result, d_result, sizeof(unsigned) * THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
	// read out result
	printf("Result: %#X\n", result[0]);
	// clean up device memory
	cudaFree(d_buffer);
	cudaFree(d_synapse);
	cudaFree(d_result);
	cudaFree(d_accum);
	// clean up host memory
	free(buffer);
	free(synapse);
	free(result);
	free(accum);
}
