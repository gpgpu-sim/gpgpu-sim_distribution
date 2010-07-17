#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#define TEXSIZE 4096
texture<float4, 1, cudaReadModeElementType> texData;

void cuda_errcheck(const char *msg) {
  cudaError_t err;
  if ((err = cudaGetLastError()) != cudaSuccess) {
    char errmsg[1024];
    sprintf(errmsg,"CUDA error %s: %s", msg, cudaGetErrorString(err));
  }
}

void cuda_bind_texture_data(const float4 *t) {
    static cudaArray *ct;
    if ( ! ct ) {
      cudaMallocArray(&ct, &texData.channelDesc, TEXSIZE, 1);
      cuda_errcheck("allocating texDataArray");
    }     cudaMemcpyToArray(ct, 0, 0, t, TEXSIZE*sizeof(float4), cudaMemcpyHostToDevice); 
    cuda_errcheck("memcpy to texDataArray");

    texData.normalized = true;
    texData.addressMode[0] = cudaAddressModeClamp;
    texData.addressMode[1] = cudaAddressModeClamp;
    texData.filterMode = cudaFilterModeLinear;

    cudaBindTextureToArray(texData, ct);
    cuda_errcheck("binding texDataArray to texture");
}


__global__ void testKernel(float4 *data, float *coord)
{
    int gsize = blockDim.x * gridDim.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    float norm_coord = (float)(gid + 32) / gsize;

    data[gid] = tex1D(texData, norm_coord);
    coord[gid] = norm_coord;
}

#define NUMTHREADS (TEXSIZE * 4)

int main(int argc, char *argv[])
{
    float4 h_texData[TEXSIZE];

    for (int t = 0; t < TEXSIZE; t++) {
        h_texData[t].x = t + 10000;
        h_texData[t].y = t + 20000;
        h_texData[t].z = t + 30000;
        h_texData[t].w = t + 40000;
    }

    float4 *h_output = new float4[NUMTHREADS];
    float4 *d_output;
    cudaMalloc(&d_output, NUMTHREADS * sizeof(float4));
    cuda_errcheck("output malloc");

    float *h_coord = new float[NUMTHREADS];
    float *d_coord;
    cudaMalloc(&d_coord, NUMTHREADS * sizeof(float));
    cuda_errcheck("coord malloc");

    cudaSetDevice(0);
    cuda_errcheck("device init");

    cuda_bind_texture_data(h_texData);

    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(NUMTHREADS / 256, 1, 1);

    testKernel<<<dimGrid, dimBlock>>> (d_output, d_coord);
    cuda_errcheck("kernel launch");

    cudaThreadSynchronize();

    cudaMemcpy(h_output, d_output, NUMTHREADS * sizeof(float4), cudaMemcpyDeviceToHost);
    cuda_errcheck("output copy");

    cudaMemcpy(h_coord, d_coord, NUMTHREADS * sizeof(float), cudaMemcpyDeviceToHost);
    cuda_errcheck("coord copy");

    for (int t = 0; t < NUMTHREADS; t++) {
        printf("output[%d] (%.06f) = (%5.3f, %5.3f, %5.3f, %5.3f)\n", t, h_coord[t], 
            h_output[t].x, h_output[t].y, h_output[t].z, h_output[t].w);
    }

    return 0;
}

