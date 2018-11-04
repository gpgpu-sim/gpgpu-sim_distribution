#include <cuda_runtime.h>
#include<stdio.h>
__global__ void add(int *a, int *b, int *c, int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
        for(int i=0;i<1000;i++){
            c[idx] = a[idx] + b[idx]+1+c[idx];
        }
}
__host__
int main(){
    const int N=10000;
    int *a,*b,*c,*da,*db,*dc;
    a=new int[N];
    b=new int[N];
    c=new int[N];
    for(int i=0;i<N;i++){
        a[i]=1;
        b[i]=2;
    }

    cudaMalloc(&da,sizeof(int)*N);
    cudaMalloc(&db,sizeof(int)*N);
    cudaMalloc(&dc,sizeof(int)*N);


    cudaMemcpy(da,a,sizeof(int)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,sizeof(int)*N,cudaMemcpyHostToDevice);

    int blockNum=(N+1023)/1024;
    add<<<blockNum,1024>>>(da,db,dc,N);

    cudaMemcpy(c,dc,sizeof(int)*N,cudaMemcpyDeviceToHost);

    for(int i=0;i<10;i++){
        printf("%d\n",c[i]);
    }
    return 0;


}