//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>

////////////////////////////////////////////////////////////////////////
// define kernel block size
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 32
#define BLOCK_Y 4

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include <laplace3d_kernel.cu>

////////////////////////////////////////////////////////////////////////
// declaration, forward
////////////////////////////////////////////////////////////////////////

extern "C" 
void Gold_laplace3d(int NX, int NY, int NZ, float* h_u1, float* h_u2);

void printHelp(void);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

  // 'h_' prefix - CPU (host) memory space

  int    NX, NY, NZ, REPEAT, bx, by, i, j, k, ind, pitch;
  size_t pitch_bytes;
  float  *h_u1, *h_u2, *h_u3, *h_foo, err;

  unsigned int   hTimer;

  // 'd_' prefix - GPU (device) memory space

  float  *d_u1, *d_u2, *d_foo;

  // check command line inputs

  if(cutCheckCmdLineFlag( argc, (const char**)argv, "help")) {
    printHelp();
    return 1;
  }

  if( cutGetCmdLineArgumenti( argc, (const char**)argv, "nx", &NX) ) {
    if( NX <= 99 ) {
      printf("Illegal argument - nx must be greater than 99\n");
      return -1;
    }
  }
  else
    NX = 100;

  if( cutGetCmdLineArgumenti( argc, (const char**)argv, "ny", &NY) ) {
    if( NY <= 99 ) {
      printf("Illegal argument - ny must be greater than 99\n");
      return -1;
    }
  }
  else
    NY = 100;

  if( cutGetCmdLineArgumenti( argc, (const char**)argv, "nz", &NZ) ) {
    if( NZ <= 99 ) {
      printf("Illegal argument - nz must be greater than 99\n");
      return -1;
    }
  }
  else
    NZ = 100;

  if( cutGetCmdLineArgumenti( argc, (const char**)argv, "repeat", &REPEAT) ) {
    if( REPEAT <= 0 ) {
      printf("Illegal argument - repeat must be greater than zero\n");
      return -1;
    }
  }
  else
    REPEAT = 1;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

  // initialise card and timer
  int deviceCount;                                                         
  CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                
  if (deviceCount == 0) {                                                  
      fprintf(stderr, "There is no device.\n");                            
      exit(EXIT_FAILURE);                                                  
  }                                                                        
  int dev;                                                                 
  for (dev = 0; dev < deviceCount; ++dev) {                                
      cudaDeviceProp deviceProp;                                           
      CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));   
      if (deviceProp.major >= 1)                                           
          break;                                                           
  }                                                                        
  if (dev == deviceCount) {                                                
      fprintf(stderr, "There is no device supporting CUDA.\n");            
      exit(EXIT_FAILURE);                                                  
  }                                                                        
  else                                                                     
      CUDA_SAFE_CALL(cudaSetDevice(dev));  
  CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
 
  // allocate memory for arrays

  h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  h_u3 = (float *)malloc(sizeof(float)*NX*NY*NZ);
  CUDA_SAFE_CALL( cudaMallocPitch((void **)&d_u1, &pitch_bytes, sizeof(float)*NX, NY*NZ) );
  CUDA_SAFE_CALL( cudaMallocPitch((void **)&d_u2, &pitch_bytes, sizeof(float)*NX, NY*NZ) );

  pitch = pitch_bytes/sizeof(float);

  // initialise u1
    
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

  // copy u1 to device

  CUT_SAFE_CALL(cutStartTimer(hTimer));
  CUDA_SAFE_CALL( cudaMemcpy2D(d_u1, pitch_bytes,
                               h_u1, sizeof(float)*NX,
                               sizeof(float)*NX, NY*NZ,
                               cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  CUT_SAFE_CALL(cutStopTimer(hTimer));
  printf("\nCopy u1 to device: %f (ms) \n", cutGetTimerValue(hTimer));
  CUT_SAFE_CALL( cutResetTimer(hTimer) );

  // Set up the execution configuration

  bx = 1 + (NX-1)/BLOCK_X;
  by = 1 + (NY-1)/BLOCK_Y;

  dim3 dimGrid(bx,by);
  dim3 dimBlock(BLOCK_X,BLOCK_Y);

  printf("\n dimGrid  = %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z);
  printf(" dimBlock = %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z);

  // Execute GPU kernel

  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  CUT_SAFE_CALL( cutResetTimer(hTimer) );
  CUT_SAFE_CALL( cutStartTimer(hTimer) );

  for (i = 1; i <= REPEAT; ++i) {
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, pitch, d_u1, d_u2);
    d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;   // swap d_u1 and d_u3

    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_CHECK_ERROR("GPU_laplace3d execution failed\n");
  }

  CUT_SAFE_CALL( cutStopTimer(hTimer) );
  printf("\n%dx GPU_laplace3d: %f (ms) \n", REPEAT, cutGetTimerValue(hTimer));

  CUT_SAFE_CALL( cutResetTimer(hTimer) );

  // Read back GPU results

  CUT_SAFE_CALL( cutStartTimer(hTimer) );
  CUDA_SAFE_CALL( cudaMemcpy2D(h_u2, sizeof(float)*NX,
                               d_u1, pitch_bytes,
                               sizeof(float)*NX, NY*NZ,
                               cudaMemcpyDeviceToHost) );
  CUT_SAFE_CALL( cutStopTimer(hTimer) );
  printf("\nCopy u2 to host: %f (ms) \n", cutGetTimerValue(hTimer));
  CUT_SAFE_CALL( cutResetTimer(hTimer) );


  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u2[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // Gold treatment

  CUT_SAFE_CALL( cutResetTimer(hTimer) );
  CUT_SAFE_CALL( cutStartTimer(hTimer) );

  for (int i = 1; i <= REPEAT; ++i) {
    Gold_laplace3d(NX, NY, NZ, h_u1, h_u3);
    h_foo = h_u1; h_u1 = h_u3; h_u3 = h_foo;   // swap h_u1 and h_u3
  }

  CUT_SAFE_CALL( cutStopTimer(hTimer) );
  printf("\n%dx Gold_laplace3d: %f (ms) \n \n", REPEAT, cutGetTimerValue(hTimer));

  // print out corner of array

  /*
  for (k=0; k<3; k++) {
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX + k*NX*NY;
        printf(" %5.2f ", h_u1[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }
  */

  // error check

  err = 0.0;

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        ind = i + j*NX + k*NX*NY;
        err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
      }
    }
  }

  printf("\n rms error = %f \n",sqrt(err/ (float)(NX*NY*NZ)));

 // Release GPU and CPU memory
  printf("CUDA_SAFE_CALL( cudaFree(d_u1) );\n"); fflush(stdout);
  CUDA_SAFE_CALL( cudaFree(d_u1) );
  printf("CUDA_SAFE_CALL( cudaFree(d_u2) );\n"); fflush(stdout);
  CUDA_SAFE_CALL( cudaFree(d_u2) );
  printf("free(h_u1);\n"); fflush(stdout);
  free(h_u1);
  printf("free(h_u2);\n"); fflush(stdout);
  free(h_u2);
  printf("free(h_u3);\n"); fflush(stdout);
  free(h_u3);

  CUT_SAFE_CALL( cutDeleteTimer(hTimer) );
  CUT_EXIT(argc, argv);
}


///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
  printf("Usage:  laplace3d [OPTION]...\n");
  printf("6-point stencil 3D Laplace test \n");
  printf("\n");
  printf("Example: run 100 iterations on a 256x128x128 grid\n");
  printf("./laplace3d --nx=256 --ny=128 --nz=128 --repeat=100\n");

  printf("\n");
  printf("Options:\n");
  printf("--help\t\t\tDisplay this help menu\n");
  printf("--nx=[SIZE]\t\tGrid width\n");
  printf("--ny=[SIZE]\t\tGrid height\n");
  printf("--nz=[SIZE]\t\tGrid depth\n");
  printf("--repeat=[COUNT]\tNumber of repetitions\n");
}
