/*==========================================================================
                                  S T O R E  G P U

* Copyright (c) 2008, NetSysLab at the University of British Columbia
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the University nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY NetSysLab ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NetSysLab BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
  Main entry of the library.


==========================================================================*/

/*==========================================================================

                                  INCLUDES

==========================================================================*/
// system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// project
#include <cutil.h>
#include <cust.h>
#include <md5_cpu.h>
#include <sha1_cpu.h>
#include <storeGPU.h>
#include <storeCPU.h>

// kernels
#include <md5_kernel.cu>
#include <sha1_kernel.cu>

/*==========================================================================

                             DATA DECLARATIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                              TYPE DEFINITIONS
--------------------------------------------------------------------------*/

// defines a GPU device properties
typedef struct sg_dev_prop {
  int max_thread_per_block;
  int max_grid_size;
  int global_mem_size;
  int warp_size;
} sg_dev_prop_type;

// defines an execution context used to lunch a kernel.
typedef struct sg_exec_context {
  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int chunk_size;
  int total_size;
  int pad_size;
} sg_exec_context_type;


/*--------------------------------------------------------------------------
                             FUNCTION PROTOTYPES
--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
                                  CONSTANTS
--------------------------------------------------------------------------*/



/*--------------------------------------------------------------------------
                              GLOBAL VARIABLES
--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------
                                    MACROS
--------------------------------------------------------------------------*/

#define GET_REAL_CHUNK_SIZE(chunk_size) ((chunk_size) - 12)

/*==========================================================================

                                  FUNCTIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                                    LOCAL FUNCTIONS
--------------------------------------------------------------------------*/
#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
/*===========================================================================

FUNCTION SG_GET_DEV_PROP

DESCRIPTION
  Probes the device for its properties

DEPENDENCIES
  None

RETURN VALUE
  device information

===========================================================================*/
static void sg_get_dev_prop(sg_dev_prop_type* dev_prop) {

  struct cudaDeviceProp prop;
  int dev;

  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  printf("\n== Device Properties ==\n");
  printf("Max global memory    : %d\n", prop.totalGlobalMem);
  printf("Registers per block  : %d\n", prop.regsPerBlock);
  printf("Warp size            : %d\n", prop.warpSize);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Block Dimensions     : %d, %d, %d\n", 
	 prop.maxThreadsDim[0],
	 prop.maxThreadsDim[1],
	 prop.maxThreadsDim[2]);
  printf("Grid Dimensions      : %d, %d, %d\n", 
	 prop.maxGridSize[0],
	 prop.maxGridSize[1],
	 prop.maxGridSize[2]);

  dev_prop->max_thread_per_block = prop.maxThreadsDim[0]; 
  dev_prop->max_grid_size        = prop.maxGridSize[0]; 
  dev_prop->global_mem_size      = prop.totalGlobalMem; 
  dev_prop->warp_size            = prop.warpSize;

}

#ifdef FEATURE_MAXIMIZE_NUM_OF_THREADS
/*===========================================================================

FUNCTION SG_GET_EXEC_CONTEXT

DESCRIPTION
  sets the required chunk size, thread per block and number of blocks
  needed for kernel execution according to client buffer size.

DEPENDENCIES
  None

RETURN VALUE
  execution context

===========================================================================*/
static sg_status_type sg_get_exec_context(int size, int hash_size, 
					  sg_exec_context_type* ctx){

  sg_dev_prop_type dev_prop;
  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int chunk_size;
  int pad_size;

  int total_chunks = 0;
  int found = 0;
  int index = 1;


  //**** Get device information ****//
  sg_get_dev_prop(&dev_prop);

  //**** Determine the execution context ****//
  /* The algorithm will try to determine the context by minimizing chunk 
   * size and maximizing total number of threads 
   * TODO: May be we can do better here 
   */
  while ( !found ) {
    // Set chunk size
    chunk_size = GET_REAL_CHUNK_SIZE(BASIC_CHUNK_SIZE * index);

    if ( chunk_size > MAX_CHUNK_SIZE )
      break;

    // Calculate the required padding for this chunk size
    pad_size = ((size % chunk_size) == 0) ? 0 : 
      chunk_size - (size % chunk_size);

    // total number of chunks required if we are going to use this chunk size
    total_chunks = (pad_size == 0) ? size / chunk_size : 
      (size / chunk_size) + 1;

    if ( total_chunks <= MAX_NUM_OF_THREADS ) {
      // Got it, this is the minimum chunk size we can use. Now determine the
      // threads and blocks numbers.
      total_threads = total_chunks;

      // Get block and grid sizes
      if (total_chunks <= MAX_THREADS_PER_BLOCK ) {
	threads_per_block = total_chunks;
	blocks_per_grid = 1;

      } else {
      threads_per_block = MAX_THREADS_PER_BLOCK;
      blocks_per_grid = ((total_threads % threads_per_block) == 0) ? 
	(total_threads/threads_per_block) : 
	(total_threads/threads_per_block) + 1;

      }

      if ( dev_prop.global_mem_size > 
	   (total_threads * (chunk_size + hash_size) + hash_size) )
	found = 1;

    }
    index++;

  }
  
  //**** Did we find a solution? ****//
  if ( !found )
    return SG_ERR_DEV_MEM_OVERFLOW;

  //**** Fill the struct with the solution ****//
  ctx->threads_per_block = threads_per_block;
  ctx->blocks_per_grid = blocks_per_grid;
  ctx->total_threads = total_threads;
  ctx->total_size = size + pad_size;
  ctx->chunk_size = chunk_size;
  ctx->pad_size = pad_size;

  return SG_OK;
  
}

#else /* FEATURE_MAXIMIZE_NUM_OF_THREADS */
/*===========================================================================

FUNCTION SG_GET_EXEC_CONTEXT

DESCRIPTION
  sets the required chunk size, thread per block and number of blocks
  needed for kernel execution according to client buffer size.

DEPENDENCIES
  None

RETURN VALUE
  execution context

===========================================================================*/
static sg_status_type sg_get_exec_context(int size, int hash_size, 
					  sg_exec_context_type* ctx){

  sg_dev_prop_type dev_prop;
  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int chunk_size;
  int pad_size;

  int total_chunks = 0;
  int found = 0;

  int index = MAX_CHUNK_SIZE / BASIC_CHUNK_SIZE;


  //**** Get device information ****//
  sg_get_dev_prop(&dev_prop);

  //**** Determine the execution context ****//
  /* The algorithm will try to determine the context by minimizing chunk 
   * size and maximizing total number of threads 
   * TODO: May be we can do better here 
   */
  while ( 1 ) {
    // Set chunk size
    chunk_size = GET_REAL_CHUNK_SIZE(BASIC_CHUNK_SIZE * index);

    // don't go less than minimum chunk size
    if ( chunk_size < GET_REAL_CHUNK_SIZE(BASIC_CHUNK_SIZE) )
      break;

    // Calculate the required padding for this chunk size
    pad_size = ((size % chunk_size) == 0) ? 0 : 
      chunk_size - (size % chunk_size);

    // total number of chunks required if we are going to use this chunk size
    total_chunks = (pad_size == 0) ? size / chunk_size : 
      (size / chunk_size) + 1;


    // don't go beyond the maximum number of threads or maximum global memory
    // TODO: it seems that the kernel breaks way before reaching the maximum
    //       global memory size (around 94MByte input plus the required 
    //       scratch space)
    if (( total_chunks > MAX_NUM_OF_THREADS) || 
	( dev_prop.global_mem_size < (total_chunks * 
				      (chunk_size + hash_size) + 
				      hash_size)))
      break;

    // each thread will take care of one chunk
    total_threads = total_chunks;

    
    // Get block and grid sizes
    if (total_chunks <= MAX_THREADS_PER_BLOCK ) {
      threads_per_block = total_chunks;
      blocks_per_grid = 1;
      
    } else {
      threads_per_block = MAX_THREADS_PER_BLOCK;
      blocks_per_grid = ((total_threads % threads_per_block) == 0) ? 
	(total_threads/threads_per_block) : 
	(total_threads/threads_per_block) + 1;

    }
    found = 1;
    
    if( total_threads > NUM_OF_MULTIPROCESSORS * dev_prop.warp_size)
      break;
    
    index--;

  }
  
  //**** Did we find a solution? ****//
  if ( !found )
    return SG_ERR_DEV_MEM_OVERFLOW;

  //**** Fill the struct with the solution ****//
  ctx->threads_per_block = threads_per_block;
  ctx->blocks_per_grid = blocks_per_grid;
  ctx->total_threads = total_threads;
  ctx->total_size = size + pad_size;
  ctx->chunk_size = chunk_size;
  ctx->pad_size = pad_size;

  return SG_OK;
  
}
#endif /* FEATURE_MAXIMIZE_NUM_OF_THREADS */

/*===========================================================================

FUNCTION SG_GET_OVERLAP_EXEC_CONTEXT

DESCRIPTION
  sets the required chunk size, thread per block and number of blocks
  needed for kernel execution according to client buffer size, offset
  and block size.

DEPENDENCIES
  None

RETURN VALUE
  execution context

===========================================================================*/
static sg_status_type sg_get_overlap_exec_context(int size, int offset, 
					         int block_size, int hash_size, 
						  sg_exec_context_type* ctx) {

  sg_dev_prop_type dev_prop;
  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int total_size;
  int pad_size;


  //**** Get device information ****//
  sg_get_dev_prop(&dev_prop);


  //**** Get the total number of threads required ****//
  total_threads = (size + offset - block_size) / offset;
  total_threads = ((size + offset - block_size) % offset) != 0 ? 
    total_threads + 1 : total_threads;


  //**** Get the required padding for the last block ****//
  pad_size = ((total_threads - 1) * offset + block_size) - size;


  //**** threads and blocks ****//
  if( total_threads > MAX_THREADS_PER_BLOCK ) {

    threads_per_block = MAX_THREADS_PER_BLOCK;
    blocks_per_grid = (total_threads % MAX_THREADS_PER_BLOCK) == 0 ?
      (total_threads / MAX_THREADS_PER_BLOCK) : 
      (total_threads / MAX_THREADS_PER_BLOCK) + 1;
  } else {

    threads_per_block = total_threads;
    blocks_per_grid = 1;
  }

  total_size = size + pad_size;
  
  //**** Check for device memory overflow ****//  
  if( dev_prop.global_mem_size < (total_size + (hash_size * total_threads))) {
    return SG_ERR_DEV_MEM_OVERFLOW;
  }


  //**** Fill the struct with the solution ****//
  ctx->threads_per_block = threads_per_block;
  ctx->blocks_per_grid = blocks_per_grid;
  ctx->total_threads = total_threads;
  ctx->total_size = total_size;
  ctx->chunk_size = block_size;
  ctx->pad_size = pad_size;

  return SG_OK;

}
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

/*===========================================================================

FUNCTION SG_PRINT_EXEC_CONTEXT

DESCRIPTION
  Prints out the passed execution context structure

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
static void sg_print_exec_context( sg_exec_context_type* ctx ) {
  printf("\n== GPU Execution Context ==\n");
  printf("Threads       : %d\n", ctx->threads_per_block);
  printf("Blocks        : %d\n", ctx->blocks_per_grid);
  printf("Total Threads : %d\n", ctx->total_threads);
  printf("Total size    : %d\n", ctx->total_size);
  printf("Chunk Size    : %d\n", ctx->chunk_size);
  printf("Padding       : %d\n\n", ctx->pad_size);
}

/*--------------------------------------------------------------------------
                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/



/*===========================================================================

FUNCTION SG_INIT

DESCRIPTION
  Initialize the library

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void sg_init( ) {

  char *buffer;

  //**** Utility library initialization ****//
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


  //**** force runtime initialization (CUDA ref. manual for more info.) ****//
  cudaMallocHost( (void**) &buffer, 4 );
  cudaFreeHost( buffer );

}

/*===========================================================================

FUNCTION SG_MALLOC

DESCRIPTION
  Allocate the required memory size.

DEPENDENCIES
  None

RETURN VALUE
  pointer to the reseved buffer

===========================================================================*/
void* sg_malloc(unsigned int size){
  
  void* buffer;
  
#ifdef FEATURE_PINNED_MODE
  cudaMallocHost( (void**) &buffer, size );
#else
  
  buffer = malloc( size );
#endif /* FEATURE_PINNED_MODE */
  
  return buffer;
}

/*===========================================================================

FUNCTION SG_FREE

DESCRIPTION
  Free the allocated buffer.

DEPENDENCIES
  None

RETURN VALUE
  pointer to the reseved buffer

===========================================================================*/
void sg_free(void* buffer){

	
#ifdef FEATURE_PINNED_MODE  
  cudaFreeHost(buffer );
#else
  free( buffer );
#endif
  
}

/*===========================================================================

FUNCTION SG_MD5

DESCRIPTION
  Returns the MD5 hash

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_md5(unsigned char* buffer, int size, 
		      unsigned char** output, int* output_size,
		      sg_time_breakdown_type* time_breakdown) {
  
  //**** Variable Declarations ****//
  sg_exec_context_type exec_context;
  sg_status_type status = SG_OK;
  unsigned char* d_scratchData;
  unsigned char* h_scratchData;
  unsigned char* d_input;
  unsigned int timer; 


  //**** create the timer ****//
  timer = 0;  
  CUT_SAFE_CALL( cutCreateTimer( &timer));
  

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  status = sg_get_exec_context(size, MD5_HASH_SIZE, &exec_context);
  if ( status != SG_OK ) {
    printf("Global memory overflow\n");
    return status;
  }
#else

  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;

#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */
  sg_print_exec_context(&exec_context);  



  //**** device memory allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* allocate input data space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_input, exec_context.total_size));

  /* allocate scratch space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_scratchData, 
			    MD5_HASH_SIZE * exec_context.total_threads));  

  /* stop the timer (device memory allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->device_mem_alloc_time = cutGetTimerValue(timer);



  //**** scratch buffer allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* allocate buffer for the results */
  cudaMallocHost((void **)&h_scratchData, MD5_HASH_SIZE * 
		 exec_context.total_threads);

  /* stop the timer (scratch buffer allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->host_output_buffer_alloc_time = cutGetTimerValue(timer);



  //**** start timer for data copy in timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* move data to the device memory */
  CUDA_SAFE_CALL(cudaMemcpy( d_input, buffer, size, 
			     cudaMemcpyHostToDevice));

  /* stop the timer (copy in) */
  CUT_SAFE_CALL( cutStopTimer( timer));
  time_breakdown->copy_in_time = cutGetTimerValue( timer );



  //**** setup execution parameters ****//
  dim3  block( exec_context.threads_per_block );
  dim3  grid( exec_context.blocks_per_grid );



  //**** start timer for kernel execution timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));


  /* execute the kernel */
  md5<<< grid, block >>>(d_input, exec_context.chunk_size, 
			 exec_context.total_threads,
			 exec_context.pad_size,
			 d_scratchData);

  // check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");
  
  /* wait till the kernel finishes execution */
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  /* stop the timer (kernel execution) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->exec_time = cutGetTimerValue(timer);



  //**** start timer for output copy out timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* get the results from the device */
  CUDA_SAFE_CALL(cudaMemcpy(h_scratchData,
			    d_scratchData,
			    MD5_HASH_SIZE * exec_context.total_threads,
			    cudaMemcpyDeviceToHost));

  /* stop the timer (output copy out) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->copy_out_time = cutGetTimerValue(timer);



  //**** start timer for last hasing stage timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* will do the last hshing stage on the CPU */
  sc_md5_standard(h_scratchData, MD5_HASH_SIZE * exec_context.total_threads, 
		  output );

  /* stop the timer (last stage) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->last_stage_time = cutGetTimerValue(timer);

  //**** free allocated memory ****//
  CUDA_SAFE_CALL(cudaFree(d_input));
  CUDA_SAFE_CALL(cudaFree(d_scratchData));
  cudaFreeHost(h_scratchData);

  *output_size = MD5_HASH_SIZE;


  return status;

}

/*===========================================================================

FUNCTION SG_MD5_OVERLAP

DESCRIPTION
  Returns the MD5 hash of each block for the provided buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_md5_overlap(unsigned char* buffer, int size,
			      int block_size, int offset,
			      unsigned char** output, int* output_size,
			      sg_time_breakdown_type* time_breakdown) {

  
  //**** Variable Declarations ****//
  sg_exec_context_type exec_context;
  sg_status_type status = SG_OK;
  unsigned char* d_output;
  unsigned char* d_input;
  unsigned int timer;



  //**** create the timer ****//
  timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  status = sg_get_overlap_exec_context(size, offset, block_size, 
			       MD5_HASH_SIZE, &exec_context);
  if ( status != SG_OK ) {
    printf("Global memory overflow\n");
    return status;
  }
#else
  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */
  sg_print_exec_context(&exec_context);



  //**** start timer for device memory allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* allocate input space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_input, exec_context.total_size));
  
  /* allocate output space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_output, 
			    MD5_HASH_SIZE * exec_context.total_threads));

  /* stop the timer (memory allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->device_mem_alloc_time = cutGetTimerValue(timer);



  //**** start timer for output memory allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /**output = (unsigned char*) sg_malloc(MD5_HASH_SIZE * 
				       exec_context.total_threads);*/
  cudaMallocHost( (void**) output, MD5_HASH_SIZE * 
				       exec_context.total_threads );

  /* stop the timer (output buffer allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->host_output_buffer_alloc_time = cutGetTimerValue(timer);



  //**** start timer for data copy in timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* move data to the device memory */
  CUDA_SAFE_CALL(cudaMemcpy(d_input, buffer, size, 
			    cudaMemcpyHostToDevice));

  /* stop the timer (copy in) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->copy_in_time = cutGetTimerValue(timer);



  //**** setup execution parameters ****//
  dim3  block( exec_context.threads_per_block );
  dim3  grid( exec_context.blocks_per_grid );



  //**** start timer for kernel execution timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* execute the kernel */
  md5_overlap<<< grid, block >>>(d_input, exec_context.chunk_size,
				 offset, exec_context.total_threads,
				 exec_context.pad_size, d_output);
  
  // check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");

  /* wait till the kernel finishes execution */
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  /* stop the timer (kernel execution) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->exec_time = cutGetTimerValue(timer);



  //**** start timer for output copy out timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* get the result from the device */
  CUDA_SAFE_CALL(cudaMemcpy(*output,
			    d_output,
			    MD5_HASH_SIZE * exec_context.total_threads,
			    cudaMemcpyDeviceToHost));

  /* stop the timer (output copy out) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->copy_out_time = cutGetTimerValue(timer);



  //**** free allocated memory ****//
  CUDA_SAFE_CALL(cudaFree(d_input));
  CUDA_SAFE_CALL(cudaFree(d_output));

  *output_size = MD5_HASH_SIZE * exec_context.total_threads;

  return status;
}

/*===========================================================================

FUNCTION SG_SHA1

DESCRIPTION
  Returns the SHA1 hash of a the provided buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_sha1(unsigned char* buffer, int size, 
		       unsigned char** output, int* output_size,
		       sg_time_breakdown_type* time_breakdown) {
  
  //**** Variable Declarations ****//
  sg_exec_context_type exec_context;
  sg_status_type status = SG_OK;
  unsigned char* d_scratchData;
  unsigned char* h_scratchData;
  unsigned char* d_input;
  unsigned int timer; 


  //**** create the timer ****//
  timer = 0;  
  CUT_SAFE_CALL( cutCreateTimer( &timer));
  

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  status = sg_get_exec_context(size, SHA1_HASH_SIZE, &exec_context);
  if ( status != SG_OK ) {
    printf("Global memory overflow\n");
    return status;
  }
#else

  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;

#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */
  sg_print_exec_context(&exec_context);  



  //**** device memory allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* allocate input data space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_input, exec_context.total_size));

  /* allocate scratch space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_scratchData, 
			    SHA1_HASH_SIZE * exec_context.total_threads));  

  /* stop the timer (device memory allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->device_mem_alloc_time = cutGetTimerValue(timer);



  //**** scratch buffer allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* allocate buffer for the results */
  cudaMallocHost((void**)&h_scratchData, SHA1_HASH_SIZE * 
		 exec_context.total_threads);

  /* stop the timer (scratch buffer allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->host_output_buffer_alloc_time = cutGetTimerValue(timer);



  //**** start timer for data copy in timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* move data to the device memory */
  CUDA_SAFE_CALL(cudaMemcpy( d_input, buffer, size, 
			     cudaMemcpyHostToDevice));

  /* stop the timer (copy in) */
  CUT_SAFE_CALL( cutStopTimer( timer));
  time_breakdown->copy_in_time = cutGetTimerValue( timer );



  //**** setup execution parameters ****//
  dim3  block( exec_context.threads_per_block );
  dim3  grid( exec_context.blocks_per_grid );



  //**** start timer for kernel execution timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));


  /* execute the kernel */
  sha1<<< grid, block >>>(d_input, exec_context.chunk_size, 
			 exec_context.total_threads,
			 exec_context.pad_size,
			 d_scratchData);

  // check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");
  
  /* wait till the kernel finishes execution */
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  /* stop the timer (kernel execution) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->exec_time = cutGetTimerValue(timer);



  //**** start timer for output copy out timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* get the results from the device */
  CUDA_SAFE_CALL(cudaMemcpy(h_scratchData,
			    d_scratchData,
			    SHA1_HASH_SIZE * exec_context.total_threads,
			    cudaMemcpyDeviceToHost));

  /* stop the timer (output copy out) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->copy_out_time = cutGetTimerValue(timer);



  //**** start timer for last hasing stage timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* will do the last hshing stage on the CPU */
  sc_sha1_standard(h_scratchData, SHA1_HASH_SIZE * exec_context.total_threads, 
		  output );

  /* stop the timer (last stage) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->last_stage_time = cutGetTimerValue(timer);


  //**** free allocated memory ****//
  CUDA_SAFE_CALL(cudaFree(d_input));
  CUDA_SAFE_CALL(cudaFree(d_scratchData));
  cudaFreeHost(h_scratchData);

  *output_size = SHA1_HASH_SIZE;

  return status;

}


/*===========================================================================

FUNCTION SG_SHA1_OVERLAP

DESCRIPTION
  Returns the SHA1 hash of each block for the provided buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_sha1_overlap(unsigned char* buffer, int size,
			      int block_size, int offset,
			      unsigned char** output, int* output_size,
			      sg_time_breakdown_type* time_breakdown) {

  
  //**** Variable Declarations ****//
  sg_exec_context_type exec_context;
  sg_status_type status = SG_OK;
  unsigned char* d_output;
  unsigned char* d_input;
  unsigned int timer;



  //**** create the timer ****//
  timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  status = sg_get_overlap_exec_context(size, offset, block_size, 
			       SHA1_HASH_SIZE, &exec_context);
  if ( status != SG_OK ) {
    printf("Global memory overflow\n");
    return status;
  }
#else
  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */
  sg_print_exec_context(&exec_context);



  //**** start timer for device memory allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* allocate input space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_input, exec_context.total_size));
  
  /* allocate output space */
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_output, 
			    SHA1_HASH_SIZE * exec_context.total_threads));

  /* stop the timer (memory allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->device_mem_alloc_time = cutGetTimerValue(timer);



  //**** start timer for output memory allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /**output = (unsigned char*) sg_malloc(SHA1_HASH_SIZE * 
				       exec_context.total_threads);*/
    cudaMallocHost( (void**) output, SHA1_HASH_SIZE * 
				       exec_context.total_threads );

  /* stop the timer (output buffer allocation) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->host_output_buffer_alloc_time = cutGetTimerValue(timer);



  //**** start timer for data copy in timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* move data to the device memory */
  CUDA_SAFE_CALL(cudaMemcpy(d_input, buffer, size, 
			    cudaMemcpyHostToDevice));

  /* stop the timer (copy in) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->copy_in_time = cutGetTimerValue(timer);



  //**** setup execution parameters ****//
  dim3  block( exec_context.threads_per_block );
  dim3  grid( exec_context.blocks_per_grid );



  //**** start timer for kernel execution timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* execute the kernel */
  sha1_overlap<<< grid, block >>>(d_input, exec_context.chunk_size,
				 offset, exec_context.total_threads,
				 exec_context.pad_size, d_output);
  
  // check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");

  /* wait till the kernel finishes execution */
  CUDA_SAFE_CALL(cudaThreadSynchronize());

  /* stop the timer (kernel execution) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->exec_time = cutGetTimerValue(timer);



  //**** start timer for output copy out timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));

  /* get the result from the device */
  CUDA_SAFE_CALL(cudaMemcpy(*output,
			    d_output,
			    SHA1_HASH_SIZE * exec_context.total_threads,
			    cudaMemcpyDeviceToHost));

  /* stop the timer (output copy out) */
  CUT_SAFE_CALL(cutStopTimer(timer));
  time_breakdown->copy_out_time = cutGetTimerValue(timer);



  //**** free allocated memory ****//
  CUDA_SAFE_CALL(cudaFree(d_input));
  CUDA_SAFE_CALL(cudaFree(d_output));

  *output_size = SHA1_HASH_SIZE * exec_context.total_threads;

  return status;
}
