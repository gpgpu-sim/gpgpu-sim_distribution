/*==========================================================================
                              S T O R E  C P U

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
  CPU version of the storeGPU library.


==========================================================================*/

/*==========================================================================

                                  INCLUDES

==========================================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "cust.h"
#include "md5_cpu.h"
#include "sha1_cpu.h"


#ifdef FEATURE_WIN32_THREADS
#include <windows.h>
#endif

/*==========================================================================

                             DATA DECLARATIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                              TYPE DEFINITIONS
--------------------------------------------------------------------------*/
// defines an execution context
typedef struct sc_exec_context {
  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int total_size;
  int chunk_size;
  int pad_size;
} sc_exec_context_type;

#ifdef FEATURE_WIN32_THREADS
typedef struct thread_data_struct {

  unsigned char *input;
  unsigned char *output;
  int ilen;

} thread_data_type, *pt_thread_data_type;
#endif  /* FEATURE_WIN32_THREADS */


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
#ifdef FEATURE_MAXIMIZE_NUM_OF_THREADS
/*===========================================================================

FUNCTION SC_GET_EXEC_CONTEXT

DESCRIPTION
  sets the execution context the algorithm will run within: chunk size, 
  thread per block, blocks, padding and total number of threads according 
  to client buffer size.

DEPENDENCIES
  None

RETURN VALUE
  execution context

===========================================================================*/
static void sc_get_exec_context(int size, sc_exec_context_type* ctx){

  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int chunk_size;
  int pad_size;

  int total_chunks = 0;
  int found = 0;
  int index = 1;


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

      found = 1;

    }
    index++;

  }
  

  //**** Fill the struct with the solution ****//
  ctx->threads_per_block = threads_per_block;
  ctx->blocks_per_grid = blocks_per_grid;
  ctx->total_threads = total_threads;
  ctx->total_size = size + pad_size;
  ctx->chunk_size = chunk_size;
  ctx->pad_size = pad_size;

}

#else /* FEATURE_MAXIMIZE_NUM_OF_THREADS */
/*===========================================================================

FUNCTION SC_GET_EXEC_CONTEXT

DESCRIPTION
  sets the required chunk size, thread per block and number of blocks
  needed for kernel execution according to client buffer size.

DEPENDENCIES
  None

RETURN VALUE
  execution context

===========================================================================*/
static void sc_get_exec_context(int size, sc_exec_context_type* ctx){

  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int chunk_size;
  int pad_size;

  int total_chunks = 0;
  int found = 0;

  int index = MAX_CHUNK_SIZE / BASIC_CHUNK_SIZE;



  //**** Determine the execution context ****//
  /* The algorithm will try to determine the context by minimizing chunk 
   * size and maximizing total number of threads 
   * TODO: May be we can do better here 
   */
  while ( 1 ) {
    // Set chunk size
    chunk_size = GET_REAL_CHUNK_SIZE(BASIC_CHUNK_SIZE * index);

    // don't go less than minimum chunk size
    if ( chunk_size < BASIC_CHUNK_SIZE )
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
    if (total_chunks > MAX_NUM_OF_THREADS)
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
    
    if( total_threads > NUM_OF_MULTIPROCESSORS * 32)
      break;
    
    index--;

  }


  //**** Fill the struct with the solution ****//
  ctx->threads_per_block = threads_per_block;
  ctx->blocks_per_grid = blocks_per_grid;
  ctx->total_threads = total_threads;
  ctx->total_size = size + pad_size;
  ctx->chunk_size = chunk_size;
  ctx->pad_size = pad_size;
  
}
#endif /* FEATURE_MAXIMIZE_NUM_OF_THREADS */


/*===========================================================================

FUNCTION SC_GET_OVERLAP_EXEC_CONTEXT

DESCRIPTION
  sets the required chunk size, thread per block and number of blocks
  needed for kernel execution according to client buffer size, offset
  and block size.

DEPENDENCIES
  None

RETURN VALUE
  execution context

===========================================================================*/
static void sc_get_overlap_exec_context( int size, int offset, 
					 int block_size,
					 sc_exec_context_type* ctx ) {

  int threads_per_block;
  int blocks_per_grid;
  int total_threads;
  int total_size;
  int pad_size;


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
  

  //**** Fill the struct with the solution ****//
  ctx->threads_per_block = threads_per_block;
  ctx->blocks_per_grid = blocks_per_grid;
  ctx->total_threads = total_threads;
  ctx->total_size = total_size;
  ctx->chunk_size = block_size;
  ctx->pad_size = pad_size;

}
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

/*===========================================================================

FUNCTION SC_PRINT_EXEC_CONTEXT

DESCRIPTION
  Prints out the passed execution context structure

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
static void sc_print_exec_context( sc_exec_context_type* ctx ) {
  printf("\n== CPU Execution Context ==\n");
  printf("Threads       : %d\n", ctx->threads_per_block);
  printf("Blocks        : %d\n", ctx->blocks_per_grid);
  printf("Total Threads : %d\n", ctx->total_threads);
  printf("Total Size    : %d\n", ctx->total_size);
  printf("Chunk Size    : %d\n", ctx->chunk_size);
  printf("Padding       : %d\n\n", ctx->pad_size);
}

#ifdef FEATURE_WIN32_THREADS
/*===========================================================================

FUNCTION MD5_CPU_MT

DESCRIPTION
  The multithread CPU implementation of the MD5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
static DWORD WINAPI md5_cpu_mt( LPVOID data ){

  pt_thread_data_type thread_data;
  
  //cast to the correct data type 
  thread_data = (pt_thread_data_type)data;
  
  md5_cpu_internal(thread_data->input, thread_data->ilen, thread_data->output);
  
  return 0;
}


/*===========================================================================

FUNCTION MD5_CPU_MT

DESCRIPTION
  The multithread CPU implementation of the MD5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
static DWORD WINAPI sha1_cpu_mt( LPVOID data ){

  pt_thread_data_type thread_data;
  
  //cast to the correct data type 
  thread_data = (pt_thread_data_type)data;
  
  sha1_cpu_internal(thread_data->input, thread_data->ilen, thread_data->output);
  
  return 0;
}
#endif  /* FEATURE_WIN32_THREADS */


/*--------------------------------------------------------------------------
                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/
/*===========================================================================

FUNCTION SC_MD5_STANDARD

DESCRIPTION
  The standard MD5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
void sc_md5_standard( unsigned char* buffer, int size, unsigned char** output) {

  unsigned char * result;

  result = (unsigned char*)malloc( MD5_HASH_SIZE );

  md5_cpu_internal( buffer, size, result );

  *output = result;
}

/*===========================================================================

FUNCTION SC_SHA1_STANDARD

DESCRIPTION
  The standard SHA1 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
void sc_sha1_standard(unsigned char* buffer, int size, unsigned char** output) {
    
  unsigned char * result;

  result = (unsigned char*)malloc( SHA1_HASH_SIZE );

  sha1_cpu_internal( buffer, size, result );
  
  *output = result;

}

/*===========================================================================

FUNCTION SC_MD5

DESCRIPTION
  CPU version of the MD5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
void sc_md5( unsigned char* buffer, int size, 
	     unsigned char** output, int* output_size) {
  
  
  //**** Variable Declarations ****//
  sc_exec_context_type exec_context;
  unsigned char* scratch_data;
  int chunk_index;
  int hash_index;
  int k;

#ifdef FEATURE_WIN32_THREADS
  
  /* This structure contains the input for a particular thread */
  pt_thread_data_type thread_data;
  
  /* Thread identifiers */
  DWORD *thread_id;
  
  /* Thread handlers */
  HANDLE *thread_handle; 

#endif  /* FEATURE_WIN32_THREADS */

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  sc_get_exec_context(size, &exec_context);
#else
  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;

#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

  sc_print_exec_context( &exec_context );
  
  scratch_data = (unsigned char *)malloc(MD5_HASH_SIZE * exec_context.total_threads);
  
#ifdef FEATURE_WIN32_THREADS
  
  //allocate memory for the thread ids
  thread_id = (DWORD *)malloc(sizeof(DWORD)*exec_context.total_threads - 1);
  
  //allocate memory for the thread handle
  thread_handle = (HANDLE *)malloc(sizeof(HANDLE)*exec_context.total_threads-1);
  
  //create structures for thread ids
  for( k = 0; k < exec_context.total_threads-1; k++ ) {
    
    // set indices 
    chunk_index = k * exec_context.chunk_size;
    hash_index = k * MD5_HASH_SIZE;
    
    // Allocate memory for thread data. 
    // thread_data_type is a type that contains the input and output buffers 
    // wrapped up into a structure
    // this is used by the thread to compute and store the hashed values. 
    thread_data = (pt_thread_data_type) HeapAlloc(GetProcessHeap(), 
						  HEAP_ZERO_MEMORY, 
						  sizeof(thread_data_type));
    
    // In case something wrong happen. That is: if memory cannot be 
    // allocated in the Heap of the current process. 
    if( thread_data == NULL ) {
      fprintf(stderr,"\n[FATAL ERROR] Unable to allocate memory in the heap for Thread[%d]",k);         
      ExitProcess(2);
    }
    
    // Generate unique data for each thread.
    thread_data->input = buffer + chunk_index;		
    thread_data->ilen = exec_context.chunk_size;
    thread_data->output = scratch_data + hash_index;
    
    // Create a thread
    thread_handle[k] = CreateThread(NULL, 0, md5_cpu_mt, thread_data, 
				    0, &thread_id[k]);
    
    int i;

    // Check whether the thread was created correctly. If it was not, close the
    // handlers and release memory
    if (thread_handle[k] == NULL) {
      
      fprintf(stderr,"\n[FATAL ERROR] Unable to spawn thread[%d].\n\t Releasing resources and saying goodbye!\n",k);
      
      for( i=0; i < exec_context.total_threads-1; i++) {
	
	if ( thread_handle[i] != NULL ) {
	  CloseHandle(thread_handle[i]);
	}
      }
      
      HeapFree(GetProcessHeap(), 0, thread_data);
      
      ExitProcess(k);
    }
  }
  
  // wait for each thread to finish
  WaitForMultipleObjects(exec_context.total_threads-1, thread_handle, 
			 TRUE, INFINITE);
  
  // Close all thread handles and free memory allocation.
  for(k=0; k < exec_context.total_threads-1; k++) {
    CloseHandle(thread_handle[k]);
  }
  
  HeapFree(GetProcessHeap(), 0, thread_data);
  
#else
  
  for( k = 0; k < exec_context.total_threads - 1; k++) {
    chunk_index = k * exec_context.chunk_size;
    hash_index = k * MD5_HASH_SIZE;
    md5_cpu_internal(buffer + chunk_index, exec_context.chunk_size, 
		     scratch_data + hash_index );
  }
#endif /* FEATURE_WIN32_THREADS */
  
  chunk_index = k * exec_context.chunk_size;
  hash_index = k * MD5_HASH_SIZE;

  if(exec_context.pad_size != 0) {

    unsigned char *last_chunk = (unsigned char*)malloc(exec_context.chunk_size);
    
    memset(last_chunk, 0, exec_context.chunk_size);
    memcpy(last_chunk, buffer + chunk_index, 
	   exec_context.chunk_size - exec_context.pad_size);
    md5_cpu_internal(last_chunk, exec_context.chunk_size, 
		     scratch_data + hash_index );
  } else {
    
    md5_cpu_internal(buffer + chunk_index, exec_context.chunk_size, 
		     scratch_data + hash_index );
  }
  
 //**** will do the last hshing stage ****//
  sc_md5_standard( scratch_data, MD5_HASH_SIZE * exec_context.total_threads, 
		   output );

  *output_size = MD5_HASH_SIZE;

}

/*===========================================================================

FUNCTION SC_MD5_OVERLAP

DESCRIPTION
  Returns the MD5 hash of each block for the provided buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
void sc_md5_overlap(unsigned char* buffer, int size, int block_size, 
		    int offset, unsigned char** output, int* output_size) {
  
   //**** Variable Declarations ****//
  sc_exec_context_type exec_context;
  unsigned char* result;
  int chunk_index;
  int hash_index;
  int k;

#ifdef FEATURE_WIN32_THREADS
	
  /* This structure contains the input for a particular thread */
  pt_thread_data_type thread_data;
  
  /* Thread identifiers */
  DWORD *thread_id;
  
  /* Thread handlers */
  HANDLE *thread_handle; 
  
#endif /* FEATURE_WIN32_THREADS */

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  sc_get_overlap_exec_context(size, offset, block_size, &exec_context);
#else
  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

  sc_print_exec_context( &exec_context ); 

  result = (unsigned char*)malloc(MD5_HASH_SIZE * exec_context.total_threads);

#ifdef FEATURE_WIN32_THREADS

	int i;

  //allocate memory for the thread ids
  thread_id = (DWORD *)malloc(sizeof(DWORD)*exec_context.total_threads-1);
  
  //allocate memory for the thread handle
  thread_handle = (HANDLE *)malloc(sizeof(HANDLE)*exec_context.total_threads-1);
  
  //create structures for thread ids
  for( k = 0; k < exec_context.total_threads-1; k++ ) {
    
    // set indices 
    chunk_index = k * offset;
    hash_index = k * MD5_HASH_SIZE;
    
    // Allocate memory for thread data. 
    // thread_data_type is a type that contains the input and output buffers 
    // wrapped up into a structure
    // this is used by the thread to compute and store the hashed values. 
    thread_data = (pt_thread_data_type) HeapAlloc(GetProcessHeap(), 
						  HEAP_ZERO_MEMORY, 
						  sizeof(thread_data_type));
    
    // In case something wrong happen. That is: if memory cannot be 
    // allocated in the Heap of the current process. 
    if( thread_data == NULL ) {
      fprintf(stderr,"\n[FATAL ERROR] Unable to allocate memory in the heap for Thread[%d]",k);         
      ExitProcess(2);
    }
    
    // Generate unique data for each thread.
    thread_data->input = buffer + chunk_index;		
    thread_data->ilen = block_size;
    thread_data->output = result + hash_index;
    
    // Create a thread
    thread_handle[k] = CreateThread(NULL, 0, md5_cpu_mt, thread_data, 
				    0, &thread_id[k]);
    
    // Check whether the thread was created correctly. If it was not, close the
    // handlers and release memory
    if (thread_handle[k] == NULL) {
      
      fprintf(stderr,"\n[FATAL ERROR] Unable to spawn thread[%d].\n\t Releasing resources and saying goodbye!\n",k);
      
      for( i=0; i < exec_context.total_threads-1; i++) {
	
	if ( thread_handle[i] != NULL ) {
	  CloseHandle(thread_handle[i]);
	}
      }
      
      HeapFree(GetProcessHeap(), 0, thread_data);
      
      ExitProcess(k);
    }
  }
  
  // wait for each thread to finish
  WaitForMultipleObjects(exec_context.total_threads-1, thread_handle, 
			 TRUE, INFINITE);
  
  // Close all thread handles and free memory allocation.
  for(k=0; k < exec_context.total_threads-1; k++) {
    CloseHandle(thread_handle[k]);
  }
  
  HeapFree(GetProcessHeap(), 0, thread_data);
  
#else 
  
  for(k = 0 ; k < exec_context.total_threads - 1; k++) {
    chunk_index = k * offset;
    hash_index = k * MD5_HASH_SIZE;
    md5_cpu_internal(buffer + chunk_index, block_size, result + hash_index );
  }

#endif /* FEATURE_WIN32_THREADS */

  chunk_index = k * offset;
  hash_index = k * MD5_HASH_SIZE;
  md5_cpu_internal(buffer + chunk_index, block_size - exec_context.pad_size,
		   result + hash_index );


  *output = result;
  *output_size = MD5_HASH_SIZE * exec_context.total_threads;
  
}

/*===========================================================================

FUNCTION SC_SHA1

DESCRIPTION
  CPU version of the SHA1 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
void sc_sha1( unsigned char* buffer, int size, 
	     unsigned char** output, int* output_size) {

   //**** Variable Declarations ****//
  sc_exec_context_type exec_context;
  unsigned char* scratch_data;
  int chunk_index;
  int hash_index;
  int k;

#ifdef FEATURE_WIN32_THREADS
	
    /* This structure contains the input for a particular thread */
	pt_thread_data_type thread_data;

	/* Thread identifiers */
    DWORD *thread_id;

	/* Thread handlers */
    HANDLE *thread_handle; 

#endif  /* FEATURE_WIN32_THREADS */

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  //**** Calculate pad size and needed block and grid sizes ****//
  sc_get_exec_context(size, &exec_context);
#else
  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

  sc_print_exec_context( &exec_context );

  scratch_data = (unsigned char*)malloc(SHA1_HASH_SIZE * exec_context.total_threads);  

#ifdef FEATURE_WIN32_THREADS

	int i;

  //allocate memory for the thread ids
  thread_id = (DWORD *)malloc(sizeof(DWORD)*exec_context.total_threads-1);
  
  //allocate memory for the thread handle
  thread_handle = (HANDLE *)malloc(sizeof(HANDLE)*exec_context.total_threads-1);
  
  //create structures for thread ids
  for( k = 0; k < exec_context.total_threads-1; k++ ) {
    
    // set indices 
    chunk_index = k * exec_context.chunk_size;
    hash_index = k * SHA1_HASH_SIZE;
    
    // Allocate memory for thread data. 
    // thread_data_type is a type that contains the input and output buffers 
    // wrapped up into a structure
    // this is used by the thread to compute and store the hashed values. 
    thread_data = (pt_thread_data_type) HeapAlloc(GetProcessHeap(), 
						  HEAP_ZERO_MEMORY, 
						  sizeof(thread_data_type));
    
    // In case something wrong happen. That is: if memory cannot be 
    // allocated in the Heap of the current process. 
    if( thread_data == NULL ) {
      fprintf(stderr,"\n[FATAL ERROR] Unable to allocate memory in the heap for Thread[%d]",k);         
      ExitProcess(2);
    }
    
    // Generate unique data for each thread.
    thread_data->input = buffer + chunk_index;		
    thread_data->ilen = exec_context.chunk_size;
    thread_data->output = scratch_data + hash_index;
    
    // Create a thread
    thread_handle[k] = CreateThread(NULL, 0, sha1_cpu_mt, thread_data, 
				    0, &thread_id[k]);
    
    // Check whether the thread was created correctly. If it was not, close the
    // handlers and release memory
    if (thread_handle[k] == NULL) {
      
      fprintf(stderr,"\n[FATAL ERROR] Unable to spawn thread[%d].\n\t Releasing resources and saying goodbye!\n",k);
      
      for( i=0; i < exec_context.total_threads-1; i++) {
	
	if ( thread_handle[i] != NULL ) {
	  CloseHandle(thread_handle[i]);
	}
      }
      
      HeapFree(GetProcessHeap(), 0, thread_data);
      
      ExitProcess(k);
    }
  }
  
  // wait for each thread to finish
  WaitForMultipleObjects(exec_context.total_threads-1, thread_handle, 
			 TRUE, INFINITE);
  
  // Close all thread handles and free memory allocation.
  for(k=0; k < exec_context.total_threads-1; k++) {
    CloseHandle(thread_handle[k]);
  }
  
  HeapFree(GetProcessHeap(), 0, thread_data);
  
#else 
  
  for( k = 0; k < exec_context.total_threads - 1; k++) {
    chunk_index = k * exec_context.chunk_size;
    hash_index = k * SHA1_HASH_SIZE;
    sha1_cpu_internal(buffer + chunk_index, exec_context.chunk_size, 
		     scratch_data + hash_index );
  }

#endif  /* FEATURE_WIN32_THREADS */

  chunk_index = k * exec_context.chunk_size;
  hash_index = k * SHA1_HASH_SIZE;

  if(exec_context.pad_size != 0) {

    unsigned char *last_chunk = (unsigned char*)malloc(exec_context.chunk_size);
    
    memset(last_chunk, 0, exec_context.chunk_size);
    memcpy(last_chunk, buffer + chunk_index, 
	   exec_context.chunk_size - exec_context.pad_size);
    sha1_cpu_internal(last_chunk, exec_context.chunk_size, 
		     scratch_data + hash_index );
  } else {
    
    sha1_cpu_internal(buffer + chunk_index, exec_context.chunk_size, 
		     scratch_data + hash_index );
  }

 //**** will do the last hshing stage ****//
  sc_sha1_standard( scratch_data, SHA1_HASH_SIZE * exec_context.total_threads, 
		   output );

  *output_size = SHA1_HASH_SIZE;
    
}

/*===========================================================================

FUNCTION SC_SHA1_OVERLAP

DESCRIPTION
  CPU version of the SHA1 overlap algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
void sc_sha1_overlap(unsigned char* buffer, int size, int block_size, 
		     int offset, unsigned char** output, int* output_size) {
  
   //**** Variable Declarations ****//
  sc_exec_context_type exec_context;
  unsigned char* result;
  int chunk_index;
  int hash_index;
  int k;

#ifdef FEATURE_WIN32_THREADS
  
  /* This structure contains the input for a particular thread */
  pt_thread_data_type thread_data;
  
  /* Thread identifiers */
  DWORD *thread_id;
  
  /* Thread handlers */
  HANDLE *thread_handle; 
  
#endif  /* FEATURE_WIN32_THREADS */
  
#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  
  //**** Calculate pad size and needed block and grid sizes ****//
  sc_get_overlap_exec_context(size, offset, block_size, &exec_context);

#else
  //**** Fill the execution context structure ****//
  exec_context.threads_per_block = THREADS_PER_BLOCK;
  exec_context.blocks_per_grid = BLOCKS_PER_GRID;
  exec_context.chunk_size = CHUNK_SIZE;
  exec_context.total_size = size;
  exec_context.total_threads = TOTAL_NUM_OF_THREADS;
  exec_context.pad_size = 0;
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

  sc_print_exec_context( &exec_context ); 

  result = (unsigned char*)malloc(SHA1_HASH_SIZE * exec_context.total_threads);

#ifdef FEATURE_WIN32_THREADS

	int i;

  //allocate memory for the thread ids
  thread_id = (DWORD *)malloc(sizeof(DWORD)*exec_context.total_threads-1);
  
  //allocate memory for the thread handle
  thread_handle = (HANDLE *)malloc(sizeof(HANDLE)*exec_context.total_threads-1);
  
  //create structures for thread ids
  for( k = 0; k < exec_context.total_threads-1; k++ ) {
    
    // set indices 
    chunk_index = k * offset;
    hash_index = k * SHA1_HASH_SIZE;
    
    // Allocate memory for thread data. 
    // thread_data_type is a type that contains the input and output buffers 
    // wrapped up into a structure
    // this is used by the thread to compute and store the hashed values. 
    thread_data = (pt_thread_data_type) HeapAlloc(GetProcessHeap(), 
						  HEAP_ZERO_MEMORY, 
						  sizeof(thread_data_type));
    
    // In case something wrong happen. That is: if memory cannot be 
    // allocated in the Heap of the current process. 
    if( thread_data == NULL ) {
      fprintf(stderr,"\n[FATAL ERROR] Unable to allocate memory in the heap for Thread[%d]",k);         
      ExitProcess(2);
    }
    
    // Generate unique data for each thread.
    thread_data->input = buffer + chunk_index;		
    thread_data->ilen = block_size;
    thread_data->output = result + hash_index;
    
    // Create a thread
    thread_handle[k] = CreateThread(NULL, 0, sha1_cpu_mt, thread_data, 
				    0, &thread_id[k]);
    
    // Check whether the thread was created correctly. If it was not, close the
    // handlers and release memory
    if (thread_handle[k] == NULL) {
      
      fprintf(stderr,"\n[FATAL ERROR] Unable to spawn thread[%d].\n\t Releasing resources and saying goodbye!\n",k);
      
      for( i=0; i < exec_context.total_threads-1; i++) {
	
	if ( thread_handle[i] != NULL ) {
	  CloseHandle(thread_handle[i]);
	}
      }
      
      HeapFree(GetProcessHeap(), 0, thread_data);
      
      ExitProcess(k);
    }
  }
  
  // wait for each thread to finish
  WaitForMultipleObjects(exec_context.total_threads-1, thread_handle, 
			 TRUE, INFINITE);
  
  // Close all thread handles and free memory allocation.
  for(k=0; k < exec_context.total_threads-1; k++) {
    CloseHandle(thread_handle[k]);
  }
  
  HeapFree(GetProcessHeap(), 0, thread_data);
  
#else
  
  for(k = 0 ; k < exec_context.total_threads - 1; k++) {
    chunk_index = k * offset;
    hash_index = k * SHA1_HASH_SIZE;
    sha1_cpu_internal(buffer + chunk_index, block_size, result + hash_index );
  }

#endif  /* FEATURE_WIN32_THREADS */

  chunk_index = k * offset;
  hash_index = k * SHA1_HASH_SIZE;
  sha1_cpu_internal(buffer + chunk_index, block_size - exec_context.pad_size,
		   result + hash_index );


  *output = result;
  *output_size = SHA1_HASH_SIZE * exec_context.total_threads;
  
}
