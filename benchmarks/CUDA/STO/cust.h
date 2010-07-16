#ifndef CUST_H
#define CUST_H
/*===========================================================================

                       CUSTOMIZATION

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
  Configuration file.

===========================================================================*/


/*===========================================================================

                         PUBLIC DATA DECLARATIONS

===========================================================================*/

/* NOTE: Whenever you change anything in this file, you have to rebuild 
   the project (Build->RebuildSolution) to take the effects. running directly
   using Debug->RunWithoutDebugging will not recompile the solution with the
   new parameters. */


/* ------------------------------------------------------------------
** GENERAL FEATURES 
** ------------------------------------------------------------------ */

// Use shared memory implementation. Note that the total size of the test
// must not exceed 96MB. This will be debugged later on.
#define FEATURE_SHARED_MEMORY

// Use pinned memory pages.
#define FEATURE_PINNED_MODE

// Hash size returned is trimmed to save bandwidth in the copy back.
// Check md5.h for the actual hash size returned.
#define FEATURE_REDUCED_HASH_SIZE

// This feature guesses at run time the best execution context the GPU will
// run within
//#define FEATURE_DYNAMIC_EXEC_CONTEXT

#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
// when in dynamic execution context setting, this feature will enable
// the algorithm that maximizes the number of threads rather than chunk size.
//#define FEATURE_MAXIMIZE_NUM_OF_THREADS
#endif

// Tests to run
// To run the overlap test instead of the original one.
#define FEATURE_RUN_OVERLAP_TEST

// Turn on to run SHA1, otherwise it will run MD5
#define FEATURE_RUN_SHA1

// Enable the multi-thread version for the CPU functions on windows
//#define FEATURE_WIN32_THREADS

// Enable the multi-thread version for the CPU functions on Linux (with pthreads)
//#define FEATURE_PTHREADS

/* ------------------------------------------------------------------
** SANITY CHECKS 
** ------------------------------------------------------------------ */



/* ------------------------------------------------------------------
** CONSTANTS 
** ------------------------------------------------------------------ */
#ifdef FEATURE_DYNAMIC_EXEC_CONTEXT
  #define MAX_THREADS_PER_BLOCK   192
  #define MAX_BLOCKS_PER_GRID     (32 * 1024)
  #define MAX_NUM_OF_THREADS      (MAX_THREADS_PER_BLOCK*MAX_BLOCKS_PER_GRID)
  #define BASIC_CHUNK_SIZE        64
  #define MAX_CHUNK_SIZE          2048
  #define NUM_OF_MULTIPROCESSORS  4
#endif /* FEATURE_DYNAMIC_EXEC_CONTEXT */

#ifdef FEATURE_REDUCED_HASH_SIZE
  #define MD5_HASH_SIZE    4
  #define SHA1_HASH_SIZE   4
#else
  #define MD5_HASH_SIZE    16
  #define SHA1_HASH_SIZE   20
#endif // FEATURE_REDUCED_HASH_SIZE


#ifdef FEATURE_RUN_OVERLAP_TEST  

// THREADS_PER_BLOCK x CHUNK_SIZE should be  < 16K if FEATURE_SHARED_MEMORY
  #define THREADS_PER_BLOCK   128
  #define BLOCKS_PER_GRID     384//32 //1024//16//384
  #define TOTAL_NUM_OF_THREADS ( THREADS_PER_BLOCK * BLOCKS_PER_GRID )

// Chunk size must be multiple of 4.
  #define CHUNK_SIZE          (52)
  //#define CHUNK_SIZE          (1024-12)	

// Offset  must be multiple of 4.
  #define OFFSET		(4)

// Size to be allocated for the original interface tests
  #define TEST_MEM_SIZE_OVERLAP (TOTAL_NUM_OF_THREADS *  OFFSET + \
			       CHUNK_SIZE - OFFSET)

#else
  
  #define THREADS_PER_BLOCK   192
  #define BLOCKS_PER_GRID     512
  #define TOTAL_NUM_OF_THREADS ( THREADS_PER_BLOCK * BLOCKS_PER_GRID )  

// Currently chunk size must be of the form (64*i - 12). 12 is reserved for 
// padding issues.
  //#define CHUNK_SIZE          (64-12)
  //#define CHUNK_SIZE          (128-12)
  //#define CHUNK_SIZE          (256-12)
  //#define CHUNK_SIZE          (512-12)
  #define CHUNK_SIZE          (1024-12)	
  //#define CHUNK_SIZE          (2048-12)	

// Size to be allocated for the tests
  #define TEST_MEM_SIZE       (CHUNK_SIZE * TOTAL_NUM_OF_THREADS)

#endif /* FEATURE_RUN_OVERLAP_TEST */

#endif /* CUST_H */
