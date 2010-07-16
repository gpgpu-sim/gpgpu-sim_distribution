/*==========================================================================
                                  M A I N

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
  Main entry.


==========================================================================*/

/*==========================================================================

                                  INCLUDES

==========================================================================*/
// system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// project
#include <cust.h>
#include <storeGPU.h>
#include <storeCPU.h>


/*==========================================================================

                             DATA DECLARATIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                              TYPE DEFINITIONS
--------------------------------------------------------------------------*/

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

/*==========================================================================

                                  FUNCTIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/

/*===========================================================================

FUNCTION SG_PRINT_TIME_BREAKDOWN

DESCRIPTION
  Prints out the given time breakdown parameter

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
static void print_gpu_time_breakdown( sg_time_breakdown_type* time_breakdown, 
				      float input_buffer_alloc_time, 
				      float gpu_init_time ) {

  printf("\n== GPU Timing ==\n");
  printf("GPU init                 : %f\n", gpu_init_time);
  printf("Host input buffer alloc  : %f\n", input_buffer_alloc_time);
  printf("-----\n");
  printf("Host output buffer alloc : %f\n", 
	 time_breakdown->host_output_buffer_alloc_time);
  printf("GPU memory alloc         : %f\n", time_breakdown->device_mem_alloc_time);
  printf("Data copy in             : %f\n", time_breakdown->copy_in_time);
  printf("Kernel execution         : %f\n", time_breakdown->exec_time);
  printf("Data copy out            : %f\n", time_breakdown->copy_out_time);
  printf("Last hasing stage        : %f\n", time_breakdown->last_stage_time);
 
}

#ifdef FEATURE_RUN_OVERLAP_TEST
/*===========================================================================

FUNCTION run_md5_overlap_test

DESCRIPTION
  run the test

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void run_md5_overlap_test( ) {

  //**** Variables ****//
  float host_input_buffer_alloc_time, gpu_init_time;
  sg_time_breakdown_type gpu_time_breakdown;
  unsigned char* sc_output;
  unsigned char* sg_output;
  unsigned char* buffer;
  unsigned int timer;
  int sg_output_size;
  int sc_output_size;


  printf( "MD5 Overlap Test\n\n" );
  
    //**** create the timer ****//
  timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));


  
  //** GPU initialization timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* GPU device initialization */
  sg_init();

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  gpu_init_time = cutGetTimerValue(timer);



  //**** Host input buffer allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* allocate test buffer */
  buffer = (unsigned char*) sg_malloc(TEST_MEM_SIZE_OVERLAP);

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  host_input_buffer_alloc_time = cutGetTimerValue(timer);



  //**** initialize test buffer with random data ****//
  for( unsigned int i = 0; i < TEST_MEM_SIZE_OVERLAP; ++i) {
    buffer[i] = i;
  }

     

  /***************/
  /***** GPU *****/
  /***************/

  //** MD5 timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* run GPU version */
  sg_md5_overlap(buffer, TEST_MEM_SIZE_OVERLAP, CHUNK_SIZE, OFFSET, 
		 &sg_output, &sg_output_size, &gpu_time_breakdown);
  
  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  print_gpu_time_breakdown( &gpu_time_breakdown, 
			    host_input_buffer_alloc_time, 
			    gpu_init_time );
  printf( "GPU Proc. Time (gpu init and input alloc are not included):  %f \n",
	  cutGetTimerValue(timer));



  
  /***************/
  /***** CPU *****/
  /***************/
  //**** start timer for cpu ****//
  CUT_SAFE_CALL( cutResetTimer( timer ) );
  CUT_SAFE_CALL( cutStartTimer( timer ) );
  

  //**** run CPU version ****//
  sc_md5_overlap(buffer, TEST_MEM_SIZE_OVERLAP, CHUNK_SIZE, OFFSET, 
		 &sc_output, &sc_output_size);
  
  
  //**** stop the timer ****//
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "CPU Processing time(ms):    %f \n", cutGetTimerValue( timer));

  
  if(sc_output_size != sg_output_size){
    printf( "\nGPU and CPU didn't converse to the same output size:\n");
    printf( "\nGPU output size: %d\n", sg_output_size);
    printf( "\nCPU output size: %d\n", sc_output_size);
  } else {
    printf( "\nOutput size: %d\n", sc_output_size);
  }
      
    
  //**** check if the results are equivalent ****//  
  CUTBoolean res = cutCompareub( sg_output, 
				 sc_output, 
				 sg_output_size);
  
  
  //**** print the results ****//  
  printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  printf("CPU GPU\n");
  for ( int i = sc_output_size - 4; i < sc_output_size; i++) {
    printf("%X %X\n",sc_output[i], sg_output[i]);
  }


  sg_free(buffer);
  //sg_free(sg_output);  
  cudaFreeHost(sg_output );
  free(sc_output);
	     
}
  
/*===========================================================================

FUNCTION run_sha1_overlap_test

DESCRIPTION
  run the test

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void run_sha1_overlap_test( ) {

  //**** Variables ****//
  float host_input_buffer_alloc_time, gpu_init_time;
  sg_time_breakdown_type gpu_time_breakdown;
  unsigned char* sc_output;
  unsigned char* sg_output;
  unsigned char* buffer;
  unsigned int timer;
  int sg_output_size;
  int sc_output_size;


  printf( "SHA1 Overlap Test\n\n" );
  
    //**** create the timer ****//
  timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));


  
  //** GPU initialization timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* GPU device initialization */
  sg_init();

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  gpu_init_time = cutGetTimerValue(timer);



  //**** Host input buffer allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* allocate test buffer */
  buffer = (unsigned char*) sg_malloc(TEST_MEM_SIZE_OVERLAP);

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  host_input_buffer_alloc_time = cutGetTimerValue(timer);



  //**** initialize test buffer with random data ****//
  for( unsigned int i = 0; i < TEST_MEM_SIZE_OVERLAP; ++i) {
    buffer[i] = i;
  }

     

  /***************/
  /***** GPU *****/
  /***************/

  //** SHA1 timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* run GPU version */
  sg_sha1_overlap(buffer, TEST_MEM_SIZE_OVERLAP, CHUNK_SIZE, OFFSET, 
		  &sg_output, &sg_output_size, &gpu_time_breakdown);
  
  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  print_gpu_time_breakdown( &gpu_time_breakdown, 
			    host_input_buffer_alloc_time, 
			    gpu_init_time );
  printf( "GPU Proc. Time (gpu init and input alloc are not included):  %f \n",
	  cutGetTimerValue(timer));



  
  /***************/
  /***** CPU *****/
  /***************/
  //**** start timer for cpu ****//
  CUT_SAFE_CALL( cutResetTimer( timer ) );
  CUT_SAFE_CALL( cutStartTimer( timer ) );
  

  //**** run CPU version ****//
  sc_sha1_overlap(buffer, TEST_MEM_SIZE_OVERLAP, CHUNK_SIZE, OFFSET, 
		 &sc_output, &sc_output_size);
  
  
  //**** stop the timer ****//
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "CPU Processing time(ms):    %f \n", cutGetTimerValue( timer));

  
  if(sc_output_size != sg_output_size){
    printf( "\nGPU and CPU didn't converse to the same output size:\n");
    printf( "\nGPU output size: %d\n", sg_output_size);
    printf( "\nCPU output size: %d\n", sc_output_size);
  } else {
    printf( "\nOutput size: %d\n", sc_output_size);
  }
      
    
  //**** check if the results are equivalent ****//  
  CUTBoolean res = cutCompareub( sg_output, 
				 sc_output, 
				 sg_output_size);
  
  
  //**** print the results ****//  
  printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  printf("CPU GPU\n");
  for ( int i = sc_output_size - 4; i < sc_output_size; i++) {
    printf("%X %X\n",sc_output[i], sg_output[i]);
  }


  sg_free(buffer);
  //sg_free(sg_output);  
  cudaFreeHost(sg_output ); 
  free(sc_output);
	     
}

#else /* FEATURE_RUN_OVERLAP_TEST */


/*===========================================================================

FUNCTION run_md5_test

DESCRIPTION
  run the test

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void run_md5_test( ) {

  //**** Variables ****//
  unsigned char *sc_output, *sc_single_output;
  unsigned char *sg_output;
  unsigned char *buffer;
  unsigned int timer;
  int sg_output_size;
  int sc_output_size;
  float host_input_buffer_alloc_time, gpu_init_time;
  sg_time_breakdown_type gpu_time_breakdown;

  printf( "MD5 Test\n\n" );

  //**** create the timer ****//
  timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));


  //** GPU initialization timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* GPU device initialization */
  sg_init();

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  gpu_init_time = cutGetTimerValue(timer);


  //**** Host input buffer allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* allocate test buffer */
  buffer = (unsigned char*) sg_malloc(TEST_MEM_SIZE);

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  host_input_buffer_alloc_time = cutGetTimerValue(timer);



  //**** initialize test buffer with random data ****//
  for( unsigned int i = 0; i < TEST_MEM_SIZE; ++i) {
    buffer[i] = i;
  }

     

  /***************/
  /***** GPU *****/
  /***************/


  //** MD5 timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* run GPU version */
  sg_md5(buffer, TEST_MEM_SIZE, &sg_output, &sg_output_size, 
	 &gpu_time_breakdown);
  
  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  print_gpu_time_breakdown( &gpu_time_breakdown, 
			    host_input_buffer_alloc_time, 
			    gpu_init_time );
  printf( "GPU Proc. Time (gpu init and input alloc are not included):  %f \n",
	  cutGetTimerValue(timer));

  
  /***************/
  /***** CPU *****/
  /***************/
  //**** start timer for cpu ****//
  CUT_SAFE_CALL( cutResetTimer( timer ) );
  CUT_SAFE_CALL( cutStartTimer( timer ) );
  
  //**** run CPU version ****//
  sc_md5(buffer, TEST_MEM_SIZE, &sc_output, &sc_output_size);
  
  //**** stop the timer ****//
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "CPU Processing time(ms):    %f \n", cutGetTimerValue( timer));

  /*****************************/
  /***** CPU Single Thread *****/
  /*****************************/
  //**** start timer for single thread cpu ****//
  CUT_SAFE_CALL( cutResetTimer( timer ) );
  CUT_SAFE_CALL( cutStartTimer( timer ) );
  
  //**** run Single Thread CPU version ****//
  sc_md5_standard(buffer, TEST_MEM_SIZE, &sc_single_output);
  
  //**** stop the timer ****//
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "CPU Single Thread Processing time(ms):    %f \n", 
	  cutGetTimerValue( timer));

  
  if(sc_output_size != sg_output_size){
    printf( "\nGPU and CPU didn't converse to the same output size:\n");
    printf( "\nGPU output size: %d\n", sg_output_size);
    printf( "\nCPU output size: %d\n", sc_output_size);
  } else {
    printf( "\nOutput size: %d\n", sc_output_size);
  }
      
    
  //**** check if the results are equivalent ****//  
  CUTBoolean res = cutCompareub( sg_output, 
				 sc_output, 
				 sg_output_size);
  
  
  //**** print the results ****//  
  printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  printf("CPU GPU\n");
  for ( int i = sc_output_size - 4; i < sc_output_size; i++) {
    printf("%X %X\n",sc_output[i], sg_output[i]);
  }


	  
  sg_free(buffer);
  free(sg_output); /* We dont need to free this using sg_free, it will always be
		      allocated using malloc. will try to come up with a 
		      cleaner way to make things more clear. */
  free(sc_output);
	     
}

/*===========================================================================

FUNCTION run_sha1_test

DESCRIPTION
  run the sha1 test

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void run_sha1_test( ) {

  //**** Variables ****//
  unsigned char *sc_output, *sc_single_output;
  unsigned char *sg_output;
  unsigned char *buffer;
  unsigned int timer;
  int sg_output_size;
  int sc_output_size;
  float host_input_buffer_alloc_time, gpu_init_time;
  sg_time_breakdown_type gpu_time_breakdown;

  printf( "SHA1 Test\n\n" );

  //**** create the timer ****//
  timer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timer));

  //** GPU initialization timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* GPU device initialization */
  sg_init();

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  gpu_init_time = cutGetTimerValue(timer);


  //**** Host input buffer allocation timing ****//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* allocate test buffer */
  buffer = (unsigned char*) sg_malloc(TEST_MEM_SIZE);

  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  host_input_buffer_alloc_time = cutGetTimerValue(timer);



  //**** initialize test buffer with random data ****//
  for( unsigned int i = 0; i < TEST_MEM_SIZE; ++i) {
    buffer[i] = i;
  }

     

  /***************/
  /***** GPU *****/
  /***************/


  //** SHA1 timing **//
  CUT_SAFE_CALL(cutResetTimer(timer));
  CUT_SAFE_CALL(cutStartTimer(timer));
  
  /* run GPU version */
  sg_sha1(buffer, TEST_MEM_SIZE, &sg_output, &sg_output_size, 
	 &gpu_time_breakdown);
  
  /* stop the timer */
  CUT_SAFE_CALL(cutStopTimer(timer));
  print_gpu_time_breakdown( &gpu_time_breakdown, 
			    host_input_buffer_alloc_time, 
			    gpu_init_time );
  printf( "GPU Proc. Time (gpu init and input alloc are not included):  %f \n",
	  cutGetTimerValue(timer));

  
  /***************/
  /***** CPU *****/
  /***************/
  //**** start timer for cpu ****//
  CUT_SAFE_CALL( cutResetTimer( timer ) );
  CUT_SAFE_CALL( cutStartTimer( timer ) );
  
  //**** run CPU version ****//
  sc_sha1(buffer, TEST_MEM_SIZE, &sc_output, &sc_output_size);
  
  //**** stop the timer ****//
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "CPU Processing time(ms):    %f \n", cutGetTimerValue( timer));

  /*****************************/
  /***** CPU Single Thread *****/
  /*****************************/
  //**** start timer for single thread cpu ****//
  CUT_SAFE_CALL( cutResetTimer( timer ) );
  CUT_SAFE_CALL( cutStartTimer( timer ) );
  
  //**** run Single Thread CPU version ****//
  sc_sha1_standard(buffer, TEST_MEM_SIZE, &sc_single_output);
  
  //**** stop the timer ****//
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "CPU Single Thread Processing time(ms):    %f \n", 
	  cutGetTimerValue( timer));

  
  if(sc_output_size != sg_output_size){
    printf( "\nGPU and CPU didn't converse to the same output size:\n");
    printf( "\nGPU output size: %d\n", sg_output_size);
    printf( "\nCPU output size: %d\n", sc_output_size);
  } else {
    printf( "\nOutput size: %d\n", sc_output_size);
  }
      
    
  //**** check if the results are equivalent ****//  
  CUTBoolean res = cutCompareub( sg_output, 
				 sc_output, 
				 sg_output_size);
  
  
  //**** print the results ****//  
  printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
  printf("CPU GPU\n");
  for ( int i = sc_output_size - 4; i < sc_output_size; i++) {
    printf("%X %X\n",sc_output[i], sg_output[i]);
  }


	  
  sg_free(buffer);
  free(sg_output); /* We dont need to free this using sg_free, it will always be
		      allocated using malloc. will try to come up with a 
		      cleaner way to make things more clear. */
  free(sc_output);
	     
}
/* void run_sha1_test( ) { */

/*   //\**** Variables ****\// */
/*   unsigned int timer = 0; */
/*   unsigned char* buffer; */
/*   unsigned char* sg_output; */
/*   int sg_output_size; */
/*   unsigned char *sc_output, *sc_single_output; */
/*   int sc_output_size; */


/*   printf( "SHA1 Test\n\n" ); */
  
  
/*   //\**** host memory management ****\// */
/*   // allocate test buffer */
/*     buffer = (unsigned char*) sg_malloc(TEST_MEM_SIZE); */


/*   //\**** initialize test buffer with random data ****\// */
/*   for( unsigned int i = 0; i < TEST_MEM_SIZE; ++i) { */
/*     buffer[i] = i; */
/*   } */

/*   //\**** create the timer ****\// */
/*   timer = 0; */
/*   CUT_SAFE_CALL( cutCreateTimer( &timer)); */
     

/*   /\***************\/ */
/*   /\***** GPU *****\/ */
/*   /\***************\/ */
/*   //\**** start timer for GPU timing ****\// */
/*   CUT_SAFE_CALL( cutResetTimer(timer) ); */
/*   CUT_SAFE_CALL( cutStartTimer( timer)); */
  

/*   //\**** run GPU version ****\// */
/*   sg_sha1(buffer, TEST_MEM_SIZE, &sg_output, &sg_output_size); */

  
/*   //\**** stop the timer ****\// */
/*   CUT_SAFE_CALL( cutStopTimer( timer)); */


/*   //\**** print results ****\// */
/*   printf( "GPU Processing time(ms):    %f \n", cutGetTimerValue( timer)); */

  
/*   /\***************\/ */
/*   /\***** CPU *****\/ */
/*   /\***************\/ */
/*   //\**** start timer for cpu ****\// */
/*   CUT_SAFE_CALL( cutResetTimer( timer ) ); */
/*   CUT_SAFE_CALL( cutStartTimer( timer ) ); */
  

/*   //\**** run CPU version ****\// */
/*   sc_sha1(buffer, TEST_MEM_SIZE, &sc_output, &sc_output_size); */
  
  
/*   //\**** stop the timer ****\// */
/*   CUT_SAFE_CALL( cutStopTimer( timer)); */
/*   printf( "CPU Processing time(ms):    %f \n", cutGetTimerValue( timer)); */

        
/*   /\*****************************\/ */
/*   /\***** CPU Single Thread *****\/ */
/*   /\*****************************\/ */
/*   //\**** start timer for single thread cpu ****\// */
/*   CUT_SAFE_CALL( cutResetTimer( timer ) ); */
/*   CUT_SAFE_CALL( cutStartTimer( timer ) ); */
  
/*   //\**** run CPU version ****\// */
/*   sc_sha1_standard(buffer, TEST_MEM_SIZE, &sc_single_output); */
  
/*   //\**** stop the timer ****\// */
/*   CUT_SAFE_CALL( cutStopTimer( timer)); */
/*   printf( "CPU Single Thread Processing time(ms):    %f \n",  */
/* 	  cutGetTimerValue( timer)); */

  
/*   if(sc_output_size != sg_output_size){ */
/*     printf( "\nGPU and CPU didn't converse to the same output size:\n"); */
/*     printf( "\nGPU output size: %d\n", sg_output_size); */
/*     printf( "\nCPU output size: %d\n", sc_output_size); */
/*   } else { */
/*     printf( "\nOutput size: %d\n", sc_output_size); */
/*   } */

    
/*   //\**** check if the results are equivalent ****\//   */
/*   CUTBoolean res = cutCompareub( sg_output,  */
/* 				 sc_output,  */
/* 				 sg_output_size); */
  
  
/*   //\**** print the results ****\//   */
/*   printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED"); */
/*   printf("CPU GPU\n"); */
/*   for ( int i = sc_output_size - 4; i < sc_output_size; i++) { */
/*     printf("%X %X\n",sc_output[i], sg_output[i]); */
/*   } */


/*   sg_free(buffer); */
/*   free(sg_output); /\* We dont need to free this using sg_free, it will always be */
/* 		      allocated using malloc. I will try to come up with a  */
/* 		      cleaner way to make things more clear. *\/ */
/*   free(sc_output); */
 
/* } */
#endif /* FEATURE_RUN_OVERLAP_TEST */



/*--------------------------------------------------------------------------
                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/
/*===========================================================================

FUNCTION main

DESCRIPTION
  main entry of the program

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
int main( int argc, char** argv)  {

#ifdef FEATURE_SHARED_MEMORY
	printf("Shared Memory Enabled\n");
#endif


#ifdef FEATURE_PINNED_MODE
	printf("Pinned Memory Enabled\n");
#endif


#ifdef FEATURE_REDUCED_HASH_SIZE
	printf("Reduced Hash Size Enabled\n");
#endif

#ifdef FEATURE_RUN_OVERLAP_TEST
#ifdef FEATURE_RUN_SHA1
  printf("Running SHA1 Overlap Test..\n");
  run_sha1_overlap_test( );
#else
  printf("Running MD5 Overlap Test..\n");
  run_md5_overlap_test( );
#endif // FEATURE_RUN_SHA1
#else
#ifdef FEATURE_RUN_SHA1
  printf("Running SHA1 Non-Overlap Test..\n");
  run_sha1_test( );
#else
  printf("Running MD5 Non-Overlap Test..\n");
  run_md5_test( );
#endif // FEATURE_RUN_SHA1
#endif // FEATURE_RUN_OVERLAP_TEST

}

