/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <template_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    unsigned int num_threads = 32;
    unsigned int mem_size = sizeof( float) * num_threads;

    // allocate host memory
    float* h_idata = (float*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads,
                                cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // compute reference solution
    float* reference = (float*) malloc( mem_size);
    computeGold( reference, h_idata, num_threads);

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat",
                                      h_odata, num_threads, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res = cutComparef( reference, h_odata, num_threads);
        printf( "%s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}
