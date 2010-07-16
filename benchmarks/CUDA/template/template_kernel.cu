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
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>

#define SDATA( index)      cutilBankChecker(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
  // shared memory
  // the size is determined by the host application
  extern  __shared__  float sdata[];

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  // use the bank checker macro to check for bank conflicts during host
  // emulation
  SDATA(tid) = g_idata[tid];
  __syncthreads();

  // perform some computations
  SDATA(tid) = (float) num_threads * SDATA( tid);
  __syncthreads();

  // write data to global memory
  g_odata[tid] = SDATA(tid);
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
