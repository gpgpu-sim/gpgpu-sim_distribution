// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/* 
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the University of British Columbia, Vancouver, 
 * BC V6T 1Z4, All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdarg.h>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
 #ifdef __APPLE__
 #include <GLUT/glut.h> // Apple's version of GLUT is here
 #else
 #include <GL/gl.h>
 #endif
#endif

#define __CUDA_RUNTIME_API_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/


#include "host_defines.h"
#include "builtin_types.h"
#include "__cudaFatFormat.h"

/*DEVICE_BUILTIN*/
struct cudaArray
{
    void *devPtr;
    int devPtr32;
    struct cudaChannelFormatDesc desc;
    int width;
    int height;
    int size; //in bytes
    unsigned dimensions;
};


#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) \
        = v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

struct gpgpu_ptx_sim_arg {
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
   struct gpgpu_ptx_sim_arg *m_next;
};

struct gpgpu_ptx_sim_arg *g_ptx_sim_params;
cudaError_t g_last_cudaError;

extern void   gpgpu_ptx_sim_init_perf();
extern void   gpgpu_ptx_sim_main_func( const char *kernel_key, dim3 gridDim, dim3 blockDim, struct gpgpu_ptx_sim_arg *);
extern void   gpgpu_ptx_sim_main_perf( const char *kernel_key, 
                                       struct dim3 gridDim, 
                                       struct dim3 blockDIm, struct gpgpu_ptx_sim_arg *grid_params );
extern void*  gpgpu_ptx_sim_malloc( size_t count );
extern void*  gpgpu_ptx_sim_mallocarray( size_t count );
extern void   gpgpu_ptx_sim_memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
extern void   gpgpu_ptx_sim_memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
extern void   gpgpu_ptx_sim_memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
extern void   gpgpu_ptx_sim_memset( size_t dst_start_addr, int c, size_t count );
extern void   gpgpu_ptx_sim_init_memory();
extern void   gpgpu_ptx_sim_load_gpu_kernels();
extern void   gpgpu_ptx_sim_register_kernel(const char *hostFun, const char *deviceFun);
extern void   gpgpu_ptx_sim_register_const_variable(void*, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to );
extern void   gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
extern struct cudaArray* gpgpu_ptx_sim_accessArrayofTexture(struct textureReference* texref);
extern void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
extern struct textureReference* gpgpu_ptx_sim_accessTextureofName(char* name);
extern char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);
extern void   gpgpu_ptx_sim_add_ptxstring( const char * );

extern int g_ptx_sim_mode;

#if defined __APPLE__
#   define __my_func__    __PRETTY_FUNCTION__
#else
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif
#endif

int g_gpgpusim_init = 0;
extern const char *g_gpgpusim_version_string;

#define GPGPUSIM_INIT \
   if( gpgpu_cuda_devices == NULL ) { \
      snprintf(the_cuda_device.name,256,"GPGPU-Sim_v%s", g_gpgpusim_version_string );\
      the_cuda_device.major = 1;\
      the_cuda_device.minor = 3;\
      the_cuda_device.totalGlobalMem = 0x40000000 /* 1 GB */;\
      the_cuda_device.sharedMemPerBlock = (16*1024);\
      the_cuda_device.regsPerBlock = (16*1024);\
      the_cuda_device.warpSize = 32;\
      the_cuda_device.memPitch = 0; \
      the_cuda_device.maxThreadsPerBlock = 512;\
      the_cuda_device.maxThreadsDim[0] = 512; \
      the_cuda_device.maxThreadsDim[1] = 512; \
      the_cuda_device.maxThreadsDim[2] = 512; \
      the_cuda_device.maxGridSize[0] = 0x40000000; \
      the_cuda_device.maxGridSize[1] = 0x40000000; \
      the_cuda_device.maxGridSize[2] = 0x40000000; \
      the_cuda_device.totalConstMem = 0x40000000; \
      the_cuda_device.clockRate = 1000000; /* 1 GHz (WARNING: ignored by performance model) */\
      the_cuda_device.textureAlignment = 0; \
      gpgpu_cuda_devices = (cudaDeviceProp **) calloc(sizeof(struct cudaDeviceProp *),1); \
      gpgpu_cuda_devices[0] = &the_cuda_device; \
   } \
   if( !g_gpgpusim_init ) { \
      gpgpu_ptx_sim_init_perf(); \
      gpgpu_ptx_sim_load_gpu_kernels(); \
      g_gpgpusim_init = 1; \
   }

void cuda_not_implemented( const char* func, unsigned line )
{
   fflush(stdout);
   fflush(stderr);
   printf("\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not been implemented yet.\n"
          "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n", 
          func,__FILE__, line );
   fflush(stdout);
   abort();
}


#define gpgpusim_ptx_error(msg, ...) gpgpusim_ptx_error_impl(__func__, __FILE__,__LINE__, msg, ##__VA_ARGS__)
#define gpgpusim_ptx_assert(cond,msg, ...) gpgpusim_ptx_assert_impl((cond),__func__, __FILE__,__LINE__, msg, ##__VA_ARGS__)

void gpgpusim_ptx_error_impl( const char *func, const char *file, unsigned line, const char *msg, ... )
{
   va_list ap;
   char buf[1024];
   va_start(ap,msg);
   vsnprintf(buf,1024,msg,ap);
   va_end(ap);

   printf("GPGPU-Sim CUDA API: %s\n", buf);
   printf("                    [%s:%u : %s]\n", file, line, func );
   abort();
}

void gpgpusim_ptx_assert_impl( int test_value, const char *func, const char *file, unsigned line, const char *msg, ... )
{
   va_list ap;
   char buf[1024];
   va_start(ap,msg);
   vsnprintf(buf,1024,msg,ap);
   va_end(ap);

   if ( test_value == 0 )
      gpgpusim_ptx_error_impl(func, file, line, msg);
}

#define MY_DEVICE_COUNT 1

int g_active_device = 0; //active gpu that runs the code

struct cudaDeviceProp the_cuda_device;

struct cudaDeviceProp **gpgpu_cuda_devices;

// global kernel parameters...  
static dim3 g_cudaGridDim;
static dim3 g_cudaBlockDim;

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern "C" {

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) 
{
   GPGPUSIM_INIT
   *devPtr = gpgpu_ptx_sim_malloc(size);
   printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",size, (unsigned long long) *devPtr);
   if ( *devPtr  ) {
       return g_last_cudaError = cudaSuccess;
   } else { 
       return g_last_cudaError = cudaErrorMemoryAllocation;
   }
}
 
__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size){
   GPGPUSIM_INIT
    *ptr = malloc(size);
    if ( *ptr  ) {
       return  cudaSuccess;
    } else { 
       return g_last_cudaError = cudaErrorMemoryAllocation;
    }
 }
 __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
   GPGPUSIM_INIT
   unsigned malloc_width_inbytes = width;
   printf("GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
   *devPtr = gpgpu_ptx_sim_malloc(malloc_width_inbytes*height);
   pitch[0] = malloc_width_inbytes;
   if ( *devPtr  ) {
      return  cudaSuccess;
   } else { 
      return g_last_cudaError = cudaErrorMemoryAllocation;
   }
}

__host__ cudaError_t CUDARTAPI cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1))
{
   unsigned size = width * height * ((desc->x + desc->y + desc->z + desc->w)/8);
   GPGPUSIM_INIT
   (*array) = (struct cudaArray*) malloc(sizeof(struct cudaArray));
   (*array)->desc = *desc;
   (*array)->width = width;
   (*array)->height = height;
   (*array)->size = size;
   (*array)->dimensions = 2;
   ((*array)->devPtr32)= (int) (long long)gpgpu_ptx_sim_mallocarray(size);
   printf("GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n", ((*array)->devPtr32));
   ((*array)->devPtr) = (void*) (long long) ((*array)->devPtr32);
   if ( ((*array)->devPtr) ) {
       return g_last_cudaError = cudaSuccess;
   } else { 
       return g_last_cudaError = cudaErrorMemoryAllocation;
   }
}

__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    // TODO...  manage g_global_mem space?
    return g_last_cudaError = cudaSuccess;
}
 __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
      free (ptr);  // this will crash the system if called twice
      return g_last_cudaError = cudaSuccess;
}

 __host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array)
{
    // TODO...  manage g_global_mem space?
    return g_last_cudaError = cudaSuccess;
};


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
   gpgpu_ptx_sim_init_memory();
   printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
   if( kind == cudaMemcpyHostToDevice ) 
      gpgpu_ptx_sim_memcpy_to_gpu( (size_t)dst, src, count ); 
   else if( kind == cudaMemcpyDeviceToHost ) 
     	gpgpu_ptx_sim_memcpy_from_gpu( dst, (size_t)src, count ); 
   else if( kind == cudaMemcpyDeviceToDevice ) 
      gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)dst, (size_t)src, count ); 
   else {
      printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n"); 
      abort();
   }
   return g_last_cudaError = cudaSuccess;
}

 __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind)
{ 
   size_t size = count;
   printf("GPGPU-Sim PTX: cudaMemcpyToArray\n"); 
   gpgpu_ptx_sim_init_memory();
   if( kind == cudaMemcpyHostToDevice ) 
      gpgpu_ptx_sim_memcpy_to_gpu( (size_t)(dst->devPtr), src, size); 
   else if( kind == cudaMemcpyDeviceToHost ) 
      gpgpu_ptx_sim_memcpy_from_gpu( dst->devPtr, (size_t)src, size); 
   else if( kind == cudaMemcpyDeviceToDevice ) 
      gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)(dst->devPtr), (size_t)src, size); 
   else {
      printf("GPGPU-Sim PTX: cudaMemcpyToArray - ERROR : unsupported cudaMemcpyKind\n"); 
      abort();
   }
   dst->devPtr32 = (unsigned) (size_t)(dst->devPtr);
   return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
   gpgpu_ptx_sim_init_memory();
   struct cudaArray *cuArray_ptr;
   size_t size = spitch*height;
   cuArray_ptr = (cudaArray*)dst;
   gpgpusim_ptx_assert( (dpitch==spitch), "different src and dst pitch not supported yet" );
   if( kind == cudaMemcpyHostToDevice ) 
      gpgpu_ptx_sim_memcpy_to_gpu( (size_t)dst, src, size ); 
   else if( kind == cudaMemcpyDeviceToHost ) 
      gpgpu_ptx_sim_memcpy_from_gpu( dst, (size_t)src, size );
   else if( kind == cudaMemcpyDeviceToDevice ) 
      gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)dst, (size_t)src, size); 
   else {
      printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n"); 
      abort();
   }
   return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
   size_t size = spitch*height;
   gpgpu_ptx_sim_init_memory();
   size_t channel_size = dst->desc.w+dst->desc.x+dst->desc.y+dst->desc.z;
   gpgpusim_ptx_assert( ((channel_size%8) == 0), "none byte multiple destination channel size not supported (sz=%u)", channel_size );
   unsigned elem_size = channel_size/8;
   gpgpusim_ptx_assert( (dst->dimensions==2), "copy to none 2D array not supported" );
   gpgpusim_ptx_assert( (wOffset==0), "non-zero wOffset not yet supported" );
   gpgpusim_ptx_assert( (hOffset==0), "non-zero hOffset not yet supported" );
   gpgpusim_ptx_assert( (dst->height == (int)height), "partial copy not supported" );
   gpgpusim_ptx_assert( (elem_size*dst->width == width), "partial copy not supported" );
   gpgpusim_ptx_assert( (spitch == width), "spitch != width not supported" );
   if( kind == cudaMemcpyHostToDevice ) 
      gpgpu_ptx_sim_memcpy_to_gpu( (size_t)(dst->devPtr), src, size); 
   else if( kind == cudaMemcpyDeviceToHost ) 
      gpgpu_ptx_sim_memcpy_from_gpu( dst->devPtr, (size_t)src, size);
   else if( kind == cudaMemcpyDeviceToDevice ) 
      gpgpu_ptx_sim_memcpy_gpu_to_gpu( (size_t)dst->devPtr, (size_t)src, size); 
   else {
      printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n"); 
      abort();
   }
   dst->devPtr32 = (unsigned) (size_t)(dst->devPtr);
   return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
   assert(kind == cudaMemcpyHostToDevice);
   printf("GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p\n", symbol);
   gpgpu_ptx_sim_memcpy_symbol(symbol,src,count,offset,1);
   return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
   assert(kind == cudaMemcpyDeviceToHost);
   printf("GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
   gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0);
   return g_last_cudaError = cudaSuccess;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   printf("GPGPU-Sim PTX: warning cudaMemcpyAsync is implemented as blocking in this version of GPGPU-Sim...\n");
   return cudaMemcpy(dst,src,count,kind);
}


 __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count)
{
   gpgpu_ptx_sim_memset((size_t)mem, c, count);
   return g_last_cudaError = cudaSuccess;
}

 __host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const char *symbol)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const char *symbol)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
 __host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
   *count =  MY_DEVICE_COUNT ; // we have a single gpu with CUDA capability 1 or higher
   GPGPUSIM_INIT
   return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 2010)
extern unsigned int gpu_n_shader;
#endif
   
extern unsigned int warp_size;   
 
 __host__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
   GPGPUSIM_INIT
   if (device <= MY_DEVICE_COUNT)  {
      *prop=*gpgpu_cuda_devices[device];
#if (CUDART_VERSION >= 2010)
      prop->multiProcessorCount = gpu_n_shader;
#endif
      prop->warpSize = warp_size;
      return g_last_cudaError = cudaSuccess;
   } else {
      return g_last_cudaError = cudaErrorInvalidDevice;
   }
}

 __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
   //goal: Choose the best matching device (just returns *device == 0 for now)
   int i;
   *device = -1; // intended to show a non-existing device
   GPGPUSIM_INIT
   for (i=0; i < MY_DEVICE_COUNT ; i++)  {
      if( *device == -1 ) {
         *device= i;  // default, pick the first device
      }
      if( prop->totalGlobalMem <=  gpgpu_cuda_devices[i]->totalGlobalMem && 
          prop->sharedMemPerBlock    <=  gpgpu_cuda_devices[i]->sharedMemPerBlock &&
          prop->regsPerBlock    <=  gpgpu_cuda_devices[i]->regsPerBlock &&
          prop->regsPerBlock    <=  gpgpu_cuda_devices[i]->regsPerBlock &&
          prop->maxThreadsPerBlock   <=  gpgpu_cuda_devices[i]->maxThreadsPerBlock  && 
          prop->totalConstMem   <=  gpgpu_cuda_devices[i]->totalConstMem )
         { 
            // if/when we study heterogenous multicpu configurations
            *device= i;
            break;
         }
   }
   if ( *device !=-1 )
      return g_last_cudaError = cudaSuccess;
   else {
      printf("GPGPU-Sim PTX: Exeuction error: no suitable GPU devices found??? in a simulator??? (%s:%u in %s)\n",
            __FILE__,__LINE__,__my_func__);
      abort();
      return g_last_cudaError = cudaErrorInvalidConfiguration;
   }
}
 
 __host__ cudaError_t CUDARTAPI cudaSetDevice(int device)
{
   //set the active device to run cuda
   if ( device <= MY_DEVICE_COUNT ) {
       g_active_device = device;
       return g_last_cudaError = cudaSuccess;
   } else {
      return g_last_cudaError = cudaErrorInvalidDevice;
   }
}
 
 __host__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
   *device = g_active_device;
   return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
{
   printf("GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = %zu\n", sizeof(struct textureReference));
   struct cudaArray *array;
   array = (struct cudaArray*) malloc(sizeof(struct cudaArray));
   array->desc = *desc;
   array->size = size;
   array->width = size;
   array->height = 1;
   array->dimensions = 1;
   array->devPtr = (void*)devPtr;
   array->devPtr32 = (int)(long long)devPtr;
   offset = 0;
   printf("GPGPU-Sim PTX:   size = %zu\n", size);
   printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
   printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
   printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpgpu_ptx_sim_findNamefromTexture(texref));
   printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n", desc->x, desc->y, desc->z, desc->w);
   printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
   gpgpu_ptx_sim_bindTextureToArray(texref, array);
   devPtr = (void*)(long long)array->devPtr32;
   printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
   return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
   printf("GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);
   printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
   printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpgpu_ptx_sim_findNamefromTexture(texref));
   printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
   gpgpu_ptx_sim_bindTextureToArray(texref, array);
   return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref)
{
    return g_last_cudaError = cudaSuccess;
}


 __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}


 __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const char *symbol)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
   *desc = array->desc;
   return g_last_cudaError = cudaSuccess;
}


 __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
   struct cudaChannelFormatDesc dummy;
   dummy.x = x;
   dummy.y = y;
   dummy.z = z;
   dummy.w = w;
   dummy.f = f;
   return dummy;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
   g_last_cudaError = cudaSuccess;
   return g_last_cudaError;
}

 __host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
{
   if( g_last_cudaError == cudaSuccess ) 
      return "no error";
   char buf[1024];
   snprintf(buf,1024,"<<GPGPU-Sim PTX: there was an error (code = %d)>>", g_last_cudaError);
   return strdup(buf);
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

 __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
   //This is the first function called for a kernel invocation #1
   //if cudaSuccess is returned then cudaSetupArgument is called 
   g_cudaGridDim.x = gridDim.x;
   g_cudaGridDim.y = gridDim.y;
   g_cudaGridDim.z = gridDim.z;

   g_cudaBlockDim.x = blockDim.x;
   g_cudaBlockDim.y = blockDim.y;
   g_cudaBlockDim.z = blockDim.z;

   return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset){
   // Called ifcudaConfigureCall is successful  #2 

   struct gpgpu_ptx_sim_arg *param = (gpgpu_ptx_sim_arg*) calloc(1,sizeof(struct gpgpu_ptx_sim_arg));
   param->m_start = arg;
   param->m_nbytes = size;
   param->m_offset = offset;
   param->m_next = g_ptx_sim_params;
   g_ptx_sim_params = param;
   
   return g_last_cudaError = cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaLaunch(const char *symbol )
{
   printf("\n\n\n");
   char *mode = getenv("PTX_SIM_MODE_FUNC");
   if( mode ) 
      sscanf(mode,"%u", &g_ptx_sim_mode);
   printf("GPGPU-Sim PTX: cudaLaunch for %p (mode=%s)\n", symbol,
          g_ptx_sim_mode?"functional simulation":"performance simulation");
   if( g_ptx_sim_mode )
      gpgpu_ptx_sim_main_func( symbol, g_cudaGridDim, g_cudaBlockDim, g_ptx_sim_params );
   else
      gpgpu_ptx_sim_main_perf( symbol, g_cudaGridDim, g_cudaBlockDim, g_ptx_sim_params );
   g_ptx_sim_params=NULL;
   return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern signed long long gpu_tot_sim_cycle;

struct timer_event
{
   int m_uid;
   int m_updates;
   time_t m_wallclock;
   double m_gpu_tot_sim_cycle;

   struct timer_event *m_next;
};

typedef struct timer_event timer_event_t;

int g_next_event_uid;
timer_event_t *g_timer_events = NULL;

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
{
   timer_event_t *t = (timer_event_t*) calloc(1,sizeof(timer_event_t));

   t->m_uid = ++g_next_event_uid;
   *event = t->m_uid;
   t->m_next = g_timer_events;
   g_timer_events = t;

   t->m_updates = 0;

   return cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
   timer_event_t *t = g_timer_events;
   while( t && t->m_uid != event ) 
      t = t->m_next;
   if( t == NULL ) 
      return cudaErrorUnknown;

   t->m_updates++;
   t->m_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
   t->m_wallclock = time((time_t *)NULL);
   return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event)
{
   printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
   return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
   printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
   return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
   timer_event_t *l = NULL;
   timer_event_t *t = g_timer_events;
   while( t && t->m_uid != event )  {
      l = t;
      t = t->m_next;
   }
   if( t == NULL ) 
      return g_last_cudaError = cudaErrorUnknown;
   if( l ) {
      l->m_next = t->m_next;
      free(t);
      return g_last_cudaError = cudaSuccess;
   } else {
      assert( g_timer_events->m_uid == event );
      l = g_timer_events;
      g_timer_events = g_timer_events->m_next;
      free(l);
      return g_last_cudaError = cudaSuccess;
   }
}


__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
   time_t elapsed_time;
   timer_event_t *s, *e;
   s = e = g_timer_events;
   while( s && s->m_uid != start ) s = s->m_next;
   while( e && e->m_uid != end ) e = e->m_next;
   if( s==NULL || e==NULL ) {
      return g_last_cudaError = cudaErrorUnknown;
   }
   elapsed_time = e->m_wallclock - s->m_wallclock;
   *ms = 1000*elapsed_time; 
   return g_last_cudaError = cudaSuccess;
}



/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void)
{
   // TODO...  manage memory resources?
   return g_last_cudaError = cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void)
{
   //Called on host side 
   //TODO This function should syncronize if we support Asyn kernel calls
   return g_last_cudaError = cudaSuccess;
};

int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
{
   //TODO This function should syncronize if we support Asyn kernel calls
   return g_last_cudaError = cudaSuccess;
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void** CUDARTAPI __cudaRegisterFatBinary( void *fatCubin ) 
{
#if (CUDART_VERSION >= 2010)
   __cudaFatCudaBinary *info =   (__cudaFatCudaBinary *)fatCubin;
   if (info->ptx->ptx)
      gpgpu_ptx_sim_add_ptxstring( info->ptx->ptx );
#endif
   return 0;
}
void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
   ;
}


void CUDARTAPI __cudaRegisterFunction(
                                     void   **fatCubinHandle,
                                     const char    *hostFun,
                                     char    *deviceFun,
                                     const char    *deviceName,
                                     int      thread_limit,
                                     uint3   *tid,
                                     uint3   *bid,
                                     dim3    *bDim,
                                     dim3    *gDim
                                     ) 
{
   gpgpu_ptx_sim_register_kernel(hostFun,deviceFun);
   return;
}

extern void __cudaRegisterVar(
                             void **fatCubinHandle,
                             char *hostVar, //pointer to...something
                             char *deviceAddress, //name of variable
                             const char *deviceName, //name of variable (same as above)
                             int ext,
                             int size,
                             int constant,
                             int global ) 
{
   printf("GPGPU-Sim PTX: __cudaRegisterVar: hostVar = %p; deviceAddress = %s; deviceName = %s\n", hostVar, deviceAddress, deviceName);
   printf("GPGPU-Sim PTX: __cudaRegisterVar: Registering const memory space of %d bytes\n", size);
   fflush(stdout);
   if ( constant && !global && !ext ) {
      gpgpu_ptx_sim_register_const_variable(hostVar,deviceName,size);
   } else if ( !constant && !global && !ext ) {
      gpgpu_ptx_sim_register_global_variable(hostVar,deviceName,size);
   } else cuda_not_implemented(__my_func__,__LINE__);
}


void __cudaRegisterShared(
                         void **fatCubinHandle,
                         void **devicePtr
                         )
{
   // we don't do anything here
   printf("GPGPU-Sim PTX: __cudaRegisterShared\n" );
}

void CUDARTAPI __cudaRegisterSharedVar(
                                      void   **fatCubinHandle,
                                      void   **devicePtr,
                                      size_t   size,
                                      size_t   alignment,
                                      int      storage
                                      )
{
   // we don't do anything here
   printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n" );
}

void __cudaRegisterTexture(
                          void **fatCubinHandle,
                          const struct textureReference *hostVar,
                          const void **deviceAddress,
                          const char *deviceName,
                          int dim,
                          int norm,
                          int ext
                          ) //passes in a newly created textureReference
{
   printf("GPGPU-Sim PTX: in __cudaRegisterTexture:\n");
   gpgpu_ptx_sim_bindNameToTexture(deviceName, hostVar);
   printf("GPGPU-Sim PTX:   int dim = %d\n", dim);
   printf("GPGPU-Sim PTX:   int norm = %d\n", norm);
   printf("GPGPU-Sim PTX:   int ext = %d\n", ext);
   printf("GPGPU-Sim PTX:   Execution warning: Not finished implementing \"%s\"\n", __my_func__ );
}

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj)
{
   printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
   return g_last_cudaError = cudaSuccess;
}

struct glbmap_entry {
   GLuint m_bufferObj;
   void *m_devPtr;
   size_t m_size;
   struct glbmap_entry *m_next;
};
typedef struct glbmap_entry glbmap_entry_t;

glbmap_entry_t* g_glbmap = NULL;

cudaError_t cudaGLMapBufferObject(void** devPtr, GLuint bufferObj) 
{
#ifdef OPENGL_SUPPORT
   GLint buffer_size=0;
   GPGPUSIM_INIT

   glbmap_entry_t *p = g_glbmap;
   while ( p && p->m_bufferObj != bufferObj )
      p = p->m_next;
   if ( p == NULL ) {
      glBindBuffer(GL_ARRAY_BUFFER,bufferObj);
      glGetBufferParameteriv(GL_ARRAY_BUFFER,GL_BUFFER_SIZE,&buffer_size); 
      assert( buffer_size != 0 );
      *devPtr = gpgpu_ptx_sim_malloc(buffer_size);

      // create entry and insert to front of list
      glbmap_entry_t *n = (glbmap_entry_t *) calloc(1,sizeof(glbmap_entry_t));
      n->m_next = g_glbmap;
      g_glbmap = n;

      // initialize entry
      n->m_bufferObj = bufferObj;
      n->m_devPtr = *devPtr;
      n->m_size = buffer_size;

      p = n;
   } else {
      buffer_size = p->m_size;
      *devPtr = p->m_devPtr;
   }

   if ( *devPtr  ) {
      char *data = (char *) calloc(p->m_size,1);
      glGetBufferSubData(GL_ARRAY_BUFFER,0,buffer_size,data);
      gpgpu_ptx_sim_memcpy_to_gpu( (size_t) *devPtr, data, buffer_size );
      free(data);
      printf("GPGPU-Sim PTX: cudaGLMapBufferObject %zu bytes starting at 0x%llx..\n", (size_t)buffer_size, 
             (unsigned long long) *devPtr);
      return g_last_cudaError = cudaSuccess;
   } else {
      return g_last_cudaError = cudaErrorMemoryAllocation;
   }

   return g_last_cudaError = cudaSuccess;
#else
   fflush(stdout);
   fflush(stderr);
   printf("GPGPU-Sim PTX: GPGPU-Sim support for OpenGL integration disabled -- exiting\n");
   fflush(stdout);
   exit(50);
#endif
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj)
{
#ifdef OPENGL_SUPPORT
   glbmap_entry_t *p = g_glbmap;
   while ( p && p->m_bufferObj != bufferObj )
      p = p->m_next;
   if ( p == NULL )
      return g_last_cudaError = cudaErrorUnknown;

   char *data = (char *) calloc(p->m_size,1);
   gpgpu_ptx_sim_memcpy_from_gpu( data,(size_t)p->m_devPtr,p->m_size );
   glBufferSubData(GL_ARRAY_BUFFER,0,p->m_size,data);
   free(data);

   return g_last_cudaError = cudaSuccess;
#else
   fflush(stdout);
   fflush(stderr);
   printf("GPGPU-Sim PTX: support for OpenGL integration disabled -- exiting\n");
   fflush(stdout);
   exit(50);
#endif
}

cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj) 
{
   printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
   return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 2010)

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost,  size_t bytes, unsigned int flags)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags( int flags )
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
{
   *driverVersion = CUDART_VERSION;
   return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
{
   *runtimeVersion = CUDART_VERSION;
   return g_last_cudaError = cudaErrorUnknown;
}

#endif

cudaError_t CUDARTAPI cudaGLSetGLDevice(int device)
{
   printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
   return g_last_cudaError = cudaErrorUnknown;
}

typedef void* HGPUNV;

cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu)
{
   cuda_not_implemented(__my_func__,__LINE__);
   return g_last_cudaError = cudaErrorUnknown;
}

void CUDARTAPI __cudaMutexOperation(int lock)
{
   cuda_not_implemented(__my_func__,__LINE__);
}

void  CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val) 
{
   cuda_not_implemented(__my_func__,__LINE__);
}

}

namespace cuda_math {

void CUDARTAPI __cudaMutexOperation(int lock)
{
   cuda_not_implemented(__my_func__,__LINE__);
}

void  CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val) 
{
   cuda_not_implemented(__my_func__,__LINE__);
}

int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
{
   //TODO This function should syncronize if we support Asyn kernel calls
   return g_last_cudaError = cudaSuccess;
}

}
