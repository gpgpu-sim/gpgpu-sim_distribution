/* 
 * opencl_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt and the University of British Columbia, 
 * Vancouver, BC V6T 1Z4, All Rights Reserved.
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#endif

#define __CUDA_RUNTIME_API_H__
#include "host_defines.h"
#include "builtin_types.h"
#include "__cudaFatFormat.h"
#include "../src/util.h"

struct gpgpu_ptx_sim_arg {
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
   struct gpgpu_ptx_sim_arg *m_next;
};
extern void   gpgpu_ptx_sim_init_memory();
extern void*  gpgpu_ptx_sim_malloc( size_t count );
extern void   gpgpu_ptx_sim_memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
extern void   gpgpu_ptx_sim_memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
extern void   gpgpu_ptx_sim_memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
extern void   gpgpu_ptx_sim_register_kernel(const char *hostFun, const char *deviceFun);
extern void   gpgpu_ptx_sim_init_perf();
extern void   gpgpu_ptx_sim_main_func( const char *kernel_key, 
                                       dim3 gridDim, 
                                       dim3 blockDim, struct gpgpu_ptx_sim_arg *);
extern void   gpgpu_ptx_sim_main_perf( const char *kernel_key, 
                                       struct dim3 gridDim, 
                                       struct dim3 blockDIm, 
                                       struct gpgpu_ptx_sim_arg *grid_params );
extern void   gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to );
extern int g_ptx_sim_mode;

struct cudaDeviceProp the_cuda_device;
struct cudaDeviceProp **gpgpu_cuda_devices;
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
      g_gpgpusim_init = 1; \
   }

# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __my_func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif

// global kernel parameters...  
static dim3 g_cudaGridDim;
static dim3 g_cudaBlockDim;
struct gpgpu_ptx_sim_arg *g_ptx_sim_params;

#include <CL/cl.h>

#include <map>
#include <string>

struct _cl_context {
   _cl_context() { m_uid = sm_context_uid++; }
   cl_mem CreateBuffer(
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret );
   cl_mem lookup_mem( cl_mem m );
private:
   unsigned m_uid;
   static unsigned sm_context_uid;

   std::map<void*/*host_ptr*/,cl_mem> m_hostptr_to_cl_mem;
   std::map<cl_mem/*device ptr*/,cl_mem> m_devptr_to_cl_mem;
};

struct _cl_device_id {
   _cl_device_id() { m_id = 0; m_next = NULL; }
   struct _cl_device_id *next() { return m_next; }
private:
   unsigned m_id;
   struct _cl_device_id *m_next;
};

struct _cl_command_queue 
{ 
   _cl_command_queue( cl_context context, cl_device_id device, cl_command_queue_properties properties ) 
   {
      m_valid = true;
      m_context = context;
      m_device = device;
      m_properties = properties;
   }
   bool is_valid() { return m_valid; }
   cl_context get_context() { return m_context; }
   cl_device_id get_device() { return m_device; }
   cl_command_queue_properties get_properties() { return m_properties; }
private:
   bool m_valid;
   cl_context                     m_context;
   cl_device_id                   m_device;
   cl_command_queue_properties    m_properties;
};

struct _cl_mem {
   _cl_mem( cl_mem_flags flags, size_t size , void *host_ptr, cl_int *errcode_ret );
   cl_mem device_ptr();
   void* host_ptr();
   bool is_on_host() { return m_is_on_host; }
private:
   bool m_is_on_host;
   size_t m_device_ptr;
   void *m_host_ptr;
   cl_mem_flags m_flags; 
   size_t m_size;
};

struct _cl_program {
   _cl_program( cl_context context,
                cl_uint           count, 
             const char **     strings,   
             const size_t *    lengths );
   void Build(const char *options);
   cl_kernel CreateKernel( const char *kernel_name, cl_int *errcode_ret );
   cl_context get_context() { return m_context; }
   char *get_ptx();
   size_t get_ptx_size();

private:
   cl_context m_context;
   std::map<cl_uint,std::string> m_strings;
   std::map<cl_uint,std::string> m_ptx;
};

struct _cl_kernel {
   _cl_kernel( cl_program prog, const char* kernel_name, void *kernel_impl );
   void SetKernelArg(
      cl_uint      arg_index,
      size_t       arg_size,
      const void * arg_value );
   cl_int bind_args( struct gpgpu_ptx_sim_arg **arg_list );
   std::string name() const { return m_kernel_name; }
   size_t get_workgroup_size();
   cl_program get_program() { return m_prog; }
private:
   unsigned m_uid;
   static unsigned sm_context_uid;
   cl_program m_prog;

   std::string m_kernel_name;

   struct arg_info {
      size_t m_arg_size;
      const void *m_arg_value;
   };
   
   std::map<unsigned, arg_info> m_args;
   void *m_kernel_impl;
};

struct _cl_platform_id {
   static const unsigned m_uid = 0;
};

struct _cl_platform_id g_gpgpu_sim_platform_id;

void *gpgpusim_opencl_getkernel_Object( const char *kernel_name );
void gpgpu_ptx_sim_load_ptx_from_string( const char *p, unsigned source_num );

void gpgpusim_exit()
{
   abort();
}

void gpgpusim_opencl_warning( const char* func, unsigned line, const char *desc )
{
   printf("GPGPU-Sim OpenCL API: Warning (%s:%u) ** %s\n", func,line,desc);
}

void gpgpusim_opencl_error( const char* func, unsigned line, const char *desc )
{
   printf("GPGPU-Sim OpenCL API: ERROR (%s:%u) ** %s\n", func,line,desc);
   gpgpusim_exit();
}

_cl_kernel::_cl_kernel( cl_program prog, const char* kernel_name, void *kernel_impl )
{
   m_uid = sm_context_uid++;
   m_kernel_name = std::string(kernel_name);
   m_kernel_impl = kernel_impl;
   m_prog = prog;
}

void _cl_kernel::SetKernelArg(
      cl_uint      arg_index,
      size_t       arg_size,
      const void * arg_value )
{
   arg_info arg;
   arg.m_arg_size = arg_size;
   arg.m_arg_value = arg_value;
   m_args[arg_index] = arg;
}

cl_int _cl_kernel::bind_args( struct gpgpu_ptx_sim_arg **arg_list )
{
   while( *arg_list ) {
      struct gpgpu_ptx_sim_arg *n = (*arg_list)->m_next;
      free( *arg_list );
      *arg_list = n;
   }
   unsigned k=0;
   std::map<unsigned, arg_info>::iterator i;
   for( i = m_args.begin(); i!=m_args.end(); i++ ) {
      if( i->first != k ) 
         return CL_INVALID_KERNEL_ARGS;
      arg_info arg = i->second;

      struct gpgpu_ptx_sim_arg *param = (gpgpu_ptx_sim_arg*) calloc(1,sizeof(struct gpgpu_ptx_sim_arg));
      param->m_start = arg.m_arg_value;
      param->m_nbytes = arg.m_arg_size;
      param->m_offset = 0;
      param->m_next = *arg_list;
      *arg_list = param;

      k++;
   }
   return CL_SUCCESS;
}


unsigned ptx_kernel_shmem_size( void *kernel_impl );
unsigned ptx_kernel_nregs( void *kernel_impl );
extern unsigned int gpgpu_shmem_size;
extern unsigned int gpgpu_shader_registers;
extern unsigned int gpu_n_thread_per_shader;

#define min(a,b) ((a<b)?(a):(b))

size_t _cl_kernel::get_workgroup_size()
{
   unsigned smem = ptx_kernel_shmem_size( m_kernel_impl );
   unsigned nregs = ptx_kernel_nregs( m_kernel_impl );
   unsigned result_shmem = (unsigned)-1;
   unsigned result_regs = (unsigned)-1;

   if( smem > 0 )
      result_shmem = gpgpu_shmem_size / smem;
   if( nregs > 0 )
      result_regs = gpgpu_shader_registers / ((nregs+3)&~3);

   unsigned result = gpu_n_thread_per_shader;
   result = min(result, result_shmem);
   result = min(result, result_regs);
   return (size_t)result;
}

cl_mem _cl_mem::device_ptr()
{
   cl_mem result = (cl_mem)(void*)m_device_ptr;
   return result;
}

void* _cl_mem::host_ptr()
{
   return m_host_ptr;
}

_cl_mem::_cl_mem(
   cl_mem_flags flags,
   size_t       size ,
   void *       host_ptr,
   cl_int *     errcode_ret )
{
   if( errcode_ret ) 
      *errcode_ret = CL_SUCCESS;

   m_is_on_host = false;
   m_flags = flags;
   m_size = size;
   m_host_ptr = host_ptr;
   m_device_ptr = 0;
   gpgpu_ptx_sim_init_memory();

   if( (flags & (CL_MEM_USE_HOST_PTR|CL_MEM_COPY_HOST_PTR)) && host_ptr == NULL ) {
      if( errcode_ret != NULL ) 
         *errcode_ret = CL_INVALID_HOST_PTR;
      return;
   }
   if( (flags & CL_MEM_COPY_HOST_PTR) && (flags & CL_MEM_USE_HOST_PTR) ) {
      if( errcode_ret ) 
         *errcode_ret = CL_INVALID_VALUE;
      return;
   }
   if( flags & CL_MEM_ALLOC_HOST_PTR ) 
      gpgpusim_opencl_error(__my_func__,__LINE__," CL_MEM_ALLOC_HOST_PTR -- not yet supported.\n");

   if( flags & CL_MEM_USE_HOST_PTR ) {
      m_is_on_host = true;
   } else {
      m_is_on_host = false;
   }
   m_device_ptr = (size_t) gpgpu_ptx_sim_malloc(size);
   if( host_ptr )
      gpgpu_ptx_sim_memcpy_to_gpu( m_device_ptr, host_ptr, size );
}

cl_mem _cl_context::CreateBuffer(
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret )
{
   if( host_ptr && (m_hostptr_to_cl_mem.find(host_ptr) != m_hostptr_to_cl_mem.end()) ) {
      printf("GPGPU-Sim OpenCL API: clCreateBuffer - buffer already created for this host variable\n");
      *errcode_ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
      return NULL;
   }
   cl_mem result = new _cl_mem(flags,size,host_ptr,errcode_ret);
   m_devptr_to_cl_mem[result->device_ptr()] = result;
   if( host_ptr ) 
      m_hostptr_to_cl_mem[host_ptr] = result;
   if( result->device_ptr() ) 
      return (cl_mem) result->device_ptr();
   else 
      return (cl_mem) host_ptr;
}

cl_mem _cl_context::lookup_mem( cl_mem m )
{
   std::map<cl_mem/*device ptr*/,cl_mem>::iterator i=m_devptr_to_cl_mem.find(m);
   if( i == m_devptr_to_cl_mem.end() ) {
      void *t = (void*)m;
      std::map<void*/*host_ptr*/,cl_mem>::iterator j = m_hostptr_to_cl_mem.find(t);
      if( j == m_hostptr_to_cl_mem.end() )
         return NULL;
      else 
         return j->second;
   } else {
      return i->second;
   }
}

_cl_program::_cl_program( cl_context        context,
                          cl_uint           count, 
                          const char **     strings, 
                          const size_t *    lengths )
{
   m_context = context;
   for( cl_uint i=0; i<count; i++ ) {
      unsigned len = lengths[i];
      char *tmp = (char*)malloc(len+1);
      memcpy(tmp,strings[i],len);
      tmp[len] = 0;
      m_strings[i] = tmp;
      free(tmp);
   }
}

extern const char *g_filename;
unsigned g_source_num;

void _cl_program::Build(const char *options)
{
   printf("GPGPU-Sim OpenCL API: compiling OpenCL kernels...\n"); 
   std::map<cl_uint,std::string>::iterator i;
   for( i = m_strings.begin(); i!= m_strings.end(); i++ ) {
      char ptx_fname[1024];
      char *use_extracted_ptx = getenv("PTX_SIM_USE_PTX_FILE");
      if( use_extracted_ptx == NULL ) {
         char *nvopencl_libdir = getenv("NVOPENCL_LIBDIR");
         char *gpgpusim_opencl_path = getenv("GPGPUSIM_ROOT");
         bool error = false;
         if( nvopencl_libdir == NULL ) {
            printf("GPGPU-Sim OpenCL API: Please set your NVOPENCL_LIBDIR environment variable to\n"
                   "                      the location of NVIDIA's libOpenCL.so file on your system.\n");
            error = true;
         }
         if( gpgpusim_opencl_path == NULL ) {
            fprintf(stderr,"GPGPU-Sim OpenCL API: Please set your GPGPUSIM_ROOT environment variable\n");
            fprintf(stderr,"                      to point to the location of your GPGPU-Sim installation\n");
            error = true;
         }
         if( error ) 
            exit(1); 

         char cl_fname[1024];
         std::string src = i->second;
         const char *source = src.c_str();

         // call wrapper
         char *ld_library_path_orig = getenv("LD_LIBRARY_PATH");

         // create temporary filenames
         snprintf(cl_fname,1024,"_cl_XXXXXX");
         snprintf(ptx_fname,1024,"_ptx_XXXXXX");
         int fd=mkstemp(cl_fname); 
         close(fd);
         fd=mkstemp(ptx_fname); 
         close(fd);

         // write OpenCL source to file
         FILE *fp = fopen(cl_fname,"w");
         if( fp == NULL ) {
            printf("GPGPU-Sim OpenCL API: ERROR ** could not create temporary files required for generating PTX\n");
            printf("                      Ensure you have write permission to the simulation directory\n");
            exit(1);
         }
         fprintf(fp,source);
         fclose(fp);

         setenv("LD_LIBRARY_PATH",nvopencl_libdir,1);
         char commandline[1024];
         const char *opt = options?options:"";
         snprintf(commandline,1024,"%s/libopencl/bin/nvopencl_wrapper %s %s %s", 
                   gpgpusim_opencl_path, cl_fname, ptx_fname, opt );
         int result = system(commandline);
         setenv("LD_LIBRARY_PATH",ld_library_path_orig,1);
         if( result != 0 ) {
            printf("GPGPU-Sim OpenCL API: ERROR ** while calling NVIDIA driver to convert OpenCL to PTX (%u)\n",
                   result );
            exit(1);
         }
         // clean up files...
         snprintf(commandline,1024,"rm -f %s", cl_fname );
         result = system(commandline);
         if( result != 0 ) 
            printf("GPGPU-Sim OpenCL API: could not remove temporary files generated while generating PTX\n");
      } else {
         snprintf(ptx_fname,1024,"_%u.ptx", g_source_num);
      }

      // read in PTX generated by wrapper
      FILE *fp = fopen(ptx_fname,"r");
      if( fp == NULL ) {
         printf("GPGPU-Sim OpenCL API: ERROR ** could not open PTX file \'%s\' for reading\n", ptx_fname );
         if( use_extracted_ptx != NULL ) 
            printf("                      Ensure PTX files are in simulation directory.\n");
         exit(1);
      }
      fseek(fp,0,SEEK_END);
      unsigned len = ftell(fp);
      if( len == 0 ) {
         exit(1);
      }
      fseek(fp,0,SEEK_SET);
      char *tmp = (char*)calloc(len+1,1);
      fread(tmp,1,len,fp);
      fclose(fp);
      if( use_extracted_ptx == NULL ) {
         // clean up files...
         char commandline[1024];
         snprintf(commandline,1024,"rm -f %s", ptx_fname );
         int result = system(commandline);
         if( result != 0 ) 
            printf("GPGPU-Sim OpenCL API: could not remove temporary files generated while generating PTX\n");
         // remove any trailing characters from string
         while( len > 0 && tmp[len] != '}' ) {
            tmp[len] = 0;
            len--;
         }
      }
      m_ptx[g_source_num] = tmp;
      gpgpu_ptx_sim_load_ptx_from_string( tmp, g_source_num );
      g_source_num++;
      free(tmp);
   }
   printf("GPGPU-Sim OpenCL API: finished compiling OpenCL kernels.\n"); 
}

cl_kernel _cl_program::CreateKernel( const char *kernel_name, cl_int *errcode_ret )
{
   cl_kernel result = NULL;
   void *kernel_impl = gpgpusim_opencl_getkernel_Object( kernel_name );
   if( kernel_impl == NULL ) 
      *errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
   else {
      result = new _cl_kernel(this,kernel_name,kernel_impl);
      gpgpu_ptx_sim_register_kernel((const char*)result,kernel_name);
   }
   return result;
}

char *_cl_program::get_ptx()
{
   if( m_ptx.empty() ) {
      printf("GPGPU-Sim PTX OpenCL API: Cannot get PTX before having built program\n");
      abort();
   }
   size_t buffer_length= get_ptx_size();
   char *tmp = (char*)calloc(buffer_length,1);
   unsigned n=0;
   std::map<cl_uint,std::string>::iterator p;
   for( p=m_ptx.begin(); p != m_ptx.end(); p++ ) {
      unsigned len = strlen( p->second.c_str() ) + 1;
      assert( (n+len-1) < buffer_length );
      memcpy(tmp+n,p->second.c_str(),len);
      n+=len;
   }
   assert( n < buffer_length );
   tmp[n]=0;
   return tmp;
}

size_t _cl_program::get_ptx_size()
{
   size_t buffer_length=0;
   std::map<cl_uint,std::string>::iterator p;
   for( p=m_ptx.begin(); p != m_ptx.end(); p++ ) {
      buffer_length += p->second.length();
      buffer_length++;
   }
   buffer_length++;
   return buffer_length;
}



unsigned _cl_context::sm_context_uid = 0;
unsigned _cl_kernel::sm_context_uid = 0;

struct _cl_device_id    g_gpgpusim_cl_device_id;
struct _cl_device_id*   g_gpgpusim_cl_device_id_list = &g_gpgpusim_cl_device_id;

void opencl_not_implemented( const char* func, unsigned line )
{
   fflush(stdout);
   fflush(stderr);
   printf("\n\nGPGPU-Sim PTX: Execution error: OpenCL API function \"%s()\" has not been implemented yet.\n"
         "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n", 
         func,__FILE__, line );
   fflush(stdout);
   abort();
}

void opencl_not_finished( const char* func, unsigned line )
{
   fflush(stdout);
   fflush(stderr);
   printf("\n\nGPGPU-Sim PTX: Execution error: OpenCL API function \"%s()\" has not been completed yet.\n"
         "                 [$GPGPUSIM_ROOT/libopencl/%s around line %u]\n\n\n", 
         func,__FILE__, line );
   fflush(stdout);
   abort();
}

extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(cl_context_properties * properties,
                        cl_device_type          device_type,
                        void (*pfn_notify)(const char *, const void *, size_t, void *),
                        void *                  user_data,
                        cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   GPGPUSIM_INIT 
   if( device_type != CL_DEVICE_TYPE_GPU ) {
      printf("GPGPU-Sim OpenCL API: unsupported device type %lx\n", device_type );
      exit(1);
   }
   if( properties != NULL ) {
      printf("GPGPU-Sim OpenCL API: do not know how to use properties in %s\n", __my_func__ );
      exit(1);
   }
   if( errcode_ret ) 
      *errcode_ret = CL_SUCCESS;
   cl_context ctx = new _cl_context;
   return ctx;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context         context, 
                 cl_context_info    param_name, 
                 size_t             param_value_size, 
                 void *             param_value, 
                 size_t *           param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( context == NULL ) return CL_INVALID_CONTEXT;
   switch( param_name ) {
   case CL_CONTEXT_DEVICES: {
      unsigned ngpu=0;
      cl_device_id device_id = g_gpgpusim_cl_device_id_list;
      while ( device_id != NULL ) {
         if( param_value ) 
            ((cl_device_id*)param_value)[ngpu] = device_id;
         device_id = device_id->next();
         ngpu++;
      }
      if( param_value_size_ret ) *param_value_size_ret = ngpu * sizeof(cl_device_id);
      break;
   }
   case CL_CONTEXT_REFERENCE_COUNT:
      opencl_not_finished(__my_func__,__LINE__);
      break;
   case CL_CONTEXT_PROPERTIES: 
      opencl_not_finished(__my_func__,__LINE__);
      break;
   default:
      opencl_not_finished(__my_func__,__LINE__);
   }
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     context, 
                     cl_device_id                   device, 
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) { *errcode_ret = CL_INVALID_CONTEXT;   return NULL; }
   gpgpusim_opencl_warning(__my_func__,__LINE__, "assuming device_id is in context");
   if( (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ) 
      gpgpusim_opencl_warning(__my_func__,__LINE__, "ignoring command queue property");
   if( (properties & CL_QUEUE_PROFILING_ENABLE) )
      gpgpusim_opencl_warning(__my_func__,__LINE__, "ignoring command queue property");
   if( errcode_ret )
       *errcode_ret = CL_SUCCESS;
   return new _cl_command_queue(context,device,properties);
}

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) { *errcode_ret = CL_INVALID_CONTEXT;   return NULL; }
   return context->CreateBuffer(flags,size,host_ptr,errcode_ret);
}

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) { *errcode_ret = CL_INVALID_CONTEXT;   return NULL; }
   *errcode_ret = CL_SUCCESS;
   return new _cl_program(context,count,strings,lengths);
}


extern CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options, 
               void (*pfn_notify)(cl_program /* program */, void * /* user_data */),
               void *               user_data ) CL_API_SUFFIX__VERSION_1_0
{
   if( !program ) return CL_INVALID_PROGRAM;
   program->Build(options);
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( kernel_name == NULL ) {
      *errcode_ret = CL_INVALID_KERNEL_NAME;
      return NULL;
   }
   cl_kernel kobj = program->CreateKernel(kernel_name,errcode_ret);
   return kobj;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value ) CL_API_SUFFIX__VERSION_1_0
{
   kernel->SetKernelArg(arg_index,arg_size,arg_value);
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
   int _global_size[3];
   int zeros[3] = { 0, 0, 0};
   printf("\n\n\n");
   char *mode = getenv("PTX_SIM_MODE_FUNC");
   if ( mode )
      sscanf(mode,"%u", &g_ptx_sim_mode);
   printf("GPGPU-Sim OpenCL API: clEnqueueNDRangeKernel '%s' (mode=%s)\n", kernel->name().c_str(),
          g_ptx_sim_mode?"functional simulation":"performance simulation");
   if ( !work_dim || work_dim > 3 ) return CL_INVALID_WORK_DIMENSION;
   for ( unsigned d=0; d < work_dim; d++ ) {
      _global_size[d] = (int)global_work_size[d];
      if ( (global_work_size[d] % local_work_size[d]) != 0 )
         return CL_INVALID_WORK_GROUP_SIZE;
   }

   assert( global_work_size[0] == local_work_size[0] * (global_work_size[0]/local_work_size[0]) ); // i.e., we can divide into equal CTAs
   g_cudaGridDim.x = global_work_size[0]/local_work_size[0];
   g_cudaGridDim.y = (work_dim < 2)?1:(global_work_size[1]/local_work_size[1]);
   g_cudaGridDim.z = (work_dim < 3)?1:(global_work_size[2]/local_work_size[2]);
   g_cudaBlockDim.x = local_work_size[0];
   g_cudaBlockDim.y = (work_dim < 2)?1:local_work_size[1];
   g_cudaBlockDim.z = (work_dim < 3)?1:local_work_size[2];

   cl_int err_val = kernel->bind_args(&g_ptx_sim_params);
   if ( err_val != CL_SUCCESS ) {
      return err_val;
   }

   gpgpu_ptx_sim_memcpy_symbol( "%_global_size", _global_size, 3 * sizeof(int), 0, 1 );
   gpgpu_ptx_sim_memcpy_symbol( "%_work_dim", &work_dim, 1 * sizeof(int), 0, 1  );
   gpgpu_ptx_sim_memcpy_symbol( "%_global_num_groups", &g_cudaGridDim, 3 * sizeof(int), 0, 1  );
   gpgpu_ptx_sim_memcpy_symbol( "%_global_launch_offset", zeros, 3 * sizeof(int), 0, 1  );
   gpgpu_ptx_sim_memcpy_symbol( "%_global_block_offset", zeros, 3 * sizeof(int), 0, 1  );

   if ( g_ptx_sim_mode )
      gpgpu_ptx_sim_main_func( (const char*)kernel, g_cudaGridDim, g_cudaBlockDim, g_ptx_sim_params );
   else
      gpgpu_ptx_sim_main_perf( (const char*)kernel, g_cudaGridDim, g_cudaBlockDim, g_ptx_sim_params );
   g_ptx_sim_params=NULL;
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              cb, 
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event ) CL_API_SUFFIX__VERSION_1_0
{
   if( !blocking_read ) 
      gpgpusim_opencl_warning(__my_func__,__LINE__, "non-blocking read treated as blocking read");
   gpgpu_ptx_sim_memcpy_from_gpu( ptr, (size_t)buffer, cb );
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue   command_queue, 
                     cl_mem             buffer, 
                     cl_bool            blocking_write, 
                     size_t             offset, 
                     size_t             cb, 
                     const void *       ptr, 
                     cl_uint            num_events_in_wait_list, 
                     const cl_event *   event_wait_list, 
                     cl_event *         event ) CL_API_SUFFIX__VERSION_1_0
{
   if( !blocking_write ) 
      gpgpusim_opencl_warning(__my_func__,__LINE__, "non-blocking write treated as blocking write");
   gpgpu_ptx_sim_memcpy_to_gpu( (size_t)buffer, ptr, cb );
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel   /* kernel */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program /* program */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context /* context */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms ) CL_API_SUFFIX__VERSION_1_0
{
   if( ((num_entries == 0) && (platforms != NULL)) ||
       ((num_platforms == NULL) && (platforms == NULL)) ) 
      return CL_INVALID_VALUE;
   if( (platforms != NULL) && (num_entries > 0) ) 
      platforms[0] = &g_gpgpu_sim_platform_id;
   if( num_platforms )
      *num_platforms = 1;
   return CL_SUCCESS;
}

#define CL_STRING_CASE( S ) \
      if( param_value && (param_value_size < strlen(S)+1) ) return CL_INVALID_VALUE; \
      if( param_value ) snprintf(buf,strlen(S)+1,S); \
      if( param_value_size_ret ) *param_value_size_ret = strlen(S)+1; 

#define CL_INT_CASE( N ) \
      if( param_value && param_value_size < sizeof(cl_int) ) return CL_INVALID_VALUE; \
      if( param_value ) *((cl_int*)param_value) = (N); \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_int);

#define CL_ULONG_CASE( N ) \
      if( param_value && param_value_size < sizeof(cl_ulong) ) return CL_INVALID_VALUE; \
      if( param_value ) *((cl_ulong*)param_value) = (N); \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_ulong);

#define CL_SIZE_CASE( N ) \
      if( param_value && param_value_size < sizeof(size_t) ) return CL_INVALID_VALUE; \
      if( param_value ) *((size_t*)param_value) = (N); \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(size_t);

#define CL_CASE( T, N ) \
      if( param_value && param_value_size < sizeof(T) ) return CL_INVALID_VALUE; \
      if( param_value ) *((T*)param_value) = (N); \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(T);

extern CL_API_ENTRY cl_int CL_API_CALL 
clGetPlatformInfo(cl_platform_id   platform, 
                  cl_platform_info param_name,
                  size_t           param_value_size, 
                  void *           param_value,
                  size_t *         param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( platform == NULL || platform->m_uid != 0 ) 
      return CL_INVALID_PLATFORM;
   char *buf = (char*)param_value;
   switch( param_name ) {
   case CL_PLATFORM_PROFILE:    CL_STRING_CASE("FULL_PROFILE"); break;
   case CL_PLATFORM_VERSION:    CL_STRING_CASE("OpenCL 1.0"); break;
   case CL_PLATFORM_NAME:       CL_STRING_CASE("GPGPU-Sim"); break;
   case CL_PLATFORM_VENDOR:     CL_STRING_CASE("GPGPU-Sim.org"); break;
   case CL_PLATFORM_EXTENSIONS: CL_STRING_CASE(" "); break;
   default:
      return CL_INVALID_VALUE;
   }
   return CL_SUCCESS;
}

#define NUM_DEVICES 1

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type, 
               cl_uint          num_entries, 
               cl_device_id *   devices, 
               cl_uint *        num_devices ) CL_API_SUFFIX__VERSION_1_0
{
   if( platform == NULL || platform->m_uid != 0 ) 
      return CL_INVALID_PLATFORM;
   if( (num_entries == 0 && devices != NULL) ||
       (num_devices == NULL && devices == NULL) )
      return CL_INVALID_VALUE;

   switch( device_type ) {
   case CL_DEVICE_TYPE_CPU: 
      opencl_not_implemented(__my_func__,__LINE__);
      break;
   case CL_DEVICE_TYPE_DEFAULT:
   case CL_DEVICE_TYPE_GPU: 
   case CL_DEVICE_TYPE_ACCELERATOR:
      if( devices != NULL ) 
         devices[0] = &g_gpgpusim_cl_device_id;
      if( num_devices ) 
         *num_devices = NUM_DEVICES;
      break;
   case CL_DEVICE_TYPE_ALL:
      opencl_not_implemented(__my_func__,__LINE__);
      break;
   default:
      return CL_INVALID_DEVICE_TYPE;
   }
   return CL_SUCCESS;
}

extern unsigned int gpu_n_shader;
extern double core_freq;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name, 
                size_t          param_value_size, 
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( device != &g_gpgpusim_cl_device_id ) 
      return CL_INVALID_DEVICE;
   char *buf = (char*)param_value;
   switch( param_name ) {
   case CL_DEVICE_NAME: CL_STRING_CASE( "GPGPU-Sim" ); break;
   case CL_DEVICE_GLOBAL_MEM_SIZE: CL_ULONG_CASE( 1024*1024*1024 ); break;
   case CL_DEVICE_MAX_COMPUTE_UNITS: CL_INT_CASE( gpu_n_shader ); break;
   case CL_DEVICE_MAX_CLOCK_FREQUENCY: CL_INT_CASE( (cl_int)core_freq ); break;
   case CL_DEVICE_VENDOR:CL_STRING_CASE("GPGPU-Sim.org"); break;
   case CL_DRIVER_VERSION: CL_STRING_CASE("1.0"); break;
   case CL_DEVICE_TYPE: CL_INT_CASE(CL_DEVICE_TYPE_GPU); break;
   case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: CL_INT_CASE( 3 ); break;
   case CL_DEVICE_MAX_WORK_ITEM_SIZES: 
      if( param_value && param_value_size < 3*sizeof(size_t) ) return CL_INVALID_VALUE; \
      if( param_value ) {
         ((size_t*)param_value)[0] = gpu_n_thread_per_shader;
         ((size_t*)param_value)[1] = gpu_n_thread_per_shader;
         ((size_t*)param_value)[2] = gpu_n_thread_per_shader;
      }
      if( param_value_size_ret ) *param_value_size_ret = 3*sizeof(cl_uint);
      break;
   case CL_DEVICE_MAX_WORK_GROUP_SIZE: CL_INT_CASE( gpu_n_thread_per_shader ); break;
   case CL_DEVICE_ADDRESS_BITS: CL_INT_CASE( 32 ); break;
   case CL_DEVICE_IMAGE_SUPPORT: CL_INT_CASE( CL_TRUE ); break;
   case CL_DEVICE_MAX_READ_IMAGE_ARGS: CL_INT_CASE( 128 ); break;
   case CL_DEVICE_MAX_WRITE_IMAGE_ARGS: CL_INT_CASE( 8 ); break;
   case CL_DEVICE_IMAGE2D_MAX_HEIGHT: CL_INT_CASE( 8192 ); break;
   case CL_DEVICE_IMAGE2D_MAX_WIDTH: CL_INT_CASE( 8192 ); break;
   case CL_DEVICE_IMAGE3D_MAX_HEIGHT: CL_INT_CASE( 2048 ); break;
   case CL_DEVICE_IMAGE3D_MAX_WIDTH: CL_INT_CASE( 2048 ); break;
   case CL_DEVICE_IMAGE3D_MAX_DEPTH: CL_INT_CASE( 2048 ); break;
   case CL_DEVICE_MAX_MEM_ALLOC_SIZE: CL_INT_CASE( 128*1024*1024 ); break;
   case CL_DEVICE_ERROR_CORRECTION_SUPPORT: CL_INT_CASE( 0 ); break;
   case CL_DEVICE_LOCAL_MEM_TYPE: CL_INT_CASE( CL_LOCAL ); break;
   case CL_DEVICE_LOCAL_MEM_SIZE: CL_ULONG_CASE( gpgpu_shmem_size ); break;
   case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: CL_ULONG_CASE( 64 * 1024 ); break;
   case CL_DEVICE_QUEUE_PROPERTIES: CL_INT_CASE( CL_QUEUE_PROFILING_ENABLE ); break;
   case CL_DEVICE_EXTENSIONS: 
      if( param_value && (param_value_size < 1) ) return CL_INVALID_VALUE; 
      if( param_value ) buf[0]=0;
      if( param_value_size_ret ) *param_value_size_ret = 1; 
      break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: CL_INT_CASE(1); break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: CL_INT_CASE(1); break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: CL_INT_CASE(1); break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: CL_INT_CASE(1); break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: CL_INT_CASE(1); break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: CL_INT_CASE(0); break;
   default:
      opencl_not_implemented(__my_func__,__LINE__);
   }
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( program == NULL ) 
      return CL_INVALID_PROGRAM;
   char *tmp=NULL;
   size_t len=0;
   switch( param_name ) {
   case CL_PROGRAM_REFERENCE_COUNT: 
      CL_INT_CASE(1);
      break;
   case CL_PROGRAM_CONTEXT:
      if( param_value && param_value_size < sizeof(cl_context)) return CL_INVALID_VALUE;
      if( param_value ) *((cl_context*)param_value) = program->get_context();
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_context);
      break;
   case CL_PROGRAM_NUM_DEVICES:
      CL_INT_CASE(NUM_DEVICES);
      break;
   case CL_PROGRAM_DEVICES:
      if( param_value && param_value_size < NUM_DEVICES * sizeof(cl_device_id) ) 
         return CL_INVALID_VALUE;
      if( param_value ) {
         assert( NUM_DEVICES == 1 );
         ((cl_device_id*)param_value)[0] = &g_gpgpusim_cl_device_id;
      }
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_device_id);
      break;
   case CL_PROGRAM_SOURCE:
      opencl_not_implemented(__my_func__,__LINE__);
      break;
   case CL_PROGRAM_BINARY_SIZES:
      if( param_value && param_value_size < NUM_DEVICES * sizeof(size_t) ) return CL_INVALID_VALUE;
      if( param_value ) *((size_t*)param_value) = program->get_ptx_size();
      if( param_value_size_ret ) *param_value_size_ret = NUM_DEVICES*sizeof(size_t);
      break;
   case CL_PROGRAM_BINARIES:
      len = program->get_ptx_size();
      if( param_value && param_value_size < NUM_DEVICES * len ) return CL_INVALID_VALUE;
      tmp = program->get_ptx();
      if( param_value ) memcpy( ((char**)param_value)[0], tmp, len );
      if( param_value_size_ret ) *param_value_size_ret = len;
      break;
   default:
      return CL_INVALID_VALUE;
      break;
   }
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue    command_queue, 
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer, 
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event ) CL_API_SUFFIX__VERSION_1_0
{
   if( num_events_in_wait_list > 0 ) 
      opencl_not_implemented(__my_func__,__LINE__);
   if( command_queue == NULL || !command_queue->is_valid() ) 
      return CL_INVALID_COMMAND_QUEUE;
   cl_context context = command_queue->get_context();
   cl_mem src = context->lookup_mem( src_buffer );
   cl_mem dst = context->lookup_mem( dst_buffer );
   if( src == NULL || dst == NULL ) 
      return CL_INVALID_MEM_OBJECT;

   if( src->is_on_host() && !dst->is_on_host() )
      gpgpu_ptx_sim_memcpy_to_gpu( ((size_t)dst->device_ptr())+dst_offset, ((char*)src->host_ptr())+src_offset, cb );
   else if( !src->is_on_host() && dst->is_on_host() ) 
      gpgpu_ptx_sim_memcpy_from_gpu( ((char*)dst->host_ptr())+dst_offset, ((size_t)src->device_ptr())+src_offset, cb );
   else if( !src->is_on_host() && !dst->is_on_host() ) 
      gpgpu_ptx_sim_memcpy_gpu_to_gpu( ((size_t)dst->device_ptr())+dst_offset, ((size_t)src->device_ptr())+src_offset, cb );
   else
      opencl_not_implemented(__my_func__,__LINE__);
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( kernel == NULL ) 
      return CL_INVALID_KERNEL;
   switch( param_name ) {
   case CL_KERNEL_WORK_GROUP_SIZE:
      CL_SIZE_CASE( kernel->get_workgroup_size() );
      break;
   case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
   case CL_KERNEL_LOCAL_MEM_SIZE:
      opencl_not_implemented(__my_func__,__LINE__);
      break;
   default:
      return CL_INVALID_VALUE;
      break;
   }
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint             /* num_events */,
                const cl_event *    /* event_list */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event /* event */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( command_queue == NULL ) 
      return CL_INVALID_COMMAND_QUEUE;
   switch( param_name ) {
   case CL_QUEUE_CONTEXT: CL_CASE(cl_context, command_queue->get_context()); break;
   case CL_QUEUE_DEVICE: CL_CASE(cl_device_id, command_queue->get_device()); break;
   case CL_QUEUE_REFERENCE_COUNT: CL_CASE(cl_uint,1); break;
   case CL_QUEUE_PROPERTIES: CL_CASE(cl_command_queue_properties, command_queue->get_properties()); break;
   default:
      return CL_INVALID_VALUE;
   }
   return CL_SUCCESS;
}
