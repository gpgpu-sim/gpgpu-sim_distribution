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
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use
 * only.
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

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>  // Apple's version of GLUT is here
#else
#include <GL/gl.h>
#endif
#endif

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "cuda_api.h"
#include "cudaProfiler.h"
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
#endif
#include "gpgpu_context.h"
#include "cuda_api_object.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpusim_entrypoint.h"
#include "../src/stream_manager.h"
#include "../src/abstract_hardware_model.h"

#include <pthread.h>
#include <semaphore.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

/*DEVICE_BUILTIN*/
struct cudaArray {
  void *devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size;  // in bytes
  unsigned dimensions;
};

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) = v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

cudaError_t g_last_cudaError = cudaSuccess;

void register_ptx_function(const char *name, function_info *impl) {
  // no longer need this
}

#if defined __APPLE__
#define __my_func__ __PRETTY_FUNCTION__
#else
#if defined __cplusplus ? __GNUC_PREREQ(2, 6) : __GNUC_PREREQ(2, 4)
#define __my_func__ __PRETTY_FUNCTION__
#else
#if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#define __my_func__ __func__
#else
#define __my_func__ ((__const char *)0)
#endif
#endif
#endif

struct _cuda_device_id *gpgpu_context::GPGPUSim_Init() {
  _cuda_device_id *the_device = the_gpgpusim->the_cude_device;
  if (!the_device) {
    gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();

    cudaDeviceProp *prop = (cudaDeviceProp *)calloc(sizeof(cudaDeviceProp), 1);
    snprintf(prop->name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
    prop->major = the_gpu->compute_capability_major();
    prop->minor = the_gpu->compute_capability_minor();
    prop->totalGlobalMem = 0x80000000 /* 2 GB */;
    prop->memPitch = 0;
    if (prop->major >= 2) {
      prop->maxThreadsPerBlock = 1024;
      prop->maxThreadsDim[0] = 1024;
      prop->maxThreadsDim[1] = 1024;
    } else {
      prop->maxThreadsPerBlock = 512;
      prop->maxThreadsDim[0] = 512;
      prop->maxThreadsDim[1] = 512;
    }

    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 0x40000000;
    prop->maxGridSize[1] = 0x40000000;
    prop->maxGridSize[2] = 0x40000000;
    prop->totalConstMem = 0x40000000;
    prop->textureAlignment = 0;
    //        * TODO: Update the .config and xml files of all GPU config files
    //        with new value of sharedMemPerBlock and regsPerBlock
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
#if (CUDART_VERSION > 5050)
    prop->regsPerMultiprocessor = the_gpu->num_registers_per_core();
    prop->sharedMemPerMultiprocessor = the_gpu->shared_mem_size();
#endif
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    prop->regsPerBlock = the_gpu->num_registers_per_block();
    prop->warpSize = the_gpu->wrp_size();
    prop->clockRate = the_gpu->shader_clock();
#if (CUDART_VERSION >= 2010)
    prop->multiProcessorCount = the_gpu->get_config().num_shader();
#endif
#if (CUDART_VERSION >= 4000)
    prop->maxThreadsPerMultiProcessor = the_gpu->threads_per_core();
#endif
    the_gpu->set_prop(prop);
    the_gpgpusim->the_cude_device = new _cuda_device_id(the_gpu);
    the_device = the_gpgpusim->the_cude_device;
  }
  start_sim_thread(1);
  return the_device;
}

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx) {
  // static CUctx_st *the_context = NULL;
  CUctx_st *the_context = ctx->the_gpgpusim->the_context;
  if (the_context == NULL) {
    _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
    ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
    the_context = ctx->the_gpgpusim->the_context;
  }
  return the_context;
}

gpgpu_context *GPGPU_Context() {
  static gpgpu_context *gpgpu_ctx = NULL;
  if (gpgpu_ctx == NULL) {
    gpgpu_ctx = new gpgpu_context();
  }
  return gpgpu_ctx;
}

void ptxinfo_data::ptxinfo_addinfo() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  if (!get_ptxinfo_kname()) {
    /* This info is not per kernel (since CUDA 5.0 some info (e.g. gmem, and
     * cmem) is added at the beginning for the whole binary ) */
    print_ptxinfo();
    context->add_ptxinfo(get_ptxinfo());
    clear_ptxinfo();
    return;
  }
  if (!strcmp("__cuda_dummy_entry__", get_ptxinfo_kname())) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  print_ptxinfo();
  context->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo());
  clear_ptxinfo();
}

void cuda_not_implemented(const char *func, unsigned line) {
  fflush(stdout);
  fflush(stderr);
  printf(
      "\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not "
      "been implemented yet.\n"
      "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n",
      func, __FILE__, line);
  fflush(stdout);
  abort();
}

void announce_call(const char *func) {
  printf("\n\nGPGPU-Sim PTX: CUDA API function \"%s\" has been called.\n",
         func);
  fflush(stdout);
}

#define gpgpusim_ptx_error(msg, ...) \
  gpgpusim_ptx_error_impl(__func__, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define gpgpusim_ptx_assert(cond, msg, ...)                           \
  gpgpusim_ptx_assert_impl((cond), __func__, __FILE__, __LINE__, msg, \
                           ##__VA_ARGS__)

void gpgpusim_ptx_error_impl(const char *func, const char *file, unsigned line,
                             const char *msg, ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  printf("GPGPU-Sim CUDA API: %s\n", buf);
  printf("                    [%s:%u : %s]\n", file, line, func);
  abort();
}

void gpgpusim_ptx_assert_impl(int test_value, const char *func,
                              const char *file, unsigned line, const char *msg,
                              ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  if (test_value == 0) gpgpusim_ptx_error_impl(func, file, line, msg);
}

typedef std::map<unsigned, CUevent_st *> event_tracker_t;

int CUevent_st::m_next_event_uid;
event_tracker_t g_timer_events;

extern int cuobjdump_lex_init(yyscan_t *scanner);
extern void cuobjdump_set_in(FILE *_in_str, yyscan_t yyscanner);
extern int cuobjdump_parse(yyscan_t scanner, struct cuobjdump_parser *parser,
                           std::list<cuobjdumpSection *> &cuobjdumpSectionList);
extern int cuobjdump_lex_destroy(yyscan_t scanner);

enum cuobjdumpSectionType { PTXSECTION = 0, ELFSECTION };

// sectiontype: 0 for ptx, 1 for elf
void addCuobjdumpSection(int sectiontype,
                         std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (sectiontype)
    cuobjdumpSectionList.push_front(new cuobjdumpELFSection());
  else
    cuobjdumpSectionList.push_front(new cuobjdumpPTXSection());
  printf("## Adding new section %s\n", sectiontype ? "ELF" : "PTX");
}

void setCuobjdumparch(const char *arch,
                      std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  unsigned archnum;
  sscanf(arch, "sm_%u", &archnum);
  assert(archnum && "cannot have sm_0");
  printf("Adding arch: %s\n", arch);
  cuobjdumpSectionList.front()->setArch(archnum);
}

void setCuobjdumpidentifier(
    const char *identifier,
    std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  printf("Adding identifier: %s\n", identifier);
  cuobjdumpSectionList.front()->setIdentifier(identifier);
}

void setCuobjdumpptxfilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  printf("Adding ptx filename: %s\n", filename);
  cuobjdumpSection *x = cuobjdumpSectionList.front();
  if (dynamic_cast<cuobjdumpPTXSection *>(x) == NULL) {
    assert(0 &&
           "You shouldn't be trying to add a ptxfilename to an elf section");
  }
  (dynamic_cast<cuobjdumpPTXSection *>(x))->setPTXfilename(filename);
}

void setCuobjdumpelffilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()) ==
      NULL) {
    assert(0 &&
           "You shouldn't be trying to add a elffilename to an ptx section");
  }
  (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()))
      ->setELFfilename(filename);
}

void setCuobjdumpsassfilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()) ==
      NULL) {
    assert(0 &&
           "You shouldn't be trying to add a sassfilename to an ptx section");
  }
  (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()))
      ->setSASSfilename(filename);
}

//! Return the executable file of the process containing the PTX/SASS code
//!
//! This Function returns the executable file ran by the process.  This
//! executable is supposed to contain the PTX/SASS code.  It provides workaround
//! for processes running on valgrind by dereferencing /proc/<pid>/exe within
//! the GPGPU-Sim process before calling cuobjdump to extract PTX/SASS.  This is
//! needed because valgrind uses x86 emulation to detect memory leak.  Other
//! processes (e.g. cuobjdump) reading /proc/<pid>/exe will see the emulator
//! executable instead of the application binary.
//!
std::string get_app_binary() {
  char self_exe_path[1025];
#ifdef __APPLE__
  uint32_t size = sizeof(self_exe_path);
  if (_NSGetExecutablePath(self_exe_path, &size) != 0) {
    printf("GPGPU-Sim ** ERROR: _NSGetExecutablePath input buffer too small\n");
    exit(1);
  }
#else
  std::stringstream exec_link;
  exec_link << "/proc/self/exe";

  ssize_t path_length = readlink(exec_link.str().c_str(), self_exe_path, 1024);
  assert(path_length != -1);
  self_exe_path[path_length] = '\0';
#endif

  printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}

// above func gives abs path whereas this give just the name of application.
char *get_app_binary_name(std::string abs_path) {
  char *self_exe_path;
#ifdef __APPLE__
  // TODO: get apple device and check the result.
  printf("WARNING: not tested for Apple-mac devices \n");
  abort();
#else
  char *buf = strdup(abs_path.c_str());
  char *token = strtok(buf, "/");
  while (token != NULL) {
    self_exe_path = token;
    token = strtok(NULL, "/");
  }
#endif
  self_exe_path = strtok(self_exe_path, ".");
  printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}

static int get_app_cuda_version() {
  int app_cuda_version = 0;
  char fname[1024];
  snprintf(fname, 1024, "_app_cuda_version_XXXXXX");
  int fd = mkstemp(fname);
  close(fd);
  std::string app_cuda_version_command =
      "ldd " + get_app_binary() +
      " | grep libcudart.so | sed  's/.*libcudart.so.\\(.*\\) =>.*/\\1/' > " +
      fname;
  system(app_cuda_version_command.c_str());
  FILE *cmd = fopen(fname, "r");
  char buf[256];
  while (fgets(buf, sizeof(buf), cmd) != 0) {
    std::cout << buf;
    app_cuda_version = atoi(buf);
  }
  fclose(cmd);
  if (app_cuda_version == 0) {
    printf("Error - Cannot detect the app's CUDA version.\n");
    exit(1);
  }
  return app_cuda_version;
}

//! Keep track of the association between filename and cubin handle
void cuda_runtime_api::cuobjdumpRegisterFatBinary(unsigned int handle,
                                                  const char *filename,
                                                  CUctx_st *context) {
  fatbinmap[handle] = filename;
}

/*******************************************************************************
 * Add internal cuda runtime API call to accept gpgpu_context *
 *******************************************************************************/
cudaError_t cudaSetDeviceInternal(int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // set the active device to run cuda
  if (device <= ctx->GPGPUSim_Init()->num_devices()) {
    ctx->api->g_active_device = device;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

cudaError_t cudaGetDeviceInternal(int *device,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *device = ctx->api->g_active_device;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimitInternal(
    size_t *pValue, cudaLimit limit, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  const struct cudaDeviceProp *prop = dev->get_prop();
  const gpgpu_sim_config &config = dev->get_gpgpu()->get_config();
  switch (limit) {
    case 0:  // cudaLimitStackSize
      *pValue = config.stack_limit();
      break;
    case 2:  // cudaLimitMallocHeapSize
      *pValue = config.heap_limit();
      break;
#if (CUDART_VERSION > 5050)
    case 3:  // cudaLimitDevRuntimeSyncDepth
      if (prop->major > 2) {
        *pValue = config.sync_depth_limit();
        break;
      } else {
        printf("ERROR:Limit %d is not supported on this architecture \n",
               limit);
        abort();
      }
    case 4:  // cudaLimitDevRuntimePendingLaunchCount
      if (prop->major > 2) {
        *pValue = config.pending_launch_count_limit();
        break;
      } else {
        printf("ERROR:Limit %d is not supported on this architecture \n",
               limit);
        abort();
      }
#endif
    default:
      printf("ERROR:Limit %d unimplemented \n", limit);
      abort();
  }
  return g_last_cudaError = cudaSuccess;
}

void **cudaRegisterFatBinaryInternal(void *fatCubin,
                                     gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION < 2010)
  printf(
      "GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or "
      "higher\n");
  exit(1);
#endif
  CUctx_st *context = GPGPUSim_Context(ctx);
  static unsigned next_fat_bin_handle = 1;
  if (context->get_device()->get_gpgpu()->get_config().use_cuobjdump()) {
    // The following workaround has only been verified on 64-bit systems.
    if (sizeof(void *) == 4)
      printf(
          "GPGPU-Sim PTX: FatBin file name extraction has not been tested on "
          "32-bit system.\n");

    // This code will get the CUDA version the app was compiled with.
    // We need this to determine how to handle the parsing of the binary.
    // Making this a runtime variable based on the app, enables GPGPU-Sim
    // compiled with a newer version of CUDA to run apps compiled with older
    // versions of CUDA. This is especially useful for PTXPLUS execution.
    // Skip cuda version check for pytorch application
    std::string app_binary_path = get_app_binary();
    int pos = app_binary_path.find("python");
    if (pos == std::string::npos) {
      // Not pytorch app : checking cuda version
      int app_cuda_version = get_app_cuda_version();
      assert(
          app_cuda_version == CUDART_VERSION / 1000 &&
          "The app must be compiled with same major version as the simulator.");
    }

    // int app_cuda_version = get_app_cuda_version();
    // assert( app_cuda_version == CUDART_VERSION / 1000  && "The app must be
    // compiled with same major version as the simulator." );
    const char *filename;
#if CUDART_VERSION < 6000
    // FatBin handle from the .fatbin.c file (one of the intermediate files
    // generated by NVCC)
    typedef struct {
      int m;
      int v;
      const unsigned long long *d;
      char *f;
    } __fatDeviceText __attribute__((aligned(8)));
    __fatDeviceText *fatDeviceText = (__fatDeviceText *)fatCubin;

    // Extract the source code file name that generate the given FatBin.
    // - Obtains the pointer to the actual fatbin structure from the FatBin
    // handle (fatCubin).
    // - An integer inside the fatbin structure contains the relative offset to
    // the source code file name.
    // - This offset differs among different CUDA and GCC versions.
    char *pfatbin = (char *)fatDeviceText->d;
    int offset = *((int *)(pfatbin + 48));
    filename = (pfatbin + 16 + offset);
#else
    filename = "default";
#endif

    // The extracted file name is associated with a fat_cubin_handle passed
    // into cudaLaunch().  Inside cudaLaunch(), the associated file name is
    // used to find the PTX/SASS section from cuobjdump, which contains the
    // PTX/SASS code for the launched kernel function.
    // This allows us to work around the fact that cuobjdump only outputs the
    // file name associated with each section.
    unsigned long long fat_cubin_handle = next_fat_bin_handle;
    next_fat_bin_handle++;
    printf(
        "GPGPU-Sim PTX: __cudaRegisterFatBinary, fat_cubin_handle = %llu, "
        "filename=%s\n",
        fat_cubin_handle, filename);
    /*!
     * This function extracts all data from all files in first call
     * then for next calls, only returns the appropriate number
     */
    assert(fat_cubin_handle >= 1);
    if (fat_cubin_handle == 1) ctx->api->cuobjdumpInit();
    ctx->api->cuobjdumpRegisterFatBinary(fat_cubin_handle, filename, context);

    return (void **)fat_cubin_handle;
  }
#if (CUDART_VERSION < 8000)
  else {
    static unsigned source_num = 1;
    unsigned long long fat_cubin_handle = next_fat_bin_handle++;
    __cudaFatCudaBinary *info = (__cudaFatCudaBinary *)fatCubin;
    assert(info->version >= 3);
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 0;
    unsigned selected_capability = 0;
    bool found = false;
    unsigned forced_max_capability = context->get_device()
                                         ->get_gpgpu()
                                         ->get_config()
                                         .get_forced_max_capability();
    if (!info->ptx) {
      printf(
          "ERROR: Cannot find ptx code in cubin file\n"
          "\tIf you are using CUDA 4.0 or higher, please enable "
          "-gpgpu_ptx_use_cuobjdump or downgrade to CUDA 3.1\n");
      exit(1);
    }
    while (info->ptx[num_ptx_versions].gpuProfileName != NULL) {
      unsigned capability = 0;
      sscanf(info->ptx[num_ptx_versions].gpuProfileName, "compute_%u",
             &capability);
      printf(
          "GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for "
          "'%s', ",
          info->ident);
      printf("capability = %s\n", info->ptx[num_ptx_versions].gpuProfileName);
      if (forced_max_capability) {
        if (capability > max_capability &&
            capability <= forced_max_capability) {
          found = true;
          max_capability = capability;
          selected_capability = num_ptx_versions;
        }
      } else {
        if (capability > max_capability) {
          found = true;
          max_capability = capability;
          selected_capability = num_ptx_versions;
        }
      }
      num_ptx_versions++;
    }
    if (found) {
      printf("GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
             info->ident, info->ptx[selected_capability].gpuProfileName);
      symbol_table *symtab;
      const char *ptx = info->ptx[selected_capability].ptx;
      if (context->get_device()
              ->get_gpgpu()
              ->get_config()
              .convert_to_ptxplus()) {
        printf(
            "GPGPU-Sim PTX: ERROR ** PTXPlus is only supported through "
            "cuobjdump\n"
            "\tEither enable cuobjdump or disable PTXPlus in your "
            "configuration file\n");
        exit(1);
      } else {
        symtab = ctx->gpgpu_ptx_sim_load_ptx_from_string(ptx, source_num);
        context->add_binary(symtab, fat_cubin_handle);
        ctx->gpgpu_ptxinfo_load_from_string(ptx, source_num, max_capability,
                                            context->no_of_ptx);
      }
      source_num++;
      ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                                    context->get_device()->get_gpgpu());
      ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                               context->get_device()->get_gpgpu());
    } else {
      printf(
          "GPGPU-Sim PTX: warning -- did not find an appropriate PTX in "
          "cubin\n");
    }
    return (void **)fat_cubin_handle;
  }
#else
  else {
    printf("ERROR **  __cudaRegisterFatBinary() needs to be updated\n");
    abort();
  }
#endif
}

void cudaRegisterFunctionInternal(void **fatCubinHandle, const char *hostFun,
                                  char *deviceFun, const char *deviceName,
                                  int thread_limit, uint3 *tid, uint3 *bid,
                                  dim3 *bDim, dim3 *gDim,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;
  printf(
      "GPGPU-Sim PTX: __cudaRegisterFunction %s : hostFun 0x%p, "
      "fat_cubin_handle = %u\n",
      deviceFun, hostFun, fat_cubin_handle);
  if (context->get_device()->get_gpgpu()->get_config().use_cuobjdump())
    ctx->cuobjdumpParseBinary(fat_cubin_handle);
  context->register_function(fat_cubin_handle, hostFun, deviceFun);
}

void cudaRegisterVarInternal(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf(
      "GPGPU-Sim PTX: __cudaRegisterVar: hostVar = %p; deviceAddress = %s; "
      "deviceName = %s\n",
      hostVar, deviceAddress, deviceName);
  printf(
      "GPGPU-Sim PTX: __cudaRegisterVar: Registering const memory space of %d "
      "bytes\n",
      size);
  if (GPGPUSim_Context(ctx)
          ->get_device()
          ->get_gpgpu()
          ->get_config()
          .use_cuobjdump())
    ctx->cuobjdumpParseBinary((unsigned)(unsigned long long)fatCubinHandle);
  fflush(stdout);
  if (constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_const_variable(hostVar, deviceName,
                                                         size);
  } else if (!constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_global_variable(hostVar, deviceName,
                                                          size);
  } else
    cuda_not_implemented(__my_func__, __LINE__);
}

cudaError_t cudaConfigureCallInternal(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  ctx->api->g_cuda_launch_stack.push_back(
      kernel_config(gridDim, blockDim, sharedMem, s));
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceCountInternal(int *count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *count = dev->num_devices();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDevicePropertiesInternal(
    struct cudaDeviceProp *prop, int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  if (device <= dev->num_devices()) {
    *prop = *dev->get_prop();
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

__host__ cudaError_t CUDARTAPI
cudaChooseDeviceInternal(int *device, const struct cudaDeviceProp *prop,
                         gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *device = dev->get_id();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaSetupArgumentInternal(const void *arg, size_t size,
                                      size_t offset,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config &config = ctx->api->g_cuda_launch_stack.back();
  config.set_arg(arg, size, offset);
  printf(
      "GPGPU-Sim PTX: Setting up arguments for %zu bytes starting at "
      "0x%llx..\n",
      size, (unsigned long long)arg);

  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaLaunchInternal(const char *hostFun,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &(ctx->func_sim->g_ptx_sim_mode));
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config config = ctx->api->g_cuda_launch_stack.back();
  {
    dim3 gridDim = config.grid_dim();
    dim3 blockDim = config.block_dim();
    if (gridDim.x * gridDim.y * gridDim.z == 0 ||
        blockDim.x * blockDim.y * blockDim.z == 0) {
      // can't launch
      printf("can't launch a empty kernel\n");
      ctx->api->g_cuda_launch_stack.pop_back();
      return g_last_cudaError = cudaErrorInvalidConfiguration;
    }
  }
  struct CUstream_st *stream = config.get_stream();

  printf("\nGPGPU-Sim PTX: cudaLaunch for 0x%p (mode=%s) on stream %u\n",
         hostFun,
         (ctx->func_sim->g_ptx_sim_mode) ? "functional simulation"
                                         : "performance simulation",
         stream ? stream->get_uid() : 0);
  kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      hostFun, config.get_args(), config.grid_dim(), config.block_dim(),
      context);
  // do dynamic PDOM analysis for performance simulation scenario
  std::string kname = grid->name();
  function_info *kernel_func_info = grid->entry();
  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kname.c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kname.c_str());
    kernel_func_info->do_pdom();
    kernel_func_info->set_pdom();
  }
  dim3 gridDim = config.grid_dim();
  dim3 blockDim = config.block_dim();

  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();
  class memory_space *global_mem;
  global_mem = gpu->get_global_memory();

  if (gpu->resume_option == 1 && (grid->get_uid() == gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    for (int i = 0; i < gpu->resume_CTA; i++) grid->increment_cta_id();
  }
  if (gpu->resume_option == 1 && (grid->get_uid() < gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    printf("Skipping kernel %d as resuming from kernel %d\n", grid->get_uid(),
           gpu->resume_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  if (gpu->checkpoint_option == 1 &&
      (grid->get_uid() > gpu->checkpoint_kernel)) {
    printf("Skipping kernel %d as checkpoint from kernel %d\n", grid->get_uid(),
           gpu->checkpoint_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  printf(
      "GPGPU-Sim PTX: pushing kernel \'%s\' to stream %u, gridDim= (%u,%u,%u) "
      "blockDim = (%u,%u,%u) \n",
      kname.c_str(), stream ? stream->get_uid() : 0, gridDim.x, gridDim.y,
      gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
  ctx->the_gpgpusim->g_stream_manager->push(op);
  ctx->api->g_cuda_launch_stack.pop_back();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaMallocInternal(void **devPtr, size_t size,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*devPtr);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*devPtr] = size;
  }
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

cudaError_t cudaMallocHostInternal(void **ptr, size_t size,
                                   gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *ptr = malloc(size);
  if (*ptr) {
    // track pinned memory size allocated in the host so that same amount of
    // memory is also allocated in GPU.
    ctx->api->pinned_memory_size[*ptr] = size;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI
cudaMallocPitchInternal(void **devPtr, size_t *pitch, size_t width,
                        size_t height, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned malloc_width_inbytes = width;
  printf("GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(
      malloc_width_inbytes * height);
  pitch[0] = malloc_width_inbytes;
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

cudaError_t cudaHostGetDevicePointerInternal(void **pDevice, void *pHost,
                                             unsigned int flags,
                                             gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // only cpu memory allocation happens in cudaHostAlloc. Linking with device
  // pointer to pinned memory happens here.
  // TODO: once kernel is executed, the contents in global pointer of GPU must
  // be copied back to CPU host pointer!
  flags = 0;
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  std::map<void *, size_t>::const_iterator i =
      ctx->api->pinned_memory_size.find(pHost);
  assert(i != ctx->api->pinned_memory_size.end());
  size_t size = i->second;
  *pDevice = gpu->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*pDevice);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*pDevice] = size;
  }
  if (*pDevice) {
    ctx->api->pinned_memory[pHost] = pDevice;
    // Copy contents in cpu to gpu
    gpu->memcpy_to_gpu((size_t)*pDevice, pHost, size);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI cudaMallocArrayInternal(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height __dv(1), gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned size =
      width * height * ((desc->x + desc->y + desc->z + desc->w) / 8);
  CUctx_st *context = GPGPUSim_Context(ctx);
  (*array) = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  (*array)->desc = *desc;
  (*array)->width = width;
  (*array)->height = height;
  (*array)->size = size;
  (*array)->dimensions = 2;
  ((*array)->devPtr32) =
      (int)(long long)context->get_device()->get_gpgpu()->gpu_mallocarray(size);
  printf("GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n",
         ((*array)->devPtr32));
  ((*array)->devPtr) = (void *)(long long)((*array)->devPtr32);
  if (((*array)->devPtr)) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI
cudaMemcpyInternal(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  // gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
  if (kind == cudaMemcpyHostToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation(src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDeviceToHost)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, dst, count, 0));
  else if (kind == cudaMemcpyDeviceToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDefault) {
    if ((size_t)src >= GLOBAL_HEAP_START) {
      if ((size_t)dst >= GLOBAL_HEAP_START)
        ctx->the_gpgpusim->g_stream_manager->push(stream_operation(
            (size_t)src, (size_t)dst, count, 0));  // device to device
      else
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation((size_t)src, dst, count, 0));  // device to host
    } else {
      if ((size_t)dst >= GLOBAL_HEAP_START)
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation(src, (size_t)dst, count, 0));
      else {
        printf(
            "GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported transfer: host to "
            "host\n");
        abort();
      }
    }
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = count;
  printf("GPGPU-Sim PTX: cudaMemcpyToArray\n");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)(dst->devPtr), src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst->devPtr, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)(dst->devPtr), (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpyToArray - ERROR : unsupported "
        "cudaMemcpyKind\n");
    abort();
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DInternal(
    void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
    size_t height, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch * height;
  gpgpusim_ptx_assert((dpitch == spitch),
                      "different src and dst pitch not supported yet");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)dst, src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)dst, (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch * height;
  size_t channel_size = dst->desc.w + dst->desc.x + dst->desc.y + dst->desc.z;
  gpgpusim_ptx_assert(
      ((channel_size % 8) == 0),
      "none byte multiple destination channel size not supported (sz=%u)",
      channel_size);
  unsigned elem_size = channel_size / 8;
  gpgpusim_ptx_assert((dst->dimensions == 2),
                      "copy to none 2D array not supported");
  gpgpusim_ptx_assert((wOffset == 0), "non-zero wOffset not yet supported");
  gpgpusim_ptx_assert((hOffset == 0), "non-zero hOffset not yet supported");
  gpgpusim_ptx_assert((dst->height == (int)height),
                      "partial copy not supported");
  gpgpusim_ptx_assert((elem_size * dst->width == width),
                      "partial copy not supported");
  gpgpusim_ptx_assert((spitch == width), "spitch != width not supported");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)(dst->devPtr), src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst->devPtr, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)dst->devPtr, (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolInternal(
    const char *symbol, const void *src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  assert(kind == cudaMemcpyHostToDevice);
  printf("GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p\n", symbol);
  // stream_operation( const char *symbol, const void *src, size_t count, size_t
  // offset )
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(src, symbol, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,src,count,offset,1,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolInternal(
    void *dst, const char *symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  assert(kind == cudaMemcpyDeviceToHost);
  printf("GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(symbol, dst, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyAsyncInternal(
    void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  switch (kind) {
    case cudaMemcpyHostToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(src, (size_t)dst, count, s));
      break;
    case cudaMemcpyDeviceToHost:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, s));
      break;
    case cudaMemcpyDeviceToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, s));
      break;
    default:
      abort();
  }
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 8000)
cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  printf(
      "GPGPU-Sim PTX: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags "
      "%p\n",
      hostFunc);
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFunc);
  printf(
      "Calculate Maxium Active Block with function ptr=%p, blockSize=%d, "
      "SMemSize=%d\n",
      hostFunc, blockSize, dynamicSMemSize);
  if (flags == cudaOccupancyDefault) {
    // create kernel_info based on entry
    dim3 gridDim(context->get_device()->get_gpgpu()->max_cta_per_core() *
                 context->get_device()->get_gpgpu()->get_config().num_shader());
    dim3 blockDim(blockSize);
    kernel_info_t result(gridDim, blockDim, entry);
    // if(entry == NULL){
    //	*numBlocks = 1;
    //	return g_last_cudaError = cudaErrorUnknown;
    //}
    *numBlocks = context->get_device()->get_gpgpu()->get_max_cta(result);
    printf("Maximum size is %d with gridDim %d and blockDim %d\n", *numBlocks,
           gridDim.x, blockDim.x);
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

#endif

__host__ cudaError_t CUDARTAPI cudaMemsetInternal(
    void *mem, int c, size_t count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI
cudaMemsetAsyncInternal(void *mem, int c, size_t count, cudaStream_t stream = 0,
                        gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: WARNING: Asynchronous memset not supported (%s)\n",
         __my_func__);
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaGLMapBufferObjectInternal(void **devPtr, GLuint bufferObj,
                                          gpgpu_context *gpgpu_ctx = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#ifdef OPENGL_SUPPORT
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  GLint buffer_size = 0;
  CUctx_st *context = GPGPUSim_Context(ctx);

  glbmap_entry_t *p = ctx->api->g_glbmap;
  while (p && p->m_bufferObj != bufferObj) p = p->m_next;
  if (p == NULL) {
    glBindBuffer(GL_ARRAY_BUFFER, bufferObj);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &buffer_size);
    assert(buffer_size != 0);
    *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(buffer_size);

    // create entry and insert to front of list
    glbmap_entry_t *n = (glbmap_entry_t *)calloc(1, sizeof(glbmap_entry_t));
    n->m_next = ctx->api->g_glbmap;
    ctx->api->g_glbmap = n;

    // initialize entry
    n->m_bufferObj = bufferObj;
    n->m_devPtr = *devPtr;
    n->m_size = buffer_size;

    p = n;
  } else {
    buffer_size = p->m_size;
    *devPtr = p->m_devPtr;
  }

  if (*devPtr) {
    char *data = (char *)calloc(p->m_size, 1);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, data);
    memcpy_to_gpu((size_t)*devPtr, data, buffer_size);
    free(data);
    printf(
        "GPGPU-Sim PTX: cudaGLMapBufferObject %zu bytes starting at 0x%llx..\n",
        (size_t)buffer_size, (unsigned long long)*devPtr);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }

  return g_last_cudaError = cudaSuccess;
#else
  fflush(stdout);
  fflush(stderr);
  printf(
      "GPGPU-Sim PTX: GPGPU-Sim support for OpenGL integration disabled -- "
      "exiting\n");
  fflush(stdout);
  exit(50);
#endif
}

#if CUDART_VERSION >= 6050
CUresult cuLinkAddFileInternal(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  static bool addedFile = false;
  if (addedFile) {
    printf(
        "GPGPU-Sim PTX: ERROR: cuLinkAddFile does not support multiple "
        "files\n");
    abort();
  }

  // blocking
  assert(type == CU_JIT_INPUT_PTX);
  CUctx_st *context = GPGPUSim_Context(ctx);
  char *file = getenv("PTX_JIT_PATH");
  if (file == NULL) {
    printf("GPGPU-Sim PTX: ERROR: PTX_JIT_PATH has not been set\n");
    abort();
  }
  strcat(file, "/");
  strcat(file, path);
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(file);
  std::string fname(path);
  ctx->api->name_symtab[fname] = symtab;
  context->add_binary(symtab, 1);
  ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                                context->get_device()->get_gpgpu());
  ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                           context->get_device()->get_gpgpu());
  addedFile = true;
  return CUDA_SUCCESS;
}
#endif

#if (CUDART_VERSION >= 2010)

cudaError_t cudaHostAllocInternal(void **pHost, size_t bytes,
                                  unsigned int flags,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *pHost = malloc(bytes);
  // need to track the size allocated so that cudaHostGetDevicePointer() can
  // function properly.
  // TODO: vary this function behavior based on flags value (following nvidia
  // documentation)
  ctx->api->pinned_memory_size[*pHost] = bytes;
  if (*pHost)
    return g_last_cudaError = cudaSuccess;
  else
    return g_last_cudaError = cudaErrorMemoryAllocation;
}

#endif

size_t getMaxThreadsPerBlock(struct cudaFuncAttributes *attr,
                             gpgpu_context *ctx) {
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  struct cudaDeviceProp prop;

  prop = *dev->get_prop();

  size_t max = prop.maxThreadsPerBlock;

  if (attr->numRegs && (prop.regsPerBlock / attr->numRegs) < max)
    max = prop.regsPerBlock / attr->numRegs;

  if (attr->sharedSizeBytes &&
      (prop.sharedMemPerBlock / attr->sharedSizeBytes) < max)
    max = prop.sharedMemPerBlock / attr->sharedSizeBytes;

  return max;
}

cudaError_t CUDARTAPI cudaFuncGetAttributesInternal(
    struct cudaFuncAttributes *attr, const char *hostFun,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  if (entry) {
    const struct gpgpu_ptx_sim_info *kinfo = entry->get_kernel_info();
    attr->sharedSizeBytes = kinfo->smem;
    attr->constSizeBytes = kinfo->cmem;
    attr->localSizeBytes = kinfo->lmem;
    attr->numRegs = kinfo->regs;
    if (kinfo->maxthreads > 0)
      attr->maxThreadsPerBlock = kinfo->maxthreads;
    else
      attr->maxThreadsPerBlock = getMaxThreadsPerBlock(attr, ctx);
#if CUDART_VERSION >= 3000
    attr->ptxVersion = kinfo->ptx_version;
    attr->binaryVersion = kinfo->sm_target;
#endif
  }
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION > 5000)
__host__ cudaError_t CUDARTAPI
cudaDeviceGetAttributeInternal(int *value, enum cudaDeviceAttr attr, int device,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  const struct cudaDeviceProp *prop;
  _cuda_device_id *dev = ctx->GPGPUSim_Init();

  if (device <= dev->num_devices()) {
    prop = dev->get_prop();
    switch (attr) {
      case 1:
        *value = prop->maxThreadsPerBlock;
        break;
      case 2:
        *value = prop->maxThreadsDim[0];
        break;
      case 3:
        *value = prop->maxThreadsDim[1];
        break;
      case 4:
        *value = prop->maxThreadsDim[2];
        break;
      case 5:
        *value = prop->maxGridSize[0];
        break;
      case 6:
        *value = prop->maxGridSize[1];
        break;
      case 7:
        *value = prop->maxGridSize[2];
        break;
      case 8:
        *value = prop->sharedMemPerBlock;
        break;
      case 9:
        *value = prop->totalConstMem;
        break;
      case 10:
        *value = prop->warpSize;
        break;
      case 11:
        *value = 16;  // dummy value
        break;
      case 12:
        *value = prop->regsPerBlock;
        break;
      case 13:
        *value = 1480000;  // for 1080ti
        break;
      case 14:
        *value = prop->textureAlignment;
        break;
      case 15:
        *value = 0;
        break;
      case 16:
        *value = prop->multiProcessorCount;
        break;
      case 17:
      case 18:
      case 19:
        *value = 0;
        break;
      case 21:
      case 22:
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 42:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 52:
      case 53:
      case 55:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
      case 62:
      case 63:
      case 64:
      case 66:
      case 67:
      case 69:
      case 70:
      case 71:
      case 73:
      case 74:
      case 77:
        *value = 1000;  // dummy value
        break;
      case 29:
      case 43:
      case 54:
      case 65:
      case 68:
      case 72:
        *value = 10;  // dummy value
        break;
      case 30:
      case 51:
        *value = 128;  // dummy value
        break;
      case 31:
        *value = 1;
        break;
      case 32:
        *value = 0;
        break;
      case 33:
      case 50:
        *value = 0;  // dummy value
        break;
      case 34:
        *value = 0;
        break;
      case 35:
        *value = 0;
        break;
      case 36:
        *value = 1250000;  // CK value for 1080ti
        break;
      case 37:
        *value = 352;  // value for 1080ti
        break;
      case 38:
        *value = 3000000;  // value for 1080ti
        break;
      case 39:
        *value = dev->get_gpgpu()->threads_per_core();
        break;
      case 40:
        *value = 0;
        break;
      case 41:
        *value = 0;
        break;
      case 75:  // cudaDevAttrComputeCapabilityMajor
        *value = prop->major;
        break;
      case 76:  // cudaDevAttrComputeCapabilityMinor
        *value = prop->minor;
        break;
      case 78:
        *value = 0;  // TODO: as of now, we dont support stream priorities.
        break;
      case 79:
        *value = 0;
        break;
      case 80:
        *value = 0;
        break;
#if (CUDART_VERSION > 5050)
      case 81:
        *value = prop->sharedMemPerMultiprocessor;
        break;
      case 82:
        *value = prop->regsPerMultiprocessor;
        break;
#endif
      case 83:
      case 84:
      case 85:
      case 86:
        *value = 0;
        break;
      case 87:
        *value = 4;  // dummy value
        break;
      case 88:
      case 89:
        *value = 0;
        break;
      default:
        printf("ERROR: Attribute number %d unimplemented \n", attr);
        abort();
    }
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}
#endif

__host__ cudaError_t CUDARTAPI cudaBindTextureInternal(
    size_t *offset, const struct textureReference *texref, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf(
      "GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = "
      "%zu\n",
      sizeof(struct textureReference));
  struct cudaArray *array;
  array = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  array->desc = *desc;
  array->size = size;
  array->width = size;
  array->height = 1;
  array->dimensions = 1;
  array->devPtr = (void *)devPtr;
  array->devPtr32 = (int)(long long)devPtr;
  offset = 0;
  printf("GPGPU-Sim PTX:   size = %zu\n", size);
  printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n",
         desc->x, desc->y, desc->z, desc->w);
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
  devPtr = (void *)(long long)array->devPtr32;
  printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArrayInternal(
    const struct textureReference *texref, const struct cudaArray *array,
    const struct cudaChannelFormatDesc *desc, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaUnbindTextureInternal(
    const struct textureReference *texref, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf(
      "GPGPU-Sim PTX: in cudaUnbindTexture: sizeof(struct textureReference) = "
      "%zu\n",
      sizeof(struct textureReference));
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));

  gpu->gpgpu_ptx_sim_unbindTexture(texref);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernelInternal(
    const char *hostFun, dim3 gridDim, dim3 blockDim, const void **args,
    size_t sharedMem, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }

  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
#if CUDART_VERSION < 10000
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream, ctx);
#endif
  for (unsigned i = 0; i < entry->num_args(); i++) {
    std::pair<size_t, unsigned> p = entry->get_param_config(i);
    cudaSetupArgumentInternal(args[i], p.first, p.second);
  }

  cudaLaunchInternal(hostFun);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamCreateInternal(
    cudaStream_t *stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: cudaStreamCreate\n");
#if (CUDART_VERSION >= 3000)
  *stream = new struct CUstream_st();
  ctx->the_gpgpusim->g_stream_manager->add_stream(*stream);
#else
  *stream = 0;
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroyInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  // per-stream synchronization required for application using external
  // libraries without explicit synchronization in the code to avoid the
  // stream_manager from spinning forever to destroy non-empty streams without
  // making any forward progress.
  stream->synchronize();
  ctx->the_gpgpusim->g_stream_manager->destroy_stream(stream);
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronizeInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  if (stream == NULL) ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
  stream->synchronize();
#else
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
#endif
  return g_last_cudaError = cudaSuccess;
}

void __cudaRegisterTextureInternal(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext,
    gpgpu_context *gpgpu_ctx =
        NULL)  // passes in a newly created textureReference
{
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  std::string devStr(deviceName);
#if (CUDART_VERSION > 4020)
  if (devStr.size() > 2 && devStr.data()[0] == ':' && devStr.data()[1] == ':')
    devStr = devStr.replace(0, 2, "");
#endif
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in __cudaRegisterTexture:\n");
  gpu->gpgpu_ptx_sim_bindNameToTexture(devStr.data(), hostVar, dim, norm, ext);
  printf("GPGPU-Sim PTX:   int dim = %d\n", dim);
  printf("GPGPU-Sim PTX:   int norm = %d\n", norm);
  printf("GPGPU-Sim PTX:   int ext = %d\n", ext);
  printf(
      "GPGPU-Sim PTX:   Execution warning: Not finished implementing \"%s\"\n",
      __my_func__);
}

cudaError_t cudaGLUnmapBufferObjectInternal(GLuint bufferObj,
                                            gpgpu_context *gpgpu_ctx = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#ifdef OPENGL_SUPPORT
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  CUctx_st *ctx = GPGPUSim_Context(ctx);
  glbmap_entry_t *p = ctx->api->g_glbmap;
  while (p && p->m_bufferObj != bufferObj) p = p->m_next;
  if (p == NULL) return g_last_cudaError = cudaErrorUnknown;

  char *data = (char *)calloc(p->m_size, 1);
  memcpy_from_gpu(data, (size_t)p->m_devPtr, p->m_size);
  glBufferSubData(GL_ARRAY_BUFFER, 0, p->m_size, data);
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

#if CUDART_VERSION >= 3000

__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfigInternal(const char *func, enum cudaFuncCache cacheConfig,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  context->get_device()->get_gpgpu()->set_cache_config(
      context->get_kernel(func)->get_name(), (FuncCache)cacheConfig);
  return g_last_cudaError = cudaSuccess;
}

#endif

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuLaunchKernelInternal(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (extra != NULL) {
    printf(
        "GPGPU-Sim CUDA DRIVER API: ERROR: Currently do not support void** "
        "extra.\n");
    abort();
  }
  const char *hostFun = (const char *)f;
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  cudaConfigureCallInternal(dim3(gridDimX, gridDimY, gridDimZ),
                            dim3(blockDimX, blockDimY, blockDimZ),
                            sharedMemBytes, (cudaStream_t)hStream, ctx);
  for (unsigned i = 0; i < entry->num_args(); i++) {
    std::pair<size_t, unsigned> p = entry->get_param_config(i);
    cudaSetupArgumentInternal(kernelParams[i], p.first, p.second, ctx);
  }
  cudaLaunchInternal(hostFun, ctx);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

CUevent_st *get_event(cudaEvent_t event) {
  unsigned event_uid;
#if CUDART_VERSION >= 3000
  event_uid = event->get_uid();
#else
  event_uid = event;
#endif
  event_tracker_t::iterator e = g_timer_events.find(event_uid);
  if (e == g_timer_events.end()) return NULL;
  return e->second;
}

__host__ cudaError_t CUDARTAPI cudaEventRecordInternal(
    cudaEvent_t event, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (!e) return g_last_cudaError = cudaErrorUnknown;
  struct CUstream_st *s = (struct CUstream_st *)stream;
  stream_operation op(e, s);
  e->issue();
  ctx->the_gpgpusim->g_stream_manager->push(op);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEventInternal(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // reference:
  // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__STREAM_gfe68d207dc965685d92d3f03d77b0876.html
  CUevent_st *e = get_event(event);
  if (!e) {
    printf(
        "GPGPU-Sim API: Error at cudaStreamWaitEvent. Event is not created "
        ".\n");
    return g_last_cudaError = cudaErrorInvalidResourceHandle;
  } else if (e->num_issued() == 0) {
    printf(
        "GPGPU-Sim API: Warning: cudaEventRecord has not been called on event "
        "before calling cudaStreamWaitEvent.\nNothin    g to be done.\n");
    return g_last_cudaError = cudaSuccess;
  }
  if (!stream) {
    ctx->the_gpgpusim->g_stream_manager->pushCudaStreamWaitEventToAllStreams(
        e, flags);
  } else {
    struct CUstream_st *s = (struct CUstream_st *)stream;
    stream_operation op(s, e, flags);
    ctx->the_gpgpusim->g_stream_manager->push(op);
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadExitInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  ctx->exit_simulation();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Called on host side
  ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI
cudaDeviceSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Blocks until the device has completed all preceding requested tasks
  ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
}

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
cudaError_t cudaPeekAtLastError(void) { return g_last_cudaError; }

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  return cudaMallocInternal(devPtr, size);
}

__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
  return cudaMallocHostInternal(ptr, size);
}
__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch,
                                               size_t width, size_t height) {
  return cudaMallocPitchInternal(devPtr, pitch, width, height);
}

__host__ cudaError_t CUDARTAPI cudaMallocArray(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height __dv(1)) {
  return cudaMallocArrayInternal(array, desc, width, height);
}

__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
}
__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  free(ptr);  // this will crash the system if called twice
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
};

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
                                          size_t count,
                                          enum cudaMemcpyKind kind) {
  return cudaMemcpyInternal(dst, src, count, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst,
                                                 size_t wOffset, size_t hOffset,
                                                 const void *src, size_t count,
                                                 enum cudaMemcpyKind kind) {
  return cudaMemcpyToArrayInternal(dst, wOffset, hOffset, src, count, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst,
                                                   const struct cudaArray *src,
                                                   size_t wOffset,
                                                   size_t hOffset, size_t count,
                                                   enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(
    struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch,
                                            const void *src, size_t spitch,
                                            size_t width, size_t height,
                                            enum cudaMemcpyKind kind) {
  return cudaMemcpy2DInternal(dst, dpitch, src, spitch, width, height, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  return cudaMemcpy2DToArrayInternal(dst, wOffset, hOffset, src, spitch, width,
                                     height, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(
    void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(
    struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t width, size_t height,
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(
    const char *symbol, const void *src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
  return cudaMemcpyToSymbolInternal(symbol, src, count, offset, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(
    void *dst, const char *symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
  return cudaMemcpyFromSymbolInternal(dst, symbol, count, offset, kind);
}

__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // placeholder; should interact with cudaMalloc and cudaFree?
  *free = 10000000000;
  *total = 10000000000;

  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                               size_t count,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  return cudaMemcpyAsyncInternal(dst, src, count, kind, stream);
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(
    void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                                 const void *src, size_t spitch,
                                                 size_t width, size_t height,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

#if (CUDART_VERSION >= 8000)
cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
      numBlocks, hostFunc, blockSize, dynamicSMemSize, flags);
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count) {
  return cudaMemsetInternal(mem, c, count);
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *mem, int c, size_t count,
                                               cudaStream_t stream = 0) {
  return cudaMemsetAsyncInternal(mem, c, count, stream = 0);
}

__host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c,
                                            size_t width, size_t height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr,
                                                    const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size,
                                                 const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  return cudaGetDeviceCountInternal(count);
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  return cudaGetDevicePropertiesInternal(prop, device);
}

#if (CUDART_VERSION > 5000)
__host__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value,
                                                      enum cudaDeviceAttr attr,
                                                      int device) {
  return cudaDeviceGetAttributeInternal(value, attr, device);
}
#endif

__host__ cudaError_t CUDARTAPI
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
  return cudaChooseDeviceInternal(device, prop);
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  return cudaSetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  return cudaGetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue,
                                                  cudaLimit limit) {
  return cudaDeviceGetLimitInternal(pValue, limit);
}

__host__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream,
                                                     int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len,
                                                     int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
                                                   void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t cudaIpcOpenMemHandle(void **devPtr,
                                          cudaIpcMemHandle_t handle,
                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI
cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t *offset, const struct textureReference *texref, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX)) {
  return cudaBindTextureInternal(offset, texref, devPtr, desc,
                                 size __dv(UINT_MAX));
}

__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array,
    const struct cudaChannelFormatDesc *desc) {
  return cudaBindTextureToArrayInternal(texref, array, desc);
}

__host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const struct textureReference *texref) {
  return cudaUnbindTextureInternal(texref);
}

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
    size_t *offset, const struct textureReference *texref) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(
    const struct textureReference **texref, const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(
    struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *desc = array->desc;
  return g_last_cudaError = cudaSuccess;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

__host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError;
}

__host__ const char *cudaGetErrorName(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return NULL;
}

__host__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (g_last_cudaError == cudaSuccess) return "no error";
  char buf[1024];
  snprintf(buf, 1024, "<<GPGPU-Sim PTX: there was an error (code = %d)>>",
           g_last_cudaError);
  return strdup(buf);
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size,
                                                 size_t offset) {
  return cudaSetupArgumentInternal(arg, size, offset);
}

__host__ cudaError_t CUDARTAPI cudaLaunch(const char *hostFun) {
  return cudaLaunchInternal(hostFun);
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const char *hostFun,
                                                dim3 gridDim, dim3 blockDim,
                                                const void **args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
  return cudaLaunchKernelInternal(hostFun, gridDim, blockDim, args, sharedMem,
                                  stream);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
  return cudaStreamCreateInternal(stream);
}

// TODO: introduce priorities
__host__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(
    cudaStream_t *stream, unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI
cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

__host__ __device__ cudaError_t CUDARTAPI
cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  return cudaStreamDestroyInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  return cudaStreamSynchronizeInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  if (stream == NULL) return g_last_cudaError = cudaErrorInvalidResourceHandle;
  return g_last_cudaError = stream->empty() ? cudaSuccess : cudaErrorNotReady;
#else
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
  return g_last_cudaError = cudaSuccess;  // it is always success because all
                                          // cuda calls are synchronous
#endif
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = new CUevent_st(false);
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                               cudaStream_t stream) {
  return cudaEventRecordInternal(event, stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                   cudaEvent_t event,
                                                   unsigned int flags) {
  return cudaStreamWaitEventInternal(stream, event, flags);
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (e == NULL) {
    return g_last_cudaError = cudaErrorInvalidValue;
  } else if (e->done()) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorNotReady;
  }
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
  fflush(stdout);
  CUevent_st *e = (CUevent_st *)event;
  while (!e->done())
    ;
  printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
  fflush(stdout);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  unsigned event_uid = e->get_uid();
  event_tracker_t::iterator pe = g_timer_events.find(event_uid);
  if (pe == g_timer_events.end())
    return g_last_cudaError = cudaErrorInvalidValue;
  g_timer_events.erase(pe);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms,
                                                    cudaEvent_t start,
                                                    cudaEvent_t end) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  time_t elapsed_time;
  CUevent_st *s = get_event(start);
  CUevent_st *e = get_event(end);
  if (s == NULL || e == NULL) return g_last_cudaError = cudaErrorUnknown;
  elapsed_time = e->clock() - s->clock();
  *ms = 1000 * elapsed_time;
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
  return cudaThreadExitInternal();
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
  return cudaThreadSynchronizeInternal();
}

int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaThreadExit();
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

#if (CUDART_VERSION >= 3010)
int dummy0() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 0;
}

int dummy1() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 2 << 20;
}

typedef int (*ExportedFunction)();

static ExportedFunction exportTable[3] = {&dummy0, &dummy0, &dummy0};

__host__ cudaError_t CUDARTAPI cudaGetExportTable(
    const void **ppExportTable, const cudaUUID_t *pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("cudaGetExportTable: UUID = ");
  for (int s = 0; s < 16; s++) {
    printf("%#2x ", (unsigned char)(pExportTableId->bytes[s]));
  }
  *ppExportTable = &exportTable;

  printf("\n");
  return g_last_cudaError = cudaSuccess;
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

//#include "../../cuobjdump_to_ptxplus/cuobjdump_parser.h"

// extracts all ptx files from binary and dumps into
// prog_name.unique_no.sm_<>.ptx files
void cuda_runtime_api::extract_ptx_files_using_cuobjdump(CUctx_st *context) {
  char command[1000];
  char *pytorch_bin = getenv("PYTORCH_BIN");
  std::string app_binary = get_app_binary();

  char ptx_list_file_name[1024];
  snprintf(ptx_list_file_name, 1024, "_cuobjdump_list_ptx_XXXXXX");
  int fd2 = mkstemp(ptx_list_file_name);
  close(fd2);

  if (pytorch_bin != NULL && strlen(pytorch_bin) != 0) {
    app_binary = std::string(pytorch_bin);
  }

  // only want file names
  snprintf(command, 1000,
           "$CUDA_INSTALL_PATH/bin/cuobjdump -lptx %s  | cut -d \":\" -f 2 | "
           "awk '{$1=$1}1' > %s",
           app_binary.c_str(), ptx_list_file_name);
  if (system(command) != 0) {
    printf("WARNING: Failed to execute cuobjdump to get list of ptx files \n");
    exit(0);
  }
  if (!gpgpu_ctx->device_runtime->g_cdp_enabled) {
    // based on the list above, dump ptx files individually. Format of dumped
    // ptx file is prog_name.unique_no.sm_<>.ptx

    std::ifstream infile(ptx_list_file_name);
    std::string line;
    while (std::getline(infile, line)) {
      // int pos = line.find(std::string(get_app_binary_name(app_binary)));
      const char *ptx_file = line.c_str();
      printf("Extracting specific PTX file named %s \n", ptx_file);
      snprintf(command, 1000, "$CUDA_INSTALL_PATH/bin/cuobjdump -xptx %s %s",
               ptx_file, app_binary.c_str());
      if (system(command) != 0) {
        printf("ERROR: command: %s failed \n", command);
        exit(0);
      }
      context->no_of_ptx++;
    }
  }

  if (!context->no_of_ptx) {
    printf(
        "WARNING: Number of ptx in the executable file are 0. One of the "
        "reasons might be\n");
    printf("\t1. CDP is enabled\n");
    printf("\t2. When using PyTorch, PYTORCH_BIN is not set correctly\n");
  }

  std::ifstream infile(ptx_list_file_name);
  std::string line;
  while (std::getline(infile, line)) {
    // int pos = line.find(std::string(get_app_binary_name(app_binary)));
    int pos1 = line.find("sm_");
    int pos2 = line.find_last_of(".");
    if (pos1 == std::string::npos && pos2 == std::string::npos) {
      printf("ERROR: PTX list is not in correct format");
      exit(0);
    }
    std::string vstr = line.substr(pos1 + 3, pos2 - pos1 - 3);
    int version = atoi(vstr.c_str());
    if (version_filename.find(version) == version_filename.end()) {
      version_filename[version] = std::set<std::string>();
    }
    version_filename[version].insert(line);
  }
}

//! Call cuobjdump to extract everything (-elf -sass -ptx)
/*!
 *	This Function extract the whole PTX (for all the files) using cuobjdump
 *	to _cuobjdump_complete_output_XXXXXX then runs a parser to chop it up
 *with each binary in its own file It is also responsible for extracting the
 *libraries linked to the binary if the option is enabled
 * */
void cuda_runtime_api::extract_code_using_cuobjdump() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);

  // prevent the dumping by cuobjdump everytime we execute the code!
  const char *override_cuobjdump = getenv("CUOBJDUMP_SIM_FILE");
  char command[1000];
  std::string app_binary = get_app_binary();
  // Running cuobjdump using dynamic link to current process
  snprintf(command, 1000, "md5sum %s ", app_binary.c_str());
  printf("Running md5sum using \"%s\"\n", command);
  if (system(command)) {
    std::cout << "Failed to execute: " << command << std::endl;
    exit(1);
  }
  // Running cuobjdump using dynamic link to current process
  // Needs the option '-all' to extract PTX from CDP-enabled binary

  // dump ptx for all individial ptx files into sepearte files which is later
  // used by ptxas.
  int result = 0;
#if (CUDART_VERSION >= 6000)
  extract_ptx_files_using_cuobjdump(context);
  return;
#endif
  // TODO: redundant to dump twice. how can it be prevented?
  // dump only for specific arch
  char fname[1024];
  if ((override_cuobjdump == NULL) || (strlen(override_cuobjdump) == 0)) {
    snprintf(fname, 1024, "_cuobjdump_complete_output_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);
    if (!gpgpu_ctx->device_runtime->g_cdp_enabled)
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass %s > %s",
               app_binary.c_str(), fname);
    else
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass -all %s > %s",
               app_binary.c_str(), fname);
    bool parse_output = true;
    result = system(command);
    if (result) {
      if (context->get_device()
              ->get_gpgpu()
              ->get_config()
              .experimental_lib_support() &&
          (result == 65280)) {
        // Some CUDA application may exclusively use kernels provided by CUDA
        // libraries (e.g. CUBLAS).  Skipping cuobjdump extraction from the
        // executable for this case.
        // 65280 is the return code from cuobjdump denoting the specific error
        // (tested on CUDA 4.0/4.1/4.2)
        printf("WARNING: Failed to execute: %s\n", command);
        printf("         Executable binary does not contain any GPU kernel.\n");
        parse_output = false;
      } else {
        printf("ERROR: Failed to execute: %s\n", command);
        exit(1);
      }
    }

    if (parse_output) {
      printf("Parsing file %s\n", fname);
      FILE *cuobjdump_in;
      cuobjdump_in = fopen(fname, "r");

      struct cuobjdump_parser parser;
      parser.elfserial = 1;
      parser.ptxserial = 1;
      cuobjdump_lex_init(&(parser.scanner));
      cuobjdump_set_in(cuobjdump_in, (parser.scanner));
      cuobjdump_parse(parser.scanner, &parser, cuobjdumpSectionList);
      cuobjdump_lex_destroy(parser.scanner);
      fclose(cuobjdump_in);
      printf("Done parsing!!!\n");
    } else {
      printf("Parsing skipped for %s\n", fname);
    }

    if (context->get_device()
            ->get_gpgpu()
            ->get_config()
            .experimental_lib_support()) {
      // Experimental library support
      // Currently only for cufft

      std::stringstream cmd;
      cmd << "ldd " << app_binary
          << " | grep $CUDA_INSTALL_PATH | awk \'{print $3}\' > _tempfile_.txt";
      int result = system(cmd.str().c_str());
      if (result) {
        std::cout << "Failed to execute: " << cmd.str() << std::endl;
        exit(1);
      }
      std::ifstream libsf;
      libsf.open("_tempfile_.txt");
      if (!libsf.is_open()) {
        std::cout << "Failed to open: _tempfile_.txt" << std::endl;
        exit(1);
      }

      // Save the original section list
      std::list<cuobjdumpSection *> tmpsl = cuobjdumpSectionList;
      cuobjdumpSectionList.clear();

      std::string line;
      std::getline(libsf, line);
      std::cout << "DOING: " << line << std::endl;
      int cnt = 1;
      while (libsf.good()) {
        std::stringstream libcodfn;
        libcodfn << "_cuobjdump_complete_lib_" << cnt << "_";
        cmd.str("");  // resetting
        cmd << "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass ";
        cmd << line;
        cmd << " > ";
        cmd << libcodfn.str();
        std::cout << "Running cuobjdump on " << line << std::endl;
        std::cout << "Using command: " << cmd.str() << std::endl;
        result = system(cmd.str().c_str());
        if (result) {
          printf("ERROR: Failed to execute: %s\n", command);
          exit(1);
        }
        std::cout << "Done" << std::endl;

        std::cout << "Trying to parse " << libcodfn.str() << std::endl;
        FILE *cuobjdump_in;
        cuobjdump_in = fopen(libcodfn.str().c_str(), "r");
        struct cuobjdump_parser parser;
        parser.elfserial = 1;
        parser.ptxserial = 1;
        cuobjdump_lex_init(&(parser.scanner));
        cuobjdump_set_in(cuobjdump_in, (parser.scanner));
        cuobjdump_parse(parser.scanner, &parser, cuobjdumpSectionList);
        cuobjdump_lex_destroy(parser.scanner);
        fclose(cuobjdump_in);
        std::getline(libsf, line);
      }
      libSectionList = cuobjdumpSectionList;

      // Restore the original section list
      cuobjdumpSectionList = tmpsl;
    }
  } else {
    printf(
        "GPGPU-Sim PTX: overriding cuobjdump with '%s' (CUOBJDUMP_SIM_FILE is "
        "set)\n",
        override_cuobjdump);
    snprintf(fname, 1024, "%s", override_cuobjdump);
  }
}

//! Read file into char*
// TODO: convert this to C++ streams, will be way cleaner
char *readfile(const std::string filename) {
  assert(filename != "");
  FILE *fp = fopen(filename.c_str(), "r");
  if (!fp) {
    std::cout << "ERROR: Could not open file %s for reading\n"
              << filename << std::endl;
    assert(0);
  }
  // finding size of the file
  int filesize = 0;
  fseek(fp, 0, SEEK_END);

  filesize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  // allocate and copy the entire ptx
  char *ret = (char *)malloc((filesize + 1) * sizeof(char));
  fread(ret, 1, filesize, fp);
  ret[filesize] = '\0';
  fclose(fp);
  return ret;
}

//! Function that helps debugging
void printSectionList(std::list<cuobjdumpSection *> sl) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sl.begin(); iter != sl.end(); iter++) {
    (*iter)->print();
  }
}

//! Remove unecessary sm versions from the section list
std::list<cuobjdumpSection *> cuda_runtime_api::pruneSectionList(
    CUctx_st *context) {
  unsigned forced_max_capability = context->get_device()
                                       ->get_gpgpu()
                                       ->get_config()
                                       .get_forced_max_capability();

  // For ptxplus, force the max capability to 19 if it's higher or
  // unspecified(0)
  if (context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus()) {
    if ((forced_max_capability == 0) || (forced_max_capability >= 20)) {
      printf(
          "GPGPU-Sim: WARNING: Capability >= 20 are not supported in "
          "PTXPlus\n\tSetting forced_max_capability to 19\n");
      forced_max_capability = 19;
    }
  }

  std::list<cuobjdumpSection *> prunedList;

  // Find the highest capability (that is lower than the forced maximum) for
  // each cubin file and set it in cuobjdumpSectionMap. Do this only for ptx
  // sections
  std::map<std::string, unsigned> cuobjdumpSectionMap;
  int min_ptx_capability_found = 0;
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (dynamic_cast<cuobjdumpPTXSection *>(*iter) != NULL) {
      if (capability < min_ptx_capability_found ||
          min_ptx_capability_found == 0)
        min_ptx_capability_found = capability;
      if (capability <= forced_max_capability || forced_max_capability == 0) {
        if ((cuobjdumpSectionMap.find((*iter)->getIdentifier()) ==
             cuobjdumpSectionMap.end()) ||
            (cuobjdumpSectionMap[(*iter)->getIdentifier()] < capability))
          cuobjdumpSectionMap[(*iter)->getIdentifier()] = capability;
      }
    }
  }

  // Throw away the sections with the lower capabilites and push those with the
  // highest in the pruned list
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (capability == cuobjdumpSectionMap[(*iter)->getIdentifier()]) {
      prunedList.push_back(*iter);
    } else {
      delete *iter;
    }
  }
  if (prunedList.empty()) {
    printf(
        "Error: No PTX sections found with sm capability that is lower than "
        "current forced maximum capability \n minimum ptx capability found = "
        "%u, maximum forced ptx capability = %u \n User might want to change "
        "either the forced maximum capability from gpgpusim configuration or "
        "update the compilation to generate the required PTX version\n",
        min_ptx_capability_found, forced_max_capability);
    abort();
  }
  return prunedList;
}

//! Merge all PTX sections that have a specific identifier into one file
std::list<cuobjdumpSection *> cuda_runtime_api::mergeMatchingSections(
    std::string identifier) {
  const char *ptxcode = "";
  std::list<cuobjdumpSection *>::iterator old_iter;
  cuobjdumpPTXSection *old_ptxsection = NULL;
  cuobjdumpPTXSection *ptxsection;
  std::list<cuobjdumpSection *> mergedList;

  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL &&
        strcmp(ptxsection->getIdentifier().c_str(), identifier.c_str()) == 0) {
      // Read and remove the last PTX section
      if (old_ptxsection != NULL) {
        ptxcode = readfile(old_ptxsection->getPTXfilename());
        // remove ptx file?
        delete *old_iter;
      }

      // Append all the PTX from the last PTX section into the current PTX
      // section Add 50 to ptxcode to ignore the information regarding
      // version/target/address_size
      if (strlen(ptxcode) >= 50) {
        FILE *ptxfile = fopen((ptxsection->getPTXfilename()).c_str(), "a");
        fprintf(ptxfile, "%s", ptxcode + 50);
        fclose(ptxfile);
      }

      old_iter = iter;
      old_ptxsection = ptxsection;
    }
    // Store all non-PTX sections and PTX sections with non-matching identifiers
    else {
      mergedList.push_back(*iter);
    }
  }

  // Store the final PTX section
  mergedList.push_back(*old_iter);

  return mergedList;
}

//! Merge any PTX sections with matching identifiers
std::list<cuobjdumpSection *> cuda_runtime_api::mergeSections() {
  std::vector<std::string> identifier;
  cuobjdumpPTXSection *ptxsection;

  // Add all identifiers present in PTX sections to a vector
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL) {
      std::string current_id = ptxsection->getIdentifier();

      // If we haven't yet seen a given identifier, add it to the vector
      if (std::find(identifier.begin(), identifier.end(), current_id) ==
          identifier.end()) {
        identifier.push_back(current_id);
      }
    }
  }

  // Call mergeMatchingSections on all identifiers in the vector
  for (std::vector<std::string>::iterator iter = identifier.begin();
       iter != identifier.end(); iter++) {
    cuobjdumpSectionList = mergeMatchingSections(*iter);
  }

  return cuobjdumpSectionList;
}

//! Within the section list, find the ELF section corresponding to a given
//! identifier
cuobjdumpELFSection *findELFSectionInList(
    std::list<cuobjdumpSection *> sectionlist, const std::string identifier) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sectionlist.begin(); iter != sectionlist.end(); iter++) {
    cuobjdumpELFSection *elfsection;
    if ((elfsection = dynamic_cast<cuobjdumpELFSection *>(*iter)) != NULL) {
      if (elfsection->getIdentifier() == identifier) return elfsection;
    }
  }
  return NULL;
}

//! Find an ELF section in all the known lists
cuobjdumpELFSection *cuda_runtime_api::findELFSection(
    const std::string identifier) {
  cuobjdumpELFSection *sec =
      findELFSectionInList(cuobjdumpSectionList, identifier);
  if (sec != NULL) return sec;
  sec = findELFSectionInList(libSectionList, identifier);
  if (sec != NULL) return sec;
  std::cout << "Could not find " << identifier << std::endl;
  assert(0 && "Could not find the required ELF section");
  return NULL;
}

//! Within the section list, find the PTX section corresponding to a given
//! identifier
cuobjdumpPTXSection *cuda_runtime_api::findPTXSectionInList(
    std::list<cuobjdumpSection *> &sectionlist, const std::string identifier) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sectionlist.begin(); iter != sectionlist.end(); iter++) {
    cuobjdumpPTXSection *ptxsection;
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL) {
      if (ptxsection->getIdentifier() == identifier)
        return ptxsection;
      else {
        if (gpgpu_ctx->device_runtime->g_cdp_enabled) {
          printf(
              "Warning: __cudaRegisterFatBinary needs %s, but find PTX section "
              "with %s\n",
              identifier.c_str(), ptxsection->getIdentifier().c_str());
          return ptxsection;
        }
      }
    }
  }
  return NULL;
}

//! Find an PTX section in all the known lists
cuobjdumpPTXSection *cuda_runtime_api::findPTXSection(
    const std::string identifier) {
  cuobjdumpPTXSection *sec =
      findPTXSectionInList(cuobjdumpSectionList, identifier);
  if (sec != NULL) return sec;
  sec = findPTXSectionInList(libSectionList, identifier);
  if (sec != NULL) return sec;
  std::cout << "Could not find " << identifier << std::endl;
  assert(0 && "Could not find the required PTX section");
  return NULL;
}

//! Extract the code using cuobjdump and remove unnecessary sections
void cuda_runtime_api::cuobjdumpInit() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  extract_code_using_cuobjdump();  // extract all the output of cuobjdump to
                                   // _cuobjdump_*.*
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0) {
    cuobjdumpSectionList = pruneSectionList(context);
    cuobjdumpSectionList = mergeSections();
  }
}

//! Either submit PTX for simulation or convert SASS to PTXPlus and submit it
void gpgpu_context::cuobjdumpParseBinary(unsigned int handle) {
  CUctx_st *context = GPGPUSim_Context(this);
  if (api->fatbin_registered[handle]) return;
  api->fatbin_registered[handle] = true;
  std::string fname = api->fatbinmap[handle];

  if (api->name_symtab.find(fname) != api->name_symtab.end()) {
    symbol_table *symtab = api->name_symtab[fname];
    context->add_binary(symtab, handle);
    return;
  }
  symbol_table *symtab;

#if (CUDART_VERSION >= 6000)
  // loops through all ptx files from smallest sm version to largest
  std::map<unsigned, std::set<std::string> >::iterator itr_m;
  for (itr_m = api->version_filename.begin();
       itr_m != api->version_filename.end(); itr_m++) {
    std::set<std::string>::iterator itr_s;
    for (itr_s = itr_m->second.begin(); itr_s != itr_m->second.end(); itr_s++) {
      std::string ptx_filename = *itr_s;
      printf("GPGPU-Sim PTX: Parsing %s\n", ptx_filename.c_str());
      symtab = gpgpu_ptx_sim_load_ptx_from_filename(ptx_filename.c_str());
    }
  }
  api->name_symtab[fname] = symtab;
  context->add_binary(symtab, handle);
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  for (itr_m = api->version_filename.begin();
       itr_m != api->version_filename.end(); itr_m++) {
    std::set<std::string>::iterator itr_s;
    for (itr_s = itr_m->second.begin(); itr_s != itr_m->second.end(); itr_s++) {
      std::string ptx_filename = *itr_s;
      printf("GPGPU-Sim PTX: Loading PTXInfo from %s\n", ptx_filename.c_str());
      gpgpu_ptx_info_load_from_filename(ptx_filename.c_str(), itr_m->first);
    }
  }
  return;
#endif

  unsigned max_capability = 0;
  for (std::list<cuobjdumpSection *>::iterator iter =
           api->cuobjdumpSectionList.begin();
       iter != api->cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (capability > max_capability) max_capability = capability;
  }
  if (max_capability > 20)
    printf("WARNING: No guarantee that PTX will be parsed for SM version %u\n",
           max_capability);
  if (max_capability == 0)
    max_capability = context->get_device()
                         ->get_gpgpu()
                         ->get_config()
                         .get_forced_max_capability();

  cuobjdumpPTXSection *ptx = NULL;
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0)
    ptx = api->findPTXSection(fname);
  char *ptxcode;
  const char *override_ptx_name = getenv("PTX_SIM_KERNELFILE");
  if (override_ptx_name == NULL or getenv("PTX_SIM_USE_PTX_FILE") == NULL or
      strlen(getenv("PTX_SIM_USE_PTX_FILE")) == 0) {
    ptxcode = readfile(ptx->getPTXfilename());
  } else {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with '%s' "
        "(PTX_SIM_USE_PTX_FILE is set)\n",
        override_ptx_name);
    ptxcode = readfile(override_ptx_name);
  }
  if (context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus()) {
    cuobjdumpELFSection *elfsection = api->findELFSection(ptx->getIdentifier());
    assert(elfsection != NULL);
    char *ptxplus_str = ptxinfo->gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(
        ptx->getPTXfilename(), elfsection->getELFfilename(),
        elfsection->getSASSfilename());
    symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxplus_str, handle);
    printf("Adding %s with cubin handle %u\n", ptx->getPTXfilename().c_str(),
           handle);
    context->add_binary(symtab, handle);
    gpgpu_ptxinfo_load_from_string(ptxcode, handle, max_capability,
                                   context->no_of_ptx);
    delete[] ptxplus_str;
  } else {
    symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxcode, handle);
    // if CUOBJDUMP_SIM_FILE is not set, ptx is NULL. So comment below.
    // printf("Adding %s with cubin handle %u\n", ptx->getPTXfilename().c_str(),
    // handle);
    context->add_binary(symtab, handle);
    gpgpu_ptxinfo_load_from_string(ptxcode, handle, max_capability,
                                   context->no_of_ptx);
  }
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  api->name_symtab[fname] = symtab;

  // TODO: Remove temporarily files as per configurations
}
}

extern "C" {

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaRegisterFatBinaryInternal(fatCubin);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st *stream = 0) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim) {
  cudaRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun, deviceName,
                               thread_limit, tid, bid, bDim, gDim);
}

extern void __cudaRegisterVar(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global) {
  cudaRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress, deviceName,
                          ext, size, constant, global);
}

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem,
                                                 cudaStream_t stream) {
  return cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

cudaError_t cudaDeviceReset(void) {
  // Should reset the simulated GPU
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
  return cudaDeviceSynchronizeInternal();
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterShared\n");
}

void CUDARTAPI __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
                                       size_t size, size_t alignment,
                                       int storage) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n");
}

void __cudaRegisterTexture(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext)  // passes in a newly created textureReference
{
  __cudaRegisterTextureInternal(fatCubinHandle, hostVar, deviceAddress,
                                deviceName, dim, norm, ext);
}

char __cudaInitModule(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufferObj) {
  return cudaGLMapBufferObjectInternal(devPtr, bufferObj);
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj) {
  return cudaGLUnmapBufferObjectInternal(bufferObj);
}

cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 2010)

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t bytes,
                                    unsigned int flags) {
  return cudaHostAllocInternal(pHost, bytes, flags);
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                               unsigned int flags) {
  return cudaHostGetDevicePointerInternal(pDevice, pHost, flags);
}

__host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer,
                                                       int device,
                                                       int peerDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice,
                                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags(int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // This flag is implicitly always on (unless you are using the driver API). It
  // is safe for GPGPU-Sim to just ignore it.
  if (cudaDeviceMapHost == flags) {
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                            const char *hostFun) {
  return cudaFuncGetAttributesInternal(attr, hostFun);
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
  CUevent_st *e = new CUevent_st(flags == cudaEventBlockingSync);
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *driverVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *runtimeVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

#if CUDART_VERSION >= 3000
__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) {
  return cudaFuncSetCacheConfigInternal(func, cacheConfig);
}

// Jin: hack for cdp
__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit,
                                                  size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

//#if CUDART_VERSION >= 9000
//__host__  cudaError_t cudaFuncSetAttribute ( const void* func, enum
// cudaFuncAttribute attr, int value ) {

// ignore this Attribute for now, and the default is that carveout =
// cudaSharedmemCarveoutDefault;   //  (-1)
//	return g_last_cudaError = cudaSuccess;
//}

#endif

#endif

#if CUDART_VERSION >= 9000
/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p entry.
 * The parameter \p entry must be a pointer to a function that executes
 * on the device. The parameter specified by \p entry must be declared as a \p
 * __global__ function. The enumeration defined by \p attr is set to the value
 * defined by \p value If the specified function does not exist, then
 * ::cudaErrorInvalidDeviceFunction is returned. If the specified attribute
 * cannot be written, or if the value is incorrect, then ::cudaErrorInvalidValue
 * is returned.
 *
 * Valid values for \p attr are:
 * ::cuFuncAttrMaxDynamicSharedMem - Maximum size of dynamic shared memory per
 * block
 * ::cudaFuncAttributePreferredSharedMemoryCarveout - Preferred shared memory-L1
 * cache split ratio
 *
 * \param entry - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void
 * **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig
 * (C++ API)", \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const
 * void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(T, size_t) "cudaSetupArgument (C++ API)"
 */
cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func,
                                           enum cudaFuncAttribute attr,
                                           int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf(
      "GPGPU-Sim PTX: Execution warning: ignoring call to \"%s ( func=%p, "
      "attr=%d, value=%d )\"\n",
      __my_func__, func, attr, value);
  return g_last_cudaError = cudaSuccess;
}
#endif

cudaError_t CUDARTAPI cudaGLSetGLDevice(int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaErrorUnknown;
}

typedef void *HGPUNV;

cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer,
                                  void *val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
}

namespace cuda_math {

void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer,
                                  void *val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}

int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO This function should syncronize if we support Asyn kernel calls
  return g_last_cudaError = cudaSuccess;
}

}  // namespace cuda_math

////////

/// static functions

int cuda_runtime_api::load_static_globals(symbol_table *symtab,
                                          unsigned min_gaddr,
                                          unsigned max_gaddr, gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading globals with explicit initializers... \n");
  fflush(stdout);
  int ng_bytes = 0;
  symbol_table::iterator g = symtab->global_iterator_begin();

  for (; g != symtab->global_iterator_end(); g++) {
    symbol *global = *g;
    if (global->has_initializer()) {
      printf("GPGPU-Sim PTX:     initializing '%s' ... ",
             global->name().c_str());
      unsigned addr = global->get_address();
      const type_info *type = global->type();
      type_info_key ti = type->get_key();
      size_t size;
      int t;
      ti.type_decode(size, t);
      int nbytes = size / 8;
      int offset = 0;
      std::list<operand_info> init_list = global->get_initializer();
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        assert((addr + offset + nbytes) <
               min_gaddr);  // min_gaddr is start of "heap" for cudaMalloc
        gpu->get_global_memory()->write(addr + offset, nbytes, &value, NULL,
                                        NULL);  // assuming little endian here
        offset += nbytes;
        ng_bytes += nbytes;
      }
      printf(" wrote %u bytes\n", offset);
    }
  }
  printf("GPGPU-Sim PTX: finished loading globals (%u bytes total).\n",
         ng_bytes);
  fflush(stdout);
  return ng_bytes;
}

int cuda_runtime_api::load_constants(symbol_table *symtab, addr_t min_gaddr,
                                     gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading constants with explicit initializers... ");
  fflush(stdout);
  int nc_bytes = 0;
  symbol_table::iterator g = symtab->const_iterator_begin();

  for (; g != symtab->const_iterator_end(); g++) {
    symbol *constant = *g;
    if (constant->is_const() && constant->has_initializer()) {
      // get the constant element data size
      int basic_type;
      size_t num_bits;
      constant->type()->get_key().type_decode(num_bits, basic_type);

      std::list<operand_info> init_list = constant->get_initializer();
      int nbytes_written = 0;
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        int nbytes = num_bits / 8;
        switch (op.get_type()) {
          case int_t:
            assert(nbytes >= 1);
            break;
          case float_op_t:
            assert(nbytes == 4);
            break;
          case double_op_t:
            assert(nbytes >= 4);
            break;  // account for double DEMOTING
          default:
            abort();
        }
        unsigned addr = constant->get_address() + nbytes_written;
        assert(addr + nbytes < min_gaddr);

        gpu->get_global_memory()->write(
            addr, nbytes, &value, NULL,
            NULL);  // assume little endian (so u8 is the first byte in u32)
        nc_bytes += nbytes;
        nbytes_written += nbytes;
      }
    }
  }
  printf(" done.\n");
  fflush(stdout);
  return nc_bytes;
}

kernel_info_t *cuda_runtime_api::gpgpu_cuda_ptx_sim_init_grid(
    const char *hostFun, gpgpu_ptx_sim_arg_list_t args, struct dim3 gridDim,
    struct dim3 blockDim, CUctx_st *context) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  function_info *entry = context->get_kernel(hostFun);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  /*
  Passing a snapshot of the GPU's current texture mapping to the kernel's info
  as kernels should use texture bindings present at the time of their launch.
  */
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  if (entry == NULL) {
    printf(
        "GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found "
        "for %p\n",
        hostFun);
    abort();
  }
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }

  entry->finalize(result->get_param_memory());
  gpgpu_ctx->func_sim->g_ptx_kernel_count++;
  fflush(stdout);

  if (g_debug_execution >= 4) {
    entry->ptx_jit_config(g_mallocPtr_Size, result->get_param_memory(),
                          (gpgpu_t *)context->get_device()->get_gpgpu(),
                          gridDim, blockDim);
  }

  return result;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
//***extra api for pytorch***

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDriverGetVersion(driverVersion);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  int deviceI = -1;
  cudaError_t e = cudaGetDevice(&deviceI);
  assert(e == cudaSuccess);
  assert(deviceI != -1);
  *device = deviceI;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetDeviceCount(count);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(len >= 10);
  strcpy(name, "GPGPU-Sim");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *bytes = 20000000000;  // dummy value
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
#if (CUDART_VERSION > 5000)
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDeviceGetAttribute(pi, (cudaDeviceAttr)attrib, dev);
  assert(e == cudaSuccess);

  return CUDA_SUCCESS;
}
#endif
CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor,
                                           CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 7000

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 7000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags,
                             CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 7000
CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 7000 */

CUresult CUDAAPI cuCtxSynchronize(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4020
CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDetach(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                   CUmodule hmod, const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                   const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                                    const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6050

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(type == CU_JIT_INPUT_PTX);
  cuda_not_implemented(__my_func__, __LINE__);
  return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  return cuLinkAddFileInternal(state, type, path, numOptions, options,
                               optionValues);
}
#endif

#if CUDART_VERSION >= 5050

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut,
                                size_t *sizeOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // all cuLink* function are implemented to block until completion so nothing
  // to do here
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5050 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                      CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuMemFreeHost(void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize,
                                unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6000

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                   unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 6000 */

#if CUDART_VERSION >= 4010

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent,
                                      CUipcEventHandle handle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 4010 */

#if CUDART_VERSION >= 6050
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
__host__ cudaError_t cudaHostRegister(void *ptr, size_t size,
                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t cudaProfilerStart() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t cudaProfilerStop() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

#endif
#if CUDART_VERSION >= 4000

CUresult CUDAAPI cuMemHostUnregister(void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                              CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                              size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                              const void *srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                              CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                   const void *srcHost, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                              unsigned char uc, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned short us, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned int ui, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayCreate(CUarray *pHandle,
                               const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                      CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuArray3DCreate(
    CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 5000

CUresult CUDAAPI
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

/** @} */ /* END CUDA_MEM */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuPointerGetAttribute(void *data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 8000
__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(
    cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc,
    const cudaTextureDesc *pTexDesc, const cudaResourceViewDesc *pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                         CUmem_range_attribute *attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 8000 */

#if CUDART_VERSION >= 6000
CUresult CUDAAPI cuPointerSetAttribute(const void *value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 6000 */

#if CUDART_VERSION >= 7000
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute *attributes,
                                        void **data, CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 7000 */

/** @} */ /* END CUDA_UNIFIED */

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                            unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6000

CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 6000 */

CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

/** @} */ /* END CUDA_STREAM */

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 8000
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 8000 */

/** @} */ /* END CUDA_EVENT */

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4020
CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  return cuLaunchKernelInternal(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                blockDimY, blockDimZ, sharedMemBytes, hStream,
                                kernelParams, extra);
}
#endif /* CUDART_VERSION >= 4000 */

/** @} */ /* END CUDA_EXEC */

CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void *ptr,
                             unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunch(CUfunction f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width,
                                   int grid_height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit,
                                  CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/** @} */ /* END CUDA_EXEC_DEPRECATED */

#if CUDART_VERSION >= 6050

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_OCCUPANCY */
#endif    /* CUDART_VERSION >= 6050 */

CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR *desc,
                                      CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                           CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                        int dim) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                                   CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                             float *pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_SURFREF */

#if CUDART_VERSION >= 5000
CUresult CUDAAPI
cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                  const CUDA_TEXTURE_DESC *pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                            CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                           CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_TEXOBJECT */

CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                                    const CUDA_RESOURCE_DESC *pResDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                             CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

#if CUDART_VERSION >= 4000
CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_PEER_ACCESS */
#endif    /* CUDART_VERSION >= 4000 */

CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 5000

CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

#if CUDART_VERSION >= 3020
CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_GRAPHICS */

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetExportTable(ppExportTable, pExportTableId);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 4000 && CUDART_VERSION < 6050)
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* defined(CUDART_VERSION_INTERNAL) || (CUDART_VERSION >= 4000 && \
          CUDART_VERSION < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 5050 && CUDART_VERSION < 6050)
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 5050 && CUDART_VERSION \
          < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 3020 && CUDART_VERSION < 4010)
CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef,
                                         const CUDA_ARRAY_DESCRIPTOR *desc,
                                         CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 3020 && CUDART_VERSION \
          < 4010) */

#if defined(CUDART_VERSION_INTERNAL) || CUDART_VERSION < 4000
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || CUDART_VERSION < 4000 */

#if defined(CUDART_VERSION_INTERNAL)
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset,
                                 const void *srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray,
                                      size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                      CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc,
                               size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                 unsigned char uc, size_t Width,
                                 size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned short us, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned int ui, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif

CUresult cuProfilerInitialize(const char *configFile, const char *outputFile,
                              CUoutput_mode outputMode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult cuProfilerStart(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
CUresult cuProfilerStop(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

//_ptds

extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                          size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeer_ptds(CUdeviceptr dstDevice,
                                              CUcontext dstContext,
                                              CUdeviceptr srcDevice,
                                              CUcontext srcContext,
                                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 const void *srcHost,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void *dstHost,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr dstDevice,
                                               unsigned char uc,
                                               unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned short us,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned int ui,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8_v2_ptds(CUdeviceptr dstDevice,
                                                 unsigned int dstPitch,
                                                 unsigned char uc,
                                                 unsigned int Width,
                                                 unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D16_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned short us,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D32_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned int ui,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

//_ptsz
extern "C" CUresult CUDAAPI
cuMemcpy3DPeer_ptsz(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                               size_t ByteCount,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeerAsync_ptsz(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
    CUcontext srcContext, size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray,
                                                      size_t dstOffset,
                                                      const void *srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyAtoHAsync_v2_ptsz(void *dstHost,
                                                      CUarray srcArray,
                                                      size_t srcOffset,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      const void *srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void *dstHost,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr dstDevice,
                                                 unsigned char uc, size_t N,
                                                 CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice,
                                                   size_t dstPitch,
                                                   unsigned char uc,
                                                   size_t Width, size_t Height,
                                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent hEvent,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWriteValue32_ptsz(CUstream stream,
                                                      CUdeviceptr addr,
                                                      cuuint32_t value,
                                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWaitValue32_ptsz(CUstream stream,
                                                     CUdeviceptr addr,
                                                     cuuint32_t value,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamBatchMemOp_ptsz(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetPriority_ptsz(CUstream hStream,
                                                     int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetFlags_ptsz(CUstream hStream,
                                                  unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream hStream,
                                                   CUevent hEvent,
                                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamAddCallback_ptsz(CUstream hStream,
                                                     CUstreamCallback callback,
                                                     void *userData,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamAttachMemAsync_ptsz(CUstream hStream,
                                                        CUdeviceptr dptr,
                                                        size_t length,
                                                        unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsMapResources_ptsz(
    unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsUnmapResources_ptsz(
    unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr,
                                                    size_t count,
                                                    CUdevice dstDevice,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
