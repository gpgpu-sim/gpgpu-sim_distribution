// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"
#include "stream_manager.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

// Help funcs to avoid multiple '->' for SST
GPGPUsim_ctx *GPGPUsim_ctx_ptr() { return GPGPU_Context()->the_gpgpusim; }

class sst_gpgpu_sim *g_the_gpu() {
  return static_cast<sst_gpgpu_sim *>(GPGPUsim_ctx_ptr()->g_the_gpu);
}

class stream_manager *g_stream_manager() {
  return GPGPUsim_ctx_ptr()->g_stream_manager;
}

// SST callback
extern void SST_callback_cudaThreadSynchronize_done();
extern void SST_callback_cudaStreamSynchronize_done(cudaStream_t stream);
__attribute__((weak)) void SST_callback_cudaThreadSynchronize_done() {}
__attribute__((weak)) void SST_callback_cudaStreamSynchronize_done(cudaStream_t stream) {}

void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  // at most one kernel running at a time
  bool done;
  do {
    sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
    done = true;
    if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
      done = false;
      ctx->the_gpgpusim->g_the_gpu->init();
      while (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      }
      ctx->the_gpgpusim->g_the_gpu->print_stats(
          ctx->the_gpgpusim->g_the_gpu->last_streamID);
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
  } while (!done);
  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

static void termination_callback() {
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);
}

void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  atexit(termination_callback);
  // concurrent kernel execution simulation thread
  do {
    if (g_debug_execution >= 3) {
      printf(
          "GPGPU-Sim: *** simulation thread starting and spinning waiting for "
          "work ***\n");
      fflush(stdout);
    }
    while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
           !ctx->the_gpgpusim->g_sim_done);
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
      ctx->the_gpgpusim->g_stream_manager->print(stdout);
      fflush(stdout);
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = true;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    bool active = false;
    bool sim_cycles = false;
    ctx->the_gpgpusim->g_the_gpu->init();
    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) &&
          !ctx->the_gpgpusim->g_the_gpu->active())
        break;

      // functional simulation
      if (ctx->the_gpgpusim->g_the_gpu->is_functional_sim()) {
        kernel_info_t *kernel =
            ctx->the_gpgpusim->g_the_gpu->get_functional_kernel();
        assert(kernel);
        ctx->the_gpgpusim->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
            *kernel);
        ctx->the_gpgpusim->g_the_gpu->finish_functional_sim(kernel);
      }

      // performance simulation
      if (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        sim_cycles = true;
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      } else {
        if (ctx->the_gpgpusim->g_the_gpu->cycle_insn_cta_max_hit()) {
          ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
          ctx->the_gpgpusim->g_sim_done = true;
          ctx->the_gpgpusim->break_limit = true;
        }
      }

      active = ctx->the_gpgpusim->g_the_gpu->active() ||
               !(ctx->the_gpgpusim->g_stream_manager->empty_protected());

    } while (active && !ctx->the_gpgpusim->g_sim_done);
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
      fflush(stdout);
    }
    if (sim_cycles) {
      ctx->the_gpgpusim->g_the_gpu->print_stats(
          ctx->the_gpgpusim->g_the_gpu->last_streamID);
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = false;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
  } while (!ctx->the_gpgpusim->g_sim_done);

  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  fflush(stdout);

  if (ctx->the_gpgpusim->break_limit) {
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    exit(1);
  }

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

bool sst_sim_cycles = false;

bool SST_Cycle() {
  // Check if Synchronize is done when SST previously requested
  // cudaThreadSynchronize
  if (GPGPU_Context()->requested_synchronize &&
      ((g_stream_manager()->empty_protected() && !GPGPUsim_ctx_ptr()->g_sim_active) ||
       GPGPUsim_ctx_ptr()->g_sim_done)) {
    SST_callback_cudaThreadSynchronize_done();
    GPGPU_Context()->requested_synchronize = false;
  }

  // Polling to check for each stream if it is marked for requested with sync
  if (g_stream_manager()->get_stream_zero()->requested_synchronize() &&
      ((g_stream_manager()->empty_protected() && !GPGPUsim_ctx_ptr()->g_sim_active) || 
      GPGPUsim_ctx_ptr()->g_sim_done)) {
    SST_callback_cudaStreamSynchronize_done(0);
    g_stream_manager()->get_stream_zero()->reset_request_synchronize();
  }

  // Iterate through each stream to check if SST is waiting on
  // it and it does not have any operation
  std::list<CUstream_st *>& streams = g_stream_manager()->get_concurrent_streams();
  for (auto it = streams.begin(); it != streams.end(); it++) {
    CUstream_st *stream = *it;
    if (stream->requested_synchronize() &&
        stream->empty()) {
      // This stream is ready
      SST_callback_cudaStreamSynchronize_done(stream);
      stream->reset_request_synchronize();
    }
  }

  if (g_stream_manager()->empty_protected() &&
      !GPGPUsim_ctx_ptr()->g_sim_done && !g_the_gpu()->active()) {
    GPGPUsim_ctx_ptr()->g_sim_active = false;
    // printf("stream is empty %d \n",  g_stream_manager->empty());
    return false;
  }

  if (g_stream_manager()->operation(&sst_sim_cycles) &&
      !g_the_gpu()->active()) {
    if (sst_sim_cycles) {
      sst_sim_cycles = false;
    }
    return false;
  }

  // printf("GPGPU-Sim: Give GPU Cycle\n");
  GPGPUsim_ctx_ptr()->g_sim_active = true;

  // functional simulation
  if (g_the_gpu()->is_functional_sim()) {
    kernel_info_t *kernel = g_the_gpu()->get_functional_kernel();
    assert(kernel);
    GPGPUsim_ctx_ptr()->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
        *kernel);
    g_the_gpu()->finish_functional_sim(kernel);
  }

  // performance simulation
  if (g_the_gpu()->active()) {
    g_the_gpu()->SST_cycle();
    sst_sim_cycles = true;
    g_the_gpu()->deadlock_check();
  } else {
    if (g_the_gpu()->cycle_insn_cta_max_hit()) {
      g_stream_manager()->stop_all_running_kernels();
      GPGPUsim_ctx_ptr()->g_sim_done = true;
      GPGPUsim_ctx_ptr()->g_sim_active = false;
      GPGPUsim_ctx_ptr()->break_limit = true;
    }
  }

  if (!g_the_gpu()->active()) {
    g_the_gpu()->print_stats(GPGPUsim_ctx_ptr()->g_the_gpu->last_streamID);
    g_the_gpu()->update_stats();
    GPGPU_Context()->print_simulation_time();
  }

  if (GPGPUsim_ctx_ptr()->break_limit) {
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    return true;
  }

  return false;
}

void gpgpu_context::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    done = (the_gpgpusim->g_stream_manager->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
}

bool gpgpu_context::synchronize_check() {
  // printf("GPGPU-Sim: synchronize checking for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
  done = (the_gpgpusim->g_stream_manager->empty() &&
          !the_gpgpusim->g_sim_active) ||
         the_gpgpusim->g_sim_done;
  pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  if (done) {
    printf(
        "GPGPU-Sim: synchronize checking: detected inactive GPU simulation "
        "thread\n");
  }
  fflush(stdout);
  return done;
}

void gpgpu_context::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
  srand(1);
  print_splash();
  func_sim->read_sim_environment_variables();
  ptx_parser->read_parser_environment_variables();
  option_parser_t opp = option_parser_create();

  ptx_reg_options(opp);
  func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);
  the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options

  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  the_gpgpusim->g_the_gpu_config->init();

  if (the_gpgpusim->g_the_gpu_config->is_SST_mode()) {
    // Create SST specific GPGPUSim
    the_gpgpusim->g_the_gpu =
        new sst_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
  } else {
    the_gpgpusim->g_the_gpu =
        new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
  }
  the_gpgpusim->g_stream_manager = new stream_manager(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

  return the_gpgpusim->g_the_gpu;
}

void gpgpu_context::start_sim_thread(int api) {
  if (the_gpgpusim->g_sim_done) {
    the_gpgpusim->g_sim_done = false;
    if (the_gpgpusim->g_the_gpu_config->is_SST_mode()) {
      // Do not create concurrent thread in SST mode
      g_the_gpu()->init();
    } else {
      if (api == 1) {
        pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                       gpgpu_sim_thread_concurrent, (void *)this);
      } else {
        pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                       gpgpu_sim_thread_sequential, (void *)this);
      }
    }
  }
}

void gpgpu_context::print_simulation_time() {
  time_t current_time, difference, d, h, m, s;
  current_time = time((time_t *)NULL);
  difference = MAX(current_time - the_gpgpusim->g_simulation_starttime, 1);

  d = difference / (3600 * 24);
  h = difference / 3600 - 24 * d;
  m = difference / 60 - 60 * (h + 24 * d);
  s = difference - 60 * (m + 60 * (h + 24 * d));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_insn / difference));
  const unsigned cycles_per_sec =
      (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle / difference);
  printf("gpgpu_simulation_rate = %u (cycle/sec)\n", cycles_per_sec);

  if (cycles_per_sec == 0) {
    printf("gpgpu_silicon_slowdown = Nan\n");
  } else {
    printf("gpgpu_silicon_slowdown = %ux\n",
           the_gpgpusim->g_the_gpu->shader_clock() * 1000 / cycles_per_sec);
  }
  fflush(stdout);
}

int gpgpu_context::gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
  the_gpgpusim->g_the_gpu->launch(grid);
  sem_post(&(the_gpgpusim->g_sim_signal_start));
  sem_wait(&(the_gpgpusim->g_sim_signal_finish));
  return 0;
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
  // calling the CUDA PTX simulator, sending the kernel by reference and a flag
  // set to true, the flag used by the function to distinguish OpenCL calls from
  // the CUDA simulation calls which it is needed by the called function to not
  // register the exit the exit of OpenCL kernel as it doesn't register entering
  // in the first place as the CUDA kernels does
  gpgpu_cuda_ptx_sim_main_func(*grid, true);
  return 0;
}
