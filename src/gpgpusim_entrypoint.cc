/*
 * gpgpusim_entrypoint.c
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

#include <stdio.h>
#include <time.h>

#include "option_parser.h"
//#include "gpgpu-sim/gpu-sim.h"

#define MAX(a,b) (((a)>(b))?(a):(b))

struct dim3 {
   unsigned int x, y, z;
};

struct gpgpu_ptx_sim_arg *grid_params;
int g_grid_num=0;
int g_argc = 3;
const char *g_argv[] = {"", "-config","gpgpusim.config"};

unsigned int run_gpu_sim(int grid_num);
void gpgpu_ptx_sim_init_grid(const char *kernel_key,struct gpgpu_ptx_sim_arg *args, struct dim3 gridDim, struct dim3 blockDim );

int   g_network_mode = 0;
char* g_network_config_filename;
option_parser_t opp;
extern void read_environment_variables();
extern void print_splash();

extern void gpu_reg_options(option_parser_t opp);
extern void init_gpu();

time_t simulation_starttime;

void gpgpu_ptx_sim_init_perf()
{
   print_splash();
   read_environment_variables();
   opp = option_parser_create();
   option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode, "Interconnection network mode", "1");
   option_parser_register(opp, "-inter_config_file", OPT_CSTR, &g_network_config_filename, "Interconnection network config file", "mesh");
   gpu_reg_options(opp); // register GPU microrachitecture options
   option_parser_cmdline(opp, g_argc, g_argv); // parse configuration options

   srand(1); 

   fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
   option_parser_print(opp, stdout);
   init_gpu(); // initialize the GPU microarchitecture model
   fprintf(stdout, "GPU performance model initialization complete.\n");

   simulation_starttime = time((time_t *)NULL);
}

extern unsigned long long  gpu_tot_sim_insn;
extern unsigned long long  gpu_tot_sim_cycle;

int gpgpu_ptx_sim_main_perf( const char *kernel_key, struct dim3 gridDim, struct dim3 blockDim, struct gpgpu_ptx_sim_arg *grid_params )
{
   time_t current_time, difference, d, h, m, s;
   gpgpu_ptx_sim_init_grid(kernel_key,grid_params,gridDim,blockDim);

   run_gpu_sim(g_grid_num); // run a CUDA grid on the GPU microarchitecture simulator

   g_grid_num++;

   current_time = time((time_t *)NULL);
   difference = MAX(current_time - simulation_starttime, 1);

   d = difference/(3600*24);
   h = difference/3600 - 24*d;
   m = difference/60 - 60*(h + 24*d);
   s = difference - 60*(m + 60*(h + 24*d));

   fflush(stderr);
   printf("\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
          (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference );
   printf("gpgpu_simulation_rate = %u (inst/sec)\n", (unsigned)(gpu_tot_sim_insn / difference) );
   printf("gpgpu_simulation_rate = %u (cycle/sec)\n", (unsigned)(gpu_tot_sim_cycle / difference) );
   fflush(stdout);

   return 0;
}
