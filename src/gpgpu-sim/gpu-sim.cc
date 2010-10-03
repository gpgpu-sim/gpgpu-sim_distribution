/* 
 * gpu-sim.c
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Ivan Sham, Henry Wong, Dan O'Connor and the 
 * University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
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

#include "gpu-sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zlib.h"

#include "../option_parser.h"
#include "shader.h"
#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "dwf.h"
#include "warp_tracker.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../intersim/statwraper.h"
#include "../intersim/interconnect_interface.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"

#include "mem_latency_stat.h"
#include "visualizer.h"
#include "stats.h"

#include <stdio.h>
#include <string.h>

#define MAX(a,b) (((a)>(b))?(a):(b))

bool g_interactive_debugger_enabled=false;

unsigned made_read_mfs = 0;
unsigned made_write_mfs = 0;
unsigned freed_read_mfs = 0;
unsigned freed_L1write_mfs = 0;
unsigned freed_L2write_mfs = 0;
unsigned freed_dummy_read_mfs = 0;
unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_sim_insn = 0;
unsigned long long  gpu_sim_prev_insn = 0;
unsigned long long  gpu_tot_sim_cycle = 0;
unsigned long long  gpu_tot_sim_insn = 0;
unsigned long long  gpu_last_sim_cycle = 0;
unsigned long long  gpu_tot_issued_cta = 0;
unsigned long long  gpu_tot_completed_thread = 0;

unsigned int **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]
unsigned int **num_activates; //num_activates[dram chip id][bank id]
unsigned int **row_access; //row_access[dram chip id][bank id]
unsigned int **max_conc_access2samerow; //max_conc_access2samerow[dram chip id][bank id]
unsigned int **max_servicetime2samerow; //max_servicetime2samerow[dram chip id][bank id]
unsigned int gpgpu_n_sent_writes = 0;
unsigned int gpgpu_n_processed_writes = 0;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_wr_back = 0;
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;

//shader cannot send to icnt because icnt buffer is full
//Note: it is accumulative for all shaders and is never reset
//so it might increase 8 times in a cycle if we have 8 shaders
char *gpgpu_runtime_stat;
int gpu_stat_sample_freq = 10000;
int gpu_runtime_stat_flag = 0;

unsigned long long  gpu_max_cycle = 0;
unsigned long long  gpu_max_insn = 0;
int gpu_deadlock = 0;
unsigned g_next_mf_request_uid = 1;
static unsigned long long  last_gpu_sim_insn = 0;
int g_total_cta_left;

// GPGPU-Sim timing model options
int   gpu_max_cycle_opt;
int   gpu_max_insn_opt;
int   gpu_max_cta_opt;
bool  gpu_deadlock_detect;
char *gpgpu_shader_core_pipeline_opt;
int   gpgpu_dram_sched_queue_size; 
bool  gpgpu_flush_cache;
int   gpgpu_mem_address_mask;
int   gpgpu_cflog_interval;

/* Defining Clock Domains
basically just the ratio is important */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  


char * gpgpu_clock_domains;

/* GPU uArch parameters */
unsigned int gpu_n_mem_per_ctrlr;
int gpu_n_tpc;
char *gpgpu_dwf_hw_opt;
bool gpgpu_thread_swizzling;
unsigned int more_thread = 1;

#define MEM_LATENCY_STAT_IMPL
#include "mem_latency_stat.h"

int   g_ptx_inst_debug_to_file;
char* g_ptx_inst_debug_file;
int   g_ptx_inst_debug_thread_uid;

int   g_ptx_convert_to_ptxplus;
int   g_ptx_save_converted_ptxplus;

void visualizer_options(option_parser_t opp);

void gpgpu_sim::reg_options(option_parser_t opp)
{
   option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &m_shader_config->model, 
               "0 = no recombination, 1 = post-dominator, 2 = MIMD, 3 = dynamic warp formation", "0");
   option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &m_memory_config->scheduler_type, 
               "0 = fifo, 1 = FR-FCFS (defaul)", "1");

   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");

   option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &m_shader_config->gpgpu_cache_texl1_opt, 
                  "per-shader L1 texture cache  (READ-ONLY) config, i.e., {<nsets>:<linesize>:<assoc>:<repl>|none}",
                  "512:64:2:L");

   option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_shader_config->gpgpu_cache_constl1_opt, 
                  "per-shader L1 constant memory cache  (READ-ONLY) config, i.e., {<nsets>:<linesize>:<assoc>:<repl>|none}",
                  "64:64:2:L");

   option_parser_register(opp, "-gpgpu_no_dl1", OPT_BOOL, &m_shader_config->gpgpu_no_dl1, 
                "no dl1 cache (voids -gpgpu_cache:dl1 option)",
                "0");

   option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &m_shader_config->gpgpu_cache_dl1_opt, 
                  "shader L1 data cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}",
                  "256:128:1:L");

   option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR, &m_shader_config->gpgpu_cache_il1_opt, 
                  "shader L1 instruction cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}",
                  "4:256:4:L");

   option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_memory_config->gpgpu_cache_dl2_opt, 
                  "unified banked L2 data cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}; disabled by default",
                  NULL); 

   option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &m_shader_config->gpgpu_perfect_mem, 
                "enable perfect memory mode (no cache miss)",
                "0");

   option_parser_register(opp, "-gpgpu_sm_uarch", OPT_CSTR, &m_shader_config->pipeline_model,
                  "shader core uarch model [GPGPUSIM_ORIG,GT200] (default=GPGPUSIM_ORIG)",
                  "GPGPUSIM_ORIG");

   option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                  "shader core pipeline config, i.e., {<nthread>:<warpsize>:<pipe_simd_width>}",
                  "256:32:32");

   option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &m_shader_config->gpgpu_shader_registers, 
                "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                "8192");

   option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &m_shader_config->max_cta_per_core, 
                "Maximum number of concurrent CTAs in shader (default 8)",
                "8");

   option_parser_register(opp, "-gpgpu_n_shader", OPT_UINT32, &m_n_shader, 
                "number of shaders in gpu",
                "8");
   option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem, 
                "number of memory modules (e.g. memory controllers) in gpu",
                "8");
   option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr, 
                "number of memory chips per memory controller",
                "1");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat, 
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");

   option_parser_register(opp, "-gpgpu_dwf_heuristic", OPT_UINT32, &gpgpu_dwf_heuristic, 
                "DWF scheduling heuristic: 0 = majority, 1 = minority, 2 = timestamp, 3 = pdom priority, 4 = pc-based, 5 = max-heap",
                "0");

   option_parser_register(opp, "-gpgpu_dwf_reg_bankconflict", OPT_BOOL, &m_shader_config->gpgpu_dwf_reg_bankconflict, 
                "bank conflict model used in MICRO'07/TACO'09 work (default=disabled)",
                "0");

   option_parser_register(opp, "-gpgpu_dwf_regbk", OPT_BOOL, &gpgpu_dwf_regbk, 
                "Have dwf scheduler to avoid bank conflict",
                "1");

   option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &m_memory_config->gpgpu_memlatency_stat, 
               "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
               "0");

   option_parser_register(opp, "-gpu_n_mshr_per_shader", OPT_UINT32, &m_shader_config->n_mshr_per_shader, 
                "Number of MSHRs per shader",
                "64");

   option_parser_register(opp, "-gpgpu_interwarp_mshr_merge", OPT_INT32, &m_shader_config->gpgpu_interwarp_mshr_merge, 
               "interwarp coalescing",
               "0");

   option_parser_register(opp, "-gpgpu_dram_sched_queue_size", OPT_INT32, &m_memory_config->gpgpu_dram_sched_queue_size, 
               "0 = unlimited (default); # entries per chip",
               "0");

   option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &m_memory_config->gpgpu_dram_buswidth, 
                "default = 4 bytes (8 bytes per cycle at DDR)",
                "4");

   option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &m_memory_config->gpgpu_dram_burst_length, 
                "Burst length of each DRAM request (default = 4 DDR cycle)",
                "4");

   option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &m_memory_config->gpgpu_dram_timing_opt, 
               "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tWTR}",
               "4:2:8:12:21:13:34:9:4:5");


   option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32, &gpgpu_mem_address_mask, 
               "0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits",
               "0");

   option_parser_register(opp, "-gpgpu_flush_cache", OPT_BOOL, &gpgpu_flush_cache, 
                "Flush cache at the end of each kernel call",
                "0");

   option_parser_register(opp, "-gpgpu_pre_mem_stages", OPT_UINT32, &m_shader_config->gpgpu_pre_mem_stages, 
                "default = 0 pre-memory pipeline stages",
                "0");

   option_parser_register(opp, "-gpgpu_no_divg_load", OPT_BOOL, &m_shader_config->gpgpu_no_divg_load, 
                "Don't allow divergence on load (meaningful for dynamic warp formation only)",
                "1");

   option_parser_register(opp, "-gpgpu_dwf_hw", OPT_CSTR, &gpgpu_dwf_hw_opt, 
                  "dynamic warp formation hw config, i.e., {<#LUT_entries>:<associativity>|none}",
                  "32:2");

   option_parser_register(opp, "-gpgpu_thread_swizzling", OPT_BOOL, &gpgpu_thread_swizzling, 
                "Thread Swizzling (1=on, 0=off)",
                "0");

   option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &m_shader_config->gpgpu_shmem_size, 
                "Size of shared memory per shader core (default 16kB)",
                "16384");

   option_parser_register(opp, "-gpgpu_shmem_pipe_speedup", OPT_INT32, &m_shader_config->gpgpu_shmem_pipe_speedup,  
                "Number of groups each warp is divided for shared memory bank conflict check",
                "2");

   option_parser_register(opp, "-gpgpu_cache_wt_through", OPT_BOOL, &m_shader_config->gpgpu_cache_wt_through, 
                "L1 cache become write through (1=on, 0=off)", 
                "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");

   option_parser_register(opp, "-gpgpu_n_cache_bank", OPT_INT32, &m_shader_config->gpgpu_n_cache_bank, 
               "Number of banks in L1 cache, also for memory coalescing stall", 
               "1");

   option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &m_shader_config->gpgpu_warpdistro_shader, 
               "Specify which shader core to collect the warp size distribution from", 
               "-1");


   option_parser_register(opp, "-gpgpu_pdom_sched_type", OPT_INT32, &m_pdom_sched_type, 
               "0 = first ready warp found, 1 = random, 8 = loose round robin", 
               "8");

   option_parser_register(opp, "-gpgpu_stall_on_use", OPT_BOOL,
                &m_shader_config->gpgpu_stall_on_use,
                "Enable stall-on-use",
                "1");

   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32, 
               &gpgpu_ptx_instruction_classification, 
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)", 
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode, 
               "Select between Performance (default) or Functional simulation (1)", 
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains, 
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");

   option_parser_register(opp, "-gpgpu_shmem_port_per_bank", OPT_INT32, &m_shader_config->gpgpu_shmem_port_per_bank, 
               "Number of access processed by a shared memory bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_cache_port_per_bank", OPT_INT32, &m_shader_config->gpgpu_cache_port_per_bank, 
               "Number of access processed by a cache bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_const_port_per_bank", OPT_INT32, &m_shader_config->gpgpu_const_port_per_bank,
               "Number of access processed by a constant cache bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-gpu_concentration", OPT_INT32, &gpu_concentration, 
               "Number of shader cores per interconnection port (default = 1)", 
               "1");
   option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL, &m_shader_config->gpgpu_local_mem_map, 
               "Mapping from local memory space address to simulated GPU physical address space (default = enabled)", 
               "1");
   option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &m_shader_config->gpgpu_num_reg_banks, 
               "Number of register banks (default = 8)", 
               "8");
   option_parser_register(opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &m_shader_config->gpgpu_reg_bank_use_warp_id,
            "Use warp ID in mapping registers to banks (default = off)",
            "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, 
                &g_ptx_inst_debug_to_file, 
                "Dump executed instructions' debug information to file", 
                "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file, 
                  "Executed instructions' debug output file",
                  "inst_debug.txt");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32, &g_ptx_inst_debug_thread_uid, 
               "Thread UID for executed instructions' debug output", 
               "1");
   option_parser_register(opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL,
                &g_ptx_convert_to_ptxplus,
                "Convert embedded ptx to ptxplus",
                "0");
   option_parser_register(opp, "-gpgpu_ptx_save_converted_ptxplus", OPT_BOOL,
                &g_ptx_save_converted_ptxplus,
                "Saved converted ptxplus to a file",
                "0");
   option_parser_register(opp, "-gpgpu_operand_collector", OPT_BOOL, &m_shader_config->gpgpu_operand_collector,
               "Enable operand collector model (default = off)",
               "0");
   option_parser_register(opp, "-gpgpu_operand_collector_num_units", OPT_INT32, &m_shader_config->gpgpu_operand_collector_num_units,
               "number of collecture units (default = 4)", 
               "4");
   option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu", OPT_INT32, &m_shader_config->gpgpu_operand_collector_num_units_sfu,
               "number of collecture units (default = 4)", 
               "4");
   option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &m_shader_config->gpgpu_coalesce_arch, 
                           "Coalescing arch (default = 13, anything else is off for now)", 
                           "13");
   addrdec_setoption(opp);
   L2c_options(opp);
   visualizer_options(opp);
   ptx_file_line_stats_options(opp);

   m_options_set = true;
}

/////////////////////////////////////////////////////////////////////////////

int mem2device(int memid) 
{
   return memid + gpu_n_tpc;
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound)
{
   i.x++;
   if ( i.x >= bound.x ) {
      i.x = 0;
      i.y++;
      if ( i.y >= bound.y ) {
         i.y = 0;
         if( i.z < bound.z ) 
            i.z++;
      }
   }
}


void gpgpu_sim::launch( kernel_info_t &kinfo )
{
   unsigned cta_size = kinfo.threads_per_cta();
   if ( cta_size > m_shader_config->n_thread_per_shader ) {
      printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
      printf("                 CTA size (x*y*z) = %u, max supported = %u\n", cta_size, 
             m_shader_config->n_thread_per_shader );
      printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
      printf("                 modify the CUDA source to decrease the kernel block size.\n");
      abort();
   }

   m_running_kernels.push_back(kinfo);
}

void gpgpu_sim::next_grid( unsigned &grid_num, class function_info *&entry )
{
   grid_num = ++m_grid_num;
   m_the_kernel = m_running_kernels.front();
   m_running_kernels.pop_front();
   entry = m_the_kernel.entry();
}

gpgpu_sim::gpgpu_sim()
{ 
   m_options_set=false;
   m_grid_num=0; 
   m_shader_config = (shader_core_config*)calloc(1,sizeof(shader_core_config));
   m_shader_stats = (shader_core_stats*)calloc(1,sizeof(shader_core_stats));
   m_memory_config = (memory_config*)calloc(1,sizeof(memory_config));
   m_memory_stats = NULL;
}

void set_ptx_warp_size(unsigned warp_size);

void gpgpu_sim::init_gpu() 
{
   assert( m_options_set );
    
   gpu_max_cycle = gpu_max_cycle_opt;
   gpu_max_insn  = gpu_max_insn_opt;

   int ntok = sscanf(gpgpu_shader_core_pipeline_opt,"%d:%d", 
                     &m_shader_config->n_thread_per_shader,
                     &m_shader_config->warp_size);
   set_ptx_warp_size(m_shader_config->warp_size);
    
   m_shader_config->max_warps_per_shader =  m_shader_config->n_thread_per_shader/m_shader_config->warp_size;
   assert( !(m_shader_config->n_thread_per_shader % m_shader_config->warp_size) );

   m_shader_stats->num_warps_issuable = (int*) calloc(m_shader_config->max_warps_per_shader+1, sizeof(int));
   m_shader_stats->num_warps_issuable_pershader = (int*) calloc(m_n_shader, sizeof(int));
   m_shader_stats->shader_cycle_distro = (unsigned int*) calloc(m_shader_config->warp_size + 3, sizeof(unsigned int));

   if(ntok != 2) {
      printf("GPGPU-Sim uArch: error while parsing configuration string gpgpu_shader_core_pipeline_opt\n");
      abort();
   }

   sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq, &gpu_runtime_stat_flag);

   m_shader_config->pdom_sched_type = m_pdom_sched_type;
   m_shader_config->gpgpu_n_shmem_bank=16;

   m_sc = (shader_core_ctx**) calloc(m_n_shader, sizeof(shader_core_ctx*));
   for (unsigned i=0;i<m_n_shader;i++) {
      m_sc[i] = (shader_core_ctx*)calloc(sizeof(shader_core_ctx),1);
      m_sc[i] = new (m_sc[i]) shader_core_ctx(this,"sh",i,i/gpu_concentration,m_shader_config,m_shader_stats);
   }

   ptx_file_line_stats_create_exposed_latency_tracker(m_n_shader);

   // initialize dynamic warp formation scheduler
   int dwf_lut_size, dwf_lut_assoc;
   sscanf(gpgpu_dwf_hw_opt,"%d:%d", &dwf_lut_size, &dwf_lut_assoc);
   char *dwf_hw_policy_opt = strchr(gpgpu_dwf_hw_opt, ';');
   int insn_size = 1; // for cuda-sim
   create_dwf_schedulers(m_n_shader, dwf_lut_size, dwf_lut_assoc, 
                         m_shader_config->warp_size, m_shader_config->warp_size, 
                         m_shader_config->n_thread_per_shader, insn_size, 
                         gpgpu_dwf_heuristic, dwf_hw_policy_opt );

   // always use no diverge on load for stack based SIMT execution (PDOM)
   m_shader_config->gpgpu_no_divg_load = (m_shader_config->model != DWF) || 
      (m_shader_config->gpgpu_no_divg_load && (m_shader_config->model == DWF)); 
   m_shader_config->m_using_dwf_rrstage = (m_shader_config->model == DWF);
   m_shader_config->using_commit_queue = (m_shader_config->model == DWF || m_shader_config->model == POST_DOMINATOR);

   m_shader_config->gpgpu_dwf_rr_stage_n_reg_banks=8;  

   assert(m_n_shader % gpu_concentration == 0);
   gpu_n_tpc = m_n_shader / gpu_concentration;

   addrdec_setnchip(m_n_mem);
   m_memory_partition_unit = new memory_partition_unit*[m_n_mem];
   for (unsigned i=0;i<m_n_mem;i++) 
      m_memory_partition_unit[i] = new memory_partition_unit(i, m_memory_config);
   m_memory_stats = new memory_stats_t(m_n_mem,m_n_shader,m_shader_config,m_memory_config);
   for (unsigned i=0;i<m_n_mem;i++) 
      m_memory_partition_unit[i]->set_stats(m_memory_stats);

   concurrent_row_access = (unsigned int**) calloc(m_n_mem, sizeof(unsigned int*));
   num_activates = (unsigned int**) calloc(m_n_mem, sizeof(unsigned int*));
   row_access = (unsigned int**) calloc(m_n_mem, sizeof(unsigned int*));
   max_conc_access2samerow = (unsigned int**) calloc(m_n_mem, sizeof(unsigned int*));
   max_servicetime2samerow = (unsigned int**) calloc(m_n_mem, sizeof(unsigned int*));

   for (unsigned i=0;i<m_n_mem ;i++ ) {
      concurrent_row_access[i] = (unsigned int*) calloc(m_memory_config->gpu_mem_n_bk, sizeof(unsigned int));
      row_access[i] = (unsigned int*) calloc(m_memory_config->gpu_mem_n_bk, sizeof(unsigned int));
      num_activates[i] = (unsigned int*) calloc(m_memory_config->gpu_mem_n_bk, sizeof(unsigned int));
      max_conc_access2samerow[i] = (unsigned int*) calloc(m_memory_config->gpu_mem_n_bk, sizeof(unsigned int));
      max_servicetime2samerow[i] = (unsigned int*) calloc(m_memory_config->gpu_mem_n_bk, sizeof(unsigned int));
   }

   m_memory_stats = new memory_stats_t(m_n_mem,m_n_shader,m_shader_config,m_memory_config);

   m_shader_stats->max_return_queue_length = (unsigned int*) calloc(m_n_shader, sizeof(unsigned int));

   icnt_init(gpu_n_tpc, m_n_mem,m_shader_config);

   time_vector_create(NUM_MEM_REQ_STAT,MR_2SH_ICNT_INJECTED);
   fprintf(stdout, "GPU performance model initialization complete.\n");
   init_clock_domains();
}

int gpgpu_sim::shared_mem_size() const
{
   return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const
{
   return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const
{
   return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const
{
   return core_freq/1000;
}

void gpgpu_sim::set_prop( cudaDeviceProp *prop )
{
   m_cuda_properties = prop;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const
{
   return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const
{
   return m_shader_config->model;
}

void gpgpu_sim::init_clock_domains(void ) 
{
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf", 
          &core_freq, &icnt_freq, &l2_freq, &dram_freq);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;        
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   l2_period = 1/l2_freq;
   core_time = 0 ;
   dram_time = 0 ;
   icnt_time = 0;
   l2_time = 0;
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
}

void gpgpu_sim::reinit_clock_domains(void)
{
   core_time = 0;
   dram_time = 0;
   icnt_time = 0;
   l2_time = 0;
}

// return the number of cycle required to run all the trace on the gpu 
unsigned int gpgpu_sim::run_gpu_sim() 
{
   // run a CUDA grid on the GPU microarchitecture simulator
   int grid_num = m_grid_num; 
   kernel_info_t &entry = m_the_kernel;
   size_t program_size = get_kernel_code_size(entry.entry());

   int not_completed;
   int mem_busy;
   int icnt2mem_busy;

   gpu_sim_cycle = 0;
   not_completed = 1;
   mem_busy = 1;
   icnt2mem_busy = 1;
   g_next_mf_request_uid = 1;
   more_thread = 1;
   gpu_sim_insn = 0;
   m_shader_stats->gpu_sim_insn_no_ld_const = 0;
   m_shader_stats->gpu_completed_thread = 0;

   reinit_clock_domains();
   set_param_gpgpu_num_shaders(m_n_shader);
   for (unsigned i=0;i<m_n_shader;i++) 
      m_sc[i]->reinit(0,m_shader_config->n_thread_per_shader,true);
   if (gpu_max_cta_opt != 0) {
      g_total_cta_left = gpu_max_cta_opt;
   } else {
      g_total_cta_left =  m_the_kernel.num_blocks();
   }
   if (gpu_max_cta_opt != 0) {
      // the maximum number of CTA has been reached, stop any further simulation
      if (gpu_tot_issued_cta >= (unsigned)gpu_max_cta_opt) {
         return 0;
      }
   }

   if (gpu_max_cycle && (gpu_tot_sim_cycle + gpu_sim_cycle) >= gpu_max_cycle) {
      return gpu_sim_cycle;
   }
   if (gpu_max_insn && (gpu_tot_sim_insn + gpu_sim_insn) >= gpu_max_insn) {
      return gpu_sim_cycle;
   }

   // refind the diverge/reconvergence pairs
   dwf_reset_reconv_pt();
   dwf_process_reconv_pts(entry.entry());
   dwf_reinit_schedulers(m_n_shader);

   // initialize the control-flow, memory access, memory latency logger
   create_thread_CFlogger( m_n_shader, m_shader_config->n_thread_per_shader, program_size, 0, gpgpu_cflog_interval );
   shader_CTA_count_create( m_n_shader, gpgpu_cflog_interval);
   if (gpgpu_cflog_interval != 0) {
      insn_warp_occ_create( m_n_shader, m_shader_config->warp_size, program_size );
      shader_warp_occ_create( m_n_shader, m_shader_config->warp_size, gpgpu_cflog_interval);
      shader_mem_acc_create( m_n_shader, m_n_mem, 4, gpgpu_cflog_interval);
      shader_mem_lat_create( m_n_shader, gpgpu_cflog_interval);
      shader_cache_access_create( m_n_shader, 3, gpgpu_cflog_interval);
      set_spill_interval (gpgpu_cflog_interval * 40);
   }

   // calcaulte the max cta count and cta size for local memory address mapping
   m_shader_config->gpu_max_cta_per_shader = m_sc[0]->max_cta(entry.entry());
   //gpu_max_cta_per_shader is limited by number of CTAs if not enough    
   if (m_the_kernel.num_blocks() < m_shader_config->gpu_max_cta_per_shader*m_n_shader) { 
      m_shader_config->gpu_max_cta_per_shader = (m_the_kernel.num_blocks() / m_n_shader);
      if (m_the_kernel.num_blocks() % m_n_shader)
         m_shader_config->gpu_max_cta_per_shader++;
   }
   unsigned int gpu_cta_size = m_the_kernel.threads_per_cta();
   m_shader_config->gpu_padded_cta_size = (gpu_cta_size%32) ? 32*((gpu_cta_size/32)+1) : gpu_cta_size;

   if (g_network_mode) {
      icnt_init_grid(); 
   }

   last_gpu_sim_insn = 0;
   while (not_completed || mem_busy || icnt2mem_busy) {
      gpu_sim_loop();
      not_completed = 0;
      for (unsigned i=0;i<m_n_shader;i++) 
         not_completed += m_sc[i]->get_not_completed();
      mem_busy = 0; 
      for (unsigned i=0;i<m_n_mem;i++) 
         mem_busy += m_memory_partition_unit[i]->busy();
      icnt2mem_busy = icnt_busy();
      if (gpu_max_cycle && (gpu_tot_sim_cycle + gpu_sim_cycle) >= gpu_max_cycle) 
         break;
      if (gpu_max_insn && (gpu_tot_sim_insn + gpu_sim_insn) >= gpu_max_insn) 
         break;
      if (gpu_deadlock_detect && gpu_deadlock) 
         break;
   }
   m_memory_stats->memlatstat_lat_pw(m_n_shader,m_shader_config->n_thread_per_shader,m_shader_config->warp_size);
   gpu_tot_sim_cycle += gpu_sim_cycle;
   gpu_tot_sim_insn += gpu_sim_insn;
   gpu_tot_completed_thread += m_shader_stats->gpu_completed_thread;
   
   ptx_file_line_stats_write_file();

   printf("stats for grid: %d\n", grid_num);
   gpu_print_stat();
   if (g_network_mode) {
      interconnect_stats();
      printf("----------------------------Interconnect-DETAILS---------------------------------" );
      icnt_overal_stat();
      printf("----------------------------END-of-Interconnect-DETAILS-------------------------" );
   }
   if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_QUEUELOGS ) {
      for (unsigned i=0;i<m_n_mem;i++) 
         m_memory_partition_unit[i]->queue_latency_log_dump(stdout);
      if (m_memory_config->gpgpu_cache_dl2_opt) {
         for(unsigned i=0; i<m_n_mem; i++) 
            m_memory_partition_unit[i]->L2c_log(DUMPLOG);
         L2c_latency_log_dump();
      }
   }

   if (gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("GPGPU-Sim uArch: ERROR ** deadlock detected: last writeback @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      unsigned num_cores=0;
      for (unsigned i=0;i<m_n_shader;i++) {
         unsigned not_completed = m_sc[i]->get_not_completed();
         if( not_completed ) {
             if ( !num_cores )  {
                 printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing instructions [core(# threads)]:\n" );
                 printf("GPGPU-Sim uArch: DEADLOCK  %u(%u)", i, not_completed);
             } else if (num_cores < 8 ) {
                 printf(" %u(%u)", i, not_completed);
             } else if (num_cores == 8 ) {
                 printf(" + others ... ");
             }
             num_cores++;
         }
      }
      printf("\n");
      for (unsigned i=0;i<m_n_mem;i++) {
         mem_busy += m_memory_partition_unit[i]->busy();
         if( mem_busy ) 
             printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u still busy\n", i);
      }
      if( icnt_busy() ) 
         printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      printf("\nRe-run the simulator in gdb and use debug routines in .gdbinit to debug this\n");
      fflush(stdout);
      abort();
   }
   return gpu_sim_cycle;
}

void gpgpu_sim::gpu_print_stat() const
{  
   unsigned i;
    
   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_sim_no_ld_const_insn = %lld\n", m_shader_stats->gpu_sim_insn_no_ld_const);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_completed_thread = %lld\n", m_shader_stats->gpu_completed_thread);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)gpu_tot_sim_insn / gpu_tot_sim_cycle);
   printf("gpu_tot_completed_thread = %lld\n", gpu_tot_completed_thread);
   printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);
   printf("gpgpu_n_sent_writes = %d\n", gpgpu_n_sent_writes);
   printf("gpgpu_n_processed_writes = %d\n", gpgpu_n_processed_writes);

   // performance counter for stalls due to congestion.
   printf("gpu_stall_by_MSHRwb= %d\n", m_shader_stats->gpu_stall_by_MSHRwb);
   printf("gpu_stall_shd_mem  = %d\n", m_shader_stats->gpu_stall_shd_mem );
   printf("gpu_stall_wr_back  = %d\n", gpu_stall_wr_back );
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh   );
   printf("gpu_stall_sh2icnt    = %d\n", m_shader_stats->gpu_stall_sh2icnt   );
   // performance counter that are not local to one shader
   shader_print_accstats(stdout);

   m_memory_stats->memlatstat_print(m_n_mem,m_memory_config->gpu_mem_n_bk);
   printf("max return queue length = ");
   for (unsigned i=0;i<m_n_shader;i++) {
      printf("%d ", m_shader_stats->max_return_queue_length[i]);
   }
   printf("\n");
   // merge misses
   printf("L1 read misses = %d\n", m_shader_stats->L1_read_miss);
   printf("L1 write misses = %d\n", m_shader_stats->L1_write_miss);
   printf("L1 write hit on misses = %d\n", m_shader_stats->L1_write_hit_on_miss);
   printf("L1 writebacks = %d\n", m_shader_stats->L1_writeback);
   printf("L1 texture misses = %d\n", m_shader_stats->L1_texture_miss);
   printf("L1 const misses = %d\n", m_shader_stats->L1_const_miss);
   m_memory_stats->print(stdout);
   printf("made_read_mfs = %d\n", made_read_mfs);
   printf("made_write_mfs = %d\n", made_write_mfs);
   printf("freed_read_mfs = %d\n", freed_read_mfs);
   printf("freed_L1write_mfs = %d\n", freed_L1write_mfs);
   printf("freed_L2write_mfs = %d\n", freed_L2write_mfs);
   printf("freed_dummy_read_mfs = %d\n", freed_dummy_read_mfs);

   printf("gpgpu_n_mem_read_local = %d\n", m_shader_stats->gpgpu_n_mem_read_local);
   printf("gpgpu_n_mem_write_local = %d\n", m_shader_stats->gpgpu_n_mem_write_local);
   printf("gpgpu_n_mem_read_global = %d\n", m_shader_stats->gpgpu_n_mem_read_global);
   printf("gpgpu_n_mem_write_global = %d\n", m_shader_stats->gpgpu_n_mem_write_global);
   printf("gpgpu_n_mem_texture = %d\n", m_shader_stats->gpgpu_n_mem_texture);
   printf("gpgpu_n_mem_const = %d\n", m_shader_stats->gpgpu_n_mem_const);

   printf("max_n_mshr_used = ");
   for (unsigned i=0; i< m_n_shader; i++) printf("%d ", m_sc[i]->get_max_mshr_used() );
   printf("\n");

   if (m_memory_config->gpgpu_cache_dl2_opt) {
      m_memory_stats->L2c_print_stat( m_n_mem );
   }
   for (unsigned i=0;i<m_n_mem;i++) 
      m_memory_partition_unit[i]->print(stdout);

   unsigned a,m;
   for (unsigned i=0, a=0, m=0;i<m_n_shader;i++) 
      m_sc[i]->L1cache_print(stdout,a,m);
   printf("L1 Data Cache Total Miss Rate = %0.3f\n", (float)m/a);
   for (i=0,a=0,m=0;i<m_n_shader;i++) 
       m_sc[i]->L1texcache_print(stdout,a,m);
   printf("L1 Texture Cache Total Miss Rate = %0.3f\n", (float)m/a);
   for (i=0,a=0,m=0;i<m_n_shader;i++) 
       m_sc[i]->L1constcache_print(stdout,a,m);
   printf("L1 Const Cache Total Miss Rate = %0.3f\n", (float)m/a);

   if (m_memory_config->gpgpu_cache_dl2_opt) 
      L2c_print_cache_stat();
   printf("n_regconflict_stall = %d\n", n_regconflict_stall);

   if (m_shader_config->model == DWF) {
      dwf_print_stat(stdout);
   }

   if (m_shader_config->model == POST_DOMINATOR) {
      printf("num_warps_issuable:");
      for (unsigned i=0;i<(m_shader_config->max_warps_per_shader+1);i++) {
         printf("%d ", m_shader_stats->num_warps_issuable[i]);
      }
      printf("\n");
   }

   printf("gpgpu_commit_pc_beyond_two = %d\n", m_shader_stats->gpgpu_commit_pc_beyond_two);

   print_shader_cycle_distro( stdout );

   print_thread_pc_histogram( stdout );

   if (gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }
   time_vector_print();

   fflush(stdout);
}


// performance counter that are not local to one shader
void gpgpu_sim::shader_print_accstats( FILE* fout ) const
{
   fprintf(fout, "gpgpu_n_load_insn  = %d\n", m_shader_stats->gpgpu_n_load_insn);
   fprintf(fout, "gpgpu_n_store_insn = %d\n", m_shader_stats->gpgpu_n_store_insn);
   fprintf(fout, "gpgpu_n_shmem_insn = %d\n", m_shader_stats->gpgpu_n_shmem_insn);
   fprintf(fout, "gpgpu_n_tex_insn = %d\n", m_shader_stats->gpgpu_n_tex_insn);
   fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", m_shader_stats->gpgpu_n_const_insn);
   fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", m_shader_stats->gpgpu_n_param_insn);

   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", m_shader_stats->gpgpu_n_shmem_bkconflict);
   fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", m_shader_stats->gpgpu_n_cache_bkconflict);   

   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n", m_shader_stats->gpgpu_n_intrawarp_mshr_merge);
   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", m_shader_stats->gpgpu_n_cmem_portconflict);

   fprintf(fout, "gpgpu_n_partial_writes = %d\n", m_shader_stats->gpgpu_n_partial_writes);

   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][bk_conf] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][bk_conf] = %d\n", 
           m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] + 
           m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] + 
           m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] + 
           m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]   
           ); // coalescing stall at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n", 
           m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] + 
           m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] + 
           m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] + 
           m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_ST][COAL_STALL]    
           ); // coalescing stall + bank conflict at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", m_shader_stats->gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

   fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n", m_shader_stats->gpu_reg_bank_conflict_stalls);
}

unsigned gpgpu_sim::threads_per_core() const 
{ 
   return m_shader_config->n_thread_per_shader; 
}

void gpgpu_sim::mem_instruction_stats(inst_t* warp)
{
   for (unsigned i=0; i< (unsigned) m_shader_config->warp_size; i++) {
      if (warp[i].hw_thread_id == -1) continue; //bubble 
         //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
      bool store = is_store(warp[i]);
      switch (warp[i].space.get_type()) {
      case undefined_space:
      case reg_space:
         break;
      case shared_space:
         m_shader_stats->gpgpu_n_shmem_insn++;
         break;
      case const_space:
         m_shader_stats->gpgpu_n_const_insn++;
         break;
      case param_space_kernel:
      case param_space_local:
         m_shader_stats->gpgpu_n_param_insn++;
         break;
      case tex_space:
         m_shader_stats->gpgpu_n_tex_insn++;
         break;
      case global_space:
      case local_space:
         if (store){ 
            m_shader_stats->gpgpu_n_store_insn++;
         } else {
            m_shader_stats->gpgpu_n_load_insn++;
         }
         break;
      default:
         abort();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////
// Wrapper function for shader cores' memory system: 
////////////////////////////////////////////////////////////////////////////////////

// Check the memory system for buffer availability
unsigned char gpgpu_sim::fq_has_buffer(unsigned long long int addr, int bsize, bool write, int sid )
{
   //requests should be single always now
   int rsize = bsize;
   //maintain similar functionality with fq_push, if its a read, bsize is the load size, not the request's size
   if (!write) {
       rsize = READ_PACKET_SIZE;
   }
   return check_icnt_has_buffer(addr, rsize, sid);
}

unsigned char gpgpu_sim::check_icnt_has_buffer(unsigned long long int addr, int bsize, int sid )
{
   int tpc_id = sid / gpu_concentration;
   return icnt_has_buffer(tpc_id, bsize);
}

int gpgpu_sim::issue_mf_from_fq(mem_fetch *mf)
{
   unsigned destination = mf->get_tlx_addr().chip;
   unsigned tpc_id = mf->get_tpc();
   mf->set_status(IN_ICNT2MEM,MR_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle);
   if (!mf->get_is_write()) {
      mf->set_type(RD_REQ);
      icnt_push(tpc_id, mem2device(destination), (void*)mf, mf->get_ctrl_size() );
   } else {
      mf->set_type(WT_REQ);
      icnt_push(tpc_id, mem2device(destination), (void*)mf, mf->size());
      gpgpu_n_sent_writes++;
   }
   return 0;
}

void shader_core_ctx::fill_shd_L1_with_new_line(mem_fetch * mf) 
{
   // When the data arrives, it flags all the appropriate MSHR
   // entries accordingly (by checking the address in each entry ) 
   if ( mf->isinst() ) {
       m_L1I->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
       m_warp[mf->get_wid()].clear_imiss_pending();
       delete mf->get_mshr();
   } else {
       m_mshr_unit->mshr_return_from_mem(mf->get_mshr());
       if (mf->istexture()) 
           m_L1T->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
       else if (mf->isconst()) 
           m_L1C->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
       else if (!m_config->gpgpu_no_dl1) 
           m_L1D->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
   }
   freed_read_mfs++;
   delete mf;
}

void shader_core_ctx::store_ack( class mem_fetch *mf )
{
   if (!strcmp("GT200",m_config->pipeline_model) )  {
    unsigned warp_id = mf->get_wid();
    m_warp[warp_id].dec_store_req();
   }
}

void gpgpu_sim::fq_pop(int tpc_id) 
{
    mem_fetch *mf = (mem_fetch*) icnt_pop(tpc_id);
    if (!mf) 
        return;
    assert(mf->get_type() == REPLY_DATA);
    mf->set_status(IN_ICNT2SHADER,MR_2SH_FQ_POP,gpu_sim_cycle+gpu_tot_sim_cycle);
    if (mf->get_is_write()) { 
        m_sc[mf->get_sid()]->store_ack(mf);
        delete mf;
    } else {
        m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
        m_sc[mf->get_sid()]->fill_shd_L1_with_new_line(mf);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA). 
 *  
 * @param kernel 
 *    object that tells us which kernel to ask for a CTA from 
 */

void shader_core_ctx::issue_block2core( kernel_info_t &kernel ) 
{
    // find a free CTA context 
    unsigned free_cta_hw_id=(unsigned)-1;
    unsigned max_concurrent_cta_this_kernel = max_cta(kernel.entry());
    assert( max_concurrent_cta_this_kernel <= MAX_CTA_PER_SHADER );
    for (unsigned i=0;i<max_concurrent_cta_this_kernel;i++ ) {
      if( m_cta_status[i]==0 ) {
         free_cta_hw_id=i;
         break;
      }
    }
    assert( free_cta_hw_id!=(unsigned)-1 );

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size; 
    if (cta_size%m_config->warp_size)
      padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);
    unsigned start_thread = free_cta_hw_id * padded_cta_size;
    unsigned end_thread  = start_thread +  cta_size;

    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread,false);
     
    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    for (unsigned i = start_thread; i<end_thread; i++) {
        m_thread[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i/m_config->warp_size;
        nthreads_in_block += ptx_sim_init_thread(kernel,&m_thread[i].m_functional_model_thread_state,m_sid,i,cta_size-(i-start_thread),m_config->n_thread_per_shader,this,free_cta_hw_id,warp_id);
        warps.set( warp_id );
    }
    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    allocate_barrier( free_cta_hw_id, warps );

    // initialize the SIMT stacks and fetch hardware
    init_warps(start_thread, end_thread);

    m_n_active_cta++;
    g_total_cta_left-=1; // used for exiting early from simulation

    shader_CTA_count_log(m_sid, 1);
    
    printf("GPGPU-Sim uArch: Shader %d initialized CTA #%d with hw tids from %d to %d @(%lld,%lld)", 
          m_sid, free_cta_hw_id, start_thread, start_thread+nthreads_in_block, gpu_sim_cycle, gpu_tot_sim_cycle );
    printf(" active threads = %d\n", get_not_completed() );
    
}


///////////////////////////////////////////////////////////////////////////////////////////
// wrapper code to to create an illusion of a memory controller with L2 cache.

void memory_partition_unit::push( mem_fetch* req, unsigned long long cycle ) 
{
    if (req) {
        m_request_tracker.insert(req);
        rop_delay_t r;
        r.req = req;
        r.ready_cycle = cycle + 115; // Add 115*4=460 delay cycles
        m_rop.push(r);
    }
    if ( !m_rop.empty() && (cycle >= m_rop.front().ready_cycle) ) {
        mem_fetch* mf = m_rop.front().req;
        m_rop.pop();
        m_stats->memlatstat_icnt2mem_pop(mf);
        if (m_config->gpgpu_cache_dl2_opt) {
            if (m_config->gpgpu_l2_readoverwrite && mf->get_is_write())
                m_icnt2cache_write_queue->push(mf,gpu_sim_cycle);
            else
                m_icnt2cache_queue->push(mf,gpu_sim_cycle);
            m_accessLocality->access(mf); 
            mf->set_status(IN_CBTOL2QUEUE,MR_DRAMQ,gpu_sim_cycle+gpu_tot_sim_cycle);
        } else {
            m_dram->push(mf); 
            mf->set_status(IN_DRAM_REQ_QUEUE,MR_DRAMQ,gpu_sim_cycle+gpu_tot_sim_cycle);
        }
    }
}

mem_fetch* memory_partition_unit::pop() 
{
   mem_fetch* mf;
   if (m_config->gpgpu_cache_dl2_opt) {
      mf = L2tocbqueue->pop(gpu_sim_cycle);
      if ( mf->isatomic() ) 
         mf->do_atomic();
   } else {
      mf = m_dram->returnq_pop(gpu_sim_cycle);
      if (mf) mf->set_type( REPLY_DATA );
      if (mf->isatomic() ) 
         mf->do_atomic();
   }
   m_request_tracker.erase(mf);
   return mf;
}

mem_fetch* memory_partition_unit::top() 
{
   if (m_config->gpgpu_cache_dl2_opt) {
      return L2tocbqueue->top();
   } else {
       mem_fetch* mf = m_dram->returnq_top();
      if (mf) mf->set_type( REPLY_DATA );
      return mf;
   }
}

void memory_partition_unit::issueCMD() 
{ 
   if (m_config->gpgpu_cache_dl2_opt) {
      // pop completed memory request from dram and push it to dram-to-L2 queue 
      if ( !(dramtoL2queue->full() || dramtoL2writequeue->full()) ) { 
         mem_fetch* mf = m_dram->pop();
         if (mf) {
            if (m_config->gpgpu_l2_readoverwrite && mf->get_is_write() )
               dramtoL2writequeue->push(mf,gpu_sim_cycle);
            else
               dramtoL2queue->push(mf,gpu_sim_cycle);
            mf->set_status(IN_DRAMTOL2QUEUE,MR_DRAM_OUTQ,gpu_sim_cycle+gpu_tot_sim_cycle);
         }
      }
   } else {
      if ( m_dram->returnq_full() ) 
         return;
      mem_fetch* mf = m_dram->pop();
      if (mf) {
         m_dram->returnq_push(mf,gpu_sim_cycle);
         mf->set_status(IN_DRAMRETURN_Q,MR_DRAM_OUTQ,gpu_sim_cycle+gpu_tot_sim_cycle);
      }
   }
   m_dram->issueCMD(); 
   m_dram->dram_log(SAMPLELOG);   
}

void dram_t::dram_log( int task ) 
{
   if (task == SAMPLELOG) {
      StatAddSample(mrqq_Dist, que_length());   
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",id);StatDisp(mrqq_Dist);
   }
}

//Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) 
{
   double smallest = min3(core_time,icnt_time,dram_time);
   int mask = 0x00;
   if (m_memory_config->gpgpu_cache_dl2_opt  //when no-L2 it will never be L2's turn
       && ( l2_time <= smallest) ) {
      smallest = l2_time;
      mask |= L2 ;
      l2_time += l2_period;
   }
   if ( icnt_time <= smallest ) {
      mask |= ICNT;
      icnt_time += icnt_period;
   }
   if ( dram_time <= smallest ) {
      mask |= DRAM;
      dram_time += dram_period;
   }
   if ( core_time <= smallest ) {
      mask |= CORE;
      core_time += core_period;
   }
   return mask;
}

unsigned long long g_single_step=0; // set this in gdb to single step the pipeline

void gpgpu_sim::gpu_sim_loop()
{
   int clock_mask = next_clock_domain();

   // shader core loading (pop from ICNT into shader core) follows CORE clock
   if (clock_mask & CORE ) {
      for (int i=0;i<gpu_n_tpc;i++) 
         fq_pop(i); 
   }
    if (clock_mask & ICNT) {
        // pop from memory controller to interconnect
        for (unsigned i=0;i<m_n_mem;i++) {
            mem_fetch* mf = m_memory_partition_unit[i]->top();
            if (mf) {
                mf->set_status(IN_ICNT2SHADER,MR_2SH_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle);
                unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();
                if ( icnt_has_buffer( mem2device(i), response_size ) ) {
                    if (!mf->get_is_write()) 
                       mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
                    else {
                        freed_L1write_mfs++;
                        gpgpu_n_processed_writes++;
                    }
                    icnt_push( mem2device(i), mf->get_tpc(), mf, response_size );
                    m_memory_partition_unit[i]->pop();
                } else {
                    gpu_stall_icnt2sh++;
                }
            }
        }
    }

   if (clock_mask & DRAM) {
      for (unsigned i=0;i<m_n_mem;i++)  
         m_memory_partition_unit[i]->issueCMD(); // Issue the dram command (scheduler + delay model) 
   }

   // L2 operations follow L2 clock domain
   if (clock_mask & L2) {
      for (unsigned i=0;i<m_n_mem;i++) 
         m_memory_partition_unit[i]->cache_cycle();
   }

   if (clock_mask & ICNT) {
      for (unsigned i=0;i<m_n_mem;i++) {
         if ( m_memory_partition_unit[i]->full() ) {
            gpu_stall_dramfull++;
            continue;
         }
         // move memory request from interconnect into memory partition (if memory controller not backed up)
         mem_fetch* mf = (mem_fetch*) icnt_pop( mem2device(i) );
         m_memory_partition_unit[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
      }
      icnt_transfer();
   }

   if (clock_mask & CORE) {
      // L1 cache + shader core pipeline stages 
      for (unsigned i=0;i<m_n_shader;i++) {
         if (m_sc[i]->get_not_completed() || more_thread) {
            if (!strcmp("GPGPUSIM_ORIG",m_shader_config->pipeline_model) ) 
               m_sc[i]->cycle();
            else if (!strcmp("GT200",m_shader_config->pipeline_model) ) 
               m_sc[i]->cycle_gt200();
         }
      }
      if( g_single_step && ((gpu_sim_cycle+gpu_tot_sim_cycle) >= g_single_step) ) {
          asm("int $03");
      }
      gpu_sim_cycle++;
      if( g_interactive_debugger_enabled ) 
         gpgpu_debug();

      for (unsigned i=0;i<m_n_shader && more_thread;i++) {
         if ( ( (m_sc[i]->get_n_active_cta()+1) <= m_sc[i]->max_cta(m_the_kernel.entry()) ) && g_total_cta_left ) {
            m_sc[i]->issue_block2core( m_the_kernel );
            if (!g_total_cta_left) 
               more_thread = 0;
            assert( g_total_cta_left > -1 );
         }
      }

      // Flush the caches once all of threads are completed.
      if (gpgpu_flush_cache) {
         int all_threads_complete = 1 ; 
         for (unsigned i=0;i<m_n_shader;i++) {
            if (m_sc[i]->get_not_completed() == 0) 
               m_sc[i]->cache_flush();
            else 
               all_threads_complete = 0 ; 
         }
         if (all_threads_complete) {
            printf("Flushed L2 caches...\n");
            if (m_memory_config->gpgpu_cache_dl2_opt) {
               int dlc = 0;
               for (unsigned i=0;i<m_n_mem;i++) {
                  dlc = m_memory_partition_unit[i]->flushL2();
                  assert (dlc == 0); // need to model actual writes to DRAM here
                  printf("Dirty lines flushed from L2 %d is %d\n", i, dlc  );
               }
            }
         }
      }

      if (!(gpu_sim_cycle % gpu_stat_sample_freq)) {
         time_t days, hrs, minutes, sec;
         time_t curr_time;
         time(&curr_time);
         unsigned long long  elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
         days    = elapsed_time/(3600*24);
         hrs     = elapsed_time/3600 - 24*days;
         minutes = elapsed_time/60 - 60*(hrs + 24*days);
         sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
         printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                (double)gpu_sim_insn/(double)gpu_sim_cycle,
                (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                ctime(&curr_time));
         fflush(stdout);
         m_memory_stats->memlatstat_lat_pw(m_n_shader,m_shader_config->n_thread_per_shader,m_shader_config->warp_size);
         visualizer_printstat();
         if (gpgpu_runtime_stat && (gpu_runtime_stat_flag != 0) ) {
            if (gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
               for (unsigned i=0;i<m_n_mem;i++) 
                  m_memory_partition_unit[i]->print_stat(stdout);
               printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
               printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_DWF_MAP) {
               printf("DWF_MS: ");
               for (unsigned i=0;i<m_n_shader;i++) {
                  printf("%u ",acc_dyn_pcs[i]);
               }
               printf("\n");
               print_thread_pc( stdout, m_n_shader );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO) {
               shader_print_runtime_stat( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_WARP_DIS) {
               print_shader_cycle_distro( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_L1MISS) {
               shader_print_l1_miss_stat( stdout );
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_PDOM ) {
               if (m_pdom_sched_type) {
                  printf ("pdom_original_warps_count %d \n",m_shader_stats->n_pdom_sc_orig_stat );
                  printf ("pdom_single_warps_count %d \n",m_shader_stats->n_pdom_sc_single_stat );
               }
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_SCHED ) {
               printf("Average Num. Warps Issuable per Shader:\n");
               for (unsigned i=0;i<m_n_shader;i++) {
                  printf("%2.2f ", (float) m_shader_stats->num_warps_issuable_pershader[i]/ gpu_stat_sample_freq);
                  m_shader_stats->num_warps_issuable_pershader[i] = 0;
               }
               printf("\n");
            }
         }
      }

      for (unsigned i=0;i<m_n_mem;i++) 
         m_memory_stats->acc_mrq_length[i] += m_memory_partition_unit[i]->dram_que_length();
      if (!(gpu_sim_cycle % 20000)) {
         // deadlock detection 
         if (gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
            gpu_deadlock = 1;
         } else {
            last_gpu_sim_insn = gpu_sim_insn;
         }
      }
      try_snap_shot(gpu_sim_cycle);
      spill_log_to_file (stdout, 0, gpu_sim_cycle);
   }
}

void shader_core_ctx::dump_istream_state( FILE *fout )
{
   fprintf(fout, "\n");
   for (unsigned w=0; w < m_config->max_warps_per_shader; w++ ) 
       m_warp[w].print(fout);
}

void gpgpu_sim::dump_pipeline( int mask, int s, int m ) const
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<m_n_shader;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) m_sc[i]->display_pipeline(stdout, 1, mask & 0x2E );
      if (!strcmp("GPGPUSIM_ORIG",m_shader_config->pipeline_model) ) 
         if(mask&0x40) m_sc[i]->dump_istream_state(stdout);
      if(mask&0x100) m_sc[i]->mshr_print(stdout, mask);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<m_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) m_memory_partition_unit[i]->print_stat(stdout);
         if(mask&0x1000000)   m_memory_partition_unit[i]->visualize();
         if(mask&0x10000000)   m_memory_partition_unit[i]->print(stdout);
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

void memory_partition_unit::visualizer_print( gzFile visualizer_file )
{
   m_dram->visualizer_print(visualizer_file);
}
