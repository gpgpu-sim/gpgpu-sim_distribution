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
#include "cflogger.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../intersim/statwraper.h"
#include "../abstract_hardware_model.h"

#include <stdio.h>
#include <string.h>
#define MAX(a,b) (((a)>(b))?(a):(b))

extern unsigned L2_write_miss;
extern unsigned L2_write_hit;
extern unsigned L2_read_hit;
extern unsigned L2_read_miss;
unsigned made_read_mfs = 0;
unsigned made_write_mfs = 0;
unsigned freed_read_mfs = 0;
unsigned freed_L1write_mfs = 0;
unsigned freed_L2write_mfs = 0;
unsigned freed_dummy_read_mfs = 0;
unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_sim_insn = 0;
unsigned long long  gpu_sim_insn_no_ld_const = 0;
unsigned long long  gpu_sim_prev_insn = 0;
unsigned long long  gpu_sim_insn_last_update = 0;
unsigned long long  gpu_tot_sim_cycle = 0;
unsigned long long  gpu_tot_sim_insn = 0;
unsigned long long  gpu_last_sim_cycle = 0;
unsigned long long  gpu_completed_thread = 0;
unsigned long long  gpu_tot_issued_cta = 0;
unsigned long long  gpu_tot_completed_thread = 0;

unsigned int **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]
unsigned int **num_activates; //num_activates[dram chip id][bank id]
unsigned int **row_access; //row_access[dram chip id][bank id]
unsigned int **max_conc_access2samerow; //max_conc_access2samerow[dram chip id][bank id]
unsigned int **max_servicetime2samerow; //max_servicetime2samerow[dram chip id][bank id]
unsigned int mergemiss = 0;
unsigned int L1_read_miss = 0;
unsigned int L1_write_miss = 0;
unsigned int L1_write_hit_on_miss = 0;
unsigned int L1_writeback = 0;
unsigned int L1_texture_miss = 0;
unsigned int L1_const_miss = 0;
unsigned int gpgpu_n_sent_writes = 0;
unsigned int gpgpu_n_processed_writes = 0;
unsigned int *max_return_queue_length;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_shd_mem = 0;
unsigned int gpu_stall_wr_back = 0;
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;
unsigned int gpu_stall_by_MSHRwb = 0;

//shader cannot send to icnt because icnt buffer is full
//Note: it is accumulative for all shaders and is never reset
//so it might increase 8 times in a cycle if we have 8 shaders
unsigned int gpu_stall_sh2icnt = 0;        
// performance counters to account for instruction distribution
extern unsigned int gpgpu_n_load_insn;
extern unsigned int gpgpu_n_store_insn;
extern unsigned int gpgpu_n_shmem_insn;
extern unsigned int gpgpu_n_tex_insn;
extern unsigned int gpgpu_n_const_insn;
extern unsigned int gpgpu_multi_unq_fetches;
char *gpgpu_runtime_stat;
int gpu_stat_sample_freq = 10000;
int gpu_runtime_stat_flag = 0;
extern int gpgpu_warpdistro_shader;

// GPGPU options
unsigned long long  gpu_max_cycle = 0;
unsigned long long  gpu_max_insn = 0;
int gpu_max_cycle_opt = 0;
int gpu_max_insn_opt = 0;
int gpu_max_cta_opt = 0;
int gpu_deadlock_detect = 0;
int gpu_deadlock = 0;
static unsigned long long  last_gpu_sim_insn = 0;
int gpgpu_dram_scheduler = DRAM_FIFO;
int g_save_embedded_ptx = 0;
int gpgpu_simd_model = 0;
int gpgpu_no_dl1 = 0;
char *gpgpu_cache_texl1_opt;
char *gpgpu_cache_constl1_opt;
char *gpgpu_cache_dl1_opt;
char *gpgpu_cache_dl2_opt;
extern int gpgpu_l2_readoverwrite;
int gpgpu_partial_write_mask = 0;

int gpgpu_perfect_mem = FALSE;
char *gpgpu_shader_core_pipeline_opt;
extern unsigned int *requests_by_warp;
unsigned int gpgpu_dram_buswidth = 4;
unsigned int gpgpu_dram_burst_length = 4;
int gpgpu_dram_sched_queue_size = 0; 
char * gpgpu_dram_timing_opt;
int gpgpu_flush_cache = 0;
int gpgpu_mem_address_mask = 0;
unsigned int recent_dram_util = 0;

int gpgpu_cflog_interval = 0;

unsigned int finished_trace = 0;

unsigned g_next_request_uid = 1;

extern struct regs_t regs;

extern long int gpu_reads;

void ptx_dump_regs( void *thd );

int g_nthreads_issued;
int g_total_cta_left;


unsigned ptx_kernel_program_size();
void visualizer_printstat();
void time_vector_create(int ld_size,int st_size);
void time_vector_print(void);
void time_vector_update(unsigned int uid,int slot ,long int cycle,int type);
void check_time_vector_update(unsigned int uid,int slot ,long int latency,int type);
void node_req_hist_clear(void *p);
void node_req_hist_dump(void *p);
void node_req_hist_update(void * p,int node, long long cycle);

/* functionally simulated memory */
extern struct mem_t *mem;

/* Defining Clock Domains
basically just the ratio is important */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  

double core_time=0;
double icnt_time=0;
double dram_time=0;
double l2_time=0;

#define MhZ *1000000
double core_freq=2 MhZ;
double icnt_freq=2 MhZ;
double dram_freq=2 MhZ;
double l2_freq=2 MhZ;

double core_period  = 1 /( 2 MhZ);
double icnt_period   = 1 /( 2 MhZ);
double dram_period = 1 /( 2 MhZ);
double l2_period = 1 / (2 MhZ);

char * gpgpu_clock_domains;

/* GPU uArch parameters */
unsigned int gpu_n_mem = 8;
unsigned int gpu_mem_n_bk = 4;
unsigned int gpu_n_mem_per_ctrlr = 1;
unsigned int gpu_n_shader = 8;
int gpu_concentration = 1;
int gpu_n_tpc = 8;
unsigned int gpu_n_mshr_per_shader;
unsigned int gpu_n_thread_per_shader = 128;
unsigned int gpu_n_warp_per_shader;
unsigned int gpu_n_mshr_per_thread = 1;

extern int gpgpu_interwarp_mshr_merge ;

extern unsigned int gpgpu_shmem_size;
extern unsigned int gpgpu_shader_registers;
extern unsigned int gpgpu_shader_cta;
extern int gpgpu_shmem_bkconflict;
extern int gpgpu_cache_bkconflict;
extern int gpgpu_n_cache_bank;
extern unsigned int warp_size; 
extern int pipe_simd_width;
extern unsigned int gpgpu_dwf_heuristic;
extern unsigned int gpgpu_dwf_regbk;
int gpgpu_reg_bankconflict = FALSE;
extern int gpgpu_shmem_port_per_bank;
extern int gpgpu_cache_port_per_bank;
extern int gpgpu_const_port_per_bank;
extern int gpgpu_shmem_pipe_speedup;  
extern int gpgpu_reg_bank_conflict_model;
extern int gpgpu_num_reg_banks;

extern unsigned int gpu_max_cta_per_shader;
extern unsigned int gpu_padded_cta_size;
extern int gpgpu_local_mem_map;

unsigned int gpgpu_pre_mem_stages = 0;
unsigned int gpgpu_no_divg_load = 0;
char *gpgpu_dwf_hw_opt;
unsigned int gpgpu_thread_swizzling = 0;
unsigned int gpgpu_strict_simd_wrbk = 0;

int pdom_sched_type = 0;
int n_pdom_sc_orig_stat = 0; //the selected pdom schedular is used 
int n_pdom_sc_single_stat = 0; //only a single warp is ready to go in that cycle.  
int *num_warps_issuable;
int *num_warps_issuable_pershader;

// Thread Dispatching Unit option 
int gpgpu_cuda_sim = 1;
int gpgpu_spread_blocks_across_cores = 1;

/* GPU uArch structures */
shader_core_ctx_t **sc;
dram_t **dram;
unsigned int common_clock = 0;
unsigned int more_thread = 1;
extern unsigned int n_regconflict_stall;
unsigned int warp_conflict_at_writeback = 0;
unsigned int gpgpu_commit_pc_beyond_two = 0;
extern int g_network_mode;
int gpgpu_cache_wt_through = 0;


//memory access classification
int gpgpu_n_mem_read_local = 0;
int gpgpu_n_mem_write_local = 0;
int gpgpu_n_mem_texture = 0;
int gpgpu_n_mem_const = 0;
int gpgpu_n_mem_read_global = 0;
int gpgpu_n_mem_write_global = 0;

#define MEM_LATENCY_STAT_IMPL
#include "mem_latency_stat.h"

unsigned char fq_has_buffer(unsigned long long int addr, int bsize, bool write, int sid );
unsigned char fq_push(unsigned long long int addr, int bsize, unsigned char write, partial_write_mask_t partial_write_mask, 
                      int sid, int wid, mshr_entry* mshr, int cache_hits_waiting,
                      enum mem_access_type mem_acc, address_type pc);
int issue_mf_from_fq(mem_fetch_t *mf);
unsigned char single_check_icnt_has_buffer(int chip, int sid, unsigned char is_write );
unsigned char fq_pop(int tpc_id);
void fill_shd_L1_with_new_line(shader_core_ctx_t * sc, mem_fetch_t * mf);

void set_option_gpgpu_spread_blocks_across_cores(int option);
void set_param_gpgpu_num_shaders(int num_shaders);
unsigned ptx_sim_grid_size();
void icnt_init_grid();
void interconnect_stats();
void icnt_overal_stat();
unsigned ptx_sim_cta_size();
unsigned ptx_sim_init_thread( void** thread_info, int sid, unsigned tid,unsigned threads_left,unsigned num_threads, core_t *core, unsigned hw_cta_id, unsigned hw_warp_id );

void gpu_sim_loop( int grid_num );

void print_shader_cycle_distro( FILE *fout ) ;
void find_reconvergence_points();
void dwf_process_reconv_pts();

extern int gpgpu_ptx_instruction_classification ;
extern int g_ptx_sim_mode;

extern int gpgpu_coalesce_arch;

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333
void L2c_log(int task);
void dram_log(int task);

void visualizer_options(option_parser_t opp);
void gpu_reg_options(option_parser_t opp)
{
   option_parser_register(opp, "-save_embedded_ptx", OPT_BOOL, &g_save_embedded_ptx, 
                "saves ptx files embedded in binary as <n>.ptx",
                "0");
   option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &gpgpu_simd_model, 
               "0 = no recombination, 1 = post-dominator, 2 = MIMD, 3 = dynamic warp formation", "0");
   option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &gpgpu_dram_scheduler, 
               "0 = fifo (default), 1 = fast ideal", "0");

   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");

   option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &gpgpu_cache_texl1_opt, 
                  "per-shader L1 texture cache  (READ-ONLY) config, i.e., {<nsets>:<linesize>:<assoc>:<repl>|none}",
                  "512:64:2:L");

   option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &gpgpu_cache_constl1_opt, 
                  "per-shader L1 constant memory cache  (READ-ONLY) config, i.e., {<nsets>:<linesize>:<assoc>:<repl>|none}",
                  "64:64:2:L");

   option_parser_register(opp, "-gpgpu_no_dl1", OPT_BOOL, &gpgpu_no_dl1, 
                "no dl1 cache (voids -gpgpu_cache:dl1 option)",
                "0");

   option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &gpgpu_cache_dl1_opt, 
                  "shader L1 data cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}",
                  "256:128:1:L");

   option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &gpgpu_cache_dl2_opt, 
                  "unified banked L2 data cache config, i.e., {<nsets>:<bsize>:<assoc>:<repl>|none}; disabled by default",
                  NULL); 

   option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem, 
                "enable perfect memory mode (no cache miss)",
                "0");

   option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                  "shader core pipeline config, i.e., {<nthread>:<warpsize>:<pipe_simd_width>}",
                  "256:32:32");

   option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers, 
                "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                "8192");

   option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &gpgpu_shader_cta, 
                "Maximum number of concurrent CTAs in shader (default 8)",
                "8");

   option_parser_register(opp, "-gpgpu_n_shader", OPT_UINT32, &gpu_n_shader, 
                "number of shaders in gpu",
                "8");
   option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &gpu_n_mem, 
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

   option_parser_register(opp, "-gpgpu_reg_bankconflict", OPT_BOOL, &gpgpu_reg_bankconflict, 
                "Check for bank conflict in the pipeline",
                "0");

   option_parser_register(opp, "-gpgpu_dwf_regbk", OPT_BOOL, (int*)&gpgpu_dwf_regbk, 
                "Have dwf scheduler to avoid bank conflict",
                "1");

   option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat, 
               "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
               "0");

   option_parser_register(opp, "-gpu_n_mshr_per_shader", OPT_UINT32, &gpu_n_mshr_per_shader, 
                "Number of MSHRs per shader",
                "64");

   option_parser_register(opp, "-gpgpu_interwarp_mshr_merge", OPT_INT32, &gpgpu_interwarp_mshr_merge, 
               "interwarp coalescing",
               "0");

   option_parser_register(opp, "-gpgpu_dram_sched_queue_size", OPT_INT32, &gpgpu_dram_sched_queue_size, 
               "0 = unlimited (default); # entries per chip",
               "0");

   option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &gpgpu_dram_buswidth, 
                "default = 4 bytes (8 bytes per cycle at DDR)",
                "4");

   option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &gpgpu_dram_burst_length, 
                "Burst length of each DRAM request (default = 4 DDR cycle)",
                "4");

   option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt, 
               "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tWTR}",
               "4:2:8:12:21:13:34:9:4:5");


   option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32, &gpgpu_mem_address_mask, 
               "0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits",
               "0");

   option_parser_register(opp, "-gpgpu_flush_cache", OPT_BOOL, &gpgpu_flush_cache, 
                "Flush cache at the end of each kernel call",
                "0");

   option_parser_register(opp, "-gpgpu_pre_mem_stages", OPT_UINT32, &gpgpu_pre_mem_stages, 
                "default = 0 pre-memory pipeline stages",
                "0");

   option_parser_register(opp, "-gpgpu_no_divg_load", OPT_BOOL, (int*)&gpgpu_no_divg_load, 
                "Don't allow divergence on load",
                "0");

   option_parser_register(opp, "-gpgpu_dwf_hw", OPT_CSTR, &gpgpu_dwf_hw_opt, 
                  "dynamic warp formation hw config, i.e., {<#LUT_entries>:<associativity>|none}",
                  "32:2");

   option_parser_register(opp, "-gpgpu_thread_swizzling", OPT_BOOL, (int*)&gpgpu_thread_swizzling, 
                "Thread Swizzling (1=on, 0=off)",
                "0");

   option_parser_register(opp, "-gpgpu_strict_simd_wrbk", OPT_BOOL, (int*)&gpgpu_strict_simd_wrbk, 
                "Applying Strick SIMD WriteBack Stage (1=on, 0=off)",
                "0");

   option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size, 
                "Size of shared memory per shader core (default 16kB)",
                "16384");

   option_parser_register(opp, "-gpgpu_shmem_bkconflict", OPT_BOOL, &gpgpu_shmem_bkconflict,  
                "Turn on bank conflict check for shared memory",
                "0");

   option_parser_register(opp, "-gpgpu_shmem_pipe_speedup", OPT_INT32, &gpgpu_shmem_pipe_speedup,  
                "Number of groups each warp is divided for shared memory bank conflict check",
                "2");

   option_parser_register(opp, "-gpgpu_cache_wt_through", OPT_BOOL, &gpgpu_cache_wt_through, 
                "L1 cache become write through (1=on, 0=off)", 
                "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");

   option_parser_register(opp, "-gpgpu_cache_bkconflict", OPT_BOOL, &gpgpu_cache_bkconflict, 
                "Turn on bank conflict check for L1 cache access", 
                "0");

   option_parser_register(opp, "-gpgpu_n_cache_bank", OPT_INT32, &gpgpu_n_cache_bank, 
               "Number of banks in L1 cache, also for memory coalescing stall", 
               "1");

   option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader, 
               "Specify which shader core to collect the warp size distribution from", 
               "-1");


   option_parser_register(opp, "-gpgpu_pdom_sched_type", OPT_INT32, &pdom_sched_type, 
               "0 = first ready warp found, 1 = random, 8 = loose round robin", 
               "8");

   option_parser_register(opp, "-gpgpu_spread_blocks_across_cores", OPT_BOOL, 
                &gpgpu_spread_blocks_across_cores, 
                "Spread block-issuing across all cores instead of filling up core by core (do NOT disable)", 
                "1");

   option_parser_register(opp, "-gpgpu_cuda_sim", OPT_BOOL, &gpgpu_cuda_sim, 
                "use PTX instruction set", 
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

   option_parser_register(opp, "-gpgpu_shmem_port_per_bank", OPT_INT32, &gpgpu_shmem_port_per_bank, 
               "Number of access processed by a shared memory bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_cache_port_per_bank", OPT_INT32, &gpgpu_cache_port_per_bank, 
               "Number of access processed by a cache bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_const_port_per_bank", OPT_INT32, &gpgpu_const_port_per_bank,
               "Number of access processed by a constant cache bank per cycle (default = 2)", 
               "2");
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-gpgpu_partial_write_mask", OPT_INT32, &gpgpu_partial_write_mask, 
               "use partial write mask to filter memory requests <1>No extra reads(use this!)<2>extra reads generated for partial chunks", 
               "0");
   option_parser_register(opp, "-gpu_concentration", OPT_INT32, &gpu_concentration, 
               "Number of shader cores per interconnection port (default = 1)", 
               "1");
   option_parser_register(opp, "-gpgpu_local_mem_map", OPT_INT32, &gpgpu_local_mem_map, 
               "Mapping from local memory space address to simulated GPU physical address space (default = 1)", 
               "1");
   option_parser_register(opp, "-gpgpu_reg_bank_conflict_model", OPT_BOOL, &gpgpu_reg_bank_conflict_model, 
               "Turn on register bank conflict model (default = off)", 
               "0");
   option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &gpgpu_num_reg_banks, 
               "Number of register banks (default = 8)", 
               "8");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch, 
                           "Coalescing arch (default = 13, anything else is off for now)", 
                           "13");
   addrdec_setoption(opp);
   L2c_options(opp);
   visualizer_options(opp);
   ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

inline int mem2device(int memid) {
   return memid + gpu_n_tpc;
}

/////////////////////////////////////////////////////////////////////////////


/* Allocate memory for uArch structures */
void init_gpu () 
{ 
   int i;

   gpu_max_cycle = gpu_max_cycle_opt;
   gpu_max_insn = gpu_max_insn_opt;

   i = sscanf(gpgpu_shader_core_pipeline_opt,"%d:%d:%d", 
              &gpu_n_thread_per_shader, &warp_size, &pipe_simd_width);
   gpu_n_warp_per_shader = gpu_n_thread_per_shader / warp_size;
   num_warps_issuable = (int*) calloc(gpu_n_warp_per_shader+1, sizeof(int));
   num_warps_issuable_pershader = (int*) calloc(gpu_n_shader, sizeof(int));
   if (i == 2) {
      pipe_simd_width = warp_size;
   } else if (i == 3) {
      assert(warp_size % pipe_simd_width == 0);
   }

   sscanf(gpgpu_runtime_stat, "%d:%x",
          &gpu_stat_sample_freq, &gpu_runtime_stat_flag);

   sc = (shader_core_ctx_t**) calloc(gpu_n_shader, sizeof(shader_core_ctx_t*));
   int mshr_que = gpu_n_mshr_per_thread;
   for (i=0;(unsigned)i<gpu_n_shader;i++) {
      sc[i] = shader_create("sh", i, /* shader id*/
                            gpu_n_thread_per_shader, /* number of threads */
                            mshr_que, /* number of MSHR per threads */
                            fq_push, fq_has_buffer, gpgpu_simd_model);
   }

   ptx_file_line_stats_create_exposed_latency_tracker(gpu_n_shader);

   // initialize dynamic warp formation scheduler
   int dwf_lut_size, dwf_lut_assoc;
   sscanf(gpgpu_dwf_hw_opt,"%d:%d", &dwf_lut_size, &dwf_lut_assoc);
   char *dwf_hw_policy_opt = strchr(gpgpu_dwf_hw_opt, ';');
   int insn_size = 1; // for cuda-sim
   create_dwf_schedulers(gpu_n_shader, dwf_lut_size, dwf_lut_assoc, 
                         warp_size, pipe_simd_width, 
                         gpu_n_thread_per_shader, insn_size, 
                         gpgpu_dwf_heuristic, dwf_hw_policy_opt );

   gpgpu_no_divg_load = gpgpu_no_divg_load && (gpgpu_simd_model == DWF);
   // always use no diverge on load for PDOM and NAIVE
   gpgpu_no_divg_load = gpgpu_no_divg_load || (gpgpu_simd_model == POST_DOMINATOR || gpgpu_simd_model == NO_RECONVERGE);
   if (gpgpu_no_divg_load)
      init_warp_tracker();

   assert(gpu_n_shader % gpu_concentration == 0);
   gpu_n_tpc = gpu_n_shader / gpu_concentration;

   dram = (dram_t**) calloc(gpu_n_mem, sizeof(dram_t*));
   // L2request = (mem_fetch_t**) calloc(gpu_n_mem, sizeof(mem_fetch_t*));
   addrdec_setnchip(gpu_n_mem);
   unsigned int nbk,tCCD,tRRD,tRCD,tRAS,tRP,tRC,CL,WL,tWTR;
   sscanf(gpgpu_dram_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",&nbk,&tCCD,&tRRD,&tRCD,&tRAS,&tRP,&tRC,&CL,&WL,&tWTR);
   gpu_mem_n_bk = nbk;
   for (i=0;(unsigned)i<gpu_n_mem;i++) {
      dram[i] = dram_create(i, nbk, tCCD, tRRD, tRCD, tRAS, tRP, tRC, 
                            CL, WL, gpgpu_dram_burst_length/*BL*/, tWTR, gpgpu_dram_buswidth/*busW*/, 
                            gpgpu_dram_sched_queue_size, gpgpu_dram_scheduler);
      if (gpgpu_cache_dl2_opt)
         L2c_create(dram[i], gpgpu_cache_dl2_opt);
   }
   dram_log(CREATELOG);
   if (gpgpu_cache_dl2_opt && 1) {
      L2c_log(CREATELOG);
   }
   concurrent_row_access = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   num_activates = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   row_access = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   max_conc_access2samerow = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   max_servicetime2samerow = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));

   for (i=0;(unsigned)i<gpu_n_mem ;i++ ) {
      concurrent_row_access[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      row_access[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      num_activates[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      max_conc_access2samerow[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      max_servicetime2samerow[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
   }

   memlatstat_init();

   L2c_init_stat();
   max_return_queue_length = (unsigned int*) calloc(gpu_n_shader, sizeof(unsigned int));
   icnt_init(gpu_n_tpc, gpu_n_mem);

   common_clock = 0;

   time_vector_create(NUM_MEM_REQ_STAT,MR_2SH_ICNT_INJECTED);
}



void gpu_print_stat();

void init_clock_domains(void ) {
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

void reinit_clock_domains(void)
{
   core_time = 0 ;
   dram_time = 0 ;
   icnt_time = 0;
   l2_time = 0;
}

void init_once(void ) {
   init_clock_domains();
}

// return the number of cycle required to run all the trace on the gpu 
unsigned int run_gpu_sim(int grid_num) 
{

   int not_completed;
   int mem_busy;
   int icnt2mem_busy;

   gpu_sim_cycle = 0;
   not_completed = 1;
   mem_busy = 1;
   icnt2mem_busy = 1;
   finished_trace = 0;
   g_next_request_uid = 1;
   more_thread = 1;
   gpu_sim_insn = 0;
   gpu_sim_insn_no_ld_const = 0;

   gpu_completed_thread = 0;

   g_nthreads_issued = 0;

   static int one_time_inits_done = 0 ;
   if (!one_time_inits_done ) {
      init_once();
   }
   reinit_clock_domains();
   assert(gpgpu_spread_blocks_across_cores); // this seems to be required, so let's make it explicit
   set_option_gpgpu_spread_blocks_across_cores(gpgpu_spread_blocks_across_cores);
   set_param_gpgpu_num_shaders(gpu_n_shader);
   for (unsigned i=0;i<gpu_n_shader;i++) {
      sc[i]->not_completed = 0;
      shader_reinit(sc[i],0,sc[i]->n_threads);
   }
   if (gpu_max_cta_opt != 0) {
      g_total_cta_left = gpu_max_cta_opt;
   } else {
      g_total_cta_left =  ptx_sim_grid_size();
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
   find_reconvergence_points();

   dwf_process_reconv_pts();
   dwf_reinit_schedulers(gpu_n_shader);

   // initialize the control-flow, memory access, memory latency logger
   create_thread_CFlogger( gpu_n_shader, gpu_n_thread_per_shader, ptx_kernel_program_size(), 0, gpgpu_cflog_interval );
   shader_CTA_count_create( gpu_n_shader, gpgpu_cflog_interval);
   if (gpgpu_cflog_interval != 0) {
      insn_warp_occ_create( gpu_n_shader, warp_size, ptx_kernel_program_size() );
      shader_warp_occ_create( gpu_n_shader, warp_size, gpgpu_cflog_interval);
      shader_mem_acc_create( gpu_n_shader, gpu_n_mem, 4, gpgpu_cflog_interval);
      shader_mem_lat_create( gpu_n_shader, gpgpu_cflog_interval);
      shader_cache_access_create( gpu_n_shader, 3, gpgpu_cflog_interval);
      set_spill_interval (gpgpu_cflog_interval * 40);
   }

   // calcaulte the max cta count and cta size for local memory address mapping
   gpu_max_cta_per_shader = max_cta_per_shader(sc[0]);
   //gpu_max_cta_per_shader is limited by number of CTAs if not enough    
   if (ptx_sim_grid_size() < gpu_max_cta_per_shader*gpu_n_shader) { 
      gpu_max_cta_per_shader = (ptx_sim_grid_size() / gpu_n_shader);
      if (ptx_sim_grid_size() % gpu_n_shader)
         gpu_max_cta_per_shader++;
   }
   unsigned int gpu_cta_size = ptx_sim_cta_size();
   gpu_padded_cta_size = (gpu_cta_size%32) ? 32*((gpu_cta_size/32)+1) : gpu_cta_size;

   if (g_network_mode) {
      icnt_init_grid(); 
   }
   last_gpu_sim_insn = 0;
   // add this condition as well? (gpgpu_n_processed_writes < gpgpu_n_sent_writes)
   while (not_completed || mem_busy || icnt2mem_busy) {
      gpu_sim_loop(grid_num);

      not_completed = 0;
      for (unsigned i=0;i<gpu_n_shader;i++) {
         not_completed += sc[i]->not_completed;
      }
      // dram_busy just check the request queue length into the dram 
      // to make sure all the memory requests (esp the writes) are done
      mem_busy = 0; 
      for (unsigned i=0;i<gpu_n_mem;i++) {
         mem_busy += dram_busy(dram[i]);
      }
     // icnt to the memory should clean of any pending tranfers as well
      icnt2mem_busy = icnt_busy( );

      if (gpu_max_cycle && (gpu_tot_sim_cycle + gpu_sim_cycle) >= gpu_max_cycle) {
         break;
      }
      if (gpu_max_insn && (gpu_tot_sim_insn + gpu_sim_insn) >= gpu_max_insn) {
         break;
      }
      if (gpu_deadlock_detect && gpu_deadlock) {
         break;
      }

   }
   memlatstat_lat_pw();
   gpu_tot_sim_cycle += gpu_sim_cycle;
   gpu_tot_sim_insn += gpu_sim_insn;
   gpu_tot_completed_thread += gpu_completed_thread;
   
   ptx_file_line_stats_write_file();

   printf("stats for grid: %d\n", grid_num);
   gpu_print_stat();
   if (g_network_mode) {
      interconnect_stats();
      printf("----------------------------Interconnect-DETAILS---------------------------------" );
      icnt_overal_stat();
      printf("----------------------------END-of-Interconnect-DETAILS-------------------------" );
   }
   if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_QUEUELOGS ) {
      dramqueue_latency_log_dump();
      dram_log(DUMPLOG);
      if (gpgpu_cache_dl2_opt) {
         L2c_log(DUMPLOG);
         L2c_latency_log_dump();
      }
   }

#define DEADLOCK 0
   if (gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("ERROR ** deadlock detected: last writeback @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      fflush(stdout);
      assert(DEADLOCK);
   }
   return gpu_sim_cycle;
}

extern void ** g_inst_classification_stat;
extern void ** g_inst_op_classification_stat;
extern int g_ptx_kernel_count; // used for classification stat collection purposes 

extern unsigned get_max_mshr_used(shader_core_ctx_t* shader);

void gpu_print_stat()
{  
   unsigned i;
   int j,k;
    
   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_sim_no_ld_const_insn = %lld\n", gpu_sim_insn_no_ld_const);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_completed_thread = %lld\n", gpu_completed_thread);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)gpu_tot_sim_insn / gpu_tot_sim_cycle);
   printf("gpu_tot_completed_thread = %lld\n", gpu_tot_completed_thread);
   printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);
   printf("gpgpu_n_sent_writes = %d\n", gpgpu_n_sent_writes);
   printf("gpgpu_n_processed_writes = %d\n", gpgpu_n_processed_writes);

   // performance counter for stalls due to congestion.
   printf("gpu_stall_by_MSHRwb= %d\n", gpu_stall_by_MSHRwb);
   printf("gpu_stall_shd_mem  = %d\n", gpu_stall_shd_mem );
   printf("gpu_stall_wr_back  = %d\n", gpu_stall_wr_back );
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh   );
   printf("gpu_stall_sh2icnt    = %d\n", gpu_stall_sh2icnt   );
   // performance counter that are not local to one shader
   shader_print_accstats(stdout);

   memlatstat_print();
   printf("max return queue length = ");
   for (unsigned i=0;i<gpu_n_shader;i++) {
      printf("%d ", max_return_queue_length[i]);
   }
   printf("\n");
   // merge misses
   printf("merge misses = %d\n", mergemiss);
   printf("L1 read misses = %d\n", L1_read_miss);
   printf("L1 write misses = %d\n", L1_write_miss);
   printf("L1 write hit on misses = %d\n", L1_write_hit_on_miss);
   printf("L1 writebacks = %d\n", L1_writeback);
   printf("L1 texture misses = %d\n", L1_texture_miss);
   printf("L1 const misses = %d\n", L1_const_miss);
   printf("L2_write_miss = %d\n", L2_write_miss);
   printf("L2_write_hit = %d\n", L2_write_hit);
   printf("L2_read_miss = %d\n", L2_read_miss);
   printf("L2_read_hit = %d\n", L2_read_hit);
   printf("made_read_mfs = %d\n", made_read_mfs);
   printf("made_write_mfs = %d\n", made_write_mfs);
   printf("freed_read_mfs = %d\n", freed_read_mfs);
   printf("freed_L1write_mfs = %d\n", freed_L1write_mfs);
   printf("freed_L2write_mfs = %d\n", freed_L2write_mfs);
   printf("freed_dummy_read_mfs = %d\n", freed_dummy_read_mfs);

   printf("gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
   printf("gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
   printf("gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
   printf("gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
   printf("gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
   printf("gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

   printf("max_n_mshr_used = ");
   for (unsigned i=0; i< gpu_n_shader; i++) printf("%d ", get_max_mshr_used(sc[i]));
   printf("\n");

   if (gpgpu_cache_dl2_opt) {
      L2c_print_stat( );
   }
   for (unsigned i=0;i<gpu_n_mem;i++) {
      dram_print(dram[i],stdout);
   }

   for (i=0, j=0, k=0;i<gpu_n_shader;i++) {
      shd_cache_print(sc[i]->L1cache,stdout);
      j+=sc[i]->L1cache->miss;
      k+=sc[i]->L1cache->access;
   }
   printf("L1 Data Cache Total Miss Rate = %0.3f\n", (float)j/k);

   for (i=0,j=0,k=0;i<gpu_n_shader;i++) {
      shd_cache_print(sc[i]->L1texcache,stdout);
      j+=sc[i]->L1texcache->miss;
      k+=sc[i]->L1texcache->access;
   }
   printf("L1 Texture Cache Total Miss Rate = %0.3f\n", (float)j/k);

   for (i=0,j=0,k=0;i<gpu_n_shader;i++) {
      shd_cache_print(sc[i]->L1constcache,stdout);
      j+=sc[i]->L1constcache->miss;
      k+=sc[i]->L1constcache->access;
   }
   printf("L1 Const Cache Total Miss Rate = %0.3f\n", (float)j/k);

   if (gpgpu_cache_dl2_opt) {
      L2c_print_cache_stat();
   }
   printf("n_regconflict_stall = %d\n", n_regconflict_stall);

   if (gpgpu_simd_model == DWF) {
      dwf_print_stat(stdout);
   }

   if (gpgpu_simd_model == POST_DOMINATOR) {
      printf("num_warps_issuable:");
      for (unsigned i=0;i<(gpu_n_warp_per_shader+1);i++) {
         printf("%d ", num_warps_issuable[i]);
      }
      printf("\n");
   }
   if (gpgpu_strict_simd_wrbk) {
      printf("warp_conflict_at_writeback = %d\n", warp_conflict_at_writeback);
   }

   printf("gpgpu_commit_pc_beyond_two = %d\n", gpgpu_commit_pc_beyond_two);

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

////////////////////////////////////////////////////////////////////////////////////
// Wrapper function for shader cores' memory system: 
////////////////////////////////////////////////////////////////////////////////////

// a hack to make the size of a packet discrete multiples of the interconnect's flit_size.
static inline
unsigned int fill_to_next_flit(unsigned int size) 
{
   assert (g_network_mode == INTERSIM);
   return size;
}



unsigned char check_icnt_has_buffer(unsigned long long int *addr, int *bsize, 
                                    int n_addr, int sid )
{
   addrdec_t tlx;
   static unsigned int *req_buffer = NULL;
   //the req_buf size can be equal to gpu_n_mem ; gpu_n_shader is added to make it compatible
   //with the case where a mem controller is sending to shd 
   if (!req_buffer) req_buffer = (unsigned int*)malloc((gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));
   memset(req_buffer, 0, (gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));

   // aggregate all buffer requirement of all memory accesses by dram chips
   for (int i=0; i< n_addr; i++) {
      addrdec_tlx(addr[i],&tlx);
      req_buffer[tlx.chip] += fill_to_next_flit(bsize[i]);
   }

   int tpc_id = sid / gpu_concentration;

   return icnt_has_buffer(tpc_id, req_buffer);
}

unsigned char single_check_icnt_has_buffer(int chip, int sid, unsigned char is_write )
{
   static unsigned int *req_buffer = NULL;
   //the req_buf size can be equal to gpu_n_mem ; gpu_n_shader is added to make it compatible
   //with the case where a mem controller is sending to shd 
   if (!req_buffer) req_buffer = (unsigned int*)malloc((gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));
   memset(req_buffer, 0, (gpu_n_mem+gpu_n_tpc)*sizeof(unsigned int));

   // aggregate all buffer requirement of all memory accesses by dram chips

   int b_size;
   if (is_write)
      b_size = sc[sid]->L1cache->line_sz;
   else
      b_size = READ_PACKET_SIZE;
   req_buffer[chip] += fill_to_next_flit(b_size);

   int tpc_id = sid / gpu_concentration;

   return icnt_has_buffer(tpc_id, req_buffer);
}

int max_n_addr = 0;

// Check the memory system for buffer availability
unsigned char fq_has_buffer(unsigned long long int addr, int bsize, bool write, int sid )
{
   //requests should be single always now
   int rsize = bsize;
   //maintain similar functionality with fq_push, if its a read, bsize is the load size, not the request's size
   if (!write) {
       rsize = READ_PACKET_SIZE;
   }
   return check_icnt_has_buffer(&addr, &rsize, 1, sid);
}

// Takes in memory address and their parameters and pushes to the fetch queue 
unsigned char fq_push(unsigned long long int addr, int bsize, unsigned char write, partial_write_mask_t partial_write_mask, 
                      int sid, int wid, mshr_entry* mshr, int cache_hits_waiting,
                      enum mem_access_type mem_acc, address_type pc) 
{
   mem_fetch_t *mf;

   mf = (mem_fetch_t*) calloc(1,sizeof(mem_fetch_t));
   mf->request_uid = g_next_request_uid++;
   mf->addr = addr;
   mf->nbytes_L1 = bsize;
   mf->sid = sid;
   mf->source_node = sid / gpu_concentration;
   mf->wid = wid;
   mf->cache_hits_waiting = cache_hits_waiting;
   mf->txbytes_L1 = 0;
   mf->rxbytes_L1 = 0;  
   mf->mshr = mshr;
   if (mshr) mshr->mf = (void*)mf; // for debugging
   mf->write = write;

   if (write)
      made_write_mfs++;
   else
      made_read_mfs++;
   memlatstat_start(mf);
   addrdec_tlx(addr,&mf->tlx);
   mf->bank = mf->tlx.bk;
   mf->chip = mf->tlx.chip;
   if (gpgpu_cache_dl2_opt)
      mf->nbytes_L2 = L2c_get_linesize( dram[mf->tlx.chip] );
   else
      mf->nbytes_L2 = 0;
   mf->txbytes_L2 = 0;
   mf->rxbytes_L2 = 0;  

   mf->write_mask = partial_write_mask;
   if (!write) assert(partial_write_mask == NO_PARTIAL_WRITE);

   // stat collection codes 
   mf->mem_acc = mem_acc;
   mf->pc = pc;

   switch (mem_acc) {
   case CONST_ACC_R: gpgpu_n_mem_const++; break;
   case TEXTURE_ACC_R: gpgpu_n_mem_texture++; break;
   case GLOBAL_ACC_R: gpgpu_n_mem_read_global++; break;
   case GLOBAL_ACC_W: gpgpu_n_mem_write_global++; break;
   case LOCAL_ACC_R: gpgpu_n_mem_read_local++; break;
   case LOCAL_ACC_W: gpgpu_n_mem_write_local++; break;
   default: assert(0);
   }

   return(issue_mf_from_fq(mf));

}

int issue_mf_from_fq(mem_fetch_t *mf){
   int destination; // where is the next level of memory?
   destination = mf->tlx.chip;
   int tpc_id = mf->sid / gpu_concentration;

   if (mf->mshr) mshr_update_status(mf->mshr,IN_ICNT2MEM);
   if (!mf->write) {
      mf->type = RD_REQ;
      assert( mf->timestamp == (gpu_sim_cycle+gpu_tot_sim_cycle) );
      time_vector_update(mf->mshr->insts[0].uid, MR_ICNT_PUSHED, gpu_sim_cycle+gpu_tot_sim_cycle, mf->type );
      icnt_push(tpc_id, mem2device(destination), (void*)mf, READ_PACKET_SIZE);
   } else {
      mf->type = WT_REQ;
      icnt_push(tpc_id, mem2device(destination), (void*)mf, mf->nbytes_L1);
      gpgpu_n_sent_writes++;
      assert( mf->timestamp == (gpu_sim_cycle+gpu_tot_sim_cycle) );
      time_vector_update(mf->request_uid, MR_ICNT_PUSHED, gpu_sim_cycle+gpu_tot_sim_cycle, mf->type ) ;
   }

   return 0;
}

extern void mshr_return_from_mem(shader_core_ctx_t * shader, mshr_entry_t* mshr);

inline void fill_shd_L1_with_new_line(shader_core_ctx_t * sc, mem_fetch_t * mf) {
   unsigned long long int repl_addr = -1;
   // When the data arrives, it flags all the appropriate MSHR
   // entries accordingly (by checking the address in each entry ) 
   memlatstat_read_done(mf);

   mshr_return_from_mem(sc, mf->mshr);

   if (mf->mshr->istexture) {
       shd_cache_fill(sc->L1texcache,mf->addr,sc->gpu_cycle);
       repl_addr = -1;
   } else if (mf->mshr->isconst) {
       shd_cache_fill(sc->L1constcache,mf->addr,sc->gpu_cycle);
       repl_addr = -1;
   } else {
       if (!gpgpu_no_dl1) {
           //if we are doing a writeback cache we may have marked off a mask in the mshr
           //only write into the cache unmasked bytes.
           //since this doesn't affect timing we don't actually do it.
           repl_addr = shd_cache_fill(sc->L1cache,mf->addr,sc->gpu_cycle);
      }
   }

   freed_read_mfs++;
   free(mf);
}

unsigned char fq_pop(int tpc_id) 
{
   mem_fetch_t *mf;

   mf = (mem_fetch_t*) icnt_pop(tpc_id);

   // if there is a memory fetch request coming back, forward it to the proper shader core
   if (mf) {
      assert(mf->type == REPLY_DATA);
      time_vector_update(mf->mshr->insts[0].uid ,MR_2SH_FQ_POP,gpu_sim_cycle+gpu_tot_sim_cycle, mf->type ) ;
      fill_shd_L1_with_new_line(sc[mf->sid], mf);
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////

int issue_block2core( shader_core_ctx_t *shdr, int grid_num  ) 
{
   int tid, nthreads_2beissued, more_threads;
   int  nthreads_in_block= 0;
   int start_thread = 0;
   int end_thread = shdr->n_threads;
   int cta_id=-1;
   int cta_size=0;
   int padded_cta_size;

   cta_size = ptx_sim_cta_size();
   padded_cta_size = cta_size;

   assert(gpgpu_spread_blocks_across_cores); //should be if  muliple CTA per shader supported

   for (unsigned i=0;i<max_cta_per_shader(shdr);i++ ) { //try to find next empty cta slot
      if (shdr->cta_status[i]==0) { //
         cta_id=i;
         break;
      }
   }
   assert( cta_id!=-1);//must have found a CTA to run
   if (padded_cta_size%warp_size) {
      padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
   }
   start_thread = cta_id * padded_cta_size;
   end_thread  = start_thread +  cta_size;
   shader_reinit(shdr,start_thread, end_thread); 

   // issue threads in blocks (if it is specified)
   warp_set_t warps;
   for (int i = start_thread; i<end_thread; i++) {  //setup the block
      unsigned warp_id = i/warp_size;
      shdr->thread[i].cta_id = cta_id;
      nthreads_in_block +=  ptx_sim_init_thread(&shdr->thread[i].ptx_thd_info,shdr->sid,i,cta_size-(i-start_thread),shdr->n_threads/*cta_size*/,shdr,cta_id,warp_id);
      warps.set( warp_id );
   }
   shdr->allocate_barrier( cta_id, warps );

   shader_init_CTA(shdr, start_thread, end_thread);
   nthreads_2beissued =  nthreads_in_block;
   shdr->cta_status[cta_id]+=nthreads_2beissued;
   assert( nthreads_2beissued ); //we should have not reached this point if there is no more thread to -

   assert( (unsigned) nthreads_2beissued <= shdr->n_threads); //confirm threads to be issued is less than or equal to number of threads supported by microarchitecture

   int n_cta_issued= nthreads_2beissued/cta_size ;//+ nthreads_2beissued%cta_size; 
   shdr->n_active_cta +=  n_cta_issued;
   shader_CTA_count_log(shdr->sid, n_cta_issued);
   g_total_cta_left-= n_cta_issued;

   more_threads = 1;
   if (gpgpu_spread_blocks_across_cores) {
      nthreads_2beissued += start_thread;
   }
   printf("Shader %d initializing CTA #%d with hw tids from %d to %d @(%lld,%lld)", 
          shdr->sid, cta_id, start_thread, nthreads_2beissued, gpu_sim_cycle, gpu_tot_sim_cycle );
   printf(" shdr->not_completed = %d\n", shdr->not_completed);

   for (tid=start_thread;tid<nthreads_2beissued;tid++) {

      // reset complete flag for stream
      shdr->not_completed += 1;
      assert( shdr->warp[tid/warp_size].n_completed > 0 );
      assert( shdr->warp[tid/warp_size].n_completed <= warp_size);
      shdr->warp[tid/warp_size].n_completed--;

      // set avail4fetch flag to ready
      shdr->thread[tid].avail4fetch = 1;
      assert( shdr->warp[tid/warp_size].n_avail4fetch < warp_size );
      shdr->warp[tid/warp_size].n_avail4fetch++;

      g_nthreads_issued++;
   }

   if (!nthreads_in_block) more_threads = 0;
   return more_threads; //if there are no more threads to be issued, return 0
}

///////////////////////////////////////////////////////////////////////////////////////////
// wrapper code to to create an illusion of a memory controller with L2 cache.
// 
int mem_ctrl_full( int mc_id ) 
{
   if (gpgpu_cache_dl2_opt) {
      return L2c_full( dram[mc_id] );
   } else {
      return( gpgpu_dram_sched_queue_size && dram_full(dram[mc_id]) );
   }
}

//#define DEBUG_PARTIAL_WRITES
void mem_ctrl_push( int mc_id, mem_fetch_t* mf ) 
{
   if (gpgpu_cache_dl2_opt) {
      L2c_push(dram[mc_id], mf);
   } else {
      addrdec_t tlx;
      addrdec_tlx(mf->addr, &tlx);
#if 0 //old chunking no longer valid.
      if (gpgpu_partial_write_mask && mf->write) {
         assert( gpgpu_no_dl1 ); // gpgpu_partial_write_mask is not supported with caches for now
      }
#endif //#if 0 //old chunking no longer valid 
      dram_push(dram[mc_id], 
                tlx.bk, tlx.row, tlx.col, 
                mf->nbytes_L1, mf->write, 
                mf->wid, mf->sid, mf->cache_hits_waiting, mf->addr, mf);
      memlatstat_dram_access(mf, mc_id, tlx.bk);
      if (mf->mshr) mshr_update_status(mf->mshr,IN_DRAM_REQ_QUEUE);
   }
}

void* mem_ctrl_pop( int mc_id ) 
{
   mem_fetch_t* mf;
   if (gpgpu_cache_dl2_opt) {
      mf = L2c_pop(dram[mc_id]);
      if (mf && mf->mshr && mf->mshr->insts[0].callback.function) {
         dram_callback_t* cb = &(mf->mshr->insts[0].callback);
         cb->function(cb->instruction, cb->thread);
      }
      return mf;
   } else {
      mf = static_cast<mem_fetch_t*> (dq_pop(dram[mc_id]->returnq)); //dram_pop(dram[mc_id]);
      if (mf) mf->type = REPLY_DATA;
      if (mf && mf->mshr && mf->mshr->insts[0].callback.function) {
         dram_callback_t* cb = &(mf->mshr->insts[0].callback);
         cb->function(cb->instruction, cb->thread);
      }
      return mf;
   }
}

void* mem_ctrl_top( int mc_id ) 
{
   mem_fetch_t* mf;
   if (gpgpu_cache_dl2_opt) {
      return L2c_top(dram[mc_id]);
   } else {
      mf = static_cast<mem_fetch_t*> (dq_top(dram[mc_id]->returnq));//dram_top(dram[mc_id]);
      if (mf) mf->type = REPLY_DATA;
      return mf ;//dram_top(dram[mc_id]);
   }
}

void get_dram_output ( dram_t* dram_p ) 
{
   mem_fetch_t* mf;
   mem_fetch_t* mf_top;
   mf_top = (mem_fetch_t*) dram_top(dram_p); //test
   if (mf_top) {
      if (mf_top->type == DUMMY_READ) {
         dram_pop(dram_p);
         free(mf_top);
         freed_dummy_read_mfs++;
         return;
      }
   }
   if (gpgpu_cache_dl2_opt) {
      L2c_get_dram_output( dram_p );
   } else {
      if ( dq_full(dram_p->returnq) ) return;
      mf = (mem_fetch_t*) dram_pop(dram_p);
      assert (mf_top==mf );
      if (mf) {
         dq_push(dram_p->returnq, mf);
         if (mf->mshr) mshr_update_status(mf->mshr,IN_DRAMRETURN_Q);
      }
   }
}

void dram_log (int task ) {
   static void ** mrqq_Dist; //memory request queue inside DRAM  
   if (task == CREATELOG) {
      mrqq_Dist = (void **)     calloc(gpu_n_mem,sizeof(void*));
      for (unsigned i=0;i<gpu_n_mem;i++) {
         if (dram[i]->queue_limit)
            mrqq_Dist[i]      = StatCreate("mrqq_length",1,dram[i]->queue_limit);
         else //queue length is unlimited; 
            mrqq_Dist[i]      = StatCreate("mrqq_length",1,64); //track up to 64 entries
      }
   } else if (task == SAMPLELOG) {
      for (unsigned i=0;i<gpu_n_mem;i++) {
         StatAddSample(mrqq_Dist[i], dram_que_length(dram[i]));   
      }
   } else if (task == DUMPLOG) {
      for (unsigned i=0;i<gpu_n_mem;i++) {
         printf ("Queue Length DRAM[%d] ",i);StatDisp(mrqq_Dist[i]);
      } 
   }
}

void dramqueue_latency_log_dump()
{
   for (unsigned i=0;i<gpu_n_mem;i++) {
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->mrqq->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i);StatDisp(dram[i]->rwq->lat_stat);   
   }
}

//Find next clock domain and increment its time
inline int next_clock_domain(void) 
{
   double smallest = min3(core_time,icnt_time,dram_time);
   int mask = 0x00;
   if (gpgpu_cache_dl2_opt  //when no-L2 it will never be L2's turn
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

extern time_t simulation_starttime;
void gpu_sim_loop( int grid_num ) 
{
   int clock_mask = next_clock_domain();

   // shader core loading (pop from ICNT into shader core) follows CORE clock
   if (clock_mask & CORE ) {
      for (int i=0;i<gpu_n_tpc;i++) {
         fq_pop(i); 
      }
   }

   if (clock_mask & ICNT) {
      // pop from memory controller to interconnect
      static unsigned int *rt_size = NULL;
      if (!rt_size) rt_size = (unsigned int*) malloc ((gpu_n_tpc+gpu_n_mem)*sizeof(unsigned int));
      memset(rt_size, 0, (gpu_n_tpc+gpu_n_mem)*sizeof(unsigned int));

      for (unsigned i=0;i<gpu_n_mem;i++) {

         mem_fetch_t* mf;

         mf = (mem_fetch_t*) mem_ctrl_top(i); //(returns L2_top or DRAM returnq top)

         if (mf) {
            mf->source_node = mem2device(i);
            assert( mf->type != RD_REQ && mf->type != WT_REQ ); // never should a request come out from L2 or dram
            if (!mf->write) {
               int return_dev = -1;
               return_dev = mf->sid / gpu_concentration;
               assert(return_dev != -1);
               // check icnt resource for READ data return
               rt_size[return_dev] = mf->nbytes_L1;
               if ( icnt_has_buffer( mem2device(i), rt_size) ) {
                  if (mf->mshr) mshr_update_status(mf->mshr,IN_ICNT2SHADER);
                  memlatstat_icnt2sh_push(mf);
                  time_vector_update(mf->mshr->insts[0].uid ,MR_2SH_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle,RD_REQ);
                  icnt_push( mem2device(i), return_dev, mf, mf->nbytes_L1);
                  mem_ctrl_pop(i);
               } else {
                  gpu_stall_icnt2sh++;
               }
               rt_size[return_dev] = 0; // clean up for the next dram_pop
            } else {
               time_vector_update(mf->request_uid ,MR_2SH_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle,WT_REQ ) ;
               mem_ctrl_pop(i);
               free(mf);
               freed_L1write_mfs++;
               gpgpu_n_processed_writes++;
            }
         }
      }
   }

   if (clock_mask & DRAM) {
      for (unsigned i=0;i<gpu_n_mem;i++) {
         get_dram_output ( dram[i] ); 
      }          
      // Issue the dram command (scheduler + delay model) 
      for (unsigned i=0;i<gpu_n_mem;i++) {
         dram_issueCMD(dram[i]);
      }
      dram_log(SAMPLELOG);   
   }

   // L2 operations follow L2 clock domain
   if (clock_mask & L2) {
      for (unsigned i=0;i<gpu_n_mem;i++) {
         L2c_process_dram_output ( dram[i], i ); // pop from dram
         L2c_push_miss_to_dram ( dram[i] );  //push to dram
         L2c_service_mem_req ( dram[i], i );   // pop(push) from(to)  icnt2l2(l2toicnt) queues; service l2 requests 
      }
      if (gpgpu_cache_dl2_opt) { // L2 cache enabled
         for (unsigned i=0;i<gpu_n_mem;i++) {
            L2c_update_stat( dram[i] ); 
         }
      }
      if (gpgpu_cache_dl2_opt) { //take a sample of l2c queue lengths
         L2c_log(SAMPLELOG); 
      }
   }

   if (clock_mask & ICNT) {
      // pop memory request from ICNT and 
      // push it to the proper memory controller (L2 or DRAM controller)
      for (unsigned i=0;i<gpu_n_mem;i++) {

         if ( mem_ctrl_full(i) ) {
            gpu_stall_dramfull++;
            continue;
         }

         mem_fetch_t* mf;    
         mf = (mem_fetch_t*) icnt_pop( mem2device(i) );

         if (mf) {
            if (mf->type==RD_REQ) {
               time_vector_update(mf->mshr->insts[0].uid ,MR_DRAMQ,gpu_sim_cycle+gpu_tot_sim_cycle,mf->type ) ;             
            } else {
               time_vector_update(mf->request_uid ,MR_DRAMQ,gpu_sim_cycle+gpu_tot_sim_cycle,mf->type ) ;
            }
            memlatstat_icnt2mem_pop(mf);
            mem_ctrl_push( i, mf );
         }
      }
      icnt_transfer( );
   }

   if (clock_mask & CORE) {
      // L1 cache + shader core pipeline stages 
      for (unsigned i=0;i<gpu_n_shader;i++) {
         if (sc[i]->not_completed || more_thread)
            shader_cycle(sc[i], i, grid_num);
         sc[i]->gpu_cycle++;
      }
      gpu_sim_cycle++;

      for (unsigned i=0;i<gpu_n_shader && more_thread;i++) {
         if (gpgpu_spread_blocks_across_cores) {
            int cta_issue_count = 1;
            if ( ( (unsigned) (sc[i]->n_active_cta + cta_issue_count) <= max_cta_per_shader(sc[i]) )
                 && g_total_cta_left ) {
               int j;
               for (j=0;j<cta_issue_count;j++) {
                  issue_block2core(sc[i], grid_num);
               }
               if (!g_total_cta_left) {
                  more_thread = 0;
               }
               assert( g_total_cta_left > -1 );
            }
         } else {
            if (!(sc[i]->not_completed))
               more_thread = issue_block2core(sc[i], grid_num);
         }
      }


      // Flush the caches once all of threads are completed.
      if (gpgpu_flush_cache) {
         int all_threads_complete = 1 ; 
         for (unsigned i=0;i<gpu_n_shader;i++) {
            if (sc[i]->not_completed == 0) {
               shader_cache_flush(sc[i]);
            } else {
               all_threads_complete = 0 ; 
            }
         }
         if (all_threads_complete) {
            printf("Flushed L1 caches...\n");
            if (gpgpu_cache_dl2_opt) {
               int dlc = 0;
               for (unsigned i=0;i<gpu_n_mem;i++) {
                  dlc = L2c_cache_flush(dram[i]);
                  printf("Dirty lines flushed from L2 %d is %d  \n", i, dlc  );
               }
            }
         }
      }

      if (!(gpu_sim_cycle % gpu_stat_sample_freq)) {
         time_t days, hrs, minutes, sec;
         time_t curr_time;
         time(&curr_time);
         unsigned long long  elapsed_time = MAX(curr_time - simulation_starttime, 1);
         days    = elapsed_time/(3600*24);
         hrs     = elapsed_time/3600 - 24*days;
         minutes = elapsed_time/60 - 60*(hrs + 24*days);
         sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
         printf("cycles: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                (double)gpu_sim_insn/(double)gpu_sim_cycle,
                (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                ctime(&curr_time));
         fflush(stdout);
         memlatstat_lat_pw();
         visualizer_printstat();
         if (gpgpu_runtime_stat && (gpu_runtime_stat_flag != 0) ) {
            if (gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
               for (unsigned i=0;i<gpu_n_mem;i++) {
                  dram_print_stat(dram[i],stdout);
               }
               printf("maxmrqlatency = %d \n", max_mrq_latency);
               printf("maxmflatency = %d \n", max_mf_latency);
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_DWF_MAP) {
               printf("DWF_MS: ");
               for (unsigned i=0;i<gpu_n_shader;i++) {
                  printf("%u ",acc_dyn_pcs[i]);
               }
               printf("\n");
               print_thread_pc( stdout );
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
               if (pdom_sched_type) {
                  printf ("pdom_original_warps_count %d \n",n_pdom_sc_orig_stat );
                  printf ("pdom_single_warps_count %d \n",n_pdom_sc_single_stat );
               }
            }
            if (gpu_runtime_stat_flag & GPU_RSTAT_SCHED ) {
               printf("Average Num. Warps Issuable per Shader:\n");
               for (unsigned i=0;i<gpu_n_shader;i++) {
                  printf("%2.2f ", (float) num_warps_issuable_pershader[i]/ gpu_stat_sample_freq);
                  num_warps_issuable_pershader[i] = 0;
               }
               printf("\n");
            }
         }
      }

      for (unsigned i=0;i<gpu_n_mem;i++) {
         acc_mrq_length[i] += dram_que_length(dram[i]);
      }
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

void dump_regs(unsigned sid, unsigned tid)
{
   if ( sid >= gpu_n_shader ) {
      printf("shader %u is out of range\n",sid);
      return;
   }
   if ( tid >= gpu_n_thread_per_shader ) {
      printf("thread %u is out of range\n",tid);
      return;
   }

   shader_core_ctx_t *s = sc[sid];

   ptx_dump_regs( s->thread[tid].ptx_thd_info );
}

int ptx_thread_done( void *thr );

void shader_dump_istream_state(shader_core_ctx_t *shader, FILE *fout )
{
   fprintf( fout, "\n");
   for (unsigned t=0; t < gpu_n_thread_per_shader/warp_size; t++ ) {
      int tid = t*warp_size;
      if ( shader->warp[t].n_completed < warp_size ) {
         fprintf( fout, "  %u:%3u fetch state = c:%u a4f:%u bw:%u (completed: ", shader->sid, tid, 
                shader->warp[t].n_completed,
                shader->warp[t].n_avail4fetch,
                shader->warp[t].n_waiting_at_barrier  );

         for (unsigned i = tid; i < (t+1)*warp_size; i++ ) {
            if ( ptx_thread_done(shader->thread[i].ptx_thd_info) ) {
               fprintf(fout,"1");
            } else {
               fprintf(fout,"0");
            }
            if ( (((i+1)%4) == 0) && (i+1) < (t+1)*warp_size ) {
               fprintf(fout,",");
            }
         }
         fprintf(fout,")\n");
      }
   }
}

void dump_pipeline_impl( int mask, int s, int m ) 
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<gpu_n_shader;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) shader_display_pipeline(sc[i], stdout, 1, mask & 0x2E );
      if(mask&0x40) shader_dump_istream_state(sc[i], stdout);
      if(mask&0x100) mshr_print(stdout, sc[i]);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<gpu_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) dram_print_stat(dram[i],stdout);
         if(mask&0x1000000)   dram_visualize( dram[i] );
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

void dump_pipeline()
{
   dump_pipeline_impl(0,-1,-1);
}
