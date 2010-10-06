/* 
 * gpu-sim.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Ivan Sham and the 
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
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files 
 * src/gpgpusim_entrypoint.c and src/simplesim-3.0/ are derived from the 
 * SimpleScalar Toolset available from http://www.simplescalar.com/ 
 * (property of SimpleScalar LLC) and the files src/intersim/ are derived 
 * from Booksim (Simulator provided with the textbook "Principles and 
 * Practices of Interconnection Networks" available from 
 * http://cva.stanford.edu/books/ppin/).  As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.
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

#ifndef GPU_SIM_H
#define GPU_SIM_H

#include "../abstract_hardware_model.h"
#include <list>
#include <stdio.h>

// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT  0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP  0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2
#define GPU_MEMLATSTAT_QUEUELOGS 0x4

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

// clock constants
#define MhZ *1000000

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

enum divergence_support_t {
   POST_DOMINATOR = 1,
   NUM_SIMD_MODEL
};

struct shader_core_config 
{
   char *pipeline_model;
   unsigned warp_size;
   bool gpgpu_perfect_mem;
   enum divergence_support_t model;
   unsigned n_thread_per_shader;
   unsigned max_warps_per_shader; 
   unsigned max_cta_per_core; //Limit on number of concurrent CTAs in shader core
   unsigned pdom_sched_type;
   bool gpgpu_no_dl1;
   char *gpgpu_cache_texl1_opt;
   char *gpgpu_cache_constl1_opt;
   char *gpgpu_cache_dl1_opt;
   char *gpgpu_cache_il1_opt;
   unsigned n_mshr_per_shader;
   bool gpgpu_dwf_reg_bankconflict;
   bool gpgpu_operand_collector;
   int gpgpu_operand_collector_num_units;
   int gpgpu_operand_collector_num_units_sfu;
   bool gpgpu_stall_on_use;
   bool gpgpu_cache_wt_through;
   //Shader core resources
   unsigned gpgpu_shmem_size;
   unsigned gpgpu_shader_registers;
   int gpgpu_warpdistro_shader;
   int gpgpu_interwarp_mshr_merge;
   int gpgpu_n_shmem_bank;
   int gpgpu_n_cache_bank;
   int gpgpu_shmem_port_per_bank;
   int gpgpu_cache_port_per_bank;
   int gpgpu_const_port_per_bank;
   int gpgpu_shmem_pipe_speedup;  
   unsigned gpgpu_num_reg_banks;
   unsigned gpu_max_cta_per_shader; // TODO: modify this for fermi... computed based upon kernel 
                                    // resource usage; used in shader_core_ctx::translate_local_memaddr 
   bool gpgpu_reg_bank_use_warp_id;
   int gpgpu_coalesce_arch;
   bool gpgpu_local_mem_map;
   int gpu_padded_cta_size;
};

enum dram_ctrl_t {
   DRAM_FIFO=0,
   DRAM_IDEAL_FAST=1
};

struct memory_config {
   char *gpgpu_cache_dl2_opt;
   char *gpgpu_dram_timing_opt;
   char *gpgpu_L2_queue_config;
   bool gpgpu_l2_readoverwrite;
   bool l2_ideal;
   unsigned gpgpu_dram_sched_queue_size;
   unsigned int gpu_mem_n_bk;
   enum dram_ctrl_t scheduler_type;
   bool gpgpu_memlatency_stat;
   unsigned gpgpu_dram_buswidth;
   unsigned gpgpu_dram_burst_length;
};

// global config
extern int gpgpu_mem_address_mask;
extern unsigned int gpu_n_mem_per_ctrlr;

extern int gpu_runtime_stat_flag;
extern int gpgpu_cflog_interval;

extern bool g_interactive_debugger_enabled;

extern int   g_ptx_inst_debug_to_file;
extern char* g_ptx_inst_debug_file;
extern int   g_ptx_inst_debug_thread_uid;




class gpgpu_sim {
public:
   gpgpu_sim();

   void reg_options(class OptionParser * opp);
   void init_gpu();
   void set_prop( struct cudaDeviceProp *prop );

   void launch( kernel_info_t &kinfo );
   void next_grid( unsigned &grid_num, class function_info *&entry );

   unsigned run_gpu_sim();

   unsigned char fq_has_buffer(unsigned long long int addr, int bsize, bool write, int sid );
   void decrement_atomic_count( unsigned sid, unsigned wid );
   void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc );
   const kernel_info_t &the_kernel() const { return m_the_kernel; }

   int shared_mem_size() const;
   int num_registers_per_core() const;
   int wrp_size() const;
   int shader_clock() const;
   const struct cudaDeviceProp *get_prop() const;
   enum divergence_support_t simd_model() const; 

   unsigned num_shader() const { return m_n_shader; }
   unsigned threads_per_core() const;
   void mem_instruction_stats( class warp_inst_t* warp);
   int issue_mf_from_fq(class mem_fetch *mf);

   void gpu_print_stat() const;
   void dump_pipeline( int mask, int s, int m ) const;

private:
   // clocks
   void init_clock_domains(void);
   void reinit_clock_domains(void);
   int next_clock_domain(void);
    
   unsigned char check_icnt_has_buffer(unsigned long long int addr, int bsize, int sid );
   void cycle();
   void fq_pop(int tpc_id);
   void L2c_options(class OptionParser *opp);
   void L2c_print_cache_stat() const;
   void L2c_print_debug();
   void L2c_latency_log_dump();
   void shader_print_runtime_stat( FILE *fout );
   void shader_print_l1_miss_stat( FILE *fout );
   void shader_print_accstats( FILE* fout ) const;
   void visualizer_printstat();
   void print_shader_cycle_distro( FILE *fout ) const;

   void gpgpu_debug();

   // data
   class shader_core_ctx **m_sc;
   class memory_partition_unit **m_memory_partition_unit;

   unsigned m_grid_num;
   kernel_info_t m_the_kernel;
   std::list<kernel_info_t> m_running_kernels;

   // clock domains - frequency
   double core_freq;
   double icnt_freq;
   double dram_freq;
   double l2_freq;
  
   // clock period  
   double core_period;
   double icnt_period;
   double dram_period;
   double l2_period;

   // time of next rising edge 
   double core_time;
   double icnt_time;
   double dram_time;
   double l2_time;

   // configuration parameters
   bool m_options_set;
   struct cudaDeviceProp     *m_cuda_properties;
   struct shader_core_config *m_shader_config;
   struct memory_config      *m_memory_config;
   unsigned int               m_n_shader;
   unsigned int               m_n_mem;
   int gpu_concentration;

   int m_pdom_sched_type;

   // options
   bool gpu_deadlock_detect;

   // stats
   struct shader_core_stats  *m_shader_stats;
   class memory_stats_t      *m_memory_stats;
   unsigned long long  gpu_tot_issued_cta;
   unsigned long long  gpu_tot_completed_thread;
   unsigned long long  last_gpu_sim_insn;

   // debug
   bool gpu_deadlock;
public:
   unsigned long long  gpu_sim_insn;
   unsigned long long  gpu_tot_sim_insn;
   unsigned long long  gpu_sim_insn_last_update;
   unsigned gpu_sim_insn_last_update_sid;
};

// global counters

extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;
extern unsigned g_next_mf_request_uid;

// stats 

extern unsigned int **max_conc_access2samerow;
extern unsigned int **max_servicetime2samerow;
extern unsigned int **row_access;
extern unsigned int **num_activates;
extern unsigned int **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]

extern unsigned int gpgpu_n_sent_writes;
extern unsigned int gpgpu_n_processed_writes;
extern unsigned made_write_mfs;
extern unsigned made_read_mfs;

#endif
