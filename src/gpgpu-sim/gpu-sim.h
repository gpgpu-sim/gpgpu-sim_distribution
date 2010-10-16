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
#include "addrdec.h"

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

enum dram_ctrl_t {
   DRAM_FIFO=0,
   DRAM_IDEAL_FAST=1
};

struct memory_config {
   memory_config()
   {
       gpgpu_cache_dl2_opt=NULL;
       gpgpu_dram_timing_opt=NULL;
       gpgpu_L2_queue_config=NULL;
   }
   void init()
   {
      assert(gpgpu_dram_timing_opt);
      sscanf(gpgpu_dram_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",&nbk,&tCCD,&tRRD,&tRCD,&tRAS,&tRP,&tRC,&CL,&WL,&tWTR);
      tRCDWR = tRCD-(WL+1);
      tRTW = (CL+(BL/2)+2-WL);
      m_address_mapping.init(m_n_mem);
   }

   char *gpgpu_cache_dl2_opt;
   char *gpgpu_dram_timing_opt;
   char *gpgpu_L2_queue_config;
   bool gpgpu_l2_readoverwrite;
   bool l2_ideal;
   unsigned gpgpu_dram_sched_queue_size;
   enum dram_ctrl_t scheduler_type;
   bool gpgpu_memlatency_stat;
   unsigned m_n_mem;
   unsigned int gpu_n_mem_per_ctrlr;

   // DRAM parameters
   unsigned int tCCD;   //column to column delay
   unsigned int tRRD;   //minimal time required between activation of rows in different banks
   unsigned int tRCD;   //row to column delay - time required to activate a row before a read
   unsigned int tRCDWR; //row to column delay for a write command
   unsigned int tRAS;   //time needed to activate row
   unsigned int tRP;    //row precharge ie. deactivate row
   unsigned int tRC;    //row cycle time ie. precharge current, then activate different row

   unsigned int CL;     //CAS latency
   unsigned int WL;     //WRITE latency
   unsigned int BL;     //Burst Length in bytes (we're using 4? could be 8)
   unsigned int tRTW;   //time to switch from read to write
   unsigned int tWTR;   //time to switch from write to read 5? look in datasheet
   unsigned int busW;

   unsigned int nbk;

   linear_to_raw_address_translation m_address_mapping;
};

// global config
extern int gpgpu_mem_address_mask;
extern int gpu_runtime_stat_flag;
extern int gpgpu_cflog_interval;
extern bool g_interactive_debugger_enabled;
extern int   g_ptx_inst_debug_to_file;
extern char* g_ptx_inst_debug_file;
extern int   g_ptx_inst_debug_thread_uid;

// global counters
extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;

class gpgpu_sim : public gpgpu_t {
public:
   gpgpu_sim();

   void reg_options(class OptionParser * opp);
   void init_gpu();
   void set_prop( struct cudaDeviceProp *prop );

   void launch( kernel_info_t &kinfo );
   void next_grid();

   unsigned run_gpu_sim();

   void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc );
   const kernel_info_t &the_kernel() const { return m_the_kernel; }

   int shared_mem_size() const;
   int num_registers_per_core() const;
   int wrp_size() const;
   int shader_clock() const;
   const struct cudaDeviceProp *get_prop() const;
   enum divergence_support_t simd_model() const; 

   unsigned num_shader() const { return m_shader_config->n_simt_clusters*m_shader_config->n_simt_cores_per_cluster; }
   unsigned threads_per_core() const;
   void mem_instruction_stats( class warp_inst_t &inst);

   void gpu_print_stat() const;
   void dump_pipeline( int mask, int s, int m ) const;

   unsigned get_forced_max_capability() const { return m_ptx_force_max_capability; }
   bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
   bool saved_converted_ptxplus() const { return m_ptx_save_converted_ptxplus; }

private:
   // clocks
   void init_clock_domains(void);
   void reinit_clock_domains(void);
   int next_clock_domain(void);
    
   void cycle();
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
   unsigned sid_to_cluster( unsigned sid ) const;

///// data /////

   class simt_core_cluster **m_cluster;
   class memory_partition_unit **m_memory_partition_unit;

   kernel_info_t m_the_kernel;
   std::list<kernel_info_t> m_running_kernels;

   unsigned int more_thread;

   // time of next rising edge 
   double core_time;
   double icnt_time;
   double dram_time;
   double l2_time;

   // debug
   bool gpu_deadlock;

   //// configuration parameters ////
   bool m_options_set;

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

   struct cudaDeviceProp     *m_cuda_properties;
   struct shader_core_config *m_shader_config;
   struct memory_config      *m_memory_config;

   int m_pdom_sched_type;

   bool gpu_deadlock_detect;
   int   m_ptx_convert_to_ptxplus;
   int   m_ptx_save_converted_ptxplus;
   unsigned m_ptx_force_max_capability;

   // stats
   struct shader_core_stats  *m_shader_stats;
   class memory_stats_t      *m_memory_stats;
   unsigned long long  gpu_tot_issued_cta;
   unsigned long long  gpu_tot_completed_thread;
   unsigned long long  last_gpu_sim_insn;

public:
   unsigned long long  gpu_sim_insn;
   unsigned long long  gpu_tot_sim_insn;
   unsigned long long  gpu_sim_insn_last_update;
   unsigned gpu_sim_insn_last_update_sid;
};

#endif
