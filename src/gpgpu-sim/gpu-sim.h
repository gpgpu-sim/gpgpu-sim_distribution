// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GPU_SIM_H
#define GPU_SIM_H

#include "../abstract_hardware_model.h"
#include "addrdec.h"
#include "shader.h"

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
   DRAM_FRFCFS=1
};

struct memory_config {
   memory_config()
   {
       m_valid = false;
       gpgpu_dram_timing_opt=NULL;
       gpgpu_L2_queue_config=NULL;
   }
   void init()
   {
      assert(gpgpu_dram_timing_opt);
      sscanf(gpgpu_dram_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             &nbk,&tCCD,&tRRD,&tRCD,&tRAS,&tRP,&tRC,&CL,&WL,&tCDLR,&tWR);
      tRCDWR = tRCD-(WL+1);
      tRTW = (CL+(BL/2)+2-WL);
      tWTR = (WL+(BL/2)+tCDLR); 
      tWTP = (WL+(BL/2)+tWR);
      m_address_mapping.init(m_n_mem);
      m_L2_config.init();
      m_valid = true;
   }
   void reg_options(class OptionParser * opp);

   bool m_valid;
   cache_config m_L2_config;
   bool m_L2_texure_only;

   char *gpgpu_dram_timing_opt;
   char *gpgpu_L2_queue_config;
   bool l2_ideal;
   unsigned gpgpu_dram_sched_queue_size;
   enum dram_ctrl_t scheduler_type;
   bool gpgpu_memlatency_stat;
   unsigned m_n_mem;
   unsigned gpu_n_mem_per_ctrlr;

   // DRAM parameters
   unsigned tCCD;   //column to column delay
   unsigned tRRD;   //minimal time required between activation of rows in different banks
   unsigned tRCD;   //row to column delay - time required to activate a row before a read
   unsigned tRCDWR; //row to column delay for a write command
   unsigned tRAS;   //time needed to activate row
   unsigned tRP;    //row precharge ie. deactivate row
   unsigned tRC;    //row cycle time ie. precharge current, then activate different row
   unsigned tCDLR;  //Last data-in to Read command (switching from write to read)
   unsigned tWR;    //Last data-in to Row precharge 

   unsigned CL;     //CAS latency
   unsigned WL;     //WRITE latency
   unsigned BL;     //Burst Length in bytes (we're using 4? could be 8)
   unsigned tRTW;   //time to switch from read to write
   unsigned tWTR;   //time to switch from write to read 
   unsigned tWTP;   //time to switch from write to precharge in the same bank
   unsigned busW;

   unsigned nbk;

   linear_to_raw_address_translation m_address_mapping;
};

// global counters and flags (please try not to add to this list!!!)
extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;
extern bool g_interactive_debugger_enabled;

class gpgpu_sim_config : public gpgpu_functional_sim_config {
public:
    gpgpu_sim_config() { m_valid = false; }
    void reg_options(class OptionParser * opp);
    void init() 
    {
        gpu_stat_sample_freq = 10000;
        gpu_runtime_stat_flag = 0;
        sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq, &gpu_runtime_stat_flag);
        m_shader_config.init();    
        ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
        m_memory_config.init();
        init_clock_domains(); 

        // initialize file name if it is not set 
        time_t curr_time;
        time(&curr_time);
        char *date = ctime(&curr_time);
        char *s = date;
        while (*s) {
            if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
            if (*s == '\n' || *s == '\r' ) *s = 0;
            s++;
        }
        char buf[1024];
        snprintf(buf,1024,"gpgpusim_visualizer__%s.log.gz",date);
        g_visualizer_filename = strdup(buf);

        m_valid=true;
    }

    unsigned num_shader() const { return m_shader_config.num_shader(); }
    unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }

private:
    void init_clock_domains(void ); 

    bool m_valid;
    shader_core_config m_shader_config;
    memory_config m_memory_config;

    // clock domains - frequency
    double core_freq;
    double icnt_freq;
    double dram_freq;
    double l2_freq;
    double core_period;
    double icnt_period;
    double dram_period;
    double l2_period;

    // GPGPU-Sim timing model options
    unsigned gpu_max_cycle_opt;
    unsigned gpu_max_insn_opt;
    unsigned gpu_max_cta_opt;
    char *gpgpu_runtime_stat;
    bool  gpgpu_flush_cache;
    bool  gpu_deadlock_detect;
    int   gpgpu_dram_sched_queue_size; 
    int   gpgpu_cflog_interval;
    char * gpgpu_clock_domains;
    unsigned max_concurrent_kernel;

    // visualizer
    bool  g_visualizer_enabled;
    char *g_visualizer_filename;
    int   g_visualizer_zlevel;

    // statistics collection
    int gpu_stat_sample_freq;
    int gpu_runtime_stat_flag;

    friend class gpgpu_sim;
};

class gpgpu_sim : public gpgpu_t {
public:
   gpgpu_sim( const gpgpu_sim_config &config );

   void set_prop( struct cudaDeviceProp *prop );

   void launch( kernel_info_t *kinfo );
   bool can_start_kernel();
   unsigned finished_kernel();
   void set_kernel_done( kernel_info_t *kernel );

   void init();
   void cycle();
   bool active(); 
   void print_stats();
   void deadlock_check();

   void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc );

   int shared_mem_size() const;
   int num_registers_per_core() const;
   int wrp_size() const;
   int shader_clock() const;
   const struct cudaDeviceProp *get_prop() const;
   enum divergence_support_t simd_model() const; 

   unsigned threads_per_core() const;
   bool get_more_cta_left() const;
   kernel_info_t *select_kernel();

   const gpgpu_sim_config &get_config() const { return m_config; }
   void gpu_print_stat() const;
   void dump_pipeline( int mask, int s, int m ) const;

private:
   // clocks
   void reinit_clock_domains(void);
   int  next_clock_domain(void);
   void issue_block2core();
    
   void L2c_print_cache_stat() const;
   void shader_print_runtime_stat( FILE *fout );
   void shader_print_l1_miss_stat( FILE *fout ) const;
   void visualizer_printstat();
   void print_shader_cycle_distro( FILE *fout ) const;

   void gpgpu_debug();

///// data /////

   class simt_core_cluster **m_cluster;
   class memory_partition_unit **m_memory_partition_unit;

   std::vector<kernel_info_t*> m_running_kernels;
   unsigned m_last_issued_kernel;

   std::list<unsigned> m_finished_kernel;
   unsigned m_total_cta_launched;
   unsigned m_last_cluster_issue;

   // time of next rising edge 
   double core_time;
   double icnt_time;
   double dram_time;
   double l2_time;

   // debug
   bool gpu_deadlock;

   //// configuration parameters ////
   const gpgpu_sim_config &m_config;
  
   const struct cudaDeviceProp     *m_cuda_properties;
   const struct shader_core_config *m_shader_config;
   const struct memory_config      *m_memory_config;

   // stats
   class shader_core_stats  *m_shader_stats;
   class memory_stats_t     *m_memory_stats;
   unsigned long long  gpu_tot_issued_cta;
   unsigned long long  last_gpu_sim_insn;

public:
   unsigned long long  gpu_sim_insn;
   unsigned long long  gpu_tot_sim_insn;
   unsigned long long  gpu_sim_insn_last_update;
   unsigned gpu_sim_insn_last_update_sid;
};

#endif
