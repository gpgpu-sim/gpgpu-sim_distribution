// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
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

#ifndef GPU_SIM_H
#define GPU_SIM_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include <fstream>
#include <functional> 
#include "../abstract_hardware_model.h"
#include "../cuda-sim/memory.h"
#include "../option_parser.h"
#include "../trace.h"
#include "addrdec.h"
#include "gpu-cache.h"
#include "shader.h"

// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT 0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP 0x8
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

class gpgpu_context;

extern tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

enum dram_ctrl_t { DRAM_FIFO = 0, DRAM_FRFCFS = 1 };

struct power_config {
  power_config() { m_valid = true; }
  void init() {
    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf1[1024];
    snprintf(buf1, 1024, "gpgpusim_power_report__%s.log", date);
    g_power_filename = strdup(buf1);
    char buf2[1024];
    snprintf(buf2, 1024, "gpgpusim_power_trace_report__%s.log.gz", date);
    g_power_trace_filename = strdup(buf2);
    char buf3[1024];
    snprintf(buf3, 1024, "gpgpusim_metric_trace_report__%s.log.gz", date);
    g_metric_trace_filename = strdup(buf3);
    char buf4[1024];
    snprintf(buf4, 1024, "gpgpusim_steady_state_tracking_report__%s.log.gz",
             date);
    g_steady_state_tracking_filename = strdup(buf4);

    if (g_steady_power_levels_enabled) {
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation, &gpu_steady_min_period);
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    if (g_use_nonlinear_model)
      sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
             &gpu_min_inc_per_active_sm);
  }
  void reg_options(class OptionParser *opp);

  char *g_power_config_name;

  bool m_valid;
  bool g_power_simulation_enabled;
  bool g_power_trace_enabled;
  bool g_steady_power_levels_enabled;
  bool g_power_per_cycle_dump;
  bool g_power_simulator_debug;
  char *g_power_filename;
  char *g_power_trace_filename;
  char *g_metric_trace_filename;
  char *g_steady_state_tracking_filename;
  int g_power_trace_zlevel;
  char *gpu_steady_state_definition;
  double gpu_steady_power_deviation;
  double gpu_steady_min_period;

  // Nonlinear power model
  bool g_use_nonlinear_model;
  char *gpu_nonlinear_model_config;
  double gpu_idle_core_power;
  double gpu_min_inc_per_active_sm;
};

class memory_config {
 public:
  memory_config(gpgpu_context *ctx) {
    m_valid = false;
    gpgpu_dram_timing_opt = NULL;
    gpgpu_L2_queue_config = NULL;
    gpgpu_ctx = ctx;
  }
  void init() {
    assert(gpgpu_dram_timing_opt);
    if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
      // dram timing option in ordered variables (legacy)
      // Disabling bank groups if their values are not specified
      nbkgrp = 1;
      tCCDL = 0;
      tRTPL = 0;
      sscanf(gpgpu_dram_timing_opt, "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             &nbk, &tCCD, &tRRD, &tRCD, &tRAS, &tRP, &tRC, &CL, &WL, &tCDLR,
             &tWR, &nbkgrp, &tCCDL, &tRTPL);
    } else {
      // named dram timing options (unordered)
      option_parser_t dram_opp = option_parser_create();

      option_parser_register(dram_opp, "nbk", OPT_UINT32, &nbk,
                             "number of banks", "");
      option_parser_register(dram_opp, "CCD", OPT_UINT32, &tCCD,
                             "column to column delay", "");
      option_parser_register(
          dram_opp, "RRD", OPT_UINT32, &tRRD,
          "minimal delay between activation of rows in different banks", "");
      option_parser_register(dram_opp, "RCD", OPT_UINT32, &tRCD,
                             "row to column delay", "");
      option_parser_register(dram_opp, "RAS", OPT_UINT32, &tRAS,
                             "time needed to activate row", "");
      option_parser_register(dram_opp, "RP", OPT_UINT32, &tRP,
                             "time needed to precharge (deactivate) row", "");
      option_parser_register(dram_opp, "RC", OPT_UINT32, &tRC, "row cycle time",
                             "");
      option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR,
                             "switching from write to read (changes tWTR)", "");
      option_parser_register(dram_opp, "WR", OPT_UINT32, &tWR,
                             "last data-in to row precharge", "");

      option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency",
                             "");
      option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write latency",
                             "");

      // Disabling bank groups if their values are not specified
      option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp,
                             "number of bank groups", "1");
      option_parser_register(
          dram_opp, "CCDL", OPT_UINT32, &tCCDL,
          "column to column delay between accesses to different bank groups",
          "0");
      option_parser_register(
          dram_opp, "RTPL", OPT_UINT32, &tRTPL,
          "read to precharge delay between accesses to different bank groups",
          "0");

      option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;");
      fprintf(stdout, "DRAM Timing Options:\n");
      option_parser_print(dram_opp, stdout);
      option_parser_destroy(dram_opp);
    }

    int nbkt = nbk / nbkgrp;
    unsigned i;
    for (i = 0; nbkt > 0; i++) {
      nbkt = nbkt >> 1;
    }
    bk_tag_length = i - 1;
    assert(nbkgrp > 0 && "Number of bank groups cannot be zero");
    tRCDWR = tRCD - (WL + 1);
    if (elimnate_rw_turnaround) {
      tRTW = 0;
      tWTR = 0;
    } else {
      tRTW = (CL + (BL / data_command_freq_ratio) + 2 - WL);
      tWTR = (WL + (BL / data_command_freq_ratio) + tCDLR);
    }
    tWTP = (WL + (BL / data_command_freq_ratio) + tWR);
    dram_atom_size =
        BL * busW * gpu_n_mem_per_ctrlr;  // burst length x bus width x # chips
                                          // per partition

    assert(m_n_sub_partition_per_memory_channel > 0);
    assert((nbk % m_n_sub_partition_per_memory_channel == 0) &&
           "Number of DRAM banks must be a perfect multiple of memory sub "
           "partition");
    m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
    fprintf(stdout, "Total number of memory sub partition = %u\n",
            m_n_mem_sub_partition);

    m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel);
    m_L2_config.init(&m_address_mapping);

    m_valid = true;

    sscanf(write_queue_size_opt, "%d:%d:%d",
           &gpgpu_frfcfs_dram_write_queue_size, &write_high_watermark,
           &write_low_watermark);
  }
  void reg_options(class OptionParser *opp);

  bool m_valid;
  mutable l2_cache_config m_L2_config;
  bool m_L2_texure_only;

  char *gpgpu_dram_timing_opt;
  char *gpgpu_L2_queue_config;
  bool l2_ideal;
  unsigned gpgpu_frfcfs_dram_sched_queue_size;
  unsigned gpgpu_dram_return_queue_size;
  enum dram_ctrl_t scheduler_type;
  bool gpgpu_memlatency_stat;
  unsigned m_n_mem;
  unsigned m_n_sub_partition_per_memory_channel;
  unsigned m_n_mem_sub_partition;
  unsigned gpu_n_mem_per_ctrlr;

  unsigned rop_latency;
  unsigned dram_latency;

  // DRAM parameters

  unsigned tCCDL;  // column to column delay when bank groups are enabled
  unsigned tRTPL;  // read to precharge delay when bank groups are enabled for
                   // GDDR5 this is identical to RTPS, if for other DRAM this is
                   // different, you will need to split them in two

  unsigned tCCD;    // column to column delay
  unsigned tRRD;    // minimal time required between activation of rows in
                    // different banks
  unsigned tRCD;    // row to column delay - time required to activate a row
                    // before a read
  unsigned tRCDWR;  // row to column delay for a write command
  unsigned tRAS;    // time needed to activate row
  unsigned tRP;     // row precharge ie. deactivate row
  unsigned
      tRC;  // row cycle time ie. precharge current, then activate different row
  unsigned tCDLR;  // Last data-in to Read command (switching from write to
                   // read)
  unsigned tWR;    // Last data-in to Row precharge

  unsigned CL;    // CAS latency
  unsigned WL;    // WRITE latency
  unsigned BL;    // Burst Length in bytes (4 in GDDR3, 8 in GDDR5)
  unsigned tRTW;  // time to switch from read to write
  unsigned tWTR;  // time to switch from write to read
  unsigned tWTP;  // time to switch from write to precharge in the same bank
  unsigned busW;

  unsigned nbkgrp;  // number of bank groups (has to be power of 2)
  unsigned
      bk_tag_length;  // number of bits that define a bank inside a bank group

  unsigned nbk;

  bool elimnate_rw_turnaround;

  unsigned
      data_command_freq_ratio;  // frequency ratio between DRAM data bus and
                                // command bus (2 for GDDR3, 4 for GDDR5)
  unsigned
      dram_atom_size;  // number of bytes transferred per read or write command

  linear_to_raw_address_translation m_address_mapping;

  unsigned icnt_flit_size;

  unsigned dram_bnk_indexing_policy;
  unsigned dram_bnkgrp_indexing_policy;
  bool dual_bus_interface;

  bool seperate_write_queue_enabled;
  char *write_queue_size_opt;
  unsigned gpgpu_frfcfs_dram_write_queue_size;
  unsigned write_high_watermark;
  unsigned write_low_watermark;
  bool m_perf_sim_memcpy;
  bool simple_dram_model;

  gpgpu_context *gpgpu_ctx;
};

// global counters and flags (please try not to add to this list!!!)
//extern unsigned long long gpu_sim_cycle;
//extern unsigned long long gpu_tot_sim_cycle;
extern bool g_interactive_debugger_enabled;

class gpgpu_sim_config : public power_config,
                         public gpgpu_functional_sim_config {
 public:
  gpgpu_sim_config(gpgpu_context *ctx)
      : m_shader_config(ctx), m_memory_config(ctx) {
    m_valid = false;
    gpgpu_ctx = ctx;
  }
  void reg_options(class OptionParser *opp);
  void init() {
    gpu_stat_sample_freq = 10000;
    gpu_runtime_stat_flag = 0;
    sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
           &gpu_runtime_stat_flag);
    m_shader_config.init();
    ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
    m_memory_config.init();
    init_clock_domains();
    power_config::init();
    Trace::init();

    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf[1024];
    snprintf(buf, 1024, "gpgpusim_visualizer__%s.log.gz", date);
    g_visualizer_filename = strdup(buf);

    m_valid = true;
  }

  unsigned num_shader() const { return m_shader_config.num_shader(); }
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
  unsigned num_core_per_cluster() const { return m_shader_config.n_simt_cores_per_cluster; }
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }
  unsigned checkpoint_option;

  size_t stack_limit() const { return stack_size_limit; }
  size_t heap_limit() const { return heap_size_limit; }
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;
  }

  bool flush_l1() const { return gpgpu_flush_l1_cache; }
  void convert_byte_string();

 private:
  void init_clock_domains(void);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
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
  unsigned long long gpu_max_cycle_opt;
  unsigned long long gpu_max_insn_opt;
  unsigned gpu_max_cta_opt;
  unsigned gpu_max_completed_cta_opt;
  char *gpgpu_runtime_stat;
  bool gpgpu_flush_l1_cache;
  bool gpgpu_flush_l2_cache;
  bool gpu_deadlock_detect;
  int gpgpu_frfcfs_dram_sched_queue_size;
  int gpgpu_cflog_interval;
  char *gpgpu_clock_domains;
  unsigned max_concurrent_kernel;

  // visualizer
  bool g_visualizer_enabled;
  char *g_visualizer_filename;
  int g_visualizer_zlevel;

  // statistics collection
  int gpu_stat_sample_freq;
  int gpu_runtime_stat_flag;
  unsigned long long page_table_walk_latency;

  int eviction_policy;
  bool invalidate_clean;
  float reserve_accessed_page_percent;
  float free_page_buffer_percentage;

  char *pcie_bandwidth_string;
  float pcie_bandwidth;

  float curve_a;
  float curve_b;

  int enable_dma;
  int multiply_dma_penalty;
  unsigned migrate_threshold;

  bool enable_smart_runtime;

  // Device Limits
  size_t stack_size_limit;
  size_t heap_size_limit;
  size_t runtime_sync_depth_limit;
  size_t runtime_pending_launch_count_limit;

  // gpu compute capability options
  unsigned int gpgpu_compute_capability_major;
  unsigned int gpgpu_compute_capability_minor;
  unsigned long long liveness_message_freq;

  friend class gpgpu_sim;

  public:
  int hardware_prefetch;
  int hwprefetch_oversub;

  friend class gmmu_t;
};

extern unsigned long long kernel_time;
extern unsigned long long memory_copy_time_h2d;
extern unsigned long long memory_copy_time_d2h;
extern unsigned long long prefetch_time;
extern unsigned long long devicesync_time;
extern unsigned long long writeback_time;
extern unsigned long long dma_time;

enum stats_type {
  prefetch = 0,
  prefetch_breakdown,
  memcpy_h2d,
  memcpy_d2h,
  memcpy_d2d,
  kernel_launch,
  page_fault,
  device_sync,
  write_back,
  invalidate,
  dma
};

class event_stats {
public:
  event_stats(enum stats_type t, unsigned long long s_time,
              unsigned long long e_time)
      : type(t), start_time(s_time), end_time(e_time) {}
  event_stats(enum stats_type t, unsigned long long s_time)
      : type(t), start_time(s_time), end_time(0) {}
  enum stats_type type;
  unsigned long long start_time;
  unsigned long long end_time;

  virtual void print(FILE *fout, float freq) = 0;
  virtual void calculate() = 0;
};

class memory_stats : public event_stats {
public:
  memory_stats(enum stats_type t, unsigned long long s_time, mem_addr_t s_addr,
               size_t sz, unsigned s_id)
      : event_stats(t, s_time), start_addr(s_addr), size(sz), stream_id(s_id) {}
  memory_stats(enum stats_type t, unsigned long long s_time,
               unsigned long long e_time, mem_addr_t s_addr, size_t sz,
               unsigned s_id)
      : event_stats(t, s_time, e_time), start_addr(s_addr), size(sz),
        stream_id(s_id) {}
  mem_addr_t start_addr;
  size_t size;
  unsigned stream_id;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t St: %x Sz: %lu \t Sm: %u \t ",
            start_time, end_time, start_addr, size, stream_id);
    if (type == memcpy_h2d)
      fprintf(fout, "T: memcpy_h2d");
    else if (type == memcpy_d2h)
      fprintf(fout, "T: memcpy_d2h");
    else if (type == memcpy_d2d)
      fprintf(fout, "T: memcpy_d2d");
    else if (type == prefetch)
      fprintf(fout, "T: prefetch");
    else if (type == prefetch_breakdown)
      fprintf(fout, "T: prefetch_breakdown");
    else if (type == device_sync)
      fprintf(fout, "T: device_sync");
    else if (type == write_back)
      fprintf(fout, "T: write_back");
    else if (type == invalidate)
      fprintf(fout, "T: invalidate");
    else if (type == dma)
      fprintf(fout, "T: dma");

    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }
  virtual void calculate() {
    if (type == memcpy_h2d) {
      memory_copy_time_h2d += end_time - start_time;
    } else if (type == memcpy_d2h) {
      memory_copy_time_d2h += end_time - start_time;
    } else if (type == prefetch_breakdown) {
      prefetch_time += end_time - start_time;
    } else if (type == device_sync) {
      devicesync_time += end_time - start_time;
    } else if (type == write_back) {
      writeback_time += end_time - start_time;
    } else if (type == dma) {
      dma_time += end_time - start_time;
    }
  }
};

class kernel_stats : public event_stats {
public:
  kernel_stats(unsigned long long s_time, unsigned s_id, unsigned k_id)
      : event_stats(kernel_launch, s_time), stream_id(s_id), kernel_id(k_id) {}
  unsigned stream_id;
  unsigned kernel_id;

  virtual void print(FILE *fout, float freq) {
    fprintf(
        fout,
        "F: %8llu----T: %8llu \t \t \t Kl: %u \t Sm: %u \t T: kernel_launch",
        start_time, end_time, kernel_id, stream_id);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() { kernel_time += end_time - start_time; }
};

class page_fault_stats : public event_stats {
public:
  page_fault_stats(unsigned long long s_time, const std::list<mem_addr_t> &pgs,
                   unsigned sz)
      : event_stats(page_fault, s_time), pages(pgs), transfering_pages(pgs),
        size(sz) {}
  std::list<mem_addr_t> pages;
  std::list<mem_addr_t> transfering_pages;
  size_t size;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t Sz: %lu \t T: page_fault",
            start_time, end_time, size);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() {}
};

extern std::map<unsigned long long, std::list<event_stats *>> sim_prof;

extern bool sim_prof_enable;

void print_sim_prof(FILE *fout, float freq);

void calculate_sim_prof(FILE *fout, gpgpu_sim *gpu);

void update_sim_prof_kernel(unsigned kernel_id, unsigned long long end_time);

void update_sim_prof_prefetch(mem_addr_t start_addr, size_t size,
                              unsigned long long end_time);

void update_sim_prof_prefetch_break_down(unsigned long long end_time);

void print_UVM_stats(gpgpu_new_stats *new_stats, gpgpu_sim *gpu, FILE *fout);

class access_info {
public:
  mem_addr_t page_no;
  mem_addr_t mem_addr;
  size_t size;
  unsigned long long cycle;
  bool is_read;
  unsigned sm_id;
  unsigned warp_id;
  access_info(mem_addr_t p_n, mem_addr_t addr, size_t s, unsigned long long c,
              bool rw, unsigned s_id, unsigned w_id)
      : page_no(p_n), mem_addr(addr), size(s), cycle(c), is_read(rw),
        sm_id(s_id), warp_id(w_id) {}
};

class gpgpu_new_stats {
public:
  gpgpu_new_stats(const gpgpu_sim_config &config);
  ~gpgpu_new_stats();
  void print(FILE *fout) const;
  void print_pcie(FILE *fout) const;
  void print_access_pattern_detail(FILE *fout) const;
  void print_access_pattern(FILE *fout) const;
  void print_time_and_access(FILE *fout) const;

  // for each shader of all global memory access

  // tlb hit
  unsigned long long *tlb_hit;
  // tlb miss
  unsigned long long *tlb_miss;

  // tlb validate
  unsigned long long *tlb_val;
  // tlb eviction
  unsigned long long *tlb_evict;
  // tlb invalidated by page eviction
  unsigned long long *tlb_page_evict;

  // in tlb miss, page hit
  unsigned long long *mf_page_hit;
  // in tlb miss, page miss
  unsigned long long *mf_page_miss;

  // in tlb miss, page miss, the first create fault
  unsigned long long mf_page_fault_outstanding;
  // in tlb miss, page miss, the following that appends to mshr
  unsigned long long mf_page_fault_pending;

  unsigned long long page_evict_dirty;

  unsigned long long page_evict_not_dirty;

  // prefetch page hit
  unsigned long long pf_page_hit;
  // prefetch page miss
  unsigned long long pf_page_miss;
  // prefetch fault page size, large page and latency
  std::vector<std::pair<unsigned long, unsigned long long>> pf_fault_latency;

  // for each page, how many time is it being accessed by each shader
  std::map<mem_addr_t, unsigned> *page_access_times;

  // for each timestamp, which page is being accessed
  std::list<access_info> time_and_page_access;

  // ready lanes utilization
  std::list<std::pair<unsigned long long, float>> pcie_read_utilization;
  // write lanes utilization
  std::list<std::pair<unsigned long long, float>> pcie_write_utilization;

  // page and its partern
  std::map<mem_addr_t, std::vector<bool>> page_thrashing;
  // tlb and its partern
  std::map<mem_addr_t, std::vector<bool>> *tlb_thrashing;

  // for each shader, the memory access latency
  std::map<unsigned, std::pair<bool, unsigned long long>> *ma_latency;

  // for mf when it is fault(not pending to prefetch), the latency
  std::map<mem_addr_t, std::list<unsigned long long>> mf_page_fault_latency;

  // for prefetch each small page latency
  std::map<mem_addr_t, std::list<unsigned long long>> pf_page_fault_latency;

  const gpgpu_sim_config &m_config;

  unsigned long long num_dma;
  unsigned long long dma_page_transfer_read;
  unsigned long long dma_page_transfer_write;
};

// this class simulate the gmmu unit on chip

class gmmu_t {
public:
  gmmu_t(class gpgpu_sim *gpu, const gpgpu_sim_config &config,
         class gpgpu_new_stats *new_stats);
  unsigned long long calculate_transfer_time(size_t data_size);
  void calculate_devicesync_time(size_t data_size);
  void cycle();
  void register_tlbflush_callback(std::function<void(mem_addr_t)> cb_tlb);
  void tlb_flush(mem_addr_t page_num);
  void page_eviction_procedure();
  bool is_block_evictable(mem_addr_t bb_addr, size_t size);

  // add a new accessed page or refresh the position of the page in the LRU page
  // list being called on detecting tlb hit or when memory fetch comes back from
  // the upward (gmmu to cu) queue
  void refresh_valid_pages(mem_addr_t page_addr);
  void sort_valid_pages();

  // check whether the page to be accessed is already in pci-e write stage queue
  // being called on tlb hit or on tlb miss but no page fault
  void check_write_stage_queue(mem_addr_t page_num, bool refresh);

  void valid_pages_erase(mem_addr_t pagenum);
  void valid_pages_clear();

  void register_prefetch(mem_addr_t m_device_addr,
                         mem_addr_t m_device_allocation_ptr, size_t m_cnt,
                         struct CUstream_st *m_stream);
  void activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt,
                         struct CUstream_st *m_stream);

  struct lp_tree_node *build_lp_tree(mem_addr_t addr, size_t size);
  void reset_large_page_info(struct lp_tree_node *node);
  void reset_lp_tree_node(struct lp_tree_node *node);
  struct lp_tree_node *get_lp_node(mem_addr_t addr);
  void evict_whole_tree(struct lp_tree_node *root);
  mem_addr_t update_basic_block(struct lp_tree_node *root, mem_addr_t addr,
                                size_t size, bool prefetch);
  mem_addr_t get_basic_block(struct lp_tree_node *root, mem_addr_t addr);

  void fill_lp_tree(struct lp_tree_node *node,
                    std::set<mem_addr_t> &scheduled_basic_blocks);
  void remove_lp_tree(struct lp_tree_node *node,
                      std::set<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_fill_lp_tree(struct lp_tree_node *node,
                                 std::set<mem_addr_t> &scheduled_basic_blocks);
  void
  traverse_and_remove_lp_tree(struct lp_tree_node *node,
                              std::set<mem_addr_t> &scheduled_basic_blocks);

  bool pcie_transfers_completed();

  void initialize_large_page(mem_addr_t start_addr, size_t size);

  unsigned long long get_ready_cycle(unsigned num_pages);
  unsigned long long get_ready_cycle_dma(unsigned size);

  float get_pcie_utilization(unsigned num_pages);

  void do_hardware_prefetch(
      std::map<mem_addr_t, std::list<mem_fetch *>> &page_fault_this_turn);

  void reserve_pages_insert(mem_addr_t addr, unsigned mem_access_uid);
  void reserve_pages_remove(mem_addr_t addr, unsigned mem_access_uid);
  bool reserve_pages_check(mem_addr_t addr);

  std::map<mem_addr_t, std::list<unsigned>> reserve_pages;

  void update_hardware_prefetcher_oversubscribed();

  // update paging, pinning, and eviction decision based on memory access
  // pattern under oversubscription
  void update_memory_management_policy();
  void log_kernel_info(unsigned kernel_id, unsigned long long time,
                       bool finish);

  void reset_large_page_info();

  mem_addr_t get_eviction_base_addr(mem_addr_t page_addr);
  size_t get_eviction_granularity(mem_addr_t page_addr);

  int get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr);
  int get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr);
  void inc_bb_access_counter(mem_addr_t addr);
  void inc_bb_round_trip(struct lp_tree_node *root);
  void traverse_and_reset_access_counter(struct lp_tree_node *root);
  void reset_bb_access_counter();
  void traverse_and_reset_round_trip(struct lp_tree_node *root);
  void reset_bb_round_trip();
  void update_access_type(mem_addr_t addr, int type);

  bool should_cause_page_migration(mem_addr_t addr, bool is_write);

private:
  // data structure to wrap memory fetch and page table walk delay
  struct page_table_walk_latency_t {
    mem_fetch *mf;
    unsigned long long ready_cycle;
  };

  // page table walk delay queue
  std::list<page_table_walk_latency_t> page_table_walk_queue;

  enum class latency_type {
    PCIE_READ,
    PCIE_WRITE_BACK,
    INVALIDATE,
    PAGE_FAULT,
    DMA
  };

  // data structure to wrap a memory page and delay to transfer over PCI-E
  struct pcie_latency_t {
    mem_addr_t start_addr;
    unsigned long long size;
    std::list<mem_addr_t> page_list;
    unsigned long long ready_cycle;

    mem_fetch *mf;
    latency_type type;
  };

  // staging queue to hold the PCI-E requests waiting for scheduling
  std::list<pcie_latency_t *> pcie_read_stage_queue;
  std::list<pcie_latency_t *> pcie_write_stage_queue;

  // read queue for fetching the page from host side
  // the request may be global memory's read (load)/ write (store)
  pcie_latency_t *pcie_read_latency_queue;

  // write back queue for page eviction requests over PCI-E
  pcie_latency_t *pcie_write_latency_queue;

  // loosely represent MSHRs to hold all memory fetches
  // corresponding to a PCI-E read requests, i.e., a common page number
  // to replay the memory fetch back upon completion
  std::map<mem_addr_t, std::list<mem_fetch *>> req_info;

  // need the gpu to do address traslation, validate page
  class gpgpu_sim *m_gpu;

  // config file
  const gpgpu_sim_config &m_config;
  const struct shader_core_config *m_shader_config;

  // callback functions to invalidate the tlb in ldst unit
  std::list<std::function<void(mem_addr_t)>> callback_tlb_flush;

  // list of valid pages (valid = 1, accessed = 1/0, dirty = 1/0) ordered as LRU
  std::list<eviction_t *> valid_pages;

  // page eviction policy
  enum class eviction_policy { LRU, TBN, SEQUENTIAL_LOCAL, RANDOM, LFU, LRU4K };

  // types of hardware prefetcher
  enum class hwardware_prefetcher { DISBALED, TBN, SEQUENTIAL_LOCAL, RANDOM };

  // types of hardware prefetcher under over-subscription
  enum class hwardware_prefetcher_oversub {
    DISBALED,
    TBN,
    SEQUENTIAL_LOCAL,
    RANDOM
  };

  // type of DMA
  enum class dma_type { DISABLED, ADAPTIVE, ALWAYS, OVERSUB };

  // type of memory access pattern per data structure
  enum class ds_pattern {
    UNDECIDED,
    RANDOM,
    LINEAR,
    MIXED,
    RANDOM_REUSE,
    LINEAR_REUSE,
    MIXED_REUSE
  };

  // list of scheduled basic blocks by their timestamps
  std::list<std::pair<unsigned long long, mem_addr_t>> block_access_list;

  // list of launch and finish cycle of kernels keyed by id
  std::map<unsigned, std::pair<unsigned long long, unsigned long long>>
      kernel_info;

  eviction_policy evict_policy;
  hwardware_prefetcher prefetcher;
  hwardware_prefetcher_oversub oversub_prefetcher;

  dma_type dma_mode;

  struct prefetch_req {
    // starting address (rolled up and down for page alignment) for the prefetch
    mem_addr_t start_addr;

    // current address from the start up to which PCI-e has already processed
    mem_addr_t cur_addr;

    // starting address of the current variable allocation
    mem_addr_t allocation_addr;

    // total size (rolled up and down for page alignment) for the prefetch
    size_t size;

    // stream associated to the prefetch
    CUstream_st *m_stream;

    // memory fetches, which are created upon page fault and are depending on
    // current prefetch, aggreagted before the prefetch is actually scheduled
    std::map<mem_addr_t, std::list<mem_fetch *>> incoming_replayable_nacks;

    // memory fetches that are finished PCI-e transfer are aggregated to be
    // replayed together upon completion of the prefetch
    std::map<mem_addr_t, std::list<mem_fetch *>> outgoing_replayable_nacks;

    // list of pages (max upto 2MB) from the current prefetch request which are
    // being served by PCI-e
    std::list<mem_addr_t> pending_prefetch;

    // stream manager upon reaching to this entry of the queue sets it to active
    bool active;
  };

  std::list<prefetch_req> prefetch_req_buffer;

  std::list<event_stats *> fault_stats;
  std::list<event_stats *> writeback_stats;

  std::list<struct lp_tree_node *> large_page_info;
  size_t total_allocation_size;

  bool over_sub;

  class gpgpu_new_stats *m_new_stats;
};

struct lp_tree_node {
  mem_addr_t addr;
  size_t size;
  size_t valid_size;
  struct lp_tree_node *left;
  struct lp_tree_node *right;
  uint32_t access_counter;
  uint8_t RW;
};

struct occupancy_stats {
  occupancy_stats()
      : aggregate_warp_slot_filled(0), aggregate_theoretical_warp_slots(0) {}
  occupancy_stats(unsigned long long wsf, unsigned long long tws)
      : aggregate_warp_slot_filled(wsf),
        aggregate_theoretical_warp_slots(tws) {}

  unsigned long long aggregate_warp_slot_filled;
  unsigned long long aggregate_theoretical_warp_slots;

  float get_occ_fraction() const {
    return float(aggregate_warp_slot_filled) /
           float(aggregate_theoretical_warp_slots);
  }

  occupancy_stats &operator+=(const occupancy_stats &rhs) {
    aggregate_warp_slot_filled += rhs.aggregate_warp_slot_filled;
    aggregate_theoretical_warp_slots += rhs.aggregate_theoretical_warp_slots;
    return *this;
  }

  occupancy_stats operator+(const occupancy_stats &rhs) const {
    return occupancy_stats(
        aggregate_warp_slot_filled + rhs.aggregate_warp_slot_filled,
        aggregate_theoretical_warp_slots +
            rhs.aggregate_theoretical_warp_slots);
  }
};

class gpgpu_context;
class ptx_instruction;

class watchpoint_event {
 public:
  watchpoint_event() {
    m_thread = NULL;
    m_inst = NULL;
  }
  watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) {
    m_thread = thd;
    m_inst = pI;
  }
  const ptx_thread_info *thread() const { return m_thread; }
  const ptx_instruction *inst() const { return m_inst; }

 private:
  const ptx_thread_info *m_thread;
  const ptx_instruction *m_inst;
};

class gpgpu_sim : public gpgpu_t {
 public:
  gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx);

  void set_prop(struct cudaDeviceProp *prop);

  void launch(kernel_info_t *kinfo);
  bool can_start_kernel();
  unsigned finished_kernel();
  void set_kernel_done(kernel_info_t *kernel);
  void stop_all_running_kernels();

  void init();
  void cycle();
  bool active();
  bool cycle_insn_cta_max_hit() {
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
  }
  void print_stats();
  void update_stats();
  void deadlock_check();
  void inc_completed_cta() { gpu_completed_cta++; }
  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc);

  int shared_mem_size() const;
  int shared_mem_per_block() const;
  int compute_capability_major() const;
  int compute_capability_minor() const;
  int num_registers_per_core() const;
  int num_registers_per_block() const;
  int wrp_size() const;
  int shader_clock() const;
  int max_cta_per_core() const;
  int get_max_cta(const kernel_info_t &k) const;
  const struct cudaDeviceProp *get_prop() const;
  enum divergence_support_t simd_model() const;

  unsigned threads_per_core() const;
  bool get_more_cta_left() const;
  bool kernel_more_cta_left(kernel_info_t *kernel) const;
  bool hit_max_cta_count() const;
  kernel_info_t *select_kernel();
  void decrement_kernel_latency();

  const gpgpu_sim_config &get_config() const { return m_config; }
  void gpu_print_stat();
  void dump_pipeline(int mask, int s, int m) const;

  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);

  // The next three functions added to be used by the functional simulation
  // function

  //! Get shader core configuration
  /*!
   * Returning the configuration of the shader core, used by the functional
   * simulation only so far
   */
  const shader_core_config *getShaderCoreConfig();

  //! Get shader core Memory Configuration
  /*!
   * Returning the memory configuration of the shader core, used by the
   * functional simulation only so far
   */
  const memory_config *getMemoryConfig();

  //! Get shader core SIMT cluster
  /*!
   * Returning the cluster of of the shader core, used by the functional
   * simulation so far
   */
  simt_core_cluster *getSIMTCluster(int index);

  gmmu_t *getGmmu();

  void hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                      const ptx_instruction *pI);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 private:
  // clocks
  void reinit_clock_domains(void);
  int next_clock_domain(void);
  void issue_block2core();
  void print_dram_stats(FILE *fout) const;
  void shader_print_runtime_stat(FILE *fout);
  void shader_print_l1_miss_stat(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout) const;
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info) const;
  void visualizer_printstat();
  void print_shader_cycle_distro(FILE *fout) const;

  void gpgpu_debug();

 protected:
  ///// data /////
  class gmmu_t *m_gmmu;
  class simt_core_cluster **m_cluster;
  class memory_partition_unit **m_memory_partition_unit;
  class memory_sub_partition **m_memory_sub_partition;

  std::vector<kernel_info_t *> m_running_kernels;
  unsigned m_last_issued_kernel;

  std::list<unsigned> m_finished_kernel;
  // m_total_cta_launched == per-kernel count. gpu_tot_issued_cta == global
  // count.
  unsigned long long m_total_cta_launched;
  unsigned long long gpu_tot_issued_cta;
  unsigned gpu_completed_cta;

  unsigned m_last_cluster_issue;
  float *average_pipeline_duty_cycle;
  float *active_sms;
  // time of next rising edge
  double core_time;
  double icnt_time;
  double dram_time;
  double l2_time;
  double gmmu_time;

  // debug
  bool gpu_deadlock;

  //// configuration parameters ////
  const gpgpu_sim_config &m_config;

  const struct cudaDeviceProp *m_cuda_properties;
  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;

  // stats
  class shader_core_stats *m_shader_stats;
  class memory_stats_t *m_memory_stats;
  class power_stat_t *m_power_stats;
  class gpgpu_sim_wrapper *m_gpgpusim_wrapper;
  unsigned long long last_gpu_sim_insn;

  unsigned long long last_liveness_message_time;

  std::map<std::string, FuncCache> m_special_cache_config;

  std::vector<std::string>
      m_executed_kernel_names;  //< names of kernel for stat printout
  std::vector<unsigned>
      m_executed_kernel_uids;  //< uids of kernel launches for stat printout
  std::map<unsigned, watchpoint_event> g_watchpoint_hits;

  std::string executed_kernel_info_string();  //< format the kernel information
                                              // into a string for stat printout
  void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout
  virtual void createSIMTCluster() = 0;

public:
  class gpgpu_new_stats *m_new_stats;

 public:
  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  occupancy_stats gpu_occupancy;
  occupancy_stats gpu_tot_occupancy;

  // performance counter for stalls due to congestion.
  unsigned int gpu_stall_dramfull;
  unsigned int gpu_stall_icnt2sh;
  unsigned long long partiton_reqs_in_parallel;
  unsigned long long partiton_reqs_in_parallel_total;
  unsigned long long partiton_reqs_in_parallel_util;
  unsigned long long partiton_reqs_in_parallel_util_total;
  unsigned long long gpu_sim_cycle_parition_util;
  unsigned long long gpu_tot_sim_cycle_parition_util;
  unsigned long long partiton_replys_in_parallel;
  unsigned long long partiton_replys_in_parallel_total;

  FuncCache get_cache_config(std::string kernel_name);
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);
  bool has_special_cache_config(std::string kernel_name);
  void change_cache_config(FuncCache cache_config);
  void set_cache_config(std::string kernel_name);

  // Jin: functional simulation for CDP
 private:
  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  kernel_info_t *m_functional_sim_kernel;

 public:
  bool is_functional_sim() { return m_functional_sim; }
  kernel_info_t *get_functional_kernel() { return m_functional_sim_kernel; }
  void functional_launch(kernel_info_t *k) {
    m_functional_sim = true;
    m_functional_sim_kernel = k;
  }
  void finish_functional_sim(kernel_info_t *k) {
    assert(m_functional_sim);
    assert(m_functional_sim_kernel == k);
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;
  }
};

class exec_gpgpu_sim : public gpgpu_sim {
 public:
  exec_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

#endif
