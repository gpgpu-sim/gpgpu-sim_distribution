// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Vijay Kandiah,
// Nikos Hardavellas Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include "../abstract_hardware_model.h"
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

// SST communication functions
/**
 * @brief Check if SST requests buffer is full
 *
 * @param core_id
 * @return true
 * @return false
 */
extern bool is_SST_buffer_full(unsigned core_id);
__attribute__((weak)) bool is_SST_buffer_full(unsigned core_id) {
  return false;
}

/**
 * @brief Send loads to SST memory backend
 *
 * @param core_id
 * @param address
 * @param size
 * @param mem_req
 */
extern void send_read_request_SST(unsigned core_id, uint64_t address,
                                  size_t size, void *mem_req);
__attribute__((weak)) void send_read_request_SST(unsigned core_id,
                                                 uint64_t address, size_t size,
                                                 void *mem_req) {}
/**
 * @brief Send stores to SST memory backend
 *
 * @param core_id
 * @param address
 * @param size
 * @param mem_req
 */
extern void send_write_request_SST(unsigned core_id, uint64_t address,
                                   size_t size, void *mem_req);
__attribute__((weak)) void send_write_request_SST(unsigned core_id,
                                                  uint64_t address, size_t size,
                                                  void *mem_req) {}

enum dram_ctrl_t { DRAM_FIFO = 0, DRAM_FRFCFS = 1 };

enum hw_perf_t {
  HW_BENCH_NAME = 0,
  HW_KERNEL_NAME,
  HW_L1_RH,
  HW_L1_RM,
  HW_L1_WH,
  HW_L1_WM,
  HW_CC_ACC,
  HW_SHRD_ACC,
  HW_DRAM_RD,
  HW_DRAM_WR,
  HW_L2_RH,
  HW_L2_RM,
  HW_L2_WH,
  HW_L2_WM,
  HW_NOC,
  HW_PIPE_DUTY,
  HW_NUM_SM_IDLE,
  HW_CYCLES,
  HW_VOLTAGE,
  HW_TOTAL_STATS
};

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
    // snprintf(buf1, 1024, "accelwattch_power_report__%s.log", date);
    snprintf(buf1, 1024, "accelwattch_power_report.log");
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
    // for(int i =0; i< hw_perf_t::HW_TOTAL_STATS; i++){
    //   accelwattch_hybrid_configuration[i] = 0;
    // }

    if (g_steady_power_levels_enabled) {
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation, &gpu_steady_min_period);
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    // if (g_use_nonlinear_model)
    //   sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
    //          &gpu_min_inc_per_active_sm);
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

  char *g_hw_perf_file_name;
  char *g_hw_perf_bench_name;
  int g_power_simulation_mode;
  bool g_dvfs_enabled;
  bool g_aggregate_power_stats;
  bool accelwattch_hybrid_configuration[hw_perf_t::HW_TOTAL_STATS];

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

  /**
   * @brief Check if the config script is in SST mode
   *
   * @return true
   * @return false
   */
  bool is_SST_mode() const { return SST_mode; }

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
  bool SST_mode;
  gpgpu_context *gpgpu_ctx;
};

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
  unsigned get_core_freq() const { return core_freq; }
  unsigned num_shader() const { return m_shader_config.num_shader(); }
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }

  /**
   * @brief Check if we are in SST mode
   *
   * @return true
   * @return false
   */
  bool is_SST_mode() const { return m_memory_config.SST_mode; }

  unsigned checkpoint_option;

  size_t stack_limit() const { return stack_size_limit; }
  size_t heap_limit() const { return heap_size_limit; }
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;
  }

  bool flush_l1() const { return gpgpu_flush_l1_cache; }

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
  friend class sst_gpgpu_sim;
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
  void print_stats(unsigned long long streamID);
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
  PowerscalingCoefficients *get_scaling_coeffs();
  void decrement_kernel_latency();

  const gpgpu_sim_config &get_config() const { return m_config; }
  void gpu_print_stat(unsigned long long streamID);
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
  simt_core_cluster *getSIMTCluster();

  void hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                      const ptx_instruction *pI);

  /**
   * @brief Check if we are in SST mode
   *
   * @return true
   * @return false
   */
  bool is_SST_mode() { return m_config.is_SST_mode(); }

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 protected:
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
  std::string executed_kernel_name();
  void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout
  virtual void createSIMTCluster() = 0;

 public:
  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  occupancy_stats gpu_occupancy;
  occupancy_stats gpu_tot_occupancy;

  typedef struct {
    unsigned long long start_cycle;
    unsigned long long end_cycle;
  } kernel_time_t;
  std::map<unsigned long long, std::map<unsigned, kernel_time_t>>
      gpu_kernel_time;
  unsigned long long last_streamID;
  unsigned long long last_uid;
  cache_stats aggregated_l1_stats;
  cache_stats aggregated_l2_stats;

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
 protected:
  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  kernel_info_t *m_functional_sim_kernel;

 public:
  bool is_functional_sim() { return m_functional_sim; }
  kernel_info_t *get_functional_kernel() { return m_functional_sim_kernel; }
  std::vector<kernel_info_t *> get_running_kernels() {
    return m_running_kernels;
  }
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

/**
 * @brief A GPGPUSim class customized to SST Balar interfacing
 *
 */
class sst_gpgpu_sim : public gpgpu_sim {
 public:
  sst_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  // SST memory handling
  std::vector<std::deque<mem_fetch *>>
      SST_gpgpu_reply_buffer; /** SST mem response queue */

  /**
   * @brief Receive mem request's response from SST and put
   *        it in a buffer (SST_gpgpu_reply_buffer)
   *
   * @param core_id
   * @param mem_req
   */
  void SST_receive_mem_reply(unsigned core_id, void *mem_req);

  /**
   * @brief Pop the head of the buffer queue to get the
   *        memory response
   *
   * @param core_id
   * @return mem_fetch*
   */
  mem_fetch *SST_pop_mem_reply(unsigned core_id);

  virtual void createSIMTCluster();

  // SST Balar interfacing
  /**
   * @brief Advance core and collect stats
   *
   */
  void SST_cycle();

  /**
   * @brief Wrapper of SST_cycle()
   *
   */
  void cycle();

  /**
   * @brief Whether the GPU is active, removed test for
   *        memory system since that is handled in SST
   *
   * @return true
   * @return false
   */
  bool active();

  /**
   * @brief SST mode use SST memory system instead, so the memcpy
   *        is empty here
   *
   * @param dst_start_addr
   * @param count
   */
  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {};

  /**
   * @brief Check if the SST config matches up with the
   *        gpgpusim.config in core number
   *
   * @param sst_numcores SST core count
   */
  void SST_gpgpusim_numcores_equal_check(unsigned sst_numcores);
};

#endif
