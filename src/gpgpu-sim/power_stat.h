// Copyright (c) 2009-2021,  Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
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

#ifndef POWER_STAT_H
#define POWER_STAT_H

#include <stdio.h>
#include <zlib.h>
#include "gpu-sim.h"
#include "mem_latency_stat.h"

typedef enum _stat_idx {
  CURRENT_STAT_IDX = 0,  // Current activity count
  PREV_STAT_IDX,         // Previous sample activity count
  NUM_STAT_IDX           // Total number of samples
} stat_idx;

struct shader_core_power_stats_pod {
  // [CURRENT_STAT_IDX] = CURRENT_STAT_IDX stat, [PREV_STAT_IDX] = last reading
  float *m_pipeline_duty_cycle[NUM_STAT_IDX];
  unsigned *m_num_decoded_insn[NUM_STAT_IDX];  // number of instructions
                                               // committed by this shader core
  unsigned
      *m_num_FPdecoded_insn[NUM_STAT_IDX];  // number of instructions committed
                                            // by this shader core
  unsigned
      *m_num_INTdecoded_insn[NUM_STAT_IDX];  // number of instructions committed
                                             // by this shader core
    unsigned *m_num_storequeued_insn[NUM_STAT_IDX];
    unsigned *m_num_loadqueued_insn[NUM_STAT_IDX];
    unsigned *m_num_tex_inst[NUM_STAT_IDX];
    double *m_num_ialu_acesses[NUM_STAT_IDX];
    double *m_num_fp_acesses[NUM_STAT_IDX];
    double *m_num_imul_acesses[NUM_STAT_IDX];
    double *m_num_imul32_acesses[NUM_STAT_IDX];
    double *m_num_imul24_acesses[NUM_STAT_IDX];
    double *m_num_fpmul_acesses[NUM_STAT_IDX];
    double *m_num_idiv_acesses[NUM_STAT_IDX];
    double *m_num_fpdiv_acesses[NUM_STAT_IDX];
    double *m_num_dp_acesses[NUM_STAT_IDX];
    double *m_num_dpmul_acesses[NUM_STAT_IDX];
    double *m_num_dpdiv_acesses[NUM_STAT_IDX];
    double *m_num_sp_acesses[NUM_STAT_IDX];
    double *m_num_sfu_acesses[NUM_STAT_IDX];
    double *m_num_sqrt_acesses[NUM_STAT_IDX];
    double *m_num_log_acesses[NUM_STAT_IDX];
    double *m_num_sin_acesses[NUM_STAT_IDX];
    double *m_num_exp_acesses[NUM_STAT_IDX];
    double *m_num_tensor_core_acesses[NUM_STAT_IDX];
    double *m_num_const_acesses[NUM_STAT_IDX];
    double *m_num_tex_acesses[NUM_STAT_IDX];
    double *m_num_mem_acesses[NUM_STAT_IDX];
    unsigned *m_num_sp_committed[NUM_STAT_IDX];
    unsigned *m_num_sfu_committed[NUM_STAT_IDX];
    unsigned *m_num_mem_committed[NUM_STAT_IDX];
    unsigned *m_active_sp_lanes[NUM_STAT_IDX];
    unsigned *m_active_sfu_lanes[NUM_STAT_IDX];
    double *m_active_exu_threads[NUM_STAT_IDX];
    double *m_active_exu_warps[NUM_STAT_IDX];    
    unsigned *m_read_regfile_acesses[NUM_STAT_IDX];
    unsigned *m_write_regfile_acesses[NUM_STAT_IDX];
    unsigned *m_non_rf_operands[NUM_STAT_IDX];
};

class power_core_stat_t : public shader_core_power_stats_pod {
 public:
  power_core_stat_t(const shader_core_config *shader_config,
                    shader_core_stats *core_stats);
  void visualizer_print(gzFile visualizer_file);
  void print(FILE *fout);
  void init();
  void save_stats();
 

 private:
  shader_core_stats *m_core_stats;
  const shader_core_config *m_config;
  float average_duty_cycle;
};

struct mem_power_stats_pod {
  // [CURRENT_STAT_IDX] = CURRENT_STAT_IDX stat, [PREV_STAT_IDX] = last reading
  class cache_stats core_cache_stats[NUM_STAT_IDX];  // Total core stats
  class cache_stats l2_cache_stats[NUM_STAT_IDX];    // Total L2 partition stats

  unsigned *shmem_access[NUM_STAT_IDX];  // Shared memory access
  // Low level DRAM stats
  unsigned *n_cmd[NUM_STAT_IDX];
  unsigned *n_activity[NUM_STAT_IDX];
  unsigned *n_nop[NUM_STAT_IDX];
  unsigned *n_act[NUM_STAT_IDX];
  unsigned *n_pre[NUM_STAT_IDX];
  unsigned *n_rd[NUM_STAT_IDX];
  unsigned *n_wr[NUM_STAT_IDX];
  unsigned *n_wr_WB[NUM_STAT_IDX];
  unsigned *n_req[NUM_STAT_IDX];

  // Interconnect stats
  long *n_simt_to_mem[NUM_STAT_IDX];
  long *n_mem_to_simt[NUM_STAT_IDX];
};

class power_mem_stat_t : public mem_power_stats_pod {
 public:
  power_mem_stat_t(const memory_config *mem_config,
                   const shader_core_config *shdr_config,
                   memory_stats_t *mem_stats, shader_core_stats *shdr_stats);
  void visualizer_print(gzFile visualizer_file);
  void print(FILE *fout) const;
  void init();
  void save_stats();

 private:
  memory_stats_t *m_mem_stats;
  shader_core_stats *m_core_stats;
  const memory_config *m_config;
  const shader_core_config *m_core_config;
};

class power_stat_t {
 public:
  power_stat_t(const shader_core_config *shader_config,
               float *average_pipeline_duty_cycle, float *active_sms,
               shader_core_stats *shader_stats, const memory_config *mem_config,
               memory_stats_t *memory_stats);
  void visualizer_print(gzFile visualizer_file);
  void print(FILE *fout) const;
  void save_stats() {
    pwr_core_stat->save_stats();
    pwr_mem_stat->save_stats();
    *m_average_pipeline_duty_cycle = 0;
    *m_active_sms = 0;
  }
  void clear();
  unsigned l1i_misses_kernel;
  unsigned l1i_hits_kernel;
  unsigned long long l1r_hits_kernel;
  unsigned long long l1r_misses_kernel;
  unsigned long long l1w_hits_kernel;
  unsigned long long l1w_misses_kernel;
  unsigned long long shared_accesses_kernel;
  unsigned long long cc_accesses_kernel;
  unsigned long long dram_rd_kernel;
  unsigned long long dram_wr_kernel;
  unsigned long long dram_pre_kernel;
  unsigned long long l2r_hits_kernel;
  unsigned long long l2r_misses_kernel;
  unsigned long long l2w_hits_kernel;
  unsigned long long l2w_misses_kernel;
  unsigned long long noc_tr_kernel;
  unsigned long long noc_rc_kernel;
  unsigned long long tot_inst_execution;
  unsigned long long tot_int_inst_execution;
  unsigned long long tot_fp_inst_execution;
  unsigned long long commited_inst_execution;
  unsigned long long ialu_acc_execution;
  unsigned long long imul24_acc_execution;
  unsigned long long imul32_acc_execution;
  unsigned long long imul_acc_execution;
  unsigned long long idiv_acc_execution;
  unsigned long long dp_acc_execution;
  unsigned long long dpmul_acc_execution;
  unsigned long long dpdiv_acc_execution;
  unsigned long long fp_acc_execution;
  unsigned long long fpmul_acc_execution;
  unsigned long long fpdiv_acc_execution;
  unsigned long long sqrt_acc_execution;
  unsigned long long log_acc_execution;
  unsigned long long sin_acc_execution;
  unsigned long long exp_acc_execution;
  unsigned long long tensor_acc_execution;
  unsigned long long tex_acc_execution;
  unsigned long long tot_fpu_acc_execution;
  unsigned long long tot_sfu_acc_execution;
  unsigned long long tot_threads_acc_execution;
  unsigned long long tot_warps_acc_execution;
  unsigned long long sp_active_lanes_execution;
  unsigned long long sfu_active_lanes_execution;
  double get_total_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_decoded_insn[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_decoded_insn[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_decoded_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_int_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
          total_inst +=
          (pwr_core_stat->m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]);
      else 
        total_inst +=
          (pwr_core_stat->m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_num_INTdecoded_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_fp_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]);
      else 
        total_inst += (pwr_core_stat->m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_FPdecoded_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_load_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst +=
          (pwr_core_stat->m_num_loadqueued_insn[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_num_loadqueued_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_store_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst +=
          (pwr_core_stat->m_num_storequeued_insn[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_num_storequeued_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_sp_committed_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sp_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_sfu_committed_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sfu_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_mem_committed_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_mem_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_committed_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_mem_committed[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sfu_committed[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sp_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_regfile_reads(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
         total_inst +=
          (pwr_core_stat->m_read_regfile_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
          (pwr_core_stat->m_read_regfile_acesses[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_read_regfile_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_regfile_writes(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst +=
          (pwr_core_stat->m_write_regfile_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
          (pwr_core_stat->m_write_regfile_acesses[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_write_regfile_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  float get_pipeline_duty() {
    float total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst +=
          (pwr_core_stat->m_pipeline_duty_cycle[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_pipeline_duty_cycle[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_non_regfile_operands(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
         total_inst += (pwr_core_stat->m_non_rf_operands[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_non_rf_operands[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_non_rf_operands[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sp_accessess() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sp_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sfu_accessess() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sfu_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sfu_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sqrt_accessess(bool aggregate_stat){
      double total_inst=0;
      for(unsigned i=0; i<m_config->num_shader();i++){
          if(aggregate_stat)
            total_inst+=(pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]);
          else
            total_inst+=(pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sqrt_acesses[PREV_STAT_IDX][i]);
      }
      return total_inst;
  }
  double get_log_accessess(bool aggregate_stat){
      double total_inst=0;
      for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)
          total_inst+=(pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]);
        else 
          total_inst+=(pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_log_acesses[PREV_STAT_IDX][i]);
      }
      return total_inst;
  }
  double get_sin_accessess(bool aggregate_stat){
      double total_inst=0;
      for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)  
          total_inst+=(pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]);
        else 
          total_inst+=(pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sin_acesses[PREV_STAT_IDX][i]);
      }
      return total_inst;
  }
  double get_exp_accessess(bool aggregate_stat){
      double total_inst=0;
      for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)  
          total_inst+=(pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]);
        else  
          total_inst+=(pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_exp_acesses[PREV_STAT_IDX][i]);
      }
      return total_inst;
  }

  double get_mem_accessess() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_mem_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_mem_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intdiv_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_idiv_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_fpdiv_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_fpdiv_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intmul32_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intmul24_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intmul_accessess(bool aggregate_stat){
      double total_inst=0;
      for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)
          total_inst+= (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]); 
        else  
          total_inst+= (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) - 
                       (pwr_core_stat->m_num_imul_acesses[PREV_STAT_IDX][i]);
      }
      return total_inst;
  }

  double get_fpmul_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)
          total_inst += (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]);
        else
          total_inst += (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_fpmul_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_fp_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)
          total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]);
        else  
          total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_fp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_dp_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)
          total_inst += (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]);
        else  
          total_inst += (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_dp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_dpmul_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
      if(aggregate_stat)  
        total_inst += (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_dpmul_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_dpdiv_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
      if(aggregate_stat)  
        total_inst += (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_dpdiv_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_tensor_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
      if(aggregate_stat)  
        total_inst += (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_tensor_core_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_const_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
        if(aggregate_stat)
          total_inst += pwr_core_stat->m_num_const_acesses[CURRENT_STAT_IDX][i];
        else
          total_inst += (pwr_core_stat->m_num_const_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_const_acesses[PREV_STAT_IDX][i]);
    }
    return (total_inst);
  }

  double get_tex_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
      if(aggregate_stat)  
        total_inst += (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]) - 
                      (pwr_core_stat->m_num_tex_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sp_active_lanes() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_active_sp_lanes[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_sp_lanes[PREV_STAT_IDX][i]);
    }
    return (total_inst / m_config->num_shader()) / m_config->gpgpu_num_sp_units;
  }

  float get_sfu_active_lanes() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_active_sfu_lanes[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_sfu_lanes[PREV_STAT_IDX][i]);
    }

    return (total_inst / m_config->num_shader()) /
           m_config->gpgpu_num_sfu_units;
  }


  float get_active_threads(bool aggregate_stat) {
    unsigned total_threads = 0;
    unsigned total_warps = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat){
        total_threads += (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]) ;
        total_warps += (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]);
      }
      else{
        total_threads += (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_exu_threads[PREV_STAT_IDX][i]);
        total_warps += (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_exu_warps[PREV_STAT_IDX][i]);
        }
    }
    if(total_warps != 0)
      return (float)((float)total_threads / (float)total_warps);
    else
      return 0;
  }

  unsigned long long get_tot_threads_kernel(bool aggregate_stat) {
    unsigned total_threads = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat){
        total_threads += (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]) ;
      }
      else{
        total_threads += (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_exu_threads[PREV_STAT_IDX][i]);
        }
    }

      return total_threads;
  }
  unsigned long long get_tot_warps_kernel(bool aggregate_stat) {
    unsigned long long total_warps = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat){
        total_warps += (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]);
      }
      else{
        total_warps += (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_exu_warps[PREV_STAT_IDX][i]);
        }
    }
      return total_warps;
  }


  double get_tot_fpu_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_fp_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_dp_acesses[PREV_STAT_IDX][i]);
    }
    //total_inst += get_total_load_inst()+get_total_store_inst()+get_tex_inst();
    return total_inst;
  }



  double get_tot_sfu_accessess(bool aggregate_stat){
    double total_inst=0;
    for(unsigned i=0; i<m_config->num_shader();i++){
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i])+
                    (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]);
        else
            total_inst += (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_idiv_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_sqrt_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_log_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_sin_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_exp_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_fpdiv_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_fpmul_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_dpmul_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_dpdiv_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_imul_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_tensor_core_acesses[PREV_STAT_IDX][i]) +
                    (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]) - 
                    (pwr_core_stat->m_num_tex_acesses[PREV_STAT_IDX][i]);

    }
    return total_inst;
  }

  double get_ialu_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_core_stat->m_num_ialu_acesses[CURRENT_STAT_IDX][i]);
      else  
        total_inst += (pwr_core_stat->m_num_ialu_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_ialu_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_tex_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_tex_inst[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_tex_inst[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_constant_c_accesses() {
    enum mem_access_type access_type[] = {CONST_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_constant_c_misses() {
    enum mem_access_type access_type[] = {CONST_ACC_R};
    enum cache_request_status request_status[] = {MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_constant_c_hits() {
    return (get_constant_c_accesses() - get_constant_c_misses());
  }
  double get_texture_c_accesses() {
    enum mem_access_type access_type[] = {TEXTURE_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_texture_c_misses() {
    enum mem_access_type access_type[] = {TEXTURE_ACC_R};
    enum cache_request_status request_status[] = {MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_texture_c_hits() {
    return (get_texture_c_accesses() - get_texture_c_misses());
  }
  double get_inst_c_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {INST_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if(aggregate_stat)
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    else
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_inst_c_misses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {INST_ACC_R};
    enum cache_request_status request_status[] = {MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if(aggregate_stat)
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    else
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_inst_c_hits(bool aggregate_stat) {
    return (get_inst_c_accesses(aggregate_stat) - get_inst_c_misses(aggregate_stat));
  }

  double get_l1d_read_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, SECTOR_MISS}; 
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if(aggregate_stat){
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
      }
  }
  double get_l1d_read_misses(bool aggregate_stat) {
    return (get_l1d_read_accesses(aggregate_stat) - get_l1d_read_hits(aggregate_stat));
  }
  double get_l1d_read_hits(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R};
    enum cache_request_status request_status[] = {HIT, MSHR_HIT};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if(aggregate_stat){
       return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
      }
  }
  double get_l1d_write_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W};
    enum cache_request_status request_status[] = {HIT, MISS, SECTOR_MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if(aggregate_stat){
       return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
      }
  }
  double get_l1d_write_misses(bool aggregate_stat) {
    return (get_l1d_write_accesses(aggregate_stat) - get_l1d_write_hits(aggregate_stat));
  }
  double get_l1d_write_hits(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W};
    enum cache_request_status request_status[] = {HIT, MSHR_HIT};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if(aggregate_stat){
       return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
      }
  }
  double get_cache_misses() {
    return get_l1d_read_misses(0) + get_constant_c_misses() +
           get_l1d_write_misses(0) + get_texture_c_misses();
  }

  double get_cache_read_misses() {
    return get_l1d_read_misses(0) + get_constant_c_misses() +
           get_texture_c_misses();
  }

  double get_cache_write_misses() { return get_l1d_write_misses(0); }

  double get_shmem_access(bool aggregate_stat) {
    unsigned total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if(aggregate_stat)
        total_inst += (pwr_mem_stat->shmem_access[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_mem_stat->shmem_access[CURRENT_STAT_IDX][i]) -
                    (pwr_mem_stat->shmem_access[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  unsigned long long  get_l2_read_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {
        GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, INST_ACC_R};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED, MISS, SECTOR_MISS}; 
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if(aggregate_stat){
       return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
  }

  unsigned long long get_l2_read_misses(bool aggregate_stat) {
    return (get_l2_read_accesses(aggregate_stat) - get_l2_read_hits(aggregate_stat));
  }

  unsigned long long get_l2_read_hits(bool aggregate_stat) {
       enum mem_access_type access_type[] = {
        GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, INST_ACC_R};
    enum cache_request_status request_status[] =  {HIT, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if(aggregate_stat){
       return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
  }

  unsigned long long get_l2_write_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W,
                                          L1_WRBK_ACC};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED, MISS, SECTOR_MISS}; 
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if(aggregate_stat){
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
  }

  unsigned long long get_l2_write_misses(bool aggregate_stat) {
    return (get_l2_write_accesses(aggregate_stat) - get_l2_write_hits(aggregate_stat));
  }
  unsigned long long get_l2_write_hits(bool aggregate_stat) {
        enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W,
                                          L1_WRBK_ACC};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if(aggregate_stat){
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
    else{
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
    }
  }
  double get_dram_cmd() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_cmd[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_activity() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_activity[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_nop() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_nop[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_act() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_act[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_act[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_pre(bool aggregate_stat) {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      if(aggregate_stat){
        total += pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i];
      }
      else{
        total += (pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_pre[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }
  double get_dram_rd(bool aggregate_stat) {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      if(aggregate_stat){
        total += pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i];
      }
      else{
        total += (pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_rd[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }
  double get_dram_wr(bool aggregate_stat) {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      if(aggregate_stat){
        total += pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i] + 
                pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i];
      }
      else{
        total += (pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i] - 
                pwr_mem_stat->n_wr[PREV_STAT_IDX][i]) +
                (pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i] - 
                pwr_mem_stat->n_wr_WB[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }
  double get_dram_req() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_req[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_req[PREV_STAT_IDX][i]);
    }
    return total;
  }

  unsigned long long get_icnt_simt_to_mem(bool aggregate_stat) {
    long total = 0;
    for (unsigned i = 0; i < m_config->n_simt_clusters; ++i){
      if(aggregate_stat){
        total += pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i];
      }
      else{
        total += (pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_simt_to_mem[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }

  unsigned long long get_icnt_mem_to_simt(bool aggregate_stat) {
    long total = 0;
    for (unsigned i = 0; i < m_config->n_simt_clusters; ++i) {
      if(aggregate_stat){
        total += pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i];
      }
      
      else{
        total += (pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_mem_to_simt[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }

  power_core_stat_t *pwr_core_stat;
  power_mem_stat_t *pwr_mem_stat;
  float *m_average_pipeline_duty_cycle;
  float *m_active_sms;
  const shader_core_config *m_config;
  const memory_config *m_mem_config;
};

#endif /*POWER_LATENCY_STAT_H*/
