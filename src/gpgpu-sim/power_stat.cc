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

#include "power_stat.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx-stats.h"
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "shader.h"
#include "stat-tool.h"
#include "visualizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

power_mem_stat_t::power_mem_stat_t(const memory_config *mem_config,
                                   const shader_core_config *shdr_config,
                                   memory_stats_t *mem_stats,
                                   shader_core_stats *shdr_stats) {
  assert(mem_config->m_valid);
  m_mem_stats = mem_stats;
  m_config = mem_config;
  m_core_stats = shdr_stats;
  m_core_config = shdr_config;

  init();
}

void power_stat_t::clear(){
  for(unsigned i=0; i< NUM_STAT_IDX; ++i){
    pwr_mem_stat->core_cache_stats[i].clear();
    pwr_mem_stat->l2_cache_stats[i].clear();
    for(unsigned j=0; j<m_config->num_shader(); ++j){
      pwr_core_stat->m_pipeline_duty_cycle[i][j]=0;                
      pwr_core_stat->m_num_decoded_insn[i][j]=0;
      pwr_core_stat->m_num_FPdecoded_insn[i][j]=0;
      pwr_core_stat->m_num_INTdecoded_insn[i][j]=0;
      pwr_core_stat->m_num_storequeued_insn[i][j]=0;
      pwr_core_stat->m_num_loadqueued_insn[i][j]=0;
      pwr_core_stat->m_num_tex_inst[i][j]=0;
      pwr_core_stat->m_num_ialu_acesses[i][j]=0;                   
      pwr_core_stat->m_num_fp_acesses[i][j]=0;                   
      pwr_core_stat->m_num_imul_acesses[i][j]=0;                   
      pwr_core_stat->m_num_imul24_acesses[i][j]=0;                   
      pwr_core_stat->m_num_imul32_acesses[i][j]=0;                   
      pwr_core_stat->m_num_fpmul_acesses[i][j]=0;                   
      pwr_core_stat->m_num_idiv_acesses[i][j]=0;                   
      pwr_core_stat->m_num_fpdiv_acesses[i][j]=0;                   
      pwr_core_stat->m_num_dp_acesses[i][j]=0;                   
      pwr_core_stat->m_num_dpmul_acesses[i][j]=0;                   
      pwr_core_stat->m_num_dpdiv_acesses[i][j]=0;                   
      pwr_core_stat->m_num_tensor_core_acesses[i][j]=0;                   
      pwr_core_stat->m_num_const_acesses[i][j]=0;                   
      pwr_core_stat->m_num_tex_acesses[i][j]=0;                   
      pwr_core_stat->m_num_sp_acesses[i][j]=0;                   
      pwr_core_stat->m_num_sfu_acesses[i][j]=0;                   
      pwr_core_stat->m_num_sqrt_acesses[i][j]=0;                   
      pwr_core_stat->m_num_log_acesses[i][j]=0;                   
      pwr_core_stat->m_num_sin_acesses[i][j]=0;                   
      pwr_core_stat->m_num_exp_acesses[i][j]=0;                   
      pwr_core_stat->m_num_mem_acesses[i][j]=0;                   
      pwr_core_stat->m_num_sp_committed[i][j]=0;
      pwr_core_stat->m_num_sfu_committed[i][j]=0;
      pwr_core_stat->m_num_mem_committed[i][j]=0;
      pwr_core_stat->m_read_regfile_acesses[i][j]=0;
      pwr_core_stat->m_write_regfile_acesses[i][j]=0;
      pwr_core_stat->m_non_rf_operands[i][j]=0;
      pwr_core_stat->m_active_sp_lanes[i][j]=0;
      pwr_core_stat->m_active_sfu_lanes[i][j]=0;
      pwr_core_stat->m_active_exu_threads[i][j]=0;                   
      pwr_core_stat->m_active_exu_warps[i][j]=0;
    }
    for (unsigned j = 0; j < m_mem_config->m_n_mem; ++j) {
      pwr_mem_stat->n_rd[i][j]=0;
      pwr_mem_stat->n_wr[i][j]=0;
      pwr_mem_stat->n_pre[i][j]=0;
    }
  }
}



void power_mem_stat_t::init() {
  shmem_access[CURRENT_STAT_IDX] =
      m_core_stats->gpgpu_n_shmem_bank_access;  // Shared memory access
  shmem_access[PREV_STAT_IDX] =
      (unsigned *)calloc(m_core_config->num_shader(), sizeof(unsigned));

  for (unsigned i = 0; i < NUM_STAT_IDX; ++i) {
    core_cache_stats[i].clear();
    l2_cache_stats[i].clear();

    n_cmd[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_activity[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_nop[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_act[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_pre[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_rd[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_wr[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_wr_WB[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));
    n_req[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));

    // Interconnect stats
    n_mem_to_simt[i] = (long *)calloc(m_core_config->n_simt_clusters,
                                      sizeof(long));  // Counted at SM
    n_simt_to_mem[i] = (long *)calloc(m_core_config->n_simt_clusters,
                                      sizeof(long));  // Counted at SM
  }
}

void power_mem_stat_t::save_stats() {
  core_cache_stats[PREV_STAT_IDX] = core_cache_stats[CURRENT_STAT_IDX];
  l2_cache_stats[PREV_STAT_IDX] = l2_cache_stats[CURRENT_STAT_IDX];

  for (unsigned i = 0; i < m_core_config->num_shader(); ++i) {
    shmem_access[PREV_STAT_IDX][i] =
        shmem_access[CURRENT_STAT_IDX][i];  // Shared memory access
  }

  for (unsigned i = 0; i < m_config->m_n_mem; ++i) {
    n_cmd[PREV_STAT_IDX][i] = n_cmd[CURRENT_STAT_IDX][i];
    n_activity[PREV_STAT_IDX][i] = n_activity[CURRENT_STAT_IDX][i];
    n_nop[PREV_STAT_IDX][i] = n_nop[CURRENT_STAT_IDX][i];
    n_act[PREV_STAT_IDX][i] = n_act[CURRENT_STAT_IDX][i];
    n_pre[PREV_STAT_IDX][i] = n_pre[CURRENT_STAT_IDX][i];
    n_rd[PREV_STAT_IDX][i] = n_rd[CURRENT_STAT_IDX][i];
    n_wr[PREV_STAT_IDX][i] = n_wr[CURRENT_STAT_IDX][i];
    n_wr_WB[PREV_STAT_IDX][i] = n_wr_WB[CURRENT_STAT_IDX][i];
    n_req[PREV_STAT_IDX][i] = n_req[CURRENT_STAT_IDX][i];
  }

  for (unsigned i = 0; i < m_core_config->n_simt_clusters; i++) {
    n_simt_to_mem[PREV_STAT_IDX][i] =
        n_simt_to_mem[CURRENT_STAT_IDX][i];  // Interconnect
    n_mem_to_simt[PREV_STAT_IDX][i] =
        n_mem_to_simt[CURRENT_STAT_IDX][i];  // Interconnect
  }
}

void power_mem_stat_t::visualizer_print(gzFile power_visualizer_file) {}

void power_mem_stat_t::print(FILE *fout) const {
  fprintf(fout, "\n\n==========Power Metrics -- Memory==========\n");
  unsigned total_mem_reads = 0;
  unsigned total_mem_writes = 0;
  for (unsigned i = 0; i < m_config->m_n_mem; ++i) {
    total_mem_reads += n_rd[CURRENT_STAT_IDX][i];
    total_mem_writes += n_wr[CURRENT_STAT_IDX][i] + n_wr_WB[CURRENT_STAT_IDX][i];
  }
  fprintf(fout, "Total memory controller accesses: %u\n",
          total_mem_reads + total_mem_writes);
  fprintf(fout, "Total memory controller reads: %u\n", total_mem_reads);
  fprintf(fout, "Total memory controller writes: %u\n", total_mem_writes);

  fprintf(fout, "Core cache stats:\n");
  core_cache_stats->print_stats(fout);
  fprintf(fout, "L2 cache stats:\n");
  l2_cache_stats->print_stats(fout);
}

power_core_stat_t::power_core_stat_t(const shader_core_config *shader_config,
                                     shader_core_stats *core_stats) {
  assert(shader_config->m_valid);
  m_config = shader_config;
  shader_core_power_stats_pod *pod = this;
  memset(pod, 0, sizeof(shader_core_power_stats_pod));
  m_core_stats = core_stats;

  init();
}

void power_core_stat_t::visualizer_print(gzFile visualizer_file) {}

void power_core_stat_t::print(FILE *fout) {
  // per core statistics
  fprintf(fout, "Power Metrics: \n");
  for (unsigned i = 0; i < m_config->num_shader(); i++) {
        fprintf(fout,"core %u:\n",i);
        fprintf(fout,"\tpipeline duty cycle =%f\n",m_pipeline_duty_cycle[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal Deocded Instructions=%u\n",m_num_decoded_insn[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal FP Deocded Instructions=%u\n",m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal INT Deocded Instructions=%u\n",m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal LOAD Queued Instructions=%u\n",m_num_loadqueued_insn[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal STORE Queued Instructions=%u\n",m_num_storequeued_insn[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal IALU Acesses=%f\n",m_num_ialu_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal FP Acesses=%f\n",m_num_fp_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal DP Acesses=%f\n",m_num_dp_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal IMUL Acesses=%f\n",m_num_imul_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal IMUL24 Acesses=%f\n",m_num_imul24_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal IMUL32 Acesses=%f\n",m_num_imul32_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal IDIV Acesses=%f\n",m_num_idiv_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal FPMUL Acesses=%f\n",m_num_fpmul_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal DPMUL Acesses=%f\n",m_num_dpmul_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal SQRT Acesses=%f\n",m_num_sqrt_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal LOG Acesses=%f\n",m_num_log_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal SIN Acesses=%f\n",m_num_sin_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal EXP Acesses=%f\n",m_num_exp_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal FPDIV Acesses=%f\n",m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal DPDIV Acesses=%f\n",m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal TENSOR Acesses=%f\n",m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal CONST Acesses=%f\n",m_num_const_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal TEX Acesses=%f\n",m_num_tex_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal SFU Acesses=%f\n",m_num_sfu_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal SP Acesses=%f\n",m_num_sp_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal MEM Acesses=%f\n",m_num_mem_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal SFU Commissions=%u\n",m_num_sfu_committed[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal SP Commissions=%u\n",m_num_sp_committed[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal MEM Commissions=%u\n",m_num_mem_committed[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal REG Reads=%u\n",m_read_regfile_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal REG Writes=%u\n",m_write_regfile_acesses[CURRENT_STAT_IDX][i]);
        fprintf(fout,"\tTotal NON REG=%u\n",m_non_rf_operands[CURRENT_STAT_IDX][i]);
  }
}
void power_core_stat_t::init() {
    m_pipeline_duty_cycle[CURRENT_STAT_IDX]=m_core_stats->m_pipeline_duty_cycle;
    m_num_decoded_insn[CURRENT_STAT_IDX]=m_core_stats->m_num_decoded_insn;
    m_num_FPdecoded_insn[CURRENT_STAT_IDX]=m_core_stats->m_num_FPdecoded_insn;
    m_num_INTdecoded_insn[CURRENT_STAT_IDX]=m_core_stats->m_num_INTdecoded_insn;
    m_num_storequeued_insn[CURRENT_STAT_IDX]=m_core_stats->m_num_storequeued_insn;
    m_num_loadqueued_insn[CURRENT_STAT_IDX]=m_core_stats->m_num_loadqueued_insn;
    m_num_ialu_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_ialu_acesses;
    m_num_fp_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_fp_acesses;
    m_num_imul_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_imul_acesses;
    m_num_imul24_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_imul24_acesses;
    m_num_imul32_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_imul32_acesses;
    m_num_fpmul_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_fpmul_acesses;
    m_num_idiv_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_idiv_acesses;
    m_num_fpdiv_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_fpdiv_acesses;
    m_num_dp_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_dp_acesses;
    m_num_dpmul_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_dpmul_acesses;
    m_num_dpdiv_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_dpdiv_acesses;
    m_num_sp_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_sp_acesses;
    m_num_sfu_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_sfu_acesses;
    m_num_sqrt_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_sqrt_acesses;
    m_num_log_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_log_acesses;
    m_num_sin_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_sin_acesses;
    m_num_exp_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_exp_acesses;
    m_num_tensor_core_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_tensor_core_acesses;
    m_num_const_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_const_acesses;
    m_num_tex_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_tex_acesses;
    m_num_mem_acesses[CURRENT_STAT_IDX]=m_core_stats->m_num_mem_acesses;
    m_num_sp_committed[CURRENT_STAT_IDX]=m_core_stats->m_num_sp_committed;
    m_num_sfu_committed[CURRENT_STAT_IDX]=m_core_stats->m_num_sfu_committed;
    m_num_mem_committed[CURRENT_STAT_IDX]=m_core_stats->m_num_mem_committed;
    m_read_regfile_acesses[CURRENT_STAT_IDX]=m_core_stats->m_read_regfile_acesses;
    m_write_regfile_acesses[CURRENT_STAT_IDX]=m_core_stats->m_write_regfile_acesses;
    m_non_rf_operands[CURRENT_STAT_IDX]=m_core_stats->m_non_rf_operands;
    m_active_sp_lanes[CURRENT_STAT_IDX]=m_core_stats->m_active_sp_lanes;
    m_active_sfu_lanes[CURRENT_STAT_IDX]=m_core_stats->m_active_sfu_lanes;
    m_active_exu_threads[CURRENT_STAT_IDX]=m_core_stats->m_active_exu_threads;
    m_active_exu_warps[CURRENT_STAT_IDX]=m_core_stats->m_active_exu_warps;
    m_num_tex_inst[CURRENT_STAT_IDX]=m_core_stats->m_num_tex_inst;

    m_pipeline_duty_cycle[PREV_STAT_IDX]=(float*)calloc(m_config->num_shader(),sizeof(float));
    m_num_decoded_insn[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_FPdecoded_insn[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_INTdecoded_insn[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_storequeued_insn[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_loadqueued_insn[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_tex_inst[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));

    m_num_ialu_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_fp_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_imul_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_imul24_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_imul32_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_fpmul_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_idiv_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_fpdiv_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_dp_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_dpmul_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_dpdiv_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_tensor_core_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_const_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_tex_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_sp_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_sfu_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_sqrt_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_log_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_sin_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_exp_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_mem_acesses[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_num_sp_committed[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_sfu_committed[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_mem_committed[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_read_regfile_acesses[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_write_regfile_acesses[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_non_rf_operands[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_active_sp_lanes[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_active_sfu_lanes[PREV_STAT_IDX]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_active_exu_threads[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));
    m_active_exu_warps[PREV_STAT_IDX]=(double *)calloc(m_config->num_shader(),sizeof(double));


}

void power_core_stat_t::save_stats() {
  for (unsigned i = 0; i < m_config->num_shader(); ++i) {
    m_pipeline_duty_cycle[PREV_STAT_IDX][i]=m_pipeline_duty_cycle[CURRENT_STAT_IDX][i];
    m_num_decoded_insn[PREV_STAT_IDX][i]= m_num_decoded_insn[CURRENT_STAT_IDX][i];
    m_num_FPdecoded_insn[PREV_STAT_IDX][i]=m_num_FPdecoded_insn[CURRENT_STAT_IDX][i];
    m_num_INTdecoded_insn[PREV_STAT_IDX][i]=m_num_INTdecoded_insn[CURRENT_STAT_IDX][i];
    m_num_storequeued_insn[PREV_STAT_IDX][i]=m_num_storequeued_insn[CURRENT_STAT_IDX][i];
    m_num_loadqueued_insn[PREV_STAT_IDX][i]=m_num_loadqueued_insn[CURRENT_STAT_IDX][i];
    m_num_ialu_acesses[PREV_STAT_IDX][i]=m_num_ialu_acesses[CURRENT_STAT_IDX][i];
    m_num_fp_acesses[PREV_STAT_IDX][i]=m_num_fp_acesses[CURRENT_STAT_IDX][i];
    m_num_tex_inst[PREV_STAT_IDX][i]=m_num_tex_inst[CURRENT_STAT_IDX][i];
    m_num_imul_acesses[PREV_STAT_IDX][i]=m_num_imul_acesses[CURRENT_STAT_IDX][i];
    m_num_imul24_acesses[PREV_STAT_IDX][i]=m_num_imul24_acesses[CURRENT_STAT_IDX][i];
    m_num_imul32_acesses[PREV_STAT_IDX][i]=m_num_imul32_acesses[CURRENT_STAT_IDX][i];
    m_num_fpmul_acesses[PREV_STAT_IDX][i]=m_num_fpmul_acesses[CURRENT_STAT_IDX][i];
    m_num_idiv_acesses[PREV_STAT_IDX][i]=m_num_idiv_acesses[CURRENT_STAT_IDX][i];
    m_num_fpdiv_acesses[PREV_STAT_IDX][i]=m_num_fpdiv_acesses[CURRENT_STAT_IDX][i];
    m_num_sp_acesses[PREV_STAT_IDX][i]=m_num_sp_acesses[CURRENT_STAT_IDX][i];
    m_num_sfu_acesses[PREV_STAT_IDX][i]=m_num_sfu_acesses[CURRENT_STAT_IDX][i];
    m_num_sqrt_acesses[PREV_STAT_IDX][i]=m_num_sqrt_acesses[CURRENT_STAT_IDX][i];
    m_num_log_acesses[PREV_STAT_IDX][i]=m_num_log_acesses[CURRENT_STAT_IDX][i];
    m_num_sin_acesses[PREV_STAT_IDX][i]=m_num_sin_acesses[CURRENT_STAT_IDX][i];
    m_num_exp_acesses[PREV_STAT_IDX][i]=m_num_exp_acesses[CURRENT_STAT_IDX][i];
    m_num_dp_acesses[PREV_STAT_IDX][i]=m_num_dp_acesses[CURRENT_STAT_IDX][i];
    m_num_dpmul_acesses[PREV_STAT_IDX][i]=m_num_dpmul_acesses[CURRENT_STAT_IDX][i];
    m_num_dpdiv_acesses[PREV_STAT_IDX][i]=m_num_dpdiv_acesses[CURRENT_STAT_IDX][i];
    m_num_tensor_core_acesses[PREV_STAT_IDX][i]=m_num_tensor_core_acesses[CURRENT_STAT_IDX][i];
    m_num_const_acesses[PREV_STAT_IDX][i]=m_num_const_acesses[CURRENT_STAT_IDX][i];
    m_num_tex_acesses[PREV_STAT_IDX][i]=m_num_tex_acesses[CURRENT_STAT_IDX][i];
    m_num_mem_acesses[PREV_STAT_IDX][i]=m_num_mem_acesses[CURRENT_STAT_IDX][i];
    m_num_sp_committed[PREV_STAT_IDX][i]=m_num_sp_committed[CURRENT_STAT_IDX][i];
    m_num_sfu_committed[PREV_STAT_IDX][i]=m_num_sfu_committed[CURRENT_STAT_IDX][i];
    m_num_mem_committed[PREV_STAT_IDX][i]=m_num_mem_committed[CURRENT_STAT_IDX][i];
    m_read_regfile_acesses[PREV_STAT_IDX][i]=m_read_regfile_acesses[CURRENT_STAT_IDX][i];
    m_write_regfile_acesses[PREV_STAT_IDX][i]=m_write_regfile_acesses[CURRENT_STAT_IDX][i];
    m_non_rf_operands[PREV_STAT_IDX][i]=m_non_rf_operands[CURRENT_STAT_IDX][i];
    m_active_sp_lanes[PREV_STAT_IDX][i]=m_active_sp_lanes[CURRENT_STAT_IDX][i];
    m_active_sfu_lanes[PREV_STAT_IDX][i]=m_active_sfu_lanes[CURRENT_STAT_IDX][i];
    m_active_exu_threads[PREV_STAT_IDX][i]=m_active_exu_threads[CURRENT_STAT_IDX][i];
    m_active_exu_warps[PREV_STAT_IDX][i]=m_active_exu_warps[CURRENT_STAT_IDX][i];
  }
}

power_stat_t::power_stat_t(const shader_core_config *shader_config,
                           float *average_pipeline_duty_cycle,
                           float *active_sms, shader_core_stats *shader_stats,
                           const memory_config *mem_config,
                           memory_stats_t *memory_stats) {
  assert(shader_config->m_valid);
  assert(mem_config->m_valid);
  pwr_core_stat = new power_core_stat_t(shader_config, shader_stats);
  pwr_mem_stat = new power_mem_stat_t(mem_config, shader_config, memory_stats,
                                      shader_stats);
  m_average_pipeline_duty_cycle = average_pipeline_duty_cycle;
  m_active_sms = active_sms;
  m_config = shader_config;
  m_mem_config = mem_config;
  l1r_hits_kernel = 0;
  l1r_misses_kernel = 0;
  l1w_hits_kernel = 0;
  l1w_misses_kernel = 0;
  shared_accesses_kernel = 0;
  cc_accesses_kernel = 0;
  dram_rd_kernel = 0;
  dram_wr_kernel = 0;
  dram_pre_kernel = 0;
  l1i_hits_kernel =0;
  l1i_misses_kernel =0;
  l2r_hits_kernel =0;
  l2r_misses_kernel =0;
  l2w_hits_kernel =0;
  l2w_misses_kernel =0;
  noc_tr_kernel = 0;
  noc_rc_kernel = 0;

  tot_inst_execution = 0;
  tot_int_inst_execution = 0;
  tot_fp_inst_execution = 0;
  commited_inst_execution = 0;
  ialu_acc_execution = 0;
  imul24_acc_execution = 0;
  imul32_acc_execution = 0;
  imul_acc_execution = 0;
  idiv_acc_execution = 0;
  dp_acc_execution = 0;
  dpmul_acc_execution = 0;
  dpdiv_acc_execution = 0;
  fp_acc_execution = 0;
  fpmul_acc_execution = 0;
  fpdiv_acc_execution = 0;
  sqrt_acc_execution = 0;
  log_acc_execution = 0;
  sin_acc_execution = 0;
  exp_acc_execution = 0;
  tensor_acc_execution = 0;
  tex_acc_execution = 0;
  tot_fpu_acc_execution = 0;
  tot_sfu_acc_execution = 0;
  tot_threads_acc_execution = 0;
  tot_warps_acc_execution = 0;
  sp_active_lanes_execution = 0;
  sfu_active_lanes_execution = 0;
}

void power_stat_t::visualizer_print(gzFile visualizer_file) {
  pwr_core_stat->visualizer_print(visualizer_file);
  pwr_mem_stat->visualizer_print(visualizer_file);
}

void power_stat_t::print(FILE *fout) const {
  fprintf(fout, "average_pipeline_duty_cycle=%f\n",
          *m_average_pipeline_duty_cycle);
  pwr_core_stat->print(fout);
  pwr_mem_stat->print(fout);
}
