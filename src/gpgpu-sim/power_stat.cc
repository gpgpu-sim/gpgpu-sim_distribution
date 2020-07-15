// Copyright (c) 2009-2011, Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington
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

void power_mem_stat_t::init() {
  shmem_read_access[CURRENT_STAT_IDX] =
      m_core_stats->gpgpu_n_shmem_bank_access;  // Shared memory access
  shmem_read_access[PREV_STAT_IDX] =
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
    shmem_read_access[PREV_STAT_IDX][i] =
        shmem_read_access[CURRENT_STAT_IDX][i];  // Shared memory access
  }

  for (unsigned i = 0; i < m_config->m_n_mem; ++i) {
    n_cmd[PREV_STAT_IDX][i] = n_cmd[CURRENT_STAT_IDX][i];
    n_activity[PREV_STAT_IDX][i] = n_activity[CURRENT_STAT_IDX][i];
    n_nop[PREV_STAT_IDX][i] = n_nop[CURRENT_STAT_IDX][i];
    n_act[PREV_STAT_IDX][i] = n_act[CURRENT_STAT_IDX][i];
    n_pre[PREV_STAT_IDX][i] = n_pre[CURRENT_STAT_IDX][i];
    n_rd[PREV_STAT_IDX][i] = n_rd[CURRENT_STAT_IDX][i];
    n_wr[PREV_STAT_IDX][i] = n_wr[CURRENT_STAT_IDX][i];
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
    total_mem_writes += n_wr[CURRENT_STAT_IDX][i];
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
    fprintf(fout, "core %u:\n", i);
    fprintf(fout, "\tpipeline duty cycle =%f\n",
            m_pipeline_duty_cycle[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal Deocded Instructions=%u\n",
            m_num_decoded_insn[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal FP Deocded Instructions=%u\n",
            m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal INT Deocded Instructions=%u\n",
            m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal LOAD Queued Instructions=%u\n",
            m_num_loadqueued_insn[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal STORE Queued Instructions=%u\n",
            m_num_storequeued_insn[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal IALU Acesses=%u\n",
            m_num_ialu_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal FP Acesses=%u\n",
            m_num_fp_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal IMUL Acesses=%u\n",
            m_num_imul_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal IMUL24 Acesses=%u\n",
            m_num_imul24_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal IMUL32 Acesses=%u\n",
            m_num_imul32_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal IDIV Acesses=%u\n",
            m_num_idiv_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal FPMUL Acesses=%u\n",
            m_num_fpmul_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal SFU Acesses=%u\n",
            m_num_trans_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal FPDIV Acesses=%u\n",
            m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal SFU Acesses=%u\n",
            m_num_sfu_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal SP Acesses=%u\n",
            m_num_sp_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal MEM Acesses=%u\n",
            m_num_mem_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal SFU Commissions=%u\n",
            m_num_sfu_committed[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal SP Commissions=%u\n",
            m_num_sp_committed[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal MEM Commissions=%u\n",
            m_num_mem_committed[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal REG Reads=%u\n",
            m_read_regfile_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal REG Writes=%u\n",
            m_write_regfile_acesses[CURRENT_STAT_IDX][i]);
    fprintf(fout, "\tTotal NON REG=%u\n",
            m_non_rf_operands[CURRENT_STAT_IDX][i]);
  }
}
void power_core_stat_t::init() {
  m_pipeline_duty_cycle[CURRENT_STAT_IDX] = m_core_stats->m_pipeline_duty_cycle;
  m_num_decoded_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_decoded_insn;
  m_num_FPdecoded_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_FPdecoded_insn;
  m_num_INTdecoded_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_INTdecoded_insn;
  m_num_storequeued_insn[CURRENT_STAT_IDX] =
      m_core_stats->m_num_storequeued_insn;
  m_num_loadqueued_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_loadqueued_insn;
  m_num_ialu_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_ialu_acesses;
  m_num_fp_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_fp_acesses;
  m_num_imul_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_imul_acesses;
  m_num_imul24_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_imul24_acesses;
  m_num_imul32_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_imul32_acesses;
  m_num_fpmul_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_fpmul_acesses;
  m_num_idiv_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_idiv_acesses;
  m_num_fpdiv_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_fpdiv_acesses;
  m_num_sp_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_sp_acesses;
  m_num_sfu_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_sfu_acesses;
  m_num_trans_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_trans_acesses;
  m_num_mem_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_mem_acesses;
  m_num_sp_committed[CURRENT_STAT_IDX] = m_core_stats->m_num_sp_committed;
  m_num_sfu_committed[CURRENT_STAT_IDX] = m_core_stats->m_num_sfu_committed;
  m_num_mem_committed[CURRENT_STAT_IDX] = m_core_stats->m_num_mem_committed;
  m_read_regfile_acesses[CURRENT_STAT_IDX] =
      m_core_stats->m_read_regfile_acesses;
  m_write_regfile_acesses[CURRENT_STAT_IDX] =
      m_core_stats->m_write_regfile_acesses;
  m_non_rf_operands[CURRENT_STAT_IDX] = m_core_stats->m_non_rf_operands;
  m_active_sp_lanes[CURRENT_STAT_IDX] = m_core_stats->m_active_sp_lanes;
  m_active_sfu_lanes[CURRENT_STAT_IDX] = m_core_stats->m_active_sfu_lanes;
  m_num_tex_inst[CURRENT_STAT_IDX] = m_core_stats->m_num_tex_inst;

  m_pipeline_duty_cycle[PREV_STAT_IDX] =
      (float *)calloc(m_config->num_shader(), sizeof(float));
  m_num_decoded_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_FPdecoded_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_INTdecoded_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_storequeued_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_loadqueued_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_ialu_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_fp_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_tex_inst[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_imul_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_imul24_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_imul32_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_fpmul_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_idiv_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_fpdiv_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_sp_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_sfu_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_trans_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_mem_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_sp_committed[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_sfu_committed[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_num_mem_committed[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_read_regfile_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_write_regfile_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_non_rf_operands[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_active_sp_lanes[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
  m_active_sfu_lanes[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));
}

void power_core_stat_t::save_stats() {
  for (unsigned i = 0; i < m_config->num_shader(); ++i) {
    m_pipeline_duty_cycle[PREV_STAT_IDX][i] =
        m_pipeline_duty_cycle[CURRENT_STAT_IDX][i];
    m_num_decoded_insn[PREV_STAT_IDX][i] =
        m_num_decoded_insn[CURRENT_STAT_IDX][i];
    m_num_FPdecoded_insn[PREV_STAT_IDX][i] =
        m_num_FPdecoded_insn[CURRENT_STAT_IDX][i];
    m_num_INTdecoded_insn[PREV_STAT_IDX][i] =
        m_num_INTdecoded_insn[CURRENT_STAT_IDX][i];
    m_num_storequeued_insn[PREV_STAT_IDX][i] =
        m_num_storequeued_insn[CURRENT_STAT_IDX][i];
    m_num_loadqueued_insn[PREV_STAT_IDX][i] =
        m_num_loadqueued_insn[CURRENT_STAT_IDX][i];
    m_num_ialu_acesses[PREV_STAT_IDX][i] =
        m_num_ialu_acesses[CURRENT_STAT_IDX][i];
    m_num_fp_acesses[PREV_STAT_IDX][i] = m_num_fp_acesses[CURRENT_STAT_IDX][i];
    m_num_tex_inst[PREV_STAT_IDX][i] = m_num_tex_inst[CURRENT_STAT_IDX][i];
    m_num_imul_acesses[PREV_STAT_IDX][i] =
        m_num_imul_acesses[CURRENT_STAT_IDX][i];
    m_num_imul24_acesses[PREV_STAT_IDX][i] =
        m_num_imul24_acesses[CURRENT_STAT_IDX][i];
    m_num_imul32_acesses[PREV_STAT_IDX][i] =
        m_num_imul32_acesses[CURRENT_STAT_IDX][i];
    m_num_fpmul_acesses[PREV_STAT_IDX][i] =
        m_num_fpmul_acesses[CURRENT_STAT_IDX][i];
    m_num_idiv_acesses[PREV_STAT_IDX][i] =
        m_num_idiv_acesses[CURRENT_STAT_IDX][i];
    m_num_fpdiv_acesses[PREV_STAT_IDX][i] =
        m_num_fpdiv_acesses[CURRENT_STAT_IDX][i];
    m_num_sp_acesses[PREV_STAT_IDX][i] = m_num_sp_acesses[CURRENT_STAT_IDX][i];
    m_num_sfu_acesses[PREV_STAT_IDX][i] =
        m_num_sfu_acesses[CURRENT_STAT_IDX][i];
    m_num_trans_acesses[PREV_STAT_IDX][i] =
        m_num_trans_acesses[CURRENT_STAT_IDX][i];
    m_num_mem_acesses[PREV_STAT_IDX][i] =
        m_num_mem_acesses[CURRENT_STAT_IDX][i];
    m_num_sp_committed[PREV_STAT_IDX][i] =
        m_num_sp_committed[CURRENT_STAT_IDX][i];
    m_num_sfu_committed[PREV_STAT_IDX][i] =
        m_num_sfu_committed[CURRENT_STAT_IDX][i];
    m_num_mem_committed[PREV_STAT_IDX][i] =
        m_num_mem_committed[CURRENT_STAT_IDX][i];
    m_read_regfile_acesses[PREV_STAT_IDX][i] =
        m_read_regfile_acesses[CURRENT_STAT_IDX][i];
    m_write_regfile_acesses[PREV_STAT_IDX][i] =
        m_write_regfile_acesses[CURRENT_STAT_IDX][i];
    m_non_rf_operands[PREV_STAT_IDX][i] =
        m_non_rf_operands[CURRENT_STAT_IDX][i];
    m_active_sp_lanes[PREV_STAT_IDX][i] =
        m_active_sp_lanes[CURRENT_STAT_IDX][i];
    m_active_sfu_lanes[PREV_STAT_IDX][i] =
        m_active_sfu_lanes[CURRENT_STAT_IDX][i];
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
