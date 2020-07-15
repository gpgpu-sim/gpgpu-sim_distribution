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

#include "power_interface.h"

void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst) {
  wrapper->init_mcpat(
      config.g_power_config_name, config.g_power_filename,
      config.g_power_trace_filename, config.g_metric_trace_filename,
      config.g_steady_state_tracking_filename,
      config.g_power_simulation_enabled, config.g_power_trace_enabled,
      config.g_steady_power_levels_enabled, config.g_power_per_cycle_dump,
      config.gpu_steady_power_deviation, config.gpu_steady_min_period,
      config.g_power_trace_zlevel, tot_inst + inst, stat_sample_freq);
}

void mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst) {
  static bool mcpat_init = true;

  if (mcpat_init) {  // If first cycle, don't have any power numbers yet
    mcpat_init = false;
    return;
  }

  if ((tot_cycle + cycle) % stat_sample_freq == 0) {
    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, stat_sample_freq,
        stat_sample_freq, power_stats->get_total_inst(),
        power_stats->get_total_int_inst(), power_stats->get_total_fp_inst(),
        power_stats->get_l1d_read_accesses(),
        power_stats->get_l1d_write_accesses(),
        power_stats->get_committed_inst());

    // Single RF for both int and fp ops
    wrapper->set_regfile_power(power_stats->get_regfile_reads(),
                               power_stats->get_regfile_writes(),
                               power_stats->get_non_regfile_operands());

    // Instruction cache stats
    wrapper->set_icache_power(power_stats->get_inst_c_hits(),
                              power_stats->get_inst_c_misses());

    // Constant Cache, shared memory, texture cache
    wrapper->set_ccache_power(power_stats->get_constant_c_hits(),
                              power_stats->get_constant_c_misses());
    wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
                              power_stats->get_texture_c_misses());
    wrapper->set_shrd_mem_power(power_stats->get_shmem_read_access());

    wrapper->set_l1cache_power(
        power_stats->get_l1d_read_hits(), power_stats->get_l1d_read_misses(),
        power_stats->get_l1d_write_hits(), power_stats->get_l1d_write_misses());

    wrapper->set_l2cache_power(
        power_stats->get_l2_read_hits(), power_stats->get_l2_read_misses(),
        power_stats->get_l2_write_hits(), power_stats->get_l2_write_misses());

    float active_sms = (*power_stats->m_active_sms) / stat_sample_freq;
    float num_cores = shdr_config->num_shader();
    float num_idle_core = num_cores - active_sms;
    wrapper->set_idle_core_power(num_idle_core);

    // pipeline power - pipeline_duty_cycle *= percent_active_sms;
    float pipeline_duty_cycle =
        ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) <
         0.8)
            ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq)
            : 0.8;
    wrapper->set_duty_cycle_power(pipeline_duty_cycle);

    // Memory Controller
    wrapper->set_mem_ctrl_power(power_stats->get_dram_rd(),
                                power_stats->get_dram_wr(),
                                power_stats->get_dram_pre());

    // Execution pipeline accesses
    // FPU (SP) accesses, Integer ALU (not present in Tesla), Sfu accesses
    wrapper->set_exec_unit_power(power_stats->get_tot_fpu_accessess(),
                                 power_stats->get_ialu_accessess(),
                                 power_stats->get_tot_sfu_accessess());

    // Average active lanes for sp and sfu pipelines
    float avg_sp_active_lanes =
        (power_stats->get_sp_active_lanes()) / stat_sample_freq;
    float avg_sfu_active_lanes =
        (power_stats->get_sfu_active_lanes()) / stat_sample_freq;
    assert(avg_sp_active_lanes <= 32);
    assert(avg_sfu_active_lanes <= 32);
    wrapper->set_active_lanes_power(
        (power_stats->get_sp_active_lanes()) / stat_sample_freq,
        (power_stats->get_sfu_active_lanes()) / stat_sample_freq);

    double n_icnt_simt_to_mem =
        (double)
            power_stats->get_icnt_simt_to_mem();  // # flits from SIMT clusters
                                                  // to memory partitions
    double n_icnt_mem_to_simt =
        (double)
            power_stats->get_icnt_mem_to_simt();  // # flits from memory
                                                  // partitions to SIMT clusters
    wrapper->set_NoC_power(
        n_icnt_mem_to_simt,
        n_icnt_simt_to_mem);  // Number of flits traversing the interconnect

    wrapper->compute();

    wrapper->update_components_power();
    wrapper->print_trace_files();
    power_stats->save_stats();

    wrapper->detect_print_steady_state(0, tot_inst + inst);

    wrapper->power_metrics_calculations();

    wrapper->dump();
  }
  // wrapper->close_files();
}

void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper) {
  wrapper->reset_counters();
}
