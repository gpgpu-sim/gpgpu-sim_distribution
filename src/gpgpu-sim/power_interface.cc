// Copyright (c) 2009-2021, Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington, Vijay Kandiah, Nikos Hardavellas, 
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
      config.g_power_trace_zlevel, tot_inst + inst, stat_sample_freq,  
      config.g_power_simulation_mode, 
      config.g_dvfs_enabled,
      config.get_core_freq()/1000000,
      config.num_shader());
}

void mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst, bool dvfs_enabled) {
  static bool mcpat_init = true;

  if (mcpat_init) {  // If first cycle, don't have any power numbers yet
    mcpat_init = false;
    return;
  }

  if ((tot_cycle + cycle) % stat_sample_freq == 0) {
    if(dvfs_enabled){
      wrapper->set_model_voltage(1); //performance model needs to support this.
    }

    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, stat_sample_freq,
        stat_sample_freq, power_stats->get_total_inst(0),
        power_stats->get_total_int_inst(0), power_stats->get_total_fp_inst(0),
        power_stats->get_l1d_read_accesses(0),
        power_stats->get_l1d_write_accesses(0),
        power_stats->get_committed_inst(0));

    // Single RF for both int and fp ops
    wrapper->set_regfile_power(power_stats->get_regfile_reads(0),
                               power_stats->get_regfile_writes(0),
                               power_stats->get_non_regfile_operands(0));

    // Instruction cache stats
    wrapper->set_icache_power(power_stats->get_inst_c_hits(0),
                              power_stats->get_inst_c_misses(0));

    // Constant Cache, shared memory, texture cache
    wrapper->set_ccache_power(power_stats->get_const_accessess(0), 0); //assuming all HITS in constant cache for now
    wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
                              power_stats->get_texture_c_misses());
    wrapper->set_shrd_mem_power(power_stats->get_shmem_access(0));

    wrapper->set_l1cache_power(
        power_stats->get_l1d_read_hits(0), power_stats->get_l1d_read_misses(0),
        power_stats->get_l1d_write_hits(0), power_stats->get_l1d_write_misses(0));

    wrapper->set_l2cache_power(
        power_stats->get_l2_read_hits(0), power_stats->get_l2_read_misses(0),
        power_stats->get_l2_write_hits(0), power_stats->get_l2_write_misses(0));

    float active_sms = (*power_stats->m_active_sms) / stat_sample_freq;
    float num_cores = shdr_config->num_shader();
    float num_idle_core = num_cores - active_sms;
    wrapper->set_num_cores(num_cores);
    wrapper->set_idle_core_power(num_idle_core);

    // pipeline power - pipeline_duty_cycle *= percent_active_sms;
    float pipeline_duty_cycle =
        ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) <
         0.8)
            ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq)
            : 0.8;
    wrapper->set_duty_cycle_power(pipeline_duty_cycle);

    // Memory Controller
    wrapper->set_mem_ctrl_power(power_stats->get_dram_rd(0),
                                power_stats->get_dram_wr(0),
                                power_stats->get_dram_pre(0));

    // Execution pipeline accesses
    // FPU (SP) accesses, Integer ALU (not present in Tesla), Sfu accesses

    wrapper->set_int_accesses(power_stats->get_ialu_accessess(0), 
                              power_stats->get_intmul24_accessess(0), 
                              power_stats->get_intmul32_accessess(0), 
                              power_stats->get_intmul_accessess(0), 
                              power_stats->get_intdiv_accessess(0));

    wrapper->set_dp_accesses(power_stats->get_dp_accessess(0), 
                              power_stats->get_dpmul_accessess(0), 
                              power_stats->get_dpdiv_accessess(0));

    wrapper->set_fp_accesses(power_stats->get_fp_accessess(0), 
                            power_stats->get_fpmul_accessess(0), 
                            power_stats->get_fpdiv_accessess(0));

    wrapper->set_trans_accesses(power_stats->get_sqrt_accessess(0), 
                                power_stats->get_log_accessess(0), 
                                power_stats->get_sin_accessess(0), 
                                power_stats->get_exp_accessess(0));

    wrapper->set_tensor_accesses(power_stats->get_tensor_accessess(0));

    wrapper->set_tex_accesses(power_stats->get_tex_accessess(0));

    wrapper->set_exec_unit_power(power_stats->get_tot_fpu_accessess(0),
                                 power_stats->get_ialu_accessess(0),
                                 power_stats->get_tot_sfu_accessess(0));

    wrapper->set_avg_active_threads(power_stats->get_active_threads(0));

    // Average active lanes for sp and sfu pipelines
    float avg_sp_active_lanes =
        (power_stats->get_sp_active_lanes()) / stat_sample_freq;
    float avg_sfu_active_lanes =
        (power_stats->get_sfu_active_lanes()) / stat_sample_freq;
    if(avg_sp_active_lanes >32.0 )
      avg_sp_active_lanes = 32.0;
    if(avg_sfu_active_lanes >32.0 )
      avg_sfu_active_lanes = 32.0;
    assert(avg_sp_active_lanes <= 32);
    assert(avg_sfu_active_lanes <= 32);
    wrapper->set_active_lanes_power(avg_sp_active_lanes, avg_sfu_active_lanes);

    double n_icnt_simt_to_mem =
        (double)
            power_stats->get_icnt_simt_to_mem(0);  // # flits from SIMT clusters
                                                  // to memory partitions
    double n_icnt_mem_to_simt =
        (double)
            power_stats->get_icnt_mem_to_simt(0);  // # flits from memory
                                                  // partitions to SIMT clusters
    wrapper->set_NoC_power(n_icnt_mem_to_simt + n_icnt_simt_to_mem);  // Number of flits traversing the interconnect

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

bool parse_hw_file(char* hwpowerfile, bool find_target_kernel, vector<string> &hw_data, char* benchname, std::string executed_kernelname){
  fstream hw_file;
  hw_file.open(hwpowerfile, ios::in);
  string line, word, temp;
  while(!hw_file.eof()){
    hw_data.clear();
    getline(hw_file, line);
    stringstream s(line);
    while (getline(s,word,',')){
      hw_data.push_back(word);
    }
    if(hw_data[HW_BENCH_NAME] == std::string(benchname)){
      if(find_target_kernel){
        if(hw_data[HW_KERNEL_NAME] == ""){
          hw_file.close();
          return true;
        }
        else{
          if(hw_data[HW_KERNEL_NAME] == executed_kernelname){
            hw_file.close();
            return true;
          }
        }
      }
      else{
        hw_file.close();
        return true;
      }
    } 
  }
  hw_file.close();
  return false;
}


void calculate_hw_mcpat(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst, int power_simulation_mode, bool dvfs_enabled, char* hwpowerfile, 
                 char* benchname, std::string executed_kernelname, 
                 const bool *accelwattch_hybrid_configuration, bool aggregate_power_stats){

  /* Reading HW data from CSV file */

  vector<string> hw_data;
  bool kernel_found = false;
  kernel_found = parse_hw_file(hwpowerfile, true, hw_data, benchname, executed_kernelname); //Searching for matching executed_kernelname.
  if(!kernel_found)
    kernel_found = parse_hw_file(hwpowerfile, false, hw_data, benchname, executed_kernelname); //Searching for any kernel with same benchname. 
  assert("Could not find perf stats for the target benchmark in hwpowerfile.\n" && (kernel_found));
  unsigned perf_cycles = static_cast<unsigned int>(std::stod(hw_data[HW_CYCLES]) + 0.5);
  if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_CYCLES]))
    perf_cycles = cycle;
  wrapper->init_mcpat_hw_mode(perf_cycles); //total PERF MODEL cycles for current kernel

  if(dvfs_enabled){
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_VOLTAGE])) 
      wrapper->set_model_voltage(1); //performance model needs to support this
    else  
      wrapper->set_model_voltage(std::stod(hw_data[HW_VOLTAGE])); //performance model needs to support this
  }

  double l1_read_hits = std::stod(hw_data[HW_L1_RH]);
  double l1_read_misses = std::stod(hw_data[HW_L1_RM]);
  double l1_write_hits = std::stod(hw_data[HW_L1_WH]);
  double l1_write_misses = std::stod(hw_data[HW_L1_WM]);

  if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L1_RH]))
    l1_read_hits = power_stats->get_l1d_read_hits(1) - power_stats->l1r_hits_kernel;
  if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L1_RM]))
    l1_read_misses = power_stats->get_l1d_read_misses(1) - power_stats->l1r_misses_kernel;
  if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L1_WH]))
    l1_write_hits = power_stats->get_l1d_write_hits(1) - power_stats->l1w_hits_kernel;
  if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L1_WM]))
    l1_write_misses = power_stats->get_l1d_write_misses(1) - power_stats->l1w_misses_kernel;

    if(aggregate_power_stats){
      power_stats->tot_inst_execution += power_stats->get_total_inst(1);
      power_stats->tot_int_inst_execution +=  power_stats->get_total_int_inst(1);
      power_stats->tot_fp_inst_execution +=  power_stats->get_total_fp_inst(1);
      power_stats->commited_inst_execution += power_stats->get_committed_inst(1);
      wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, cycle, //TODO: core.[0] cycles counts don't matter, remove this
        cycle, power_stats->tot_inst_execution,
        power_stats->tot_int_inst_execution, power_stats->tot_fp_inst_execution,
        l1_read_hits + l1_read_misses,
        l1_write_hits + l1_write_misses,
        power_stats->commited_inst_execution);
    }
    else{
    wrapper->set_inst_power(
        shdr_config->gpgpu_clock_gated_lanes, cycle, //TODO: core.[0] cycles counts don't matter, remove this
        cycle, power_stats->get_total_inst(1),
        power_stats->get_total_int_inst(1), power_stats->get_total_fp_inst(1),
        l1_read_hits + l1_read_misses,
        l1_write_hits + l1_write_misses,
        power_stats->get_committed_inst(1));
    }

    // Single RF for both int and fp ops -- activity factor set to 0 for Accelwattch HW and Accelwattch Hybrid because no HW Perf Stats for register files
    wrapper->set_regfile_power(power_stats->get_regfile_reads(1),
                               power_stats->get_regfile_writes(1),
                               power_stats->get_non_regfile_operands(1));

    // Instruction cache stats -- activity factor set to 0 for Accelwattch HW and Accelwattch Hybrid because no HW Perf Stats for instruction cache
    wrapper->set_icache_power(power_stats->get_inst_c_hits(1) - power_stats->l1i_hits_kernel,
                              power_stats->get_inst_c_misses(1) - power_stats->l1i_misses_kernel);

    // Constant Cache, shared memory, texture cache
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_CC_ACC]))
      wrapper->set_ccache_power(power_stats->get_const_accessess(1) - power_stats->cc_accesses_kernel, 0); //assuming all HITS in constant cache for now
    else  
      wrapper->set_ccache_power(std::stod(hw_data[HW_CC_ACC]), 0); //assuming all HITS in constant cache for now

    
    // wrapper->set_tcache_power(power_stats->get_texture_c_hits(),
    //                           power_stats->get_texture_c_misses());

    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_SHRD_ACC]))
      wrapper->set_shrd_mem_power(power_stats->get_shmem_access(1) - power_stats->shared_accesses_kernel);
    else  
      wrapper->set_shrd_mem_power(std::stod(hw_data[HW_SHRD_ACC]));

    wrapper->set_l1cache_power( l1_read_hits,  l1_read_misses, l1_write_hits,  l1_write_misses);

    double l2_read_hits = std::stod(hw_data[HW_L2_RH]);
    double l2_read_misses = std::stod(hw_data[HW_L2_RM]);
    double l2_write_hits = std::stod(hw_data[HW_L2_WH]);
    double l2_write_misses = std::stod(hw_data[HW_L2_WM]);

    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L2_RH]))
      l2_read_hits = power_stats->get_l2_read_hits(1) - power_stats->l2r_hits_kernel;
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L2_RM]))
      l2_read_misses = power_stats->get_l2_read_misses(1)  - power_stats->l2r_misses_kernel;
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L2_WH]))
      l2_write_hits = power_stats->get_l2_write_hits(1) - power_stats->l2w_hits_kernel;
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_L2_WM]))
      l2_write_misses = power_stats->get_l2_write_misses(1) - power_stats->l2w_misses_kernel;

    wrapper->set_l2cache_power(l2_read_hits, l2_read_misses, l2_write_hits, l2_write_misses);
    
    float active_sms = (*power_stats->m_active_sms) / stat_sample_freq;
    float num_cores = shdr_config->num_shader();
    float num_idle_core = num_cores - active_sms;
    wrapper->set_num_cores(num_cores);
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_NUM_SM_IDLE]))
      wrapper->set_idle_core_power(num_idle_core);
    else 
      wrapper->set_idle_core_power(std::stod(hw_data[HW_NUM_SM_IDLE])); 

    float pipeline_duty_cycle =
        ((*power_stats->m_average_pipeline_duty_cycle / (stat_sample_freq)) <
         0.8)
            ? ((*power_stats->m_average_pipeline_duty_cycle) / stat_sample_freq)
            : 0.8;
    
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_PIPE_DUTY]))
      wrapper->set_duty_cycle_power(pipeline_duty_cycle);
    else
      wrapper->set_duty_cycle_power(std::stod(hw_data[HW_PIPE_DUTY]));

    // Memory Controller
  
    double dram_reads = std::stod(hw_data[HW_DRAM_RD]);
    double dram_writes = std::stod(hw_data[HW_DRAM_WR]);
    double dram_pre = 0;
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_DRAM_RD]))
      dram_reads = power_stats->get_dram_rd(1) - power_stats->dram_rd_kernel;
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_DRAM_WR]))
      dram_writes = power_stats->get_dram_wr(1) - power_stats->dram_wr_kernel;
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_DRAM_RD]))
      dram_pre = power_stats->get_dram_pre(1) - power_stats->dram_pre_kernel;


    wrapper->set_mem_ctrl_power(dram_reads, dram_writes, dram_pre);

    if(aggregate_power_stats){
      power_stats->ialu_acc_execution += power_stats->get_ialu_accessess(1);
      power_stats->imul24_acc_execution += power_stats->get_intmul24_accessess(1);
      power_stats->imul32_acc_execution += power_stats->get_intmul32_accessess(1);
      power_stats->imul_acc_execution += power_stats->get_intmul_accessess(1);
      power_stats->idiv_acc_execution += power_stats->get_intdiv_accessess(1);
      power_stats->dp_acc_execution += power_stats->get_dp_accessess(1);
      power_stats->dpmul_acc_execution += power_stats->get_dpmul_accessess(1);
      power_stats->dpdiv_acc_execution += power_stats->get_dpdiv_accessess(1);
      power_stats->fp_acc_execution += power_stats->get_fp_accessess(1);
      power_stats->fpmul_acc_execution += power_stats->get_fpmul_accessess(1);
      power_stats->fpdiv_acc_execution += power_stats->get_fpdiv_accessess(1);
      power_stats->sqrt_acc_execution += power_stats->get_sqrt_accessess(1);
      power_stats->log_acc_execution += power_stats->get_log_accessess(1);
      power_stats->sin_acc_execution += power_stats->get_sin_accessess(1);
      power_stats->exp_acc_execution += power_stats->get_exp_accessess(1);
      power_stats->tensor_acc_execution += power_stats->get_tensor_accessess(1);
      power_stats->tex_acc_execution += power_stats->get_tex_accessess(1);
      power_stats->tot_fpu_acc_execution += power_stats->get_tot_fpu_accessess(1);
      power_stats->tot_sfu_acc_execution += power_stats->get_tot_sfu_accessess(1);
      power_stats->tot_threads_acc_execution += power_stats->get_tot_threads_kernel(1);
      power_stats->tot_warps_acc_execution += power_stats->get_tot_warps_kernel(1);
      
      power_stats->sp_active_lanes_execution += (power_stats->get_sp_active_lanes() * shdr_config->num_shader() * shdr_config->gpgpu_num_sp_units);
      power_stats->sfu_active_lanes_execution += (power_stats->get_sfu_active_lanes() * shdr_config->num_shader() * shdr_config->gpgpu_num_sp_units);

      wrapper->set_int_accesses(power_stats->ialu_acc_execution, 
                                power_stats->imul24_acc_execution, 
                                power_stats->imul32_acc_execution, 
                                power_stats->imul_acc_execution, 
                                power_stats->idiv_acc_execution);

      wrapper->set_dp_accesses(power_stats->dp_acc_execution, 
                                power_stats->dpmul_acc_execution, 
                                power_stats->dpdiv_acc_execution);

      wrapper->set_fp_accesses(power_stats->fp_acc_execution, 
                              power_stats->fpmul_acc_execution, 
                              power_stats->fpdiv_acc_execution);

      wrapper->set_trans_accesses(power_stats->sqrt_acc_execution, 
                                  power_stats->log_acc_execution, 
                                  power_stats->sin_acc_execution, 
                                  power_stats->exp_acc_execution);

      wrapper->set_tensor_accesses(power_stats->tensor_acc_execution);

      wrapper->set_tex_accesses(power_stats->tex_acc_execution);

      wrapper->set_exec_unit_power(power_stats->ialu_acc_execution,
                                   power_stats->tot_fpu_acc_execution,
                                   power_stats->tot_sfu_acc_execution);

      wrapper->set_avg_active_threads((double)((double)power_stats->tot_threads_acc_execution / (double)power_stats->tot_warps_acc_execution));

      // Average active lanes for sp and sfu pipelines
      float avg_sp_active_lanes =
          (power_stats->sp_active_lanes_execution) / shdr_config->num_shader() / shdr_config->gpgpu_num_sp_units / stat_sample_freq;
      float avg_sfu_active_lanes =
          (power_stats->sfu_active_lanes_execution) / shdr_config->num_shader() / shdr_config->gpgpu_num_sp_units / stat_sample_freq;
      if(avg_sp_active_lanes >32.0 )
        avg_sp_active_lanes = 32.0;
      if(avg_sfu_active_lanes >32.0 )
        avg_sfu_active_lanes = 32.0;
      assert(avg_sp_active_lanes <= 32);
      assert(avg_sfu_active_lanes <= 32);
      wrapper->set_active_lanes_power(avg_sp_active_lanes, avg_sfu_active_lanes);
    }
    else{
      wrapper->set_int_accesses(power_stats->get_ialu_accessess(1), 
                                power_stats->get_intmul24_accessess(1), 
                                power_stats->get_intmul32_accessess(1), 
                                power_stats->get_intmul_accessess(1), 
                                power_stats->get_intdiv_accessess(1));

      wrapper->set_dp_accesses(power_stats->get_dp_accessess(1), 
                                power_stats->get_dpmul_accessess(1), 
                                power_stats->get_dpdiv_accessess(1));

      wrapper->set_fp_accesses(power_stats->get_fp_accessess(1), 
                              power_stats->get_fpmul_accessess(1), 
                              power_stats->get_fpdiv_accessess(1));

      wrapper->set_trans_accesses(power_stats->get_sqrt_accessess(1), 
                                  power_stats->get_log_accessess(1), 
                                  power_stats->get_sin_accessess(1), 
                                  power_stats->get_exp_accessess(1));

      wrapper->set_tensor_accesses(power_stats->get_tensor_accessess(1));

      wrapper->set_tex_accesses(power_stats->get_tex_accessess(1));

      wrapper->set_exec_unit_power(power_stats->get_tot_fpu_accessess(1),
                                   power_stats->get_ialu_accessess(1),
                                   power_stats->get_tot_sfu_accessess(1));

      wrapper->set_avg_active_threads(power_stats->get_active_threads(1));

      // Average active lanes for sp and sfu pipelines
      float avg_sp_active_lanes =
          (power_stats->get_sp_active_lanes()) / stat_sample_freq;
      float avg_sfu_active_lanes =
          (power_stats->get_sfu_active_lanes()) / stat_sample_freq;
      if(avg_sp_active_lanes >32.0 )
        avg_sp_active_lanes = 32.0;
      if(avg_sfu_active_lanes >32.0 )
        avg_sfu_active_lanes = 32.0;
      assert(avg_sp_active_lanes <= 32);
      assert(avg_sfu_active_lanes <= 32);
      wrapper->set_active_lanes_power(avg_sp_active_lanes, avg_sfu_active_lanes);
    }

  
    double n_icnt_simt_to_mem =
      (double)
          (power_stats->get_icnt_simt_to_mem(1) - power_stats->noc_tr_kernel);  // # flits from SIMT clusters
                                                // to memory partitions
    double n_icnt_mem_to_simt =
      (double)
          (power_stats->get_icnt_mem_to_simt(1)- power_stats->noc_rc_kernel);  // # flits from memory
                                                // partitions to SIMT clusters
    if((power_simulation_mode == 2) && (accelwattch_hybrid_configuration[HW_NOC]))   
      wrapper->set_NoC_power(n_icnt_mem_to_simt + n_icnt_simt_to_mem);  // Number of flits traversing the interconnect from Accel-Sim
    else
      wrapper->set_NoC_power(std::stod(hw_data[HW_NOC]));  // Number of flits traversing the interconnect from HW
   
    wrapper->compute();

    wrapper->update_components_power();

    wrapper->power_metrics_calculations();

    wrapper->dump();
    power_stats->l1r_hits_kernel = power_stats->get_l1d_read_hits(1);
    power_stats->l1r_misses_kernel = power_stats->get_l1d_read_misses(1);
    power_stats->l1w_hits_kernel = power_stats->get_l1d_write_hits(1);
    power_stats->l1w_misses_kernel = power_stats->get_l1d_write_misses(1);
    power_stats->shared_accesses_kernel = power_stats->get_const_accessess(1);
    power_stats->cc_accesses_kernel = power_stats->get_shmem_access(1);
    power_stats->dram_rd_kernel = power_stats->get_dram_rd(1);
    power_stats->dram_wr_kernel = power_stats->get_dram_wr(1);
    power_stats->dram_pre_kernel = power_stats->get_dram_pre(1);
    power_stats->l1i_hits_kernel = power_stats->get_inst_c_hits(1);
    power_stats->l1i_misses_kernel = power_stats->get_inst_c_misses(1);
    power_stats->l2r_hits_kernel = power_stats->get_l2_read_hits(1);
    power_stats->l2r_misses_kernel = power_stats->get_l2_read_misses(1);
    power_stats->l2w_hits_kernel =  power_stats->get_l2_write_hits(1); 
    power_stats->l2w_misses_kernel = power_stats->get_l2_write_misses(1);
    power_stats->noc_tr_kernel = power_stats->get_icnt_simt_to_mem(1);
    power_stats->noc_rc_kernel =  power_stats->get_icnt_mem_to_simt(1);


    power_stats->clear();
}