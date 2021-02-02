// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
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

#include "gpu-sim.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

bool g_interactive_debugger_enabled = false;

bool sim_prof_enable = false;

std::map<unsigned long long, std::list<event_stats *>> sim_prof;

void print_sim_prof(FILE *fout, float freq) {
  freq /= 1000;
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      (*iter2)->print(fout, freq);
    }
  }
}

unsigned long long kernel_time = 0;
unsigned long long memory_copy_time_h2d = 0;
unsigned long long memory_copy_time_d2h = 0;
unsigned long long prefetch_time = 0;
unsigned long long devicesync_time = 0;
unsigned long long writeback_time = 0;
unsigned long long dma_time = 0;

//unsigned long long gpu_sim_cycle = 0;
//unsigned long long gpu_tot_sim_cycle = 0;

void calculate_sim_prof(FILE *fout, gpgpu_sim *gpu) {
  float freq = gpu->shader_clock() / 1000.0;
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      (*iter2)->calculate();
    }
  }

  unsigned long long page_fault_time = 0;
  if (!gpu->get_config().enable_accurate_simulation) {
    page_fault_time = gpu->m_new_stats->mf_page_fault_outstanding *
                      gpu->get_config().page_fault_latency;
  }

  fprintf(fout, "Tot_prefetch_time: %llu(cycle), %f(us)\n", prefetch_time,
          ((float)prefetch_time) / freq);
  fprintf(fout, "Tot_kernel_exec_time: %llu(cycle), %f(us)\n", kernel_time,
          ((float)kernel_time) / freq);

  if (!gpu->get_config().enable_accurate_simulation) {
    fprintf(fout, "Tot_kernel_exec_time_and_fault_time: %llu(cycle), %f(us)\n",
            kernel_time + page_fault_time,
            ((float)(kernel_time + page_fault_time)) / freq);
  }

  fprintf(fout, "Tot_memcpy_h2d_time: %llu(cycle), %f(us)\n",
          memory_copy_time_h2d, ((float)memory_copy_time_h2d) / freq);
  fprintf(fout, "Tot_memcpy_d2h_time: %llu(cycle), %f(us)\n",
          memory_copy_time_d2h, ((float)memory_copy_time_d2h) / freq);
  fprintf(fout, "Tot_memcpy_time: %llu(cycle), %f(us)\n",
          memory_copy_time_h2d + memory_copy_time_d2h,
          ((float)(memory_copy_time_h2d + memory_copy_time_d2h)) / freq);
  fprintf(fout, "Tot_devicesync_time: %llu(cycle), %f(us)\n", devicesync_time,
          ((float)devicesync_time) / freq);
  fprintf(fout, "Tot_writeback_time: %llu(cycle), %f(us)\n", writeback_time,
          ((float)writeback_time) / freq);
  fprintf(fout, "Tot_dma_time: %llu(cycle), %f(us)\n", dma_time,
          ((float)dma_time) / freq);
  fprintf(fout, "Tot_memcpy_d2h_sync_wb_time: %llu(cycle), %f(us)\n",
          writeback_time + devicesync_time + memory_copy_time_d2h,
          ((float)(writeback_time + devicesync_time + memory_copy_time_d2h) /
           freq));
}

void update_sim_prof_kernel(unsigned kernel_id, unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == kernel_launch &&
          ((kernel_stats *)(*iter2))->kernel_id == kernel_id) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void update_sim_prof_prefetch(mem_addr_t start_addr, size_t size,
                              unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::reverse_iterator
           iter = sim_prof.rbegin();
       iter != sim_prof.rend(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == prefetch &&
          ((memory_stats *)(*iter2))->start_addr == start_addr &&
          ((memory_stats *)(*iter2))->size == size) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void update_sim_prof_prefetch_break_down(unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::reverse_iterator
           iter = sim_prof.rbegin();
       iter != sim_prof.rend(); iter++) {
    for (std::list<event_stats *>::reverse_iterator iter2 =
             iter->second.rbegin();
         iter2 != iter->second.rend(); iter2++) {
      if ((*iter2)->type == prefetch_breakdown) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void print_UVM_stats(gpgpu_new_stats *new_stats, gpgpu_sim *gpu, FILE *fout) {
  new_stats->print(stdout);

  /*
      FILE* f1 = fopen("Pcie_trace.txt", "w");

      g_the_gpu->m_new_stats->print_pcie(f1);

      fclose(f1);

      FILE* f2 = fopen("Access_pattern_detail.txt", "w");

      g_the_gpu->m_new_stats->print_access_pattern_detail(f2);

      fclose(f2);

      FILE* f3 = fopen("Access_pattern.txt", "w");

      g_the_gpu->m_new_stats->print_access_pattern(f3);

      fclose(f3);

  */
  FILE *f4 = fopen("access.txt", "w");

  new_stats->print_time_and_access(f4);

  fclose(f4);

  if (sim_prof_enable) {
    print_sim_prof(stdout, gpu->shader_clock());
    calculate_sim_prof(stdout, gpu);
  }
}

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08
#define GMMU 0x10

#define MEM_LATENCY_STAT_IMPL

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 8)", "8");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_UINT32,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_INT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_INT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_INT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_INT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");
  option_parser_register(opp, "-tlb_size", OPT_INT32, &tlb_size,
                         "Number of tlb entries per SM.", "4096");                       

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "GPUWattch XML file",
                         "gpuwattch.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  m_address_mapping.addrdec_setoption(opp);
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU", "8");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
  option_parser_register(
      opp, "-gddr_size", OPT_CSTR, &gddr_size_string,
      "Size of GDDR in MB/GB.(GLOBAL_HEAP_START, GLOBAL_HEAP_START + "
      "gddr_size) would be used for unmanged memory, (GLOBAL_HEAP_START + "
      "gddr_size, GLOBAL_HEAP_START + gddr_size*2) would be used for managed "
      "memory. ",
      "1GB");

  option_parser_register(
      opp, "-page_table_walk_latency", OPT_INT64, &page_table_walk_latency,
      "Average page table walk latency (in core cycle).", "100");

  option_parser_register(opp, "-eviction_policy", OPT_INT32, &eviction_policy,
                         "Select page eviction policy", "0");

  option_parser_register(opp, "-invalidate_clean", OPT_BOOL, &invalidate_clean,
                         "Should directly invalidate clean pages", "0");

  option_parser_register(
      opp, "-reserve_accessed_page_percent", OPT_FLOAT,
      &reserve_accessed_page_percent,
      "Percentage of accessed pages reserved from eviction in hope that they "
      "will be accessed in next iteration.",
      "0.0");

  option_parser_register(
      opp, "-percentage_of_free_page_buffer", OPT_FLOAT,
      &free_page_buffer_percentage,
      "Percentage of free page buffer to trigger the page eviction.", "0.0");

  option_parser_register(opp, "-page_size", OPT_CSTR, &page_size_string,
                         "GDDR page size, only 4KB/2MB avaliable.", "4KB");

  option_parser_register(opp, "-pcie_bandwidth", OPT_CSTR,
                         &pcie_bandwidth_string,
                         "PCI-e bandwidth per direction, in GB/s.", "16.0GB/s");

  option_parser_register(opp, "-enable_dma", OPT_INT32, &enable_dma,
                         "Enable direct access to CPU memory", "0");

  option_parser_register(
      opp, "-multiply_dma_penalty", OPT_INT32, &multiply_dma_penalty,
      "Oversubscription Multiplicative Penalty Factor for Adaptive DMA", "0");

  option_parser_register(
      opp, "-migrate_threshold", OPT_INT32, &migrate_threshold,
      "Access counter threshold for migrating the page from cpu to gpu", "10");

  option_parser_register(opp, "-sim_prof_enable", OPT_BOOL, &sim_prof_enable,
                         "Enable gpgpu-sim profiler", "0");

  option_parser_register(opp, "-hardware_prefetch", OPT_INT32,
                         &hardware_prefetch,
                         "Select gpgpu-sim hardware prefetcher", "1");

  option_parser_register(
      opp, "-hwprefetch_oversub", OPT_INT32, &hwprefetch_oversub,
      "Select gpgpu-sim hardware prefetcher under over-subscription", "0");

  option_parser_register(opp, "-page_fault_latency", OPT_INT64,
                         &page_fault_latency,
                         "Average fault latency (in core cycle).", "66645");

  option_parser_register(opp, "-enable_accurate_simulation", OPT_BOOL,
                         &enable_accurate_simulation,
                         "Enable page fault functional simulation.", "0");

  option_parser_register(opp, "-enable_smart_runtime", OPT_BOOL,
                         &enable_smart_runtime,
                         "Enable access pattern detection, policy engine, and "
                         "adaptive memory management.",
                         "0");
}

void gpgpu_sim_config::convert_byte_string() {
  gpgpu_functional_sim_config::convert_byte_string();
  if (strstr(pcie_bandwidth_string, "GB/s")) {
    pcie_bandwidth = strtof(pcie_bandwidth_string, NULL);
    if (pcie_bandwidth == 16.0) {
      curve_a = 12.0;
    } else if (pcie_bandwidth == 32.0) {
      curve_a = 24.0;
    } else if (pcie_bandwidth == 64.0) {
      curve_a = 48.0;
    } else {
      printf("-pcie_bandwidth should be 16.0GB/s, 32.0GB/s or 64.0GB/s\n");
    }

    curve_b = 0.07292;

  } else {
    printf("-pcie_bandwidth should be in GB/s\n");
    exit(1);
  }
}
/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats,
                                   m_new_stats);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,
                                             config.g_power_config_name);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  m_new_stats = new gpgpu_new_stats(m_config);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  m_gmmu = new gmmu_t(this, config, m_new_stats);

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;

  m_memory_partition_unit =
      new memory_partition_unit *[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i] =
        new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  icnt_wrapper_init();
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition);

  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
  gmmu_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

void gpgpu_sim::print_stats() {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat();

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  printf("partiton_reqs_in_parallel_total    = %lld\n",
  partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("partiton_reqs_in_parallel_util = %lld\n",
  partiton_reqs_in_parallel_util);
  printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  partiton_reqs_in_parallel_util_total ); 
  printf("gpu_sim_cycle_parition_util = %lld\n", gpu_sim_cycle_parition_util);
  printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.icnt_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  print_UVM_stats(m_new_stats, this, stdout);
  fflush(stdout);

  clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld)\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle);
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  // to get the cycles spent for any cuda stream operation before and after the
  // kernel is launched monotonically increase the total simulation cycle
  if (!active()) {
    int mask = 0x00;
    mask |= GMMU;
    gpu_tot_sim_cycle++;
    return mask;
  }

  double smallest = min4(core_time, icnt_time, dram_time, gmmu_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  if (gmmu_time <= smallest) {
    mask |= GMMU;
    gmmu_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

gpgpu_new_stats::gpgpu_new_stats(const gpgpu_sim_config &config)
    : m_config(config) {
  tlb_hit = new unsigned long long[m_config.num_cluster()];
  tlb_miss = new unsigned long long[m_config.num_cluster()];
  tlb_val = new unsigned long long[m_config.num_cluster()];
  tlb_evict = new unsigned long long[m_config.num_cluster()];
  tlb_page_evict = new unsigned long long[m_config.num_cluster()];

  mf_page_hit = new unsigned long long[m_config.num_cluster()];
  mf_page_miss = new unsigned long long[m_config.num_cluster()];

  mf_page_fault_outstanding = 0;
  mf_page_fault_pending = 0;

  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    tlb_hit[i] = 0;
    tlb_miss[i] = 0;
    tlb_val[i] = 0;
    tlb_evict[i] = 0;
    tlb_page_evict[i] = 0;
    mf_page_hit[i] = 0;
    mf_page_miss[i] = 0;
  }

  pf_page_hit = 0;
  pf_page_miss = 0;

  page_evict_not_dirty = 0;
  page_evict_dirty = 0;

  num_dma = 0;
  dma_page_transfer_read = 0;
  dma_page_transfer_write = 0;

  tlb_thrashing =
      new std::map<mem_addr_t, std::vector<bool>>[m_config.num_cluster()];

  ma_latency =
      new std::map<unsigned,
                   std::pair<bool, unsigned long long>>[m_config.num_cluster()];

  page_access_times =
      new std::map<mem_addr_t, unsigned>[m_config.num_cluster()];
}

void gpgpu_new_stats::print_pcie(FILE *fout) const {
  fprintf(fout, "Read lanes:\n");
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_read_utilization.begin();
       iter != pcie_read_utilization.end(); iter++) {
    fprintf(fout, "%llu %f\n", iter->first, iter->second);
  }
  fprintf(fout, "Write lanes:\n");
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_write_utilization.begin();
       iter != pcie_write_utilization.end(); iter++) {
    fprintf(fout, "%llu %f\n", iter->first, iter->second);
  }
}

void gpgpu_new_stats::print_access_pattern_detail(FILE *fout) const {
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout, "Shader %u\n", i);
    for (std::map<mem_addr_t, unsigned>::const_iterator iter =
             page_access_times[i].begin();
         iter != page_access_times[i].end(); iter++) {
      fprintf(fout, "%u %u\n", iter->first, iter->second);
    }
  }
}

void gpgpu_new_stats::print_access_pattern(FILE *fout) const {
  std::map<mem_addr_t, unsigned> tot_access;
  fprintf(fout, "Total page access pttern:\n");
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    for (std::map<mem_addr_t, unsigned>::const_iterator iter =
             page_access_times[i].begin();
         iter != page_access_times[i].end(); iter++) {
      tot_access[iter->first] += iter->second;
    }
  }
  for (std::map<mem_addr_t, unsigned>::const_iterator iter = tot_access.begin();
       iter != tot_access.end(); iter++) {
    fprintf(fout, "%u %u\n", iter->first, iter->second);
  }
}

void gpgpu_new_stats::print_time_and_access(FILE *fout) const {
  for (std::list<access_info>::const_iterator iter =
           time_and_page_access.begin();
       iter != time_and_page_access.end(); iter++) {
    fprintf(fout, "%u 0x%x %u %llu %d %u %u\n", iter->page_no, iter->mem_addr,
            iter->size, iter->cycle, iter->is_read, iter->sm_id, iter->warp_id);
  }

  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == kernel_launch) {
        fprintf(fout, "K: %llu %llu\n", (*iter2)->start_time,
                (*iter2)->end_time);
      }
    }
  }
}

void gpgpu_new_stats::print(FILE *fout) const {
  fprintf(fout, "========================================UVM "
                "statistics==============================\n");

  fprintf(fout, "========================================TLB "
                "statistics(access)==============================\n");
  unsigned long long tot_tlb_hit = 0;
  unsigned long long tot_tlb_miss = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout,
            "Shader%u: Tlb_access: %llu Tlb_hit: %llu Tlb_miss: %llu "
            "Tlb_hit_rate: %f\n",
            i, tlb_hit[i] + tlb_miss[i], tlb_hit[i], tlb_miss[i],
            ((float)tlb_hit[i]) / ((float)(tlb_hit[i] + tlb_miss[i])));
    tot_tlb_hit += tlb_hit[i];
    tot_tlb_miss += tlb_miss[i];
  }

  fprintf(fout,
          "Tlb_tot_access: %llu Tlb_tot_hit: %llu, Tlb_tot_miss: %llu, "
          "Tlb_tot_hit_rate: %f\n",
          tot_tlb_hit + tot_tlb_miss, tot_tlb_hit, tot_tlb_miss,
          ((float)tot_tlb_hit) / ((float)(tot_tlb_hit + tot_tlb_miss)));

  fprintf(fout, "========================================TLB "
                "statistics(validate)==============================\n");
  unsigned long long tot_tlb_val = 0;
  unsigned long long tot_tlb_inval_te = 0;
  unsigned long long tot_tlb_inval_pe = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout,
            "Shader%u: Tlb_validate: %llu Tlb_invalidate: %llu Tlb_evict: %llu "
            "Tlb_page_evict: %llu\n",
            i, tlb_val[i], tlb_evict[i] + tlb_page_evict[i], tlb_evict[i],
            tlb_page_evict[i]);
    tot_tlb_val += tlb_val[i];
    tot_tlb_inval_te += tlb_evict[i];
    tot_tlb_inval_pe += tlb_page_evict[i];
  }

  fprintf(fout,
          "Tlb_tot_valiate: %llu Tlb_invalidate: %llu, Tlb_tot_evict: %llu, "
          "Tlb_tot_evict page: %llu\n",
          tot_tlb_val, tot_tlb_inval_te + tot_tlb_inval_pe, tot_tlb_inval_te,
          tot_tlb_inval_pe);

  fprintf(fout, "========================================TLB "
                "statistics(thrashing)==============================\n");
  std::map<mem_addr_t, unsigned> tlb_thrash[m_config.num_cluster()];
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
             tlb_thrashing[i].begin();
         iter != tlb_thrashing[i].end(); iter++) {
      for (unsigned j = 0; j != iter->second.size(); j++) {
        if (j + 2 >= iter->second.size())
          break;
        if (iter->second[j] == true && iter->second[j + 1] == false &&
            iter->second[j + 2] == true)
          tlb_thrash[i][iter->first]++;
      }
    }
  }

  unsigned tot_tlb_thrash = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    unsigned s_thrash = 0;
    fprintf(fout, "Shader%u: ", i);
    for (std::map<mem_addr_t, unsigned>::iterator iter = tlb_thrash[i].begin();
         iter != tlb_thrash[i].end(); iter++) {
      fprintf(fout, "Page: %u Trashed: %u | ", iter->first, iter->second);
      s_thrash += iter->second;
    }
    fprintf(fout, "Total %u\n", s_thrash);
    tot_tlb_thrash += s_thrash;
  }
  fprintf(fout, "Tlb_tot_thrash: %u\n", tot_tlb_thrash);

  fprintf(fout, "========================================Page fault "
                "statistics==============================\n");

  unsigned long long tot_page_hit = 0;
  unsigned long long tot_page_miss = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(
        fout,
        "Shader%u: Page_table_access:%llu Page_hit: %llu Page_miss: %llu "
        "Page_hit_rate: %f\n",
        i, mf_page_hit[i] + mf_page_miss[i], mf_page_hit[i], mf_page_miss[i],
        ((float)mf_page_hit[i]) / ((float)(mf_page_hit[i] + mf_page_miss[i])));
    tot_page_hit += mf_page_hit[i];
    tot_page_miss += mf_page_miss[i];
  }

  fprintf(fout,
          "Page_table_tot_access: %llu Page_tot_hit: %llu, Page_tot_miss %llu, "
          "Page_tot_hit_rate: %f Page_tot_fault: %llu Page_tot_pending: %llu\n",
          tot_page_hit + tot_page_miss, tot_page_hit, tot_page_miss,
          ((float)tot_page_hit) / ((float)(tot_page_hit + tot_page_miss)),
          mf_page_fault_outstanding, mf_page_fault_pending);

  float avg_mf_latency = 0;
  unsigned long long tot_mf_fault = 0;
  for (std::map<mem_addr_t, std::list<unsigned long long>>::const_iterator
           iter = mf_page_fault_latency.begin();
       iter != mf_page_fault_latency.end(); iter++) {
    for (std::list<unsigned long long>::const_iterator iter2 =
             iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      avg_mf_latency =
          ((float)tot_mf_fault) / ((float)(tot_mf_fault + 1)) * avg_mf_latency +
          ((float)(*iter2)) / ((float)(tot_mf_fault + 1));
      tot_mf_fault++;
    }
  }
  fprintf(fout, "Total_memory_access_page_fault: %llu, Average_latency: %f\n",
          tot_mf_fault, avg_mf_latency);

  fprintf(fout, "========================================Page thrashing "
                "statistics==============================\n");

  unsigned long long tot_validate = 0;
  for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
           page_thrashing.begin();
       iter != page_thrashing.end(); iter++) {
    for (std::vector<bool>::const_iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if (*iter2 == true)
        tot_validate++;
    }
  }

  fprintf(
      fout,
      "Page_validate: %llu Page_evict_dirty: %llu Page_evict_not_dirty: %llu\n",
      tot_validate, page_evict_dirty, page_evict_not_dirty);

  std::map<mem_addr_t, unsigned> page_thrash;
  for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
           page_thrashing.begin();
       iter != page_thrashing.end(); iter++) {
    for (unsigned j = 0; j != iter->second.size(); j++) {
      if (j + 2 >= iter->second.size())
        break;
      if (iter->second[j] == true && iter->second[j + 1] == false &&
          iter->second[j + 2] == true)
        page_thrash[iter->first]++;
    }
  }

  unsigned tot_page_thrash = 0;
  for (std::map<mem_addr_t, unsigned>::iterator iter = page_thrash.begin();
       iter != page_thrash.end(); iter++) {
    fprintf(fout, "Page: %u Thrashed: %u\n", iter->first, iter->second);
    tot_page_thrash += iter->second;
  }
  fprintf(fout, "Page_tot_thrash: %u\n", tot_page_thrash);

  fprintf(fout, "========================================Memory access "
                "statistics==============================\n");

  /*
     unsigned long long* ma_num = new unsigned long
     long[m_config.num_cluster()]; float* avg_ma_latency = new
     float[m_config.num_cluster()];

     unsigned long long tot_ma_num = 0;
     float tot_avg_ma_latency = 0;

     for(unsigned i = 0; i < m_config.num_cluster(); i++) {
         ma_num[i] = 0;
         avg_ma_latency[i] = 0;
         for(std::map<unsigned, std::pair<bool, unsigned long long>
     >::const_iterator iter = ma_latency[i].begin(); iter !=
     ma_latency[i].end(); iter++) { assert(iter->second.first);
             avg_ma_latency[i] = ((float)ma_num[i]) / ((float)(ma_num[i]+1)) *
     avg_ma_latency[i] + ((float)(iter->second.second)) /
     ((float)(ma_num[i]+1)); ma_num[i]++;
         }
         fprintf(fout, "Shader%u: Memory_access: %u, Avg_memory_access_latency:
     %llu\n", i, ma_latency[i].size(), ((unsigned long long)
     (avg_ma_latency[i])));
     }

     for(unsigned i = 0; i < m_config.num_cluster(); i++) {
         tot_avg_ma_latency = ((float)tot_ma_num) /
     ((float)(tot_ma_num+ma_num[i])) * tot_avg_ma_latency + avg_ma_latency[i] /
     ((float)(tot_ma_num+ma_num[i])) * ((float)ma_num[i]); tot_ma_num +=
     ma_num[i];
     }
     fprintf(fout,"Tot_memory_access: %u, Tot_avg_memory_access_latency:
     %llu\n", tot_ma_num, ((unsigned long long)tot_avg_ma_latency));

     delete[] ma_num;
     delete[] avg_ma_latency;
  */
  fprintf(fout, "========================================Prefetch "
                "statistics==============================\n");

  fprintf(fout,
          "Tot_page_hit: %llu, Tot_page_miss: %llu, Tot_page_fault: %lu\n",
          pf_page_hit, pf_page_miss, pf_fault_latency.size());

  float avg_pf_latency = 0;
  float avg_pref_size = 0;
  float avg_pref_latency = 0;

  unsigned long long tot_pf_fault = 0;
  unsigned long long tot_pref_fault = 0;
  for (std::map<mem_addr_t, std::list<unsigned long long>>::const_iterator
           iter = pf_page_fault_latency.begin();
       iter != pf_page_fault_latency.end(); iter++) {
    for (std::list<unsigned long long>::const_iterator iter2 =
             iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      avg_pf_latency =
          ((float)tot_pf_fault) / ((float)(tot_pf_fault + 1)) * avg_pf_latency +
          ((float)(*iter2)) / ((float)(tot_pf_fault + 1));
      tot_pf_fault++;
    }
  }

  for (std::vector<std::pair<unsigned long, unsigned long long>>::const_iterator
           iter = pf_fault_latency.begin();
       iter != pf_fault_latency.end(); iter++) {
    avg_pref_size += iter->first;
    avg_pref_latency = ((float)tot_pref_fault) / ((float)(tot_pref_fault + 1)) *
                           avg_pref_latency +
                       ((float)(iter->second)) / ((float)(tot_pref_fault + 1));
    tot_pref_fault++;
  }

  avg_pref_size /= ((float)pf_fault_latency.size());
  fprintf(
      fout,
      "Avg_page_latency: %f, Avg_prefetch_size: %f, Avg_prefetch_latency: %f\n",
      avg_pf_latency, avg_pref_size, avg_pref_latency);

  fprintf(fout, "========================================Rdma "
                "statistics==============================\n");
  fprintf(fout, "dma_read: %llu\n", num_dma);
  fprintf(fout, "dma_migration_read %llu\n", dma_page_transfer_read);
  fprintf(fout, "dma_migration_write %llu\n", dma_page_transfer_write);

  fprintf(fout, "========================================PCI-e "
                "statistics==============================\n");
  float avg_r = 0;
  unsigned long long r_0 = 0;
  unsigned long long r_25 = 0;
  unsigned long long r_50 = 0;
  unsigned long long r_75 = 0;
  unsigned long long r_tot = 0;
  float avg_w = 0;
  unsigned long long w_0 = 0;
  unsigned long long w_25 = 0;
  unsigned long long w_50 = 0;
  unsigned long long w_75 = 0;
  unsigned long long w_tot = 0;
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_read_utilization.begin();
       iter != pcie_read_utilization.end(); iter++) {
    if (iter->second <= 0.25) {
      r_0++;
    } else if (iter->second <= 0.5) {
      r_25++;
    } else if (iter->second <= 0.75) {
      r_50++;
    } else {
      r_75++;
    }
    avg_r = (avg_r * ((float)r_tot) + iter->second) / ((float)(r_tot + 1));
    r_tot++;
  }
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_write_utilization.begin();
       iter != pcie_write_utilization.end(); iter++) {
    if (iter->second <= 0.25) {
      w_0++;
    } else if (iter->second <= 0.5) {
      w_25++;
    } else if (iter->second <= 0.75) {
      w_50++;
    } else {
      w_75++;
    }
    avg_w = (avg_w * ((float)w_tot) + iter->second) / ((float)(w_tot + 1));
    w_tot++;
  }

  fprintf(fout, "Pcie_read_utilization: %f\n", avg_r);
  fprintf(fout, "[0-25]: %f, [26-50]: %f, [51-75]: %f, [76-100]: %f\n",
          ((float)r_0) / ((float)r_tot), ((float)r_25) / ((float)r_tot),
          ((float)r_50) / ((float)r_tot), ((float)r_75) / ((float)r_tot));
  fprintf(fout, "Pcie_write_utilization: %f\n", avg_w);
  fprintf(fout, "[0-25]: %f, [26-50]: %f, [51-75]: %f, [76-100]: %f\n",
          ((float)w_0) / ((float)w_tot), ((float)w_25) / ((float)w_tot),
          ((float)w_50) / ((float)w_tot), ((float)w_75) / ((float)w_tot));
}

gpgpu_new_stats::~gpgpu_new_stats() {
  delete[] tlb_hit;
  delete[] tlb_miss;
  delete[] tlb_val;
  delete[] tlb_evict;
  delete[] tlb_page_evict;
  delete[] mf_page_hit;
  delete[] mf_page_miss;
  delete[] page_access_times;
  delete[] tlb_thrashing;
  delete[] ma_latency;
}

gmmu_t::gmmu_t(class gpgpu_sim *gpu, const gpgpu_sim_config &config,
               class gpgpu_new_stats *new_stats)
    : m_gpu(gpu), m_config(config), m_new_stats(new_stats) {
  m_shader_config = &m_config.m_shader_config;

  if (m_config.enable_dma == 0) {
    dma_mode = dma_type::DISABLED;
  } else if (m_config.enable_dma == 1) {
    dma_mode = dma_type::ADAPTIVE;
  } else if (m_config.enable_dma == 2) {
    dma_mode = dma_type::ALWAYS;
  } else if (m_config.enable_dma == 3) {
    dma_mode = dma_type::OVERSUB;
  } else {
    printf("Unknown DMA mode\n");
    exit(1);
  }

  if (m_config.eviction_policy == 0) {
    evict_policy = eviction_policy::LRU;
  } else if (m_config.eviction_policy == 1) {
    evict_policy = eviction_policy::TBN;
  } else if (m_config.eviction_policy == 2) {
    evict_policy = eviction_policy::SEQUENTIAL_LOCAL;
  } else if (m_config.eviction_policy == 3) {
    evict_policy = eviction_policy::RANDOM;
  } else if (m_config.eviction_policy == 4) {
    evict_policy = eviction_policy::LFU;
  } else if (m_config.eviction_policy == 5) {
    evict_policy = eviction_policy::LRU4K;
  } else {
    printf("Unknown eviction policy");
    exit(1);
  }

  if (m_config.hardware_prefetch == 0) {
    prefetcher = hwardware_prefetcher::DISBALED;
  } else if (m_config.hardware_prefetch == 1) {
    prefetcher = hwardware_prefetcher::TBN;
  } else if (m_config.hardware_prefetch == 2) {
    prefetcher = hwardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (m_config.hardware_prefetch == 3) {
    prefetcher = hwardware_prefetcher::RANDOM;
  } else {
    printf("Unknown hardware prefeching policy");
    exit(1);
  }

  if (m_config.hwprefetch_oversub == 0) {
    oversub_prefetcher = hwardware_prefetcher_oversub::DISBALED;
  } else if (m_config.hwprefetch_oversub == 1) {
    oversub_prefetcher = hwardware_prefetcher_oversub::TBN;
  } else if (m_config.hwprefetch_oversub == 2) {
    oversub_prefetcher = hwardware_prefetcher_oversub::SEQUENTIAL_LOCAL;
  } else if (m_config.hwprefetch_oversub == 3) {
    oversub_prefetcher = hwardware_prefetcher_oversub::RANDOM;
  } else {
    printf("Unknown hardware prefeching policy under over-subscription");
    exit(1);
  }

  pcie_read_latency_queue = NULL;
  pcie_write_latency_queue = NULL;

  total_allocation_size = 0;

  over_sub = false;

  //gpu_sim_cycle = m_gpu->gpu_sim_cycle;
  //gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
}

unsigned long long gmmu_t::calculate_transfer_time(size_t data_size) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));

  if (data_size >= 2 * 1024 * 1024) {
    speed /= 2;
  }

  return (unsigned long long)((float)(data_size)*m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}

void gmmu_t::calculate_devicesync_time(size_t data_size) {

  unsigned cur_turn = 0;
  unsigned cur_size = 0;

  float speed;

  while (data_size != 0) {

    unsigned long long cur_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
    unsigned long long cur_time = 0;

    if (cur_turn == 0) {
      cur_size = MIN_PREFETCH_SIZE;
    } else {
      cur_size = MIN_PREFETCH_SIZE * pow(2, cur_turn - 1);
    }

    if (data_size < 4096) {
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(data_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, data_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;

      return;
    } else {
      cur_size -= 4096;
      data_size -= 4096;
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(4096) / (float)(1024)));
      cur_time = (unsigned long long)((float)(4096) * m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, 4096, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
    }

    if (data_size < cur_size) {
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(data_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, data_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;

      return;
    } else {
      data_size -= cur_size;
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(cur_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(cur_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, cur_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
    }

    cur_turn++;
    if (cur_turn == 6) {
      cur_turn = 0;
    }
  }
  return;
}

bool gmmu_t::pcie_transfers_completed() {
  return pcie_write_stage_queue.empty() && pcie_write_latency_queue == NULL &&
         pcie_read_stage_queue.empty() && pcie_read_latency_queue == NULL;
}

void gmmu_t::register_tlbflush_callback(
    std::function<void(mem_addr_t)> cb_tlb) {
  callback_tlb_flush.push_back(cb_tlb);
}

void gmmu_t::tlb_flush(mem_addr_t page_num) {
  for (list<std::function<void(mem_addr_t)>>::iterator iter =
           callback_tlb_flush.begin();
       iter != callback_tlb_flush.end(); iter++) {
    (*iter)(page_num);
  }
}

void gmmu_t::check_write_stage_queue(mem_addr_t page_num, bool refresh) {
  // the page, about to be accessed, was selected for eviction earlier
  // so don't evict that page
  for (std::list<pcie_latency_t *>::iterator iter =
           pcie_write_stage_queue.begin();
       iter != pcie_write_stage_queue.end(); iter++) {
    if (std::find((*iter)->page_list.begin(), (*iter)->page_list.end(),
                  page_num) != (*iter)->page_list.end()) {
      // on tlb hit refresh position of pages in the valid page list
      for (std::list<mem_addr_t>::iterator pg_iter = (*iter)->page_list.begin();
           pg_iter != (*iter)->page_list.end(); pg_iter++) {
        m_gpu->get_global_memory()->set_page_access(*pg_iter);

        m_gpu->get_global_memory()->set_page_dirty(*pg_iter);

        // reclaim valid size in large page tree for unique basic blocks
        // corresponding to all pages
        mem_addr_t page_addr =
            m_gpu->get_global_memory()->get_mem_addr(*pg_iter);
        struct lp_tree_node *root = get_lp_node(page_addr);
        update_basic_block(root, page_addr, m_config.page_size, true);

        refresh_valid_pages(page_addr);
      }

      pcie_write_stage_queue.erase(iter);
      break;
    }
  }
}

// check if the block is already scheduled for eviction or is not valid at all
bool gmmu_t::is_block_evictable(mem_addr_t addr, size_t size) {
  for (mem_addr_t start = addr; start != addr + size;
       start += m_config.page_size) {
    if (!m_gpu->get_global_memory()->is_valid(
            m_gpu->get_global_memory()->get_page_num(start))) {
      return false;
    }
  }

  for (std::list<pcie_latency_t *>::iterator iter =
           pcie_write_stage_queue.begin();
       iter != pcie_write_stage_queue.end(); iter++) {
    if ((addr >= (*iter)->start_addr) &&
        (addr < (*iter)->start_addr + (*iter)->size)) {
      return false;
    }
  }

  for (mem_addr_t start = addr; start != addr + size;
       start += m_config.page_size) {
    if (!reserve_pages_check(start)) {
      return false;
    }
  }

  return true;
}

void gmmu_t::page_eviction_procedure() {
  sort_valid_pages();

  std::list<std::pair<mem_addr_t, size_t>> evicted_pages;

  int eviction_start =
      (int)(valid_pages.size() * m_config.reserve_accessed_page_percent / 100);

  if (evict_policy == eviction_policy::LRU4K) {
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);

      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    }
  } else if (evict_policy == eviction_policy::LRU ||
             evict_policy == eviction_policy::LFU ||
             m_config.page_size == MAX_PREFETCH_SIZE) {
    // in lru, only evict the least recently used pages at the front of accessed
    // pages queue in lfu, only evict the page accessed least number of times
    // from the front of accessed pages queue
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      evict_whole_tree(root);

      evicted_pages.push_back(std::make_pair(root->addr, root->size));
    }
  } else if (evict_policy == eviction_policy::RANDOM) {
    // in random eviction, select a random page
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(
        iter, eviction_start +
                  (rand() %
                   (int)(valid_pages.size() *
                         (1 - m_config.reserve_accessed_page_percent / 100))));

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);

      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    }
  } else if (evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    // we evict sixteen 4KB pages in the 2 MB allocation where this evictable
    // belong to
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        break;
      }
    }

    if (iter != valid_pages.end()) {
      evicted_pages.push_back(std::make_pair(bb_addr, MIN_PREFETCH_SIZE));
    }
  } else if (evict_policy == eviction_policy::TBN) {
    // we evict multiple 64KB pages in the 2 MB allocation where this evictable
    // belong
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    // find all basic blocks which are not staged/scheduled for write back or
    // not invalid or not in ld/st unit
    std::set<mem_addr_t> all_basic_blocks;

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        break;
      }
    }

    if (iter != valid_pages.end()) {
      all_basic_blocks.insert(bb_addr);
      traverse_and_remove_lp_tree(root, all_basic_blocks);
    }

    // group all contiguous basic blocks if possible
    std::set<mem_addr_t>::iterator bb = all_basic_blocks.begin();

    while (bb != all_basic_blocks.end()) {
      std::set<mem_addr_t>::iterator next_bb = bb;
      size_t cur_num = 0;

      do {
        next_bb++;
        cur_num++;
      } while (next_bb != all_basic_blocks.end() &&
               ((*next_bb) == ((*bb) + cur_num * MIN_PREFETCH_SIZE)));

      evicted_pages.push_back(
          std::make_pair((*bb), (cur_num * MIN_PREFETCH_SIZE)));

      bb = next_bb;
    }
  }

  // always write back the chunk no matter what it has not dirty pages or dirty
  // pages
  for (std::list<std::pair<mem_addr_t, size_t>>::iterator iter =
           evicted_pages.begin();
       iter != evicted_pages.end(); iter++) {
    pcie_latency_t *p_t = new pcie_latency_t();

    p_t->start_addr = iter->first;
    p_t->size = iter->second;

    latency_type ltype = latency_type::PCIE_WRITE_BACK;

    for (std::list<eviction_t *>::iterator it = valid_pages.begin();
         it != valid_pages.end(); it++) {
      if ((*it)->addr <= iter->first &&
          iter->first < (*it)->addr + (*it)->size) {
        if ((*it)->RW == 1) {
          ltype = latency_type::INVALIDATE;
          break;
        }
      }
    }

    p_t->type = ltype;

    if (m_config.page_size == MAX_PREFETCH_SIZE) {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter->first);

      p_t->page_list.push_back(page_num);

      valid_pages_erase(page_num);
    } else {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter->first);

      for (int i = 0; i < (int)(iter->second / m_config.page_size); i++) {
        p_t->page_list.push_back(page_num + i);

        valid_pages_erase(page_num + i);
      }
    }

    pcie_write_stage_queue.push_back(p_t);
  }
}

void gmmu_t::valid_pages_erase(mem_addr_t page_num) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  for (std::list<eviction_t *>::iterator it = valid_pages.begin();
       it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      valid_pages.erase(it);
      break;
    }
  }
}

void gmmu_t::valid_pages_clear() { valid_pages.clear(); }

void gmmu_t::refresh_valid_pages(mem_addr_t page_addr) {
  bool valid = false;
  for (std::list<eviction_t *>::iterator it = valid_pages.begin();
       it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      (*it)->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      valid = true;
      break;
    }
  }

  if (!valid) {
    eviction_t *item = new eviction_t();
    item->addr = get_eviction_base_addr(page_addr);
    item->size = get_eviction_granularity(page_addr);
    item->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    valid_pages.push_back(item);
  }
}

void gmmu_t::sort_valid_pages() {
  for (std::list<eviction_t *>::iterator vp_iter = valid_pages.begin();
       vp_iter != valid_pages.end(); vp_iter++) {
    for (std::list<struct lp_tree_node *>::iterator lp_iter =
             large_page_info.begin();
         lp_iter != large_page_info.end(); lp_iter++) {
      if ((*vp_iter)->addr == (*lp_iter)->addr) {
        (*vp_iter)->access_counter = (*lp_iter)->access_counter;
        (*vp_iter)->RW = (*lp_iter)->RW;
        break;
      }
    }
  }

  if (evict_policy == eviction_policy::LFU) {
    valid_pages.sort([](const eviction_t *i, const eviction_t *j) {
      return (i->access_counter < j->access_counter) ||
             ((i->access_counter == j->access_counter) && (i->RW < j->RW)) ||
             ((i->access_counter == j->access_counter) && (i->RW == j->RW) &&
              (i->cycle < j->cycle));
    });
  } else {
    if (evict_policy == eviction_policy::TBN ||
        evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
      std::map<mem_addr_t, std::list<eviction_t *>> tempMap;

      for (std::list<eviction_t *>::iterator it = valid_pages.begin();
           it != valid_pages.end(); it++) {
        struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node((*it)->addr);
        tempMap[root->addr].push_back(*it);
      }

      for (std::map<mem_addr_t, std::list<eviction_t *>>::iterator it =
               tempMap.begin();
           it != tempMap.end(); it++) {
        it->second.sort([](const eviction_t *i, const eviction_t *j) {
          return i->cycle > j->cycle;
        });
      }

      std::list<pair<mem_addr_t, std::list<eviction_t *>>> tempList;

      for (std::map<mem_addr_t, std::list<eviction_t *>>::iterator it =
               tempMap.begin();
           it != tempMap.end(); it++) {
        tempList.push_back(make_pair(it->first, it->second));
      }

      tempList.sort([](const pair<mem_addr_t, std::list<eviction_t *>> i,
                       const pair<mem_addr_t, std::list<eviction_t *>> j) {
        return i.second.front()->cycle < j.second.front()->cycle;
      });

      std::list<eviction_t *> new_valid_pages;

      for (std::list<pair<mem_addr_t, std::list<eviction_t *>>>::iterator it =
               tempList.begin();
           it != tempList.end(); it++) {
        (*it).second.sort([](const eviction_t *i, const eviction_t *j) {
          return i->cycle < j->cycle;
        });
        new_valid_pages.insert(new_valid_pages.end(), it->second.begin(),
                               it->second.end());
      }

      valid_pages = new_valid_pages;
    } else {
      valid_pages.sort([](const eviction_t *i, const eviction_t *j) {
        return i->cycle < j->cycle;
      });
    }
  }
}

unsigned long long gmmu_t::get_ready_cycle(unsigned num_pages) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b *
                     ((float)(num_pages * m_config.page_size) / 1024.0));

  return m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle +
         (unsigned long long)((float)(m_config.page_size * num_pages) *
                              m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}

unsigned long long gmmu_t::get_ready_cycle_dma(unsigned size) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)(size) / 1024.0));
  return m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + 200;
}

float gmmu_t::get_pcie_utilization(unsigned num_pages) {
  return 2.0 * m_config.curve_a / M_PI *
         atan(m_config.curve_b *
              ((float)(num_pages * m_config.page_size) / 1024.0)) /
         m_config.pcie_bandwidth;
}

void gmmu_t::activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt,
                               struct CUstream_st *m_stream) {
  for (std::list<prefetch_req>::iterator iter = prefetch_req_buffer.begin();
       iter != prefetch_req_buffer.end(); iter++) {
    if (iter->start_addr == m_device_addr && iter->size == m_cnt &&
        iter->m_stream->get_uid() == m_stream->get_uid()) {
      assert(iter->cur_addr == m_device_addr);
      iter->active = true;
      return;
    }
  }
}

void gmmu_t::register_prefetch(mem_addr_t m_device_addr,
                               mem_addr_t m_device_allocation_ptr, size_t m_cnt,
                               struct CUstream_st *m_stream) {
  struct prefetch_req pre_q;

  pre_q.start_addr = m_device_addr;
  pre_q.cur_addr = m_device_addr;
  pre_q.allocation_addr = m_device_allocation_ptr;
  pre_q.size = m_cnt;
  pre_q.active = false;
  pre_q.m_stream = m_stream;

  prefetch_req_buffer.push_back(pre_q);
}

struct lp_tree_node *gmmu_t::build_lp_tree(mem_addr_t addr, size_t size) {
  struct lp_tree_node *node = new lp_tree_node();
  node->addr = addr;
  node->size = size;
  node->valid_size = 0;
  node->access_counter = 0;
  node->RW = 0;

  if (size == MIN_PREFETCH_SIZE) {
    node->left = NULL;
    node->right = NULL;
  } else {
    node->left = build_lp_tree(addr, size / 2);
    node->right = build_lp_tree(addr + size / 2, size / 2);
  }
  return node;
}

void gmmu_t::initialize_large_page(mem_addr_t start_addr, size_t size) {
  struct lp_tree_node *root = build_lp_tree(start_addr, size);

  large_page_info.push_back(root);

  total_allocation_size += size;
}

struct lp_tree_node *gmmu_t::get_lp_node(mem_addr_t addr) {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    if ((*iter)->addr <= addr && addr < (*iter)->addr + (*iter)->size) {
      return *iter;
    }
  }
  return NULL;
}

mem_addr_t gmmu_t::get_basic_block(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->addr;
}

void gmmu_t::evict_whole_tree(struct lp_tree_node *node) {
  if (node != NULL) {
    node->valid_size = 0;
    evict_whole_tree(node->left);
    evict_whole_tree(node->right);
  }
}

mem_addr_t gmmu_t::update_basic_block(struct lp_tree_node *node,
                                      mem_addr_t addr, size_t size,
                                      bool prefetch) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (prefetch) {
      if (node->valid_size != node->size) {
        node->valid_size += size;
      }
    } else {
      if (node->valid_size != 0) {
        node->valid_size -= size;
      }
    }

    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  if (prefetch) {
    if (node->valid_size != node->size) {
      node->valid_size += size;
    }
  } else {
    if (node->valid_size != 0) {
      node->valid_size -= size;
    }
  }

  return node->addr;
}

void gmmu_t::fill_lp_tree(struct lp_tree_node *node,
                          std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == MIN_PREFETCH_SIZE) {
    if (node->valid_size == 0) {
      node->valid_size = MIN_PREFETCH_SIZE;
      scheduled_basic_blocks.insert(node->addr);
    }
  } else {
    fill_lp_tree(node->left, scheduled_basic_blocks);
    fill_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
  }
}

void gmmu_t::remove_lp_tree(struct lp_tree_node *node,
                            std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == MIN_PREFETCH_SIZE) {
    if (node->valid_size == MIN_PREFETCH_SIZE &&
        is_block_evictable(node->addr, MIN_PREFETCH_SIZE)) {
      node->valid_size = 0;
      scheduled_basic_blocks.insert(node->addr);
    }
  } else {
    remove_lp_tree(node->left, scheduled_basic_blocks);
    remove_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
  }
}

void gmmu_t::traverse_and_fill_lp_tree(
    struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size != MIN_PREFETCH_SIZE) {
    traverse_and_fill_lp_tree(node->left, scheduled_basic_blocks);
    traverse_and_fill_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;

    if (node->valid_size != node->size && node->valid_size > node->size / 2) {
      fill_lp_tree(node, scheduled_basic_blocks);
    }
  }
}

void gmmu_t::traverse_and_remove_lp_tree(
    struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size != MIN_PREFETCH_SIZE) {
    traverse_and_remove_lp_tree(node->left, scheduled_basic_blocks);
    traverse_and_remove_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;

    if (node->valid_size != 0 && node->valid_size < node->size / 2) {
      remove_lp_tree(node, scheduled_basic_blocks);
    }
  }
}

void gmmu_t::reserve_pages_insert(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  printf("Yechen - gmmu_t::reserve_pages_insert : page %llu and uid %u will be inserted\n", page_num, ma_uid);
  if (find(reserve_pages[page_num].begin(), reserve_pages[page_num].end(),
           ma_uid) == reserve_pages[page_num].end()) {
    reserve_pages[page_num].push_back(ma_uid);
  }
}

void gmmu_t::reserve_pages_remove(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  printf("Yechen - gmmu_t::reserve_pages_remove : page %llu and uid %u will be removed\n", page_num, ma_uid);
  fflush(stdout);
  assert(reserve_pages.find(page_num) != reserve_pages.end());

  std::list<unsigned>::iterator iter = std::find(
      reserve_pages[page_num].begin(), reserve_pages[page_num].end(), ma_uid);

  assert(iter != reserve_pages[page_num].end());

  reserve_pages[page_num].erase(iter);

  if (reserve_pages[page_num].empty()) {
    reserve_pages.erase(page_num);
  }
}

bool gmmu_t::reserve_pages_check(mem_addr_t addr) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  return reserve_pages.find(page_num) == reserve_pages.end();
}

void gmmu_t::update_hardware_prefetcher_oversubscribed() {
  if (oversub_prefetcher == hwardware_prefetcher_oversub::DISBALED) {
    prefetcher = hwardware_prefetcher::DISBALED;
  } else if (oversub_prefetcher == hwardware_prefetcher_oversub::TBN) {
    prefetcher = hwardware_prefetcher::TBN;
  } else if (oversub_prefetcher ==
             hwardware_prefetcher_oversub::SEQUENTIAL_LOCAL) {
    prefetcher = hwardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (oversub_prefetcher == hwardware_prefetcher_oversub::RANDOM) {
    prefetcher = hwardware_prefetcher::RANDOM;
  }
}

void gmmu_t::log_kernel_info(unsigned kernel_id, unsigned long long time,
                             bool finish) {
  if (!finish) {
    kernel_info.insert(std::make_pair(kernel_id, std::make_pair(time, 0)));
  } else {
    std::map<unsigned,
             std::pair<unsigned long long, unsigned long long>>::iterator it =
        kernel_info.find(kernel_id);
    if (it != kernel_info.end()) {
      it->second.second = time;
    }
  }
}

void gmmu_t::update_memory_management_policy() {
  std::map<std::string, ds_pattern> accessPatterns;

  int i = 1;
  std::map<std::pair<mem_addr_t, size_t>, std::string> dataStructures;
  std::map<std::string, std::list<mem_addr_t>> dsUniqueBlocks;

  // get the managed allocations
  const std::map<uint64_t, struct allocation_info *> &managedAllocations =
      m_gpu->gpu_get_managed_allocations();

  // loop over managed allocations to create three maps
  // 1. data structures - key: pair of start addr and size; value: ds_i
  // 2. access pattern: key: ds_i; value: UNDECIDED pattern
  // 3. unique accessed blocks for reuse: key: ds_i; value: empty list of block
  // start address
  for (std::map<uint64_t, struct allocation_info *>::const_iterator iter =
           managedAllocations.begin();
       iter != managedAllocations.end(); iter++) {
    dataStructures.insert(
        std::make_pair(std::make_pair(iter->second->gpu_mem_addr,
                                      iter->second->allocation_size),
                       std::string("ds" + std::to_string(i))));

    accessPatterns.insert(std::make_pair(std::string("ds" + std::to_string(i)),
                                         ds_pattern::UNDECIDED));
    dsUniqueBlocks.insert(std::make_pair(std::string("ds" + std::to_string(i)),
                                         std::list<mem_addr_t>()));
    i++;
  }

  // create three level hierarchy for kernel-wise then data-structure wise block
  // address first level: name of kernel (k_i); second level: ds_i; third level:
  // block addresses ordered by access time
  std::map<unsigned, std::map<std::string, std::list<mem_addr_t>>>
      kernel_pattern;

  for (std::map<unsigned,
                std::pair<unsigned long long, unsigned long long>>::iterator
           k_iter = kernel_info.begin();
       k_iter != kernel_info.end(); k_iter++) {

    unsigned long long start = k_iter->second.first;
    unsigned long long end = k_iter->second.second;

    std::map<std::string, std::list<mem_addr_t>> dsAccess;

    for (std::list<std::pair<unsigned long long, mem_addr_t>>::iterator
             acc_iter = block_access_list.begin();
         acc_iter != block_access_list.end(); acc_iter++) {

      unsigned long long access_cycle = acc_iter->first;
      mem_addr_t block_addr = acc_iter->second;

      if (access_cycle >= start && ((end == 0) || (access_cycle <= end))) {

        for (std::map<std::pair<mem_addr_t, size_t>, std::string>::iterator
                 ds_iter = dataStructures.begin();
             ds_iter != dataStructures.end(); ds_iter++) {

          if (block_addr >= ds_iter->first.first &&
              block_addr < ds_iter->first.first + ds_iter->first.second) {
            dsAccess[ds_iter->second].push_back(block_addr);
          }
        }
      }
    }

    kernel_pattern.insert(std::make_pair(k_iter->first, dsAccess));
  }

  // determine pattern per data structure
  // first loop on kernel level then on data structures accessed in that kernel
  for (std::map<unsigned,
                std::map<std::string, std::list<mem_addr_t>>>::iterator k_iter =
           kernel_pattern.begin();
       k_iter != kernel_pattern.end(); k_iter++) {

    for (std::map<std::string, std::list<mem_addr_t>>::iterator da_iter =
             k_iter->second.begin();
         da_iter != k_iter->second.end(); da_iter++) {

      // get the sorted list of block addresses belonging to the current
      // data-structure in current kernel
      std::list<mem_addr_t> curBlocks = std::list<mem_addr_t>(da_iter->second);
      curBlocks.sort();
      curBlocks.unique();

      // check for data reuse
      bool reuse = false;

      // first within this kernel
      // if the number of unique blocks accessed and total number of blocks
      // accessed are not same then there is repetition
      if (curBlocks.size() != da_iter->second.size()) {
        reuse = true;
      }

      // second check if the current accessed blocks are already seen or not
      std::map<std::string, std::list<mem_addr_t>>::iterator ub_it =
          dsUniqueBlocks.find(da_iter->first);

      // check for intersection between unique blocks accessed in current kernel
      // and the previous kernels is null set or not
      std::list<int> intersection;
      std::set_intersection(curBlocks.begin(), curBlocks.end(),
                            ub_it->second.begin(), ub_it->second.end(),
                            std::back_inserter(intersection));

      if (intersection.size() != 0) {
        reuse = true;
      }

      // add the current blocks to the seen set per data structure
      ub_it->second.merge(curBlocks);
      ub_it->second.sort();
      ub_it->second.unique();

      // now update the pattern
      std::map<std::string, ds_pattern>::iterator dsp_it =
          accessPatterns.find(da_iter->first);
      ds_pattern curPattern;

      // check for linearity or randomness in current kernel
      if (std::is_sorted(da_iter->second.begin(), da_iter->second.end())) {
        if (reuse) {
          curPattern = ds_pattern::LINEAR_REUSE;
        } else {
          curPattern = ds_pattern::LINEAR;
        }
      } else {
        if (reuse) {
          curPattern = ds_pattern::RANDOM_REUSE;
        } else {
          curPattern = ds_pattern::RANDOM;
        }
      }

      // determine the pattern
      if (dsp_it->second == ds_pattern::UNDECIDED) {
        dsp_it->second = curPattern;
      } else if (dsp_it->second == ds_pattern::LINEAR) {
        if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::LINEAR_REUSE;
        } else if (curPattern == ds_pattern::RANDOM) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::LINEAR_REUSE) {
        if (curPattern == ds_pattern::RANDOM ||
            curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM) {
        if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::RANDOM_REUSE;
        } else if (curPattern == ds_pattern::LINEAR) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM_REUSE) {
        if (curPattern == ds_pattern::LINEAR ||
            curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      }
    }
  }

  bool is_random = false, is_random_reuse = false, is_linear = false,
       is_linear_reuse = false, is_mixed = false, is_mixed_reuse = false;

  for (std::map<std::string, ds_pattern>::iterator ap_iter =
           accessPatterns.begin();
       ap_iter != accessPatterns.end(); ap_iter++) {
    if (ap_iter->second == ds_pattern::RANDOM) {
      is_random = true;
    } else if (ap_iter->second == ds_pattern::RANDOM_REUSE) {
      is_random_reuse = true;
    } else if (ap_iter->second == ds_pattern::LINEAR) {
      is_linear = true;
    } else if (ap_iter->second == ds_pattern::LINEAR_REUSE) {
      is_linear_reuse = true;
    } else if (ap_iter->second == ds_pattern::MIXED) {
      is_mixed = true;
    } else if (ap_iter->second == ds_pattern::MIXED_REUSE) {
      is_mixed_reuse = true;
    }
  }

  if (is_random || is_random_reuse || is_mixed || is_mixed_reuse) {
    dma_mode = dma_type::OVERSUB;
    evict_policy = eviction_policy::TBN;
  } else if (is_linear_reuse) {
    evict_policy = eviction_policy::TBN;
  }
}

void gmmu_t::reset_lp_tree_node(struct lp_tree_node *node) {
  node->valid_size = 0;
  node->access_counter = 0;
  node->RW = 0;

  if (node->size != MIN_PREFETCH_SIZE) {
    reset_lp_tree_node(node->left);
    reset_lp_tree_node(node->right);
  }
}

void gmmu_t::reset_large_page_info() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    reset_lp_tree_node(*iter);
  }

  over_sub = false;
}

mem_addr_t gmmu_t::get_eviction_base_addr(mem_addr_t page_addr) {
  mem_addr_t lru_addr;

  struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(page_addr);

  if (evict_policy == eviction_policy::TBN ||
      evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_addr = m_gpu->getGmmu()->get_basic_block(root, page_addr);
  } else if (evict_policy == eviction_policy::LRU4K ||
             evict_policy == eviction_policy::RANDOM) {
    lru_addr = page_addr;
  } else {
    lru_addr = root->addr;
  }

  return lru_addr;
}

size_t gmmu_t::get_eviction_granularity(mem_addr_t page_addr) {
  size_t lru_size;

  struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(page_addr);

  if (evict_policy == eviction_policy::TBN ||
      evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_size = MIN_PREFETCH_SIZE;
  } else if (evict_policy == eviction_policy::LRU4K ||
             evict_policy == eviction_policy::RANDOM) {
    lru_size = m_config.page_size;
  } else {
    lru_size = root->size;
  }

  return lru_size;
}

void gmmu_t::update_access_type(mem_addr_t addr, int type) {
  struct lp_tree_node *node = m_gpu->getGmmu()->get_lp_node(addr);

  while (node->size != MIN_PREFETCH_SIZE) {
    node->RW |= type;

    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  node->RW |= type;
}

int gmmu_t::get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->access_counter & ((1 << 27) - 1);
}

int gmmu_t::get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
}

void gmmu_t::inc_bb_access_counter(mem_addr_t addr) {
  struct lp_tree_node *node = m_gpu->getGmmu()->get_lp_node(addr);

  while (node->size != MIN_PREFETCH_SIZE) {
    node->access_counter++;

    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  if (node->access_counter == ((1 << 27) - 1)) {
    reset_bb_access_counter();
  }

  node->access_counter++;
}

void gmmu_t::inc_bb_round_trip(struct lp_tree_node *node) {
  if (node->size != MIN_PREFETCH_SIZE) {
    inc_bb_round_trip(node->left);
    inc_bb_round_trip(node->right);
  } else {
    uint16_t round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;

    if (round_trip == ((1 << 6) - 1)) {
      reset_bb_round_trip();
    }

    round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    round_trip++;

    node->access_counter =
        (round_trip << 27) | (node->access_counter & ((1 << 27) - 1));
  }
}

void gmmu_t::traverse_and_reset_access_counter(struct lp_tree_node *node) {
  if (node->size == MIN_PREFETCH_SIZE) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    int access_counter = (node->access_counter & ((1 << 27) - 1)) >> 1;

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
    node->access_counter = node->access_counter >> 1;
  }
}

void gmmu_t::reset_bb_access_counter() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    traverse_and_reset_access_counter(*iter);
  }
}

void gmmu_t::traverse_and_reset_round_trip(struct lp_tree_node *node) {
  if (node->size == MIN_PREFETCH_SIZE) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 28;
    int access_counter = node->access_counter & ((1 << 27) - 1);

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
  }
}

void gmmu_t::reset_bb_round_trip() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    traverse_and_reset_round_trip(*iter);
  }
}

bool gmmu_t::should_cause_page_migration(mem_addr_t addr, bool is_write) {
  if (dma_mode == dma_type::DISABLED) {
    return true;
  } else if (dma_mode == dma_type::ALWAYS) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

      if (get_bb_access_counter(root, addr) < m_config.migrate_threshold) {
        return false;
      } else {
        return true;
      }
    }
  } else if (dma_mode == dma_type::OVERSUB) {
    if (over_sub) {
      if (is_write) {
        return true;
      } else {
        struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

        if (get_bb_access_counter(root, addr) < m_config.migrate_threshold) {
          return false;
        } else {
          return true;
        }
      }
    } else {
      return true;
    }
  } else if (dma_mode == dma_type::ADAPTIVE) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

      int derived_threshold;

      if (over_sub) {
        derived_threshold = m_config.migrate_threshold *
                            m_config.multiply_dma_penalty *
                            (get_bb_round_trip(root, addr) + 1);
      } else {
        size_t num_read_stage_queue = 0;

        for (std::list<pcie_latency_t *>::iterator iter =
                 pcie_read_stage_queue.begin();
             iter != pcie_read_stage_queue.end(); iter++) {
          num_read_stage_queue += (*iter)->page_list.size();
        }

        size_t num_write_stage_queue = 0;

        for (std::list<pcie_latency_t *>::iterator iter =
                 pcie_write_stage_queue.begin();
             iter != pcie_write_stage_queue.end(); iter++) {
          num_write_stage_queue += (*iter)->page_list.size();
        }

        derived_threshold =
            (int)(1.0 + m_config.migrate_threshold *
                            m_gpu->get_global_memory()->get_projected_occupancy(
                                num_read_stage_queue, num_write_stage_queue,
                                m_config.free_page_buffer_percentage));
      }

      if (get_bb_access_counter(root, addr) < derived_threshold) {
        return false;
      } else {
        return true;
      }
    }
  }
}

void gmmu_t::cycle() {
  int simt_cluster_id = 0;

  size_t num_read_stage_queue = 0;

  for (std::list<pcie_latency_t *>::iterator iter =
           pcie_read_stage_queue.begin();
       iter != pcie_read_stage_queue.end(); iter++) {
    num_read_stage_queue += (*iter)->page_list.size();
  }

  size_t num_write_stage_queue = 0;

  for (std::list<pcie_latency_t *>::iterator iter =
           pcie_write_stage_queue.begin();
       iter != pcie_write_stage_queue.end(); iter++) {
    num_write_stage_queue += (*iter)->page_list.size();
  }

  num_write_stage_queue += pcie_write_latency_queue != NULL
                               ? pcie_write_latency_queue->page_list.size()
                               : 0;

  if (m_gpu->get_global_memory()->should_evict_page(
          num_read_stage_queue, num_write_stage_queue,
          m_config.free_page_buffer_percentage)) {

    if (m_config.enable_smart_runtime) {
      update_memory_management_policy();
    }

    page_eviction_procedure();
  }

  // check whether current transfer in the pcie write latency queue is finished
  if (pcie_write_latency_queue != NULL &&
      (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
          pcie_write_latency_queue->ready_cycle) {

    for (std::list<mem_addr_t>::iterator iter =
             pcie_write_latency_queue->page_list.begin();
         iter != pcie_write_latency_queue->page_list.end(); iter++) {
      m_gpu->gpu_writeback(m_gpu->get_global_memory()->get_mem_addr(*iter));
    }

    if (sim_prof_enable) {
      for (std::list<event_stats *>::iterator iter = writeback_stats.begin();
           iter != writeback_stats.end(); iter++) {
        if (((memory_stats *)(*iter))->start_addr ==
            m_gpu->get_global_memory()->get_mem_addr(
                pcie_write_latency_queue->page_list.front())) {
          event_stats *wb = *iter;
          wb->end_time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
          sim_prof[wb->start_time].push_back(wb);
          writeback_stats.erase(iter);
          break;
        }
      }
    }

    pcie_write_latency_queue = NULL;
  }

  // schedule a write back transfer if there is a write back request in staging
  // queue and a free lane
  if (!pcie_write_stage_queue.empty() && pcie_write_latency_queue == NULL) {
    pcie_write_latency_queue = pcie_write_stage_queue.front();
    pcie_write_latency_queue->ready_cycle =
        get_ready_cycle(pcie_write_latency_queue->page_list.size());

    for (unsigned long long write_period = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
         write_period != pcie_write_latency_queue->ready_cycle; write_period++)
      m_new_stats->pcie_write_utilization.push_back(std::make_pair(
          write_period,
          get_pcie_utilization(pcie_write_latency_queue->page_list.size())));

    for (std::list<mem_addr_t>::iterator iter =
             pcie_write_latency_queue->page_list.begin();
         iter != pcie_write_latency_queue->page_list.end(); iter++) {
      m_new_stats->page_thrashing[*iter].push_back(false);

      if (m_gpu->get_global_memory()->is_page_dirty(*iter)) {
        m_new_stats->page_evict_dirty++;
      } else {
        m_new_stats->page_evict_not_dirty++;
      }

      m_gpu->get_global_memory()->invalidate_page(*iter);
      m_gpu->get_global_memory()->clear_page_dirty(*iter);
      m_gpu->get_global_memory()->clear_page_access(*iter);

      m_gpu->get_global_memory()->free_pages(1);

      tlb_flush(*iter);
    }

    struct lp_tree_node *root =
        m_gpu->getGmmu()->get_lp_node(m_gpu->get_global_memory()->get_mem_addr(
            pcie_write_latency_queue->page_list.front()));
    inc_bb_round_trip(root);

    if (sim_prof_enable) {
      if (pcie_write_latency_queue->type == latency_type::INVALIDATE &&
          m_config.invalidate_clean) {
        event_stats *inv = new memory_stats(
            invalidate, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            m_gpu->get_global_memory()->get_mem_addr(
                pcie_write_latency_queue->page_list.front()),
            pcie_write_latency_queue->page_list.size() * m_config.page_size, 0);
        sim_prof[inv->start_time].push_back(inv);
      } else {
        event_stats *wb = new memory_stats(
            write_back, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            m_gpu->get_global_memory()->get_mem_addr(
                pcie_write_latency_queue->page_list.front()),
            pcie_write_latency_queue->page_list.size() * m_config.page_size, 0);
        writeback_stats.push_back(wb);
      }
    }

    pcie_write_stage_queue.pop_front();

    if (pcie_write_latency_queue->type == latency_type::INVALIDATE &&
        m_config.invalidate_clean) {
      pcie_write_latency_queue = NULL;
    }
  }

  list<mem_addr_t> page_finsihed_for_mf;

  // check whether the current transfer in the pcie latency queue is finished
  if (pcie_read_latency_queue != NULL &&
      (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
          pcie_read_latency_queue->ready_cycle) {

    if (pcie_read_latency_queue->type == latency_type::PCIE_READ) {

      for (std::list<mem_addr_t>::iterator iter =
               pcie_read_latency_queue->page_list.begin();
           iter != pcie_read_latency_queue->page_list.end(); iter++) {
        // validate the page in page table
        m_gpu->get_global_memory()->validate_page(*iter);

        // add to the valid pages list
        refresh_valid_pages(m_gpu->get_global_memory()->get_mem_addr(*iter));

        m_new_stats->page_thrashing[*iter].push_back(true);

        // check if the transferred page is part of a prefetch request
        if (!prefetch_req_buffer.empty()) {

          prefetch_req &pre_q = prefetch_req_buffer.front();

          std::list<mem_addr_t>::iterator iter2 =
              find(pre_q.pending_prefetch.begin(), pre_q.pending_prefetch.end(),
                   *iter);

          if (iter2 != pre_q.pending_prefetch.end()) {

            // pending prefetch holds the list of 4KB pages of a big chunk of
            // tranfer (max upto 2MB) remove it from the list as the PCI-e has
            // transferred the page
            pre_q.pending_prefetch.erase(iter2);

            // if this page is part of current prefecth request
            // add all the dependant memory requests to the
            // outgoing_replayable_nacks these should be replayed only when
            // current block of memory transfer is finished
            pre_q.outgoing_replayable_nacks[*iter].merge(req_info[*iter]);

            // erase the page from the MSHR map
            req_info.erase(req_info.find(*iter));

            m_new_stats->pf_page_fault_latency[*iter].back() =
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
                m_new_stats->pf_page_fault_latency[*iter].back();
          }
        }

        // this page request is created by core on page fault and not part of a
        // prefetch
        if (req_info.find(*iter) != req_info.end()) {

          page_finsihed_for_mf.push_back(*iter);

          // for all memory fetches that were waiting for this page, should be
          // replayed back for cache access
          for (std::list<mem_fetch *>::iterator iter2 = req_info[*iter].begin();
               iter2 != req_info[*iter].end(); iter2++) {
            mem_fetch *mf = *iter2;

            simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();

            // push the memory fetch into the gmmu to cu queue
            (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
          }

          // erase the page from the MSHR map
          req_info.erase(req_info.find(*iter));

          m_new_stats->mf_page_fault_latency[*iter].back() =
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
              m_new_stats->pf_page_fault_latency[*iter].back();
        }
      }
    } else if (pcie_read_latency_queue->type ==
               latency_type::PAGE_FAULT) { // processed far-fault is returned to
                                           // upward queue

      if (sim_prof_enable) {
        for (std::list<event_stats *>::iterator iter = fault_stats.begin();
             iter != fault_stats.end(); iter++) {
          if (((page_fault_stats *)(*iter))->transfering_pages.front() ==
              pcie_read_latency_queue->page_list.front()) {
            event_stats *mf_fault = *iter;
            mf_fault->end_time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
            sim_prof[mf_fault->start_time].push_back(mf_fault);
            fault_stats.erase(iter);
            break;
          }
        }
      }
    } else if (pcie_read_latency_queue->type ==
               latency_type::DMA) { // processed DMA request is returned to
                                    // upward queue
      mem_fetch *mf = pcie_read_latency_queue->mf;

      simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();

      // push the memory fetch into the gmmu to cu queue
      (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
    }
    pcie_read_latency_queue = NULL;
  }

  // schedule a transfer if there is a pending item in staging queue and
  // nothing is being served at the read latency queue and we have available
  // free pages

  if (!pcie_read_stage_queue.empty() && pcie_read_latency_queue == NULL &&
      m_gpu->get_global_memory()->get_free_pages() >=
          pcie_read_stage_queue.front()->page_list.size()) {

    std::list<pcie_latency_t *>::const_iterator iter =
        pcie_read_stage_queue.begin();
    for (; iter != pcie_read_stage_queue.end(); iter++) {
      if ((*iter)->type == latency_type::DMA) {
        break;
      }
    }

    // prioritize dma before page migration
    if (iter == pcie_read_stage_queue.end()) {
      pcie_read_latency_queue = pcie_read_stage_queue.front();
    } else {
      pcie_read_latency_queue = *iter;
    }

    if (pcie_read_latency_queue->type == latency_type::PCIE_READ) {
      pcie_read_latency_queue->ready_cycle =
          get_ready_cycle(pcie_read_latency_queue->page_list.size());

      if (sim_prof_enable) {
        event_stats *cp_h2d =
            new memory_stats(memcpy_h2d, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                             pcie_read_latency_queue->ready_cycle,
                             pcie_read_latency_queue->start_addr,
                             pcie_read_latency_queue->size, 0);
        sim_prof[m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle].push_back(cp_h2d);
      }

      for (unsigned long long read_period = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
           read_period != pcie_read_latency_queue->ready_cycle; read_period++)
        m_new_stats->pcie_read_utilization.push_back(std::make_pair(
            read_period,
            get_pcie_utilization(pcie_read_latency_queue->page_list.size())));

      m_gpu->get_global_memory()->alloc_pages(
          pcie_read_latency_queue->page_list.size());
    } else if (pcie_read_latency_queue->type ==
               latency_type::PAGE_FAULT) { // schedule far-fault for transfer

      pcie_read_latency_queue->ready_cycle =
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle +
          m_config.page_fault_latency *
              pcie_read_latency_queue->page_list.size();

      if (sim_prof_enable) {
        event_stats *mf_fault = new page_fault_stats(
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            pcie_read_latency_queue->page_list,
            pcie_read_latency_queue->page_list.size() * m_config.page_size);
        fault_stats.push_back(mf_fault);
      }
    } else if (pcie_read_latency_queue->type ==
               latency_type::DMA) { // schedule DMA request for transfer
      pcie_read_latency_queue->ready_cycle =
          get_ready_cycle_dma(pcie_read_latency_queue->mf->get_access_size());
      if (sim_prof_enable) {
        event_stats *ma_dma =
            new memory_stats(dma, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                             pcie_read_latency_queue->ready_cycle,
                             pcie_read_latency_queue->mf->get_addr(),
                             pcie_read_latency_queue->mf->get_access_size(), 0);
        sim_prof[m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle].push_back(ma_dma);
      }
    }

    // remove the scheduled transfer from read stage queue
    if (iter == pcie_read_stage_queue.end()) {
      pcie_read_stage_queue.pop_front();
    } else {
      pcie_read_stage_queue.erase(iter);
    }
  }

  std::map<mem_addr_t, std::list<mem_fetch *>> page_fault_this_turn;

  // check the page_table_walk_delay_queue
  while (!page_table_walk_queue.empty() &&
         ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
          page_table_walk_queue.front().ready_cycle)) {

    mem_fetch *mf = page_table_walk_queue.front().mf;

    list<mem_addr_t> page_list = m_gpu->get_global_memory()->get_faulty_pages(
        mf->get_addr(), mf->get_access_size());

    simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
    // if there is no page fault, directly return to the upward queue of cluster
    if (page_list.empty()) {
      mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(
          mf->get_mem_access().get_addr());
      check_write_stage_queue(page_num, false);

      (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);

      m_new_stats->mf_page_hit[simt_cluster_id]++;
    } else {
      assert(page_list.size() == 1);

      m_new_stats->mf_page_miss[simt_cluster_id]++;

      // the page request is already there in MSHR either as a page fault or as
      // part of scheduled prefetch request
      if (req_info.find(*(page_list.begin())) != req_info.end()) {
        m_new_stats->mf_page_fault_pending++;
        req_info[*(page_list.begin())].push_back(mf);
      } else {

        // if the memory fetch is part of any requests in the prefetch command
        // buffer then add it to the incoming replayable_nacks
        std::list<prefetch_req>::iterator iter;

        for (iter = prefetch_req_buffer.begin();
             iter != prefetch_req_buffer.end(); iter++) {

          if (iter->start_addr <= mf->get_addr() &&
              mf->get_addr() < iter->start_addr + iter->size) {

            m_new_stats->mf_page_fault_pending++;

            iter->incoming_replayable_nacks[page_list.front()].push_back(mf);
            break;
          }
        }

        // if the memory fetch is not part of any request in the prefetch
        // command buffer
        if (iter == prefetch_req_buffer.end()) {

          // if dma is enabled/it is a write access/read access counter hasn't
          // reached thresold
          if (!should_cause_page_migration(mf->get_mem_access().get_addr(),
                                           mf->get_mem_access().get_type() ==
                                               GLOBAL_ACC_W)) {

            m_new_stats->num_dma++;
            pcie_latency_t *p_t = new pcie_latency_t();

            mf->set_dma();

            p_t->mf = mf;
            p_t->type = latency_type::DMA;

            pcie_read_stage_queue.push_back(p_t);
          } else {
            if (dma_mode != dma_type::DISABLED &&
                mf->get_mem_access().get_type() == GLOBAL_ACC_W) {
              m_new_stats->dma_page_transfer_write++;
            } else if (dma_mode != dma_type::DISABLED &&
                       mf->get_mem_access().get_type() == GLOBAL_ACC_R) {
              m_new_stats->dma_page_transfer_read++;
            }

            page_fault_this_turn[page_list.front()].push_back(mf);
          }
        }
      }
    }

    page_table_walk_queue.pop_front();
  }

  // call hardware prefetcher based on the current page faults
  do_hardware_prefetch(page_fault_this_turn);

  // fetch from cluster's cu to gmmu queue and push it into the page table way
  // delay queue
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {

    if (!(m_gpu->getSIMTCluster(i))->empty_cu_gmmu_queue()) {

      mem_fetch *mf = (m_gpu->getSIMTCluster(i))->front_cu_gmmu_queue();

      struct page_table_walk_latency_t pt_t;
      pt_t.mf = mf;
      pt_t.ready_cycle =
          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config.page_table_walk_latency;

      page_table_walk_queue.push_back(pt_t);

      (m_gpu->getSIMTCluster(i))->pop_cu_gmmu_queue();
    }
  }

  // check if there is an active outstanding prefetch request
  if (!prefetch_req_buffer.empty() && prefetch_req_buffer.front().active) {

    prefetch_req &pre_q = prefetch_req_buffer.front();

    // schedule for page transfers from the active prefetch request when there
    // is no pending transfer for the same can be the very first time or a
    // scheduled big chunk of pages (2MB) is finsihed just now
    if (pre_q.pending_prefetch.empty()) {

      // case when the last schedule finished, it is not the first time
      if (pre_q.cur_addr > pre_q.start_addr) {

        if (sim_prof_enable) {
          update_sim_prof_prefetch_break_down(m_gpu->gpu_sim_cycle +
                                              m_gpu->gpu_tot_sim_cycle);
        }

        m_new_stats->pf_fault_latency.back().second =
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
            m_new_stats->pf_fault_latency.back().second;

        // all the memory fetches created by core on page fault were aggreagted
        // earlier now they are replayed back together to the core
        for (map<mem_addr_t, std::list<mem_fetch *>>::iterator iter =
                 pre_q.outgoing_replayable_nacks.begin();
             iter != pre_q.outgoing_replayable_nacks.end(); iter++) {

          for (std::list<mem_fetch *>::iterator iter2 = iter->second.begin();
               iter2 != iter->second.end(); iter2++) {

            mem_fetch *mf = *iter2;

            simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
            // push them to the upward queue to replay them back to the
            // corresponding core in bulk
            (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
          }
        }
        pre_q.outgoing_replayable_nacks.clear();
      }

      // all the memory fetches have been replayed and
      // the prefetch request is completed entirely
      // now signal the stream that the operation is finished so that it can
      // schedule something else
      if (pre_q.cur_addr == pre_q.start_addr + pre_q.size) {

        pre_q.m_stream->record_next_done();

        if (sim_prof_enable) {
          update_sim_prof_prefetch(pre_q.start_addr, pre_q.size,
                                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        }

        prefetch_req_buffer.pop_front();
        return;
      }

      mem_addr_t start_addr = 0;

      pcie_latency_t *p_t = new pcie_latency_t();

      // break the loop if
      //  Case 1: reach the end of this prefetch
      //  Case 2: it reaches the 2MB line from starting of the allocation
      //  Case 3: it encounters a valid page in between
      do {
        // get the page number for the current updated address
        mem_addr_t page_num =
            m_gpu->get_global_memory()->get_page_num(pre_q.cur_addr);

        // update the current address by page size as we break a big chunk (2MB)
        // in the granularity of the smallest unit of page
        pre_q.cur_addr += m_config.page_size;

        // check for Case 3, i.e., we encounter a valid page
        if (m_gpu->get_global_memory()->is_valid(page_num)) {

          m_new_stats->pf_page_hit++;

          // check if this page is currently written back
          check_write_stage_queue(page_num, false);

          // break out of loop only when we have already scheduled some pages
          // for transfer if not we will continue skipping valid pages if any
          // until we find some invalid pages to transfer
          if (!pre_q.pending_prefetch.empty()) {
            break;
          }
        } else {

          m_new_stats->pf_page_miss++;

          // remember this page as pending under the prefetch request
          pre_q.pending_prefetch.push_back(page_num);

          if (start_addr == 0) {
            start_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
            p_t->start_addr = pre_q.cur_addr;
          }

          // just create a placeholder in MSHR for the memory fetches created by
          // core on page fault later in the time so that they go to outgoing
          // replayable nacks, rather than incoming
          req_info[page_num];

          // incoming nacks hold the list of page faults for the transfer which
          // has not been scheduled yet so instead of pushing them to MSHR and
          // then again getting back to the outgoing list directly switch
          // between the incoming and outgoing list of replayable nacks
          if (pre_q.incoming_replayable_nacks.find(page_num) !=
              pre_q.incoming_replayable_nacks.end()) {
            pre_q.outgoing_replayable_nacks[page_num].merge(
                pre_q.incoming_replayable_nacks[page_num]);
            pre_q.incoming_replayable_nacks.erase(page_num);
          }

          // schedule this page as it is not valid to the read stage queue
          p_t->page_list.push_back(page_num);
          m_new_stats->pf_page_fault_latency[page_num].push_back(
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        }

      } while (
          pre_q.cur_addr !=
              (pre_q.start_addr +
               pre_q.size) && // check for Case 1, i.e., we reached the end of
                              // prefetch request
          ((unsigned long long)(pre_q.cur_addr - pre_q.allocation_addr)) %
              ((unsigned long long)
                   MAX_PREFETCH_SIZE)); // Case 2: allowing maximum transfer
                                        // size as huge page size of 2MB

      if (!p_t->page_list.empty()) {
        p_t->size = p_t->page_list.size() * m_config.page_size;
        p_t->type = latency_type::PCIE_READ;
        pcie_read_stage_queue.push_back(p_t);
      }

      m_new_stats->pf_fault_latency.push_back(
          std::make_pair(pre_q.pending_prefetch.size() * m_config.page_size,
                         m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle));

      if (sim_prof_enable && !pre_q.pending_prefetch.empty()) {
        event_stats *cp_pref_bd = new memory_stats(
            prefetch_breakdown, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, start_addr,
            pre_q.pending_prefetch.size() * m_config.page_size,
            pre_q.m_stream->get_uid());
        sim_prof[m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle].push_back(cp_pref_bd);
      }
    }
  }
}

void gmmu_t::do_hardware_prefetch(
    std::map<mem_addr_t, std::list<mem_fetch *>> &page_fault_this_turn) {
  // now decide on transfers as a group of page faults and prefetches
  if (!page_fault_this_turn.empty()) {
    unsigned long long num_pages_read_stage_queue = 0;

    for (std::list<pcie_latency_t *>::iterator iter =
             pcie_read_stage_queue.begin();
         iter != pcie_read_stage_queue.end(); iter++) {
      num_pages_read_stage_queue += (*iter)->page_list.size();
    }

    std::list<std::list<mem_addr_t>> all_transfer_all_page;
    std::list<std::list<mem_addr_t>> all_transfer_faulty_pages;
    std::map<mem_addr_t, std::list<mem_fetch *>> temp_req_info;

    // create a tree structure large page -> basic blocks -> faulty pages
    std::map<mem_addr_t, std::map<mem_addr_t, std::list<mem_addr_t>>>
        block_tree;

    if (prefetcher == hwardware_prefetcher::DISBALED ||
        prefetcher == hwardware_prefetcher::RANDOM) {
      for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator it =
               page_fault_this_turn.begin();
           it != page_fault_this_turn.end(); it++) {
        std::list<mem_addr_t> temp_pages;
        temp_pages.push_back(it->first);

        mem_addr_t page_addr =
            m_gpu->get_global_memory()->get_mem_addr(it->first);
        struct lp_tree_node *root = get_lp_node(page_addr);
        update_basic_block(root, page_addr, m_config.page_size, true);

        all_transfer_all_page.push_back(temp_pages);
        all_transfer_faulty_pages.push_back(temp_pages);

        temp_req_info[it->first];

        if (prefetcher == hwardware_prefetcher::RANDOM) {
          struct lp_tree_node *root =
              get_lp_node(m_gpu->get_global_memory()->get_mem_addr(it->first));

          size_t random_size =
              (rand() % (root->size / m_config.page_size)) * m_config.page_size;

          if (random_size > root->size) {
            random_size -= root->size;
          }

          mem_addr_t prefetch_addr = root->addr + random_size;

          mem_addr_t prefetch_page_num =
              m_gpu->get_global_memory()->get_page_num(prefetch_addr);

          if (!m_gpu->get_global_memory()->is_valid(prefetch_page_num) &&
              page_fault_this_turn.find(prefetch_addr) ==
                  page_fault_this_turn.end() &&
              temp_req_info.find(prefetch_page_num) == temp_req_info.end() &&
              req_info.find(prefetch_page_num) == req_info.end()) {

            mem_addr_t page_addr =
                m_gpu->get_global_memory()->get_mem_addr(prefetch_page_num);
            struct lp_tree_node *root = get_lp_node(page_addr);
            update_basic_block(root, page_addr, m_config.page_size, true);

            all_transfer_all_page.back().push_back(prefetch_page_num);

            temp_req_info[prefetch_page_num];
          }
        }
      }
    } else {
      std::map<mem_addr_t, std::set<mem_addr_t>> lp_pf_groups;

      for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator it =
               page_fault_this_turn.begin();
           it != page_fault_this_turn.end(); it++) {
        mem_addr_t page_addr =
            m_gpu->get_global_memory()->get_mem_addr(it->first);

        struct lp_tree_node *root = get_lp_node(page_addr);

        lp_pf_groups[root->addr].insert(page_addr);
      }

      for (std::map<mem_addr_t, std::set<mem_addr_t>>::iterator lp_pf_iter =
               lp_pf_groups.begin();
           lp_pf_iter != lp_pf_groups.end(); lp_pf_iter++) {
        std::set<mem_addr_t> schedulable_basic_blocks;

        // list of all invalid pages and pages with fault from all basic blocks
        // to satisfy current transfer size
        std::list<mem_addr_t> cur_transfer_all_pages;
        std::list<mem_addr_t> cur_transfer_faulty_pages;

        for (std::set<mem_addr_t>::iterator pf_iter =
                 lp_pf_iter->second.begin();
             pf_iter != lp_pf_iter->second.end(); pf_iter++) {
          mem_addr_t page_addr = *pf_iter;

          struct lp_tree_node *root = get_lp_node(page_addr);

          mem_addr_t bb_addr =
              update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, true);

          schedulable_basic_blocks.insert(bb_addr);

          cur_transfer_faulty_pages.push_back(
              m_gpu->get_global_memory()->get_page_num(page_addr));
        }

        if (prefetcher == hwardware_prefetcher::TBN) {
          struct lp_tree_node *root = get_lp_node(lp_pf_iter->first);
          traverse_and_fill_lp_tree(root, schedulable_basic_blocks);
        }

        for (std::set<mem_addr_t>::iterator bb =
                 schedulable_basic_blocks.begin();
             bb != schedulable_basic_blocks.end(); bb++) {

          block_access_list.push_back(
              std::make_pair(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, *bb));

          // all the invalid pages in the current 64 K basic block of transfer
          std::list<mem_addr_t> all_block_pages =
              m_gpu->get_global_memory()->get_faulty_pages(*bb,
                                                           MIN_PREFETCH_SIZE);

          for (std::list<mem_addr_t>::iterator pg_iter =
                   all_block_pages.begin();
               pg_iter != all_block_pages.end(); pg_iter++) {
            if (temp_req_info.find(*pg_iter) == temp_req_info.end()) {
              // mark entry into mshr for all pages in the current basic block
              temp_req_info[*pg_iter];
              cur_transfer_all_pages.push_back(*pg_iter);
            }
          }
        }

        all_transfer_all_page.push_back(cur_transfer_all_pages);
        all_transfer_faulty_pages.push_back(cur_transfer_faulty_pages);
      }
    }

    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter =
             temp_req_info.begin();
         iter != temp_req_info.end(); iter++) {
      req_info[iter->first];
      req_info[iter->first].merge(iter->second);
    }

    std::list<std::list<mem_addr_t>>::iterator all_pg_iter =
        all_transfer_all_page.begin();
    std::list<std::list<mem_addr_t>>::iterator all_pf_iter =
        all_transfer_faulty_pages.begin();

    for (; all_pg_iter != all_transfer_all_page.end();
         all_pg_iter++, all_pf_iter++) {
      // now we found all the basic blocks for the current transfer size
      // we now decide on the splits based on page faults
      std::list<mem_addr_t>::iterator pf_iter = all_pf_iter->begin();
      std::list<mem_addr_t>::iterator pg_iter = all_pg_iter->begin();

      std::list<mem_addr_t>::iterator prev_pg_iter;

      while (pg_iter != all_pg_iter->end()) {

        // if there is a gap between current and last page
        // it can be if two basic blocks selected for current transfer size
        // is separated by other basic blocks
        // then we send this basic block (or remaining of so) for transfer
        if (pg_iter != all_pg_iter->begin()) {
          prev_pg_iter = pg_iter;
          --prev_pg_iter;

          if ((*pg_iter) != ((*prev_pg_iter) + 1)) {

            // add the current split for transfer
            pcie_latency_t *p_t = new pcie_latency_t();
            p_t->start_addr =
                m_gpu->get_global_memory()->get_mem_addr(all_pg_iter->front());
            p_t->page_list =
                std::list<mem_addr_t>(all_pg_iter->begin(), pg_iter);
            p_t->size = p_t->page_list.size() * m_config.page_size;
            p_t->type = latency_type::PCIE_READ;

            pcie_read_stage_queue.push_back(p_t);

            // remove the scheduled pages from all pages and move the pointer
            pg_iter = all_pg_iter->erase(all_pg_iter->begin(), pg_iter);
          }
        }

        // we found a page on which a page fault request is pending
        // now we split upto this and create a memory transfer
        if ((pf_iter != all_pf_iter->end()) && ((*pf_iter) == (*pg_iter))) {

          if (m_config.enable_accurate_simulation) {
            pcie_latency_t *f_t = new pcie_latency_t();
            f_t->page_list.push_back(*pf_iter);
            f_t->type = latency_type::PAGE_FAULT;
            pcie_read_stage_queue.push_back(f_t);
          }

          // add the current split for transfer
          pcie_latency_t *p_t = new pcie_latency_t();
          p_t->start_addr =
              m_gpu->get_global_memory()->get_mem_addr(all_pg_iter->front());
          p_t->page_list =
              std::list<mem_addr_t>(all_pg_iter->begin(), ++pg_iter);
          p_t->size = p_t->page_list.size() * m_config.page_size;
          p_t->type = latency_type::PCIE_READ;

          pcie_read_stage_queue.push_back(p_t);

          // remove the scheduled pages from all pages and move the pointer
          pg_iter = all_pg_iter->erase(all_pg_iter->begin(), pg_iter);
          pf_iter++;
        } else {
          pg_iter++;
        }
      }

      // prefetch the remaining from the 64K basic block
      if (!all_pg_iter->empty()) {
        pcie_latency_t *p_t = new pcie_latency_t();
        p_t->start_addr =
            m_gpu->get_global_memory()->get_mem_addr(all_pg_iter->front());
        p_t->page_list = *all_pg_iter;
        p_t->size = p_t->page_list.size() * m_config.page_size;
        p_t->type = latency_type::PCIE_READ;

        pcie_read_stage_queue.push_back(p_t);
      }
    }

    // adding statistics for prefetch
    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter2 =
             page_fault_this_turn.begin();
         iter2 != page_fault_this_turn.end(); iter2++) {
      assert(req_info[iter2->first].size() == 0);

      // add the pending prefecthes to the MSHR entry
      req_info[iter2->first] = iter2->second;

      m_new_stats->mf_page_fault_outstanding++;
      m_new_stats->mf_page_fault_pending += req_info[iter2->first].size() - 1;

      m_new_stats->mf_page_fault_latency[iter2->first].push_back(
          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    }

    if (!over_sub && m_gpu->get_global_memory()->should_evict_page(
                         num_pages_read_stage_queue + temp_req_info.size(), 0,
                         m_config.free_page_buffer_percentage)) {

      if (m_config.enable_smart_runtime) {
        update_memory_management_policy();
      } else {
        update_hardware_prefetcher_oversubscribed();
      }

      over_sub = true;
    }
  }
}

void gpgpu_sim::cycle() {
  int clock_mask = next_clock_domain();

  // the gmmu has the same clock as the core
  if (clock_mask & GMMU) {
    m_gmmu->cycle();
  }
  
  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        //if (mf) {
        //  printf("MEM_FETCH DEBUG: gpgpu_sim::cycle :: mf info %p\n", mf);
        //  mf->print(stdout);
        //}
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      m_memory_sub_partition[i]->accumulate_L2cache_stats(
          m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle();
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for GPUWattch
      m_cluster[i]->get_icnt_stats(
          m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
      m_cluster[i]->get_cache_stats(
          m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      m_cluster[i]->get_current_occupancy(
          gpu_occupancy.aggregate_warp_slot_filled,
          gpu_occupancy.aggregate_theoretical_warp_slots);
    }
    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    gpu_sim_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                  m_power_stats, m_config.gpu_stat_sample_freq,
                  gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                  gpu_sim_insn);
    }
#endif

    issue_block2core();
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f\% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster(int index) { return *(m_cluster + index); }

gmmu_t *gpgpu_sim::getGmmu() { return m_gmmu; }
