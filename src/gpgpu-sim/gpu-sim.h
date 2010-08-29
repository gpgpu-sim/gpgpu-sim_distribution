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

extern void         init_gpu();
extern void         gpu_reg_options(class OptionParser * opp);
extern unsigned int run_gpu_sim(int grid_num);
extern unsigned int get_converge_point(unsigned int pc, void *thd);
extern void         gpu_print_stat();
extern int          mem_ctrl_full( int mc_id );
extern void         dramqueue_latency_log_dump();
extern void         dump_pipeline_impl( int mask, int s, int m );

extern unsigned int L1_write_miss;
extern unsigned int L1_read_miss;
extern unsigned int L1_texture_miss;
extern unsigned int L1_const_miss;
extern unsigned int L1_write_hit_on_miss;
extern unsigned int L1_writeback;
extern unsigned int L1_const_miss;
extern bool gpgpu_perfect_mem;
extern bool gpgpu_no_dl1;
extern char *gpgpu_cache_texl1_opt;
extern char *gpgpu_cache_constl1_opt;
extern char *gpgpu_cache_dl1_opt;
extern unsigned int gpu_n_thread_per_shader;
extern unsigned int gpu_n_mshr_per_shader;
extern unsigned int gpu_n_shader;
extern unsigned int gpu_n_mem;
extern bool gpgpu_reg_bankconflict;
extern int gpgpu_dram_sched_queue_size;
extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;
extern unsigned long long  gpu_sim_insn;
extern unsigned int gpu_n_warp_per_shader;
extern unsigned int **max_conc_access2samerow;
extern unsigned int **max_servicetime2samerow;
extern unsigned int **row_access;
extern unsigned int **num_activates;
extern struct dram_timing **dram;
extern int *num_warps_issuable;
extern int *num_warps_issuable_pershader;
extern unsigned long long  gpu_sim_insn_no_ld_const;
extern unsigned long long  gpu_sim_insn_last_update;
extern unsigned long long  gpu_completed_thread;
extern class shader_core_ctx **sc;
extern unsigned int gpgpu_pre_mem_stages;
extern bool gpgpu_no_divg_load;
extern bool gpgpu_thread_swizzling;
extern bool gpgpu_strict_simd_wrbk;
extern unsigned int warp_conflict_at_writeback;
extern unsigned int gpgpu_commit_pc_beyond_two;
extern bool gpgpu_spread_blocks_across_cores;
extern int gpgpu_cflog_interval;
extern unsigned int gpu_stall_by_MSHRwb;
extern unsigned int gpu_stall_shd_mem;
extern unsigned int gpu_stall_sh2icnt;
extern bool gpgpu_operand_collector;
extern int gpgpu_operand_collector_num_units;
extern int gpgpu_operand_collector_num_units_sfu;
extern int gpu_runtime_stat_flag;
extern unsigned int *max_return_queue_length;
extern int gpgpu_partial_write_mask;
extern int gpgpu_n_mem_write_local;
extern int gpgpu_n_mem_write_global;
extern bool gpgpu_cache_wt_through;
extern double core_freq;
extern double icnt_freq;
extern double dram_freq;
extern double l2_freq;
extern int pdom_sched_type;
extern int n_pdom_sc_orig_stat;
extern int n_pdom_sc_single_stat;
extern bool gpgpu_cuda_sim;
extern int gpgpu_mem_address_mask;
extern bool g_interactive_debugger_enabled;
extern unsigned int gpu_n_mem_per_ctrlr;
extern unsigned int **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]
extern unsigned long long  gpu_tot_sim_insn;
extern unsigned int gpgpu_n_sent_writes;
extern unsigned int gpgpu_n_processed_writes;
extern int gpgpu_simd_model;
extern unsigned int gpu_mem_n_bk;
extern unsigned g_next_mf_request_uid;
extern int   g_ptx_inst_debug_to_file;
extern char* g_ptx_inst_debug_file;
extern int   g_ptx_inst_debug_thread_uid;

#endif
