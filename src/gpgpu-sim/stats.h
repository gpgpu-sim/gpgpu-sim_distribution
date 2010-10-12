/* 
 * shader.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt and the
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
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
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

#ifndef STATS_INCLUDED
#define STATS_INCLUDED



enum mem_stage_access_type {
   C_MEM,
   T_MEM,
   S_MEM,
   G_MEM_LD,
   L_MEM_LD,
   G_MEM_ST,
   L_MEM_ST,
   N_MEM_STAGE_ACCESS_TYPE
};

enum mem_stage_stall_type {
   NO_RC_FAIL = 0, 
   BK_CONF,
   MSHR_RC_FAIL,
   ICNT_RC_FAIL,
   COAL_STALL,
   WB_ICNT_RC_FAIL,
   WB_CACHE_RSRV_FAIL,
   N_MEM_STAGE_STALL_TYPE
};

struct shader_core_stats 
{
   unsigned int gpgpu_n_load_insn;
   unsigned int gpgpu_n_store_insn;
   unsigned int gpgpu_n_shmem_insn;
   unsigned int gpgpu_n_tex_insn;
   unsigned int gpgpu_n_const_insn;
   unsigned int gpgpu_n_param_insn;
   unsigned int gpgpu_n_shmem_bkconflict;
   unsigned int gpgpu_n_cache_bkconflict;
   int          gpgpu_n_intrawarp_mshr_merge;
   unsigned int gpgpu_n_cmem_portconflict;
   int          gpgpu_n_partial_writes;
   unsigned int gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE][N_MEM_STAGE_STALL_TYPE];
   unsigned int gpu_reg_bank_conflict_stalls;
   unsigned int *shader_cycle_distro;
   unsigned int L1_write_miss;
   unsigned int L1_read_miss;
   unsigned int L1_texture_miss;
   unsigned int L1_const_miss;
   unsigned int L1_write_hit_on_miss;
   unsigned int L1_writeback;
   int *num_warps_issuable;
   int *num_warps_issuable_pershader;
   unsigned long long  gpu_sim_insn_no_ld_const;
   unsigned long long  gpu_completed_thread;
   unsigned int gpgpu_commit_pc_beyond_two;
   unsigned int gpu_stall_by_MSHRwb;
   unsigned int gpu_stall_shd_mem;
   unsigned int gpu_stall_sh2icnt;

   //memory access classification
   int gpgpu_n_mem_read_local;
   int gpgpu_n_mem_write_local;
   int gpgpu_n_mem_texture;
   int gpgpu_n_mem_const;
   int gpgpu_n_mem_read_global;
   int gpgpu_n_mem_write_global;
   int gpgpu_n_mem_read_inst;

   int n_pdom_sc_orig_stat;
   int n_pdom_sc_single_stat;

   unsigned made_write_mfs;
   unsigned made_read_mfs;
};

#endif
