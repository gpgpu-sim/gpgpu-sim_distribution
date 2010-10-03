/* 
 * mem_latency_stat.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the 
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


#ifndef MEM_LATENCY_STAT_H
#define MEM_LATENCY_STAT_H

#include <stdio.h>

class memory_stats_t {
public:
   memory_stats_t( unsigned n_mem, 
                   unsigned n_shader, 
                   struct shader_core_config *shader_config, 
                   struct memory_config *mem_config );

   unsigned memlatstat_done( class mem_fetch *mf, unsigned n_warp_per_shader );
   void memlatstat_read_done( class mem_fetch *mf, unsigned n_warp_per_shader);
   void memlatstat_dram_access( class mem_fetch *mf );
   void memlatstat_icnt2mem_pop( class mem_fetch *mf);
   void memlatstat_lat_pw( unsigned n_shader, unsigned n_thread_per_shader, unsigned warp_size );
   void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);

   void L2c_print_stat(unsigned n_mem);

   void print( FILE *fp );

   unsigned m_n_mem;
   unsigned m_n_shader;

   const struct shader_core_config *m_shader_config;
   const struct memory_config *m_memory_config;

   unsigned max_mrq_latency;
   unsigned max_dq_latency;
   unsigned max_mf_latency;
   unsigned max_icnt2mem_latency;
   unsigned max_icnt2sh_latency;
   unsigned mrq_lat_table[32];
   unsigned dq_lat_table[32];
   unsigned mf_lat_table[32];
   unsigned icnt2mem_lat_table[24];
   unsigned icnt2sh_lat_table[24];
   unsigned mf_lat_pw_table[32]; //table storing values of mf latency Per Window
   unsigned mf_num_lat_pw;
   unsigned max_warps;
   unsigned mf_tot_lat_pw; //total latency summed up per window. divide by mf_num_lat_pw to obtain average latency Per Window
   unsigned long long int mf_total_lat;
   unsigned long long int ** mf_total_lat_table; //mf latency sums[dram chip id][bank id]
   unsigned ** mf_max_lat_table; //mf latency sums[dram chip id][bank id]
   unsigned num_mfs;
   unsigned int ***bankwrites; //bankwrites[shader id][dram chip id][bank id]
   unsigned int ***bankreads; //bankreads[shader id][dram chip id][bank id]
   unsigned int **totalbankwrites; //bankwrites[dram chip id][bank id]
   unsigned int **totalbankreads; //bankreads[dram chip id][bank id]
   unsigned int **totalbankaccesses; //bankaccesses[dram chip id][bank id]
   unsigned int *num_MCBs_accessed; //tracks how many memory controllers are accessed whenever any thread in a warp misses in cache
   unsigned int *position_of_mrq_chosen; //position of mrq in m_queue chosen 
   unsigned *mf_num_lat_pw_perwarp;
   unsigned *mf_tot_lat_pw_perwarp; //total latency summed up per window per warp. divide by mf_num_lat_pw_perwarp to obtain average latency Per Window
   unsigned long long int *mf_total_lat_perwarp;
   unsigned *num_mfs_perwarp;
   unsigned *acc_mrq_length;
   
   unsigned ***mem_access_type_stats; // dram access type classification


   // stats
   unsigned L2_write_miss;
   unsigned L2_write_hit;
   unsigned L2_read_hit;
   unsigned L2_read_miss;
   unsigned int *L2_cbtoL2length;
   unsigned int *L2_cbtoL2writelength;
   unsigned int *L2_L2tocblength;
   unsigned int *L2_dramtoL2length;
   unsigned int *L2_dramtoL2writelength;
   unsigned int *L2_L2todramlength;
};

#endif /*MEM_LATENCY_STAT_H*/
