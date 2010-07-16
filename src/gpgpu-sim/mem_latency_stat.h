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

extern unsigned long long  gpu_sim_cycle;
extern unsigned int gpu_n_mem;
extern unsigned int gpu_n_shader;
extern int gpgpu_dram_sched_queue_size;
extern int gpgpu_dram_scheduler;
extern unsigned int gpu_mem_n_bk;
#ifdef MEM_LATENCY_STAT_IMPL
   #define EXTERN_DEF
#else
   #define EXTERN_DEF extern
#endif

EXTERN_DEF int gpgpu_memlatency_stat = FALSE;

EXTERN_DEF unsigned max_mrq_latency;
EXTERN_DEF unsigned max_dq_latency;
EXTERN_DEF unsigned max_mf_latency;
EXTERN_DEF unsigned max_icnt2mem_latency;
EXTERN_DEF unsigned max_icnt2sh_latency;
EXTERN_DEF unsigned mrq_lat_table[32];
EXTERN_DEF unsigned dq_lat_table[32];
EXTERN_DEF unsigned mf_lat_table[32];
EXTERN_DEF unsigned icnt2mem_lat_table[24];
EXTERN_DEF unsigned icnt2sh_lat_table[24];
EXTERN_DEF unsigned mf_lat_pw_table[32]; //table storing values of mf latency Per Window
EXTERN_DEF unsigned mf_num_lat_pw;
EXTERN_DEF unsigned mf_tot_lat_pw; //total latency summed up per window. divide by mf_num_lat_pw to obtain average latency Per Window
EXTERN_DEF unsigned long long int mf_total_lat;
EXTERN_DEF unsigned long long int ** mf_total_lat_table; //mf latency sums[dram chip id][bank id]
EXTERN_DEF unsigned ** mf_max_lat_table; //mf latency sums[dram chip id][bank id]
EXTERN_DEF unsigned num_mfs;
EXTERN_DEF unsigned int ***bankwrites; //bankwrites[shader id][dram chip id][bank id]
EXTERN_DEF unsigned int ***bankreads; //bankreads[shader id][dram chip id][bank id]
EXTERN_DEF unsigned int **totalbankwrites; //bankwrites[dram chip id][bank id]
EXTERN_DEF unsigned int **totalbankreads; //bankreads[dram chip id][bank id]
EXTERN_DEF unsigned int **totalbankaccesses; //bankaccesses[dram chip id][bank id]
EXTERN_DEF unsigned int *requests_by_warp;
EXTERN_DEF unsigned int *MCB_accesses; //upon cache miss, tracks which memory controllers accessed by a warp
EXTERN_DEF unsigned int *num_MCBs_accessed; //tracks how many memory controllers are accessed whenever any thread in a warp misses in cache
EXTERN_DEF unsigned int *position_of_mrq_chosen; //position of mrq in m_queue chosen 
EXTERN_DEF unsigned *mf_num_lat_pw_perwarp;
EXTERN_DEF unsigned *mf_tot_lat_pw_perwarp; //total latency summed up per window per warp. divide by mf_num_lat_pw_perwarp to obtain average latency Per Window
EXTERN_DEF unsigned long long int *mf_total_lat_perwarp;
EXTERN_DEF unsigned *num_mfs_perwarp;
EXTERN_DEF unsigned *acc_mrq_length;

EXTERN_DEF unsigned ***mem_access_type_stats; // dram access type classification


void memlatstat_init( )
{
   unsigned i,j;

   max_mrq_latency = 0;
   max_dq_latency = 0;
   max_mf_latency = 0;
   max_icnt2mem_latency = 0;
   max_icnt2sh_latency = 0;
   memset(mrq_lat_table, 0, sizeof(unsigned)*32);
   memset(dq_lat_table, 0, sizeof(unsigned)*32);
   memset(mf_lat_table, 0, sizeof(unsigned)*32);
   memset(icnt2mem_lat_table, 0, sizeof(unsigned)*24);
   memset(icnt2sh_lat_table, 0, sizeof(unsigned)*24);
   memset(mf_lat_pw_table, 0, sizeof(unsigned)*32);
   mf_num_lat_pw = 0;
   mf_num_lat_pw_perwarp = (unsigned *) calloc((gpu_n_shader * gpu_n_thread_per_shader / warp_size)+1, sizeof(unsigned int));
   mf_tot_lat_pw_perwarp = (unsigned *) calloc((gpu_n_shader * gpu_n_thread_per_shader / warp_size)+1, sizeof(unsigned int));
   mf_total_lat_perwarp = (unsigned long long int *) calloc((gpu_n_shader * gpu_n_thread_per_shader / warp_size)+1, sizeof(unsigned long long int));
   num_mfs_perwarp = (unsigned *) calloc((gpu_n_shader * gpu_n_thread_per_shader / warp_size)+1, sizeof(unsigned int));
   acc_mrq_length = (unsigned *) calloc(gpu_n_mem, sizeof(unsigned int));
   mf_tot_lat_pw = 0; //total latency summed up per window. divide by mf_num_lat_pw to obtain average latency Per Window
   mf_total_lat = 0;
   num_mfs = 0;
   printf("*** Initializing Memory Statistics ***\n");
   requests_by_warp = (unsigned int*) calloc((gpu_n_shader * gpu_n_thread_per_shader / warp_size)+1, sizeof(unsigned int));
   totalbankreads = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   totalbankwrites = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   totalbankaccesses = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
   mf_total_lat_table = (unsigned long long int **) calloc(gpu_n_mem, sizeof(unsigned long long *));
   mf_max_lat_table = (unsigned **) calloc(gpu_n_mem, sizeof(unsigned *));
   bankreads = (unsigned int***) calloc(gpu_n_shader, sizeof(unsigned int**));
   bankwrites = (unsigned int***) calloc(gpu_n_shader, sizeof(unsigned int**));
   MCB_accesses = (unsigned int*) calloc(gpu_n_mem*4, sizeof(unsigned int));
   num_MCBs_accessed = (unsigned int*) calloc(gpu_n_mem*4+1, sizeof(unsigned int));
   if (gpgpu_dram_sched_queue_size) {
      position_of_mrq_chosen = (unsigned int*) calloc(gpgpu_dram_sched_queue_size, sizeof(unsigned int));
   } else
      position_of_mrq_chosen = (unsigned int*) calloc(1024, sizeof(unsigned int));
   for (i=0;i<gpu_n_shader ;i++ ) {
      bankreads[i] = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
      bankwrites[i] = (unsigned int**) calloc(gpu_n_mem, sizeof(unsigned int*));
      for (j=0;j<gpu_n_mem ;j++ ) {
         bankreads[i][j] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
         bankwrites[i][j] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      }
   }

   for (i=0;i<gpu_n_mem ;i++ ) {
      totalbankreads[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      totalbankwrites[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      totalbankaccesses[i] = (unsigned int*) calloc(gpu_mem_n_bk, sizeof(unsigned int));
      mf_total_lat_table[i] = (unsigned long long int*) calloc(gpu_mem_n_bk, sizeof(unsigned long long int));
      mf_max_lat_table[i] = (unsigned *) calloc(gpu_mem_n_bk, sizeof(unsigned));
   }

   mem_access_type_stats = (unsigned ***) malloc(NUM_MEM_ACCESS_TYPE * sizeof(unsigned **));
   for (i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
      int j;
      mem_access_type_stats[i] = (unsigned **) calloc(gpu_n_mem, sizeof(unsigned*));
      for (j=0; (unsigned) j< gpu_n_mem; j++) {
         mem_access_type_stats[i][j] = (unsigned *) calloc((gpu_mem_n_bk+1), sizeof(unsigned*));
      }
   }
}

void memlatstat_start(mem_fetch_t *mf)
{
   mf->timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
   mf->timestamp2 = 0;
}

// recorder the total latency
unsigned memlatstat_done(mem_fetch_t *mf)
{
   unsigned mf_latency;
   unsigned wid = mf->sid*gpu_n_warp_per_shader + mf->wid;
   mf_latency = (gpu_sim_cycle+gpu_tot_sim_cycle) - mf->timestamp;
   mf_num_lat_pw++;
   mf_num_lat_pw_perwarp[wid]++;
   mf_tot_lat_pw_perwarp[wid] += mf_latency;
   mf_tot_lat_pw += mf_latency;
   check_time_vector_update(mf->mshr->insts[0].uid,MR_2SH_FQ_POP,mf_latency, mf->type ) ;
   mf_lat_table[LOGB2(mf_latency)]++;
   shader_mem_lat_log(mf->sid, mf_latency);
   mf_total_lat_table[mf->chip][mf->bank] += mf_latency;
   if (mf_latency > max_mf_latency)
      max_mf_latency = mf_latency;
   return mf_latency;
}

void memlatstat_icnt2sh_push(mem_fetch_t *mf)
{
   mf->timestamp2 = gpu_sim_cycle+gpu_tot_sim_cycle;
}

void memlatstat_read_done(mem_fetch_t *mf)
{
   if (gpgpu_memlatency_stat) {
      unsigned mf_latency = memlatstat_done(mf);

      if (mf_latency > mf_max_lat_table[mf->chip][mf->bank]) {
         mf_max_lat_table[mf->chip][mf->bank] = mf_latency;
      }

      unsigned icnt2sh_latency;
      icnt2sh_latency = (gpu_tot_sim_cycle+gpu_sim_cycle) - mf->timestamp2;
      icnt2sh_lat_table[LOGB2(icnt2sh_latency)]++;
      if (icnt2sh_latency > max_icnt2sh_latency)
         max_icnt2sh_latency = icnt2sh_latency;
   }
}

void memlatstat_dram_access(mem_fetch_t *mf, unsigned dram_id, unsigned bank)
{
   assert(dram_id < gpu_n_mem);
   assert(bank < gpu_mem_n_bk);
   if (gpgpu_memlatency_stat) { 
      if (mf->write) {
         if ( (unsigned) mf->sid  <  gpu_n_shader  ) {   //do not count L2_writebacks here 
            bankwrites[mf->sid][dram_id][bank]++;
            shader_mem_acc_log( mf->sid, dram_id, bank, 'w');
         }
         totalbankwrites[dram_id][bank]++;
      } else {
         bankreads[mf->sid][dram_id][bank]++;
         shader_mem_acc_log( mf->sid, dram_id, bank, 'r');
         totalbankreads[dram_id][bank]++;
      }

      if (mf->pc != (unsigned) -1) {
         ptx_file_line_stats_add_dram_traffic(mf->pc, 1);
      }
      
      mem_access_type_stats[mf->mem_acc][dram_id][bank]++;
   }
}

void memlatstat_icnt2mem_pop(mem_fetch_t *mf)
{
   if (gpgpu_memlatency_stat) {
      unsigned icnt2mem_latency;
      icnt2mem_latency = (gpu_tot_sim_cycle+gpu_sim_cycle) - mf->timestamp;
      icnt2mem_lat_table[LOGB2(icnt2mem_latency)]++;
      if (icnt2mem_latency > max_icnt2mem_latency)
         max_icnt2mem_latency = icnt2mem_latency;
   }
}

void memlatstat_lat_pw( )
{
   unsigned i;
   if (mf_num_lat_pw && gpgpu_memlatency_stat) {
      assert(mf_tot_lat_pw);
      mf_total_lat = mf_tot_lat_pw;
      num_mfs = mf_num_lat_pw;
      mf_lat_pw_table[LOGB2(mf_tot_lat_pw/mf_num_lat_pw)]++;
      mf_tot_lat_pw = 0;
      mf_num_lat_pw = 0;
   }
   for (i=0;i < ((gpu_n_shader * gpu_n_thread_per_shader / warp_size)+1); i++) {
      if (mf_num_lat_pw_perwarp[i] && gpgpu_memlatency_stat) {
         assert(mf_tot_lat_pw_perwarp[i]);
         mf_total_lat_perwarp[i] += mf_tot_lat_pw_perwarp[i];
         num_mfs_perwarp[i] += mf_num_lat_pw_perwarp[i];
         //mf_lat_pw_table[LOGB2(mf_tot_lat_pw/mf_num_lat_pw)]++;
         mf_tot_lat_pw_perwarp[i] = 0;
         mf_num_lat_pw_perwarp[i] = 0;
      }
   }
}


void memlatstat_print( )
{
   unsigned i,j,k,l,m;
   unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses, min_chip_accesses;

   if (gpgpu_memlatency_stat) {
      printf("maxmrqlatency = %d \n", max_mrq_latency);
      printf("maxdqlatency = %d \n", max_dq_latency);
      printf("maxmflatency = %d \n", max_mf_latency);
      if (num_mfs) {
         printf("averagemflatency = %lld \n", mf_total_lat/num_mfs);
      }
      printf("max_icnt2mem_latency = %d \n", max_icnt2mem_latency);
      printf("max_icnt2sh_latency = %d \n", max_icnt2sh_latency);
      printf("mrq_lat_table:");
      for (i=0; i< 32; i++) {
         printf("%d \t", mrq_lat_table[i]);
      }
      printf("\n");
      printf("dq_lat_table:");
      for (i=0; i< 32; i++) {
         printf("%d \t", dq_lat_table[i]);
      }
      printf("\n");
      printf("mf_lat_table:");
      for (i=0; i< 32; i++) {
         printf("%d \t", mf_lat_table[i]);
      }
      printf("\n");
      printf("icnt2mem_lat_table:");
      for (i=0; i< 24; i++) {
         printf("%d \t", icnt2mem_lat_table[i]);
      }
      printf("\n");
      printf("icnt2sh_lat_table:");
      for (i=0; i< 24; i++) {
         printf("%d \t", icnt2sh_lat_table[i]);
      }
      printf("\n");
      printf("mf_lat_pw_table:");
      for (i=0; i< 32; i++) {
         printf("%d \t", mf_lat_pw_table[i]);
      }
      printf("\n");

      /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
      printf("maximum concurrent accesses to same row:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            printf("%9d ",max_conc_access2samerow[i][j]);
         }
         printf("\n");
      }

      /*MAXIMUM SERVICE TIME TO SAME ROW*/
      printf("maximum service time to same row:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            printf("%9d ",max_servicetime2samerow[i][j]);
         }
         printf("\n");
      }

      /*AVERAGE ROW ACCESSES PER ACTIVATE*/
      int total_row_accesses = 0;
      int total_num_activates = 0;
      printf("average row accesses per activate:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            total_row_accesses += row_access[i][j];
            total_num_activates += num_activates[i][j];
            printf("%9f ",(float) row_access[i][j]/num_activates[i][j]);
         }
         printf("\n");
      }
      printf("average row locality = %d/%d = %f\n", total_row_accesses, total_num_activates, (float)total_row_accesses/total_num_activates);
      /*MEMORY ACCESSES*/
      k = 0;
      l = 0;
      m = 0;
      max_bank_accesses = 0;
      max_chip_accesses = 0;
      min_bank_accesses = 0xFFFFFFFF;
      min_chip_accesses = 0xFFFFFFFF;
      printf("number of total memory accesses made:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            l = totalbankaccesses[i][j];
            if (l < min_bank_accesses)
               min_bank_accesses = l;
            if (l > max_bank_accesses)
               max_bank_accesses = l;
            k += l;
            m += l;
            printf("%9d ",l);
         }
         if (m < min_chip_accesses)
            min_chip_accesses = m;
         if (m > max_chip_accesses)
            max_chip_accesses = m;
         m = 0;
         printf("\n");
      }
      printf("total accesses: %d\n", k);
      if (min_bank_accesses)
         printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses, (float)max_bank_accesses/min_bank_accesses);
      else
         printf("min_bank_accesses = 0!\n");
      if (min_chip_accesses)
         printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses, (float)max_chip_accesses/min_chip_accesses);
      else
         printf("min_chip_accesses = 0!\n");

      /*READ ACCESSES*/
      k = 0;
      l = 0;
      m = 0;
      max_bank_accesses = 0;
      max_chip_accesses = 0;
      min_bank_accesses = 0xFFFFFFFF;
      min_chip_accesses = 0xFFFFFFFF;
      printf("number of total read accesses:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            l = totalbankreads[i][j];
            if (l < min_bank_accesses)
               min_bank_accesses = l;
            if (l > max_bank_accesses)
               max_bank_accesses = l;
            k += l;
            m += l;
            printf("%9d ",l);
         }
         if (m < min_chip_accesses)
            min_chip_accesses = m;
         if (m > max_chip_accesses)
            max_chip_accesses = m;
         m = 0;
         printf("\n");
      }
      printf("total reads: %d\n", k);
      if (min_bank_accesses)
         printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses, (float)max_bank_accesses/min_bank_accesses);
      else
         printf("min_bank_accesses = 0!\n");
      if (min_chip_accesses)
         printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses, (float)max_chip_accesses/min_chip_accesses);
      else
         printf("min_chip_accesses = 0!\n");

      /*WRITE ACCESSES*/
      k = 0;
      l = 0;
      m = 0;
      max_bank_accesses = 0;
      max_chip_accesses = 0;
      min_bank_accesses = 0xFFFFFFFF;
      min_chip_accesses = 0xFFFFFFFF;
      printf("number of total write accesses:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            l = totalbankwrites[i][j];
            if (l < min_bank_accesses)
               min_bank_accesses = l;
            if (l > max_bank_accesses)
               max_bank_accesses = l;
            k += l;
            m += l;
            printf("%9d ",l);
         }
         if (m < min_chip_accesses)
            min_chip_accesses = m;
         if (m > max_chip_accesses)
            max_chip_accesses = m;
         m = 0;
         printf("\n");
      }
      printf("total reads: %d\n", k);
      if (min_bank_accesses)
         printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses, (float)max_bank_accesses/min_bank_accesses);
      else
         printf("min_bank_accesses = 0!\n");
      if (min_chip_accesses)
         printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses, (float)max_chip_accesses/min_chip_accesses);
      else
         printf("min_chip_accesses = 0!\n");


      /*AVERAGE MF LATENCY PER BANK*/
      printf("average mf latency per bank:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            k = totalbankwrites[i][j] + totalbankreads[i][j];
            if (k)
               printf("%10lld", mf_total_lat_table[i][j] / k);
            else
               printf("    none  ");
         }
         printf("\n");
      }

      /*MAXIMUM MF LATENCY PER BANK*/
      printf("maximum mf latency per bank:\n");
      for (i=0;i<gpu_n_mem ;i++ ) {
         printf("dram[%d]: ", i);
         for (j=0;j<4 ;j++ ) {
            printf("%10d", mf_max_lat_table[i][j]);
         }
         printf("\n");
      }
   }

   if (gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
      printf("\nNumber of Memory Banks Accessed per Memory Operation per Warp (from 0):\n");
      unsigned long long accum_MCBs_accessed = 0;
      unsigned long long tot_mem_ops_per_warp = 0;
      for (i=0;i<= gpu_n_mem*4 ; i++ ) {
         accum_MCBs_accessed += i*num_MCBs_accessed[i];
         tot_mem_ops_per_warp += num_MCBs_accessed[i];
         printf("%d\t", num_MCBs_accessed[i]);
      }

      printf("\nAverage # of Memory Banks Accessed per Memory Operation per Warp=%f\n", (float)accum_MCBs_accessed/tot_mem_ops_per_warp);

      //printf("\nAverage Difference Between First and Last Response from Memory System per warp = ");


      printf("\nposition of mrq chosen\n");

      if (!gpgpu_dram_sched_queue_size)
         j = 1024;
      else
         j = gpgpu_dram_sched_queue_size;
      k=0;l=0;
      for (i=0;i< j; i++ ) {
         printf("%d\t", position_of_mrq_chosen[i]);
         k += position_of_mrq_chosen[i];
         l += i*position_of_mrq_chosen[i];
      }
      printf("\n");
      printf("\naverage position of mrq chosen = %f\n", (float)l/k);
   }
}

#endif /*MEM_LATENCY_STAT_H*/
