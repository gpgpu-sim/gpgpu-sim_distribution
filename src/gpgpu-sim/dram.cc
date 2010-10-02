/* 
 * dram.c
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, George L. Yuan,
 * Ivan Sham, Justin Kwong, Dan O'Connor and the 
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

#include "gpu-sim.h"
#include "gpu-misc.h"
#include "dram.h"
#include "mem_latency_stat.h"
#include "dram_sched.h"

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

template class fifo_pipeline<mem_fetch>;
template class fifo_pipeline<dram_req_t>;

dram_t::dram_t( unsigned int partition_id, struct memory_config *config )
{
   id = partition_id;
   m_stats = NULL;
   m_config = config;

   BL=m_config->gpgpu_dram_burst_length;
   busW=m_config->gpgpu_dram_buswidth;

   sscanf(m_config->gpgpu_dram_timing_opt,"%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",&nbk,&tCCD,&tRRD,&tRCD,&tRAS,&tRP,&tRC,&CL,&WL,&tWTR);
   m_config->gpu_mem_n_bk = nbk;

   tRCDWR = tRCD - (WL + 1); //formula given in datasheet
   tRTW = (CL+(BL/2)+2-WL); //read to write time according to datasheet

   CCDc = 0;
   RRDc = 0;
   RTWc = 0;
   WTRc = 0;

   rw = READ; //read mode is default

   bk = (bank_t**) calloc(sizeof(bank_t*),nbk);
   bk[0] = (bank_t*) calloc(sizeof(bank_t),nbk);
   for (unsigned i=1;i<nbk;i++) 
      bk[i] = bk[0] + i;
   for (unsigned i=0;i<nbk;i++) 
      bk[i]->state = BANK_IDLE;

   prio = 0;  
   rwq = new fifo_pipeline<dram_req_t>("rwq",CL,CL+1,gpu_sim_cycle);
   mrqq = new fifo_pipeline<dram_req_t>("mrqq",0,0,gpu_sim_cycle);
   returnq = new fifo_pipeline<mem_fetch>("dramreturnq",0,m_config->gpgpu_dram_sched_queue_size,gpu_sim_cycle); 
   m_fast_ideal_scheduler = NULL;
   if ( m_config->scheduler_type == DRAM_IDEAL_FAST )
      m_fast_ideal_scheduler = new ideal_dram_scheduler(this);
   n_cmd = 0;
   n_activity = 0;
   n_nop = 0; 
   n_act = 0; 
   n_pre = 0; 
   n_rd = 0;
   n_wr = 0;
   n_req = 0;
   max_mrqs_temp = 0;
   bwutil = 0;
   max_mrqs = 0;

   for (unsigned i=0;i<10;i++) {
      dram_util_bins[i]=0;
      dram_eff_bins[i]=0;
   }
   last_n_cmd = last_n_activity = last_bwutil = 0;

   n_cmd_partial = 0;
   n_activity_partial = 0;
   n_nop_partial = 0;  
   n_act_partial = 0;  
   n_pre_partial = 0;  
   n_req_partial = 0;
   ave_mrqs_partial = 0;
   bwutil_partial = 0;

   if ( queue_limit() )
      mrqq_Dist = StatCreate("mrqq_length",1, queue_limit());
   else //queue length is unlimited; 
      mrqq_Dist = StatCreate("mrqq_length",1,64); //track up to 64 entries
}

int dram_t::full() 
{
   int full = 0;
   if ( m_config->gpgpu_dram_sched_queue_size == 0 ) return 0;
   if ( m_config->scheduler_type == DRAM_IDEAL_FAST ) {
      unsigned nreqs = m_fast_ideal_scheduler->num_pending() + mrqq->get_n_element();
      full = (nreqs >= m_config->gpgpu_dram_sched_queue_size);
   } else {
      full = (mrqq->get_length() >= m_config->gpgpu_dram_sched_queue_size);
   }

   return full;
}

unsigned int dram_t::que_length() const
{
   unsigned nreqs = 0;
   if (m_config->scheduler_type == DRAM_IDEAL_FAST ) {
      nreqs = m_fast_ideal_scheduler->num_pending();
   } else {
      nreqs = mrqq->get_length();
   }
   return nreqs;
}

bool dram_t::returnq_full() const
{
   return returnq->full();
}

unsigned int dram_t::queue_limit() const 
{ 
   return m_config->gpgpu_dram_sched_queue_size; 
}


dram_req_t::dram_req_t( class mem_fetch *mf )
{
   bk = mf->tlx.bk; 
   row = mf->tlx.row; 
   col = mf->tlx.col; 
   nbytes = mf->nbytes_L1;
   txbytes = 0;
   dqbytes = 0;
   data = mf;
   timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
   cache_hits_waiting = mf->cache_hits_waiting;
   addr = mf->addr;
   insertion_time = (unsigned) gpu_sim_cycle;
   rw = data->m_write?WRITE:READ;
}

void dram_t::push( class mem_fetch *data ) 
{
   assert(data->tlx.bk<nbk);
   dram_req_t *mrq = new dram_req_t(data);
   mrqq->push(mrq,gpu_sim_cycle);

   // stats...
   n_req += 1;
   n_req_partial += 1;
   if ( m_config->scheduler_type == DRAM_IDEAL_FAST ) {
      unsigned nreqs = m_fast_ideal_scheduler->num_pending();
      if ( nreqs > max_mrqs_temp)
         max_mrqs_temp = nreqs;
   } else {
      max_mrqs_temp = (max_mrqs_temp > mrqq->get_length())? max_mrqs_temp : mrqq->get_length();
   }
   m_stats->memlatstat_dram_access(data);
}

void dram_t::scheduler_fifo()
{
   if (!mrqq->empty()) {
      unsigned int bkn;
      dram_req_t *head_mrqq = mrqq->top();
      bkn = head_mrqq->bk;
      if (!bk[bkn]->mrq) {
         bk[bkn]->mrq = mrqq->pop(gpu_sim_cycle);
      }
   }
}


#define DEC2ZERO(x) x = (x)? (x-1) : 0;
#define SWAP(a,b) a ^= b; b ^= a; a ^= b;

void dram_t::issueCMD()
{
   unsigned i,j,k;
   unsigned char issued;
   issued = 0;

   /* check if the upcoming request is on an idle bank */
   /* Should we modify this so that multiple requests are checked? */

   switch (m_config->scheduler_type) {
   case DRAM_FIFO: scheduler_fifo(); break;
   case DRAM_IDEAL_FAST: fast_scheduler_ideal(); break;
	default:
		printf("Error: Unknown DRAM scheduler type\n");
		assert(0);
   }
   if ( m_config->scheduler_type == DRAM_IDEAL_FAST ) {
      unsigned nreqs = m_fast_ideal_scheduler->num_pending();
      if ( nreqs > max_mrqs) {
         max_mrqs = nreqs;
      }
      ave_mrqs += nreqs;
      ave_mrqs_partial += nreqs;
   } else {
      if (mrqq->get_length() > max_mrqs) {
         max_mrqs = mrqq->get_length();
      }
      ave_mrqs += mrqq->get_length();
      ave_mrqs_partial +=  mrqq->get_length();
   }
   k=nbk;
   // check if any bank is ready to issue a new read
   for (i=0;i<nbk;i++) {
      j = (i + prio) % nbk;
      if (bk[j]->mrq) { //if currently servicing a memory request
         // correct row activated for a READ
         if ( !issued && !CCDc && !bk[j]->RCDc &&
              (bk[j]->curr_row == bk[j]->mrq->row) && 
              (bk[j]->mrq->rw == READ) && (WTRc == 0 )  &&
              (bk[j]->state == BANK_ACTIVE) &&
              !rwq->full() ) {
            if (rw==WRITE) {
               rw=READ;
               rwq->set_min_length(CL,gpu_sim_cycle);
            }
            rwq->push(bk[j]->mrq,gpu_sim_cycle);
            bk[j]->mrq->txbytes += BL * busW * gpu_n_mem_per_ctrlr; //16 bytes
            CCDc = tCCD;
            RTWc = tRTW;
            issued = 1;
            n_rd++;
            bwutil+= BL/2;
            bwutil_partial += BL/2;
            bk[j]->n_access++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tRD  Bk:%d Row:%03x Col:%03x \n",
                   j, bk[j]->curr_row,
                   bk[j]->mrq->col+bk[j]->mrq->txbytes-BL*busW);
#endif            
            // transfer done
            if ( !(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes) ) {
               bk[j]->mrq = NULL;
            }
         } else
            // correct row activated for a WRITE
            if ( !issued && !CCDc && !bk[j]->RCDWRc &&
                 (bk[j]->curr_row == bk[j]->mrq->row)  && 
                 (bk[j]->mrq->rw == WRITE) && (RTWc == 0 )  &&
                 (bk[j]->state == BANK_ACTIVE) &&
                 !rwq->full() ) {
            if (rw==READ) {
               rw=WRITE;
               rwq->set_min_length(WL,gpu_sim_cycle);
            }
            rwq->push(bk[j]->mrq,gpu_sim_cycle);

            bk[j]->mrq->txbytes += BL * busW * gpu_n_mem_per_ctrlr; /*16 bytes*/
            CCDc = tCCD;
            issued = 1;
            n_wr++;
            bwutil+=2;
            bwutil_partial += BL/2;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tWR  Bk:%d Row:%03x Col:%03x \n",
                   j, bk[j]->curr_row, 
                   bk[j]->mrq->col+bk[j]->mrq->txbytes-BL*busW);
#endif  
            // transfer done 
            if ( !(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes) ) {
               bk[j]->mrq = NULL;
            }
         }

         else
            // bank is idle
            if ( !issued && !RRDc && 
                 (bk[j]->state == BANK_IDLE) &&
                 !bk[j]->RPc && !bk[j]->RCc ) {
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tACT BK:%d NewRow:%03x From:%03x \n",
                   j,bk[j]->mrq->row,bk[j]->curr_row);
#endif
            // activate the row with current memory request 
            bk[j]->curr_row = bk[j]->mrq->row;
            bk[j]->state = BANK_ACTIVE;
            RRDc = tRRD;
            bk[j]->RCDc = tRCD;
            bk[j]->RCDWRc = tRCDWR;
            bk[j]->RASc = tRAS;
            bk[j]->RCc = tRC;
            prio = (j + 1) % nbk;
            issued = 1;
            n_act_partial++;
            n_act++;
         }

         else
            // different row activated
            if ( (!issued) && 
                 (bk[j]->curr_row != bk[j]->mrq->row) &&
                 (bk[j]->state == BANK_ACTIVE) && 
                 (!bk[j]->RASc) ) {
            // make the bank idle again
            bk[j]->state = BANK_IDLE;
            bk[j]->RPc = tRP;
            prio = (j + 1) % nbk;
            issued = 1;
            n_pre++;
            n_pre_partial++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tPRE BK:%d Row:%03x \n", j,bk[j]->curr_row);
#endif
         }
      } else {
         if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc
             && !bk[j]->RCc && !bk[j]->RPc  && !bk[j]->RCDWRc) k--;
         bk[i]->n_idle++;
      }
   }
   if (!issued) {
      n_nop++;
      n_nop_partial++;
#ifdef DRAM_VIEWCMD
      printf("\tNOP                        ");
#endif
   }
   if (k) {
      n_activity++;
      n_activity_partial++;
   }
   n_cmd++;
   n_cmd_partial++;

   // decrements counters once for each time dram_issueCMD is called
   DEC2ZERO(RRDc);
   DEC2ZERO(CCDc);
   DEC2ZERO(RTWc);
   DEC2ZERO(WTRc);
   for (j=0;j<nbk;j++) {
      DEC2ZERO(bk[j]->RCDc);
      DEC2ZERO(bk[j]->RASc);
      DEC2ZERO(bk[j]->RCc);
      DEC2ZERO(bk[j]->RPc);
      DEC2ZERO(bk[j]->RCDWRc);
   }

#ifdef DRAM_VISUALIZE
   visualize();
#endif
}

//if mrq is being serviced by dram, gets popped after CL latency fulfilled
class mem_fetch* dram_t::pop() 
{
   class mem_fetch *data = NULL;
   dram_req_t *mrq = rwq->pop(gpu_sim_cycle);
   if (mrq) {
#ifdef DRAM_VIEWCMD 
      printf("\tDQ: BK%d Row:%03x Col:%03x",
             mrq->bk, mrq->row, mrq->col + mrq->dqbytes);
#endif
      mrq->dqbytes += BL * busW * gpu_n_mem_per_ctrlr; /*16 bytes*/
      if (mrq->dqbytes >= mrq->nbytes) {
         if (m_config->gpgpu_memlatency_stat) {
            unsigned dq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - mrq->timestamp;
            m_stats->dq_lat_table[LOGB2(dq_latency)]++;
            if (dq_latency > m_stats->max_dq_latency)
               m_stats->max_dq_latency = dq_latency;
         }
         data = mrq->data; 
         delete mrq;
      }
   }
#ifdef DRAM_VIEWCMD 
   printf("\n");
#endif
   return data;
}

void dram_t::returnq_push( class mem_fetch *mf, unsigned long long gpu_sim_cycle)
{
   returnq->push(mf,gpu_sim_cycle);
}

class mem_fetch* dram_t::returnq_pop( unsigned long long gpu_sim_cycle)
{
   return returnq->pop(gpu_sim_cycle);
}

class mem_fetch* dram_t::returnq_top()
{
   return returnq->top();
}

void dram_t::print( FILE* simFile) const
{
   unsigned i;
   fprintf(simFile,"DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", 
           id, nbk, busW, BL, CL );
   fprintf(simFile,"tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
           tCCD, tRRD, tRCD, tRAS, tRP, tRC );
   fprintf(simFile,"n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g\n",
           n_cmd, n_nop, n_act, n_pre, n_req, n_rd, n_wr,
           (float)bwutil/n_cmd);
   fprintf(simFile,"n_activity=%d dram_eff=%.4g\n",
           n_activity, (float)bwutil/n_activity);
   for (i=0;i<nbk;i++) {
      fprintf(simFile, "bk%d: %da %di ",i,bk[i]->n_access,bk[i]->n_idle);
   }
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dram_eff_bins[i]);
   fprintf(simFile, "\n");
   fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs, (float)ave_mrqs/n_cmd);
}

void dram_t::visualize() const
{
   printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", 
          RRDc, CCDc, mrqq->get_length(),rwq->get_length());
   for (unsigned i=0;i<nbk;i++) {
      printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", 
             i, bk[i]->state, bk[i]->curr_row,
             bk[i]->RCDc, bk[i]->RASc,
             bk[i]->RPc, bk[i]->RCc,
             bk[i]->mrq );
      if (bk[i]->mrq)
         printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
      printf("\n");
   }
   if ( m_fast_ideal_scheduler ) 
      m_fast_ideal_scheduler->print(stdout);
}

void dram_t::print_stat( FILE* simFile ) 
{
   fprintf(simFile,"DRAM (%d): n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g ",
           id, n_cmd, n_nop, n_act, n_pre, n_req, n_rd, n_wr,
           (float)bwutil/n_cmd);
   fprintf(simFile, "mrqq: %d %.4g mrqsmax=%d ", max_mrqs, (float)ave_mrqs/n_cmd, max_mrqs_temp);
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (unsigned i=0;i<10;i++) fprintf(simFile, " %d", dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (unsigned i=0;i<10;i++) fprintf(simFile, " %d", dram_eff_bins[i]);
   fprintf(simFile, "\n");
   max_mrqs_temp = 0;
}

void dram_t::queue_latency_log_dump( FILE *fp ) 
{
   fprintf(fp,"(LOGB2)Latency DRAM[%d] ",id);
   StatDisp(mrqq->get_lat_stat());
   fprintf(fp,"(LOGB2)Latency DRAM[%d] ",id);
   StatDisp(rwq->get_lat_stat());   
   dram_log(DUMPLOG);
}

void dram_t::visualizer_print( gzFile visualizer_file )
{
   // dram specific statistics
   gzprintf(visualizer_file,"dramncmd: %u %u\n",id, n_cmd_partial);  
   gzprintf(visualizer_file,"dramnop: %u %u\n",id,n_nop_partial);
   gzprintf(visualizer_file,"dramnact: %u %u\n",id,n_act_partial);
   gzprintf(visualizer_file,"dramnpre: %u %u\n",id,n_pre_partial);
   gzprintf(visualizer_file,"dramnreq: %u %u\n",id,n_req_partial);
   gzprintf(visualizer_file,"dramavemrqs: %u %u\n",id,
            n_cmd_partial?(ave_mrqs_partial/n_cmd_partial ):0);

   // utilization and efficiency
   gzprintf(visualizer_file,"dramutil: %u %u\n",  
            id,n_cmd_partial?100*bwutil_partial/n_cmd_partial:0);
   gzprintf(visualizer_file,"drameff: %u %u\n", 
            id,n_activity_partial?100*bwutil_partial/n_activity_partial:0);

   // reset for next interval
   bwutil_partial = 0;
   n_activity_partial = 0;
   ave_mrqs_partial = 0; 
   n_cmd_partial = 0;
   n_nop_partial = 0;
   n_act_partial = 0;
   n_pre_partial = 0;
   n_req_partial = 0;

   // dram access type classification
   for (unsigned j = 0; j < m_config->gpu_mem_n_bk; j++) {
      gzprintf(visualizer_file,"dramglobal_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[GLOBAL_ACC_R][id][j]);
      gzprintf(visualizer_file,"dramglobal_acc_w: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[GLOBAL_ACC_W][id][j]);
      gzprintf(visualizer_file,"dramlocal_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[LOCAL_ACC_R][id][j]);
      gzprintf(visualizer_file,"dramlocal_acc_w: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[LOCAL_ACC_W][id][j]);
      gzprintf(visualizer_file,"dramconst_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[CONST_ACC_R][id][j]);
      gzprintf(visualizer_file,"dramtexture_acc_r: %u %u %u\n", id, j, 
               m_stats->mem_access_type_stats[TEXTURE_ACC_R][id][j]);
   }
}
