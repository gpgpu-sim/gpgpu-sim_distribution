// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"
#include "gpu-misc.h"
#include "dram.h"
#include "mem_latency_stat.h"
#include "dram_sched.h"
#include "mem_fetch.h"
#include "l2cache.h"

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

template class fifo_pipeline<mem_fetch>;
template class fifo_pipeline<dram_req_t>;

dram_t::dram_t( unsigned int partition_id, const struct memory_config *config, memory_stats_t *stats,
                memory_partition_unit *mp )
{
   id = partition_id;
   m_memory_partition_unit = mp;
   m_stats = stats;
   m_config = config;

   CCDc = 0;
   RRDc = 0;
   RTWc = 0;
   WTRc = 0;

   rw = READ; //read mode is default

	bkgrp = (bankgrp_t**) calloc(sizeof(bankgrp_t*), m_config->nbkgrp);
	bkgrp[0] = (bankgrp_t*) calloc(sizeof(bank_t), m_config->nbkgrp);
	for (unsigned i=1; i<m_config->nbkgrp; i++) {
		bkgrp[i] = bkgrp[0] + i;
	}
	for (unsigned i=0; i<m_config->nbkgrp; i++) {
		bkgrp[i]->CCDLc = 0;
		bkgrp[i]->RTPLc = 0;
	}

   bk = (bank_t**) calloc(sizeof(bank_t*),m_config->nbk);
   bk[0] = (bank_t*) calloc(sizeof(bank_t),m_config->nbk);
   for (unsigned i=1;i<m_config->nbk;i++) 
      bk[i] = bk[0] + i;
   for (unsigned i=0;i<m_config->nbk;i++) {
      bk[i]->state = BANK_IDLE;
      bk[i]->bkgrpindex = i/(m_config->nbk/m_config->nbkgrp);
   }
   prio = 0;  
   rwq = new fifo_pipeline<dram_req_t>("rwq",m_config->CL,m_config->CL+1);
   mrqq = new fifo_pipeline<dram_req_t>("mrqq",0,2);
   returnq = new fifo_pipeline<mem_fetch>("dramreturnq",0,m_config->gpgpu_dram_return_queue_size==0?1024:m_config->gpgpu_dram_return_queue_size); 
   m_frfcfs_scheduler = NULL;
   if ( m_config->scheduler_type == DRAM_FRFCFS )
      m_frfcfs_scheduler = new frfcfs_scheduler(m_config,this,stats);
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
   ave_mrqs = 0;

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

bool dram_t::full() const 
{
    if(m_config->scheduler_type == DRAM_FRFCFS ){
        if(m_config->gpgpu_frfcfs_dram_sched_queue_size == 0 ) return false;
        return m_frfcfs_scheduler->num_pending() >= m_config->gpgpu_frfcfs_dram_sched_queue_size;
    }
   else return mrqq->full();
}

unsigned dram_t::que_length() const
{
   unsigned nreqs = 0;
   if (m_config->scheduler_type == DRAM_FRFCFS ) {
      nreqs = m_frfcfs_scheduler->num_pending();
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
   return m_config->gpgpu_frfcfs_dram_sched_queue_size; 
}


dram_req_t::dram_req_t( class mem_fetch *mf )
{
   txbytes = 0;
   dqbytes = 0;
   data = mf;

   const addrdec_t &tlx = mf->get_tlx_addr();

   bk  = tlx.bk; 
   row = tlx.row; 
   col = tlx.col; 
   nbytes = mf->get_data_size();

   timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
   addr = mf->get_addr();
   insertion_time = (unsigned) gpu_sim_cycle;
   rw = data->get_is_write()?WRITE:READ;
}

void dram_t::push( class mem_fetch *data ) 
{
   assert(id == data->get_tlx_addr().chip); // Ensure request is in correct memory partition

   dram_req_t *mrq = new dram_req_t(data);
   data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
   mrqq->push(mrq);

   // stats...
   n_req += 1;
   n_req_partial += 1;
   if ( m_config->scheduler_type == DRAM_FRFCFS ) {
      unsigned nreqs = m_frfcfs_scheduler->num_pending();
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
      head_mrqq->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
      bkn = head_mrqq->bk;
      if (!bk[bkn]->mrq) 
         bk[bkn]->mrq = mrqq->pop();
   }
}


#define DEC2ZERO(x) x = (x)? (x-1) : 0;
#define SWAP(a,b) a ^= b; b ^= a; a ^= b;

void dram_t::cycle()
{

   if( !returnq->full() ) {
       dram_req_t *cmd = rwq->pop();
       if( cmd ) {
#ifdef DRAM_VIEWCMD 
           printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row, cmd->col + cmd->dqbytes);
#endif
           cmd->dqbytes += m_config->dram_atom_size; 
           if (cmd->dqbytes >= cmd->nbytes) {
              mem_fetch *data = cmd->data; 
              data->set_status(IN_PARTITION_MC_RETURNQ,gpu_sim_cycle+gpu_tot_sim_cycle); 
              if( data->get_access_type() != L1_WRBK_ACC && data->get_access_type() != L2_WRBK_ACC ) {
                 data->set_reply();
                 returnq->push(data);
              } else {
                 m_memory_partition_unit->set_done(data);
                 delete data;
              }
              delete cmd;
           }
#ifdef DRAM_VIEWCMD 
           printf("\n");
#endif
       }
   }

   /* check if the upcoming request is on an idle bank */
   /* Should we modify this so that multiple requests are checked? */

   switch (m_config->scheduler_type) {
   case DRAM_FIFO: scheduler_fifo(); break;
   case DRAM_FRFCFS: scheduler_frfcfs(); break;
	default:
		printf("Error: Unknown DRAM scheduler type\n");
		assert(0);
   }
   if ( m_config->scheduler_type == DRAM_FRFCFS ) {
      unsigned nreqs = m_frfcfs_scheduler->num_pending();
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

   unsigned k=m_config->nbk;
   bool issued = false;

   // check if any bank is ready to issue a new read
   for (unsigned i=0;i<m_config->nbk;i++) {
      unsigned j = (i + prio) % m_config->nbk;
	  unsigned grp = j>>m_config->bk_tag_length;
      if (bk[j]->mrq) { //if currently servicing a memory request
          bk[j]->mrq->data->set_status(IN_PARTITION_DRAM,gpu_sim_cycle+gpu_tot_sim_cycle);
         // correct row activated for a READ
         if ( !issued && !CCDc && !bk[j]->RCDc &&
              !(bkgrp[grp]->CCDLc) &&
              (bk[j]->curr_row == bk[j]->mrq->row) && 
              (bk[j]->mrq->rw == READ) && (WTRc == 0 )  &&
              (bk[j]->state == BANK_ACTIVE) &&
              !rwq->full() ) {
            if (rw==WRITE) {
               rw=READ;
               rwq->set_min_length(m_config->CL);
            }
            rwq->push(bk[j]->mrq);
            bk[j]->mrq->txbytes += m_config->dram_atom_size; 
            CCDc = m_config->tCCD;
            bkgrp[grp]->CCDLc = m_config->tCCDL;
            RTWc = m_config->tRTW;
            bk[j]->RTPc = m_config->BL/m_config->data_command_freq_ratio;
            bkgrp[grp]->RTPLc = m_config->tRTPL;
            issued = true;
            n_rd++;
            bwutil += m_config->BL/m_config->data_command_freq_ratio;
            bwutil_partial += m_config->BL/m_config->data_command_freq_ratio;
            bk[j]->n_access++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tRD  Bk:%d Row:%03x Col:%03x \n",
                   j, bk[j]->curr_row,
                   bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif            
            // transfer done
            if ( !(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes) ) {
               bk[j]->mrq = NULL;
            }
         } else
            // correct row activated for a WRITE
            if ( !issued && !CCDc && !bk[j]->RCDWRc &&
                 !(bkgrp[grp]->CCDLc) &&
                 (bk[j]->curr_row == bk[j]->mrq->row)  && 
                 (bk[j]->mrq->rw == WRITE) && (RTWc == 0 )  &&
                 (bk[j]->state == BANK_ACTIVE) &&
                 !rwq->full() ) {
            if (rw==READ) {
               rw=WRITE;
               rwq->set_min_length(m_config->WL);
            }
            rwq->push(bk[j]->mrq);

            bk[j]->mrq->txbytes += m_config->dram_atom_size; 
            CCDc = m_config->tCCD;
            bkgrp[grp]->CCDLc = m_config->tCCDL;
            WTRc = m_config->tWTR; 
            bk[j]->WTPc = m_config->tWTP; 
            issued = true;
            n_wr++;
            bwutil += m_config->BL/m_config->data_command_freq_ratio;
            bwutil_partial += m_config->BL/m_config->data_command_freq_ratio;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tWR  Bk:%d Row:%03x Col:%03x \n",
                   j, bk[j]->curr_row, 
                   bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
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
            RRDc = m_config->tRRD;
            bk[j]->RCDc = m_config->tRCD;
            bk[j]->RCDWRc = m_config->tRCDWR;
            bk[j]->RASc = m_config->tRAS;
            bk[j]->RCc = m_config->tRC;
            prio = (j + 1) % m_config->nbk;
            issued = true;
            n_act_partial++;
            n_act++;
         }

         else
            // different row activated
            if ( (!issued) && 
                 (bk[j]->curr_row != bk[j]->mrq->row) &&
                 (bk[j]->state == BANK_ACTIVE) && 
                 (!bk[j]->RASc && !bk[j]->WTPc && 
				  !bk[j]->RTPc &&
				  !bkgrp[grp]->RTPLc) ) {
            // make the bank idle again
            bk[j]->state = BANK_IDLE;
            bk[j]->RPc = m_config->tRP;
            prio = (j + 1) % m_config->nbk;
            issued = true;
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
         bk[j]->n_idle++;
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
   for (unsigned j=0;j<m_config->nbk;j++) {
      DEC2ZERO(bk[j]->RCDc);
      DEC2ZERO(bk[j]->RASc);
      DEC2ZERO(bk[j]->RCc);
      DEC2ZERO(bk[j]->RPc);
      DEC2ZERO(bk[j]->RCDWRc);
      DEC2ZERO(bk[j]->WTPc);
      DEC2ZERO(bk[j]->RTPc);
   }
   for (unsigned j=0; j<m_config->nbkgrp; j++) {
	   DEC2ZERO(bkgrp[j]->CCDLc);
	   DEC2ZERO(bkgrp[j]->RTPLc);
   }

#ifdef DRAM_VISUALIZE
   visualize();
#endif
}

//if mrq is being serviced by dram, gets popped after CL latency fulfilled
class mem_fetch* dram_t::return_queue_pop() 
{
    return returnq->pop();
}

class mem_fetch* dram_t::return_queue_top() 
{
    return returnq->top();
}

void dram_t::print( FILE* simFile) const
{
   unsigned i;
   fprintf(simFile,"DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", 
           id, m_config->nbk, m_config->busW, m_config->BL, m_config->CL );
   fprintf(simFile,"tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
           m_config->tCCD, m_config->tRRD, m_config->tRCD, m_config->tRAS, m_config->tRP, m_config->tRC );
   fprintf(simFile,"n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g\n",
           n_cmd, n_nop, n_act, n_pre, n_req, n_rd, n_wr,
           (float)bwutil/n_cmd);
   fprintf(simFile,"n_activity=%d dram_eff=%.4g\n",
           n_activity, (float)bwutil/n_activity);
   for (i=0;i<m_config->nbk;i++) {
      fprintf(simFile, "bk%d: %da %di ",i,bk[i]->n_access,bk[i]->n_idle);
   }
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dram_eff_bins[i]);
   fprintf(simFile, "\n");
   if(m_config->scheduler_type== DRAM_FRFCFS) 
       fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs, (float)ave_mrqs/n_cmd);
}

void dram_t::visualize() const
{
   printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", 
          RRDc, CCDc, mrqq->get_length(),rwq->get_length());
   for (unsigned i=0;i<m_config->nbk;i++) {
      printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", 
             i, bk[i]->state, bk[i]->curr_row,
             bk[i]->RCDc, bk[i]->RASc,
             bk[i]->RPc, bk[i]->RCc,
             bk[i]->mrq );
      if (bk[i]->mrq)
         printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
      printf("\n");
   }
   if ( m_frfcfs_scheduler ) 
      m_frfcfs_scheduler->print(stdout);
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
   for (unsigned j = 0; j < m_config->nbk; j++) {
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


void dram_t::set_dram_power_stats(	unsigned &cmd,
									unsigned &activity,
									unsigned &nop,
									unsigned &act,
									unsigned &pre,
									unsigned &rd,
									unsigned &wr,
									unsigned &req) const{

	// Point power performance counters to low-level DRAM counters
	cmd = n_cmd;
	activity = n_activity;
	nop = n_nop;
	act = n_act;
	pre = n_pre;
	rd = n_rd;
	wr = n_wr;
	req = n_req;
}
