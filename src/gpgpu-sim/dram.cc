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

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

dram_t* dram_create( unsigned int id, unsigned int nbk, 
                     unsigned int tCCD, unsigned int tRRD,
                     unsigned int tRCD, unsigned int tRAS,
                     unsigned int tRP, unsigned int tRC,
                     unsigned int CL, unsigned int WL,
                     unsigned int BL, unsigned int tWTR,
                     unsigned int busW, unsigned int queue_limit,
                     unsigned char scheduler_type ) 
{
   dram_t *dm;
   unsigned i;

   dm = (dram_t*)calloc(1,sizeof(dram_t));

   dm->id = id;

   dm->nbk = nbk;
   dm->tCCD = tCCD;
   dm->tRRD = tRRD;
   dm->tRCD = tRCD;
   dm->tRCDWR = tRCD - (WL + 1); //formula given in datasheet
   dm->tRAS = tRAS;
   dm->tRP  = tRP;
   dm->tRC  = tRC;
   dm->CL = CL;
   dm->WL = WL;
   dm->BL = BL;

   dm->tRTW = (CL+(BL/2)+2-WL); //read to write time according to datasheet
   dm->tWTR = tWTR;

   dm->busW = busW;

   dm->CCDc = 0;
   dm->RRDc = 0;
   dm->RTWc = 0;
   dm->WTRc = 0;

   dm->rw = READ; //read mode is default

   dm->bk = (bank_t**) calloc(sizeof(bank_t*),dm->nbk);
   dm->bk[0] = (bank_t*) calloc(sizeof(bank_t),dm->nbk);
   for (i=1;i<dm->nbk;i++) {
      dm->bk[i] = dm->bk[0] + i;
   }
   for (i=0;i<dm->nbk;i++) {
      dm->bk[i]->state = BANK_IDLE;
   }
   dm->prio = 0;  
   dm->rwq = dq_create("rwq",0,dm->CL,dm->CL+1);
   dm->mrqq = dq_create("mrqq",0,0,0);
   dm->queue_limit = queue_limit;

   dm->returnq = dq_create("dramreturnq",0,0, queue_limit); 

   dm->m_fast_ideal_scheduler = NULL;
   if ( scheduler_type == DRAM_IDEAL_FAST )
      dm->m_fast_ideal_scheduler = alloc_fast_ideal_scheduler(dm);


   dm->n_cmd = 0;
   dm->n_activity = 0;
   dm->n_nop = 0; 
   dm->n_act = 0; 
   dm->n_pre = 0; 
   dm->n_rd = 0;
   dm->n_wr = 0;
   dm->n_req = 0;
   dm->max_mrqs_temp = 0;

   dm->bwutil = 0;

   dm->max_mrqs = 0;

   dm->scheduler_type = scheduler_type;

   dm->realistic_scheduler_mode = READ; //realistic scheduler defaults to read
   for (i=0;i<10;i++) {
      dm->dram_util_bins[i]=0;
      dm->dram_eff_bins[i]=0;
   }
   dm->last_n_cmd = dm->last_n_activity = dm->last_bwutil = 0;

   dm->n_cmd_partial = 0;
   dm->n_activity_partial = 0;
   dm->n_nop_partial = 0;  
   dm->n_act_partial = 0;  
   dm->n_pre_partial = 0;  
   dm->n_req_partial = 0;
   dm->ave_mrqs_partial = 0;
   dm->bwutil_partial = 0;
   return dm;
}

void dram_free( dram_t *dm ) 
{
   dq_free(dm->mrqq);
   dq_free(dm->rwq);
   dq_free( dm->returnq );

   free(dm->bk[0]);
   free(dm->bk);
   free(dm);
}

int dram_full( dram_t *dm ) 
{
   int full = 0;
   if ( dm->queue_limit == 0 ) return 0;
   if ( dm->scheduler_type == DRAM_IDEAL_FAST ) {
      unsigned nreqs = fast_scheduler_queue_length(dm) + dq_n_element(dm->mrqq);
      full = (nreqs >= dm->queue_limit);
   } else {
      full = (dm->mrqq->length >= dm->queue_limit);
   }

   return full;
}

unsigned int dram_que_length( dram_t *dm ) 
{
   unsigned nreqs = 0;
   if (dm->scheduler_type == DRAM_IDEAL_FAST ) {
      nreqs = fast_scheduler_queue_length(dm);
   } else {
      nreqs = dm->mrqq->length ;
   }
   return nreqs;
}

void dram_push( dram_t *dm, unsigned int bank,
                unsigned int row, unsigned int col,
                unsigned int nbytes, unsigned int write,
                unsigned int wid, 
                unsigned int sid, int cache_hits_waiting, unsigned long long addr,
                void *data ) 
{
   dram_req_t *mrq;

   if (bank>=dm->nbk) printf("ERROR: no such bank exist in DRAM %d\n", bank);

   mrq = (dram_req_t *) malloc(sizeof(dram_req_t));

   mrq->bk = bank;
   mrq->row = row;
   mrq->col = col;
   mrq->nbytes = nbytes;
   mrq->txbytes = 0;
   mrq->dqbytes = 0;
   mrq->data = data;
   mrq->timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
   mrq->cache_hits_waiting = cache_hits_waiting;
   mrq->addr = addr;
   mrq->insertion_time = (unsigned) gpu_sim_cycle;

   if (!write) {
      mrq->rw = READ;   //request is a read
   } else {
      mrq->rw = WRITE;  //request is a write
   }

   dq_push(dm->mrqq,mrq);
   dm->n_req += 1;
   dm->n_req_partial += 1;

   if ( dm->scheduler_type == DRAM_IDEAL_FAST ) {
      unsigned nreqs = fast_scheduler_queue_length(dm);
      if ( nreqs > dm->max_mrqs_temp)
         dm->max_mrqs_temp = nreqs;
   } else {
      dm->max_mrqs_temp = (dm->max_mrqs_temp > dm->mrqq->length)? dm->max_mrqs_temp : dm->mrqq->length;
   }
}

void scheduler_fifo(dram_t* dm) 
{
   if (dm->mrqq->head) {
      dram_req_t *head_mrqq;
      unsigned int bkn;
      head_mrqq = (dram_req_t *)dm->mrqq->head->data;
      bkn = head_mrqq->bk;
      if (!dm->bk[bkn]->mrq) {
         dm->bk[bkn]->mrq = (dram_req_t*) dq_pop(dm->mrqq);
      }
   }
}


#define DEC2ZERO(x) x = (x)? (x-1) : 0;
#define SWAP(a,b) a ^= b; b ^= a; a ^= b;

void dram_issueCMD (dram_t* dm) 
{
   unsigned i,j,k;
   unsigned char issued;
   issued = 0;

   /* check if the upcoming request is on an idle bank */
   /* Should we modify this so that multiple requests are checked? */

   switch (dm->scheduler_type) {
   case DRAM_FIFO:
      scheduler_fifo(dm);
      break;
   case DRAM_IDEAL_FAST:
      fast_scheduler_ideal(dm);
      break;
	default:
		printf("Error: Unknown DRAM scheduler type\n");
		assert(0);
   }
   if ( dm->scheduler_type == DRAM_IDEAL_FAST ) {
      unsigned nreqs = fast_scheduler_queue_length(dm);
      if ( nreqs > dm->max_mrqs) {
         dm->max_mrqs = nreqs;
      }
      dm->ave_mrqs += nreqs;
      dm->ave_mrqs_partial += nreqs;
   } else {
      if (dm->mrqq->length > dm->max_mrqs) {
         dm->max_mrqs = dm->mrqq->length;
      }
      dm->ave_mrqs += dm->mrqq->length;
      dm->ave_mrqs_partial +=  dm->mrqq->length;
   }
   k=dm->nbk;
   // check if any bank is ready to issue a new read
   for (i=0;i<dm->nbk;i++) {
      j = (i + dm->prio) % dm->nbk;
      if (dm->bk[j]->mrq) { //if currently servicing a memory request
         // correct row activated for a READ
         if ( !issued && !dm->CCDc && !dm->bk[j]->RCDc &&
              (dm->bk[j]->curr_row == dm->bk[j]->mrq->row) && 
              (dm->bk[j]->mrq->rw == READ) && (dm->WTRc == 0 )  &&
              (dm->bk[j]->state == BANK_ACTIVE) &&
              !dq_full(dm->rwq) ) {
            if (dm->rw==WRITE) {
               dm->rw=READ;
               dq_set_min_length(dm->rwq, dm->CL);
            }
            dq_push(dm->rwq,(void*)dm->bk[j]->mrq); //only push when rwq empty?
            dm->bk[j]->mrq->txbytes += dm->BL * dm->busW * gpu_n_mem_per_ctrlr; //16 bytes
            dm->CCDc = dm->tCCD;
            dm->RTWc = dm->tRTW;
            issued = 1;
            dm->n_rd++;
            //printf("\tn_rd++ Bank: %d Row: %d Col: %d\n", j, dm->bk[j]->mrq->row, dm->bk[j]->mrq->col);
            dm->bwutil+= dm->BL/2;
            dm->bwutil_partial += dm->BL/2;
            dm->bk[j]->n_access++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tRD  Bk:%d Row:%03x Col:%03x \n",
                   j, dm->bk[j]->curr_row,
                   dm->bk[j]->mrq->col+dm->bk[j]->mrq->txbytes-dm->BL*dm->busW);
#endif            
            // transfer done
            if ( !(dm->bk[j]->mrq->txbytes < dm->bk[j]->mrq->nbytes) ) {
               dm->bk[j]->mrq = NULL;
            }
         } else
            // correct row activated for a WRITE
            if ( !issued && !dm->CCDc && !dm->bk[j]->RCDWRc &&
                 (dm->bk[j]->curr_row == dm->bk[j]->mrq->row)  && 
                 (dm->bk[j]->mrq->rw == WRITE) && (dm->RTWc == 0 )  &&
                 (dm->bk[j]->state == BANK_ACTIVE) &&
                 !dq_full(dm->rwq) ) {
            if (dm->rw==READ) {
               dm->rw=WRITE;
               dq_set_min_length(dm->rwq, dm->WL);
            }
            dq_push(dm->rwq,(void*)dm->bk[j]->mrq);

            dm->bk[j]->mrq->txbytes += dm->BL * dm->busW * gpu_n_mem_per_ctrlr; /*16 bytes*/
            dm->CCDc = dm->tCCD;
            issued = 1;
            dm->n_wr++;
            dm->bwutil+=2;
            dm->bwutil_partial += dm->BL/2;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tWR  Bk:%d Row:%03x Col:%03x \n",
                   j, dm->bk[j]->curr_row, 
                   dm->bk[j]->mrq->col+dm->bk[j]->mrq->txbytes-dm->BL*dm->busW);
#endif  
            // transfer done 
            if ( !(dm->bk[j]->mrq->txbytes < dm->bk[j]->mrq->nbytes) ) {
               dm->bk[j]->mrq = NULL;
            }
         }

         else
            // bank is idle
            if ( !issued && !dm->RRDc && 
                 (dm->bk[j]->state == BANK_IDLE) &&
                 !dm->bk[j]->RPc && !dm->bk[j]->RCc ) {
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tACT BK:%d NewRow:%03x From:%03x \n",
                   j,dm->bk[j]->mrq->row,dm->bk[j]->curr_row);
#endif
            // activate the row with current memory request 
            dm->bk[j]->curr_row = dm->bk[j]->mrq->row;
            dm->bk[j]->state = BANK_ACTIVE;
            dm->RRDc = dm->tRRD;
            dm->bk[j]->RCDc = dm->tRCD;
            dm->bk[j]->RCDWRc = dm->tRCDWR;
            dm->bk[j]->RASc = dm->tRAS;
            dm->bk[j]->RCc = dm->tRC;
            dm->prio = (j + 1) % dm->nbk;
            issued = 1;
            dm->n_act_partial++;
            dm->n_act++;
         }

         else
            // different row activated
            if ( (!issued) && 
                 (dm->bk[j]->curr_row != dm->bk[j]->mrq->row) &&
                 (dm->bk[j]->state == BANK_ACTIVE) && 
                 (!dm->bk[j]->RASc) ) {
            //printf("\tRASc: %d \n", dm->bk[j]->RASc);
            // make the bank idle again
            dm->bk[j]->state = BANK_IDLE;
            dm->bk[j]->RPc = dm->tRP;
            dm->prio = (j + 1) % dm->nbk;
            issued = 1;
            dm->n_pre++;
            dm->n_pre_partial++;
#ifdef DRAM_VERIFY
            PRINT_CYCLE=1;
            printf("\tPRE BK:%d Row:%03x \n", j,dm->bk[j]->curr_row);
            //printf("\tRASc: %d \n", dm->bk[j]->RASc);
#endif
         }
      } else {
         if (!dm->CCDc && !dm->RRDc && !dm->RTWc && !dm->WTRc && !dm->bk[j]->RCDc && !dm->bk[j]->RASc
             && !dm->bk[j]->RCc && !dm->bk[j]->RPc  && !dm->bk[j]->RCDWRc) k--;
         dm->bk[i]->n_idle++;
      }
   }
   if (!issued) {
      dm->n_nop++;
      dm->n_nop_partial++;
#ifdef DRAM_VIEWCMD
      printf("\tNOP                        ");
#endif
   }
   if (k) {
      dm->n_activity++;
      dm->n_activity_partial++;
   }
   dm->n_cmd++;
   dm->n_cmd_partial++;

   // decrements counters once for each time dram_issueCMD is called
   DEC2ZERO(dm->RRDc);
   DEC2ZERO(dm->CCDc);
   DEC2ZERO(dm->RTWc);
   DEC2ZERO(dm->WTRc);
   for (j=0;j<dm->nbk;j++) {
      DEC2ZERO(dm->bk[j]->RCDc);
      DEC2ZERO(dm->bk[j]->RASc);
      DEC2ZERO(dm->bk[j]->RCc);
      DEC2ZERO(dm->bk[j]->RPc);
      DEC2ZERO(dm->bk[j]->RCDWRc);
   }

#ifdef DRAM_VISUALIZE
   dram_visualize(dm);
#endif
}

//if mrq is being serviced by dram, gets popped after CL latency fulfilled
void* dram_pop( dram_t *dm ) 
{ 
   dram_req_t *mrq;
   void *data;
   unsigned dq_latency;

   data = NULL;
   mrq = (dram_req_t*)dq_pop(dm->rwq);
   if (mrq) {
      // data = mrq->data; 
#ifdef DRAM_VIEWCMD 
      printf("\tDQ: BK%d Row:%03x Col:%03x",
             mrq->bk, mrq->row, mrq->col + mrq->dqbytes);
#endif
      mrq->dqbytes += dm->BL * dm->busW * gpu_n_mem_per_ctrlr; /*16 bytes*/
      if (mrq->dqbytes >= mrq->nbytes) {

         if (gpgpu_memlatency_stat) {
            dq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - mrq->timestamp;
            dq_lat_table[LOGB2(dq_latency)]++;
            if (dq_latency > max_dq_latency)
               max_dq_latency = dq_latency;
         }
         data = mrq->data; 

         free(mrq);
      }
   }
#ifdef DRAM_VIEWCMD 
   printf("\n");
#endif

   return data;
}

// a hack to allow peeking into what memory request will be serviced.
void* dram_top( dram_t *dm )
{
   dram_req_t *mrq;
   void *data;

   data = NULL;
   mrq = (dram_req_t*)dq_top(dm->rwq);
   if (mrq) {
      // number of bytes returned from dram if this is ever popped
      unsigned tobe_dqbytes = mrq->dqbytes + dm->BL * dm->busW * gpu_n_mem_per_ctrlr; 
      if (tobe_dqbytes >= mrq->nbytes) {
         data = mrq->data; 
      }
   }

   return data;
}

void dram_print( dram_t* dm, FILE* simFile) 
{
   unsigned i;
   fprintf(simFile,"DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", 
           dm->id, dm->nbk, dm->busW, dm->BL, dm->CL );
   fprintf(simFile,"tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
           dm->tCCD, dm->tRRD, dm->tRCD, dm->tRAS, dm->tRP, dm->tRC );
   fprintf(simFile,"n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g\n",
           dm->n_cmd, dm->n_nop, dm->n_act, dm->n_pre, dm->n_req, dm->n_rd, dm->n_wr,
           (float)dm->bwutil/dm->n_cmd);
   fprintf(simFile,"n_activity=%d dram_eff=%.4g\n",
           dm->n_activity, (float)dm->bwutil/dm->n_activity);
   for (i=0;i<dm->nbk;i++) {
      fprintf(simFile, "bk%d: %da %di ",i,dm->bk[i]->n_access,dm->bk[i]->n_idle);
   }
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dm->dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dm->dram_eff_bins[i]);
   fprintf(simFile, "\n");
   /*
   {
   delay_data* mrq;
   mrq = dm->mrqq->head;
   while (mrq) {
      printf("%d",((dram_req_t*)mrq->data)->bk);
      mrq = mrq->next;
   }
   printf("\n");
   }
   */
   fprintf(simFile, "mrqq: max=%d avg=%g\n", dm->max_mrqs, (float)dm->ave_mrqs/dm->n_cmd);
}

void dram_visualize( dram_t* dm ) 
{
   unsigned i;

   printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", 
          dm->RRDc, dm->CCDc, dm->mrqq->length,dm->rwq->length);
   for (i=0;i<dm->nbk;i++) {
      printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", 
             i, dm->bk[i]->state, dm->bk[i]->curr_row,
             dm->bk[i]->RCDc, dm->bk[i]->RASc,
             dm->bk[i]->RPc, dm->bk[i]->RCc,
             dm->bk[i]->mrq );
      if (dm->bk[i]->mrq)
         printf("txf: %d %d", dm->bk[i]->mrq->nbytes, dm->bk[i]->mrq->txbytes);
      printf("\n");
   }
   if ( dm->m_fast_ideal_scheduler ) {
      dump_fast_ideal_scheduler( dm );
   }

}

void dram_print_stat( dram_t* dm, FILE* simFile ) 
{
   int i;
   fprintf(simFile,"DRAM (%d): n_cmd=%d n_nop=%d n_act=%d n_pre=%d n_req=%d n_rd=%d n_write=%d bw_util=%.4g ",
           dm->id, dm->n_cmd, dm->n_nop, dm->n_act, dm->n_pre, dm->n_req, dm->n_rd, dm->n_wr,
           (float)dm->bwutil/dm->n_cmd);
   fprintf(simFile, "mrqq: %d %.4g mrqsmax=%d ", dm->max_mrqs, (float)dm->ave_mrqs/dm->n_cmd, dm->max_mrqs_temp);
   fprintf(simFile, "\n");
   fprintf(simFile, "dram_util_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dm->dram_util_bins[i]);
   fprintf(simFile, "\ndram_eff_bins:");
   for (i=0;i<10;i++) fprintf(simFile, " %d", dm->dram_eff_bins[i]);
   fprintf(simFile, "\n");
   dm->max_mrqs_temp = 0;
}


unsigned dram_busy( dram_t* dm) 
{
   unsigned busy = 0;

   switch (dm->scheduler_type) {
   case DRAM_FIFO:
      busy = (dm->mrqq->length > 0);
      break;
   case DRAM_IDEAL_FAST:
      busy = (fast_scheduler_queue_length(dm) > 0) || (dm->mrqq->length > 0);
      break;
   }

   return busy;
}

