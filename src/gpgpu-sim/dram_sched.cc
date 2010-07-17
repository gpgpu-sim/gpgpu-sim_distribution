/* 
 * dram_sched.cc
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

#include "dram_sched.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "../abstract_hardware_model.h"

extern unsigned long long  gpu_sim_cycle;
extern signed long long gpu_tot_sim_cycle;
extern unsigned max_mrq_latency;
extern unsigned mrq_lat_table[24];
extern int gpgpu_memlatency_stat;
extern int gpgpu_dram_sched_queue_size;
extern unsigned int **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]
extern unsigned int **row_access; //concurrent_row_access[dram chip id][bank id]
extern unsigned int **num_activates; //num_activates[dram chip id][bank id]
extern unsigned int **max_conc_access2samerow; //max_conc_access2samerow[dram chip id][bank id]
extern unsigned int **max_servicetime2samerow;

ideal_dram_scheduler::ideal_dram_scheduler( dram_t *dm )
{

   m_num_pending = 0;
   m_dram = dm;
   m_queue = new std::list<dram_req_t*>[dm->nbk];
   m_bins = new std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >[ dm->nbk ];
   m_last_row = new std::list<std::list<dram_req_t*>::iterator>*[ dm->nbk ];
   curr_row_service_time = new unsigned[dm->nbk];
   row_service_timestamp = new unsigned[dm->nbk];
   for ( unsigned i=0; i < dm->nbk; i++ ) {
      m_queue[i].clear();
      m_bins[i].clear();
      m_last_row[i] = NULL;
      curr_row_service_time[i] = 0;
      row_service_timestamp[i] = 0;
   }

}

void ideal_dram_scheduler::add_req( dram_req_t *req )
{
   m_num_pending++;

   m_queue[req->bk].push_front(req);
   std::list<dram_req_t*>::iterator ptr = m_queue[req->bk].begin();

   m_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front


}


inline void ideal_dram_scheduler::data_collection(unsigned int bank)
{
   if (gpu_sim_cycle > row_service_timestamp[bank]) {
      curr_row_service_time[bank] = gpu_sim_cycle - row_service_timestamp[bank];
      if (curr_row_service_time[bank] > max_servicetime2samerow[m_dram->id][bank])
         max_servicetime2samerow[m_dram->id][bank] = curr_row_service_time[bank];
   }
   curr_row_service_time[bank] = 0;
   row_service_timestamp[bank] = gpu_sim_cycle;
   if (concurrent_row_access[m_dram->id][bank] > max_conc_access2samerow[m_dram->id][bank]) {
      max_conc_access2samerow[m_dram->id][bank] = concurrent_row_access[m_dram->id][bank];
   }
   concurrent_row_access[m_dram->id][bank] = 0;
   num_activates[m_dram->id][bank]++;
}

dram_req_t *ideal_dram_scheduler::schedule( unsigned bank, unsigned curr_row )
{
   int row_hit = 0;
   if ( m_last_row[bank] == NULL ) {
      if ( m_queue[bank].empty() )
         return NULL;

      std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >::iterator bin_ptr = m_bins[bank].find( curr_row );
      if ( bin_ptr == m_bins[bank].end()) {
         dram_req_t *req = m_queue[bank].back();
         bin_ptr = m_bins[bank].find( req->row );
         assert( bin_ptr != m_bins[bank].end() ); // where did the request go???
         m_last_row[bank] = &(bin_ptr->second);
         data_collection(bank);
      } else {
         m_last_row[bank] = &(bin_ptr->second);

      }
   }
   row_hit=1;
   std::list<dram_req_t*>::iterator next = m_last_row[bank]->back();
   dram_req_t *req = (*next);

   concurrent_row_access[m_dram->id][bank]++;
   row_access[m_dram->id][bank]++;
   m_last_row[bank]->pop_back();

   m_queue[bank].erase(next);
   if ( m_last_row[bank]->empty() ) {
      m_bins[bank].erase( req->row );
      m_last_row[bank] = NULL;
   }
#ifdef DEBUG_FAST_IDEAL_SCHED
   if ( req )
      printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", 
             (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif
   assert( req != NULL && m_num_pending != 0 ); 
   m_num_pending--;

   return req;
}


void ideal_dram_scheduler::print( FILE *fp )
{
   for ( unsigned b=0; b < m_dram->nbk; b++ ) {
      printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size() );
   }
}

void* alloc_fast_ideal_scheduler(dram_t *dm)
{
   return new ideal_dram_scheduler(dm);
}

void fast_scheduler_ideal(dram_t* dm)
{


   unsigned mrq_latency;
   // replacement for scheduler_ideal()

   ideal_dram_scheduler *sched = reinterpret_cast<ideal_dram_scheduler*>( dm->m_fast_ideal_scheduler );
   while ( !dq_empty(dm->mrqq) && (!gpgpu_dram_sched_queue_size || sched->num_pending() < (unsigned) gpgpu_dram_sched_queue_size)) {
      dram_req_t *req = (dram_req_t*)dq_pop(dm->mrqq);
      sched->add_req(req);
   }

   dram_req_t *req;
   unsigned i;
   for ( i=0; i < dm->nbk; i++ ) {
      unsigned b = (i+dm->prio)%dm->nbk;
      if ( !dm->bk[b]->mrq ) {

         req = sched->schedule(b, dm->bk[b]->curr_row);

         if ( req ) {
            dm->prio = (dm->prio+1)%dm->nbk;
            dm->bk[b]->mrq = req;
            if (gpgpu_memlatency_stat) {
               mrq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - dm->bk[b]->mrq->timestamp;
               dm->bk[b]->mrq->timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
               mrq_lat_table[LOGB2(mrq_latency)]++;
               if (mrq_latency > max_mrq_latency) {
                  max_mrq_latency = mrq_latency;
               }
            }

            break;
         }
      }
   }
}



void dump_fast_ideal_scheduler( dram_t *dm )
{
   ideal_dram_scheduler *sched = reinterpret_cast<ideal_dram_scheduler*>( dm->m_fast_ideal_scheduler );
   sched->print(stdout);
}

unsigned fast_scheduler_queue_length(dram_t *dm)
{
   if (dm->m_fast_ideal_scheduler ) {
      ideal_dram_scheduler *sched = reinterpret_cast<ideal_dram_scheduler*>( dm->m_fast_ideal_scheduler );
      return sched->num_pending();
   } else {
      printf("fast_scheduler_queue_length(): Where did the scheduler go?\n");
      exit(1);
   }
}


