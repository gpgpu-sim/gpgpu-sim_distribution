/* 
 * l2cache.cc
 *
 * Copyright (c) 2009 by Tor M. Aamodt and 
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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <set>

#include "../option_parser.h"
#include "mem_fetch.h"
#include "dram.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../intersim/statwraper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"
#include "shader.h"
#include "mem_latency_stat.h"

template class fifo_pipeline<mem_fetch>;

// external dependencies
extern unsigned long long int addrdec_mask[5];
extern unsigned made_write_mfs;
extern unsigned freed_L1write_mfs;
extern unsigned freed_L2write_mfs;

address_type L2c_mshr::cache_tag(const mem_fetch *mf) const 
{
   return (mf->addr & ~(m_linesize - 1));
}

address_type L2c_miss_tracker::cache_tag(const mem_fetch *mf) const 
{
   return (mf->addr & ~(m_linesize - 1));
}

address_type L2c_access_locality::cache_tag(const mem_fetch *mf) const 
{
   return (mf->addr & ~(m_linesize - 1));
}

void gpgpu_sim::L2c_options(option_parser_t opp)
{
   option_parser_register(opp, "-gpgpu_L2_queue", OPT_CSTR, &m_memory_config->gpgpu_L2_queue_config, 
                  "L2 data cache queue length and latency config",
                  "0:0:0:0:0:0:10:10");

   option_parser_register(opp, "-gpgpu_l2_readoverwrite", OPT_BOOL, &m_memory_config->gpgpu_l2_readoverwrite, 
                "Prioritize read requests over write requests for L2",
                "0");

   option_parser_register(opp, "-l2_ideal", OPT_BOOL, &m_memory_config->l2_ideal, 
                "Use a ideal L2 cache that always hit",
                "0");
}


////////////////////////////////////////////////
// L2 MSHR model

bool L2c_mshr::new_miss(const mem_fetch *mf)
{
   address_type cacheTag = cache_tag(mf);
   mem_fetch_list &missGroup = m_L2missgroup[cacheTag];

   bool mshr_hit = not missGroup.empty();

   missGroup.push_front(mf);
   
   m_n_miss += 1;
   if (mshr_hit) 
      m_n_mshr_hits += 1;
   m_entries_used += 1;
   m_max_entries_used = std::max(m_max_entries_used, m_entries_used);

   return mshr_hit;
}

void L2c_mshr::miss_serviced(const mem_fetch *mf) 
{
   assert(m_active_mshr_chain.list == NULL);
   address_type cacheTag = cache_tag(mf);
   L2missGroup::iterator missGroup = m_L2missgroup.find(cacheTag);
   if (missGroup == m_L2missgroup.end() || mf->type == L2_WTBK_DATA) {
      assert(mf->type == L2_WTBK_DATA); // only this returning mem req can be missed by the MSHR
      return; 
   } 
   assert(missGroup->first == cacheTag);

   m_active_mshr_chain.cacheTag = cacheTag;
   m_active_mshr_chain.list = &(missGroup->second);

   m_n_miss_serviced_by_dram += 1;
}

bool L2c_mshr::mshr_chain_empty()
{
   return (m_active_mshr_chain.list == NULL);
}

mem_fetch *L2c_mshr::mshr_chain_top()
{
   const mem_fetch *mf = m_active_mshr_chain.list->back();
   assert(cache_tag(mf) == m_active_mshr_chain.cacheTag);

   return const_cast<mem_fetch*>(mf);
}

void L2c_mshr::mshr_chain_pop()
{
   m_entries_used -= 1;
   m_active_mshr_chain.list->pop_back();
   if (m_active_mshr_chain.list->empty()) {
      address_type cacheTag = m_active_mshr_chain.cacheTag;
      m_L2missgroup.erase(cacheTag);
      m_active_mshr_chain.list = NULL;
   }
}

void L2c_mshr::print(FILE *fout)
{
   fprintf(fout, "L2c MSHR: n_entries_used = %zu\n", m_entries_used);
   L2missGroup::iterator missGroup;
   for (missGroup = m_L2missgroup.begin(); missGroup != m_L2missgroup.end(); ++missGroup) {
      fprintf(fout, "%#08x: ", missGroup->first); 
      mem_fetch_list &mf_list = missGroup->second; 
      for (mem_fetch_list::iterator imf = mf_list.begin(); imf != mf_list.end(); ++imf) {
         fprintf(fout, "%p:%d ", *imf, (*imf)->request_uid);
      }
      fprintf(fout, "\n");
   }
}

void L2c_mshr::print_stat(FILE *fout) const
{
   fprintf(fout, "L2c MSHR: max_entry = %zu, n_miss = %d, n_mshr_hits = %d, n_serviced_by_dram %d\n", 
           m_max_entries_used, m_n_miss, m_n_mshr_hits, m_n_miss_serviced_by_dram);
}

////////////////////////////////////////////////
// track redundant dram access generated by L2 cache

void L2c_miss_tracker::new_miss(mem_fetch *mf)
{
   address_type cacheTag = cache_tag(mf);
   mem_fetch_set &missGroup = m_L2missgroup[cacheTag];

   if (missGroup.size() != 0) {
      m_L2redundantCnt[cacheTag] += 1;
      m_totalL2redundantAcc += 1;
   }

   missGroup.insert(mf);
}

void L2c_miss_tracker::miss_serviced(mem_fetch *mf)
{
   address_type cacheTag = cache_tag(mf);
   L2missGroup::iterator iMissGroup = m_L2missgroup.find(cacheTag);
   if (iMissGroup == m_L2missgroup.end()) return; // this is possible for write miss 
   mem_fetch_set &missGroup = iMissGroup->second;

   missGroup.erase(mf);

   // remove the miss group if it goes empty
   if (missGroup.empty()) {
      m_L2missgroup.erase(iMissGroup);
   }
}

void L2c_miss_tracker::print(FILE *fout, bool brief)
{
   L2missGroup::iterator iMissGroup;
   for (iMissGroup = m_L2missgroup.begin(); iMissGroup != m_L2missgroup.end(); ++iMissGroup) {
      fprintf(fout, "%#08x: ", iMissGroup->first); 
      for (mem_fetch_set::iterator iMemSet = iMissGroup->second.begin(); iMemSet != iMissGroup->second.end(); ++iMemSet) { 
         fprintf(fout, "%p ", *iMemSet);
      }
      fprintf(fout, "\n");
   }
}

void L2c_miss_tracker::print_stat(FILE *fout, bool brief) const
{
   fprintf(fout, "RedundantMiss = %d\n", m_totalL2redundantAcc);
   if (brief == true) return;
   fprintf(fout, "  Detail:");
   for (L2redundantCnt::const_iterator iL2rc = m_L2redundantCnt.begin(); iL2rc != m_L2redundantCnt.end(); ++iL2rc) {
      fprintf(fout, "%#08x:%d ", iL2rc->first, iL2rc->second);
   }
   fprintf(fout, "\n");
}

////////////////////////////////////////////////
// track all locality of L2 cache access
void L2c_access_locality::access(mem_fetch *mf)
{
   address_type cacheTag = cache_tag(mf);
   m_L2accCnt[cacheTag] += 1;
   m_totalL2accAcc += 1;
}

void L2c_access_locality::print_stat(FILE *fout, bool brief) const
{
   float access_locality = (float) m_totalL2accAcc / m_L2accCnt.size();
   fprintf(fout, "Access Locality = %d / %zu (%f) \n", m_totalL2accAcc, m_L2accCnt.size(), access_locality);
   if (brief == true) return;
   fprintf(fout, "  Detail:");
   pow2_histogram locality_histo(" Hits");
   for (L2accCnt::const_iterator iL2rc = m_L2accCnt.begin(); iL2rc != m_L2accCnt.end(); ++iL2rc) {
      locality_histo.add2bin(iL2rc->second);
   }
   locality_histo.fprint(fout);
   fprintf(fout, "\n");
}

memory_partition_unit::~memory_partition_unit()
{
   delete m_mshr;
   delete m_missTracker;
   delete m_accessLocality; 
}

void memory_partition_unit::set_stats( class memory_stats_t *stats )
{
    m_stats=stats;
    m_dram->set_stats(stats);
}

void memory_partition_unit::cache_cycle()
{
   process_dram_output(); // pop from dram
   L2c_push_miss_to_dram();   // push to dram
   L2c_service_mem_req();     // pop(push) from(to)  icnt2l2(l2toicnt) queues; service l2 requests 
   if (m_config->gpgpu_cache_dl2_opt) { // L2 cache enabled
      L2c_update_stat(); 
      L2c_log(SAMPLELOG); 
   }
}

unsigned memory_partition_unit::L2c_get_linesize() 
{ 
   return m_L2cache->get_line_sz(); 
}

bool memory_partition_unit::full() const
{
   if (m_config->gpgpu_cache_dl2_opt) {
      return m_icnt2cache_queue->full() || m_icnt2cache_write_queue->full();
   } else {
      return( m_config->gpgpu_dram_sched_queue_size && m_dram->full() );
   }
}

//////////////////////////////////////////////// 
// L2 access functions

// L2 Cache Creation 
memory_partition_unit::memory_partition_unit( unsigned partition_id, struct memory_config *config )
{
   m_id = partition_id;
   m_config=config;
   m_stats=NULL;
   m_dram = new dram_t(m_id, m_config);

   if( m_config->gpgpu_cache_dl2_opt ) {
      char L2c_name[32];
      snprintf(L2c_name, 32, "L2_%03d", m_id);
      m_L2cache = new cache_t(L2c_name,m_config->gpgpu_cache_dl2_opt, ~addrdec_mask[CHIP], write_through, -1, -1 );
      m_mshr = new L2c_mshr(m_L2cache->get_line_sz());
      m_missTracker = new L2c_miss_tracker(m_L2cache->get_line_sz()); 
      m_accessLocality = new L2c_access_locality(m_L2cache->get_line_sz());
   } else {
      m_L2cache=NULL;
      m_mshr=NULL;
      m_missTracker=NULL;
      m_accessLocality=NULL;
   }

   unsigned int L2c_cb_L2_length;
   unsigned int L2c_cb_L2w_length;
   unsigned int L2c_L2_dm_length;
   unsigned int L2c_dm_L2_length;
   unsigned int L2c_dm_L2w_length;
   unsigned int L2c_L2_cb_length;
   unsigned int L2c_L2_cb_minlength;
   unsigned int L2c_L2_dm_minlength;

   sscanf(m_config->gpgpu_L2_queue_config,"%d:%d:%d:%d:%d:%d:%d:%d", 
          &L2c_cb_L2_length, &L2c_cb_L2w_length, &L2c_L2_dm_length, 
          &L2c_dm_L2_length, &L2c_dm_L2w_length, &L2c_L2_cb_length,
          &L2c_L2_cb_minlength, &L2c_L2_dm_minlength );
   //(<name>,<latency>,<min_length>,<max_length>)
   m_icnt2cache_queue        = new fifo_pipeline<mem_fetch>("cbtoL2queue",       0,L2c_cb_L2_length, gpu_sim_cycle); 
   m_icnt2cache_write_queue   = new fifo_pipeline<mem_fetch>("cbtoL2writequeue",  0,L2c_cb_L2w_length, gpu_sim_cycle); 
   L2todramqueue      = new fifo_pipeline<mem_fetch>("L2todramqueue",     L2c_L2_dm_minlength, L2c_L2_dm_length, gpu_sim_cycle);
   dramtoL2queue      = new fifo_pipeline<mem_fetch>("dramtoL2queue",     0,L2c_dm_L2_length, gpu_sim_cycle);
   dramtoL2writequeue = new fifo_pipeline<mem_fetch>("dramtoL2writequeue",0,L2c_dm_L2w_length, gpu_sim_cycle);
   L2tocbqueue        = new fifo_pipeline<mem_fetch>("L2tocbqueue",       L2c_L2_cb_minlength, L2c_L2_cb_length, gpu_sim_cycle);
   L2todram_wbqueue   = new fifo_pipeline<mem_fetch>("L2todram_wbqueue",  L2c_L2_dm_minlength, L2c_L2_dm_minlength + m_config->gpgpu_dram_sched_queue_size + L2c_dm_L2_length, gpu_sim_cycle);
   L2dramout = NULL;
   wb_addr=-1;
   if (m_config->gpgpu_cache_dl2_opt && 1) {
      cbtol2_Dist      = StatCreate("cbtoL2",1, m_icnt2cache_queue->get_max_len());
      cbtoL2wr_Dist    = StatCreate("cbtoL2write",1, m_icnt2cache_write_queue->get_max_len());
      L2tocb_Dist      = StatCreate("L2tocb",1, L2tocbqueue->get_max_len());
      dramtoL2_Dist    = StatCreate("dramtoL2",1, dramtoL2queue->get_max_len());
      dramtoL2wr_Dist  = StatCreate("dramtoL2write",1, dramtoL2writequeue->get_max_len());
      L2todram_Dist    = StatCreate("L2todram",1, L2todramqueue->get_max_len());
      L2todram_wb_Dist = StatCreate("L2todram_wb",1, L2todram_wbqueue->get_max_len());
   } else {
      cbtol2_Dist      = NULL;
      cbtoL2wr_Dist    = NULL;
      L2tocb_Dist      = NULL;
      dramtoL2_Dist    = NULL;
      dramtoL2wr_Dist  = NULL;
      L2todram_Dist    = NULL;
      L2todram_wb_Dist = NULL;
   }
}

void memory_partition_unit::L2c_service_mem_req()
{
   // service memory request in icnt-to-L2 queue, writing to L2 as necessary
   if( L2tocbqueue->full() || L2todramqueue->full() ) 
      return;
   mem_fetch* mf = m_icnt2cache_queue->pop(gpu_sim_cycle);
   if( !mf ) 
      mf = m_icnt2cache_write_queue->pop(gpu_sim_cycle);
   if( !mf ) 
      return;
   switch (mf->type) {
   case RD_REQ:
   case WT_REQ: {
         address_type rep_block;
         enum cache_request_status status = m_L2cache->access( mf->addr, 4, mf->m_write, gpu_sim_cycle, &rep_block);
         if( (status==HIT) || m_config->l2_ideal ) {
            mf->type = REPLY_DATA;
            L2tocbqueue->push(mf,gpu_sim_cycle);
            if (!mf->m_write) { 
               m_stats->L2_read_hit++;
               m_stats->memlatstat_icnt2sh_push(mf);
               if (mf->mshr) 
                  mf->mshr->set_status(IN_L2TOCBQUEUE_HIT);
            } else { 
               m_stats->L2_write_hit++;
               freed_L1write_mfs++;
               gpgpu_n_processed_writes++;
            }
         } else {
            // L2 Cache Miss
            // if a miss hits in the mshr, that means there is another inflight request for the same data
            // this miss just need to access the cache later when this request is serviced
            bool mshr_hit = m_mshr->new_miss(mf);
            if (not mshr_hit) {
               if (mf->m_write) {
                  // if request is writeback from L1 and misses, 
                  // then redirect mf writes to dram (no write allocate)
                  mf->nbytes_L2 = mf->nbytes_L1 - READ_PACKET_SIZE;
               }
               L2todramqueue->push(mf,gpu_sim_cycle);
            }
            if (mf->mshr) 
               mf->mshr->set_status(IN_L2TODRAMQUEUE);
         }
      }
      break;
   default: assert(0);
   }
}

// service memory request in L2todramqueue, pushing to dram 
void memory_partition_unit::L2c_push_miss_to_dram()
{
   if ( m_config->gpgpu_dram_sched_queue_size && m_dram->full() ) 
      return;
   mem_fetch* mf = L2todram_wbqueue->pop(gpu_sim_cycle); //prioritize writeback
   if (!mf) mf = L2todramqueue->pop(gpu_sim_cycle);
   if (mf) {
      if (mf->m_write) {
         m_stats->L2_write_miss++;
      } else {
         m_stats->L2_read_miss++;
      }
      m_missTracker->new_miss(mf);
      m_dram->push(mf);
      if (mf->mshr) mf->mshr->set_status(IN_DRAM_REQ_QUEUE);
   }
}

// service memory request in dramtoL2queue, writing to L2 as necessary
// (may cause cache eviction and subsequent writeback) 
void memory_partition_unit::process_dram_output() 
{
   if (L2dramout == NULL) {
      // pop from mshr chain if it is not empty, otherwise, pop a new cacheline from dram output queue
      if (m_mshr->mshr_chain_empty() == false) {
         L2dramout = m_mshr->mshr_chain_top();
         m_mshr->mshr_chain_pop();
      } else {
         L2dramout = dramtoL2queue->pop(gpu_sim_cycle);
         if (!L2dramout) 
            L2dramout = dramtoL2writequeue->pop(gpu_sim_cycle);
         if (L2dramout != NULL) {
            m_mshr->miss_serviced(L2dramout);
            if (m_mshr->mshr_chain_empty() == false) { // possible if this is a L2 writeback
               L2dramout = m_mshr->mshr_chain_top();
               m_mshr->mshr_chain_pop();
            }
         }
      }
   }
   mem_fetch* mf = L2dramout;
   if (mf) {
      if (!mf->m_write) { //service L2 read miss
         // it is a pre-fill dramout mf
         if (wb_addr == (unsigned long long int)-1) {
            if ( L2tocbqueue->full() ) {
               assert (L2dramout || wb_addr == (unsigned long long int)-1);
               return;
            }
            if (mf->mshr) 
               mf->mshr->set_status(IN_L2TOCBQUEUE_MISS);
            //only transfer across icnt once the whole line has been received by L2 cache
            mf->type = REPLY_DATA;
            L2tocbqueue->push(mf,gpu_sim_cycle);
            wb_addr = m_L2cache->shd_cache_fill(mf->addr, gpu_sim_cycle);
         }
         // only perform a write on cache eviction (write-back policy)
         // it is the 1st or nth time trial to writeback
         if (wb_addr != (unsigned long long int)-1) {
            // performing L2 writeback (no false sharing for memory-side cache)
            int wb_succeed = L2c_write_back(wb_addr, m_L2cache->get_line_sz()); 
            if (!wb_succeed) {
               assert (L2dramout || wb_addr == (unsigned long long int)-1);
               return;
            }
         }
         m_missTracker->miss_serviced(mf);
         L2dramout = NULL;
         wb_addr = -1;
      } else { //service L2 write miss
         m_missTracker->miss_serviced(mf);
         freed_L2write_mfs++;
         m_request_tracker.erase(mf);
         delete mf;
         gpgpu_n_processed_writes++;
         L2dramout = NULL;
         wb_addr = -1;
      }
   }
   assert (L2dramout || wb_addr == (unsigned long long int)-1);
}

// Writeback from L2 to DRAM: 
// - Takes in memory address and their parameters and pushes to dram request queue
// - This is used only for L2 writeback 
bool memory_partition_unit::L2c_write_back( unsigned long long int addr, int bsize ) 
{
   if ( L2todram_wbqueue->full() ) 
      return false;
   mem_fetch *mf = new mem_fetch(addr,
                                 bsize+READ_PACKET_SIZE/*l1*/,
                                 bsize/*l2*/,
                                 0/*sid*/,0/*tpc*/,0/*wid*/,0/*cache_hits_waiting*/,NULL,true,
                                 partial_write_mask_t(),
                                 L2_WRBK_ACC,
                                 L2_WTBK_DATA,
                                 -1/*pc*/);
   m_stats->memlatstat_start(mf);
   made_write_mfs++;
   L2todram_wbqueue->push(mf,gpu_sim_cycle);
   gpgpu_n_sent_writes++;
   return true;
}

void memory_partition_unit::L2c_print_cache_stat(unsigned &accesses, unsigned &misses) const
{
   FILE *fp = stdout;
   m_L2cache->shd_cache_print(fp,accesses,misses);
   m_mshr->print_stat(fp); 
   m_missTracker->print_stat(fp);
   m_accessLocality->print_stat(fp, false);
}

void memory_partition_unit::print( FILE *fp ) const
{
   if( !m_request_tracker.empty() ) {
      fprintf(fp,"Memory Parition %u: pending memory requests:\n", m_id);
      for( std::set<mem_fetch*>::const_iterator r=m_request_tracker.begin(); r != m_request_tracker.end(); ++r ) {
         mem_fetch *mf = *r;
         if( mf ) 
            mf->print(fp);
         else 
            fprintf(fp," <NULL mem_fetch?>\n");
      }
   }
   m_dram->print(fp); 
}

void memory_partition_unit::L2c_update_stat()
{
   unsigned i=m_id;
   if (m_icnt2cache_queue->get_length() > m_stats->L2_cbtoL2length[i])
      m_stats->L2_cbtoL2length[i] = m_icnt2cache_queue->get_length();
   if (m_icnt2cache_write_queue->get_length() > m_stats->L2_cbtoL2writelength[i])
      m_stats->L2_cbtoL2writelength[i] = m_icnt2cache_write_queue->get_length();
   if (L2tocbqueue->get_length() > m_stats->L2_L2tocblength[i])
      m_stats->L2_L2tocblength[i] = L2tocbqueue->get_length();
   if (dramtoL2queue->get_length() > m_stats->L2_dramtoL2length[i])
      m_stats->L2_dramtoL2length[i] = dramtoL2queue->get_length();
   if (dramtoL2writequeue->get_length() > m_stats->L2_dramtoL2writelength[i])
      m_stats->L2_dramtoL2writelength[i] = dramtoL2writequeue->get_length();
   if (L2todramqueue->get_length() > m_stats->L2_L2todramlength[i])
      m_stats->L2_L2todramlength[i] = L2todramqueue->get_length();
}

void memory_stats_t::L2c_print_stat( unsigned n_mem )
{
   unsigned i;

   printf("                                     ");
   for (i=0;i<n_mem;i++) {
      printf(" dram[%d]", i);
   }
   printf("\n");

   printf("cbtoL2 queue maximum length         ="); 
   for (i=0;i<n_mem;i++) {
      printf("%8d", L2_cbtoL2length[i]);
   }
   printf("\n");

   printf("cbtoL2 write queue maximum length   ="); 
   for (i=0;i<n_mem;i++) {
      printf("%8d", L2_cbtoL2writelength[i]);
   }
   printf("\n");

   printf("L2tocb queue maximum length         =");
   for (i=0;i<n_mem;i++) {
      printf("%8d", L2_L2tocblength[i]);
   }
   printf("\n");

   printf("dramtoL2 queue maximum length       =");
   for (i=0;i<n_mem;i++) {
      printf("%8d", L2_dramtoL2length[i]);
   }
   printf("\n");

   printf("dramtoL2 write queue maximum length ="); 
   for (i=0;i<n_mem;i++) {
      printf("%8d", L2_dramtoL2writelength[i]);
   }
   printf("\n");

   printf("L2todram queue maximum length       =");
   for (i=0;i<n_mem;i++) {
      printf("%8d", L2_L2todramlength[i]);
   }
   printf("\n");
}

void memory_stats_t::print( FILE *fp )
{
   fprintf(fp,"L2_write_miss = %d\n", L2_write_miss);
   fprintf(fp,"L2_write_hit = %d\n", L2_write_hit);
   fprintf(fp,"L2_read_miss = %d\n", L2_read_miss);
   fprintf(fp,"L2_read_hit = %d\n", L2_read_hit);
}

void gpgpu_sim::L2c_print_cache_stat() const
{
   unsigned i, j, k;
   for (i=0,j=0,k=0;i<m_n_mem;i++) 
      m_memory_partition_unit[i]->L2c_print_cache_stat(k,j);
   printf("L2 Cache Total Miss Rate = %0.3f\n", (float)j/k);
}

void gpgpu_sim::L2c_print_debug()
{
   unsigned i;

   printf("                                     ");
   for (i=0;i<m_n_mem;i++) 
      printf(" dram[%d]", i);
   printf("\n");

   printf("cbtoL2 queue length         ="); 
   for (i=0;i<m_n_mem;i++) 
      printf("%8d", m_memory_partition_unit[i]->get_cbtoL2queue_length() );
   printf("\n");

   printf("cbtoL2 write queue length   ="); 
   for (i=0;i<m_n_mem;i++) 
      printf("%8d", m_memory_partition_unit[i]->get_cbtoL2writequeue_length());
   printf("\n");

   printf("L2tocb queue length         =");
   for (i=0;i<m_n_mem;i++) {
      printf("%8d", m_memory_partition_unit[i]->get_L2tocbqueue_length());
   }
   printf("\n");

   printf("dramtoL2 queue length       =");
   for (i=0;i<m_n_mem;i++) {
      printf("%8d", m_memory_partition_unit[i]->get_dramtoL2queue_length());
   }
   printf("\n");

   printf("dramtoL2 write queue length ="); 
   for (i=0;i<m_n_mem;i++) {
      printf("%8d", m_memory_partition_unit[i]->get_dramtoL2writequeue_length());
   }
   printf("\n");

   printf("L2todram queue length       =");
   for (i=0;i<m_n_mem;i++) {
      printf("%8d", m_memory_partition_unit[i]->get_L2todramqueue_length());
   }
   printf("\n");

   printf("L2todram writeback queue length       =");
   for (i=0;i<m_n_mem;i++) {
      printf("%8d", m_memory_partition_unit[i]->get_L2todram_wbqueue_length());
   }
   printf("\n");
}

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

void memory_partition_unit::L2c_log(int task)
{
   if (task == SAMPLELOG) {
      StatAddSample(cbtol2_Dist,       m_icnt2cache_queue->get_length());
      StatAddSample(cbtoL2wr_Dist,     m_icnt2cache_write_queue->get_length());
      StatAddSample(L2tocb_Dist,       L2tocbqueue->get_length());
      StatAddSample(dramtoL2_Dist,     dramtoL2queue->get_length());
      StatAddSample(dramtoL2wr_Dist,   dramtoL2writequeue->get_length());
      StatAddSample(L2todram_Dist,     L2todramqueue->get_length());
      StatAddSample(L2todram_wb_Dist,  L2todram_wbqueue->get_length());
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(cbtol2_Dist);
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(cbtoL2wr_Dist);
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(L2tocb_Dist);
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(dramtoL2_Dist);
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(dramtoL2wr_Dist);
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(L2todram_Dist);
      printf ("Queue Length DRAM[%d] ",m_id); StatDisp(L2todram_wb_Dist);
   }
}

unsigned memory_partition_unit::flushL2() 
{ 
   return m_L2cache->flush(); 
}

void gpgpu_sim::L2c_latency_log_dump()
{
   for (unsigned i=0;i<m_n_mem;i++) 
      m_memory_partition_unit[i]->L2c_latency_log_dump();
}

void memory_partition_unit::L2c_latency_log_dump()
{
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(m_icnt2cache_queue->get_lat_stat());
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(m_icnt2cache_write_queue->get_lat_stat());
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(L2tocbqueue->get_lat_stat());
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(dramtoL2queue->get_lat_stat());
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(dramtoL2writequeue->get_lat_stat());
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(L2todramqueue->get_lat_stat());
   printf ("(LOGB2)Latency DRAM[%u] ",m_id); StatDisp(L2todram_wbqueue->get_lat_stat());
}

bool memory_partition_unit::busy() const 
{
   return !m_request_tracker.empty();
}

