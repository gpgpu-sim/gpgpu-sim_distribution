/* 
 * l2cache.h
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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "dram.h"
#include "../tr1_hash_map.h"
#include "../abstract_hardware_model.h"

#include <list>
#include <queue>

class mem_fetch;

class L2c_mshr 
{
private:
   typedef std::list<const mem_fetch*> mem_fetch_list;
   typedef tr1_hash_map<address_type, mem_fetch_list> L2missGroup;
   L2missGroup m_L2missgroup; // structure tracking redundant dram access

   struct active_chain {
      address_type cacheTag;
      mem_fetch_list *list;
      active_chain() : cacheTag(0xDEADBEEF), list(NULL) { }
   };
   active_chain m_active_mshr_chain; 
   size_t m_linesize; // L2 cache line size

   const size_t m_n_entries; // total number of entries available
   size_t m_entries_used; // number of entries in use

   int m_n_miss; 
   int m_n_miss_serviced_by_dram;
   int m_n_mshr_hits;
   size_t m_max_entries_used; 
   
   address_type cache_tag(const mem_fetch *mf) const;

public:
   L2c_mshr(size_t linesize, size_t n_entries = 64) 
   : m_linesize(linesize), m_n_entries(n_entries), m_entries_used(0), 
     m_n_miss(0), m_n_miss_serviced_by_dram(0), m_n_mshr_hits(0), m_max_entries_used(0) { }
  
   // add a cache miss to MSHR, return true if this access is hit another existing entry and merges with it
   bool new_miss(const mem_fetch *mf);

   // notify MSHR that a new cache line has been fetched, activate the associated MSHR chain
   void miss_serviced(const mem_fetch *mf);

   // probe if there are pending hits left in this MSHR chain
   bool mshr_chain_empty();

   // peek the first entry in the active MSHR chain
   mem_fetch *mshr_chain_top();

   // pop the first entry in the active MSHR chain
   void mshr_chain_pop(); 

   void print(FILE *fout = stdout); 
   void print_stat(FILE *fout = stdout) const; 
};

class L2c_miss_tracker
{
private:
   typedef std::set<mem_fetch*> mem_fetch_set;
   typedef tr1_hash_map<address_type, mem_fetch_set> L2missGroup;
   L2missGroup m_L2missgroup; // structure tracking redundant dram access
   size_t m_linesize; // L2 cache line size

   typedef tr1_hash_map<address_type, int> L2redundantCnt; 
   L2redundantCnt m_L2redundantCnt; 

   int m_totalL2redundantAcc;

   address_type cache_tag(const mem_fetch *mf) const;

public:
   L2c_miss_tracker(size_t linesize) : m_linesize(linesize), m_totalL2redundantAcc(0) { }
   void new_miss(mem_fetch *mf);
   void miss_serviced(mem_fetch *mf);

   void print(FILE *fout, bool brief = true);
   void print_stat(FILE *fout, bool brief = true) const;
};

class L2c_access_locality
{
public:
   L2c_access_locality(size_t linesize) : m_linesize(linesize), m_totalL2accAcc(0) { }
   void print_stat(FILE *fout, bool brief = true) const;
   void access(mem_fetch *mf);
private:
   address_type cache_tag(const mem_fetch *mf) const;

   size_t m_linesize; // L2 cache line size

   typedef tr1_hash_map<address_type, int> L2accCnt; 
   L2accCnt m_L2accCnt; 
   int m_totalL2accAcc;
};

class memory_partition_unit 
{
public:
   memory_partition_unit( unsigned partition_id, struct memory_config *config);
   ~memory_partition_unit(); 

   void set_stats( class memory_stats_t *stats );

   void cache_cycle();

   bool has_cache() { return L2cache != NULL; }
   unsigned L2c_get_linesize();
   bool full() const;
   bool busy() const;

   void push( class mem_fetch* mf, unsigned long long clock_cycle );
   class mem_fetch* pop(); 
   class mem_fetch* top();
   void issueCMD();
   void visualizer_print( gzFile visualizer_file );
   void L2c_latency_log_dump();
   void L2c_log(int task);
   unsigned L2c_cache_flush();
   void L2c_print_cache_stat(unsigned &accesses, unsigned &misses) const;

   unsigned get_cbtoL2queue_length() const { return cbtoL2queue->get_length(); }
   unsigned get_cbtoL2writequeue_length() const { return cbtoL2writequeue->get_length(); }
   unsigned get_dramtoL2queue_length() const { return dramtoL2queue->get_length(); }
   unsigned get_dramtoL2writequeue_length() const { return dramtoL2writequeue->get_length(); }
   unsigned get_L2todramqueue_length() const { return L2todramqueue->get_length(); }
   unsigned get_L2todram_wbqueue_length() const { return L2todram_wbqueue->get_length(); }
   unsigned get_L2tocbqueue_length() const { return L2tocbqueue->get_length(); }

   void print_stat( FILE *fp ) { m_dram->print_stat(fp); }
   void visualize() const { m_dram->visualize(); }
   unsigned dram_que_length() const { return m_dram->que_length(); }
   void queue_latency_log_dump( FILE *fp ) { m_dram->queue_latency_log_dump(fp); }
   void print( FILE *fp ) { m_dram->print(fp); }

private:
   void request_tracker_insert(class mem_fetch *mf);
   void request_tracker_erase(class mem_fetch *mf);

   // pop completed memory request from dram and push it to dram-to-L2 queue 
   void L2c_get_dram_output();

   // service memory request in icnt-to-L2 queue, writing to L2 as necessary
   // (if L2 writeback miss, writeback to memory) 
   void L2c_service_mem_req();
   
   // service memory request in L2todramqueue, pushing to dram 
   void L2c_push_miss_to_dram();
   
   // service memory request in dramtoL2queue, writing to L2 as necessary
   // (may cause cache eviction and subsequent writeback) 
   void L2c_process_dram_output();
   
   bool L2c_write_back( unsigned long long int addr, int bsize );
   
   // probe L2 cache for fullness 
   struct mem_fetch* L2c_pop( dram_t *dram_p );
   
   void L2c_init_stat(unsigned n_mem);
   void L2c_update_stat();
   void L2c_print_debug();

// data
   unsigned m_id;
   struct memory_config *m_config;
   class dram_t *m_dram;
   struct shd_cache_t *L2cache;

   // model delay of ROP units with a fixed latency
   struct rop_delay_t
   {
    	unsigned long long ready_cycle;
    	class mem_fetch* req;
   };
   std::queue<rop_delay_t> m_rop; 

   // these are various FIFOs between units within a memory partition
   fifo_pipeline<mem_fetch> *cbtoL2queue;
   fifo_pipeline<mem_fetch> *cbtoL2writequeue;
   fifo_pipeline<mem_fetch> *dramtoL2queue;
   fifo_pipeline<mem_fetch> *dramtoL2writequeue;
   fifo_pipeline<mem_fetch> *L2todramqueue;
   fifo_pipeline<mem_fetch> *L2todram_wbqueue;
   fifo_pipeline<mem_fetch> *L2tocbqueue;

   mem_fetch *L2request; //request currently being serviced by the L2 Cache

   L2c_mshr *m_mshr; // mshr model 
   L2c_miss_tracker *m_missTracker; // tracker observing for redundant misses
   L2c_access_locality *m_accessLocality; // tracking true locality of L2 Cache access 

   class mem_fetch *L2dramout; 
   unsigned long long int wb_addr;

   class memory_stats_t *m_stats;

   class Stats *cbtol2_Dist;  
   class Stats *cbtoL2wr_Dist;  
   class Stats *L2tocb_Dist; 
   class Stats *dramtoL2_Dist;
   class Stats *dramtoL2wr_Dist;
   class Stats *L2todram_Dist;
   class Stats *L2todram_wb_Dist;

   std::set<mem_fetch*> m_request_tracker;
};

#endif
