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
#include "../abstract_hardware_model.h"

#include <list>
#include <queue>

class mem_fetch;

class partition_mf_allocator : public mem_fetch_allocator {
public:
    partition_mf_allocator( const memory_config *config )
    {
        m_memory_config = config;
    }
    virtual mem_fetch * alloc(const class warp_inst_t &inst, const mem_access_t &access) const 
    {
        abort();
        return NULL;
    }
    virtual mem_fetch * alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr) const;
private:
    const memory_config *m_memory_config;
};

class memory_partition_unit 
{
public:
   memory_partition_unit( unsigned partition_id, const struct memory_config *config, class memory_stats_t *stats );
   ~memory_partition_unit(); 

   bool busy() const;

   void cache_cycle( unsigned cycle );
   void dram_cycle();

   bool full() const;
   void push( class mem_fetch* mf, unsigned long long clock_cycle );
   class mem_fetch* pop(); 
   class mem_fetch* top();

   unsigned flushL2();

   void visualizer_print( gzFile visualizer_file );
   void print_cache_stat(unsigned &accesses, unsigned &misses) const;
   void print_stat( FILE *fp ) { m_dram->print_stat(fp); }
   void visualize() const { m_dram->visualize(); }
   void print( FILE *fp ) const;

private:
// data
   unsigned m_id;
   const struct memory_config *m_config;
   class dram_t *m_dram;
   class data_cache *m_L2cache;
   class L2interface *m_L2interface;
   partition_mf_allocator *m_mf_allocator;

   // model delay of ROP units with a fixed latency
   struct rop_delay_t
   {
    	unsigned long long ready_cycle;
    	class mem_fetch* req;
   };
   std::queue<rop_delay_t> m_rop; 

   // these are various FIFOs between units within a memory partition
   fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_dram_queue;
   fifo_pipeline<mem_fetch> *m_dram_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_icnt_queue; // L2 cache hit response queue

   class mem_fetch *L2dramout; 
   unsigned long long int wb_addr;

   class memory_stats_t *m_stats;

   std::set<mem_fetch*> m_request_tracker;

   friend class L2interface;
};

class L2interface : public mem_fetch_interface {
public:
    L2interface( memory_partition_unit *unit ) { m_unit=unit; }
    virtual bool full( unsigned size, bool write) const 
    {
        // assume read and write packets all same size
        return m_unit->m_L2_dram_queue->full();
    }
    virtual void push(mem_fetch *mf) 
    {
        mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,0/*FIXME*/);
        m_unit->m_L2_dram_queue->push(mf);
    }
private:
    memory_partition_unit *m_unit;
};

#endif
