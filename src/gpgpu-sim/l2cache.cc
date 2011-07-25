// Copyright (c) 2009-2011, Tor M. Aamodt
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


mem_fetch * partition_mf_allocator::alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const 
{
    assert( wr );
    mem_access_t access( type, addr, size, wr );
    mem_fetch *mf = new mem_fetch( access, 
                                   NULL,
                                   WRITE_PACKET_SIZE, 
                                   -1, 
                                   -1, 
                                   -1,
                                   m_memory_config );
    return mf;
}

memory_partition_unit::memory_partition_unit( unsigned partition_id, 
                                              const struct memory_config *config,
                                              class memory_stats_t *stats )
{
    m_id = partition_id;
    m_config=config;
    m_stats=stats;
    m_dram = new dram_t(m_id,m_config,m_stats,this);

    char L2c_name[32];
    snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
    m_L2interface = new L2interface(this);
    m_mf_allocator = new partition_mf_allocator(config);
    m_L2cache = new data_cache(L2c_name,m_config->m_L2_config,-1,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE);

    unsigned int icnt_L2;
    unsigned int L2_dram;
    unsigned int dram_L2;
    unsigned int L2_icnt;

    sscanf(m_config->gpgpu_L2_queue_config,"%u:%u:%u:%u", &icnt_L2,&L2_dram,&dram_L2,&L2_icnt );
    m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2",0,icnt_L2); 
    m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram",0,L2_dram);
    m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2",0,dram_L2);
    m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt",0,L2_icnt);
    wb_addr=-1;
}

memory_partition_unit::~memory_partition_unit()
{
    delete m_icnt_L2_queue;
    delete m_L2_dram_queue;
    delete m_dram_L2_queue;
    delete m_L2_icnt_queue;
    delete m_L2cache;
    delete m_L2interface;
}

void memory_partition_unit::cache_cycle( unsigned cycle )
{
    // L2 fill responses 
    if ( m_L2cache->access_ready() && !m_L2_icnt_queue->full() ) {
        mem_fetch *mf = m_L2cache->next_access();
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
    }
    // DRAM to L2 (texture) and icnt (not texture)
    if ( !m_dram_L2_queue->empty() ) {
        mem_fetch *mf = m_dram_L2_queue->top();
        if ( m_L2cache->waiting_for_fill(mf) ) { 
            mf->set_status(IN_PARTITION_L2_FILL_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_dram_L2_queue->pop();
        } else if ( !m_L2_icnt_queue->full() ) {
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_icnt_queue->push(mf);
            m_dram_L2_queue->pop();
        }
    }

    // prior L2 misses inserted into m_L2_dram_queue here
    m_L2cache->cycle(); 

    // new L2 texture accesses and/or non-texture accesses
    if ( !m_L2_dram_queue->full() && !m_icnt_L2_queue->empty() ) {
        mem_fetch *mf = m_icnt_L2_queue->top();
        if ( (m_config->m_L2_texure_only && mf->istexture()) || (!m_config->m_L2_texure_only) ) {
            if ( !m_L2_icnt_queue->full() ) {
                std::list<cache_event> events;
                enum cache_request_status status = m_L2cache->access(mf->get_partition_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
                bool write_sent = was_write_sent(events);
                bool read_sent = was_read_sent(events);

                if ( status == HIT ) {
                    if( !write_sent ) {
                        // L2 cache replies
                        assert(!read_sent);
                        if( mf->get_access_type() == L1_WRBK_ACC ) {
                            m_request_tracker.erase(mf);
                            delete mf;
                        } else {
                            mf->set_reply();
                            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                            m_L2_icnt_queue->push(mf);
                        }
                        m_icnt_L2_queue->pop();
                    } else {
                        assert(write_sent);
                        m_icnt_L2_queue->pop();
                    }
                } else if ( status != RESERVATION_FAIL ) {
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
                } else {
                    assert(!write_sent);
                    assert(!read_sent);
                    // L2 cache lock-up: will try again next cycle
                }
            }
        } else {
            // non-texture access
            mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_dram_queue->push(mf);
            m_icnt_L2_queue->pop();
        }
    }

    // ROP delay queue
    if( !m_rop.empty() && (cycle >= m_rop.front().ready_cycle) && !m_icnt_L2_queue->full() ) {
        mem_fetch* mf = m_rop.front().req;
        m_rop.pop();
        m_icnt_L2_queue->push(mf);
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
    }
}

bool memory_partition_unit::full() const
{
    return m_icnt_L2_queue->full();
}

void memory_partition_unit::print_cache_stat(unsigned &accesses, unsigned &misses) const
{
    FILE *fp = stdout;
    m_L2cache->print(fp,accesses,misses);
}

void memory_partition_unit::print( FILE *fp ) const
{
    if ( !m_request_tracker.empty() ) {
        fprintf(fp,"Memory Parition %u: pending memory requests:\n", m_id);
        for ( std::set<mem_fetch*>::const_iterator r=m_request_tracker.begin(); r != m_request_tracker.end(); ++r ) {
            mem_fetch *mf = *r;
            if ( mf )
                mf->print(fp);
            else
                fprintf(fp," <NULL mem_fetch?>\n");
        }
    }
    m_L2cache->display_state(fp);
    m_dram->print(fp); 
}

void memory_stats_t::print( FILE *fp )
{
    fprintf(fp,"L2_write_miss = %d\n", L2_write_miss);
    fprintf(fp,"L2_write_hit = %d\n", L2_write_hit);
    fprintf(fp,"L2_read_miss = %d\n", L2_read_miss);
    fprintf(fp,"L2_read_hit = %d\n", L2_read_hit);
}

void memory_stats_t::visualizer_print( gzFile visualizer_file )
{
   gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
   gzprintf(visualizer_file, "Ltwowritehit: %d\n",  L2_write_hit);
   gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
   gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
   if (num_mfs) 
      gzprintf(visualizer_file, "averagemflatency: %lld\n", mf_total_lat/num_mfs);
}

void gpgpu_sim::L2c_print_cache_stat() const
{
    unsigned i, j, k;
    for (i=0,j=0,k=0;i<m_memory_config->m_n_mem;i++)
        m_memory_partition_unit[i]->print_cache_stat(k,j);
    printf("L2 Cache Total Miss Rate = %0.3f\n", (float)j/k);
}

unsigned memory_partition_unit::flushL2() 
{ 
    m_L2cache->flush(); 
    return 0; // L2 is read only in this version
}

bool memory_partition_unit::busy() const 
{
    return !m_request_tracker.empty();
}

void memory_partition_unit::push( mem_fetch* req, unsigned long long cycle ) 
{
    if (req) {
        m_request_tracker.insert(req);
        m_stats->memlatstat_icnt2mem_pop(req);
        if( req->istexture() ) {
            m_icnt_L2_queue->push(req);
            req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        } else {
            rop_delay_t r;
            r.req = req;
            r.ready_cycle = cycle + 115; // Add 115*4=460 delay cycles
            m_rop.push(r);
            req->set_status(IN_PARTITION_ROP_DELAY,gpu_sim_cycle+gpu_tot_sim_cycle);
        }
    }
}

mem_fetch* memory_partition_unit::pop() 
{
    mem_fetch* mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if ( mf && mf->isatomic() )
        mf->do_atomic();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        delete mf;
        mf = NULL;
    } 
    return mf;
}

mem_fetch* memory_partition_unit::top() 
{
    mem_fetch *mf = m_L2_icnt_queue->top();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        m_L2_icnt_queue->pop();
        m_request_tracker.erase(mf);
        delete mf;
        mf = NULL;
    } 
    return mf;
}

void memory_partition_unit::set_done( mem_fetch *mf )
{
    m_request_tracker.erase(mf);
}

void memory_partition_unit::dram_cycle() 
{ 
    // pop completed memory request from dram and push it to dram-to-L2 queue 
    if ( !m_dram_L2_queue->full() ) {
        mem_fetch* mf = m_dram->pop();
        if (mf) {
            if( mf->get_access_type() == L1_WRBK_ACC ) {
                m_request_tracker.erase(mf);
                delete mf;
            } else {
                m_dram_L2_queue->push(mf);
                mf->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            }
        }
    }
    m_dram->cycle(); 
    m_dram->dram_log(SAMPLELOG);   

    if( !m_dram->full() && !m_L2_dram_queue->empty() ) {
        mem_fetch *mf = m_L2_dram_queue->pop();
        m_dram->push(mf);
    }
}
