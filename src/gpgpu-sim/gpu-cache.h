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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"

enum cache_block_state {
    INVALID,
    RESERVED,
    VALID,
    MODIFIED
};

enum cache_request_status {
    HIT,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL
};

enum cache_event {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT
};

struct cache_block_t {
    cache_block_t()
    {
        m_tag=0;
        m_block_addr=0;
        m_alloc_time=0;
        m_fill_time=0;
        m_last_access_time=0;
        m_status=INVALID;
    }
    void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time )
    {
        m_tag=tag;
        m_block_addr=block_addr;
        m_alloc_time=time;
        m_last_access_time=time;
        m_fill_time=0;
        m_status=RESERVED;
    }
    void fill( unsigned time )
    {
        assert( m_status == RESERVED );
        m_status=VALID;
        m_fill_time=time;
    }

    new_addr_type    m_tag;
    new_addr_type    m_block_addr;
    unsigned         m_alloc_time;
    unsigned         m_last_access_time;
    unsigned         m_fill_time;
    cache_block_state    m_status;
};

enum replacement_policy_t {
    LRU,
    FIFO
};

enum write_policy_t {
    READ_ONLY,
    WRITE_BACK,
    WRITE_THROUGH
};

enum allocation_policy_t {
    ON_MISS,
    ON_FILL
};

enum mshr_config_t {
    TEX_FIFO,
    ASSOC // normal cache 
};

class cache_config {
public:
    cache_config() 
    { 
        m_valid = false; 
        m_disabled = false;
        m_config_string = NULL; // set by option parser
    }
    void init()
    {
        assert( m_config_string );
        char rp, wp, ap, mshr_type;
        int ntok = sscanf(m_config_string,"%u:%u:%u:%c:%c:%c,%c:%u:%u,%u:%u", 
                          &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap,
                          &mshr_type,&m_mshr_entries,&m_mshr_max_merge,&m_miss_queue_size,&m_result_fifo_entries);
        if ( ntok < 10 ) {
            if ( !strcmp(m_config_string,"none") ) {
                m_disabled = true;
                return;
            }
            exit_parse_error();
        }
        switch (rp) {
        case 'L': m_replacement_policy = LRU; break;
        case 'F': m_replacement_policy = FIFO; break;
        default: exit_parse_error();
        }
        switch (wp) {
        case 'R': m_write_policy = READ_ONLY; break;
        case 'B': m_write_policy = WRITE_BACK; break;
        case 'T': m_write_policy = WRITE_THROUGH; break;
        default: exit_parse_error();
        }
        switch (ap) {
        case 'm': m_alloc_policy = ON_MISS; break;
        case 'f': m_alloc_policy = ON_FILL; break;
        default: exit_parse_error();
        }
        switch (mshr_type) {
        case 'F': m_mshr_type = TEX_FIFO; assert(ntok==11); break;
        case 'A': m_mshr_type = ASSOC; break;
        default: exit_parse_error();
        }
        m_line_sz_log2 = LOGB2(m_line_sz);
        m_nset_log2 = LOGB2(m_nset);
        m_valid = true;
    }
    bool disabled() const { return m_disabled;}
    unsigned get_line_sz() const
    {
        assert( m_valid );
        return m_line_sz;
    }
    unsigned get_num_lines() const
    {
        assert( m_valid );
        return m_nset * m_assoc;
    }

    void print( FILE *fp ) const
    {
        fprintf( fp, "Size = %d B (%d Set x %d-way x %d byte line)\n", 
                 m_line_sz * m_nset * m_assoc,
                 m_nset, m_assoc, m_line_sz );
    }

    unsigned set_index( new_addr_type addr ) const 
    {
        return(addr >> m_line_sz_log2) & (m_nset-1);
    }
    new_addr_type tag( new_addr_type addr ) const
    {
        return addr >> (m_line_sz_log2+m_nset_log2);
    }
    new_addr_type block_addr( new_addr_type addr ) const
    {
        return addr & ~(m_line_sz-1);
    }

    char *m_config_string;

private:
    void exit_parse_error()
    {
        printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n", m_config_string );
        abort();
    }

    bool m_valid;
    bool m_disabled;
    unsigned m_line_sz;
    unsigned m_line_sz_log2;
    unsigned m_nset;
    unsigned m_nset_log2;
    unsigned m_assoc;

    enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
    enum write_policy_t m_write_policy;             // 'T' = write through, 'B' = write back, 'R' = read only
    enum allocation_policy_t m_alloc_policy;        // 'm' = allocate on miss, 'f' = allocate on fill
    enum mshr_config_t m_mshr_type;

    union {
        unsigned m_mshr_entries;
        unsigned m_fragment_fifo_entries;
    };
    union {
        unsigned m_mshr_max_merge;
        unsigned m_request_fifo_entries;
    };
    union {
        unsigned m_miss_queue_size;
        unsigned m_rob_entries;
    };
    unsigned m_result_fifo_entries;

    friend class tag_array;
    friend class read_only_cache;
    friend class tex_cache;
    friend class data_cache;
};

class tag_array {
public:
    tag_array( const cache_config &config, int core_id, int type_id ); 
    ~tag_array();

    enum cache_request_status probe( new_addr_type addr, unsigned &idx ) const;
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx );
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ); 

    void fill( new_addr_type addr, unsigned time );
    void fill( unsigned idx, unsigned time );

    unsigned size() const { return m_config.get_num_lines();}
    cache_block_t &get_block(unsigned idx) { return m_lines[idx];}

    void flush(); // flash invalidate all entries 
    void new_window();

    void print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const;
    float windowed_miss_rate( bool minus_pending_hit ) const;

protected:

    const cache_config &m_config;

    cache_block_t *m_lines; /* nbanks x nset x assoc lines in total */

    unsigned m_access;
    unsigned m_miss;
    unsigned m_pending_hit; // number of cache miss that hit a line that is allocated but not filled

    // performance counters for calculating the amount of misses within a time window
    unsigned m_prev_snapshot_access;
    unsigned m_prev_snapshot_miss;
    unsigned m_prev_snapshot_pending_hit; 

    int m_core_id; // which shader core is using this
    int m_type_id; // what kind of cache is this (normal, texture, constant)
};

class mshr_table {
public:
    mshr_table( unsigned num_entries, unsigned max_merged )
    : m_num_entries(num_entries),
    m_max_merged(max_merged),
#ifndef USE_MAP
    m_data(2*num_entries)
#endif
    {
    }

    // is there a pending request to the lower memory level already?
    bool probe( new_addr_type block_addr ) const
    {
        table::const_iterator a = m_data.find(block_addr);
        return a != m_data.end();
    }

    // is there space for tracking a new memory access?
    bool full( new_addr_type block_addr ) const 
    { 
        table::const_iterator i=m_data.find(block_addr);
        if ( i != m_data.end() )
            return i->second.size() >= m_max_merged;
        else
            return m_data.size() >= m_num_entries; 
    }

    // add or merge this access
    void add( new_addr_type block_addr, mem_fetch *mf )
    {
        m_data[block_addr].push_back(mf);
        assert( m_data.size() <= m_num_entries );
        assert( m_data[block_addr].size() <= m_max_merged );
    }

    // true if cannot accept new fill responses
    bool busy() const 
    { 
        return false;
    }

    // accept a new cache fill response: mark entry ready for processing
    void mark_ready( new_addr_type block_addr )
    {
        assert( !busy() );
        table::iterator a = m_data.find(block_addr);
        assert( a != m_data.end() ); // don't remove same request twice
        m_current_response.push_back( block_addr );
        assert( m_current_response.size() <= m_data.size() );
    }

    // true if ready accesses exist
    bool access_ready() const 
    {
        return !m_current_response.empty(); 
    }

    // next ready access
    mem_fetch *next_access()
    {
        assert( access_ready() );
        new_addr_type block_addr = m_current_response.front();
        assert( !m_data[block_addr].empty() );
        mem_fetch *result = m_data[block_addr].front();
        m_data[block_addr].pop_front();
        if ( m_data[block_addr].empty() ) {
            // release entry
            m_data.erase(block_addr); 
            m_current_response.pop_front();
        }
        return result;
    }

    void display( FILE *fp ) const
    {
        fprintf(fp,"MSHR contents\n");
        for ( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
            unsigned block_addr = e->first;
            fprintf(fp,"MSHR: tag=0x%06x, %zu entries : ", block_addr, e->second.size());
            if ( !e->second.empty() ) {
                mem_fetch *mf = e->second.front();
                fprintf(fp,"%p :",mf);
                mf->print(fp);
            } else {
                fprintf(fp," no memory requests???\n");
            }
        }
    }

private:

    // finite sized, fully associative table, with a finite maximum number of merged requests
    const unsigned m_num_entries;
    const unsigned m_max_merged;

    typedef std::list<mem_fetch*> entry;
    typedef my_hash_map<new_addr_type,entry> table;
    table m_data;

    // it may take several cycles to process the merged requests
    bool m_current_response_ready;
    std::list<new_addr_type> m_current_response;
};

class cache_t {
public:
    virtual ~cache_t() {}
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;
};

bool was_write_sent( const std::list<cache_event> &events );
bool was_read_sent( const std::list<cache_event> &events );

class read_only_cache : public cache_t {
public:
    read_only_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                     enum mem_fetch_status status )
    : m_config(config), m_tag_array(config,core_id,type_id), m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge)
    {
        m_name = name;
        assert(config.m_mshr_type == ASSOC);
        m_memport=memport;
        m_miss_queue_status = status;
    }

    // access cache: returns RESERVATION_FAIL if request could not be accepted (for any reason)
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) 
    {
        assert(m_config.m_write_policy == READ_ONLY);
        assert(!mf->get_is_write());
        new_addr_type block_addr = m_config.block_addr(addr);
        unsigned cache_index = (unsigned)-1;
        enum cache_request_status status = m_tag_array.probe(block_addr,cache_index);
        if ( status == HIT ) {
            m_tag_array.access(block_addr,time,cache_index); // update LRU state 
            return HIT;
        }
        if ( status != RESERVATION_FAIL ) {
            bool mshr_hit = m_mshrs.probe(block_addr);
            bool mshr_avail = !m_mshrs.full(block_addr);
            if ( mshr_hit && mshr_avail ) {
                m_tag_array.access(addr,time,cache_index);
                m_mshrs.add(block_addr,mf);
                return MISS;
            } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
                m_tag_array.access(addr,time,cache_index);
                m_mshrs.add(block_addr,mf);
                m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
                mf->set_data_size( m_config.get_line_sz() );
                m_miss_queue.push_back(mf);
                mf->set_status(m_miss_queue_status,time);
                events.push_back(READ_REQUEST_SENT);
                return MISS;
            }
        }
        return RESERVATION_FAIL;
    }

    void cycle() 
    {
        // send next request to lower level of memory
        if ( !m_miss_queue.empty() ) {
            mem_fetch *mf = m_miss_queue.front();
            if ( !m_memport->full(mf->get_data_size(),mf->get_is_write()) ) {
                m_miss_queue.pop_front();
                m_memport->push(mf);
            }
        }
    }

    // interface for response from lower memory level (model bandwidth restictions in caller)
    void fill( mem_fetch *mf, unsigned time )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); 
        assert( e != m_extra_mf_fields.end() );
        assert( e->second.m_valid );
        mf->set_data_size( e->second.m_data_size );
        if ( m_config.m_alloc_policy == ON_MISS )
            m_tag_array.fill(e->second.m_cache_index,time);
        else if ( m_config.m_alloc_policy == ON_FILL )
            m_tag_array.fill(e->second.m_block_addr,time);
        else abort();
        m_mshrs.mark_ready(e->second.m_block_addr);
        m_extra_mf_fields.erase(mf);
    }

    bool waiting_for_fill( mem_fetch *mf )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); 
        return e != m_extra_mf_fields.end();
    }

    // are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const
    {
        return m_mshrs.access_ready();
    }

    // pop next ready access (does not include accesses that "HIT")
    mem_fetch *next_access() 
    { 
        return m_mshrs.next_access(); 
    }

    // flash invalidate all entries in cache
    void flush()
    {
        m_tag_array.flush();
    }

    void print(FILE *fp, unsigned &accesses, unsigned &misses) const
    {
        fprintf( fp, "Cache %s:\t", m_name.c_str() );
        m_tag_array.print(fp,accesses,misses);
    }

    void display_state( FILE *fp ) const
    {
        fprintf(fp,"Cache %s:\n", m_name.c_str() );
        m_mshrs.display(fp);
        fprintf(fp,"\n");
    }


    protected:
    std::string m_name;
    const cache_config &m_config;
    tag_array  m_tag_array;
    mshr_table m_mshrs;
    std::list<mem_fetch*> m_miss_queue;
    enum mem_fetch_status m_miss_queue_status;
    mem_fetch_interface *m_memport;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( new_addr_type a, unsigned i, unsigned d ) 
        {
            m_valid = true;
            m_block_addr = a;
            m_cache_index = i;
            m_data_size = d;
        }
        bool m_valid;
        new_addr_type m_block_addr;
        unsigned m_cache_index;
        unsigned m_data_size;
    };

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;
};

// This is meant to model the first level data cache in Fermi.
// It is write-evict (global) or write-back (local) at the granularity 
// of individual blocks (the policy used in fermi according to the CUDA manual)

class data_cache : public read_only_cache {
public:
    data_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status )
    : read_only_cache(name,config,core_id,type_id,memport,status)
    {
        m_memfetch_creator=mfcreator;
    }
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) 
    {
        bool wr = mf->get_is_write();
        enum mem_access_type type = mf->get_access_type();
        bool evict = (type == GLOBAL_ACC_W); // evict a line that hits on global memory write

        new_addr_type block_addr = m_config.block_addr(addr);
        unsigned cache_index = (unsigned)-1;
        enum cache_request_status status = m_tag_array.probe(block_addr,cache_index);
        if ( status == HIT ) {
            if ( evict ) {
                if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
                    return RESERVATION_FAIL; // cannot handle request this cycle

                // generate a write through
                cache_block_t &block = m_tag_array.get_block(cache_index);
                assert( block.m_status != MODIFIED ); // fails if block was allocated by a ld.local and now accessed by st.global
                m_miss_queue.push_back(mf);
                mf->set_status(m_miss_queue_status,time); 
                events.push_back(WRITE_REQUEST_SENT);

                // invalidate block 
                block.m_status = INVALID;
            } else {
                m_tag_array.access(block_addr,time,cache_index); // update LRU state 
                if ( wr ) {
                    assert( type == LOCAL_ACC_W || /*l2 only*/ type == L1_WRBK_ACC );
                    // treated as write back...
                    cache_block_t &block = m_tag_array.get_block(cache_index);
                    block.m_status = MODIFIED;
                }
            }
            return HIT;
        } else if ( status != RESERVATION_FAIL ) {
            if ( wr ) {
                if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
                    return RESERVATION_FAIL; // cannot handle request this cycle

                // on miss, generate write through (no write buffering -- too many threads for that)
                m_miss_queue.push_back(mf); 
                mf->set_status(m_miss_queue_status,time); 
                events.push_back(WRITE_REQUEST_SENT);
                return MISS;
            } else {
                if ( (m_miss_queue.size()+1) >= m_config.m_miss_queue_size )
                    return RESERVATION_FAIL; // cannot handle request this cycle (might need to generate two requests)

                bool do_miss = false;
                bool wb = false;
                cache_block_t evicted;

                bool mshr_hit = m_mshrs.probe(block_addr);
                bool mshr_avail = !m_mshrs.full(block_addr);
                if ( mshr_hit && mshr_avail ) {
                    m_tag_array.access(addr,time,cache_index,wb,evicted);
                    m_mshrs.add(block_addr,mf);
                    do_miss = true;
                } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
                    m_tag_array.access(addr,time,cache_index,wb,evicted);
                    m_mshrs.add(block_addr,mf);
                    m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
                    mf->set_data_size( m_config.get_line_sz() );
                    m_miss_queue.push_back(mf);
                    mf->set_status(m_miss_queue_status,time); 
                    events.push_back(READ_REQUEST_SENT);
                    do_miss = true;
                }
                if( wb ) {
                    assert(do_miss);
                    mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,L1_WRBK_ACC,m_config.get_line_sz(),true);
                    events.push_back(WRITE_BACK_REQUEST_SENT);
                    m_miss_queue.push_back(wb);
                    wb->set_status(m_miss_queue_status,time); 
                }
                if( do_miss ) 
                    return MISS;
            }
        }

        return RESERVATION_FAIL;
    }
    private:
    mem_fetch_allocator *m_memfetch_creator;
};

// See the following paper to understand this cache model:
// 
// Igehy, et al., Prefetching in a Texture Cache Architecture, 
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
public:
    tex_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
               enum mem_fetch_status request_status, 
               enum mem_fetch_status rob_status )
    : m_config(config), 
    m_tags(config,core_id,type_id), 
    m_fragment_fifo(config.m_fragment_fifo_entries), 
    m_request_fifo(config.m_request_fifo_entries),
    m_rob(config.m_rob_entries),
    m_result_fifo(config.m_result_fifo_entries)
    {
        m_name = name;
        assert(config.m_mshr_type == TEX_FIFO);
        assert(config.m_write_policy == READ_ONLY);
        assert(config.m_alloc_policy == ON_MISS);
        m_memport=memport;
        m_cache = new data_block[ config.get_num_lines() ];
        m_request_queue_status = request_status;
        m_rob_status = rob_status;
    }

    // return values: RESERVATION_FAIL if request could not be accepted 
    // otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT 
    // since unlike a normal CPU cache, a "HIT" in texture cache does not 
    // mean the data is ready (still need to get through fragment fifo)
    enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) {
        if ( m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full() )
            return RESERVATION_FAIL;

        // at this point, we will accept the request : access tags and immediately allocate line
        new_addr_type block_addr = m_config.block_addr(addr);
        unsigned cache_index = (unsigned)-1;
        enum cache_request_status status = m_tags.access(block_addr,time,cache_index);
        assert( status != RESERVATION_FAIL );
        assert( status != HIT_RESERVED ); // as far as tags are concerned: HIT or MISS 
        m_fragment_fifo.push( fragment_entry(mf,cache_index,status==MISS,mf->get_data_size()) );
        if ( status == MISS ) {
            // we need to send a memory request...
            unsigned rob_index = m_rob.push( rob_entry(cache_index, mf, block_addr) );
            m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
            mf->set_data_size(m_config.get_line_sz());
            m_tags.fill(cache_index,time); // mark block as valid 
            m_request_fifo.push(mf);
            mf->set_status(m_request_queue_status,time);
            events.push_back(READ_REQUEST_SENT);
            return MISS;
        } else {
            // the value *will* *be* in the cache already
            return HIT_RESERVED;
        }
    }

    void cycle() 
    {
        // send next request to lower level of memory
        if ( !m_request_fifo.empty() ) {
            mem_fetch *mf = m_request_fifo.peek();
            if ( !m_memport->full(mf->get_ctrl_size(),false) ) {
                m_request_fifo.pop();
                m_memport->push(mf);
            }
        }
        // read ready lines from cache
        if ( !m_fragment_fifo.empty() && !m_result_fifo.full() ) {
            const fragment_entry &e = m_fragment_fifo.peek();
            if ( e.m_miss ) {
                // check head of reorder buffer to see if data is back from memory
                unsigned rob_index = m_rob.next_pop_index();
                const rob_entry &r = m_rob.peek(rob_index);
                assert( r.m_request == e.m_request );
                assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr()) );
                if ( r.m_ready ) {
                    assert( r.m_index == e.m_cache_index );
                    m_cache[r.m_index].m_valid = true;
                    m_cache[r.m_index].m_block_addr = r.m_block_addr;
                    m_result_fifo.push(e.m_request);
                    m_rob.pop();
                    m_fragment_fifo.pop();
                }
            } else {
                // hit:
                assert( m_cache[e.m_cache_index].m_valid ); 
                assert( m_cache[e.m_cache_index].m_block_addr = m_config.block_addr(e.m_request->get_addr()) );
                m_result_fifo.push( e.m_request );
                m_fragment_fifo.pop();
            }
        }
    }

    // place returning cache block into reorder buffer
    void fill( mem_fetch *mf, unsigned time )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); 
        assert( e != m_extra_mf_fields.end() );
        assert( e->second.m_valid );
        assert( !m_rob.empty() );
        mf->set_status(m_rob_status,time);

        unsigned rob_index = e->second.m_rob_index;
        rob_entry &r = m_rob.peek(rob_index);
        assert( !r.m_ready );
        r.m_ready = true;
        r.m_time = time;
        assert( r.m_block_addr == m_config.block_addr(mf->get_addr()) );
    }

    // are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const
    {
        return !m_result_fifo.empty();
    }

    // pop next ready access (includes both accesses that "HIT" and those that "MISS")
    mem_fetch *next_access() 
    { 
        return m_result_fifo.pop();
    }

    void display_state( FILE *fp ) const
    {
        fprintf(fp,"%s (texture cache) state:\n", m_name.c_str() );
        fprintf(fp,"fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(), m_fragment_fifo.capacity() );
        fprintf(fp,"reorder buffer entries = %u / %u\n", m_rob.size(), m_rob.capacity() );
        fprintf(fp,"request fifo entries   = %u / %u\n", m_request_fifo.size(), m_request_fifo.capacity() );
        if ( !m_rob.empty() )
            fprintf(fp,"reorder buffer contents:\n");
        for ( int n=m_rob.size()-1; n>=0; n-- ) {
            unsigned index = (m_rob.next_pop_index() + n)%m_rob.capacity();
            const rob_entry &r = m_rob.peek(index);
            fprintf(fp, "tex rob[%3d] : %s ", index, (r.m_ready?"ready  ":"pending") );
            if ( r.m_ready )
                fprintf(fp,"@%6u", r.m_time );
            else
                fprintf(fp,"       ");
            fprintf(fp,"[idx=%4u]",r.m_index);
            r.m_request->print(fp,false);
        }
        if ( !m_fragment_fifo.empty() ) {
            fprintf(fp,"fragment fifo (oldest) :");
            fragment_entry &f = m_fragment_fifo.peek();
            fprintf(fp,"%s:          ", f.m_miss?"miss":"hit ");
            f.m_request->print(fp,false);
        }
    }


    private:
    std::string m_name;
    const cache_config &m_config;

    struct fragment_entry {
        fragment_entry() {}
        fragment_entry( mem_fetch *mf, unsigned idx, bool m, unsigned d )
        {
            m_request=mf;
            m_cache_index=idx;
            m_miss=m;
            m_data_size=d;
        }
        mem_fetch *m_request;     // request information
        unsigned   m_cache_index; // where to look for data
        bool       m_miss;        // true if sent memory request
        unsigned   m_data_size;
    };

    struct rob_entry {
        rob_entry() { m_ready = false; m_time=0; m_request=NULL;}
        rob_entry( unsigned i, mem_fetch *mf, new_addr_type a ) 
        { 
            m_ready=false; 
            m_index=i;
            m_time=0;
            m_request=mf; 
            m_block_addr=a;
        }
        bool m_ready;
        unsigned m_time; // which cycle did this entry become ready?
        unsigned m_index; // where in cache should block be placed?
        mem_fetch *m_request;
        new_addr_type m_block_addr;
    };

    struct data_block {
        data_block() { m_valid = false;}
        bool m_valid;
        new_addr_type m_block_addr;
    };

    // TODO: replace fifo_pipeline with this?
    template<class T> class fifo {
    public:
        fifo( unsigned size ) 
        { 
            m_size=size; 
            m_num=0; 
            m_head=0; 
            m_tail=0; 
            m_data = new T[size];
        }
        bool full() const { return m_num == m_size;}
        bool empty() const { return m_num == 0;}
        unsigned size() const { return m_num;}
        unsigned capacity() const { return m_size;}
        unsigned push( const T &e ) 
        { 
            assert(!full()); 
            m_data[m_head] = e; 
            unsigned result = m_head;
            inc_head(); 
            return result;
        }
        T pop() 
        { 
            assert(!empty()); 
            T result = m_data[m_tail];
            inc_tail();
            return result;
        }
        const T &peek( unsigned index ) const 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek( unsigned index ) 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek() const
        { 
            return m_data[m_tail]; 
        }
        unsigned next_pop_index() const 
        {
            return m_tail;
        }
    private:
        void inc_head() { m_head = (m_head+1)%m_size; m_num++;}
        void inc_tail() { assert(m_num>0); m_tail = (m_tail+1)%m_size; m_num--;}

        unsigned   m_head; // next entry goes here
        unsigned   m_tail; // oldest entry found here
        unsigned   m_num;  // how many in fifo?
        unsigned   m_size; // maximum number of entries in fifo
        T         *m_data;
    };

    tag_array               m_tags;
    fifo<fragment_entry>    m_fragment_fifo;
    fifo<mem_fetch*>        m_request_fifo;
    fifo<rob_entry>         m_rob;
    data_block             *m_cache;
    fifo<mem_fetch*>        m_result_fifo; // next completed texture fetch

    mem_fetch_interface    *m_memport;
    enum mem_fetch_status   m_request_queue_status;
    enum mem_fetch_status   m_rob_status;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( unsigned i ) 
        {
            m_valid = true;
            m_rob_index = i;
        }
        bool m_valid;
        unsigned m_rob_index;
    };

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;
};

#endif
