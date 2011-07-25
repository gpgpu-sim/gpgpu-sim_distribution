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

#include "gpu-cache.h"
#include "stat-tool.h"
#include <assert.h>

tag_array::~tag_array() 
{
    delete m_lines;
}

tag_array::tag_array( const cache_config &config, int core_id, int type_id ) 
: m_config(config)
{
    assert( m_config.m_write_policy == READ_ONLY );
    m_lines = new cache_block_t[ config.get_num_lines()];

    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id; 
    m_type_id = type_id;
}

enum cache_request_status tag_array::probe( new_addr_type addr, unsigned &idx ) const {
    assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;

    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag) {
            if ( line->m_status == RESERVED ) {
                idx = index;
                return HIT_RESERVED;
            } else if ( line->m_status == VALID ) {
                idx = index;
                return HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                return HIT;
            } else {
                assert( line->m_status == INVALID );
            }
        }
        if (line->m_status != RESERVED) {
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 

    return MISS;
}

enum cache_request_status tag_array::access( new_addr_type addr, unsigned time, unsigned &idx )
{
    bool wb=false;
    cache_block_t evicted;
    enum cache_request_status result = access(addr,time,idx,wb,evicted);
    assert(!wb);
    return result;
}

enum cache_request_status tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ) 
{
    m_access++;
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    enum cache_request_status status = probe(addr,idx);
    switch (status) {
    case HIT_RESERVED: 
        m_pending_hit++;
    case HIT: 
        m_lines[idx].m_last_access_time=time; 
        break;
    case MISS:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS ) {
            if( m_lines[idx].m_status == MODIFIED ) {
                wb = true;
                evicted = m_lines[idx];
            }
            m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time );
        }
        break;
    case RESERVATION_FAIL:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    }
    return status;
}

void tag_array::fill( new_addr_type addr, unsigned time )
{
    assert( m_config.m_alloc_policy == ON_FILL );
    unsigned idx;
    enum cache_request_status status = probe(addr,idx);
    assert(status==MISS); // MSHR should have prevented redundant memory request
    m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time );
    m_lines[idx].fill(time);
}

void tag_array::fill( unsigned index, unsigned time ) 
{
    assert( m_config.m_alloc_policy == ON_MISS );
    m_lines[index].fill(time);
}

void tag_array::flush() 
{
    for (unsigned i=0; i < m_config.get_num_lines(); i++)
        m_lines[i].m_status = INVALID;
}

float tag_array::windowed_miss_rate( bool minus_pending_hit ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    if (minus_pending_hit)
        n_miss -= n_pending_hit;
    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float) n_miss / n_access;
    return missrate;
}

void tag_array::new_window()
{
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const
{
    m_config.print(stream);
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), -MgHts = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_miss - m_pending_hit, (float) (m_miss - m_pending_hit) / m_access);
    total_misses+=m_miss;
    total_access+=m_access;
}

bool was_write_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_read_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == READ_REQUEST_SENT ) 
            return true;
    }
    return false;
}
