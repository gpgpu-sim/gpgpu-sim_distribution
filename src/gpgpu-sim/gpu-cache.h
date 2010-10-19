/* 
 * gpu-cache.h
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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"

class mem_fetch; // mem_fetch opaque to cache and mshrs

enum cache_block_state {
    INVALID,
    RESERVED,
    VALID
};

enum cache_request_status {
    HIT,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL
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

   new_addr_type 	m_tag;
   new_addr_type 	m_block_addr;
   unsigned 	 	m_alloc_time;
   unsigned 	 	m_last_access_time;
   unsigned 		m_fill_time;
   cache_block_state 	m_status;
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
	m_config_string = NULL;	// set by option parser
    }
    void init()
    {
	assert( m_config_string );
	char rp, wp, ap, mshr_type;
	int ntok = sscanf(m_config_string,"%u:%u:%u:%c:%c:%c,%c:%u:%u,%u", 
			  &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap,
			  &mshr_type,&m_mshr_entries,&m_mshr_max_merge,&m_miss_queue_size);
	if( ntok < 10 ) 
	    exit_parse_error();
	switch(rp) {
	case 'L': m_replacement_policy = LRU; break;
	case 'F': m_replacement_policy = FIFO; break;
	default: exit_parse_error();
	}
	switch(wp) {
	case 'R': m_write_policy = READ_ONLY; break;
	case 'B': m_write_policy = WRITE_BACK; break;
	case 'T': m_write_policy = WRITE_THROUGH; break;
	default: exit_parse_error();
	}
	switch(ap) {
	case 'm': m_alloc_policy = ON_MISS; break;
	case 'f': m_alloc_policy = ON_FILL; break;
	default: exit_parse_error();
	}
	switch(mshr_type) {
	case 'F': m_mshr_type = TEX_FIFO; break;
	case 'A': m_mshr_type = ASSOC; break;
	default: exit_parse_error();
	}
	m_line_sz_log2 = LOGB2(m_line_sz);
	m_nset_log2 = LOGB2(m_nset);
	m_valid = true;
    }
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
	return (addr >> m_line_sz_log2) & (m_nset-1);
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
    unsigned m_line_sz;
    unsigned m_line_sz_log2;
    unsigned m_nset;
    unsigned m_nset_log2;
    unsigned m_assoc;

    enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
    enum write_policy_t m_write_policy;             // 'T' = write through, 'B' = write back, 'R' = read only
    enum allocation_policy_t m_alloc_policy;        // 'm' = allocate on miss, 'f' = allocate on fill
    enum mshr_config_t m_mshr_type;

    unsigned m_mshr_entries;
    unsigned m_mshr_max_merge;
    unsigned m_miss_queue_size;

    friend class tag_array;
    friend class cache_t;
};

class tag_array {
public:
    tag_array( const cache_config &config, int core_id, int type_id ); 
   ~tag_array();

    enum cache_request_status probe( new_addr_type addr, unsigned &idx ) const;
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx );

    void fill( new_addr_type addr, unsigned time );
    void fill( unsigned idx, unsigned time );

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
	if( i != m_data.end() ) 
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
	if( m_data[block_addr].empty() ) {
	    // release entry
	    m_data.erase(block_addr); 
	    m_current_response.pop_front();
	}
	return result;
    }

    void display( FILE *fp ) const
    {
	fprintf(fp,"MSHR contents\n");
	for( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
	    unsigned block_addr = e->first;
	    fprintf(fp,"MSHR: tag=0x%06x, %zu entries : ", block_addr, e->second.size());
	    if( !e->second.empty() ) {
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
    cache_t( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport ) 
    	: m_config(config), m_tag_array(config,core_id,type_id), m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge)
    {
	m_name = name;
	assert(config.m_mshr_type == ASSOC);
	assert(config.m_write_policy == READ_ONLY);
	m_memport=memport;
    }

    // access cache: returns RESERVATION_FAIL if request could not be accepted (for any reason)
    enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time )
    {
	new_addr_type block_addr = m_config.block_addr(addr);
	unsigned cache_index = (unsigned)-1;
	enum cache_request_status status = m_tag_array.probe(block_addr,cache_index);
	if( status == HIT ) {
	    m_tag_array.access(block_addr,time,cache_index); // update LRU state 
	    return HIT;
	}
	if( status != RESERVATION_FAIL ) {
	    bool mshr_hit = m_mshrs.probe(block_addr);
	    bool mshr_avail = !m_mshrs.full(block_addr);
	    if( mshr_hit && mshr_avail ) {
		m_tag_array.access(addr,time,cache_index);
		m_mshrs.add(block_addr,mf);
		return MISS;
	    } else if( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
		m_tag_array.access(addr,time,cache_index);
		m_mshrs.add(block_addr,mf);
		m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index);
		m_miss_queue.push_back(mf);
		return MISS;
	    }
	}
	return RESERVATION_FAIL;
    }

    void cycle() 
    {
	// send next request to lower level of memory
	if( !m_miss_queue.empty() ) {
	    mem_fetch *mf = m_miss_queue.front();
	    if( !m_memport->full(mf->get_data_size(),mf->get_is_write()) ) {
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
	if ( m_config.m_alloc_policy == ON_MISS )
	    m_tag_array.fill(e->second.m_cache_index,time);
	else if ( m_config.m_alloc_policy == ON_FILL ) 
	    m_tag_array.fill(e->second.m_block_addr,time);
	else abort();
	m_mshrs.mark_ready(e->second.m_block_addr);
	m_extra_mf_fields.erase(mf);
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

private:
    std::string m_name;
    const cache_config &m_config;
    tag_array  m_tag_array;
    mshr_table m_mshrs;
    std::list<mem_fetch*> m_miss_queue;
    mem_fetch_interface *m_memport;

    struct extra_mf_fields {
	extra_mf_fields()  { m_valid = false; }
	extra_mf_fields( new_addr_type a, unsigned i ) 
	{
	    m_block_addr = a;
	    m_cache_index = i;
	    m_valid = true;
	}
	bool m_valid;
	new_addr_type m_block_addr;
	unsigned m_cache_index;
    };
    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;
    extra_mf_fields_lookup m_extra_mf_fields;
};


#endif
