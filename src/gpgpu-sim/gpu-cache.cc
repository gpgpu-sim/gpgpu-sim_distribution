// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
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


void l2_cache_config::init(linear_to_raw_address_translation *address_mapping){
	cache_config::init();
	m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const{
	if(!m_address_mapping){
		return(addr >> m_line_sz_log2) & (m_nset-1);
	}else{
		// Calculate set index without memory partition bits to reduce set camping
		new_addr_type part_addr = m_address_mapping->partition_address(addr);
		return(part_addr >> m_line_sz_log2) & (m_nset -1);
	}
}

tag_array::~tag_array() 
{
    delete[] m_lines;
}

tag_array::tag_array( const cache_config &config,
                      int core_id,
                      int type_id,
                      cache_block_t* new_lines)
    : m_config( config ),
      m_lines( new_lines )
{
    init( core_id, type_id );
}

tag_array::tag_array( const cache_config &config,
                      int core_id,
                      int type_id )
    : m_config( config )
{
    //assert( m_config.m_write_policy == READ_ONLY ); Old assert
    m_lines = new cache_block_t[ config.get_num_lines()];
    init( core_id, type_id );
}

void tag_array::init( int core_id, int type_id )
{
    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id; 
    m_type_id = type_id;
}

enum cache_request_status tag_array::probe( new_addr_type addr, unsigned &idx ) const {
    //assert( m_config.m_write_policy == READ_ONLY );
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

float tag_array::windowed_miss_rate( ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

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
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), PendingHit = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_pending_hit, (float) m_pending_hit / m_access);
    total_misses+=m_miss;
    total_access+=m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses) const{
	// Get the access and miss counts from the tag array
	total_misses = m_miss;
	total_access = m_access;
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
/****************************************************************** MSHR ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe( new_addr_type block_addr ) const{
    table::const_iterator a = m_data.find(block_addr);
    return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full( new_addr_type block_addr ) const{
    table::const_iterator i=m_data.find(block_addr);
    if ( i != m_data.end() )
        return i->second.m_list.size() >= m_max_merged;
    else
        return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add( new_addr_type block_addr, mem_fetch *mf ){
	m_data[block_addr].m_list.push_back(mf);
	assert( m_data.size() <= m_num_entries );
	assert( m_data[block_addr].m_list.size() <= m_max_merged );
	// indicate that this MSHR entry contains an atomic operation
	if ( mf->isatomic() ) {
		m_data[block_addr].m_has_atomic = true;
	}
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready( new_addr_type block_addr, bool &has_atomic ){
    assert( !busy() );
    table::iterator a = m_data.find(block_addr);
    assert( a != m_data.end() ); // don't remove same request twice
    m_current_response.push_back( block_addr );
    has_atomic = a->second.m_has_atomic;
    assert( m_current_response.size() <= m_data.size() );
}

/// Returns next ready access
mem_fetch *mshr_table::next_access(){
    assert( access_ready() );
    new_addr_type block_addr = m_current_response.front();
    assert( !m_data[block_addr].m_list.empty() );
    mem_fetch *result = m_data[block_addr].m_list.front();
    m_data[block_addr].m_list.pop_front();
    if ( m_data[block_addr].m_list.empty() ) {
        // release entry
        m_data.erase(block_addr);
        m_current_response.pop_front();
    }
    return result;
}

void mshr_table::display( FILE *fp ) const{
    fprintf(fp,"MSHR contents\n");
    for ( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
        unsigned block_addr = e->first;
        fprintf(fp,"MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr, e->second.m_has_atomic, e->second.m_list.size());
        if ( !e->second.m_list.empty() ) {
            mem_fetch *mf = e->second.m_list.front();
            fprintf(fp,"%p :",mf);
            mf->print(fp);
        } else {
            fprintf(fp," no memory requests???\n");
        }
    }
}
/***************************************************************** Caches *****************************************************************/
/// Sends next request to lower level of memory
void baseline_cache::cycle(){
    if ( !m_miss_queue.empty() ) {
        mem_fetch *mf = m_miss_queue.front();
        if ( !m_memport->full(mf->get_data_size(),mf->get_is_write()) ) {
            m_miss_queue.pop_front();
            m_memport->push(mf);
            n_simt_to_mem+=mf->get_num_flits(true); // Interconnect power stats
        }
    }
}

/// Interface for response from lower memory level (model bandwidth restictions in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time){
	extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
	assert( e != m_extra_mf_fields.end() );
	assert( e->second.m_valid );
	mf->set_data_size( e->second.m_data_size );
	if ( m_config.m_alloc_policy == ON_MISS )
		m_tag_array->fill(e->second.m_cache_index,time);
	else if ( m_config.m_alloc_policy == ON_FILL )
		m_tag_array->fill(e->second.m_block_addr,time);
	else abort();
	bool has_atomic = false;
	m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
	if (has_atomic) {
		assert(m_config.m_alloc_policy == ON_MISS);
		cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index);
		block.m_status = MODIFIED; // mark line as dirty for atomic operation
	}
	m_extra_mf_fields.erase(mf);
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill( mem_fetch *mf ){
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses, unsigned &misses) const{
    fprintf( fp, "Cache %s:\t", m_name.c_str() );
    m_tag_array->print(fp,accesses,misses);
}

void baseline_cache::display_state( FILE *fp ) const{
    fprintf(fp,"Cache %s:\n", m_name.c_str() );
    m_mshrs.display(fp);
    fprintf(fp,"\n");
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
		unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa){

	bool wb=false;
	cache_block_t e;
	send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e, events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
		unsigned time, bool &do_miss, bool &wb, cache_block_t &evicted, std::list<cache_event> &events, bool read_only, bool wa){

    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if ( mshr_hit && mshr_avail ) {
    	if(read_only)
    		m_tag_array->access(block_addr,time,cache_index);
    	else
    		m_tag_array->access(block_addr,time,cache_index,wb,evicted);

        m_mshrs.add(block_addr,mf);
        do_miss = true;
    } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
    	if(read_only)
    		m_tag_array->access(block_addr,time,cache_index);
    	else
    		m_tag_array->access(block_addr,time,cache_index,wb,evicted);

        m_mshrs.add(block_addr,mf);
        m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
        mf->set_data_size( m_config.get_line_sz() );
        m_miss_queue.push_back(mf);
        mf->set_status(m_miss_queue_status,time);
        if(!wa)
        	events.push_back(READ_REQUEST_SENT);
        do_miss = true;
    }
}


/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request, unsigned time, std::list<cache_event> &events){
    events.push_back(request);
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status,time);
}


/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(block_addr,time,cache_index); // update LRU state
	cache_block_t &block = m_tag_array->get_block(cache_index);
	block.m_status = MODIFIED;

	m_write_access++;
	return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(block_addr,time,cache_index); // update LRU state
	cache_block_t &block = m_tag_array->get_block(cache_index);
	block.m_status = MODIFIED;

	// generate a write-through
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	m_write_access++;
	return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	// generate a write-through/evict
	cache_block_t &block = m_tag_array->get_block(cache_index);
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	// Invalidate block
	block.m_status = INVALID;

	m_write_access++;
	return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	bool evict = (mf->get_access_type() == GLOBAL_ACC_W); // evict a line that hits on global memory write
	if(evict)
		return wr_hit_we(addr, cache_index, mf, time, events, status); // Write-evict
	else
		return wr_hit_wb(addr, cache_index, mf, time, events, status); // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory and send a read request for the same block
enum cache_request_status data_cache::wr_miss_wa(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status) {

	new_addr_type block_addr = m_config.block_addr(addr);

	// Write allocate, maximum 3 requests (write miss, read request, write back request)
	// Conservatively ensure the worst-case request can be handled this cycle
	bool mshr_hit = m_mshrs.probe(block_addr);
	bool mshr_avail = !m_mshrs.full(block_addr);
	if(miss_queue_full(2) || (!(mshr_hit && mshr_avail) && !(!mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size))))
		return RESERVATION_FAIL;

	send_write_request(mf, WRITE_REQUEST_SENT, time, events);
	// Tries to send write allocate request, returns true on success and false on failure
	//if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
	//	return RESERVATION_FAIL;

	const mem_access_t *ma = new  mem_access_t( L2_WR_ALLOC_R,
						mf->get_addr(),
						mf->get_data_size(),
						false, // Now performing a read
						mf->get_access_warp_mask(),
						mf->get_access_byte_mask() );

	mem_fetch *n_mf = new mem_fetch( *ma,
					NULL,
					mf->get_ctrl_size(),
					mf->get_wid(),
					mf->get_sid(),
					mf->get_tpc(),
					mf->get_mem_config());

	bool do_miss = false;
	bool wb = false;
	cache_block_t evicted;

	// Send read request resulting from write miss
	send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb, evicted, events, false, true);

	if( wb && (m_config.m_write_policy != WRITE_THROUGH) ) { // If evicted block is modified and not a write-through (already modified lower level)
		mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,L2_WRBK_ACC,m_config.get_line_sz(),true);
		m_miss_queue.push_back(wb);
		wb->set_status(m_miss_queue_status,time);
	}
	if( do_miss ){
		m_write_access++;
		m_write_miss++;
		return MISS;
	}

	return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	// on miss, generate write through (no write buffering -- too many threads for that)
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	m_write_access++;
	m_write_miss++;
	return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block. Special case for atomic instructions -> Mark block as modified
enum cache_request_status data_cache::rd_hit_base(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(block_addr,time,cache_index);
	if(mf->isatomic()){ // Atomics treated as global read/write requests - Perform read, mark line as MODIFIED
		assert(mf->get_access_type() == GLOBAL_ACC_R);
		cache_block_t &block = m_tag_array->get_block(cache_index);
        block.m_status = MODIFIED;  // mark line as dirty
	}

	m_read_access++;
	return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory, perform write-back as necessary
enum cache_request_status data_cache::rd_miss_base(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	if(miss_queue_full(1))
		return RESERVATION_FAIL; // cannot handle request this cycle (might need to generate two requests)

	new_addr_type block_addr = m_config.block_addr(addr);
	bool do_miss = false;
	bool wb = false;
	cache_block_t evicted;
	send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, evicted, events, false, false);

	if(wb && (m_config.m_write_policy != WRITE_THROUGH) ){ // If evicted block is modified and not a write-through (already modified lower level)
		mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr, L1_WRBK_ACC,m_config.get_line_sz(),true);
		send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
	}
	if( do_miss ){
		m_read_access++;
		m_read_miss++;
		return MISS;
	}
	return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) {
	assert( mf->get_data_size() <= m_config.get_line_sz());
	assert(m_config.m_write_policy == READ_ONLY);
	assert(!mf->get_is_write());
	new_addr_type block_addr = m_config.block_addr(addr);
	unsigned cache_index = (unsigned)-1;
	enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
	if ( status == HIT ) {
		m_tag_array->access(block_addr,time,cache_index); // update LRU state
		return HIT;
	}else if ( status != RESERVATION_FAIL ) {
		if(!miss_queue_full(0)){
			bool do_miss=false;
			send_read_request(addr, block_addr, cache_index, mf, time, do_miss, events, true, false);
			if(do_miss)
				return MISS;
		}
	}
	return RESERVATION_FAIL;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status l1_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ){

	assert( mf->get_data_size() <= m_config.get_line_sz());
	bool wr = mf->get_is_write();
	new_addr_type block_addr = m_config.block_addr(addr);
	unsigned cache_index = (unsigned)-1;
	enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);

	// Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the data_cache constructor to reflect the corresponding cache configuration options.
	// Function pointers were used to avoid many long conditional branches resulting from many cache configuration options.
	if(wr){	// Write
		if(status == HIT){
			return (this->*m_wr_hit)(addr, cache_index, mf, time, events, status);
		}else if ( status != RESERVATION_FAIL ) {
			return (this->*m_wr_miss)(addr, cache_index,  mf, time, events, status);
		}
	}else{ // Read
		if(status == HIT){
			return (this->*m_rd_hit)(addr, cache_index,  mf, time, events, status);
		}else if ( status != RESERVATION_FAIL ) {
			return (this->*m_rd_miss)(addr, cache_index,  mf, time, events, status);
		}
	}
	return RESERVATION_FAIL;
}

/// Models second level shared cache with global write-back and write-allocate policies
/// Currently the same as l1_cache, but separated to allow for different implementations
enum cache_request_status l2_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ){

	assert( mf->get_data_size() <= m_config.get_line_sz());
	bool wr = mf->get_is_write();
	new_addr_type block_addr = m_config.block_addr(addr);
	unsigned cache_index = (unsigned)-1;
	enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);

	// Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the data_cache constructor to reflect the corresponding cache configuration options.
	// Function pointers were used to avoid many long conditional branches resulting from many cache configuration options.
	if(wr){	// Write
		if(status == HIT){
			return (this->*m_wr_hit)(addr, cache_index,  mf, time, events, status);
		}else if ( status != RESERVATION_FAIL ) {
			return (this->*m_wr_miss)(addr, cache_index,  mf, time, events, status);
		}
	}else{ // Read
		if(status == HIT){
			return (this->*m_rd_hit)(addr, cache_index,  mf, time, events, status);
		}else if ( status != RESERVATION_FAIL ) {
			return (this->*m_rd_miss)(addr, cache_index,  mf, time, events, status);
		}
	}
	return RESERVATION_FAIL;
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) {
    if ( m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full() )
        return RESERVATION_FAIL;

    assert( mf->get_data_size() <= m_config.get_line_sz());

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

void tex_cache::cycle(){
    // send next request to lower level of memory
    if ( !m_request_fifo.empty() ) {
        mem_fetch *mf = m_request_fifo.peek();
        if ( !m_memport->full(mf->get_ctrl_size(),false) ) {
            m_request_fifo.pop();
            m_memport->push(mf);
            n_simt_to_mem+=mf->get_num_flits(true); // Interconnect power stats
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

/// Place returning cache block into reorder buffer
void tex_cache::fill( mem_fetch *mf, unsigned time )
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

void tex_cache::display_state( FILE *fp ) const
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
/******************************************************************************************************************************************/

