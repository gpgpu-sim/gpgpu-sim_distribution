/* 
 * gpu-cache.c
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

#include <stdio.h>
#include <stdlib.h>
#include "../abstract_hardware_model.h"

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#define VALID 0x01    // block is valid (and present in cache)
#define DIRTY 0x02    // block is dirty
#define RESERVED 0x04 // there is an outstanding request for this block, but it has not returned yet

enum cache_request_status {
    HIT,
    HIT_W_WT,       // Hit, but write through cache, still needs to send to memory 
    MISS_NO_WB,     // miss, but witeback not necessary
    MISS_W_WB,      // miss, must do writeback 
    WB_HIT_ON_MISS, // request hit on a reservation in wb cache
    RESERVATION_FAIL,
    NUM_CACHE_REQUEST_STATUS
};

struct cache_block_t {
   cache_block_t()
   {
      tag=0;
      addr=0;
      fetch_time=0;
      last_used=0;
      status=0;
   }
   new_addr_type tag;
   new_addr_type addr;
   unsigned fetch_time;
   unsigned last_used;
   unsigned char status; /* valid, dirty... etc */
};

enum replacement_policy {
    LRU,
    FIFO
};

enum write_policy {
    no_writes,    // line replacement when new line arrives
    write_back,   // line replacement when new line arrives
    write_through // reservation based, use much handle reservation full error.
};

enum allocation_policy {
    on_miss,
    on_fill
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
	int ntok = sscanf(m_config_string,"%d:%d:%d:%c:%c:%c", &nset, &line_sz, &assoc, &replacement_policy, &write_policy, &alloc_policy);
	if( ntok != 6 ) {
	    printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n", m_config_string );
	    abort();
	}
	m_valid = true;
    }
    unsigned get_line_sz() const
    {
	assert( m_valid );
	return line_sz;
    }
    unsigned get_num_lines() const
    {
	assert( m_valid );
	return nset * assoc;
    }

    enum write_policy get_write_policy() const
    {
	if( write_policy == 'R' ) 
	    return no_writes;
	else if( write_policy == 'B' ) 
	    return write_back;
	else if( write_policy == 'T' ) 
	    return write_through;
	else
	    abort();
    }

    char *m_config_string;

private:
    bool m_valid;
    unsigned int nset;
    unsigned int line_sz;
    unsigned int assoc;
    unsigned char replacement_policy; // 'L' = LRU, 'F' = FIFO, 'R' = RANDOM
    unsigned char write_policy;       // 'T' = write through, 'B' = write back, 'R' = read only
    unsigned char alloc_policy;       // 'm' = allocate on miss, 'f' = allocate on fill

    friend class cache_t;
};

class cache_t {
public:
    cache_t( const char *name, const cache_config &config, int core_id, int type_id ); 
   ~cache_t();

   enum cache_request_status access( new_addr_type addr, 
                                     bool write,
                                     unsigned int sim_cycle, 
                                     address_type *wb_address );

   new_addr_type fill( new_addr_type addr, unsigned int sim_cycle );

   unsigned flush();
   
   void     print( FILE *stream, unsigned &total_access, unsigned &total_misses );
   float    windowed_cache_miss_rate(int);
   void     new_window();

private:
   std::string m_name;
   const cache_config &m_config;

   cache_block_t *m_lines; /* nbanks x nset x assoc lines in total */
   unsigned m_n_banks;
   unsigned m_nset;
   unsigned m_nset_log2;
   unsigned m_assoc;
   unsigned m_line_sz; // bytes 
   unsigned m_line_sz_log2;

   enum write_policy m_write_policy; 
   unsigned char m_replacement_policy;

   unsigned m_access;
   unsigned m_miss;
   unsigned m_merge_hit; // number of cache miss that hit the same line (and merged as a result)

   // performance counters for calculating the amount of misses within a time window
   unsigned m_prev_snapshot_access;
   unsigned m_prev_snapshot_miss;
   unsigned m_prev_snapshot_merge_hit; 
   
   int m_core_id; // which shader core is using this
   int m_type_id; // what kind of cache is this (normal, texture, constant)
};

#endif
