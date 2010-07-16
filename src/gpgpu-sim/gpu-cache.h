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
#include "../util.h"

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#define VALID 0x01
#define DIRTY 0x02
#define RESERVED 0x04

enum cache_request_status {
    HIT,
    HIT_W_WT, /* Hit, but write through cache, still needs to send to memory */
    MISS_NO_WB, /* miss, but witeback not necessary*/
    MISS_W_WB, /* miss, must do writeback */
    WB_HIT_ON_MISS, /* request hit on a reservation in wb cache*/
    RESERVATION_FAIL,
    NUM_CACHE_REQUEST_STATUS
};


typedef struct {
   unsigned long long int tag;
   unsigned long long int addr;
   unsigned int set;
   unsigned int line_sz; /* bytes */
   unsigned int fetch_time;
   unsigned int last_used;
   unsigned char status; /* valid, dirty... etc */
} shd_cache_line_t;


#define LRU 'L'
#define FIFO 'F'
#define RANDOM 'R'

enum cache_write_policy{
    no_writes, //line replacement when new line arrives
    write_back, //line replacement when new line arrives
    write_through //reservation based, use much handle reservation full error.
};

typedef struct {

   char *name;

   shd_cache_line_t *lines; /* nset x assoc lines in total */
   unsigned int nset;
   unsigned int nset_log2;
   unsigned int assoc;
   unsigned int line_sz; // bytes 
   unsigned int line_sz_log2;
   enum cache_write_policy write_policy; 
   unsigned char policy;
   unsigned int hit_latency;

   unsigned int access;
   unsigned int miss;
   unsigned int merge_hit; // number of cache miss that hit the same line (and merged as a result)

   // performance counters for calculating the amount of misses within a time window
   unsigned int prev_snapshot_access;
   unsigned int prev_snapshot_miss;
   unsigned int prev_snapshot_merge_hit; 
   
   int core_id; // which shader core is using this
   int type_id; // what kind of cache is this (normal, texture, constant)

   unsigned long long int bank_mask;

} shd_cache_t;

shd_cache_t * shd_cache_create( char *name,
                                unsigned int nset,
                                unsigned int assoc,
                                unsigned int line_sz,
                                unsigned char policy,
                                unsigned int hit_latency,
                                unsigned long long int bank_mask,
                                enum cache_write_policy wp);

void shd_cache_destroy( shd_cache_t* cp );

// hook up with shader core logger
void shd_cache_bind_logger(shd_cache_t* cp, int core_id, int type_id);

//depercated, use _wb
shd_cache_line_t* shd_cache_access( shd_cache_t *cp, 
                                    unsigned long long int addr, 
                                    unsigned int nbytes, 
                                    unsigned char write,
                                    unsigned int sim_cycle );

//cache check checks for wb and forwards information over.
enum cache_request_status shd_cache_access_wb( shd_cache_t *cp, 
                                    unsigned long long int addr, 
                                    unsigned int nbytes, 
                                    unsigned char write,
                                    unsigned int sim_cycle, 
                                    address_type *wb_address);


//just probe the tag array to see if addr is in the cache or not
//does not update LRU or stats...
shd_cache_line_t* shd_cache_probe( shd_cache_t *cp, 
                                   unsigned long long int addr);

// undo the statistic record when the memory access is stalled/squashed and will try again next cycle
void shd_cache_undo_stats( shd_cache_t *cp, int miss );

void shd_cache_mergehit( shd_cache_t *cp, unsigned long long int addr );

unsigned long long int shd_cache_fill( shd_cache_t *cp, 
                                       unsigned long long int addr,
                                       unsigned int sim_cycle );

unsigned long long int L2_shd_cache_fill( shd_cache_t *cp, 
                                          unsigned long long int addr,
                                          unsigned int sim_cycle );

void shd_cache_print( shd_cache_t *cp,  FILE *stream);


#endif
