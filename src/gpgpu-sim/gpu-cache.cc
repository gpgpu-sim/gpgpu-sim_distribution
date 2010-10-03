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

#include "gpu-cache.h"
#include "gpu-misc.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "gpu-sim.h"
#include <assert.h>
#include <string.h>

cache_t::~cache_t() 
{
   delete m_lines;
}

cache_t::cache_t( const char *name, 
                  const char *opt, 
                  unsigned long long int bank_mask, 
                  enum cache_write_policy wp, 
                  int core_id, 
                  int type_id) 
{
   unsigned int nset;
   unsigned int line_sz;
   unsigned int assoc;
   unsigned char policy;
   int ntok = sscanf(opt,"%d:%d:%d:%c", &nset, &line_sz, &assoc, &policy);
   if( ntok != 4 ) {
      printf("GPGPU-Sim uArch: cache configuration string parsing error for cache %s\n", name);
      abort();
   }
   assert(nset && assoc);

   cache_t *cp = this;
   unsigned int nlines;
   unsigned int i;


   nlines = nset * assoc;
   cp->bank_mask = bank_mask;
   cp->m_name = (char*) malloc(sizeof(char) * (strlen(name) + 1));
   strcpy(cp->m_name, name);
   cp->m_nset = nset;
   cp->m_nset_log2 = LOGB2(nset);
   cp->m_assoc = assoc;
   cp->m_line_sz = line_sz;
   cp->line_sz_log2 = LOGB2(line_sz);
   cp->policy = policy;
   cp->m_lines = (cache_block_t*) calloc(nlines, sizeof(cache_block_t));
   cp->write_policy = wp;

   for (i=0; i<nlines; i++) {
      cp->m_lines[i].line_sz = line_sz;
      cp->m_lines[i].status = 0;
   }

   // don't hook up with any logger
   cp->core_id = -1; 
   cp->type_id = -1;

   // initialize snapshot counters for visualizer
   cp->prev_snapshot_access = 0;
   cp->prev_snapshot_miss = 0;
   cp->prev_snapshot_merge_hit = 0;
   cp->core_id = core_id; 
   cp->type_id = type_id;
}

enum cache_request_status cache_t::access( unsigned long long int addr, 
                                           unsigned char write, 
                                           unsigned int sim_cycle, 
                                           address_type *wb_address ) 
{
   cache_t *cp = this; 
   unsigned long long int bank_addr; // offset within bank
   bool all_reserved = true;
   cache_block_t *pending_line = NULL;
   cache_block_t *clean_line = NULL;

   if (cp->bank_mask)
      bank_addr = addrdec_packbits(cp->bank_mask, addr, 64, 0);
   else
      bank_addr = addr;

   unsigned set = (bank_addr >> cp->line_sz_log2) & ( (1<<cp->m_nset_log2) - 1 );
   unsigned long long tag = bank_addr >> (cp->line_sz_log2 + cp->m_nset_log2);

   cp->m_access++;
   shader_cache_access_log(cp->core_id, cp->type_id, 0);

   for (unsigned way=0; way<cp->m_assoc; way++) {
      cache_block_t *line = &(cp->m_lines[set*cp->m_assoc+way] );
      if (line->tag == tag) {
         if (line->status & RESERVED) {
            pending_line = line;
            break;
         } else if (line->status & VALID) {
            line->last_used = sim_cycle;
            if (write) 
               line->status |= DIRTY;
            if (cp->write_policy == write_through) 
               return HIT_W_WT;
            return HIT;
         }
      } 
      if (!(line->status & RESERVED)) {
         all_reserved = false;
         if (!(line->status & VALID)) 
             clean_line = line;
      }
   }
   cp->miss++;
   shader_cache_access_log(cp->core_id, cp->type_id, 1);

   if (pending_line || cp->write_policy != write_back || write) {
      if (pending_line) {
         if( write ) // write hit-under-miss (irrelevant whether write-back or write-through)
                     // - timing assumes a large enough write buffer in shader core that we never
                     //   encounter a structural hazard
                     // - write buffer merged with returning cache block in zero cycles
            pending_line->status |= DIRTY;
         return WB_HIT_ON_MISS;
      }
      return MISS_NO_WB;
   }

   // at this point: this must be a write back cache (and not a hit-under-miss)
   assert( cp->write_policy == write_back );

   if (all_reserved) 
      // cannot service this request, because we can't garantee that we have room for the line when it comes back
      return RESERVATION_FAIL;

   if (clean_line) {
      // found a clean line in the cache so, no need to do a writeback
      clean_line->status |= RESERVED;
      clean_line->tag = tag;
      return MISS_NO_WB; 
   }

   // no clean lines, need to kick a line out to reserve a spot
   cache_block_t *wb_line = NULL;
           
   for (unsigned way=0; way<cp->m_assoc; way++) {
      cache_block_t *line = &(cp->m_lines[set*cp->m_assoc+way] );
      if (line->status & VALID && !(line->status & RESERVED)) {
         if (!wb_line) {
            wb_line = line;
            continue;
         }
         switch (cp->policy) {
         case LRU: 
            if (line->last_used < wb_line->last_used)
               wb_line = line;
            break;
         case FIFO:
            if (line->fetch_time < wb_line->fetch_time)
               wb_line = line;
            break;
         default:
            abort(); 
         }   
      }
   }
   assert(wb_line); // should always find a line
   assert((wb_line->status & (DIRTY|VALID)) == (DIRTY|VALID)); // should be dirty (or we would have found a clean line earlier)
   
   // reserve line 
   wb_line->status = RESERVED;
   wb_line->tag = tag; 
   *wb_address = wb_line->addr; 

   return MISS_W_WB;
}

// Obtain the windowed cache miss rate for visualizer
float cache_t::shd_cache_windowed_cache_miss_rate( int minus_merge_hit )
{
   cache_t *cp = this;
   unsigned int n_access = cp->m_access - cp->prev_snapshot_access;
   unsigned int n_miss = cp->miss - cp->prev_snapshot_miss;
   unsigned int n_merge_hit = cp->merge_hit - cp->prev_snapshot_merge_hit;
   
   if (minus_merge_hit) 
      n_miss -= n_merge_hit;
   float missrate = 0.0f;
   if (n_access != 0) 
      missrate = (float) n_miss / n_access;
   
   return missrate;
}

// start a new sampling window
void cache_t::shd_cache_new_window()
{
   cache_t *cp = this;
   cp->prev_snapshot_access = cp->m_access;
   cp->prev_snapshot_miss = cp->miss;
   cp->prev_snapshot_merge_hit = cp->merge_hit;
}

static unsigned int _n_line_existed = 0; // debug counter

// Fetch requested data into cache line. 
// Returning address on the replaced line if it is dirty, or -1 if it is clean
// Assume the line is filled all at once. 
new_addr_type cache_t::shd_cache_fill( new_addr_type addr, unsigned int sim_cycle ) 
{
   cache_t *cp = this;
   unsigned int base = 0 ; 
   unsigned int maxway = cp->m_assoc ; 
   cache_block_t *pline, *cline;
   unsigned long long int packed_addr;
   if (cp->bank_mask)
      packed_addr = addrdec_packbits(cp->bank_mask, addr, 64, 0);
   else
      packed_addr = addr;
   unsigned set = (packed_addr >> cp->line_sz_log2) & ( (1<<cp->m_nset_log2) - 1 );
   unsigned long long tag = packed_addr >> (cp->line_sz_log2 + cp->m_nset_log2);

   if (cp->write_policy == write_back) {
      //this request must have a reserved spot
      cline = NULL;
      for (unsigned i=base; i<maxway; i++) {
         pline = &(cp->m_lines[set*cp->m_assoc+i] );
         if ((pline->tag == tag) && (pline->status & RESERVED)) { 
            cline = pline;
            break;
         }
         if ((pline->tag == tag) && (pline->status & VALID)) {
            //A second fill has returned to a line in the cache
            //discard it as line in cache may have been modified, or is the same
            _n_line_existed++;

            return -1;
         }
      }

      if (!cline) printf("----!!! about to abort - this probably happened because global memory msrh merging is not enabled with a writeback cache !!!----\n");

      assert(cline); //error if it doesn't have a reserved space
      
      /* Fetch data into block */
      cline->status &= ~RESERVED;
      cline->status |= VALID;
      //cline->status &= ~DIRTY; Don't clear dirty bit, as might be dirty from write.
      cline->tag = tag; 
      cline->addr = addr;
      cline->last_used = sim_cycle;
      cline->fetch_time = sim_cycle;

      // no wb, already handled.
      return -1;
   }  

   //behavior unchanged for write through cache... probably not all necessary.
   
   // Look for any free slots and the possibility that the line is in the cache already
   bool nofreeslot = true;
   bool line_exists = false;
   for (unsigned i=base; i<maxway; i++) {
      pline = &(cp->m_lines[set*cp->m_assoc+i] );
      if (!(pline->status & VALID)) {
         cline = pline;
         nofreeslot = false;
         break;
      } else if (pline->tag == tag) {
         cline = pline;
         line_exists = true;
         break;
      }
   }

   if (line_exists) {
      _n_line_existed += 1;
      return -1; // don't need to spill any line, nor it needs to be filled
   }

   if (nofreeslot) {
      cline = &(cp->m_lines[set*cp->m_assoc+base] );
      for (unsigned i=1+base; i<maxway; i++) {
         pline = &(cp->m_lines[set*cp->m_assoc+i] );
         if (pline->status & VALID) {
            switch (cp->policy) {
            case LRU: 
               if (pline->last_used < cline->last_used)
                  cline = pline;
               break;
            case FIFO:
               if (pline->fetch_time < cline->fetch_time)
                  cline = pline;
               break;
            default:
               break;
            }   
         }
      }
   }

   /* Set the replaced cache line address */
   unsigned long long int repl_addr;
   if ((cline->status & (DIRTY|VALID)) == (DIRTY|VALID)) {
      repl_addr = cline->addr; 
   } else {
      repl_addr = -1;
   }

   /* Fetch data into block */
   cline->status |= VALID;
   cline->status &= ~DIRTY;
   cline->tag = tag; 
   cline->addr = addr;
   cline->last_used = sim_cycle;
   cline->fetch_time = sim_cycle;

   return repl_addr;
}

unsigned int cache_t::flush() 
{
   cache_t *cp = this;
   int dirty_lines_flushed = 0 ;
   for (unsigned i = 0; i < cp->m_nset * cp->m_assoc ; i++) {
      if ( (cp->m_lines[i].status & (DIRTY|VALID)) == (DIRTY|VALID) ) {
         dirty_lines_flushed++;
      }
      cp->m_lines[i].status &= ~VALID;
      cp->m_lines[i].status &= ~DIRTY;
   }
   return dirty_lines_flushed;
}

void cache_t::shd_cache_print( FILE *stream, unsigned &total_access, unsigned &total_misses ) 
{
   cache_t *cp = this;
   fprintf( stream, "Cache %s:\t", cp->m_name);
   fprintf( stream, "Size = %d B (%d Set x %d-way x %d byte line)\n", 
            cp->m_line_sz * cp->m_nset * cp->m_assoc,
            cp->m_nset, cp->m_assoc, cp->m_line_sz );
   fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), -MgHts = %d (%.3g)\n", 
            cp->m_access, cp->miss, (float) cp->miss / cp->m_access, 
            cp->miss - cp->merge_hit, (float) (cp->miss - cp->merge_hit) / cp->m_access);
   total_misses+=cp->miss;
   total_access+=cp->m_access;
}
