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
#include <assert.h>
#include <string.h>

// both shd_cache_access and shd_cache_probe functions use 
// shd_cache_access_internal to access/probe cache
shd_cache_line_t* shd_cache_access_internal( shd_cache_t *cp, 
                                             unsigned long long int addr, 
                                             unsigned int nbytes, 
                                             unsigned char write,
                                             unsigned int sim_cycle,
                                             unsigned int real_access); 

shd_cache_t * shd_cache_create( char *name,
                                unsigned int nset,
                                unsigned int assoc,
                                unsigned int line_sz,
                                unsigned char policy,
                                unsigned int hit_latency,
                                unsigned long long int bank_mask,
                                enum cache_write_policy wp) {

   shd_cache_t *cp;
   unsigned int nlines;

   unsigned int i;

   if (!nset || !assoc) {
      printf("Creating non-existing cache!\n");
      return 0;
   }

   nlines = nset * assoc;
   cp = (shd_cache_t*) calloc(1, sizeof(shd_cache_t));

   cp->bank_mask = bank_mask;
   cp->name = (char*) malloc(sizeof(char) * (strlen(name) + 1));
   strcpy(cp->name, name);
   cp->nset = nset;
   cp->nset_log2 = LOGB2(nset);
   cp->assoc = assoc;
   cp->line_sz = line_sz;
   cp->line_sz_log2 = LOGB2(line_sz);
   cp->policy = policy;
   cp->hit_latency = hit_latency; 
   cp->lines = (shd_cache_line_t*) calloc(nlines, sizeof(shd_cache_line_t));
   cp->write_policy = wp;

   for (i=0; i<nlines; i++) {
      cp->lines[i].line_sz = line_sz;
      cp->lines[i].status = 0;
   }

   // don't hook up with any logger
   cp->core_id = -1; 
   cp->type_id = -1;

   // initialize snapshot counters for visualizer
   cp->prev_snapshot_access = 0;
   cp->prev_snapshot_miss = 0;
   cp->prev_snapshot_merge_hit = 0;

//   printf("%s: %d(%x) x %d x %d(%x) %c, %d\n",
//          cp->name, cp->nset, cp->nset_log2, cp->assoc, cp->line_sz,
//          cp->line_sz_log2, cp->policy, nlines);

   return cp;
}

void shd_cache_destroy( shd_cache_t* cp ) {

   free(cp->lines);
   free(cp);
}

extern void shader_cache_miss_log( int logger_id, int type );
// hook up with shader core logger
void shd_cache_bind_logger(shd_cache_t* cp, int core_id, int type_id) {
   cp->core_id = core_id; 
   cp->type_id = type_id;
}

extern unsigned long long int addrdec_packbits(unsigned long long int mask, 
                                               unsigned long long int val,
                                               unsigned char high, unsigned char low);
extern void shader_cache_access_log( int logger_id, int type, int miss);
extern void shader_cache_access_unlog( int logger_id, int type, int miss);

shd_cache_line_t* shd_cache_access_internal( shd_cache_t *cp, 
                                             unsigned long long int addr, 
                                             unsigned int nbytes, 
                                             unsigned char write,
                                             unsigned int sim_cycle,
                                             unsigned int real_access) {
   //if real_access==0 then its only a cache probe stats and LRU tags should not be updated 
   unsigned int i;
   unsigned int set;
   unsigned long long int tag; 
   unsigned long long int packed_addr;
   shd_cache_line_t *pline;

   if (cp->bank_mask)
      packed_addr = addrdec_packbits(cp->bank_mask, addr, 64, 0);
   else
      packed_addr = addr;

   set = (packed_addr >> cp->line_sz_log2) & ( (1<<cp->nset_log2) - 1 );
   tag = packed_addr >> (cp->line_sz_log2 + cp->nset_log2);

   if (real_access) {
      cp->access++;
      shader_cache_access_log(cp->core_id, cp->type_id, 0);
   }

   for (i=0; i<cp->assoc; i++) {
      pline = &(cp->lines[set*cp->assoc+i] );
      if (pline->status & VALID) {
         if (pline->tag == tag) {
            //printf("Cache Hit! Addr=%08x Set=%x Way=%x Tag=%x\n", packed_addr, set, i, tag);
            if (real_access) {
               pline->last_used = sim_cycle;
               if (write) {
                  pline->status |= DIRTY;
               }
            }
            return pline;
         }
      }
   }
   if (real_access) {
      cp->miss++;
      shader_cache_access_log(cp->core_id, cp->type_id, 1);
   }
   return 0;
}

shd_cache_line_t* shd_cache_access( shd_cache_t *cp, 
                                    unsigned long long int addr, 
                                    unsigned int nbytes, 
                                    unsigned char write,
                                    unsigned int sim_cycle ) 
{
   return shd_cache_access_internal(cp,addr,nbytes,write,sim_cycle,1/*this is a real access*/);
}

extern int gpgpu_cache_wt_through;
shd_cache_t *test = NULL;
enum cache_request_status shd_cache_access_wb( shd_cache_t *cp, 
                                    unsigned long long int addr, 
                                    unsigned int nbytes, 
                                    unsigned char write,
                                    unsigned int sim_cycle, address_type *wb_address) 
{
   unsigned int i;
   unsigned int set;
   unsigned long long int tag; 
   unsigned long long int packed_addr;
   shd_cache_line_t *pline;

   unsigned already_reserved = 0;
   unsigned all_reserved = 1;
   shd_cache_line_t *free_line = NULL;

   if (cp->bank_mask)
      packed_addr = addrdec_packbits(cp->bank_mask, addr, 64, 0);
   else
      packed_addr = addr;

   set = (packed_addr >> cp->line_sz_log2) & ( (1<<cp->nset_log2) - 1 );
   tag = packed_addr >> (cp->line_sz_log2 + cp->nset_log2);

   cp->access++;
   shader_cache_access_log(cp->core_id, cp->type_id, 0);

   for (i=0; i<cp->assoc; i++) {
      pline = &(cp->lines[set*cp->assoc+i] );
      if (pline->tag == tag) {
         if (pline->status & RESERVED) {
            already_reserved = 1;
            break;
         } else if (pline->status & VALID) {
            //printf("Cache Hit! Addr=%08x Set=%x Way=%x Tag=%x\n", packed_addr, set, i, tag);
            pline->last_used = sim_cycle;
            if (write) {
               pline->status |= DIRTY;
            }            
            //return pline;
            if (cp->write_policy == write_through) return HIT_W_WT;
            return HIT;
         }
      } 
      if (!(pline->status & RESERVED)) {
         all_reserved = 0;
         if (!(pline->status & VALID)) { 
             free_line = pline;
         }
      }
   }
   cp->miss++;
   shader_cache_access_log(cp->core_id, cp->type_id, 1);

   if (already_reserved || cp->write_policy != write_back || write) {
      //not in cache yet, but no wb as place is reserved.
      //or wt caches never nead to worry about it (as do no_write caches) 
      //or is a write
      if (already_reserved && write) {
         //write the data into the cache line, make it dirty
         //up to mshrs to save write mask to not overwrite this data when the read returns
         pline->status |= DIRTY;
         return WB_HIT_ON_MISS;
      } else if (already_reserved) {
         return WB_HIT_ON_MISS;
      }
      return MISS_NO_WB;
   }

   //if not in cache, and a write back cache, and not already allocated, need to allocate a place for this request

   if (all_reserved) {
      //cannot service this request, because we can't garantee that we have room for the line when it comes back
      return RESERVATION_FAIL;
   }

   //printf("RESRV %d\n",tag);

   if (free_line) {
      //reserve fo this request
      free_line->status |= RESERVED;
      free_line->tag = tag;
      //no writeback
      return MISS_NO_WB;
   }

   // need to kick a line out to reserve a spot
   shd_cache_line_t *rline = NULL;
           
   for (i=0; i<cp->assoc; i++) {
      pline = &(cp->lines[set*cp->assoc+i] );
      if (pline->status & VALID && !(pline->status & RESERVED)) {
         if (!rline) {
            rline = pline; //select first available for ejection for later comparison
            continue;
         }
         switch (cp->policy) {
         case LRU: 
            if (pline->last_used < rline->last_used)
               rline = pline;
            break;
         case FIFO:
            if (pline->fetch_time < rline->fetch_time)
               rline = pline;
            break;
         default:
            rline = pline; //pick one, ie. the last valied one.
         }   
      }
   }
   assert(rline); //ensure we actually found one.

   unsigned needs_wb = (rline->status & (DIRTY|VALID)) == (DIRTY|VALID);
   /* Set the replaced cache line address */
   if (needs_wb) {   
      *wb_address = rline->addr; 
   }
   
   /* reserve this new line */
   rline->status |= RESERVED;
   rline->status &= ~VALID;
   rline->status &= ~DIRTY;
   rline->tag = tag; 

   /* printf("Fetching! Addr=%08x ReplAddr=%08x(%d) Set=%x Tag=%x\n",
             packed_addr, repl_addr, nofreeslot, set, tag);
   */
   if (needs_wb) {
      return MISS_W_WB;
   } else {
      return MISS_NO_WB;
   }
}

shd_cache_line_t* shd_cache_probe( shd_cache_t *cp, 
                                   unsigned long long int addr)
{
   return shd_cache_access_internal(cp,addr,
                                    1,0,0, /*do not matter*/
                                    0/*this is just a probe*/);  
}

void shd_cache_undo_stats( shd_cache_t *cp, int miss )
{
   if (miss) {
      cp->miss--;
      shader_cache_access_unlog(cp->core_id, cp->type_id, 1);
   }
   cp->access--;
   shader_cache_access_unlog(cp->core_id, cp->type_id, 0);
}

// Obtain the windowed cache miss rate for visualizer
float shd_cache_windowed_cache_miss_rate( shd_cache_t *cp, int minus_merge_hit )
{
   unsigned int n_access = cp->access - cp->prev_snapshot_access;
   unsigned int n_miss = cp->miss - cp->prev_snapshot_miss;
   unsigned int n_merge_hit = cp->merge_hit - cp->prev_snapshot_merge_hit;
   
   if (minus_merge_hit) {
      n_miss -= n_merge_hit;
   }
   float missrate = 0.0f;
   if (n_access != 0) {
      missrate = (float) n_miss / n_access;
   }
   
   return missrate;
}

// start a new sampling window
void shd_cache_new_window( shd_cache_t *cp )
{
   cp->prev_snapshot_access = cp->access;
   cp->prev_snapshot_miss = cp->miss;
   cp->prev_snapshot_merge_hit = cp->merge_hit;
}

unsigned long long int L2_shd_cache_fill( shd_cache_t *cp, 
                                          unsigned long long int addr,
                                          unsigned int sim_cycle ) {
   unsigned long long int result = shd_cache_fill(cp, addr, sim_cycle);
   return result;
}

static unsigned int _n_line_existed = 0; // debug counter

// Fetch requested data into cache line. 
// Returning address on the replaced line if it is dirty, or -1 if it is clean
// Assume the line is filled all at once. 
unsigned long long int shd_cache_fill( shd_cache_t *cp, 
                                       unsigned long long int addr,
                                       unsigned int sim_cycle ) {

   unsigned int i;
   unsigned int set;
   unsigned long long int tag;
   unsigned long long int packed_addr;
   unsigned long long int repl_addr;

   unsigned char nofreeslot; 
   unsigned char line_exists;
   unsigned int base = 0 ; 
   unsigned int maxway = cp->assoc ; 

   shd_cache_line_t *pline, *cline;

   if (cp->bank_mask)
      packed_addr = addrdec_packbits(cp->bank_mask, addr, 64, 0);
   else
      packed_addr = addr;
   set = (packed_addr >> cp->line_sz_log2) & ( (1<<cp->nset_log2) - 1 );
   tag = packed_addr >> (cp->line_sz_log2 + cp->nset_log2);

   if (cp->write_policy == write_back) {
      //this request must have a reserved spot
      cline = NULL;
      for (i=base; i<maxway; i++) {
         pline = &(cp->lines[set*cp->assoc+i] );
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

      //if (cline) printf("FOUND %d\n",tag);
      //else printf("UNFOUND!!! %d\n", tag);

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
   nofreeslot = 1;
   line_exists = 0;
   for (i=base; i<maxway; i++) {
      pline = &(cp->lines[set*cp->assoc+i] );
      if (!(pline->status & VALID)) {
         cline = pline;
         nofreeslot = 0;
         break;
      } else if (pline->tag == tag) {
         cline = pline;
         line_exists = 1;
         break;
      }
   }

   if (line_exists) {
      _n_line_existed += 1;
      return -1; // don't need to spill any line, nor it needs to be filled
   }

   if (nofreeslot) {
      cline = &(cp->lines[set*cp->assoc+base] );
      for (i=1+base; i<maxway; i++) {
         pline = &(cp->lines[set*cp->assoc+i] );
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

/*    printf("Fetching! Addr=%08x ReplAddr=%08x(%d) Set=%x Tag=%x\n",
          packed_addr, repl_addr, nofreeslot, set, tag);
 */
   return repl_addr;
}

void shd_cache_mergehit( shd_cache_t *cp, unsigned long long int addr )
{
   cp->merge_hit += 1;
}

void shd_cache_print( shd_cache_t *cp,  FILE *stream) {
   fprintf( stream, "Cache %s:\t", cp->name);
   fprintf( stream, "Size = %d B (%d Set x %d-way x %d byte line)\n", 
            cp->line_sz * cp->nset * cp->assoc,
            cp->nset, cp->assoc, cp->line_sz );
   fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), -MgHts = %d (%.3g)\n", 
            cp->access, cp->miss, (float) cp->miss / cp->access, 
            cp->miss - cp->merge_hit, (float) (cp->miss - cp->merge_hit) / cp->access);
}

#ifdef UNIT_TEST

int main() {
   shd_cache_t *cp[3];
   unsigned int addr, i;
   unsigned int cachenum;
   unsigned int sim_cycle;

   unsigned int test_addrs[8] = { 0x100, 0x200, 0x300, 0x400, 
      0x104, 0x204, 0x500, 0x100}; 
   unsigned int repl_addr[8] = {0,0,0,0,0,0,0,0};
   unsigned int rdwr[8] = {0,1,0,0,0,0,0,0};

   sim_cycle = 0;
   cp[0] = shd_cache_create ("cp1", 16, 4, 16, LRU, 1);
   cp[1] = shd_cache_create ("cp2", 16, 4, 16, FIFO, 1);

   for (cachenum = 0; cachenum<2; cachenum++)
      for (i=0; i<8; i++) {
         if ( !shd_cache_access(cp[cachenum], test_addrs[i], 4, rdwr[i], sim_cycle) ) {
            repl_addr[i] = shd_cache_fill(cp[cachenum], test_addrs[i], sim_cycle);
            shd_cache_access(cp[cachenum], test_addrs[i], 4, rdwr[i], sim_cycle);
         }
         sim_cycle++;
      }

   printf("replaced address:");
   for (i=0; i<8; i++) {
      printf("0x%x ", repl_addr[i]);
   }
   printf("\n");
   shd_cache_print(cp[0],stdout);
   shd_cache_print(cp[1],stdout);

   shd_cache_fill(cp[0], 0x104b3ecb0, sim_cycle);
   printf("Accessing 64-bit address tag: %d\n",
          shd_cache_access(cp[0], 0x104b3ecb2, 4, 0, sim_cycle));
   printf("Accessing 64-bit address tag: %d\n",
          shd_cache_access(cp[0], 0x103433330, 4, 0, sim_cycle));


   shd_set_coherency_policy(2);
   cp[2] = shd_cache_create("cp2", 16, 4, 16, LRU, 1);
   shd_cache_fill(cp[2], 0x12345000, 0);
   shd_cache_access(cp[2], 0x12345000, 4, 1, 0);
   shd_cache_access(cp[2], 0x12345004, 4, 0, 0);
   shd_cache_access(cp[2], 0x12345008, 4, 0, 0);
   shd_cache_access(cp[2], 0x1234500C, 4, 1, 0);
   printf("Checking Dirty Vector %x, Result = %d (Expect %d)\n", 0xf,
          shd_cache_linedirty(cp[2], 0x12345000, 0xf), 1 );
   printf("Checking Dirty Vector %x, Result = %d (Expect %d)\n", 0x6,
          shd_cache_linedirty(cp[2], 0x12345000, 0x6), 0 );
   printf("Checking Dirty Vector %x, Result = %d (Expect %d)\n", 0x1,
          shd_cache_linedirty(cp[2], 0x12345000, 0x1), 1 );
   printf("Checking Dirty Vector %x, Result = %d (Expect %d)\n", 0x8,
          shd_cache_linedirty(cp[2], 0x12345000, 0x8), 1 );
   printf("Checking Dirty Vector %x, Result = %d (Expect %d)\n", 0x9,
          shd_cache_linedirty(cp[2], 0x12345000, 0x9), 1 );

}

#endif
