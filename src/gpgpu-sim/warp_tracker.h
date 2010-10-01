/* 
 * waro_tracker.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda and the 
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

#ifndef warp_tracker_h_INCLUDED
#define warp_tracker_h_INCLUDED

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <map>
#include <list>
#include <deque>
#include <queue>
#include <vector>
#include <map>
#include <list>

#include "../abstract_hardware_model.h"
#include "shader.h"

void print_thread_pc_histogram( FILE *fout );
void print_thread_pc( FILE *fout, unsigned n_shader );

class warp_tracker {
public:
   warp_tracker( int *tid, address_type pc, unsigned warp_size ) 
   {
      // makes a copy of pc and thread ids in warp 
      m_pc = pc;
      m_n_thd = 0;
      m_warp_size=warp_size;
      m_tid.reserve(warp_size);
      std::copy(tid, tid+warp_size, m_tid.begin());
      for (unsigned i=0; i<warp_size; i++) 
         if (tid[i] >= 0) m_n_thd++;
      m_n_notavail = m_n_thd;
   }

   void print_info( unsigned sid ) const
   {
	   printf("sid=%d tid=[", sid);
	   for(unsigned i=0; i<m_tid.size(); i++)
		   if(m_tid[i] > -1) printf("%d ", m_tid[i]);
	   printf("]\n");
   }
   unsigned warp_size() const { return m_warp_size; }
   address_type pc() const { return m_pc; }
   int tid(unsigned i) const { return m_tid[i]; }

   bool avail_thd() 
   {
      // signal that this thread is available for fetch
      // if all threads in the warp are available, change all their status
      // and return true
      assert( m_n_notavail > 0 );
      m_n_notavail--;
      return (m_n_notavail==0);
   }

   bool complete_thd( int tid_in ) 
   {
      // a bookkeeping method to allow a warp to be deallocated
      // when its threads have finished executing.
      assert( m_n_notavail > 0 );
      m_n_notavail--;
      if (m_n_notavail) {
         return false;
      } else {
         return true;
      }
   }

private:
   address_type     m_pc;
   std::vector<int> m_tid;
   int              m_n_thd;      // total number of threads in this warp with active mask set to enabled
   int              m_n_notavail; // number of threads waiting (preventing this warp from being issued again)
   unsigned m_warp_size;
};


class warp_tracker_pool {
public:
   warp_tracker_pool( class shader_core_ctx *my_shader );
   
   void wpt_register_warp( int *tid_in, address_type pc, unsigned n_thread_in_warp, unsigned warp_size );
   int  wpt_signal_avail( int tid, address_type pc );
   void wpt_deregister_warp( int tid, address_type pc );
   int  wpt_signal_complete( int tid, address_type pc );
   bool wpt_thread_in_wpt( int tid );

private:
   warp_tracker* map_get_warp_tracker(int tid, address_type pc) {
      // Returns NULL pointer if no warp_tracker assigned
      if(warp_tracker_map[tid].find(pc) == warp_tracker_map[tid].end())
         return NULL;
      return warp_tracker_map[tid][pc];
   }

   void map_set_warp_tracker(int tid, address_type pc, warp_tracker* wpt) {
      // Make sure that warp tracker is not already assigned here
      if(warp_tracker_map[tid].find(pc) != warp_tracker_map[tid].end())
         assert(warp_tracker_map[tid][pc] == NULL);
      warp_tracker_map[tid][pc] = wpt;
   }

   void map_clear_warp_tracker( warp_tracker *wpt ) {
      // Make sure that warp tracker was previously assigned
      address_type pc = wpt->pc();
      for (unsigned i=0; i<wpt->warp_size(); i++) {
         int tid = wpt->tid(i);
         if (tid >= 0) {
            assert(warp_tracker_map[tid].find(pc) != warp_tracker_map[tid].end());
            assert(warp_tracker_map[tid][pc] != NULL);
            warp_tracker_map[tid][pc] = NULL;
         }
      }
   }

// data

   class shader_core_ctx *m_shader;
   unsigned gpu_n_thread_per_shader;
   unsigned warp_size;

   // Warp tracker map: vector (index thread id) of maps (index pc)
   std::vector< std::map<address_type, warp_tracker*> > warp_tracker_map;
};

class thread_pc_tracker {
public:
   address_type *thd_pc; // tracks the pc of each thread
   std::map<address_type, unsigned> pc_count;
   unsigned acc_pc_count;
   int simd_width;
   static std::map<unsigned, unsigned> histogram; // static so automatically aggregated across cores

   thread_pc_tracker( ) {
      this->acc_pc_count = 0;
      this->simd_width = 0;
      this->thd_pc = NULL;
   }

   thread_pc_tracker(int simd_width, int thread_count) {
      this->acc_pc_count = 0;
      this->simd_width = simd_width;
      this->thd_pc = new address_type[thread_count];
      memset( this->thd_pc, 0, sizeof(address_type)*thread_count);
   }

   void add_threads( int *tid, address_type pc ) {
      for (int i=0; i<simd_width; i++) {
         if (tid[i] != -1) {
            pc_count[pc] += 1; // automatically create a new entry if not exist
            thd_pc[tid[i]] = pc;
         }
      }
   }

   void sub_threads( int *tid ) {
      for (int i=0; i<simd_width; i++) {
         if (tid[i] != -1) {
            address_type pc = thd_pc[tid[i]];
            if (pc == 0) break;
            pc_count[pc] -= 1;
            assert((int)pc_count[pc] >= 0);
            if (pc_count[pc] == 0) pc_count.erase(pc); // manually erasing entries with 0 count
         }
      }
   }

   void update_acc_count( ) { 
      acc_pc_count += pc_count.size(); 
      histogram[pc_count.size()] += 1;
   }

   void set_threads_pc ( int *tid, address_type pc ) {
      sub_threads(tid);
      add_threads(tid, pc);
      update_acc_count( );
   }

   unsigned get_acc_pc_count( ) { return acc_pc_count;}

   unsigned count( ) { return pc_count.size();}

   static void histo_print( FILE* fout ) {
      if (histogram.empty()) return; // do not output anything if the histogram is empty
      std::map<unsigned, unsigned>::iterator i;
      fprintf(fout, "Thread PC Histogram: ");
      for (i = histogram.begin(); i != histogram.end(); i++) {
         fprintf(fout, "%d:%d ", i->first, i->second);
      }
      fprintf(fout, "\n");
   }
};

#endif
