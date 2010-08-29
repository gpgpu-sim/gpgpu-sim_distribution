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

#include <vector>
#include <map>
#include <list>

#ifndef warp_tracker_h_INCLUDED
#define warp_tracker_h_INCLUDED

#ifdef __cplusplus

   #include <cstdio>
   #include <cstdlib>
   #include <cstring>
   #include <cassert>
   #include <map>
   #include <list>
   #include <deque>
   #include <queue>

#endif

#include "../abstract_hardware_model.h"
#include "shader.h"

extern unsigned int warp_size;
extern unsigned int gpu_n_shader;
extern unsigned int gpu_n_thread_per_shader;

//void init_warp_tracker( );

//void wpt_register_warp( int *tid_in, shader_core_ctx_t *shd );

//int wpt_signal_avail( int tid, shader_core_ctx_t *shd );

//void wpt_unlock_threads( int tid, shader_core_ctx_t *shd );

//int wpt_signal_complete( int tid, shader_core_ctx_t *shd );

void print_thread_pc_histogram( FILE *fout );
void print_thread_pc( FILE *fout );
void track_thread_pc( int shader_id, int *tid, address_type pc );

int* alloc_commit_warp( );
void free_commit_warp( int *commit_warp );

class warp_tracker {
public:
   std::vector<int> tid;

   int n_thd; // total number of threads in this warp
   int n_notavail; // number of threads still not available
   shader_core_ctx_t *shd; // reference to shader core
   address_type pc;

   warp_tracker () {
	  tid.resize(warp_size,-1);

      n_thd = 0;
      n_notavail = 0;
      shd = NULL;
   }

   void print_info(){
	   printf("sid=%d ", shd->sid);

	   printf("tid=[");
	   for(unsigned i=0; i<warp_size; i++)
		   if(tid[i] > -1)
			   printf("%d ", tid[i]);
	   printf("]\n");
   }

   void copy_tid( int *tid ) {
	   std::copy(tid, tid+warp_size, this->tid.begin());
   }

   // set the warp to be consist of the given threads
   void set_warp ( int *tid, shader_core_ctx_t *shd, address_type pc) {
	  copy_tid(tid);

      this->n_thd = 0;
      this->n_notavail = 0;
      for (unsigned i=0; i<warp_size; i++) {
         if (this->tid[i] >= 0) {
            this->n_thd++;
         }
      }
      this->n_notavail = this->n_thd;
      this->shd = shd;
      this->pc = pc;
   }

   // signal that this thread is available for fetch
   // if all threads in the warp are available, change all their status
   // and return true
   bool avail_thd ( ) {
      n_notavail--;
      return (n_notavail==0);
   }

   // a bookkeeping method to allow a warp to be deallocated
   // when its threads have finished executing.
   bool complete_thd ( int tid_in ) {
      n_notavail--;
      if (n_notavail) {
         return false;
      } else {
         return true;
      }
   }
};

//-------------------------------------------------------------------

class warp_tracker_pool
{
	private:
		unsigned gpu_n_shader;
		unsigned gpu_n_thread_per_shader;

		// Warp tracker map: a vector (index shader id) of vectors (index thread id) of maps (index pc)
		std::vector< std::vector< std::map<address_type, warp_tracker*> > > warp_tracker_map;

		// Pool of warp trackers
		std::list<warp_tracker> warp_tracker_list;

		// List to keep track of free warp trackers
		std::list<warp_tracker*> warp_tracker_free_list;

		warp_tracker* map_get_warp_tracker(int sid, int tid, address_type pc) {
			// Return NULL pointer if no warp_tracker assigned
			if(warp_tracker_map[sid][tid].find(pc) == warp_tracker_map[sid][tid].end())
				return NULL;

			return warp_tracker_map[sid][tid][pc];
		}

		void map_set_warp_tracker(int sid, int tid, address_type pc, warp_tracker* wpt) {
			// Make sure that warp tracker is not already assigned here
			if(warp_tracker_map[sid][tid].find(pc) != warp_tracker_map[sid][tid].end())
				assert(warp_tracker_map[sid][tid][pc] == NULL);

			warp_tracker_map[sid][tid][pc] = wpt;
		}

		void map_clear_warp_tracker(int sid, int tid, address_type pc) {
			// Make sure that warp tracker was previously assigned
			assert(warp_tracker_map[sid][tid].find(pc) != warp_tracker_map[sid][tid].end());
			assert(warp_tracker_map[sid][tid][pc] != NULL);

			warp_tracker_map[sid][tid][pc] = NULL;
		}

		warp_tracker* alloc_warp_tracker( int *tid_in, shader_core_ctx_t *shd, address_type pc );
		void free_warp_tracker(warp_tracker* wpt);

	public:
		warp_tracker_pool( unsigned gpu_n_shader, unsigned gpu_n_thread_per_shader  );

		void wpt_register_warp( int *tid_in, shader_core_ctx_t *shd, address_type pc );
		int wpt_signal_avail( int tid, shader_core_ctx_t *shd, address_type pc );
		void wpt_deregister_warp( int tid, shader_core_ctx_t *shd, address_type pc );
		int wpt_signal_complete( int tid, shader_core_ctx_t *shd, address_type pc );

		unsigned size() { return warp_tracker_list.size(); }
		unsigned free_size() { return warp_tracker_free_list.size(); }

		bool wpt_thread_in_wpt(shader_core_ctx *shd, int tid);

};

warp_tracker_pool& get_warp_tracker_pool();

#endif
