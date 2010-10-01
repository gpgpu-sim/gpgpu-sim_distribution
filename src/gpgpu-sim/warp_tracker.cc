/* 
 * warp_tracker.cc
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

#include "warp_tracker.h"
#include "gpu-sim.h"
#include "shader.h"
#include <set>

using namespace std;

/*
 * Constructor for warp_tracker_pool.
 *
 * Resizes the warp_tracker map and pool and allocates empty warp_trackers.
 *
 * @param tid_in Array of thread id's corresponding to a warp
 * @param *my_shader Pointer to the shader core
 *
 */
warp_tracker_pool::warp_tracker_pool(class shader_core_ctx *my_shader) 
{
   m_shader=my_shader;
   const shader_core_config *config = my_shader->get_config();
	gpu_n_thread_per_shader = config->n_thread_per_shader;
   warp_size = config->warp_size;
	warp_tracker_map.resize(gpu_n_thread_per_shader);
}

/*
 * Register a new warp_tracker with a warp
 *
 * A warp_tracker is fetched from the pool of warp_trackers and assigned to
 * track the input warp. A (sid,tid,pc) to warp_tracker mapping is also stored.
 *
 * @param tid_in Array of thread id's corresponding to a single warp (array of size warp_size)
 * @param *shd Pointer to the shader core
 *
 */
void warp_tracker_pool::wpt_register_warp( int *tid_in, address_type pc, unsigned n_thd, unsigned warp_size )
{
   assert(n_thd); 
   warp_tracker *wpt = new warp_tracker(tid_in,pc,warp_size);
   // assign the new warp_tracker to warp_tracker_map
   for (unsigned i=0; i<warp_size; i++) {
      if (tid_in[i] >= 0) {
         assert( map_get_warp_tracker(tid_in[i],pc) == NULL );
         map_set_warp_tracker(tid_in[i], pc, wpt);
      }
   }
}

/*
 * Signal that the current thread has completed and ready to be unlocked.
 *
 * @param tid Thread that is exiting
 * @param *shd Pointer to the shader core
 *
 * @return Returns true is all threads in the warp have completed.
 */
int warp_tracker_pool::wpt_signal_avail( int tid, address_type pc )
{
   warp_tracker *wpt = map_get_warp_tracker(tid,pc);
   assert(wpt != NULL);


   // signal the warp tracker
   if (wpt->avail_thd()) {
      return 1;
   } else {
      return 0;
   }
}

/*
 * Unlock a warp
 *
 * Unlocks a warp for re-fetching. Sets avail4fetch = 1 for all threads and increments n_avail4fetch
 * by number of active threads.
 *
 * @param tid Thread that is exiting
 *
 */
void warp_tracker_pool::wpt_deregister_warp( int tid, address_type pc ) {
   warp_tracker *wpt = map_get_warp_tracker(tid,pc);
   assert(wpt != NULL);
   map_clear_warp_tracker(wpt);
   delete wpt;
}


/*
 * Signal that the a thread is done and is exiting (exit instruction)
 *
 * Marks a thread as completed. If all threads in the warp have completed, call register_cta_thread_exit on all
 * threads and removed the warp from warp tracker.
 *
 * @param tid Thread that is exiting
 *
 * @return The warp's mask of active threads.
 */
int warp_tracker_pool::wpt_signal_complete( int tid, address_type pc )
{
   warp_tracker *wpt = map_get_warp_tracker(tid,pc);
   assert(wpt != NULL);

   // signal the warp tracker
   if (wpt->complete_thd(tid)) {
      // if the warp has completed execution, remove this warp_tracker
      map_clear_warp_tracker(wpt);
      int warp_mask = 0;
      for (unsigned i=0; i<warp_size; i++) {
         if (wpt->tid(i) >= 0) {
            m_shader->register_cta_thread_exit( wpt->tid(i) );
            warp_mask |= (1 << i);
         }
      }
      delete wpt;
      return warp_mask;
   } else {
      return 0;
   }
}

/*
 * Check if this thread is being tracked by the warp tracker currently
 *
 * Checks if any pc of the given tid maps to a warp_tracker
 *
 * @param tid Thread to check
 *
 * @return True is thread is being tracked
 */
bool warp_tracker_pool::wpt_thread_in_wpt(int tid) {
	std::map<address_type, warp_tracker*>::iterator it;
	for(it=warp_tracker_map[tid].begin(); it!=warp_tracker_map[tid].end(); it++)
		if((*it).second != NULL)
			return true;

	return false;
}

//------------------------------------------------------------------------------------

map<unsigned, unsigned> thread_pc_tracker::histogram;

thread_pc_tracker *thread_pc_tracker = NULL;

void print_thread_pc_histogram( FILE *fout )
{
   thread_pc_tracker::histo_print(fout);
}

void print_thread_pc( FILE *fout, unsigned n_shader )
{
   fprintf(fout, "SHD_PC_C: ");
   for (unsigned i=0; i<n_shader; i++) {
      fprintf(fout, "%d ", thread_pc_tracker[i].get_acc_pc_count() );
   }
   fprintf(fout, "\n");
}

// uncomment to enable checking for warp consistency
// #define CHECK_WARP_CONSISTENCY

void shader_core_ctx::check_stage_pcs( unsigned stage )
{
#ifdef CHECK_WARP_CONSISTENCY
   address_type inst_pc = (address_type)-1;
   unsigned tid;
   if( m_config->model == MIMD ) 
      return;

   std::set<unsigned> tids;

   for ( int i = 0; i < m_config->warp_size; i++) {
      if (m_pipeline_reg[i][stage].hw_thread_id == -1 ) 
         continue;
      if ( inst_pc == (address_type)-1 ) 
         inst_pc = m_pipeline_reg[i][stage].pc;
      tid = m_pipeline_reg[i][stage].hw_thread_id;
      assert( tids.find(tid) == tids.end() );
      tids.insert(tid);
      assert( inst_pc == m_pipeline_reg[i][stage].pc );
   }
#endif
}

void shader_core_ctx::check_pm_stage_pcs( unsigned stage )
{
#ifdef CHECK_WARP_CONSISTENCY
   address_type inst_pc = (address_type)-1;
   unsigned tid;
   if( m_config->model == MIMD ) 
      return;

   std::set<unsigned> tids;

   for (int i = 0; i < m_config->warp_size; i++) {
      if (pre_mem_pipeline[i][stage].hw_thread_id == -1 ) 
         continue;
      if ( inst_pc == (address_type)-1 ) 
         inst_pc = pre_mem_pipeline[i][stage].pc;
      tid = pre_mem_pipeline[i][stage].hw_thread_id;
      assert( tids.find(tid) == tids.end() );
      tids.insert(tid);
      assert( inst_pc == pre_mem_pipeline[i][stage].pc );
   }
#endif
}
