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


void register_cta_thread_exit(shader_core_ctx_t *shader, int cta_num );

/*
 * Constructor for warp_tracker_pool.
 *
 * Resizes the warp_tracker map and pool and allocates empty warp_trackers.
 *
 * @param tid_in Array of thread id's corresponding to a warp
 * @param *shd Pointer to the shader core
 *
 * @return Pointer to a warp_tracker
 */
warp_tracker_pool::warp_tracker_pool(unsigned gpu_n_shader, unsigned gpu_n_thread_per_shader) {
	this->gpu_n_shader = gpu_n_shader;
	this->gpu_n_thread_per_shader = gpu_n_thread_per_shader;

	// Resize the warp tracker map
	warp_tracker_map.resize(gpu_n_shader);
	for(unsigned i=0; i<gpu_n_shader; i++) {
		warp_tracker_map[i].resize(gpu_n_thread_per_shader);
	}

	// Create a pool of warp_trackers
	warp_tracker_list.resize(gpu_n_shader * gpu_n_thread_per_shader);

	// Add all warp_trackers to the list of free warp_trackers
	std::list<warp_tracker>::iterator it;
	for(it=warp_tracker_list.begin(); it!=warp_tracker_list.end(); it++)
		warp_tracker_free_list.push_back(&(*it));

}

/*
 * Fetch a free warp_tracker from the pool of warp_trackers. Assigns the warp_tracker to
 * the input warp.
 *
 * If there are no free warp_trackers in the pool, the pool is extended by allocating more
 * warp_trackers.
 *
 * @param tid_in Array of thread id's corresponding to a warp
 * @param *shd Pointer to the shader core
 *
 * @return Pointer to a warp_tracker
 */
warp_tracker* warp_tracker_pool::alloc_warp_tracker( int *tid_in, shader_core_ctx_t *shd, address_type pc ) {
	// If no free warp trackers are available, allocate some more
	if(warp_tracker_free_list.empty()) {
		printf("warp_tracker_list empty (size=%d) - allocating new warp_trackers\n", size());
		fflush(stdout);
		// Warp tracker list is empty, resize the list
		unsigned previous_size = warp_tracker_list.size();
		warp_tracker_list.resize( previous_size + this->gpu_n_thread_per_shader);

		// Add newly allocated warp trackers to list of free warp trackers
		std::list<warp_tracker>::iterator it = warp_tracker_list.begin();
		for(unsigned i=0; i<previous_size; i++)
			it++; // Increment iterator
		for(; it!=warp_tracker_list.end(); it++)
				warp_tracker_free_list.push_back(&(*it));
	}

	assert(!warp_tracker_free_list.empty());
	// Fetch a free warp_tracker
	warp_tracker* wpt = warp_tracker_free_list.front();
	warp_tracker_free_list.pop_front();
	wpt->set_warp(tid_in, shd, pc);

	return wpt;
}

/*
 * Free the warp_tracker.
 *
 * Puts the warp_tracker back into the pool of free warp_trackers.
 *
 * @param wpt Pointer to a warp_tracker
 *
 */
void warp_tracker_pool::free_warp_tracker(warp_tracker* wpt) {
	warp_tracker_free_list.push_back(wpt);
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
void warp_tracker_pool::wpt_register_warp( int *tid_in, shader_core_ctx_t *shd, address_type pc)
{
   int sid = shd->sid;
   unsigned i;
   int n_thd = 0;
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] >= 0) n_thd++;
   }

   if (!n_thd) return;

   warp_tracker *wpt = this->alloc_warp_tracker(tid_in, shd, pc);

   // assign the new warp_tracker to warp_tracker_map
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] >= 0) {
         assert( map_get_warp_tracker(sid,tid_in[i],pc) == NULL );
         map_set_warp_tracker(sid, tid_in[i], pc, wpt);
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
int warp_tracker_pool::wpt_signal_avail( int tid, shader_core_ctx_t *shd, address_type pc )
{
   int sid = shd->sid;
   warp_tracker *wpt = map_get_warp_tracker(sid,tid,pc);
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
 * @param *shd Pointer to the shader core
 *
 */
void warp_tracker_pool::wpt_deregister_warp( int tid, shader_core_ctx_t *shd, address_type pc ) {
   int sid = shd->sid;
   warp_tracker *wpt = map_get_warp_tracker(sid,tid,pc);
   assert(wpt != NULL);

    // the warp is ready to be fetched again, remove this warp_tracker
    for (unsigned i=0; i<warp_size; i++) {
	   if (wpt->tid[i] >= 0) {
		  map_clear_warp_tracker(sid,wpt->tid[i],pc);
	   }
    }

    free_warp_tracker( wpt );
}


/*
 * Signal that the a thread is done and is exiting (exit instruction)
 *
 * Marks a thread as completed. If all threads in the warp have completed, call register_cta_thread_exit on all
 * threads and removed the warp from warp tracker.
 *
 * @param tid Thread that is exiting
 * @param *shd Pointer to the shader core
 *
 * @return The warp's mask of active threads.
 */
int warp_tracker_pool::wpt_signal_complete( int tid, shader_core_ctx_t *shd, address_type pc )
{
   int sid = shd->sid;
   warp_tracker *wpt = map_get_warp_tracker(sid,tid,pc);
   assert(wpt != NULL);

   // signal the warp tracker
   if (wpt->complete_thd(tid)) {
      // if the warp has completed execution, remove this warp_tracker
      int warp_mask = 0;
      for (unsigned i=0; i<warp_size; i++) {
         if (wpt->tid[i] >= 0) {
            register_cta_thread_exit(shd, wpt->tid[i] );
            map_clear_warp_tracker(sid,wpt->tid[i],pc);
            warp_mask |= (1 << i);
         }
      }

      free_warp_tracker( wpt );

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
 * @param *shd Pointer to the shader core
 * @param tid Thread to check
 *
 * @return True is thread is being tracked
 */
bool warp_tracker_pool::wpt_thread_in_wpt(shader_core_ctx *shd, int tid) {
	int sid = shd->sid;
	std::map<address_type, warp_tracker*>::iterator it;
	for(it=warp_tracker_map[sid][tid].begin(); it!=warp_tracker_map[sid][tid].end(); it++)
		if((*it).second != NULL)
			return true;

	return false;
}



warp_tracker_pool& get_warp_tracker_pool(){
	static warp_tracker_pool* wpt_pool = new warp_tracker_pool(gpu_n_shader, gpu_n_thread_per_shader);
	return *wpt_pool;
}


//-------------------------------------------------------------------------------

/*

static warp_tracker ***warp_tracker_map;
static unsigned **g_warp_tracker_map_setl_cycle;
static warp_tracker *warp_tracker_pool = NULL;
static list<warp_tracker*> free_wpt;

warp_tracker* alloc_warp_tracker( int *tid_in, shader_core_ctx_t *shd ) 
{
   assert(!free_wpt.empty());
   warp_tracker* wpt = free_wpt.front();
   free_wpt.pop_front();

   wpt->set_warp(tid_in, shd);

   return wpt;
}

void free_warp_tracker(warp_tracker* wpt)
{
   free_wpt.push_back(wpt);
}

void init_warp_tracker( ) 
{
   unsigned int i;

   warp_tracker_map = (warp_tracker ***)calloc(gpu_n_shader, sizeof(warp_tracker **));
   g_warp_tracker_map_setl_cycle = (unsigned**)calloc(gpu_n_shader, sizeof(unsigned*));
   for (i=0; i<gpu_n_shader; i++) {
      warp_tracker_map[i] = (warp_tracker **)calloc(gpu_n_thread_per_shader, sizeof(warp_tracker *));
      g_warp_tracker_map_setl_cycle[i] = (unsigned*)calloc(gpu_n_thread_per_shader, sizeof(unsigned));
   }

   // max possible number of warps is just when each thread has its own warp
   warp_tracker_pool = new warp_tracker[gpu_n_shader * gpu_n_thread_per_shader]; 
   printf("%d %d %d %d\n", warp_size, gpu_n_shader, gpu_n_thread_per_shader, 
          warp_size * gpu_n_shader * gpu_n_thread_per_shader);
   for (i=0; i<gpu_n_shader*gpu_n_thread_per_shader; i++) {
      free_wpt.push_back(&(warp_tracker_pool[i]));
   }
   printf("%zd\n", free_wpt.size());
}

void wpt_register_warp( int *tid_in, shader_core_ctx_t *shd ) 
{
   int sid = shd->sid;
   unsigned i;
   int n_thd = 0;
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] >= 0) n_thd++;
   }

   if (!n_thd) return;

   warp_tracker *wpt = alloc_warp_tracker(tid_in, shd);

   // assign the new warp_tracker to warp_tracker_map
   for (i=0; i<warp_size; i++) {
      if (tid_in[i] >= 0) {
         assert( warp_tracker_map[sid][tid_in[i]] == NULL );
         warp_tracker_map[sid][tid_in[i]] = wpt;
         g_warp_tracker_map_setl_cycle[sid][tid_in[i]] = gpu_tot_sim_cycle + gpu_sim_cycle;
      }
   }
}

int wpt_signal_avail( int tid, shader_core_ctx_t *shd ) 
{
   int sid = shd->sid;
   warp_tracker *wpt = warp_tracker_map[sid][tid];
   assert(wpt != NULL);


   // signal the warp tracker
   if (wpt->avail_thd()) {
      return 1;
   } else {
      return 0;
   }
}

// Unlock a warp
void wpt_unlock_threads( int tid, shader_core_ctx_t *shd ) {
   int sid = shd->sid;
   warp_tracker *wpt = warp_tracker_map[sid][tid];
   assert(wpt != NULL);

    int thd_unlocked = 0;
    // Unlock
    for (unsigned i=0; i<warp_size; i++) {
		   if (wpt->tid[i] >= 0) {
			  shd->thread[wpt->tid[i]].avail4fetch++;
			  assert(shd->thread[wpt->tid[i]].avail4fetch <= 1);
			  assert( shd->warp[wpt->tid[i]/warp_size].n_avail4fetch < warp_size );
			  shd->warp[wpt->tid[i]/warp_size].n_avail4fetch++;
			   thd_unlocked = 1;
		   }
	   }

    if (shd->model == POST_DOMINATOR || shd->model == NO_RECONVERGE) {
   	 // Do nothing
    } else {
       // For this case, submit to commit_queue
       if (shd->using_commit_queue && thd_unlocked) {
          int *tid_unlocked = alloc_commit_warp();
          memcpy(tid_unlocked, wpt->tid, sizeof(int)*warp_size);
          dq_push(shd->thd_commit_queue,(void*)tid_unlocked);
       }
    }

    // the warp is ready to be fetched again, remove this warp_tracker
    for (unsigned i=0; i<warp_size; i++) {
	   if (wpt->tid[i] >= 0) {
		  warp_tracker_map[sid][wpt->tid[i]] = NULL;
		  g_warp_tracker_map_setl_cycle[sid][wpt->tid[i]] = gpu_tot_sim_cycle + gpu_sim_cycle;
	   }
    }

    free_warp_tracker( wpt );
}

*/

/*
 * Signal that the a thread is done and is exiting (exit instruction)
 *
 * Marks a thread as completed. If all threads in the warp have completed, call register_cta_thread_exit on all
 * threads and removed the warp from warp tracker.
 *
 * @param tid Thread that is exiting
 * @param *shd Pointer to the shader core
 *
 * @return The warp's mask of active threads.
 */
//

/*
int wpt_signal_complete( int tid, shader_core_ctx_t *shd ) 
{
   int sid = shd->sid;
   warp_tracker *wpt = warp_tracker_map[sid][tid];
   assert(wpt != NULL);

   // signal the warp tracker
   if (wpt->complete_thd(tid)) {
      // if the warp has completed execution, remove this warp_tracker
      int warp_mask = 0;
      for (unsigned i=0; i<warp_size; i++) {
         if (wpt->tid[i] >= 0) {
            register_cta_thread_exit(shd, wpt->tid[i] );
            warp_tracker_map[sid][wpt->tid[i]] = NULL;
            g_warp_tracker_map_setl_cycle[sid][wpt->tid[i]] = gpu_tot_sim_cycle + gpu_sim_cycle;
            warp_mask |= (1 << i);
         }
      }

      free_warp_tracker( wpt );

      return warp_mask;
   } else {
      return 0;
   }
}
*/

//------------------------------------------------------------------------------------

class thread_pc_tracker_class {
public:
   address_type *thd_pc; // tracks the pc of each thread
   map<address_type, unsigned> pc_count;
   unsigned acc_pc_count;
   int simd_width;
   static map<unsigned, unsigned> histogram;

   thread_pc_tracker_class( ) {
      this->acc_pc_count = 0;
      this->simd_width = 0;
      this->thd_pc = NULL;
   }

   thread_pc_tracker_class(int simd_width, int thread_count) {
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
      map<unsigned, unsigned>::iterator i;
      fprintf(fout, "Thread PC Histogram: ");
      for (i = histogram.begin(); i != histogram.end(); i++) {
         fprintf(fout, "%d:%d ", i->first, i->second);
      }
      fprintf(fout, "\n");
   }
};

map<unsigned, unsigned> thread_pc_tracker_class::histogram;

thread_pc_tracker_class *thread_pc_tracker = NULL;

void print_thread_pc_histogram( FILE *fout )
{
   thread_pc_tracker_class::histo_print(fout);
}

void print_thread_pc( FILE *fout )
{
   fprintf(fout, "SHD_PC_C: ");
   for (unsigned i=0; i<gpu_n_shader; i++) {
      fprintf(fout, "%d ", thread_pc_tracker[i].get_acc_pc_count() );
   }
   fprintf(fout, "\n");
}

void track_thread_pc( int shader_id, int *tid, address_type pc ) 
{
   if (!thread_pc_tracker) {
      thread_pc_tracker = new thread_pc_tracker_class[gpu_n_shader];
      for (unsigned i=0; i<gpu_n_shader; i++) {
         thread_pc_tracker[i] = thread_pc_tracker_class(warp_size, gpu_n_thread_per_shader);
      }
   }
   thread_pc_tracker[shader_id].set_threads_pc( tid, pc );
}

//------------------------------------------------------------------------------------

static int *commit_warp_pool = NULL;
static queue<int*> free_commit_warp_q;

void init_commit_warp( )
{
   unsigned int num_warp = warp_size * gpu_n_shader * gpu_n_thread_per_shader;
   commit_warp_pool = new int[num_warp];
   for (unsigned int i=0; i<num_warp; i+=warp_size) {
      free_commit_warp_q.push(&(commit_warp_pool[i]));
   }
}

int* alloc_commit_warp( )
{
   if (!commit_warp_pool) {
      init_commit_warp( );
   }

   assert(!free_commit_warp_q.empty());
   int *new_commit_warp = free_commit_warp_q.front();
   free_commit_warp_q.pop();

   return new_commit_warp;
}

void free_commit_warp( int *commit_warp )
{
   free_commit_warp_q.push(commit_warp);
}

// uncomment to enable checking for warp consistency
// #define CHECK_WARP_CONSISTENCY

void check_stage_pcs( shader_core_ctx_t *shader, unsigned stage )
{
#ifdef CHECK_WARP_CONSISTENCY
   address_type inst_pc = (address_type)-1;
   unsigned tid;
   if( shader->model == MIMD ) 
      return;

   std::set<unsigned> tids;

   for ( int i = 0; i < pipe_simd_width; i++) {
      if (shader->pipeline_reg[i][stage].hw_thread_id == -1 ) 
         continue;
      if ( inst_pc == (address_type)-1 ) 
         inst_pc = shader->pipeline_reg[i][stage].pc;
      tid = shader->pipeline_reg[i][stage].hw_thread_id;
      assert( tids.find(tid) == tids.end() );
      tids.insert(tid);
      assert( inst_pc == shader->pipeline_reg[i][stage].pc );
   }
#endif
}

void check_pm_stage_pcs( shader_core_ctx_t *shader, unsigned stage )
{
#ifdef CHECK_WARP_CONSISTENCY
   address_type inst_pc = (address_type)-1;
   unsigned tid;
   if( shader->model == MIMD ) 
      return;

   std::set<unsigned> tids;

   for (int i = 0; i < pipe_simd_width; i++) {
      if (shader->pre_mem_pipeline[i][stage].hw_thread_id == -1 ) 
         continue;
      if ( inst_pc == (address_type)-1 ) 
         inst_pc = shader->pre_mem_pipeline[i][stage].pc;
      tid = shader->pre_mem_pipeline[i][stage].hw_thread_id;
      assert( tids.find(tid) == tids.end() );
      tids.insert(tid);
      assert( inst_pc == shader->pre_mem_pipeline[i][stage].pc );
   }
#endif
}
