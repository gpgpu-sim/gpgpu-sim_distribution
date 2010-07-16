/* 
 * dwf.cc
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, and the
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


#include "dwf.h"
#include "histogram.h"
#include <map>
#include <set>
#include <deque>
#include <queue>
#include <string.h>

using namespace std;

unsigned int gpgpu_dwf_regbk = 1;
unsigned int gpgpu_dwf_heuristic = 0;
enum {
   MAJORITY  = 0,
   MINORITY  = 1,
   FIFO      = 2,
   PDOMPRIO  = 3,
   PC        = 4,
   MAJORITY_MAXHEAP = 5,
   N_DWFMODE
};

typedef struct warp_entry {
   address_type pc;
   int* tid; // thread id's
   int occ; // occupancy vector
   int pdom_prio; // pdom_priority
   int pdom_occ; // pdom_priority's aux data
   int next_warp; // index to next warp in an implicit queue
   void* lut_ptr; // pointer to the warp lut entry that last update this warp (a hack), done to decouple warp lut and warp pool
   int uid; // unique id of a warp
} warp_entry_t;

class issue_warp_majority {
public:

   virtual void add_threads( address_type pc, int *tid) = 0;
   virtual void push_warp( address_type pc, int idx) = 0;
   virtual int pop_warp( ) = 0;
   virtual void print( FILE *fout ) = 0;
   virtual ~issue_warp_majority( ) {}
};

typedef struct maxheap_lut_entry {
   address_type pc;    // pc of the warps
   int maxheap_idx; // index to the max heap
} maxheap_lut_entry_t;

typedef struct maxheap_entry {
   address_type pc;    // pc of the warps
   int n_thds;  // number of threads with this pc (from lut)
   int wpool_head;  // the first warp with this pc
   int wpool_tail;  // the last warp with this pc
   int lut_idx; // reverse index to the lut (for update in entry movement)
} maxheap_entry_t;

class mh_lut_class {
private:

   maxheap_lut_entry_t *lut_data;
   list<int> *lru_stack; // front = LRU
   int n_set;
   int insn_size_lgb2;

public:

   int size;
   int assoc;
   int n_read;
   int n_write;
   int n_read_per_cycle;
   int n_write_per_cycle;

   int n_aliased;
   static maxheap_lut_entry_t clean_entry;

   mh_lut_class (int size, int assoc, int n_read_per_cycle, int n_write_per_cycle ) {
      int i;

      this->size = size;
      this->assoc = assoc;
      lut_data = new maxheap_lut_entry_t[size];

      for (i=0; i<size; i++) {
         lut_data[i] = clean_entry;
      }

      n_set = size/assoc;
      assert(n_set && !((n_set - 1) & n_set)); // make sure n_set is a power of 2

      insn_size_lgb2 = 0;

      lru_stack = new list<int>[n_set];
      for (i=0; i<n_set; i++) {
         int j;
         for (j=0; j<assoc; j++) {
            lru_stack[i].push_back(i * assoc + j);
         }
      }

      this->n_read_per_cycle = n_read_per_cycle;
      this->n_write_per_cycle = n_write_per_cycle;
      this->n_read = 0;
      this->n_write = 0;
      this->n_aliased = 0;
   }

   ~mh_lut_class ( ) {
      delete[] lut_data;
   }

   // obtain entry at a known location
   maxheap_lut_entry_t get( int lut_idx ) {
      assert(lut_idx < size);
      n_read++;
      return lut_data[lut_idx];
   }

   // modify an entry at a known location
   void set( int lut_idx, maxheap_lut_entry_t lut_entry ) {
      n_write++;
      lut_data[lut_idx] = lut_entry;
   }

   // update a lut entry with a new index
   void update_mh_idx( int lut_idx, int mh_idx ) {
      n_write++;
      lut_data[lut_idx].maxheap_idx = mh_idx;
   }

   // lookup an entry with a pc
   int lookup( address_type pc ) {
      int i;
      int lut_idx = -1;
      int set_start_idx = get_set(pc) * assoc;

      // look for the matched entry within the set
      for (i = set_start_idx; i < (set_start_idx + assoc); i++) {
         if (lut_data[i].pc == pc) {
            lut_idx = i;
         }
      }

      // update lru stack if hit
      if (lut_idx != -1) {
         int set_idx = set_start_idx / assoc;
         list<int>::iterator it;
         it = find(lru_stack[set_idx].begin(), lru_stack[set_idx].end(), lut_idx);

         if (it != lru_stack[set_idx].end()) {
            lru_stack[set_idx].erase(it);
            lru_stack[set_idx].push_back(lut_idx);
         }
      }

      return lut_idx;
   }

   void free(int lut_idx) {
      set(lut_idx, clean_entry);

      int set_idx = lut_idx / assoc;
      list<int>::iterator it;
      it = find(lru_stack[set_idx].begin(), lru_stack[set_idx].end(), lut_idx);

      if (it != lru_stack[set_idx].end()) {
         lru_stack[set_idx].erase(it);
         lru_stack[set_idx].push_front(lut_idx);
      }
   }

   // find the LRU entry to be replaced
   int find_lru( maxheap_lut_entry_t lut_entry ) {
      int set_idx = get_set(lut_entry.pc);
      int lru_idx = lru_stack[set_idx].front();

      return lru_idx;
   }

   // actually replacing the LRU entry
   int replace_lru( maxheap_lut_entry_t lut_entry ) {
      int set_idx = get_set(lut_entry.pc);
      int lru_idx = lru_stack[set_idx].front();
      lru_stack[set_idx].pop_front();

      // counting the number of overwritten entries
      if (lut_data[lru_idx].maxheap_idx != 0) n_aliased++;

      set(lru_idx, lut_entry);
      lru_stack[set_idx].push_back(lru_idx);

      return lru_idx;
   }

   // reset the number of accesses to zero
   void reset_access( ) {
      n_read = 0;
      n_write = 0;
   }

   // clear the number of accesses - done at the end of scheduler cycle
   void clear_access( ) {
      n_read -= n_read_per_cycle;
      n_read = (n_read >= 0)? n_read : 0;
      n_write -= n_write_per_cycle;
      n_write = (n_write >= 0)? n_write : 0;
   }

   // test if the structure is done with all the required accesses
   int all_access_done( ) {
      return(n_read == 0 && n_write == 0);
   }

   void print_lut_e(FILE *fout, maxheap_lut_entry_t lut_e) {
      fprintf(fout, "[%08x]mh%02d", 
              lut_e.pc, lut_e.maxheap_idx);
   }

   void print(FILE *fout) {
      int i, j;
      for (i=0; i<n_set; i++) {
         fprintf(fout, "S%02d", i);
         for (j=0; j<assoc; j++) {
            fprintf(fout, " |%02d:", i * assoc + j);
            print_lut_e(fout, lut_data[i * assoc + j]);
         }
         fprintf(fout, "  ");
         list<int>::iterator it = lru_stack[i].begin();
         for (; it != lru_stack[i].end(); it++) {
            fprintf(fout, "%02d-", *it);
         }
         fprintf(fout, "\n");
      }
   }

private:

   inline int get_set(address_type pc) {
      return((pc >> insn_size_lgb2) & (n_set - 1));
   }
};

maxheap_lut_entry_t mh_lut_class::clean_entry = {0xDEADBEEF, 0};

// A class tracking the number of accesses done to the maxheap structure
// and the index ranges from 1..n_entries with 1 being the root
class maxheap_class {
private:

   maxheap_entry_t *maxheap_data;
   mh_lut_class *lut;

public:

   int n_read;
   int n_write;
   int n_entries;
   int size;
   int n_read_per_cycle;
   int n_write_per_cycle;

   int max_n_entries;
   static maxheap_entry_t clean_entry;

   maxheap_class( int size, mh_lut_class *lut, int n_read_per_cycle, int n_write_per_cycle ) {
      n_read = 0;
      n_write = 0;
      n_entries = 0; // index to the last element
      this->size = size;
      maxheap_data = new maxheap_entry_t[size];

      for (int i=0; i<size; i++) {
         maxheap_data[i] = clean_entry;
      }

      this->lut = lut;

      this->n_read_per_cycle = n_read_per_cycle;
      this->n_write_per_cycle = n_write_per_cycle;
      this->n_read = 0;
      this->n_write = 0;
      this->max_n_entries = 0;
   }

   ~maxheap_class( ) {
      delete[] maxheap_data;
   }

   // insert a new entry into the maxheap
   // return: the index to the new entry
   int insert( maxheap_entry_t mh_entry ) {
      assert(n_entries + 1 < size); 
      n_write++;
      n_entries++;
      maxheap_data[n_entries] = mh_entry;
      max_n_entries = (max_n_entries >= n_entries)? max_n_entries : n_entries;
      return n_entries;
   }

   // retrieve the max heap entry at index [mh_idx]
   maxheap_entry_t get( int mh_idx ) {
      assert(mh_idx > 0);
      assert(mh_idx <= n_entries);
      n_read++;
      return maxheap_data[mh_idx];
   }

   // replace the max heap entry at index [mh_idx]
   void set( int mh_idx, maxheap_entry_t mh_entry ) {
      assert(mh_idx > 0);
      assert(mh_idx <= n_entries);
      n_write++;
      maxheap_data[mh_idx] = mh_entry;
   }

   // a special version of set that only reset the lut_idx
   void remove_lut_idx( int mh_idx ) {
      assert(mh_idx > 0);
      assert(mh_idx <= n_entries);
      n_write++;
      maxheap_data[mh_idx].lut_idx = -1;
   }

   // read both childrens of a given node, count as one read
   // return the number of child read
   int get_childof(int mh_idx, maxheap_entry_t *child) {
      int child_idx = childof(mh_idx);
      int child_read = 0;

      if (child_idx <= n_entries) {
         n_read++;
         child[0] = maxheap_data[child_idx]; 
         child_read++;
      }
      if (child_idx + 1 <= n_entries) {
         child[1] = maxheap_data[child_idx + 1]; 
         child_read++;
      }

      return child_read;
   }

   // pop the root entry of max heap
   maxheap_entry_t pop_root( ) {
      maxheap_entry_t old_root = get(1);
      maxheap_entry_t curr_mhe[3];
      curr_mhe[0] = get(n_entries);

      set(1, curr_mhe[0]);
      if (curr_mhe[0].lut_idx >= 0)
         lut->update_mh_idx(curr_mhe[0].lut_idx, 1);

      n_entries--;

      int curr_node = 1;
      int n_child = 0;

      n_child = get_childof(curr_node, curr_mhe + 1);
      while (n_child > 0) {
         int max_child = 0;
         int i;
         for (i = 1; i < n_child + 1; i++) {
            if (cmp_mh(curr_mhe[i], curr_mhe[max_child])) {
               max_child = i;
            }
         }

         n_child = 0;
         if (max_child > 0) {
            int max_child_node = childof(curr_node) + max_child - 1;
            set(curr_node, curr_mhe[max_child]);
            set(max_child_node, curr_mhe[0]);

            // update the lut for this swap
            if (curr_mhe[max_child].lut_idx >= 0)
               lut->update_mh_idx(curr_mhe[max_child].lut_idx, curr_node);
            if (curr_mhe[0].lut_idx >= 0)
               lut->update_mh_idx(curr_mhe[0].lut_idx, max_child_node);

            // get the next child
            curr_node = max_child_node;
            n_child = get_childof(curr_node, curr_mhe + 1);
         }
      }

      return old_root;
   }

   // probe if the maxheap is empty
   int empty( ) {
      return(n_entries == 0);
   }

   // reset the number of accesses to zero
   void reset_access( ) {
      n_read = 0;
      n_write = 0;
   }

   // clear the number of accesses - done at the end of scheduler cycle
   void clear_access( ) {
      n_read -= n_read_per_cycle;
      n_read = (n_read >= 0)? n_read : 0;
      n_write -= n_write_per_cycle;
      n_write = (n_write >= 0)? n_write : 0;
   }

   // test if the structure is done with all the required accesses
   int all_access_done( ) {
      return(n_read == 0 && n_write == 0);
   }

   // sort the max heap again starting from start_idx 
   // (this entry can only go up in the tree to the root)
   void sort_bottomup(int start_idx) {
      maxheap_entry_t mh_entry;
      maxheap_entry_t mh_parent;

      if (start_idx == 1) return; // no need to resort if the root is incremented

      int curr_idx = start_idx;
      int parent_idx = parentof(start_idx);

      int continue_sort = 1;
      while (curr_idx > 1 && continue_sort) {
         mh_entry = get(curr_idx);
         mh_parent = get(parent_idx);

         // swap the entries if it is now larger than it's parent
         if (cmp_mh(mh_entry, mh_parent)) {
            set(parent_idx, mh_entry);
            set(curr_idx, mh_parent);

            // update the lut for this swap
            if (mh_entry.lut_idx >= 0)
               lut->update_mh_idx(mh_entry.lut_idx, parent_idx);
            if (mh_parent.lut_idx >= 0)
               lut->update_mh_idx(mh_parent.lut_idx, curr_idx);

            // update index for next iteration
            curr_idx = parent_idx;
            parent_idx = parentof(curr_idx);
         } else {
            // swap did not happen, no need to sort anymore 
            continue_sort = 0;
         }
      }
   }

   void print_mh_e(FILE *fout, maxheap_entry_t mh_e) {
      fprintf(fout, "[%08x]%03d(H%03dT%03d)p%02d | ", 
              mh_e.pc, mh_e.n_thds, mh_e.wpool_head, mh_e.wpool_tail, mh_e.lut_idx);
   }

   void print(FILE *fout) {
      fprintf(fout, "MaxHeap: ");
      fprintf(fout, "N_entries = %d\n", n_entries);
      for (int i=0; i<n_entries; i++) {
         print_mh_e(fout, maxheap_data[i + 1]);
         if (!((i + 2) & (i + 1))) fprintf(fout, "\n");
      }
      fprintf(fout, "\n");
   }

private:

   static inline int parentof(int mh_idx) {
      assert(mh_idx > 0);
      return(mh_idx / 2);
   }

   static inline int childof(int mh_idx) {
      return(mh_idx * 2);
   }

   static inline int cmp_mh(maxheap_entry_t &a, maxheap_entry_t &b) {
      if (a.n_thds > b.n_thds) return 1;
      if (a.n_thds == b.n_thds) {
         if (a.pc < b.pc) return 1;
      }
      return 0;
   }

};

maxheap_entry_t maxheap_class::clean_entry = {0, 0, -1, -1, -1};

typedef struct mh_update_struct {
   int n_maxheap_read;
   int n_maxheap_write;
   int n_mhlut_read;
   int n_mhlut_write;
} mh_update;

// heap implementation of majority policy
class issue_warp_majority_heap : public issue_warp_majority {
public:

   mh_lut_class mh_lut;
   maxheap_class maxheap;

   maxheap_lut_entry_t major_lut_e;
   maxheap_entry_t major_mh_e;

   vector<warp_entry_t> *warp_pool;
   int simd_width;

   int n_stall_on_maxheap;

   queue<mh_update> update_queue;
   static pow2_histogram n_pending_updates_histo;

   issue_warp_majority_heap (int simd_width = 0, vector<warp_entry_t> *bp = NULL,
                             int lut_size = 32, int lut_assoc = 4, int maxheap_size = 128,
                             int n_read_lut = 4, int n_write_lut = 4, 
                             int n_read_mh = 4, int n_write_mh = 4) 
   : mh_lut(lut_size, lut_assoc, n_read_lut, n_write_lut), 
   maxheap(maxheap_size, &mh_lut, n_read_mh, n_write_mh)
   {
      this->simd_width = simd_width;
      this->warp_pool = bp;

      this->major_lut_e = mh_lut_class::clean_entry;
      this->major_mh_e  = maxheap_class::clean_entry;

      this->n_stall_on_maxheap = 0;
   }

   // adding more threads to a specify pc
   // these threads may end up in different warpes
   void add_threads( address_type pc, int *tid) {
      int i;
      int n_thds = 0;
      for (i=0; i<simd_width; i++) {
         if (tid[i] >= 0) n_thds++;
      }

      // handle special case with adding threads to current majority pc
      if (major_lut_e.pc == pc) {
         assert(major_mh_e.pc == pc);
         major_mh_e.n_thds += n_thds;
         return;
      }

      maxheap_lut_entry_t lut_e;
      maxheap_entry_t mh_entry;

      // snapshot the current maxheap read/write demand
      mh_update new_mh_update;
      new_mh_update.n_maxheap_read  = maxheap.n_read;
      new_mh_update.n_maxheap_write = maxheap.n_write;
      new_mh_update.n_mhlut_read  = mh_lut.n_read;
      new_mh_update.n_mhlut_write = mh_lut.n_write;

      int lut_idx = mh_lut.lookup(pc);

      int sort_from_idx = 0;

      if (lut_idx >= 0) {
         // obtain the entry
         lut_e = mh_lut.get(lut_idx);

         // get the maxheap entry and update its number of threads
         mh_entry = maxheap.get(lut_e.maxheap_idx);
         mh_entry.n_thds += n_thds;
         maxheap.set(lut_e.maxheap_idx, mh_entry);

         // sort from this specific entry
         sort_from_idx = lut_e.maxheap_idx;
      } else {
         // create a new lut entry 
         lut_e = mh_lut_class::clean_entry;
         lut_e.pc = pc;

         // get index to the LRU lut entry in this set
         lut_idx = mh_lut.find_lru(lut_e);

         // get the replaced lut entry and remove its link with the maxheap entry
         maxheap_lut_entry_t lut_old = mh_lut.get(lut_idx);
         if (lut_old.maxheap_idx > 0) maxheap.remove_lut_idx(lut_old.maxheap_idx);

         // create a new maxheap entry
         mh_entry = maxheap_class::clean_entry;
         mh_entry.pc = pc;
         mh_entry.n_thds = n_thds;
         mh_entry.lut_idx = lut_idx;

         // push the new entry into the maxheap and lut respectively
         lut_e.maxheap_idx = maxheap.insert(mh_entry);
         mh_lut.replace_lru(lut_e);

         // start sorting from the bottom?
         sort_from_idx = lut_e.maxheap_idx;
      }

      maxheap.sort_bottomup(sort_from_idx);

      // record the newly generated maxheap read/write demand from this update
      new_mh_update.n_maxheap_read  = maxheap.n_read - new_mh_update.n_maxheap_read;
      new_mh_update.n_maxheap_write = maxheap.n_write - new_mh_update.n_maxheap_write;
      new_mh_update.n_mhlut_read  = mh_lut.n_read - new_mh_update.n_mhlut_read;
      new_mh_update.n_mhlut_write = mh_lut.n_write - new_mh_update.n_mhlut_write;

      update_queue.push(new_mh_update);
   }

   // call this when a new warp allocated for a specific pc
   void push_warp( address_type pc, int idx) {
      maxheap_entry_t *p_mh_e = NULL;
      maxheap_entry_t mh_e;
      maxheap_lut_entry_t lut_e = mh_lut_class::clean_entry;
      int lut_idx = -1;

      if (major_mh_e.pc == pc) {
         p_mh_e = &major_mh_e;
      } else {
         lut_idx = mh_lut.lookup(pc);
         assert(lut_idx >= 0); // if it is a miss, a new entry should have been created already
         lut_e = mh_lut.get(lut_idx);
         mh_e = maxheap.get(lut_e.maxheap_idx);
         p_mh_e = &mh_e;

         // discounting these 'gets'
         // because they should be combined with the 'gets' in add_threads()
         mh_lut.n_read--;
         maxheap.n_read--;
      }

      if (p_mh_e->wpool_head == -1) {
         p_mh_e->wpool_head = idx;
         p_mh_e->wpool_tail = idx;
      } else {
         (*warp_pool)[p_mh_e->wpool_tail].next_warp = idx;
         p_mh_e->wpool_tail = idx;
      }

      if (major_mh_e.pc == pc) {
      } else {
         maxheap.set(lut_e.maxheap_idx, mh_e);
         // discounting this 'set'
         // because it should be combined with the 'set' in add_threads()
         maxheap.n_write--;
      }
   }

   // obtain a warp index from this issue logic
   int pop_warp( ) {
      int bidx = -1;
      if (major_mh_e.wpool_head == -1 && !maxheap.empty()) {
         if (this->all_access_done( )) {
            // pop the majority PC from max heap
            major_mh_e = maxheap.pop_root();

            // pop its corresponding entry from the lut as well (if it exists)
            if (major_mh_e.lut_idx >= 0) {
               major_lut_e = mh_lut.get(major_mh_e.lut_idx);
               mh_lut.free(major_mh_e.lut_idx);
            } else {
               major_lut_e = mh_lut_class::clean_entry;
            }
         } else {
            n_stall_on_maxheap += 1;
            bidx = -1;
            return bidx;
         }
      }

      // just pop and entry to from the virtual queue (and set the head pointer to next warp)
      bidx = major_mh_e.wpool_head;
      if (bidx >= 0) {
         major_mh_e.wpool_head = (*warp_pool)[major_mh_e.wpool_head].next_warp;
      }

      return bidx;
   }

   void reset_access( ) {
      maxheap.reset_access();
      mh_lut.reset_access();

      while (!update_queue.empty()) {
         update_queue.pop();
      }
   }

   inline void consume_access( int &req_acc, int &avl_acc) {
      if (req_acc > avl_acc) {
         req_acc -= avl_acc;
         avl_acc = 0;
      } else {
         avl_acc -= req_acc;
         req_acc = 0;
      }
   }

   void clear_access( ) {
      maxheap.clear_access();
      mh_lut.clear_access();

      int n_maxheap_read_bw  = maxheap.n_read_per_cycle;
      int n_maxheap_write_bw = maxheap.n_write_per_cycle;
      int n_mhlut_read_bw  = mh_lut.n_read_per_cycle;
      int n_mhlut_write_bw = mh_lut.n_write_per_cycle;

      while ((n_maxheap_read_bw > 0 || n_maxheap_read_bw > 0 || 
              n_mhlut_read_bw > 0 || n_mhlut_write_bw > 0) && !update_queue.empty()) {
         mh_update &c_update = update_queue.front();

         consume_access (c_update.n_maxheap_read,  n_maxheap_read_bw);
         consume_access (c_update.n_maxheap_write, n_maxheap_write_bw);
         consume_access (c_update.n_mhlut_read,   n_mhlut_read_bw);
         consume_access (c_update.n_mhlut_write,  n_mhlut_write_bw);

         if (c_update.n_maxheap_read == 0 && c_update.n_maxheap_write == 0 && 
             c_update.n_mhlut_read == 0 && c_update.n_mhlut_write == 0) {
            update_queue.pop();
         } else {
            break;
         }
      }

      n_pending_updates_histo.add2bin(update_queue.size());
   }

   void print( FILE *fout ) {
      fprintf(fout, "LUT: ");
      mh_lut.print_lut_e(fout, major_lut_e);
      fprintf(fout, " \tMH: ");
      maxheap.print_mh_e(fout, major_mh_e);
      fprintf(fout, "\n");
      mh_lut.print(fout);
      maxheap.print(fout);
   }

   static void print_stat( FILE *fout) {
      fprintf(fout, "n_pending_maxheap_updates = ");
      n_pending_updates_histo.fprint(fout);
      fprintf(fout, "\n");
   }

private:

   int all_access_done( ) {
      return(maxheap.all_access_done() && mh_lut.all_access_done());

   }
};
pow2_histogram issue_warp_majority_heap::n_pending_updates_histo;

class warp_queue {
public:
   int m_pc;
   int n_thds;
   int simd_width;
   deque<int> idx_queue;

   warp_queue( address_type pc, int simd_width) {
      this->m_pc = pc;
      this->n_thds = 0;
      this->simd_width = simd_width;
   }

   // called right after a lut_entry is looked up
   void add_threads( int *tid ) {
      for (int i=0; i<simd_width; i++) {
         if (tid[i] >= 0) this->n_thds++;
      }
   }

   // called right after a warp is issued
   void sub_threads( int *tid ) {
      for (int i=0; i<simd_width; i++) {
         if (tid[i] >= 0) this->n_thds--;
      }
   }

   // if other warp queue should be ahead
   bool operator<(const warp_queue& other) const {
      if (n_thds == other.n_thds) {
         return(m_pc > other.m_pc); // smaller pc first
      } else {
         return(n_thds < other.n_thds);
      }
   }
   bool operator>(const warp_queue& other) const {
      if (n_thds == other.n_thds) {
         return(m_pc > other.m_pc); // smaller pc first
      } else {
         return(n_thds > other.n_thds);
      }
   }

   void print( FILE *fout ) {
      fprintf(fout, "0x%08x(%03d)=[", m_pc, n_thds);
      deque<int>::iterator dit = idx_queue.begin();
      for (; dit != idx_queue.end(); dit++) {
         fprintf(fout, "%03d ", *dit);
      }
      fprintf(fout, "]\n");
   }
};

bool minor_warp( const warp_queue* a, const warp_queue* b ) {
   return(*a<*b);
}

// queue implementation of majority scheduling policy
class issue_warp_majority_queue : public issue_warp_majority {
public:
   map<address_type, warp_queue* > majority_map;
   set<warp_queue*> warpq_set;
   warp_queue* maj_warp;

   vector<warp_entry_t> *warp_pool;
   int simd_width;

   issue_warp_majority_queue(int simd_width = 0, vector<warp_entry_t> *bp = NULL) {
      this->maj_warp = NULL;
      this->simd_width = simd_width;
      this->warp_pool = bp;
   }

   // adding more threads to a specify pc
   // these threads may end up in different warps
   void add_threads( address_type pc, int *tid) {
      warp_queue* bq = majority_map[pc];
      if (bq == NULL) {
         bq = new warp_queue(pc,simd_width);
         warpq_set.insert(bq);
         majority_map[pc] = bq;
      }
      bq->add_threads(tid);
   }

   // call this when a new warp allocated for a specific pc
   void push_warp( address_type pc, int idx) {
      warp_queue* bq = majority_map[pc];
      assert(bq != NULL);
      bool check_redundant_idx = false;
      if (check_redundant_idx) {
         deque<int>::iterator dit = find(bq->idx_queue.begin(), bq->idx_queue.end(), idx);
         assert(dit == bq->idx_queue.end());
      }
      bq->idx_queue.push_back(idx);
   }

   // obtain a warp index from this issue logic
   int pop_warp( ) {
      int bidx = -1;

      // find the new majority pc if it didn't exist
      if (maj_warp == NULL && warpq_set.size()) {
         maj_warp = *max_element(warpq_set.begin(), warpq_set.end(), minor_warp);
      }

      // if a majority pc indeed exist
      if (maj_warp) {
         assert(!maj_warp->idx_queue.empty());
         bidx = maj_warp->idx_queue.front();
         maj_warp->idx_queue.pop_front();
         maj_warp->sub_threads((*warp_pool)[bidx].tid);

         // when the majority pc runs out of thread
         if (maj_warp->n_thds == 0) {
            // remove that warp queue
            warpq_set.erase(maj_warp);
            majority_map.erase(maj_warp->m_pc);
            delete maj_warp;
            maj_warp = NULL;
         }
      }

      return bidx;
   }

   void print( FILE *fout ) {
      fprintf(fout, "issue_warp_majority:\n");
      set<warp_queue*>::iterator dit = warpq_set.begin();
      for (; dit != warpq_set.end(); dit++) {
         fprintf(fout, " %c ", ((*dit)==maj_warp)? 'M':' ');
         (*dit)->print(fout);
      }
   }

   void check_consistency( ) {
      set<warp_queue*>::iterator set_it = warpq_set.begin();
      for (; set_it != warpq_set.end(); set_it++) {
         warp_queue* bq = (*set_it);

         int real_nthds = 0;
         deque<int>::iterator dit = bq->idx_queue.begin();
         for (; dit != bq->idx_queue.end(); dit++) {
            int *tid = (*warp_pool)[*dit].tid;
            for (int i = 0; i < simd_width; i++) {
               real_nthds += (tid[i] >= 0)? 1 : 0;
            }
         }

         assert(real_nthds == bq->n_thds);
      }
   }
};

// pdom priority 
class lesspdom_first {
public:
   vector<warp_entry_t> *warp_pool;
   lesspdom_first( vector<warp_entry_t> *bp=NULL ) {
      this->warp_pool = bp;
   }
   bool operator() (const int &idx_a, const int &idx_b) const {
      if ((*warp_pool)[idx_a].pdom_prio != (*warp_pool)[idx_b].pdom_prio) {
         return((*warp_pool)[idx_a].pdom_prio < (*warp_pool)[idx_b].pdom_prio);
      } else {
         return((*warp_pool)[idx_a].occ > (*warp_pool)[idx_b].occ);
      }
   }
};


class issue_warp_pdom_prio {
public:
   vector<warp_entry_t> *warp_pool;
   int* thd_pdom_prio; 
   int simd_width;
   int n_threads;

   int resort_needed;
   list<int> pdom_pqueue; //the queue holding all index

   lesspdom_first lesspdom_cmp;

   static set<address_type> reconvgence_pt; //table holding all recvg pt

   issue_warp_pdom_prio (int simd_width = 0, vector<warp_entry_t> *bp = NULL, 
                         int n_threads = 0) 
   : lesspdom_cmp(bp)
   {
      this->simd_width = simd_width;
      this->warp_pool = bp;
      this->n_threads = n_threads;
      this->thd_pdom_prio = new int[n_threads];
      memset(this->thd_pdom_prio, 0, sizeof(int)*n_threads);
      this->resort_needed = 0;
   }

   ~issue_warp_pdom_prio( ) {
      delete[] this->thd_pdom_prio;
   }

   void reinit( ) {
      memset(this->thd_pdom_prio, 0, sizeof(int)*n_threads);
   }

   // adding more threads to a warp
   void add_threads( int idx, address_type pc) {
      assert((*warp_pool)[idx].pc == pc);

      // check to see if this is a newly allocated warp
      bool check_pdom = false;
      if ((*warp_pool)[idx].pdom_prio == -1) {
         check_pdom = true;
      }

      // check for newly assigned threads to the warp
      int pdom_occ = (*warp_pool)[idx].pdom_occ;
      int *tid = (*warp_pool)[idx].tid;
      for (int i=0; i<simd_width; i++) {
         if (tid[i] >= 0 && !(pdom_occ & (1<<i))) {
            if ((*warp_pool)[idx].pdom_prio < thd_pdom_prio[tid[i]]) {
               (*warp_pool)[idx].pdom_prio = thd_pdom_prio[tid[i]];
               resort_needed = 1;
            }
            pdom_occ |= (1<<i);
         }
      }
      if (check_pdom) {
         if (reconvgence_pt.find(pc) != reconvgence_pt.end()) {
            (*warp_pool)[idx].pdom_prio += 1;
         }
      }
   }

   // call this when a new warp allocated for a specific pc
   void push_warp( address_type pc, int idx ) {
      assert((*warp_pool)[idx].pc == pc);
      // initialize the pdom_prio for this newly allocated warp
      (*warp_pool)[idx].pdom_prio = -1;
      (*warp_pool)[idx].pdom_occ = 0;
      pdom_pqueue.push_back(idx);
   }

   // obtain a warp index from this issue logic
   int front_warp( ) {
      int bidx = -1;

      if (!pdom_pqueue.empty()) {
         if (resort_needed) {
            pdom_pqueue.sort(lesspdom_cmp);
            resort_needed = 0;
         }

         bidx = pdom_pqueue.front();
      }

      return bidx;
   }

   int size( ) {
      return pdom_pqueue.size();
   }

   void enforce_resort( ) {
      resort_needed = 1;
   }

   int pop_warp( ) {
      int bidx = -1;

      if (!pdom_pqueue.empty()) {
         if (resort_needed) {
            pdom_pqueue.sort(lesspdom_cmp);
            resort_needed = 0;
         }

         bidx = pdom_pqueue.front();
         pdom_pqueue.pop_front();

         // update the pdom prio of each thread inside a warp
         for (int i=0; i<simd_width; i++) {
            if ((*warp_pool)[bidx].tid[i] >= 0) {
               thd_pdom_prio[(*warp_pool)[bidx].tid[i]] = (*warp_pool)[bidx].pdom_prio;
            }
         }
      }

      return bidx;
   }

};

set<address_type> issue_warp_pdom_prio::reconvgence_pt = set<address_type>(); 
//*/


class npc_tracker_class {
public:
   map<address_type, unsigned> pc_count;
   unsigned* acc_pc_count;
   int simd_width;
   static map<unsigned, unsigned> histogram;

   npc_tracker_class( ) {
      this->acc_pc_count = NULL;
      this->simd_width = 0;
   }

   npc_tracker_class(unsigned* acc_pc_count, int simd_width) {
      this->acc_pc_count = acc_pc_count;
      this->simd_width = simd_width;
   }

   void add_threads( int *tid, address_type pc ) {
      for (int i=0; i<simd_width; i++) {
         if (tid[i] != -1) pc_count[pc] += 1; // automatically create a new entry if not exist
      }
   }

   void sub_threads( int *tid, address_type pc ) {
      for (int i=0; i<simd_width; i++) {
         if (tid[i] != -1) {
            pc_count[pc] -= 1;
            assert((int)pc_count[pc] >= 0);
            if (pc_count[pc] == 0) pc_count.erase(pc); // manually erasing entries with 0 count
         }
      }
   }

   void update_acc_count( ) { 
      (*acc_pc_count) += pc_count.size(); 
      histogram[pc_count.size()] += 1;
   }

   unsigned count( ) { return pc_count.size();}

   static void histo_print( FILE* fout ) {
      map<unsigned, unsigned>::iterator i;
      fprintf(fout, "DYHW nPC Histogram: ");
      for (i = histogram.begin(); i != histogram.end(); i++) {
         fprintf(fout, "%d:%d ", i->first, i->second);
      }
      fprintf(fout, "\n");
   }
};

map<unsigned, unsigned> npc_tracker_class::histogram;

class pc_tag {
private:

   address_type m_pc;

public:

   pc_tag () {
      this->reset();
   }

   pc_tag (const pc_tag& p) { this->m_pc = p.m_pc;}
   pc_tag (const address_type& other_pc) { this->m_pc = other_pc;}

   pc_tag& operator=(const pc_tag& p) { m_pc = p.m_pc; return *this;}
   pc_tag& operator=(const address_type& other_pc) { m_pc = other_pc; return *this;}

   inline bool operator==(const pc_tag& p) const { return(m_pc == p.m_pc);}
   inline bool operator==(const address_type& other_pc) const { return(m_pc == other_pc);}

   inline bool operator!=(const pc_tag& p) const { return(m_pc != p.m_pc);}
   inline bool operator!=(const address_type& other_pc) const { return(m_pc != other_pc);}

   inline bool operator<(const pc_tag& p) const { return(m_pc < p.m_pc);}

   inline void reset() {
      m_pc = -1;
   }

   inline address_type get_pc() const { return m_pc;}

   // the hash function to warp LUT
   inline unsigned lut_hash( int insn_size_lgb2, int lut_nsets ) const {
      return(m_pc >> insn_size_lgb2) & (lut_nsets - 1);
   }

   inline void to_print(char *buffer, unsigned length) {
      snprintf(buffer, length, "0x%08x", m_pc);
   }
};

template <class Tag>
class tag2warp_entry_t {
public: 

   Tag tag;
   int idx; // pointing to warp pool
   int occ; // occupancy vector
   int accessed; // is the entry accessed this cycle

   tag2warp_entry_t () {
      this->reset();
   }

   ~tag2warp_entry_t () {}

   tag2warp_entry_t (const tag2warp_entry_t& p) {
      this->tag = p.tag;
      this->idx = p.idx;
      this->occ = p.occ;
      this->accessed = p.accessed;
   }

   tag2warp_entry_t& operator=(const tag2warp_entry_t& p) {
      if (this != &p) {
         tag = p.tag;
         idx = p.idx;
         occ = p.occ;
         accessed = p.accessed;
      }
      return *this;
   }

   inline bool operator==(const tag2warp_entry_t& p) const {
      return(tag == p.tag);
   }

   inline bool operator==(const Tag& test_tag) const {
      return(tag == test_tag);
   }

   inline bool operator()(const tag2warp_entry_t& p) const {
      return(tag == p.tag);
   }

   inline void reset() {
      tag.reset();
      idx = 0; 
      occ = 0; 
      accessed = 0; 
   }

   void print( FILE *fout ) {
      static char buffer[20];
      tag.to_print(buffer,20);
      fprintf(fout, "\t%s->%03d (%02x)\n", buffer, idx, occ);
   }

};

template <class Tag>
class tag2warp_set {
public:
   vector< tag2warp_entry_t<Tag> > entry;
   list< tag2warp_entry_t<Tag>* > lru_stack;

   tag2warp_set(int assoc = 0) : entry(assoc) {
      for (unsigned j=0; j<this->entry.size(); j++) {
         this->lru_stack.push_back(&(this->entry[j]));
      }
   }

   tag2warp_set(const tag2warp_set& other) : entry(other.entry.size()) {
      for (unsigned j=0; j<this->entry.size(); j++) {
         this->lru_stack.push_back(&(this->entry[j]));
      }
   }

   tag2warp_set& operator=(const tag2warp_set& p) {
      printf("tag2warp_set assignment operator called!\n");
      return *this;
   }

   ~tag2warp_set() {}
};

template <class Tag>
class warp_lut {
public:
   virtual ~warp_lut() {}
   virtual tag2warp_entry_t<Tag>* lookup_pc2warp( const Tag& tag, bool& lut_missed ) = 0;
   virtual void invalidate_entry( tag2warp_entry_t<Tag>* lut_entry, int warp_idx ) = 0;
   virtual void clear_accessed( ) = 0;
   virtual void print( FILE* fout) = 0;
};

template <class Tag>
class warp_lut_sa : public warp_lut<Tag> {
private: 
   int lut_size;
   int lut_assoc;
   vector< tag2warp_set<Tag> > tag2warp_lut;
   int insn_size_lgb2;

   queue< tag2warp_entry_t<Tag>* > lut_accessed_q; // store accessed lut entry for clear

   struct same_tag {
      Tag tag;
      bool operator()(tag2warp_entry_t<Tag>* a) {
         return(a->tag == tag);
      }
   };

   static unsigned int lut_aliased;

public:
   warp_lut_sa(int lut_size, int lut_assoc, int insn_size) {
      this->lut_size = lut_size;
      this->lut_assoc = lut_assoc;

      // optimize for LUT hash function
      insn_size_lgb2 = 0;
      while ( (1 << insn_size_lgb2) < insn_size ) insn_size_lgb2++;

      // initialize the pc2warp LUT
      // note: lut_size is the absolute size of LUT regardless of assoc.
      this->tag2warp_lut.assign(lut_size/lut_assoc, tag2warp_set<Tag>(lut_assoc));

      // assert on #set in LUT to be power of 2
      int lut_nset_pow2 = 1;
      while ( lut_nset_pow2 < (int)tag2warp_lut.size() ) lut_nset_pow2 <<= 1;
      assert((int)tag2warp_lut.size() == lut_nset_pow2);
   }

   tag2warp_entry_t<Tag>* lookup_pc2warp( const Tag& tag, bool& lut_missed );
   void invalidate_entry( tag2warp_entry_t<Tag>* lut_entry, int warp_idx ) {
      if (lut_entry != NULL) { // check for warp lut entry invalidation
         if (lut_entry->idx == warp_idx) {
            lut_entry->reset();
         }
      }
   }

   void clear_accessed( );

   void print( FILE* fout) {
      for (unsigned i=0; i< tag2warp_lut.size(); i++) {
         for (unsigned j=0; j< tag2warp_lut[i].entry.size(); j++) {
            fprintf(fout, "lut%03d-%02d:", i, j);
            tag2warp_lut[i].entry[j].print(fout);
         }
      }
   }

   static void print_stats ( FILE* fout ) {
      fprintf( fout, "lut_aliased = %d\n", lut_aliased);
   }
};
template <class Tag> unsigned int warp_lut_sa<Tag>::lut_aliased = 0;


// lookup function in LUT
// may return an entry that has different PC for replacement
// or return a NULL pointer to indicate that the entry is accessed by another port
template <class Tag>
tag2warp_entry_t<Tag>* warp_lut_sa<Tag>::lookup_pc2warp( const Tag &tag, bool &lut_missed )
{
   tag2warp_entry_t<Tag>* lut_entry = NULL;
   unsigned hashed_pc = tag.lut_hash(insn_size_lgb2, tag2warp_lut.size());
   list< tag2warp_entry_t<Tag>* > &hashed_lru_stack = tag2warp_lut.at(hashed_pc).lru_stack;
   struct same_tag same_tag_f;

   same_tag_f.tag = tag;
   typename list< tag2warp_entry_t<Tag>* >::iterator lut_it;
   lut_it = find_if(hashed_lru_stack.begin(), 
                    hashed_lru_stack.end(),
                    same_tag_f);
   if (lut_it != hashed_lru_stack.end()) {
      lut_entry = *lut_it;
      lut_entry->accessed = 1;
      lut_accessed_q.push(lut_entry);
      hashed_lru_stack.splice(hashed_lru_stack.end(), hashed_lru_stack, lut_it);
      assert(lut_entry == hashed_lru_stack.back());
      lut_missed = false;
   } else {
      assert(!hashed_lru_stack.empty());
      lut_entry = hashed_lru_stack.front();
      if (lut_entry->accessed) {
         lut_entry = NULL;
      } else {
         lut_entry->accessed = 1;
         lut_accessed_q.push(lut_entry);
         hashed_lru_stack.splice(hashed_lru_stack.end(), hashed_lru_stack, hashed_lru_stack.begin());
         assert(lut_entry == hashed_lru_stack.back());
         lut_aliased++;
      }
      lut_missed = true;
   }
   assert(hashed_lru_stack.size() == tag2warp_lut[hashed_pc].entry.size());

   return lut_entry;
}

template <class Tag>
void warp_lut_sa<Tag>::clear_accessed( ) {
   while ( !lut_accessed_q.empty() ) {
      lut_accessed_q.front()->accessed = 0;
      lut_accessed_q.pop();
   }
}

// a perfect warp lut that never misses.
template <class Tag>
class warp_lut_perfect : public warp_lut<Tag> {
private:
   typedef map< Tag, tag2warp_entry_t<Tag>* > warp_map_t;
   warp_map_t m_tag2entry_map;

   static unsigned int lut_max_size;
public:
   warp_lut_perfect() {}
   ~warp_lut_perfect() {
      typename warp_map_t::iterator mit = m_tag2entry_map.begin();
      for (; mit != m_tag2entry_map.end(); mit++) {
         delete mit->second;
      }
   }

   // idealistic implementation of lookup: the entry is never aliased, 
   // and a new one is created automatically if it does not exist
   tag2warp_entry_t<Tag>* lookup_pc2warp( const Tag& tag, bool& lut_missed ) {
      typename warp_map_t::iterator mit = m_tag2entry_map.find(tag);

      tag2warp_entry_t<Tag>* lut_entry = NULL;
      if (mit != m_tag2entry_map.end()) {
         lut_entry = mit->second;
         assert(lut_entry->tag == tag);
      } else {
         lut_entry = new tag2warp_entry_t<Tag>();
         m_tag2entry_map.insert(make_pair(tag, lut_entry));
      }

      lut_missed = false;
      lut_max_size = (lut_max_size < m_tag2entry_map.size())? m_tag2entry_map.size() : lut_max_size;

      return lut_entry;
   }

   void invalidate_entry( tag2warp_entry_t<Tag>* lut_entry, int warp_idx ) {
      if (lut_entry == NULL) return;
      if (lut_entry->idx != warp_idx) return;

      typename warp_map_t::iterator mit = m_tag2entry_map.find(lut_entry->tag);
      if (mit != m_tag2entry_map.end()) {
         assert(mit->second == lut_entry);
         mit->second->reset();
         delete mit->second;
         m_tag2entry_map.erase(mit);
      }
   }

   void clear_accessed( ) {}

   void print( FILE* fout) {
      typename warp_map_t::iterator mit = m_tag2entry_map.begin();
      for (; mit != m_tag2entry_map.end(); mit++) {
         mit->second->print(fout);
      }
   }

   static void print_stats ( FILE* fout ) {
      fprintf( fout, "lut_max_size = %d\n", lut_max_size);
   }
};
template <class Tag> unsigned int warp_lut_perfect<Tag>::lut_max_size = 0;


typedef tag2warp_entry_t<pc_tag> warplut_entry_t;
typedef pc_tag warp_tag_t;

class dwf_hw_sche_class {
public:
   int m_id;
   warp_lut<pc_tag> *warp_lut_pc;
   vector<warp_entry_t> warp_pool;
   deque<int> free_warp_q; // the warp allocator
   int simd_width;
   int regf_width;
   int insn_size_lgb2;
   bool just_resume;

   vector<char> m_req;  // request vector from incoming warp
   vector<char> m_occ_new; // occupancy vector of the new warp, double as conflict vector
   vector<char> m_occ_upd; // occupancy vector of the updated existing warp
   vector<char> m_occ_ext; // occupancy vector of the existing warp

   dwf_hw_sche_class( int lut_size, int lut_assoc, 
                      int simd_width, int regf_width, 
                      int n_threads, int insn_size, 
                      int heuristic, int id, 
                      char *policy_opt = NULL );
   ~dwf_hw_sche_class();

   warplut_entry_t* lookup_pc2warp( const warp_tag_t& lookup_tag );
   int update_warp( int* tid, address_type pc );

   // barrier handling
   int m_nbarriers; 
   class dwf_barrier {
   public:
      bool m_release; // see if a barrier is to be released (ie. all warp in cta hit already)
      deque<int> m_queue; // queue storing warps currently hitting a barrier, skipping warplut and scheduler

      dwf_barrier() : m_release(false) {}
      dwf_barrier(const dwf_barrier& that) 
      : m_release(that.m_release), m_queue(that.m_queue) {}
      bool ready_to_issue() {
         return(m_release && !m_queue.empty());
      }
   };
   set< int > m_cta_released_barrier; // set of cta with released barrier 
   map< int, dwf_barrier > m_barrier; // map <barrier id == cta id, barrier>
   int update_warp_at_barrier( int* tid, address_type pc, int cta_id, int barrier_num = 0 );
   void hit_barrier( int cta_id, int barrier_num = 0 );
   void release_barrier( int cta_id, int barrier_num = 0 );

   int allocate_warp( address_type pc, bool update_scheduler = true );
   void free_warp( int idx, bool update_warplut = true );

   void issue_warp( int *tid, address_type *pc );

   void clear_accessed( ) {
      warp_lut_pc->clear_accessed();
   }

   void init_cta(int start_thread, int cta_size, address_type start_pc);

   void print_pc2warp_lut( FILE *fout );
   void print_warp_pool( FILE *fout );
   void print_free_warp_q( FILE *fout );

   int heuristic;

   // FIFO warp issue logic
   queue<int> issue_warp_FIFO_q; 

   // PC warp issue logic
   class pc_first {
   public:
      vector<warp_entry_t> &warp_pool;
      pc_first( vector<warp_entry_t> &bp ) : warp_pool(bp) {}
      bool operator() (const int &idx_a, const int &idx_b) const {
         if (warp_pool[idx_a].pc != warp_pool[idx_b].pc) {
            return(warp_pool[idx_a].pc > warp_pool[idx_b].pc);
         } else {
            return(warp_pool[idx_a].occ < warp_pool[idx_b].occ);
         }
      }
   };
   pc_first mypc_first;
   priority_queue<int, vector<int>, pc_first > issue_warp_PC_q;

   // Majority warp issue logic
   issue_warp_majority *issue_warp_MAJ;
   void clear_policy_access( );
   void reset_policy_access( );

   // PDOM Priority issue logic
   issue_warp_pdom_prio issue_warp_pdom;

   // statistics
   npc_tracker_class npc_tracker;
   int max_warppool_occ;
   int *warppool_occ_histo; // histogram of warppool occupancy
   static unsigned int lut_realmiss;
   static unsigned int uid_cnt;
   static unsigned int warp_fragmentation;
   static unsigned int warp_merge_conflict;
   static void print_stats ( FILE* fout ) {
      warp_lut_perfect<warp_tag_t>::print_stats( fout );
      warp_lut_sa<warp_tag_t>::print_stats( fout );
      fprintf( fout, "lut_realmiss = %d\n", lut_realmiss);
      fprintf( fout, "warp_fragmentation = %d\n", warp_fragmentation);
      fprintf( fout, "warp_merge_conflict = %d\n", warp_merge_conflict);
   }
};

unsigned int dwf_hw_sche_class::lut_realmiss = 0;
unsigned int dwf_hw_sche_class::uid_cnt = 0;
unsigned int dwf_hw_sche_class::warp_fragmentation = 0;
unsigned int dwf_hw_sche_class::warp_merge_conflict = 0;


dwf_hw_sche_class::dwf_hw_sche_class( int lut_size, int lut_assoc, 
                                      int simd_width, int regf_width,
                                      int n_threads, int insn_size, 
                                      int heuristic, int id,
                                      char *policy_opt ) 
: m_id(id), 
// WarpLUT w/ pc tag
warp_lut_pc( (lut_size == 0)? (warp_lut<pc_tag> *) new warp_lut_perfect<pc_tag>() : 
             (warp_lut<pc_tag> *) new warp_lut_sa<pc_tag>(lut_size, lut_assoc, insn_size) ), 
m_nbarriers(1), // for barrier
mypc_first( warp_pool ), issue_warp_PC_q( mypc_first ),   // DPC
issue_warp_pdom(simd_width, &warp_pool, n_threads),       // DPdPri
npc_tracker( NULL, simd_width )
{
   unsigned i;

   this->simd_width = simd_width;
   this->regf_width = regf_width;
   this->m_req.resize(regf_width);
   this->m_occ_new.resize(regf_width);
   this->m_occ_upd.resize(regf_width);
   this->m_occ_ext.resize(regf_width);

   // initialize the warp pool 
   // (make sure the thread id's are init to -1)
   this->warp_pool.resize(n_threads);
   for (i=0; i<warp_pool.size(); i++) {
      warp_pool[i].pc = -1;
      warp_pool[i].tid = new int[simd_width];
      memset(warp_pool[i].tid, -1, sizeof(int)*simd_width);
      warp_pool[i].occ = 0;
      warp_pool[i].next_warp = -1;

      // push the index to the warp allocator
      free_warp_q.push_back(i);
   }

   // setup for various heuristics
   this->heuristic = heuristic;
   switch (heuristic) {
   case MAJORITY: 
      issue_warp_MAJ = new issue_warp_majority_queue(simd_width, &warp_pool);
      break;
   case MAJORITY_MAXHEAP: {
         int mh_lut_size = 32;
         int mh_lut_assoc = 4;
         int n_reads_per_cycle_lut = 4;
         int n_writes_per_cycle_lut = 4;
         int mh_size = 128;
         int n_reads_per_cycle_mh = 4;
         int n_writes_per_cycle_mh = 4;
         if (policy_opt != NULL) {
            sscanf(policy_opt, ";LUT=%d:%dr%dw%d;MH=%dr%dw%d", 
                   &mh_lut_size, &mh_lut_assoc, &n_reads_per_cycle_lut, &n_writes_per_cycle_lut,
                   &mh_size, &n_reads_per_cycle_mh, &n_writes_per_cycle_mh);
         }
         issue_warp_MAJ = new issue_warp_majority_heap(simd_width, &warp_pool, 
                                                       mh_lut_size, mh_lut_assoc, mh_size,
                                                       n_reads_per_cycle_lut, n_writes_per_cycle_lut,
                                                       n_reads_per_cycle_mh, n_writes_per_cycle_mh);
      }
      break;
   }

   this->just_resume = false;

   this->max_warppool_occ = 0;
   this->warppool_occ_histo = new int[n_threads];
   memset(this->warppool_occ_histo, 0, n_threads*sizeof(int));
}

// should never be called (only at exit?)
dwf_hw_sche_class::~dwf_hw_sche_class( )
{
   unsigned i;

   for (i=0; i<warp_pool.size(); i++) {
      free(warp_pool[i].tid);
   }

   delete[] this->warppool_occ_histo;

   delete warp_lut_pc;
}

// allocate a new warp in warp pool
int dwf_hw_sche_class::allocate_warp( address_type pc, bool update_scheduler )
{
   int idx;
   assert(!free_warp_q.empty());
   idx = free_warp_q.front();
   free_warp_q.pop_front();
   warp_pool[idx].uid = uid_cnt;
   uid_cnt++;
   warp_pool[idx].pc = pc;
   warp_pool[idx].next_warp = -1;
   warp_pool[idx].lut_ptr = NULL;

   if (update_scheduler) {
      if (heuristic == FIFO) issue_warp_FIFO_q.push(idx);
      if (heuristic == PC) issue_warp_PC_q.push(idx);
      if (heuristic == MAJORITY || heuristic == MAJORITY_MAXHEAP)
         issue_warp_MAJ->push_warp(pc, idx);
      if (heuristic == PDOMPRIO) issue_warp_pdom.push_warp(pc, idx);
   }

   return idx;
}

// free a warp in warp pool
// it will reset the content of the warp entry as well
void dwf_hw_sche_class::free_warp( int idx, bool update_warplut ) 
{
   bool redundant_idx_check = false;
   if (redundant_idx_check) {
      deque<int>::iterator dit = find(free_warp_q.begin(), free_warp_q.end(), idx);
      assert(dit == free_warp_q.end());
   }

   warp_pool[idx].pc = -1;
   memset(warp_pool[idx].tid, -1, sizeof(int)*simd_width);
   warp_pool[idx].occ = 0;
   warp_pool[idx].next_warp = -1;
   if (update_warplut) {
      warp_lut_pc->invalidate_entry( (warplut_entry_t*)warp_pool[idx].lut_ptr, idx );
   }

   free_warp_q.push_back(idx);
   assert(free_warp_q.size() <= warp_pool.size());
}

warplut_entry_t* dwf_hw_sche_class::lookup_pc2warp( const warp_tag_t& lookup_tag )
{
   bool lut_missed = false;

   warplut_entry_t* lut_entry;
   lut_entry = warp_lut_pc->lookup_pc2warp( lookup_tag, lut_missed );

   if (!lut_missed) {
      if (lut_entry->tag != warp_pool[lut_entry->idx].pc) lut_missed = true;
   }

   if (lut_missed) {
      if (npc_tracker.pc_count.find(lookup_tag.get_pc()) != npc_tracker.pc_count.end()) {
         lut_realmiss++; // ie. the incoming warp lost an opportunity to merge
      }
   }

   return lut_entry;
}   


void fill_all (vector<char>& container, const char& value)
{
   fill(container.begin(), container.end(), value);
}

int regfile_hash(signed istream_number, unsigned simd_size, unsigned n_banks);
int dwf_hw_sche_class::update_warp( int *tid, address_type pc )
{
   int i;
   bool newwarp = false;
   bool newwarp_alloc = false;
   warplut_entry_t* lut_entry;
   warp_tag_t warp_tag(pc);
   lut_entry = lookup_pc2warp(warp_tag);

   // no LUT entry returned, stall
   if (!lut_entry) {
      assert(0);
   }

   if (heuristic == MAJORITY || heuristic == MAJORITY_MAXHEAP) {
      issue_warp_MAJ->add_threads(pc, tid);
   }

   npc_tracker.add_threads( tid, pc );

   // if the pc of the LUT entry does not match,
   // allocate a new entry
   if (lut_entry->tag != warp_tag) {
      lut_entry->idx = allocate_warp(pc);
      lut_entry->tag = warp_tag;
      lut_entry->occ = 0;
      assert(warp_pool[lut_entry->idx].pc == pc);
      newwarp = true;
      newwarp_alloc = true;
   }

   // create the request vector
   bool tid_has_valid_entry = false;
   fill_all(m_req, 0);
   for (i = 0; i<simd_width; i++) {
      if (tid[i] != -1) {
         int lane = regfile_hash(tid[i],simd_width,regf_width);
         // make sure we are not having two threads going to same lane
         assert(lane < regf_width);
         m_req[lane] += 1;
         tid_has_valid_entry = true;
      }
   }
   assert(tid_has_valid_entry);

   // read the old idx pointing to an existing warp
   int old_idx = lut_entry->idx;

   // create the conflict vector
   fill_all(m_occ_ext, 0);
   int regf_mask = regf_width - 1;
   for (i = 0; i<simd_width; i++) {
      m_occ_ext[i & regf_mask] += ((lut_entry->occ & (1 << i)) == 0)? 0 : 1;
   }
   fill_all(m_occ_upd, 0);
   fill_all(m_occ_new, 0);
   int n_regf_slot = simd_width / regf_width;
   bool conflict = false;
   for (i = 0; i<regf_width; i++) {
      if (m_occ_ext[i] + m_req[i] > n_regf_slot) {
         m_occ_new[i] = m_occ_ext[i] + m_req[i] - n_regf_slot;
         m_occ_upd[i] = n_regf_slot - m_occ_ext[i];
         conflict = true;
      } else {
         m_occ_upd[i] = m_req[i];
      }
   }

   // if the pc of the warp mismatch with lut, 
   // set conflict vector to all one. 
   // that force all threads to the newly allocated warp
   if (warp_pool[old_idx].pc != pc) {
      conflict = true;
      for (i = 0; i<regf_width; i++) {
         m_occ_new[i] = m_req[i];
         m_occ_upd[i] = 0;
         m_occ_ext[i] = n_regf_slot;
      }
   }

   // if there are conflicted entries, get a new warp
   int new_idx = -1;
   if (conflict) {
      new_idx = allocate_warp(pc);
      lut_entry->idx = new_idx;
      lut_entry->occ = 0; //update the lut_entry
      assert(warp_pool[new_idx].pc == pc);

      int total_occ = 0;
      for (i = 0; i < regf_width; i++)
         total_occ += m_occ_ext[i] + m_req[i];
      if (total_occ <= simd_width) warp_fragmentation += 1;
      warp_merge_conflict += 1;

      newwarp_alloc = true;
   }

   // update the warp as indicated by the LUT
   // if the lane is conflicted, or the old warp is just not 
   // write to the new warp 
   int new_occ = 0;
   fill_all(m_occ_new, 0);
   for (i = 0; i<simd_width; i++) {
      if (tid[i] != -1) {
         int rfbank = regfile_hash(tid[i],simd_width,regf_width);
         int lane = -1;
         if ((m_occ_ext[rfbank] < n_regf_slot) || newwarp) {
            lane = rfbank + m_occ_ext[rfbank] * regf_width;
            assert(lane < simd_width);
            warp_pool[old_idx].tid[lane] = tid[i];
            warp_pool[old_idx].occ++;
            lut_entry->occ |= (1<<lane);
            m_occ_ext[rfbank]++;
         } else {
            lane = rfbank + m_occ_new[rfbank] * regf_width;
            assert(lane < simd_width);
            warp_pool[new_idx].tid[lane] = tid[i];
            warp_pool[new_idx].occ++;
            new_occ |= (1<<lane);
            m_occ_new[rfbank]++;
            assert(m_occ_new[rfbank] <= n_regf_slot);
         }
      }
   }

   // to cover the case where the pc of the warp mismatch with lut 
   // (because the warp is issued) 
   if (warp_pool[old_idx].pc == pc) {
      issue_warp_pdom.add_threads(old_idx, pc);
   }
   if (conflict) {
      lut_entry->occ = new_occ;
      issue_warp_pdom.add_threads(new_idx, pc);
   }

   warp_pool[lut_entry->idx].lut_ptr = lut_entry; // link up the lut entry and warp 

   bool scheduler_consistency_check = false;
   if (scheduler_consistency_check && heuristic == MAJORITY) {
      ((issue_warp_majority_queue*)issue_warp_MAJ)->check_consistency();
   }

   return 1;
}

// called AFTER threads hit a barrier to insert them into the barrier queue
// ASSUME: threads from released barrier are not hitting second barrier right away
int dwf_hw_sche_class::update_warp_at_barrier( int* tid, address_type pc, int cta_id, int barrier_num )
{
   assert(barrier_num < m_nbarriers);
   assert(cta_id >= 0);

   int i;
   int warp_index = 0xDEADBEEF;

   npc_tracker.add_threads( tid, pc );

   // always allocate new warp
   warp_index = allocate_warp(pc, false);
   assert(warp_pool[warp_index].pc == pc);

   // no need to create the request vector
   // no need to create the conflict vector

   // assign threads into the new warp
   fill_all(m_occ_ext, 0);
   int max_nthreads_per_rfbank = simd_width / regf_width;
   for (i = 0; i<simd_width; i++) {
      if (tid[i] != -1) {
         int rfbank = regfile_hash(tid[i],simd_width,regf_width);
         int lane = -1;

         assert(m_occ_ext[rfbank] < max_nthreads_per_rfbank);
         lane = rfbank + m_occ_ext[rfbank] * regf_width;
         assert(lane < simd_width);
         warp_pool[warp_index].tid[lane] = tid[i];
         warp_pool[warp_index].occ++;
         m_occ_ext[rfbank]++;
      }
   }

   warp_pool[warp_index].lut_ptr = NULL; // no link to any lut entry 

   // put the warp id into barrier queue
   m_barrier[cta_id].m_queue.push_back(warp_index);

   // notify issue module to check this barrier at issue
   if ( m_barrier[cta_id].ready_to_issue() ) {
      m_cta_released_barrier.insert(cta_id);
   }

   return 1;
}

// called at decode stage when thread hit a barrier
// ASSUME: threads from released barrier are not hitting second barrier right away
void dwf_hw_sche_class::hit_barrier( int cta_id, int barrier_num )
{
   assert(barrier_num < m_nbarriers);
   assert(cta_id >= 0);

   m_barrier[cta_id].m_release = false;
}

// called at decode stage when all thread in cta hit the barrier
// ASSUME: threads from released barrier are not hitting second barrier right away
void dwf_hw_sche_class::release_barrier( int cta_id, int barrier_num )
{
   assert(barrier_num < m_nbarriers);
   assert(cta_id >= 0);

   map<int, dwf_barrier>::iterator i_barrier = m_barrier.find(cta_id);
   assert(i_barrier != m_barrier.end()); // barrier has to exists in the first place!
   i_barrier->second.m_release = true;
}

void dwf_hw_sche_class::issue_warp( int *tid, address_type *pc )
{
   int i;
   bool warp_issued = false;

   // scan the released barriers for ready warp
   // TODO: arbitrate between different queues?
   set<int>::iterator i_ctabar = m_cta_released_barrier.begin();
   for (; i_ctabar != m_cta_released_barrier.end(); ++i_ctabar) {
      int cta_id = *i_ctabar;
      map<int, dwf_barrier>::iterator i_barrier = m_barrier.find(cta_id);

      if ( i_barrier->second.ready_to_issue() ) {
         int warp_idx = i_barrier->second.m_queue.front();

         for (i = 0; i < simd_width; i++) {
            tid[i] =  warp_pool[warp_idx].tid[i];
         }
         *pc = warp_pool[warp_idx].pc;

         i_barrier->second.m_queue.pop_front();
         free_warp(warp_idx, false); // don't update warplut as the warp is not linked to it

         // remove cta from checking list if the queue is emptied 
         // (if the last threads haven't made it back to scheduler in time, 
         //  update_warp_at_barrier will insert the cta id again)
         if (i_barrier->second.m_queue.empty()) {
            m_cta_released_barrier.erase(i_ctabar); 
         }

         warp_issued = true;

         break;
      }
   }

   if (!warp_issued) {
      switch (heuristic) {
      case FIFO:
         // Oldest warp are issued first
         if (!issue_warp_FIFO_q.empty()) {
            int idx = issue_warp_FIFO_q.front();
            for (i = 0; i < simd_width; i++) {
               tid[i] =  warp_pool[idx].tid[i];
            }
            *pc = warp_pool[idx].pc;

            issue_warp_FIFO_q.pop();
            free_warp(idx);
         } else {
            memset(tid, -1, sizeof(int)*simd_width);
            *pc = -1;
         }
         break;
      case PC:
         // lowest PC warp are issued first
         if (!issue_warp_PC_q.empty()) {
            int idx = issue_warp_PC_q.top();
            for (i = 0; i < simd_width; i++) {
               tid[i] =  warp_pool[idx].tid[i];
            }
            *pc = warp_pool[idx].pc;

            issue_warp_PC_q.pop();
            free_warp(idx);
         } else {
            memset(tid, -1, sizeof(int)*simd_width);
            *pc = -1;
         }
         break;
      case MAJORITY:
      case MAJORITY_MAXHEAP:
         // issue the most common PC first
         {
            int idx = issue_warp_MAJ->pop_warp();
            if (idx >= 0) {
               for (i = 0; i < simd_width; i++) {
                  tid[i] =  warp_pool[idx].tid[i];
               }
               *pc = warp_pool[idx].pc;
               free_warp(idx);
            } else {
               memset(tid, -1, sizeof(int)*simd_width);
               *pc = -1;
            }
         }
         break;
      case PDOMPRIO:
         // issue the warp with lowest PDOM count
         {
            int idx = issue_warp_pdom.front_warp();
            if (idx >= 0) {
               issue_warp_pdom.pop_warp();

               for (i = 0; i < simd_width; i++) {
                  tid[i] =  warp_pool[idx].tid[i];
               }
               *pc = warp_pool[idx].pc;
               free_warp(idx);

               just_resume = false;
            } else {
               memset(tid, -1, sizeof(int)*simd_width);
               *pc = -1;
            }
         }
         break;
      default:
         printf("Unsupported Heuristics!\n");
         abort();
         break;
      }
   }

   npc_tracker.sub_threads( tid, *pc );

   int warppool_occ = warp_pool.size() - free_warp_q.size();
   if (max_warppool_occ < warppool_occ) {
      max_warppool_occ = warppool_occ;
   }
   warppool_occ_histo[warppool_occ] += 1;
}

void dwf_hw_sche_class::init_cta(int start_thread, int cta_size, address_type start_pc)
{
   assert((start_thread % simd_width) == 0); // thread id starting at a warp

   int n_warp_2assign = cta_size / simd_width;
   n_warp_2assign += (cta_size % simd_width)? 1 : 0; // round up

   static int *thd_id = NULL;
   if (thd_id == NULL) thd_id = new int[simd_width];

   for (int w = 0; w < n_warp_2assign; w++) {
      // generate the warp update register for each warp
      fill_n(thd_id, simd_width, -1);
      int warp_start_tid = start_thread + w * simd_width;
      for (int i = 0; (i < simd_width) && (warp_start_tid + i) < (start_thread + cta_size); i++) {
         thd_id[i] = warp_start_tid + i;
      }

      // push these warps into DWF scheduler
      update_warp( thd_id, start_pc );
   }
}

void dwf_hw_sche_class::print_free_warp_q( FILE *fout )
{
   fprintf(fout, "free_node_q (%zd)= ", free_warp_q.size() );
   deque<int>::iterator dit = free_warp_q.begin();
   for (; dit != free_warp_q.end(); dit++) {
      fprintf(fout, "%03d ", *dit);
   }
   fprintf(fout, "\n");
}

void print_warp( FILE *fout, warp_entry_t warp_e, int simd_width ) 
{
   fprintf(fout, "\t%02d 0x%08x: (", warp_e.pdom_prio, warp_e.pc );
   for (int i=0;i<simd_width;i++) {
      fprintf(fout, "%03d ", warp_e.tid[i]);
   }
   fprintf(fout, ")\n");
}

void dwf_hw_sche_class::print_warp_pool( FILE *fout )
{
   for (unsigned i=0; i< warp_pool.size(); i++) {
      if (warp_pool[i].pc != (address_type)-1) {
         fprintf(fout, "bp%03d:", i);
         print_warp(fout, warp_pool[i], simd_width);
      }
   }
}

void dwf_hw_sche_class::clear_policy_access( ) {
   if (heuristic == MAJORITY_MAXHEAP) {
      ((issue_warp_majority_heap*)issue_warp_MAJ)->clear_access( );
   }
}

void dwf_hw_sche_class::reset_policy_access( ) {
   if (heuristic == MAJORITY_MAXHEAP) {
      ((issue_warp_majority_heap*)issue_warp_MAJ)->reset_access( );
   }
}

///////////////////////////////////////////////////////////////////////////
// c-wrapper interface
///////////////////////////////////////////////////////////////////////////

int dwf_hw_n_sche = 0;
dwf_hw_sche_class **dwf_hw_sche;
unsigned *acc_dyn_pcs = NULL;

void create_dwf_schedulers( int n_shaders, 
                                       int lut_size, int lut_assoc, 
                                       int simd_width, int regf_width,
                                       int n_threads, int insn_size, 
                                       int heuristic,
                                       char *policy_opt )
{
   dwf_hw_n_sche = n_shaders;
   dwf_hw_sche = new dwf_hw_sche_class*[n_shaders];
   for (int i=0; i<n_shaders; i++) {
      dwf_hw_sche[i] = new dwf_hw_sche_class( lut_size, lut_assoc, 
                                              simd_width, regf_width,
                                              n_threads, insn_size, 
                                              heuristic, i, 
                                              policy_opt );
   }

   if (acc_dyn_pcs == NULL) {
      acc_dyn_pcs = new unsigned[n_shaders];
      std::fill_n(acc_dyn_pcs, n_shaders, 0);
   }
   for (int i=0; i<n_shaders; i++) {
      dwf_hw_sche[i]->npc_tracker.acc_pc_count = &acc_dyn_pcs[i];
   }   
}

int dwf_update_warp( int shd_id, int* tid, address_type pc )
{
   return dwf_hw_sche[shd_id]->update_warp( tid, pc );
}

int dwf_update_warp_at_barrier( int shd_id, int* tid, address_type pc, int cta_id )
{
   return dwf_hw_sche[shd_id]->update_warp_at_barrier( tid, pc, cta_id);
}

void dwf_hit_barrier( int shd_id, int cta_id )
{
   dwf_hw_sche[shd_id]->hit_barrier( cta_id );
}

void dwf_release_barrier( int shd_id, int cta_id )
{
   dwf_hw_sche[shd_id]->release_barrier( cta_id );
}

void dwf_issue_warp( int shd_id, int *tid, address_type *pc )
{
   dwf_hw_sche[shd_id]->issue_warp( tid, pc );
}

void dwf_clear_accessed( int shd_id )
{
   dwf_hw_sche[shd_id]->clear_accessed( );
}

void dwf_clear_policy_access( int shd_id )
{
   dwf_hw_sche[shd_id]->clear_policy_access( );
}

void dwf_reset_policy_access( int shd_id )
{
   dwf_hw_sche[shd_id]->reset_policy_access( );
}

void dwf_init_CTA(int shd_id, int start_thread, int cta_size, address_type start_pc)
{
   dwf_hw_sche[shd_id]->init_cta(start_thread, cta_size, start_pc);
   dwf_hw_sche[shd_id]->clear_accessed( );
   dwf_hw_sche[shd_id]->clear_policy_access( );
}

void dwf_print_stat( FILE* fout )
{
   dwf_hw_sche_class::print_stats( fout );
   npc_tracker_class::histo_print( fout );
   fprintf(fout, "max_warppool_occ = ");
   for (int i=0; i<dwf_hw_n_sche; i++) {
      fprintf(fout, "%d ", dwf_hw_sche[i]->max_warppool_occ);
   }
   fprintf(fout, "\n");
   for (int i=0; i<dwf_hw_n_sche; i++) {
      fprintf(fout, "warppool_occ[%d] = ", i);
      for (int j=0; j<dwf_hw_sche[i]->max_warppool_occ; j++) {
         fprintf(fout, "%d ", dwf_hw_sche[i]->warppool_occ_histo[j]);
      }
      fprintf(fout, "\n");
   }
   if (dwf_hw_sche[0]->heuristic == MAJORITY_MAXHEAP) {
      fprintf(fout, "n_stall_on_maxheap = ");
      for (int i=0; i<dwf_hw_n_sche; i++) {
         fprintf(fout, "%d ", 
                 ((issue_warp_majority_heap*)dwf_hw_sche[i]->issue_warp_MAJ)->n_stall_on_maxheap);
      }
      fprintf(fout, "\n");
      fprintf(fout, "maxheap_n_entries = ");
      for (int i=0; i<dwf_hw_n_sche; i++) {
         fprintf(fout, "%d ", 
                 ((issue_warp_majority_heap*)dwf_hw_sche[i]->issue_warp_MAJ)->maxheap.max_n_entries);
      }
      fprintf(fout, "\n");
      fprintf(fout, "maxheap_lut_n_aliased = ");
      for (int i=0; i<dwf_hw_n_sche; i++) {
         fprintf(fout, "%d ", 
                 ((issue_warp_majority_heap*)dwf_hw_sche[i]->issue_warp_MAJ)->mh_lut.n_aliased);
      }
      fprintf(fout, "\n");
      issue_warp_majority_heap::print_stat(fout);
   }
}

void dwf_reset_reconv_pt() 
{
   issue_warp_pdom_prio::reconvgence_pt.clear();
}

void dwf_insert_reconv_pt(address_type pc) 
{
   issue_warp_pdom_prio::reconvgence_pt.insert(pc);
}

void dwf_reinit_schedulers( int n_shaders )
{
   for (int i=0; i<n_shaders; i++) {
      dwf_hw_sche[i]->issue_warp_pdom.reinit();
   }
}

void dwf_update_statistics( int shader_id )
{
   dwf_hw_sche[shader_id]->npc_tracker.update_acc_count();
}

void g_print_dmaj_scheduler(int sid) {
   dwf_hw_sche[sid]->issue_warp_MAJ->print(stdout);
}

void g_print_warp_lut(int sid) {
   dwf_hw_sche[sid]->warp_lut_pc->print(stdout);
}

void g_print_free_warp_q(int sid) {
   dwf_hw_sche[sid]->print_free_warp_q(stdout);
}

void g_print_warp_pool(int sid) {
   dwf_hw_sche[sid]->print_warp_pool(stdout);
}

void g_print_max_heap(int sid) {
   dwf_hw_sche[sid]->issue_warp_MAJ->print(stdout);
}

#ifdef UNIT_TEST

   #undef UNIT_TEST
   #include "stat-tool.cc"

unsigned gpgpu_thread_swizzling = 0;
unsigned long long  gpu_sim_cycle = 0;

int regfile_hash(signed istream_number, unsigned simd_size, unsigned n_banks) {
   if (gpgpu_thread_swizzling) {
      signed warp_ID = istream_number / simd_size;
      return((istream_number + warp_ID) % n_banks);
   } else {
      return(istream_number % n_banks);
   }
}

int log2i(int n) {
   int lg;
   lg = -1;
   while (n) {
      n>>=1;lg++;
   }
   return lg;
}

int test_FIFO() 
{
   dwf_hw_sche_class *dwf_sche;
   int i;
   int tid[6][4] = {
      { 0, 1, 2, 3},
      { 4, 5, 6, 7},
      { 8,-1,10,-1},
      {-1, 1,-1, 3},
      { 4, 9,-1,11},
      {-1,13,14,-1}
   };

   int expect_out[12][4] = {
      { 0, 1, 2, 3},
      { 0, 1, 2, 3},
      { 0, 1, 2, 3},
      { 4, 5, 6, 7},
      { 8, 1,10, 3},
      { 4, 9,14,11},
      {-1,13,-1,-1},
      { 4, 9,-1,11},
      {-1,13,14,-1},
      { 8,-1,10,-1},
      { 4, 9,14,11},
      { 8,13,10,-1}
   };

   int tid_out[4];
   address_type pc_out;

   dwf_sche = new dwf_hw_sche_class(16, 2, 4, 4, 16, 1, FIFO);

   // same threads - different pc
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[0], 0x409a80);
   dwf_sche->update_warp(tid[0], 0x409a88);

   // different threads - different pc
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[0], 0x409a90);
   dwf_sche->update_warp(tid[1], 0x409a80);

   // different threads - same pc 
   // expect two warp to merge into one as there is no lane conflict
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[2], 0x409a90);
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[3], 0x409a90);

   // same as above, but with lane conflict
   // expect a new warp allocated, 
   // but only the conflicting threads goes to new warp
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[4], 0x409a80);
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[5], 0x409a80);

   // different threads - different pc
   // purposely try to alias an existing mapping 
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[4], 0x410a80);
   dwf_sche->update_warp(tid[5], 0x411a80);

   // going back to that mapping 
   // a new warp should be allocated (despite lack of conflict)
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[2], 0x409a80);

   // testing the occupancy vector
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[4], 0x409aa0);
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[5], 0x409aa0);
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[2], 0x409aa0);

   // fill the warp pool up
   for (i=12; i<64; ) {
      dwf_sche->clear_accessed();
      dwf_sche->update_warp(tid[1], 0x409a80 + 8 * i++);
      dwf_sche->update_warp(tid[4], 0x409a80 + 8 * i++);
   }
   // issue all the warp (do some auto checking on the way)
   for (i=0; i<64; i++) {
      dwf_sche->issue_warp(tid_out, &pc_out);
      printf("0x%08x [%d %d %d %d]\n", pc_out, tid_out[0], tid_out[1], tid_out[2], tid_out[3]);
      if (i<12) {
         if ( memcmp(tid_out, expect_out[i], 4*sizeof(int) ) ) {
            printf("%d warp mismatches\n", i);
            assert(0);
         }
      }
   }

   // now that all warpes are issue, no entries in the lut is valid
   // updating warp with an old address that remains in the lut
   // to see if detects the invalid lut entry
   dwf_sche->clear_accessed();
   dwf_sche->update_warp(tid[2], 0x409a80 + 8 * 63);
   dwf_sche->update_warp(tid[3], 0x409a80 + 8 * 62);
   dwf_sche->issue_warp(tid_out, &pc_out);
   assert(!memcmp(tid_out, tid[2], 4*sizeof(int) ));
   dwf_sche->issue_warp(tid_out, &pc_out);
   assert(!memcmp(tid_out, tid[3], 4*sizeof(int) ));

   dwf_sche->print_warp_pool(stdout);
   dwf_sche->warp_lut_pc->print(stdout);
   dwf_hw_sche_class::print_stats(stdout);

   delete dwf_sche;

   return 0;
}

int test_PC ()
{
   dwf_hw_sche_class *dwf_sche;
   int i;
   int tid[4][4] = {
      { 0, 1, 2, 3},
      { 4, 5, 6, 7},
      { 8,-1,10,-1},
      {-1,13,14,-1}
   };

   int tid_out[4];
   address_type pc_out;

   dwf_sche = new dwf_hw_sche_class(16, 2, 4, 4, 16, 1, PC);

   // fill the warp pool up in reverse PC order
   for (i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
         dwf_sche->clear_accessed();
         dwf_sche->update_warp(tid[j], 0x409a80 - 8 * i);
      }
   }

   // issue the warps, expect them to be in PC order, with higher occ warp issued first
   printf("PC Issue Logic:\n");
   for (i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
         dwf_sche->issue_warp(tid_out, &pc_out);
         printf("0x%08x [%d %d %d %d]\n", pc_out, tid_out[0], tid_out[1], tid_out[2], tid_out[3]);
      }
   }  

}

int test_MAJ ()
{
   dwf_hw_sche_class *dwf_sche;
   int i;
   int tid[4][4] = {
      { 0, 1, 2, 3},
      { 4, 5, 6, 7},
      { 8,-1,10,-1},
      {-1,13,14,-1}
   };

   int tid_out[4];
   address_type pc_out;

   dwf_sche = new dwf_hw_sche_class(16, 2, 4, 4, 16, 1, MAJORITY);

   // fill the warp pool up in reverse PC order
   for (i=0; i<4; i++) {
      for (int j=0; j<(4-i); j++) {
         dwf_sche->clear_accessed();
         dwf_sche->update_warp(tid[j], 0x409a80 - 8 * i);
      }
   }

   // issue the warps, expect them to be in PC order, with higher occ warp issued first
   printf("Majority Issue Logic:\n");
   for (i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
         dwf_sche->issue_warp(tid_out, &pc_out);
         printf("0x%08x [%d %d %d %d]\n", pc_out, tid_out[0], tid_out[1], tid_out[2], tid_out[3]);
      }
   }  
}

int test_MAJ_HEAP ()
{
   printf("\ntest_MAJ_HEAP:\n");
   dwf_hw_sche_class *dwf_sche;
   int i;
   int tid[4][4] = {
      { 0, 1, 2, 3},
      { 4, 5, 6, 7},
      { 8,-1,10,-1},
      {-1,13,14,-1}
   };

   int tid_out[4];
   address_type pc_out;

   dwf_sche = new dwf_hw_sche_class(16, 2, 4, 4, 16, 1, MAJORITY_MAXHEAP);

   // fill the warp pool up in reverse PC order
   for (i=0; i<4; i++) {
      for (int j=0; j<(i+1); j++) {
         dwf_sche->clear_accessed();
         dwf_sche->update_warp(tid[j], 0x409a80 + 8 * i);
      }
   }

   dwf_sche->reset_policy_access();
   dwf_sche->issue_warp_MAJ->print(stdout);

   // issue the warps, expect them to be in PC order, with higher occ warp issued first
   printf("Majority (Max Heap) Issue Logic:\n");
   for (i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
         dwf_sche->issue_warp(tid_out, &pc_out);
         printf("0x%08x [%d %d %d %d]\n", pc_out, tid_out[0], tid_out[1], tid_out[2], tid_out[3]);
      }
      dwf_sche->reset_policy_access();
   }  
}

void test_warp_lut_pc ()
{
   printf("\ntest_warp_lut_pc:\n");
   warp_lut_sa<pc_tag> warp_lut_pc(16,   // size
                                   4,   // assoc
                                   1);  // insn_size

   address_type pc_value[] = {0, 4, 0, 8, 12, 16, 20, 8, 8, 0};
   int n_entry = sizeof(pc_value) / sizeof(address_type);
   vector<pc_tag> pc_stream(pc_value, pc_value + n_entry);

   int misses = 0;
   for (int n = 0; n < n_entry * 100; n++) {
      int i = n % n_entry;
      tag2warp_entry_t<pc_tag> *lut_entry = NULL;
      bool lut_miss = false;

      lut_entry = warp_lut_pc.lookup_pc2warp(pc_stream[i], lut_miss);

      if (lut_entry->tag != pc_stream[i]) {
         lut_entry->tag = pc_stream[i];
         lut_entry->occ = 1;
         misses += 1;
      }
      warp_lut_pc.clear_accessed();
      lut_entry->accessed = 0;
   }

   printf("Number of Miss = %d\n", misses);
}

int main () {
   //test_FIFO();
   //test_PC();
   //test_MAJ();
   test_MAJ_HEAP();
   test_warp_lut_pc();
   return 0;
}

#endif
