/* 
 * shader.c
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Ivan Sham, Henry Wong, Dan O'Connor, Henry Tran and the 
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

#include <float.h>
#include "shader.h"
#include "gpu-sim.h"
#include "addrdec.h"
#include "dram.h"
#include "dwf.h"
#include "warp_tracker.h"
#include "stat-tool.h"
#include "gpu-misc.h"
#include "../cuda-sim/ptx_sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/dram_callback.h"
#include "../cuda-sim/cuda-sim.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "visualizer.h"
#include <string.h>
#include <limits.h>

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a,b) (((a)>(b))?(a):(b))

extern bool gpgpu_stall_on_use;
enum mem_stage_access_type {
   C_MEM,
   T_MEM,
   S_MEM,
   G_MEM_LD,
   L_MEM_LD,
   G_MEM_ST,
   L_MEM_ST,
   N_MEM_STAGE_ACCESS_TYPE
};

enum mem_stage_stall_type {
   NO_RC_FAIL = 0, 
   BK_CONF,
   MSHR_RC_FAIL,
   ICNT_RC_FAIL,
   COAL_STALL,
   WB_ICNT_RC_FAIL,
   WB_CACHE_RSRV_FAIL,
   N_MEM_STAGE_STALL_TYPE
};

unsigned int gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE][N_MEM_STAGE_STALL_TYPE] = { {0} };
unsigned warp_size = 4; 
int pipe_simd_width;
unsigned int *shader_cycle_distro;
unsigned int g_waiting_at_barrier = 0;
unsigned int gpgpu_shmem_size;
unsigned int gpgpu_shader_registers;
unsigned int gpgpu_shader_cta;
unsigned int gpgpu_n_load_insn = 0;
unsigned int gpgpu_n_store_insn = 0;
unsigned int gpgpu_n_shmem_insn = 0;
unsigned int gpgpu_n_tex_insn = 0;
unsigned int gpgpu_n_const_insn = 0;
unsigned int gpgpu_n_param_insn = 0;
unsigned int gpgpu_multi_unq_fetches = 0;
bool         gpgpu_shmem_bkconflict;
unsigned int gpgpu_n_shmem_bkconflict = 0;
int          gpgpu_n_shmem_bank = 16;
bool         gpgpu_cache_bkconflict;
unsigned int gpgpu_n_cache_bkconflict = 0;
unsigned int gpgpu_n_cmem_portconflict = 0;
int          gpgpu_n_cache_bank;
int          gpgpu_warpdistro_shader;
int          gpgpu_interwarp_mshr_merge;
int          gpgpu_n_intrawarp_mshr_merge = 0;
int          gpgpu_n_partial_writes = 0;
int          gpgpu_shmem_port_per_bank;
int          gpgpu_cache_port_per_bank;
int          gpgpu_const_port_per_bank;
int          gpgpu_shmem_pipe_speedup;
unsigned int gpu_max_cta_per_shader = 8;
unsigned int gpu_padded_cta_size = 32;
int          gpgpu_local_mem_map;

/////////////////////////////////////////////////////////////////////////////
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/

static const char* MSHR_Status_str[] = {
   "INITIALIZED",
   "IN_ICNT2MEM",
   "IN_ICNTOL2QUEUE",
   "IN_L2TODRAMQUEUE",
   "IN_DRAM_REQ_QUEUE",
   "IN_DRAMTOL2QUEUE",
   "IN_L2TOICNTQUEUE_HIT",
   "IN_L2TOICNTQUEUE_MISS",
   "IN_ICNT2SHADER",
   "FETCHED",
};

// a helper function that deduce if a mshr contains an atomic operation 
bool isatomic(mshr_entry_t *mshr)
{
   return (mshr->insts[0].callback.function != NULL);
}

#include <map>
#include <utility>
#include <algorithm>
// a class that speeds up mshr lookup with a C++ multimap 
class mshr_lookup {
private:
   typedef std::multimap<unsigned long long int, mshr_entry*> mshr_lut_t;
   mshr_lut_t m_lut; // multiple mshr entries can have the same tag
private:
   void insert(mshr_entry* mshr) 
   {
      using namespace std;
      unsigned long long int tag_addr = mshr->addr;
      m_lut.insert(make_pair(tag_addr, mshr));
   }

   mshr_entry* lookup(unsigned long long int addr) const 
   {
      using namespace std;
      // mshr_lut_t::const_iterator i_lut = m_lut.find(tag_addr);
      pair<mshr_lut_t::const_iterator, mshr_lut_t::const_iterator> i_range = m_lut.equal_range(addr);
      if (i_range.first == i_range.second) {
         return NULL;
      } else {
         mshr_lut_t::const_iterator i_lut = i_range.first; 
         mshr_entry* mshr_hit = i_lut->second;
         //follow match to end of merge chain:
         //this won't really work for different sized requests, ie can't merge a 64b request to a 32b
         while (mshr_hit->merged_requests) {
            mshr_hit = mshr_hit->merged_requests;
         }
         return mshr_hit;
      }
   }     

   void remove(mshr_entry* mshr)
   {
      using namespace std;
      std::pair<mshr_lut_t::iterator, mshr_lut_t::iterator> i_range = m_lut.equal_range(mshr->addr);

      assert(i_range.first != i_range.second);

      for (mshr_lut_t::iterator i_lut = i_range.first; i_lut != i_range.second; ++i_lut) {
         if (i_lut->second == mshr) {
            m_lut.erase(i_lut);
            break;
         }
      }
   }
public:
   //checks if we should do mshr merging for this mshr
   bool can_merge(mshr_entry_t * mshr)
   {
      if (mshr->iswrite) return false; // can't merge a write
      if (isatomic(mshr)) return false; // can't merge a atomic operation
      bool interwarp_mshr_merge = gpgpu_interwarp_mshr_merge & GLOBAL_MSHR_MERGE;
      if (mshr->istexture) {
         interwarp_mshr_merge = gpgpu_interwarp_mshr_merge & TEX_MSHR_MERGE;
      } else if (mshr->isconst) {
         interwarp_mshr_merge = gpgpu_interwarp_mshr_merge & CONST_MSHR_MERGE;
      }
      return interwarp_mshr_merge;
   }

   void mshr_fast_lookup_insert(mshr_entry* mshr) 
   {        
      if (!can_merge(mshr)) return;  
      insert(mshr);
   }
   
   void mshr_fast_lookup_remove(mshr_entry* mshr) 
   {
      if (!can_merge(mshr)) return;  
      remove(mshr);
   }
   
   mshr_entry* shader_get_mergeable_mshr(mshr_entry_t* mshr)
   {
      if (!can_merge(mshr)) return NULL;
      return lookup(mshr->addr);
   }
};

class mem_access_t;
int is_tex ( int space );
int is_const ( int space );
int is_local ( int space );
#include <iostream>
class mshr_shader_unit{
public:
   mshr_shader_unit(unsigned max_mshr): m_max_mshr(max_mshr), m_max_mshr_used(0){
      m_mshrs.resize(max_mshr);
      for (std::vector<mshr_entry_t>::iterator i = m_mshrs.begin(); i != m_mshrs.end(); i++) m_free_list.push_back(i);
   }
   bool has_mshr(unsigned num){return (num <= m_free_list.size());}
   unsigned mshr_used(){ return m_max_mshr - m_free_list.size();}
   mshr_entry_t* add_mshr(mem_access_t &access, inst_t* warp);

   //return queue access; (includes texture pipeline return)
   mshr_entry_t* return_head(){
      if (has_return()) 
         return &(*(choose_return_queue().front())); 
      else 
         return NULL;
      }
   //return queue pop; (includes texture pipeline return)
   void pop_return_head() {
      free_mshr(return_head()->this_mshr);
      choose_return_queue().pop_front(); 
   }

   static void mshr_update_status(mshr_entry *mshr, enum mshr_status new_status );
   void mshr_return_from_mem(mshr_entry *mshr);
   void check_mshr(mshr_entry *mshr){
      assert(find(m_free_list.begin(),m_free_list.end(),mshr->this_mshr)==m_free_list.end());
      assert(mshr->insts.size());
   }
   unsigned get_max_mshr_used(){return m_max_mshr_used;}  
   void print(FILE* fp, shader_core_ctx_t* shader);
private:
   typedef std::vector<mshr_entry_t> mshr_storage_type;//list might be less complicated, but slower?
   mshr_storage_type m_mshrs; 
   std::deque< mshr_storage_type::iterator > m_free_list;
   std::deque< mshr_storage_type::iterator > m_mshr_return_queue;
   std::deque< mshr_storage_type::iterator > m_texture_mshr_pipeline;
   unsigned m_max_mshr;
   unsigned m_max_mshr_used;
   mshr_lookup m_mshr_lookup;

   mshr_entry_t *alloc_free_mshr(bool istexture){
      assert(!m_free_list.empty());
      std::vector<mshr_entry_t>::iterator i = m_free_list.back();
      m_free_list.pop_back();
      i->this_mshr = i;
      if (istexture) {
         //put in texture pipeline
         m_texture_mshr_pipeline.push_back(i);
      }
      if (mshr_used() > m_max_mshr_used) m_max_mshr_used = mshr_used();
      return &(*i);
   }
   void free_mshr(std::vector<mshr_entry_t>::iterator &i){
      //clean up up for next time, since not reallocating memory.
      m_mshr_lookup.mshr_fast_lookup_remove(&(*i)); //need to remove before clearing insts, as they are accessed
      i->insts.clear(); //add expects this to be clear
      m_free_list.push_back(i);
   }
   bool has_return() { return (not m_mshr_return_queue.empty()) or ((not m_texture_mshr_pipeline.empty()) and m_texture_mshr_pipeline.front()->fetched());}
   std::deque< std::vector<mshr_entry_t>::iterator > & choose_return_queue() {
      //prioritize a ready texture over a global/const...
      if ((not m_texture_mshr_pipeline.empty()) and m_texture_mshr_pipeline.front()->fetched()) return m_texture_mshr_pipeline;
      assert(!m_mshr_return_queue.empty());
      return m_mshr_return_queue;
   }
};



void mshr_shader_unit::mshr_update_status(mshr_entry *mshr, enum mshr_status new_status ) {
   mshr->status = new_status;
#if DEBUGL1MISS 
   printf("cycle %d Addr %x  %d \n",gpu_sim_cycle,CACHE_TAG_OF_64(mshr->addr),new_status);
#endif
   mshr_entry * merged_req = mshr->merged_requests;
   while (merged_req) {
      merged_req->status = new_status;
      merged_req = merged_req->merged_requests;
   }
}

inline void mshr_shader_unit::mshr_return_from_mem(mshr_entry *mshr){
   mshr_update_status(mshr, FETCHED);
   if (not mshr->istexture) {
       //place in return queue
       m_mshr_return_queue.push_back(mshr->this_mshr);
       //place all merged requests in return queue
       mshr_entry * merged_req = mshr->merged_requests;
       while (merged_req) {
          m_mshr_return_queue.push_back(merged_req->this_mshr);
          merged_req = merged_req->merged_requests;
       } 
   }
}

void mshr_return_from_mem(shader_core_ctx_t * shader, mshr_entry_t* mshr){
    shader->mshr_unit->mshr_return_from_mem(mshr);
}

unsigned get_max_mshr_used(shader_core_ctx_t * shader){
   return shader->mshr_unit->get_max_mshr_used();
}


void mshr_print(FILE* fp, shader_core_ctx_t *shader) {
   shader->mshr_unit->print(fp, shader);
}

void mshr_shader_unit::print(FILE* fp, shader_core_ctx_t* shader){
   for (mshr_storage_type::iterator it = m_mshrs.begin(); it != m_mshrs.end(); it++) {
      //valid if not in free list;
      if (find(m_free_list.begin(),m_free_list.end(), it) == m_free_list.end()) {
         mshr_entry *mshr = &(*it);
         fprintf(fp, "MSHR(%d): %s Addr:0x%llx Fetched:%d Merged:%d Status:%s\n", 
		 shader->sid, 
                 (mshr->iswrite)? "=>" : "<=",
                 mshr->addr, mshr->fetched(), 
                 (mshr->merged_requests != NULL or mshr->merged_on_other_reqest), MSHR_Status_str[mshr->status]);
         for (unsigned i = 0; i < mshr->insts.size(); i++) {
            fprintf(fp,"\tthread: UID:%d HW:%d ReqAddr:0x%llx\n", mshr->insts[i].uid, mshr->insts[i].hw_thread_id, mshr->insts[i].memreqaddr);
         }
      }
   }
}

void mshr_update_status(mshr_entry* mshr, enum mshr_status new_status) {
   mshr_entry *merged_req;
   mshr->status = new_status;
#if DEBUGL1MISS 
   printf("cycle %d Addr %x  %d \n",gpu_sim_cycle,CACHE_TAG_OF_64(mshr->addr),new_status);
#endif
   merged_req = mshr->merged_requests;
   while (merged_req) {
      merged_req->status = new_status;
      merged_req = merged_req->merged_requests;
   }
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/*-------------------------------------------------------------------------*/

inst_t create_nop_inst() // just because C++ does not have designated initializer list....
{
   inst_t nop_inst;
   nop_inst.pc = 0;
   nop_inst.op=NO_OP; 
   memset(nop_inst.out, 0, sizeof(nop_inst.out)); 
   memset(nop_inst.in, 0, sizeof(nop_inst.in)); 
   nop_inst.is_vectorin=0; 
   nop_inst.is_vectorout=0;
   nop_inst.memreqaddr=0; 
   nop_inst.reg_bank_access_pending=0;
   nop_inst.reg_bank_conflict_stall_checked=0,
   nop_inst.hw_thread_id=-1; 
   nop_inst.wlane=-1;
   nop_inst.uid = (unsigned)-1;
   nop_inst.priority = (unsigned)-1;
   nop_inst.ptx_thd_info = NULL;
   nop_inst.warp_active_mask = 0;
   nop_inst.ts_cycle = 0;
   nop_inst.id_cycle = 0;
   nop_inst.ex_cycle = 0;
   nop_inst.mm_cycle = 0;
   nop_inst.cache_miss = false;
   nop_inst.space = memory_space_t();
   nop_inst.cycles = 0;
   return nop_inst;
}

static inst_t nop_inst = create_nop_inst();

inst_t *first_valid_thread( inst_t *warp )
{
   for(unsigned t=0; t < ::warp_size; t++ ) 
      if( warp[t].hw_thread_id != -1 ) 
         return warp+t;
   return NULL;
}

void move_warp( inst_t *dst, inst_t *src )
{
   memcpy(dst,src,::warp_size * sizeof(inst_t)); 
   for( unsigned t=0; t < ::warp_size; t++) 
      src[t] = nop_inst;
}

bool pipeline_regster_empty( inst_t *reg )
{
   return first_valid_thread(reg) == NULL;
}

std::list<unsigned> get_regs_written( inst_t *warp )
{
   std::list<unsigned> result;
   inst_t *fvi = first_valid_thread(warp);
   if( fvi == NULL ) 
      return result;
   for( unsigned op=0; op < 4; op++ ) {
      int reg_num = fvi->arch_reg[op]; // this math needs to match that used in function_info::ptx_decode_inst
      if( reg_num >= 0 ) // valid register
         result.push_back(reg_num);
   }
   return result;
}

int log2i(int n) {
   int lg;
   lg = -1;
   while (n) {
      n>>=1;lg++;
   }
   return lg;
}

shader_core_ctx_t* shader_create( const char *name, int sid,
                                  unsigned int n_threads,
                                  unsigned int n_mshr,
                                  fq_push_t fq_push,
                                  fq_has_buffer_t fq_has_buffer,
                                  unsigned int model )
{
   shader_core_ctx_t *sc;
   sc = (shader_core_ctx_t*)calloc(sizeof(shader_core_ctx_t),1);
   sc = new (sc) shader_core_ctx(name,sid,n_threads,n_mshr,fq_push,fq_has_buffer,model,
                                 gpu_n_warp_per_shader,gpgpu_shader_cta);
   return sc;
}

shader_core_ctx::shader_core_ctx( const char *name, int sid, 
                                  unsigned int n_threads, 
                                  unsigned int n_mshr, 
                                  fq_push_t fq_push, 
                                  fq_has_buffer_t fq_has_buffer,
                                  unsigned model,
                                  unsigned max_warps_per_cta, unsigned max_cta_per_core )
   : m_barriers( max_warps_per_cta, max_cta_per_core )
{
   shader_core_ctx *sc = this;
   assert( !((model == DWF) && gpgpu_operand_collector) );

   int i;
   unsigned int shd_n_set;
   unsigned int shd_linesize;
   unsigned int shd_n_assoc;
   unsigned char shd_policy;

   unsigned int l1tex_cache_n_set; //L1 texture cache parameters
   unsigned int l1tex_cache_linesize;
   unsigned int l1tex_cache_n_assoc;
   unsigned char l1tex_cache_policy;

   unsigned int l1const_cache_n_set; //L1 constant cache parameters
   unsigned int l1const_cache_linesize;
   unsigned int l1const_cache_n_assoc;
   unsigned char l1const_cache_policy;

   if ( gpgpu_cuda_sim ) {
      unsigned cta_size = ptx_sim_cta_size();
      if ( cta_size > n_threads ) {
         printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
         printf("                 This can cause problems with applications that use __syncthreads.\n");
         printf("                 CTA size (x*y*z) = %u, n_threads = %u\n", cta_size, n_threads );
         printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
         printf("                 modify the CUDA source to decrease the kernel block size.\n");
         abort();
      }
   }

   sc->name = name;
   sc->sid = sid;

   sc->RR_k = 0;

   sc->model = model;

   sc->pipeline_reg = (inst_t**) calloc(N_PIPELINE_STAGES, sizeof(inst_t*));
   for (int j = 0; j<N_PIPELINE_STAGES; j++) {
      sc->pipeline_reg[j] = (inst_t*) calloc(warp_size, sizeof(inst_t));
      for (unsigned i=0; i<warp_size; i++) {
         sc->pipeline_reg[j][i] = nop_inst;
      }
   }

   if (gpgpu_pre_mem_stages) {
      sc->pre_mem_pipeline = (inst_t**) calloc(gpgpu_pre_mem_stages+1, sizeof(inst_t*));
      for (unsigned j = 0; j<=gpgpu_pre_mem_stages; j++) {
         sc->pre_mem_pipeline[j] = (inst_t*) calloc(pipe_simd_width, sizeof(inst_t));
         for (int i=0; i<pipe_simd_width; i++) {
            sc->pre_mem_pipeline[j][i] = nop_inst;
         }
      }
   }
   sc->n_threads = n_threads;
   sc->thread = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), n_threads);
   sc->not_completed = 0;

   unsigned n_warp = (n_threads/warp_size) + ((n_threads%warp_size)?1:0);
   sc->warp.resize(n_warp, shd_warp_t(warp_size));
   for (unsigned j = 0; j < n_warp; j++) {
      sc->warp[j].wid = j;
   }

   sc->n_active_cta = 0;
   for (i = 0; i<MAX_CTA_PER_SHADER; i++  ) {
      sc->cta_status[i]=0;
   }
   //Warp variable initializations
   sc->next_warp = 0;
   sc->branch_priority = 0;
   sc->max_branch_priority = (int*) malloc(sizeof(int)*n_threads);


   for (unsigned i = 0; i<n_threads; i++) {
      sc->max_branch_priority[i] = INT_MAX;
      sc->thread[i].id = i;

      sc->thread[i].warp_priority = sc->max_branch_priority[i];
      sc->thread[i].avail4fetch = 0;
      sc->thread[i].m_waiting_at_barrier = 0;

      sc->thread[i].ptx_thd_info = NULL;
      sc->thread[i].cta_id = -1;
   }

   sscanf(gpgpu_cache_dl1_opt,"%d:%d:%d:%c", 
          &shd_n_set, &shd_linesize, &shd_n_assoc, &shd_policy);
   sscanf(gpgpu_cache_texl1_opt,"%d:%d:%d:%c", 
          &l1tex_cache_n_set, &l1tex_cache_linesize, &l1tex_cache_n_assoc, &l1tex_cache_policy);
   sscanf(gpgpu_cache_constl1_opt,"%d:%d:%d:%c", 
          &l1const_cache_n_set, &l1const_cache_linesize, &l1const_cache_n_assoc, &l1const_cache_policy);
#define STRSIZE 32
   char L1c_name[STRSIZE];
   char L1texc_name[STRSIZE];
   char L1constc_name[STRSIZE];
   snprintf(L1c_name, STRSIZE, "L1c_%03d", sc->sid);
   sc->L1cache = shd_cache_create(L1c_name,shd_n_set,shd_n_assoc,shd_linesize,shd_policy,1,0, 
				  gpgpu_cache_wt_through?write_through:write_back);
   shd_cache_bind_logger(sc->L1cache, sc->sid, get_shader_normal_cache_id());
   snprintf(L1texc_name, STRSIZE, "L1texc_%03d", sc->sid);
   sc->L1texcache = shd_cache_create(L1texc_name,l1tex_cache_n_set,l1tex_cache_n_assoc,l1tex_cache_linesize,l1tex_cache_policy,1,0, no_writes );
   shd_cache_bind_logger(sc->L1texcache, sc->sid, get_shader_texture_cache_id());
   snprintf(L1constc_name, STRSIZE, "L1constc_%03d", sc->sid);
   sc->L1constcache = shd_cache_create(L1constc_name,l1const_cache_n_set,l1const_cache_n_assoc,l1const_cache_linesize,l1const_cache_policy,1,0, no_writes );
   shd_cache_bind_logger(sc->L1constcache, sc->sid, get_shader_constant_cache_id());
   //at this point, should set the parameters used by addressing schemes of all textures
   ptx_set_tex_cache_linesize(l1tex_cache_linesize);

   sc->mshr_unit = new mshr_shader_unit(gpu_n_mshr_per_shader);

   sc->fq_push = fq_push;
   sc->fq_has_buffer = fq_has_buffer;

   sc->pdom_warp = (pdom_warp_ctx_t*)calloc(n_threads / warp_size, sizeof(pdom_warp_ctx_t));
   for (unsigned i = 0; i < n_threads / warp_size; ++i) {
      sc->pdom_warp[i].m_stack_top = 0;
      sc->pdom_warp[i].m_pc = (address_type*)calloc(warp_size * 2, sizeof(address_type));
      sc->pdom_warp[i].m_calldepth = (unsigned int*)calloc(warp_size * 2, sizeof(unsigned int));
      sc->pdom_warp[i].m_active_mask = (unsigned int*)calloc(warp_size * 2, sizeof(unsigned int));
      sc->pdom_warp[i].m_recvg_pc = (address_type*)calloc(warp_size * 2, sizeof(address_type));
      sc->pdom_warp[i].m_branch_div_cycle = (unsigned long long *)calloc(warp_size * 2, sizeof(unsigned long long ));

      memset(sc->pdom_warp[i].m_pc, -1, warp_size * 2 * sizeof(address_type));
      memset(sc->pdom_warp[i].m_calldepth, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_active_mask, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_recvg_pc, -1, warp_size * 2 * sizeof(address_type));
   }

   sc->waiting_at_barrier = 0;

   sc->last_issued_thread = sc->n_threads - 1;

   sc->using_dwf = (sc->model == DWF);

   sc->using_rrstage = (sc->model == DWF);

   sc->using_commit_queue = (sc->model == DWF
                             || sc->model == POST_DOMINATOR || sc->model == NO_RECONVERGE);

   if (sc->using_commit_queue) {
      sc->thd_commit_queue = dq_create("thd_commit_queue", 0, 0, 0);
   }

   sc->shmem_size = gpgpu_shmem_size;
   sc->n_registers = gpgpu_shader_registers;
   sc->n_cta = gpgpu_shader_cta;

   sc->shader_memory_new_instruction_processed = false;

   // Initialize scoreboard
   sc->scrb = new Scoreboard(sc->sid, n_warp);

   if( gpgpu_operand_collector ) {
      m_opndcoll_new.init( gpgpu_operand_collector_num_units, 
                           gpgpu_operand_collector_num_units_sfu, 
                           gpgpu_num_reg_banks, this );
   }
}


unsigned shader_reinit(shader_core_ctx_t *sc, int start_thread, int end_thread ) 
{
   int i;
   unsigned result=0;

   if ( gpgpu_cuda_sim ) {
      unsigned cta_size = ptx_sim_cta_size();
      if ( cta_size > sc->n_threads ) {
         printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
         printf("                 This can cause problems with applications that use __syncthreads.\n");
         printf("                 CTA size (x*y*z) = %u, n_threads = %u\n", cta_size, sc->n_threads );
         printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
         printf("                 modify the CUDA source to decrease the kernel block size.\n");
         abort();
      }
   }

   sc->next_warp = 0;
   sc->branch_priority = 0;

   for (i = start_thread; i<end_thread; i++)
      ptx_sim_free_sm(&sc->thread[i].ptx_thd_info);

   for (i = start_thread; i<end_thread; i++) {
      sc->max_branch_priority[i] = INT_MAX;
      sc->thread[i].warp_priority = sc->max_branch_priority[i];
      sc->thread[i].n_insn = 0;
      sc->thread[i].cta_id = -1;
   }

   for (unsigned i = start_thread / warp_size; i < end_thread / warp_size; ++i) {
      sc->warp[i].reset(warp_size);
      sc->pdom_warp[i].m_stack_top = 0;
      memset(sc->pdom_warp[i].m_pc, -1, warp_size * 2 * sizeof(address_type));
      memset(sc->pdom_warp[i].m_calldepth, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_active_mask, 0, warp_size * 2 * sizeof(unsigned int));
      memset(sc->pdom_warp[i].m_recvg_pc, -1, warp_size * 2 * sizeof(address_type));
      memset(sc->pdom_warp[i].m_branch_div_cycle, 0, warp_size * 2 * sizeof(unsigned long long ));
   }

   sc->waiting_at_barrier = 0;
   sc->last_issued_thread = end_thread - 1; 

   if (sc->using_commit_queue) {
      if (!gpgpu_spread_blocks_across_cores) //assertion no longer holds with multiple blocks per core  
         assert(dq_empty(sc->thd_commit_queue));
   }
   sc->pending_shmem_bkacc = 0;
   sc->pending_cache_bkacc = 0;
   sc->pending_cmem_acc = 0;

   //do not reset this here, shader memory may be in the middle of processing another cta's instruction.
   //sc->shader_memory_new_instruction_processed = false;

   return result;
}

// initialize a CTA in the shader core, currently only useful for PDOM and DWF

void shader_init_CTA(shader_core_ctx_t *shader, int start_thread, int end_thread)
{
   int i;
   int n_thread = end_thread - start_thread;
   address_type start_pc = ptx_thread_get_next_pc(shader->thread[start_thread].ptx_thd_info);
   if (shader->model == POST_DOMINATOR) {
      int start_warp = start_thread / warp_size;
      int end_warp = end_thread / warp_size + ((end_thread % warp_size)? 1 : 0);
      for (i = start_warp; i < end_warp; ++i) {
         shader->pdom_warp[i].m_stack_top = 0;
         memset(shader->pdom_warp[i].m_pc, -1, warp_size * 2 * sizeof(address_type));
         memset(shader->pdom_warp[i].m_calldepth, 0, warp_size * 2 * sizeof(unsigned int));
         memset(shader->pdom_warp[i].m_active_mask, 0, warp_size * 2 * sizeof(unsigned int));
         memset(shader->pdom_warp[i].m_recvg_pc, -1, warp_size * 2 * sizeof(address_type));
         memset(shader->pdom_warp[i].m_branch_div_cycle, 0, warp_size * 2 * sizeof(unsigned long long ));
         shader->pdom_warp[i].m_pc[0] = start_pc;
         shader->pdom_warp[i].m_calldepth[0] = 1;
         int t = 0;
         for (t = 0; t < (int)warp_size; t++) {
            if ( i * (int)warp_size + t < end_thread ) {
               shader->pdom_warp[i].m_active_mask[0] |= (1 << t);
            }
         }
      }
   } else if (shader->model == DWF) {
      dwf_init_CTA(shader->sid, start_thread, n_thread, start_pc);
   }

   for (i = start_thread; i<end_thread; i++) {
      shader->thread[i].in_scheduler = 1;
   } 
}




// register id for unused register slot in instruction
#define DNA       (0)

unsigned g_next_shader_inst_uid=1;

// check to see if the fetch stage need to be stalled
int shader_fetch_stalled(shader_core_ctx_t *shader)
{
   int n_warp_parts = warp_size/pipe_simd_width;

   if (shader->warp_part2issue < n_warp_parts) {
      return 1;
   }

   for (unsigned i=0; i<warp_size; i++) {
      if (shader->pipeline_reg[TS_IF][i].hw_thread_id != -1 ) {
         return 1;  // stalled 
      }
   }
   for (int i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[IF_ID][i].hw_thread_id != -1 ) {
         return 1;  // stalled 
      }
   }

   shader->warp_part2issue = 0; // reset pointer to first warp part
   shader->new_warp_TS = 1;

   return 0; // not stalled
}

// initalize the pipeline stage register to nops
void shader_clear_stage_reg(shader_core_ctx_t *shader, int stage)
{
   for (unsigned i=0; i<warp_size; i++) {
      shader->pipeline_reg[stage][i] = nop_inst;
   }
}

// return the next pc of a thread 
address_type shader_thread_nextpc(shader_core_ctx_t *shader, int tid)
{
   assert( gpgpu_cuda_sim );
   address_type pc = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
   return pc;
}

// issue thread to the warp 
// tid - thread id, warp_id - used by PDOM, wlane - position in warp
void shader_issue_thread(shader_core_ctx_t *shader, int tid, int wlane, unsigned active_mask )
{
   if ( gpgpu_cuda_sim ) {
      shader->pipeline_reg[TS_IF][wlane].hw_thread_id = tid;
      shader->pipeline_reg[TS_IF][wlane].wlane = wlane;
      shader->pipeline_reg[TS_IF][wlane].pc = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
      shader->pipeline_reg[TS_IF][wlane].ptx_thd_info = shader->thread[tid].ptx_thd_info;
      shader->pipeline_reg[TS_IF][wlane].memreqaddr = 0;
      shader->pipeline_reg[TS_IF][wlane].reg_bank_conflict_stall_checked = 0;
      shader->pipeline_reg[TS_IF][wlane].reg_bank_access_pending = 0;
      shader->pipeline_reg[TS_IF][wlane].uid = g_next_shader_inst_uid++;
      shader->pipeline_reg[TS_IF][wlane].warp_active_mask = active_mask;
      shader->pipeline_reg[TS_IF][wlane].ts_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
   }
   assert( shader->thread[tid].avail4fetch > 0 );
   shader->thread[tid].avail4fetch--;
   assert( shader->warp[wid_from_hw_tid(tid,warp_size)].n_avail4fetch > 0 );
   shader->warp[wid_from_hw_tid(tid,warp_size)].n_avail4fetch--;
}

void update_max_branch_priority(shader_core_ctx_t *shader, unsigned warp_hw_id, unsigned grid_num )
{
   int temp_max = 0;
   // This means that a group of threads has completed,
   // hence need to update max_priority
   for (unsigned i = 0; i<warp_size; i++) {
      if ( !ptx_thread_done( shader->thread[hw_tid_from_wid(warp_hw_id,warp_size,i)].ptx_thd_info ) ) {
         if (shader->thread[hw_tid_from_wid(warp_hw_id,warp_size,i)].warp_priority>=temp_max) {
            temp_max = shader->thread[hw_tid_from_wid(warp_hw_id,warp_size,i)].warp_priority;
         }
      }
   }
   for (unsigned i = 0; i<warp_size; i++) {
      shader->max_branch_priority[hw_tid_from_wid(warp_hw_id,warp_size,i)] = temp_max;
   }
}

void shader_fetch_simd_no_reconverge(shader_core_ctx_t *shader, unsigned int shader_number, int grid_num )
{
   int i;
   int tid;
   int new_tid = 0;
   address_type pc = 0;
   int warp_ok = 0;
   int n_warp = shader->n_threads/warp_size;
   int complete = 0;

   assert(gpgpu_cuda_sim);

   // First, check to see if entire program is completed, 
   // if it is, then break out of loop
   for (unsigned i=0; i<shader->n_threads; i++) {
      if (!ptx_thread_done( shader->thread[i].ptx_thd_info )) {
         complete = 0;
         break;
      } else {
         complete = 1;
      }
   }
   if (complete) {
      // printf("Shader has completed program.\n");
      return;
   }

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   // Finds a warp where all threads in it are available for fetching 
   // simultaneously(all threads are not yet in pipeline, or, the ones 
   // that are not available, are completed already
   for (i=0; i<n_warp; i++) {
      int n_completed = shader->warp[shader->next_warp].n_completed;
      int n_avail4fetch = shader->warp[shader->next_warp].n_avail4fetch;
      if (((n_completed) == (int)warp_size) ||
          ((n_completed + n_avail4fetch) < (int)warp_size) ) {
         //All threads in this warp have completed, hence go to next warp
         //Or, some of the threads are still in pipeline
         warp_ok = 0; // hey look, it's a silent register update / store instruction! (this operation is redundant)
         shader->next_warp = (shader->next_warp+1)%n_warp;
      } else {
         int n_waiting_at_barrier = shader->warp[shader->next_warp].n_waiting_at_barrier;
         if ( n_waiting_at_barrier >= (int)warp_size ) {
            warp_ok = 0; // hey look, it's a silent register update / store instruction! (this operation is redundant)
            continue;
         }
         warp_ok = 1;
         break;
      }
   }
   // None of the instructions from inside the warp can be scheduled -> should   
   // probably just stall, ie nops into pipeline
   if (!warp_ok) {
      shader_clear_stage_reg(shader, TS_IF);  // NOTE: is this needed?
      shader->next_warp = (shader->next_warp+1)%n_warp;  // NOTE: this is not round-robin.
      return;
   }

   tid = warp_size*shader->next_warp;

   for (i = 0; i<(int)warp_size; i++) {
      if (shader->thread[tid+i].warp_priority == shader->max_branch_priority[tid+i]) {
         pc = shader_thread_nextpc(shader, tid+i);
         new_tid = tid+i;
         break;
      }
   }
   //Determine which instructions inside this 'warp' will be scheduled together at this run
   //If they are cannot be scheduled together then 'save' their branch priority
   for (i = 0; i<(int)warp_size; i++) {
      if (!ptx_thread_done( shader->thread[tid+i].ptx_thd_info )) {
         address_type next_pc;
         next_pc = shader_thread_nextpc(shader, tid+i);
         if (next_pc != pc ||
             shader->thread[tid+i].warp_priority != shader->max_branch_priority[tid+i] ||
             shader->thread[tid+i].m_waiting_at_barrier) {
            if (!ptx_thread_done( shader->thread[tid+i].ptx_thd_info )) {
               if ( !shader->thread[tid + i].m_waiting_at_barrier ) {
                  shader->thread[tid + i].warp_priority = shader->branch_priority;
               }
            }
         } else {
            shader_issue_thread(shader, tid+i, i,(unsigned)-1); 
         }
      }
   }
   shader->branch_priority++;

   shader->next_warp = (shader->next_warp+1)%n_warp;
}

int pdom_sched_find_next_warp (shader_core_ctx_t *shader,int pdom_sched_policy, int* ready_warps
                               , int ready_warp_count, int* last_warp, int w_comp_c, int w_pipe_c, int w_barr_c)
{
   int n_warp = shader->n_threads/warp_size;
   int i=0;
   int selected_warp = ready_warps[0];
   int found =0; 

   switch (pdom_sched_policy) {
   case 0: 
      selected_warp = ready_warps[0]; //first ok warp found
      found=1;
      break;
   case 1  ://random
      selected_warp = ready_warps[rand()%ready_warp_count];
      found=1;  
      break;
   case 8  :// execute the first available warp which is after the warp execued last time
      found=0;
      selected_warp = (last_warp[shader->sid] + 1 ) % n_warp;
      while (!found) {
         for (i=0;i<ready_warp_count;i++) {
            if (selected_warp==ready_warps[i]) {
               found=1;
            }
         }
         if (!found)
            selected_warp = (selected_warp + 1 ) % n_warp;
      }
      break;         
   default:
      assert(0);
   }
   if (found) {
      if (ready_warp_count==1) {
         n_pdom_sc_single_stat++;
      } else {
         n_pdom_sc_orig_stat++;
      }
      return selected_warp;
   } else {
      return -1;
   }
}

void shader_fetch_simd_postdominator(shader_core_ctx_t *shader, unsigned int shader_number, int grid_num) {
   int i;
   int warp_ok = 0;
   int n_warp = shader->n_threads/warp_size;
   int complete = 0;
   int tmp_warp;
   int warp_id;

   address_type check_pc = -1;

   assert(gpgpu_cuda_sim);

   // First, check to see if entire program is completed, 
   // if it is, then break out of loop
   for (unsigned i=0; i<shader->n_threads; i++) {
      if (!ptx_thread_done( shader->thread[i].ptx_thd_info )) {
         complete = 0;
         break;
      } else {
         complete = 1;
      }
   }
   if (complete) {
      return;
   }

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   int ready_warp_count = 0;
   int w_comp_c = 0 ;
   int w_pipe_c = 0 ;
   int w_barr_c = 0 ;
   static int * ready_warps = NULL;
   static int * tmp_ready_warps = NULL;
   if (!ready_warps) {
      ready_warps = (int*)calloc(n_warp,sizeof(int));
   }
   if (!tmp_ready_warps) {
      tmp_ready_warps = (int*)calloc(n_warp,sizeof(int));
   }
   for (i=0; i<n_warp; i++) {
      ready_warps[i]=-1;
      tmp_ready_warps[i]=-1;
   }

   static int* last_warp; //keeps track of last warp issued per shader
   if (!last_warp) {
      last_warp = (int*)calloc(gpu_n_shader,sizeof(int));
   }


   // Finds a warp where all threads in it are available for fetching 
   // simultaneously(all threads are not yet in pipeline, or, the ones 
   // that are not available, are completed already
   for (i=0; i<n_warp; i++) {
      int n_completed = shader->warp[shader->next_warp].n_completed;
      int n_avail4fetch = shader->warp[shader->next_warp].n_avail4fetch;

      if ((n_completed) == (int)warp_size) {
         //All threads in this warp have completed 
         w_comp_c++;
      } else if ((n_completed+n_avail4fetch) < (int)warp_size) {
         //some of the threads are still in pipeline
         w_pipe_c++;
      } else if ( shader->warp_waiting_at_barrier(shader->next_warp) ) {
         w_barr_c++;
      } else if ( shader_warp_scoreboard_hazard(shader, shader->next_warp) ) {
    	  // Do nothing - warp is filtered out
    	  //printf("SCOREBOARD COLLISION - wid=%d\n", shader->next_warp);
      } else {
         // A valid warp is found at this point
         tmp_ready_warps[ready_warp_count] = shader->next_warp;
         ready_warp_count++;
      }
      shader->next_warp = (shader->next_warp + 1) % n_warp;
   }
   for (i=0;i<ready_warp_count;i++) {
      ready_warps[i]=tmp_ready_warps[i];
   }

   num_warps_issuable[ready_warp_count]++;
   num_warps_issuable_pershader[shader->sid]+= ready_warp_count;

   if (ready_warp_count) {
      tmp_warp = pdom_sched_find_next_warp (shader, pdom_sched_type ,ready_warps
                                            , ready_warp_count, last_warp, w_comp_c, w_pipe_c ,w_barr_c);
      if (tmp_warp != -1) {
         shader->next_warp = tmp_warp;
         warp_ok=1;  
      }
   }

   static int no_warp_issued; 
   // None of the instructions from inside the warp can be scheduled -> should  
   // probably just stall, ie nops into pipeline
   if (!warp_ok) {
      shader_clear_stage_reg(shader, TS_IF);  
      shader->next_warp = (shader->next_warp+1) % n_warp;  
      no_warp_issued = 1 ; 
      return;
   }

   /************************************************************/
   //at this point we have a warp to execute which is pointed to by
   //shader->next_warp

   warp_id = shader->next_warp;
   last_warp[shader->sid] = warp_id;
   int wtid = warp_size*warp_id;

   pdom_warp_ctx_t *scheduled_warp = &(shader->pdom_warp[warp_id]);

   // schedule threads according to active mask on the top of pdom stack
   for (i = 0; i < (int)warp_size; i++) {
      unsigned int mask = (1 << i);
      if ((scheduled_warp->m_active_mask[scheduled_warp->m_stack_top] & mask) == mask) {
         assert (!ptx_thread_done( shader->thread[wtid+i].ptx_thd_info ));
         shader_issue_thread(shader, wtid+i, i, scheduled_warp->m_active_mask[scheduled_warp->m_stack_top]);
      }
   }
   shader->next_warp = (shader->next_warp+1)%n_warp;

   // check if all issued threads have the same pc
   for (i = 0; i < (int) warp_size; i++) {
      if ( shader->pipeline_reg[TS_IF][i].hw_thread_id != -1 ) {
         if ( check_pc == (unsigned)-1 ) {
            check_pc = shader->pipeline_reg[TS_IF][i].pc;
         } else {
            assert( check_pc == shader->pipeline_reg[TS_IF][i].pc );
         }
      }
   }
}

bool shader_warp_scoreboard_hazard(shader_core_ctx_t *shader, int warp_id) {
	static inst_t active_inst;
	static op_type op = NO_OP;
	static int i1, i2, i3, i4, o1, o2, o3, o4; //4 outputs needed for texture fetches in cuda-sim
	static int vectorin, vectorout;
	static int arch_reg[MAX_REG_OPERANDS] = { -1 };
	static int pred;
	static int ar1, ar2; // address registers for memory operands

	// Get an active thread in the warp
	int wtid = warp_size*warp_id;
	pdom_warp_ctx_t *scheduled_warp = &(shader->pdom_warp[warp_id]);
	thread_ctx_t *active_thread = NULL;
	for (int i = 0; i < (int)warp_size; i++) {
		unsigned int mask = (1 << i);
		if ((scheduled_warp->m_active_mask[scheduled_warp->m_stack_top] & mask) == mask) {
			active_thread = &(shader->thread[wtid+i]);
		}
	}
	if(active_thread == NULL) return false;

	// Decode instruction
	ptx_decode_inst( active_thread->ptx_thd_info, (unsigned*)&op, &i1, &i2, &i3, &i4, &o1, &o2, &o3, &o4, &vectorin, &vectorout, arch_reg, &pred, &ar1, &ar2);
	active_inst.op = op;
	active_inst.in[0] = i1;
	active_inst.in[1] = i2;
	active_inst.in[2] = i3;
	active_inst.in[3] = i4;
	active_inst.out[0] = o1;
	active_inst.out[1] = o2;
	active_inst.out[2] = o3;
	active_inst.out[3] = o4;
	active_inst.is_vectorin = vectorin;
	active_inst.is_vectorout = vectorout;
	active_inst.pred = pred;
	active_inst.ar1 = ar1;
	active_inst.ar2 = ar2;

	return shader->scrb->checkCollision(warp_id, &active_inst);
}

void shader_pdom_update_warp_mask(shader_core_ctx_t *shader, int warp_id) {
	int wtid = warp_size*warp_id;

	pdom_warp_ctx_t *scheduled_warp = &(shader->pdom_warp[warp_id]);

	int stack_top = scheduled_warp->m_stack_top;

	address_type top_pc = scheduled_warp->m_pc[stack_top];
	unsigned int top_active_mask = scheduled_warp->m_active_mask[stack_top];
	address_type top_recvg_pc = scheduled_warp->m_recvg_pc[stack_top];

	assert(top_active_mask != 0);

	const address_type null_pc = 0;
	int warp_diverged = 0;
	address_type new_recvg_pc = null_pc;
	while (top_active_mask != 0) {

	  // extract a group of threads with the same next PC among the active threads in the warp
	  address_type tmp_next_pc = null_pc;
	  unsigned int tmp_active_mask = 0;
	  void *first_active_thread=NULL;
	  for (int i = warp_size - 1; i >= 0; i--) {
		 unsigned int mask = (1 << i);
		 if ((top_active_mask & mask) == mask) { // is this thread active?
			if (ptx_thread_done( shader->thread[wtid+i].ptx_thd_info )) {
			   top_active_mask &= ~mask; // remove completed thread from active mask
			} else if (tmp_next_pc == null_pc) {
			   first_active_thread = shader->thread[wtid+i].ptx_thd_info;
			   tmp_next_pc = shader_thread_nextpc(shader, wtid+i);
			   tmp_active_mask |= mask;
			   top_active_mask &= ~mask;
			} else if (tmp_next_pc == shader_thread_nextpc(shader, wtid+i)) {
			   tmp_active_mask |= mask;
			   top_active_mask &= ~mask;
			}
		 }
	  }

	  // discard the new entry if its PC matches with reconvergence PC
	  // that automatically reconverges the entry
	  if (tmp_next_pc == top_recvg_pc) continue;

	  // this new entry is not converging
	  // if this entry does not include thread from the warp, divergence occurs
	  if (top_active_mask != 0 && warp_diverged == 0) {
		 warp_diverged = 1;
		 // modify the existing top entry into a reconvergence entry in the pdom stack
		 new_recvg_pc = get_converge_point(top_pc,first_active_thread);
		 if (new_recvg_pc != top_recvg_pc) {
			scheduled_warp->m_pc[stack_top] = new_recvg_pc;
			scheduled_warp->m_branch_div_cycle[stack_top] = gpu_sim_cycle;
			stack_top += 1;
			scheduled_warp->m_branch_div_cycle[stack_top] = 0;
		 }
	  }

	  // discard the new entry if its PC matches with reconvergence PC
	  if (warp_diverged && tmp_next_pc == new_recvg_pc) continue;

	  // update the current top of pdom stack
	  scheduled_warp->m_pc[stack_top] = tmp_next_pc;
	  scheduled_warp->m_active_mask[stack_top] = tmp_active_mask;
	  if (warp_diverged) {
		 scheduled_warp->m_calldepth[stack_top] = 0;
		 scheduled_warp->m_recvg_pc[stack_top] = new_recvg_pc;
	  } else {
		 scheduled_warp->m_recvg_pc[stack_top] = top_recvg_pc;
	  }
	  stack_top += 1; // set top to next entry in the pdom stack
	}
	scheduled_warp->m_stack_top = stack_top - 1;

	assert(scheduled_warp->m_stack_top >= 0);
	assert(scheduled_warp->m_stack_top < (int)warp_size * 2);
}


void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
   unsigned warp_id = tid/warp_size;
   pdom_warp_ctx_t *warp_info = &(sc[sid]->pdom_warp[warp_id]);
   unsigned idx = warp_info->m_stack_top;
   *pc = warp_info->m_pc[idx];
   *rpc = warp_info->m_recvg_pc[idx];
}

void shader_fetch_mimd( shader_core_ctx_t *shader, unsigned int shader_number ) 
{
   unsigned int last_issued_thread = 0;

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   // some form of barrel processing: 
   // - checking availability from the thread after the last issued thread
   for (int i=0, j=0;i<(int)shader->n_threads && j< (int) warp_size;i++) {
      int thd_id = (i + shader->last_issued_thread + 1) % shader->n_threads;
      if (shader->thread[thd_id].avail4fetch && !shader->thread[thd_id].m_waiting_at_barrier ) {
         shader_issue_thread(shader, thd_id, j,(unsigned)-1);
         last_issued_thread = thd_id;
         j++;
      }
   }
   shader->last_issued_thread = last_issued_thread;
}

// seperate the incoming warp into multiple warps with seperate pcs
int split_warp_by_pc(int *tid_in, shader_core_ctx_t *shader, int **tid_split, address_type *pc) {
   unsigned n_pc = 0;
   static int *pc_cnt = NULL; // count the number of threads with the same pc

   assert(tid_in);
   assert(tid_split);
   assert(pc);
   memset(pc,0,sizeof(address_type)*warp_size);

   if (!pc_cnt) pc_cnt = (int*) malloc(sizeof(int)*warp_size);
   memset(pc_cnt,0,sizeof(int)*warp_size);

   // go through each thread in the given warp
   for (unsigned i=0; i< warp_size; i++) {
      if (tid_in[i] < 0) continue;
      int matched = 0;
      address_type thd_pc;
      thd_pc = shader_thread_nextpc(shader, tid_in[i]);

      // check to see if the pc has occured before
      for (unsigned j=0; j<n_pc; j++) {
         if (thd_pc == pc[j]) {
            tid_split[j][pc_cnt[j]] = tid_in[i];
            pc_cnt[j]++;
            matched = 1;
            break;
         }
      }
      // if not, put the tid in a seperate warp
      if (!matched) {
         assert(n_pc < warp_size);
         tid_split[n_pc][0] = tid_in[i];
         pc[n_pc] = thd_pc;
         pc_cnt[n_pc] = 1;
         n_pc++;
      }
   }
   return n_pc;
}

// see if this warp just executed the barrier instruction 
int warp_reached_barrier(int *tid_in, shader_core_ctx_t *shader)
{
   int reached_barrier = 0;
   for (unsigned i=0; i<warp_size; i++) {
      if (tid_in[i] < 0) continue;
      if (shader->thread[tid_in[i]].m_reached_barrier) {
         reached_barrier = 1;
         break;
      }
   }
   return reached_barrier;
}

// seperate the incoming warp into multiple warps with seperate pcs and cta
int split_warp_by_cta(int *tid_in, shader_core_ctx_t *shader, int **tid_split, address_type *pc, int *cta) {
   unsigned n_pc = 0;
   static int *pc_cnt = NULL; // count the number of threads with the same pc

   assert(tid_in);
   assert(tid_split);
   assert(pc);
   memset(pc,0,sizeof(address_type)*warp_size);

   if (!pc_cnt) pc_cnt = (int*) malloc(sizeof(int)*warp_size);
   memset(pc_cnt,0,sizeof(int)*warp_size);

   // go through each thread in the given warp
   for (unsigned i=0; i<warp_size; i++) {
      if (tid_in[i] < 0) continue;
      int matched = 0;
      address_type thd_pc;
      thd_pc = shader_thread_nextpc(shader, tid_in[i]);

      int thd_cta = ptx_thread_get_cta_uid( shader->thread[tid_in[i]].ptx_thd_info );

      // check to see if the pc has occured before
      for (unsigned j=0; j<n_pc; j++) {
         if (thd_pc == pc[j] && thd_cta == cta[j]) {
            tid_split[j][pc_cnt[j]] = tid_in[i];
            pc_cnt[j]++;
            matched = 1;
            break;
         }
      }
      // if not, put the tid in a seperate warp
      if (!matched) {
         assert(n_pc < warp_size);
         tid_split[n_pc][0] = tid_in[i];
         pc[n_pc] = thd_pc;
         cta[n_pc] = thd_cta;
         pc_cnt[n_pc] = 1;
         n_pc++;
      }
   }
   return n_pc;
}

void shader_fetch_simd_dwf( shader_core_ctx_t *shader, unsigned int shader_number ) {

   static int *tid_in = NULL;
   static int *tid_out = NULL;

   if (!tid_in) {
      tid_in = (int*) malloc(sizeof(int)*warp_size);
      memset(tid_in, -1, sizeof(int)*warp_size);
   }
   if (!tid_out) {
      tid_out = (int*) malloc(sizeof(int)*warp_size);
      memset(tid_out, -1, sizeof(int)*warp_size);
   }


   static int **tid_split = NULL;
   if (!tid_split) {
      tid_split = (int**)malloc(sizeof(int*)*warp_size);
      tid_split[0] = (int*)malloc(sizeof(int)*warp_size*warp_size);
      for (unsigned i=1; i<warp_size; i++) {
         tid_split[i] = tid_split[0] + warp_size * i;
      }
   }

   static address_type *thd_pc = NULL;
   if (!thd_pc) thd_pc = (address_type*)malloc(sizeof(address_type)*warp_size);
   static int *thd_cta = NULL;
   if (!thd_cta) thd_cta = (int*)malloc(sizeof(int)*warp_size);

   int warpupdate_bw = 1;
   while (!dq_empty(shader->thd_commit_queue) && warpupdate_bw > 0) {
      // grab a committed warp, split it into multiple BRUs (tid_split) by PC
      int *tid_commit = (int*)dq_pop(shader->thd_commit_queue);
      memset(tid_split[0], -1, sizeof(int)*warp_size*warp_size);
      memset(thd_pc, 0, sizeof(address_type)*warp_size);
      memset(thd_cta, -1, sizeof(int)*warp_size);

      int reached_barrier = warp_reached_barrier(tid_commit, shader);

      unsigned n_warp_update;
      if (reached_barrier) {
         n_warp_update = split_warp_by_cta(tid_commit, shader, tid_split, thd_pc, thd_cta);
      } else {
         n_warp_update = split_warp_by_pc(tid_commit, shader, tid_split, thd_pc);
      }

      if (n_warp_update > 2) gpgpu_commit_pc_beyond_two++;
      warpupdate_bw -= n_warp_update;
      // put the splitted warp updates into the DWF scheduler
      for (unsigned i=0;i<n_warp_update;i++) {
         for (unsigned j=0;j<warp_size;j++) {
            if (tid_split[i][j] < 0) continue;
            assert(shader->thread[tid_split[i][j]].avail4fetch);
            assert(!shader->thread[tid_split[i][j]].in_scheduler);
            shader->thread[tid_split[i][j]].in_scheduler = 1;
         }
         dwf_clear_accessed(shader->sid);
         if (reached_barrier) {
            dwf_update_warp_at_barrier(shader->sid, tid_split[i], thd_pc[i], thd_cta[i]);
         } else {
            dwf_update_warp(shader->sid, tid_split[i], thd_pc[i]);
         }
      }

      free_commit_warp(tid_commit);
   }

   // Track the #PC right after the warps are input to the scheduler
   dwf_update_statistics(shader->sid);
   dwf_clear_policy_access(shader->sid);

   if (shader_fetch_stalled(shader)) {
      return; 
   }
   shader_clear_stage_reg(shader, TS_IF);

   address_type scheduled_pc;
   dwf_issue_warp(shader->sid, tid_out, &scheduled_pc);

   for (unsigned i=0; i<warp_size; i++) {
      int issue_tid = tid_out[i];
      if (issue_tid >= 0) {
         shader_issue_thread(shader, issue_tid, i, (unsigned)-1);
         shader->thread[issue_tid].in_scheduler = 0;
         shader->thread[issue_tid].m_reached_barrier = 0;
         shader->last_issued_thread = issue_tid;
         assert(shader->pipeline_reg[TS_IF][i].pc == scheduled_pc);
      }
   }   
}

void print_shader_cycle_distro( FILE *fout ) 
{
   fprintf(fout, "Warp Occupancy Distribution:\n");
   fprintf(fout, "Stall:%d\t", shader_cycle_distro[0]);
   fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[1]);
   fprintf(fout, "W0_Mem:%d", shader_cycle_distro[2]);
   for (unsigned i = 3; i < warp_size + 3; i++) {
      fprintf(fout, "\tW%d:%d", i-2, shader_cycle_distro[i]);
   }
   fprintf(fout, "\n");
}

void inflight_memory_insn_add( shader_core_ctx_t *shader, inst_t *mem_insn)
{
   if (enable_ptx_file_line_stats) {
      ptx_file_line_stats_add_inflight_memory_insn(shader->sid, mem_insn->pc);
   }
}

void inflight_memory_insn_sub( shader_core_ctx_t *shader, inst_t *mem_insn)
{
   if (enable_ptx_file_line_stats) {
      ptx_file_line_stats_sub_inflight_memory_insn(shader->sid, mem_insn->pc);
   }
}

void report_exposed_memory_latency( shader_core_ctx_t *shader )
{
   if (enable_ptx_file_line_stats) {
      ptx_file_line_stats_commit_exposed_latency(shader->sid, 1);
   }
}

static int gpgpu_warp_occ_detailed = 0;
static int **warp_occ_detailed = NULL;

void check_stage_pcs( shader_core_ctx_t *shader, unsigned stage );
void check_pm_stage_pcs( shader_core_ctx_t *shader, unsigned stage );

void shader_fetch( shader_core_ctx_t *shader, unsigned int shader_number, int grid_num ) 
{
   assert(shader->model < NUM_SIMD_MODEL);
   int n_warp_parts = warp_size/pipe_simd_width;

   // check if decode stage is stalled
   int decode_stalled = 0;
   for (int i = 0; i < pipe_simd_width; i++) {
      if (shader->pipeline_reg[IF_ID][i].hw_thread_id != -1 )
         decode_stalled = 1;
   }

   if (shader->gpu_cycle % n_warp_parts == 0) {

      switch (shader->model) {
      case NO_RECONVERGE:
         shader_fetch_simd_no_reconverge(shader, shader_number, grid_num );
         break;
      case POST_DOMINATOR:
         shader_fetch_simd_postdominator(shader, shader_number, grid_num);
         break;
      case MIMD:
         shader_fetch_mimd(shader, shader_number);
         break;
      case DWF:
         shader_fetch_simd_dwf(shader, shader_number);
         break;
      default:
         fprintf(stderr, "Unknown scheduler: %d\n", shader->model);
         assert(0);
         break;
      }

      static int *tid_out = NULL;
      if (!tid_out) {
         tid_out = (int*) malloc(sizeof(int) * warp_size);
      }
      memset(tid_out, -1, sizeof(int)*warp_size);

      if (!shader_cycle_distro) {
         shader_cycle_distro = (unsigned int*) calloc(warp_size + 3, sizeof(unsigned int));
      }

      if (gpgpu_no_divg_load && shader->new_warp_TS && !decode_stalled) {
         int n_thd_in_warp = 0;
         address_type pc_out = 0xDEADBEEF;
         for (unsigned i=0; i<warp_size; i++) {
            tid_out[i] = shader->pipeline_reg[TS_IF][i].hw_thread_id;
            if (tid_out[i] >= 0) {
               n_thd_in_warp += 1;
               pc_out = shader->pipeline_reg[TS_IF][i].pc;
            }
         }

         //wpt_register_warp(tid_out, shader);
         get_warp_tracker_pool().wpt_register_warp(tid_out, shader, pc_out);

         if (gpu_runtime_stat_flag & GPU_RSTAT_DWF_MAP) {
            track_thread_pc( shader->sid, tid_out, pc_out );
         }
         if (gpgpu_cflog_interval != 0) {
            insn_warp_occ_log( shader->sid, pc_out, n_thd_in_warp);
            shader_warp_occ_log( shader->sid, n_thd_in_warp);
         }
         if ( gpgpu_warpdistro_shader < 0 || shader->sid == gpgpu_warpdistro_shader ) {
            shader_cycle_distro[n_thd_in_warp + 2] += 1;
            if (n_thd_in_warp == 0) {
               if (shader->pending_mem_access == 0) shader_cycle_distro[1]++;
            }
         }
         shader->new_warp_TS = 0;

         if (enable_ptx_file_line_stats && n_thd_in_warp > 0) {
            //ptx_file_line_stats_add_warp_issued(pc_out);
            //ptx_file_line_stats_add_warp_occ_total(pc_out, n_thd_in_warp);
         }

         if ( gpgpu_warp_occ_detailed && 
              n_thd_in_warp && (shader->model == POST_DOMINATOR) ) {
            int n_warp = gpu_n_thread_per_shader / warp_size;
            if (!warp_occ_detailed) {
               warp_occ_detailed = (int**) malloc(sizeof(int*) * gpu_n_shader * n_warp);
               warp_occ_detailed[0] = (int*) calloc(sizeof(int), gpu_n_shader * n_warp * warp_size);
               for (unsigned i = 0; i < n_warp * gpu_n_shader; i++) {
                  warp_occ_detailed[i] = warp_occ_detailed[0] + i * warp_size;
               }
            }

            int wid = -1;
            for (unsigned i=0; i<warp_size; i++) {
               if (tid_out[i] >= 0) wid = tid_out[i] / warp_size;
            }
            assert(wid != -1);
            warp_occ_detailed[shader->sid * n_warp + wid][n_thd_in_warp - 1] += 1;

            if (shader->sid == 0 && wid == 16 && 0) {
               printf("wtrace[%08x] ", pc_out);
               for (unsigned i=0; i<warp_size; i++) {
                  printf("%03d ", tid_out[i]);
               }
               printf("\n");
            }
         }
      } else {
         if ( gpgpu_warpdistro_shader < 0 || shader->sid == gpgpu_warpdistro_shader ) {
            shader_cycle_distro[0] += 1;
         }
      }

      if (!decode_stalled) {
         for (unsigned i = 0; i < warp_size; i++) {
            int tid_tsif = shader->pipeline_reg[TS_IF][i].hw_thread_id;
            address_type pc_out = shader->pipeline_reg[TS_IF][i].pc;
            cflog_update_thread_pc(shader->sid, tid_tsif, pc_out);
         }
      }

      if (enable_ptx_file_line_stats && !decode_stalled) {
         int TS_stage_empty = 1;
         for (unsigned i = 0; i < warp_size; i++) {
            if (shader->pipeline_reg[TS_IF][i].hw_thread_id >= 0) {
               TS_stage_empty = 0;
               break;
            }
         }
         if (TS_stage_empty) {
            report_exposed_memory_latency(shader);
         }
      }
   }

   // if not, send the warp part to decode stage
   if (!decode_stalled && shader->warp_part2issue < n_warp_parts) {
      check_stage_pcs(shader,TS_IF);
      for (int i = 0; i < pipe_simd_width; i++) {
         int wlane_idx = shader->warp_part2issue * pipe_simd_width + i;
         shader->pipeline_reg[IF_ID][i] = shader->pipeline_reg[TS_IF][wlane_idx];
         shader->pipeline_reg[IF_ID][i].if_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
         shader->pipeline_reg[TS_IF][wlane_idx] = nop_inst;
      }
      shader->warp_part2issue += 1;
   }
}

inline int is_load ( op_type op ) {
   return op == LOAD_OP;
}

inline int is_store ( op_type op ) {
   return op == STORE_OP;
}

inline int is_tex ( memory_space_t space ) {
   return((space) == tex_space);
}

inline int is_const ( memory_space_t space ) {
   return((space.get_type() == const_space) || (space == param_space_kernel));
}

inline int is_local ( memory_space_t space ) {
   return (space == local_space) || (space == param_space_local);
}

inline int is_param ( memory_space_t space ) {
   return (space == param_space_kernel);
}

inline int is_shared ( memory_space_t space ) {
   return((space) == shared_space);
}

inline int shmem_bank ( address_type addr ) {
   return((int)(addr/((address_type)WORD_SIZE)) % gpgpu_n_shmem_bank);
}

inline int cache_bank ( address_type addr, shader_core_ctx_t *shader ) {
   return(int)( addr >> (address_type)shader->L1cache->line_sz_log2 ) & ( gpgpu_n_cache_bank - 1 );
}

inline address_type coalesced_segment(address_type addr, unsigned segment_size_lg2bytes)
{
   return  (addr >> segment_size_lg2bytes);
}


inline address_type translate_local_memaddr(address_type localaddr, shader_core_ctx_t *shader, int tid)
{
   // During functional execution, each thread sees its own memory space for local memory, but these
   // need to be mapped to a shared address space for timing simulation.  We do that mapping here.
   localaddr -= 0x100;
   localaddr /=4;
   if (gpgpu_local_mem_map) {
      // Dnew = D*nTpC*nCpS*nS + nTpC*C + T%nTpC
      // C = S + nS*(T/nTpC)
      // D = data index; T = thread; C = CTA; S = shader core; p = per
      // keep threads in a warp contiguous
      // then distribute across memory space by CTAs from successive shader cores first, 
      // then by successive CTA in same shader core
      localaddr *= gpu_padded_cta_size * gpu_max_cta_per_shader * gpu_n_shader;
      localaddr += gpu_padded_cta_size * (shader->sid + gpu_n_shader * (tid / gpu_padded_cta_size));
      localaddr += tid % gpu_padded_cta_size; 
   } else {
      // legacy mapping that maps the same address in the local memory space of all threads 
      // to a single contiguous address region 
      localaddr *= gpu_n_shader * gpu_n_thread_per_shader;
      localaddr += (gpu_n_thread_per_shader*shader->sid) + tid;
   }
   localaddr *= 4;
   localaddr += 0x100;

   return localaddr;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Register Bank Conflict Structures

bool gpgpu_reg_bank_conflict_model;

#define MAX_REG_BANKS 32
unsigned int gpgpu_num_reg_banks; // this needs to be less than MAX_REG_BANKS
bool gpgpu_reg_bank_use_warp_id;

#define MAX_BANK_CONFLICT 8 /* tex can have four source and four destination regs */

class reg_bank_access {
public:
  reg_bank_access():tot(0),rd(0),wr(0){
    for (unsigned i = 0; i < 4; i++) rd_regs[i] = -1; 
  }
	unsigned tot;
	unsigned rd;
	unsigned wr;
	int rd_regs[4];
}; 

int register_bank(int regnum, int tid)
{
   int bank = regnum;
   if (gpgpu_reg_bank_use_warp_id)
      bank += tid >> 5/*log2(warp_size)*/;
   return bank % gpgpu_num_reg_banks;
}

reg_bank_access g_reg_bank_access[MAX_REG_BANKS];

// just to use as "shorthand" for clearing accesses each cycle
static const struct reg_bank_access empty_reg_bank_access;

unsigned int gpu_reg_bank_conflict_stalls = 0;

void shader_opnd_collect_read(shader_core_ctx_t* shader)
{
   const int prevstage = ID_OC;
   shader->m_opndcoll_new.step(shader->pipeline_reg[prevstage]);
}

void shader_opnd_collect_write(shader_core_ctx_t* shader)
{
   shader->m_opndcoll_new.writeback(shader->pipeline_reg[WB_RT]);
}

/////////////////////////////////////////////////////////////////////////////////////////

void shader_decode( shader_core_ctx_t *shader, 
                    unsigned int shader_number,
                    unsigned int grid_num ) {

   address_type addr;
   dram_callback_t callback;
   op_type op = NO_OP;
   int tid;
   int i1, i2, i3, i4, o1, o2, o3, o4; //4 outputs needed for texture fetches in cuda-sim
   int i;
   int touched_priority=0;
   int warp_tid=0;
   unsigned data_size;
   memory_space_t space;
   unsigned cycles;
   int vectorin, vectorout;
   int arch_reg[MAX_REG_OPERANDS] = { -1 };
   int pred;
   int ar1, ar2; // address registers for memory operands
   address_type regs_regs_PC = 0xDEADBEEF;
   address_type warp_current_pc = 0x600DBEEF;
   address_type warp_next_pc = 0x600DBEEF;
   int       warp_diverging = 0;
   const int nextstage = (gpgpu_operand_collector) ? ID_OC : \
      (shader->using_rrstage ? ID_RR : ID_EX);
   unsigned warp_id = -1;
   unsigned cta_id = -1;

   // stalling for register bank conflict 
   if ( gpgpu_reg_bank_conflict_model ) {
      for (i=0; i<pipe_simd_width;i++) {
         if ( shader->pipeline_reg[IF_ID][i].reg_bank_conflict_stall_checked ) {
            if ( shader->pipeline_reg[IF_ID][i].reg_bank_access_pending > 0 ) {
               assert( shader->pipeline_reg[IF_ID][i].reg_bank_access_pending <= 8 );
               shader->pipeline_reg[IF_ID][i].reg_bank_access_pending--;
               gpu_reg_bank_conflict_stalls++;
               return;
            }
         }
      }
   }

   for (i=0; i<pipe_simd_width;i++) {
      if (shader->pipeline_reg[nextstage][i].hw_thread_id != -1 ) {
         return;  /* stalled */
      }
   }

   check_stage_pcs(shader,IF_ID);

   // decode the instruction 
   int first_valid_thread = -1;
   for (i=0; i<pipe_simd_width;i++) {

      if (shader->pipeline_reg[IF_ID][i].hw_thread_id == -1 )
         continue; /* bubble */

      /* get the next instruction to execute from fetch stage */
      tid = shader->pipeline_reg[IF_ID][i].hw_thread_id;
      if (first_valid_thread == -1) {
         first_valid_thread = i;
         warp_id = tid/warp_size;
         assert( !shader->warp_waiting_at_barrier(warp_id) );
         cta_id = shader->thread[tid].cta_id;
      }

      if ( gpgpu_cuda_sim ) {
         ptx_decode_inst( shader->thread[tid].ptx_thd_info, (unsigned*)&op, &i1, &i2, &i3, &i4, &o1, &o2, &o3, &o4, &vectorin, &vectorout, arch_reg, &pred, &ar1, &ar2);
         shader->pipeline_reg[IF_ID][i].op = op;
         shader->pipeline_reg[IF_ID][i].pc = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
         shader->pipeline_reg[IF_ID][i].ptx_thd_info = shader->thread[tid].ptx_thd_info;

      } else {
         abort();
      }
      // put the info into the shader instruction structure 
      // - useful in tracking instruction dependency (not needed for now)
      shader->pipeline_reg[IF_ID][i].in[0] = i1;
      shader->pipeline_reg[IF_ID][i].in[1] = i2;
      shader->pipeline_reg[IF_ID][i].in[2] = i3;
      shader->pipeline_reg[IF_ID][i].in[3] = i4;
      shader->pipeline_reg[IF_ID][i].out[0] = o1;
      shader->pipeline_reg[IF_ID][i].out[1] = o2;
      shader->pipeline_reg[IF_ID][i].out[2] = o3;
      shader->pipeline_reg[IF_ID][i].out[3] = o4;

   }

   // checking for register bank conflict and stall accordingly
   if ( gpgpu_reg_bank_conflict_model && 
        first_valid_thread != -1 && 
        !shader->pipeline_reg[first_valid_thread][IF_ID].reg_bank_conflict_stall_checked ) 
   {
      for (i = 4; i < 8; i++) {
         if( arch_reg[i] != -1 ) {
            assert( arch_reg[i] >=0 );
            assert( gpgpu_num_reg_banks <= MAX_REG_BANKS );
            int skip = 0;
            int bank = arch_reg[i] % gpgpu_num_reg_banks;
            int opndreg = shader->pipeline_reg[first_valid_thread][IF_ID].in[i-4];
            assert(opndreg >= 0); 
            int j;
            for (j = 0; j < 4; j++) {
	            if (g_reg_bank_access[bank].rd_regs[j] == -1)
		            break;
	            else if (g_reg_bank_access[bank].rd_regs[j] == opndreg) {
		            // two operands reading from same register in same bank, can be merged into a single read
		            skip = 1;
		            break;
	            }
            }
            if (!skip) {
	            g_reg_bank_access[bank].tot++;
	            g_reg_bank_access[bank].rd++;
	            g_reg_bank_access[bank].rd_regs[j] = opndreg;
            }
         }
      }
      
      unsigned max_access=0;   
      inst_t* conflict_inst = &shader->pipeline_reg[first_valid_thread][IF_ID];
      for(unsigned r = 0; r < gpgpu_num_reg_banks; r++ ) {
         if( g_reg_bank_access[r].tot > max_access )
            max_access = g_reg_bank_access[r].tot;
         g_reg_bank_access[r] = empty_reg_bank_access;
      }
      if( max_access >= 1 ) {
         assert( max_access <= MAX_REG_OPERANDS );
         conflict_inst->reg_bank_access_pending = max_access - 1;
         if( max_access > 1 ) {
            conflict_inst->reg_bank_conflict_stall_checked = 1;
            return; // stall pipeline
         }
      }
      shader->pipeline_reg[first_valid_thread][IF_ID].reg_bank_conflict_stall_checked = 1;
   }

   // execute the instruction functionally
   short last_hw_thread_id = -1;
   bool first_thread_in_warp = true;
   for (i=0; i<pipe_simd_width;i++) {
      if (shader->pipeline_reg[IF_ID][i].hw_thread_id == -1 )
         continue; /* bubble */

      if(last_hw_thread_id > -1)
    	  first_thread_in_warp = false;
      last_hw_thread_id = shader->pipeline_reg[IF_ID][i].hw_thread_id;

      /* get the next instruction to execute from fetch stage */
      tid = shader->pipeline_reg[IF_ID][i].hw_thread_id;
      if ( gpgpu_cuda_sim ) {
         int arch_reg[MAX_REG_OPERANDS];

         // Decode instruction
         ptx_decode_inst( shader->thread[tid].ptx_thd_info, (unsigned*)&op, &i1, &i2, &i3, &i4, &o1, &o2, &o3, &o4, &vectorin, &vectorout, arch_reg, &pred, &ar1, &ar2 );

         // Functionally execute instruction
         ptx_exec_inst( shader->thread[tid].ptx_thd_info, &addr, &space, &data_size, &cycles, &callback, shader->pipeline_reg[IF_ID][i].warp_active_mask );

         shader->pipeline_reg[IF_ID][i].callback = callback;
         shader->pipeline_reg[IF_ID][i].space = space;
         if (is_local(space) && (is_load(op) || is_store(op))) {
            addr = translate_local_memaddr(addr, shader, tid);
         }
         shader->pipeline_reg[IF_ID][i].is_vectorin = vectorin;
         shader->pipeline_reg[IF_ID][i].is_vectorout = vectorout;
         shader->pipeline_reg[IF_ID][i].pred = pred;
         shader->pipeline_reg[IF_ID][i].ar1 = ar1;
         shader->pipeline_reg[IF_ID][i].ar2 = ar2;
         shader->pipeline_reg[IF_ID][i].data_size = data_size;
         shader->pipeline_reg[IF_ID][i].cycles = cycles;

         // Mark destination registers as write-pending in scoreboard
	     // Only do this for the first thread in warp
	     if(first_thread_in_warp) {
   		    shader->scrb->reserveRegisters(warp_id, &(shader->pipeline_reg[IF_ID][i]));
			//shader->scrb->printContents();
	     }

         warp_current_pc = shader->pipeline_reg[IF_ID][i].pc;
         memcpy( shader->pipeline_reg[IF_ID][i].arch_reg, arch_reg, sizeof(arch_reg) );
         regs_regs_PC = ptx_thread_get_next_pc( shader->thread[tid].ptx_thd_info );
      }

      shader->pipeline_reg[IF_ID][i].memreqaddr = addr;
      if ( op == LOAD_OP ) {
         shader->pipeline_reg[IF_ID][i].inst_type = LOAD_OP;
      } else if ( op == STORE_OP ) {
         shader->pipeline_reg[IF_ID][i].inst_type = STORE_OP;
      }

      if ( gpgpu_cuda_sim && ptx_thread_at_barrier( shader->thread[tid].ptx_thd_info ) ) {
         if (shader->model == DWF) {
            shader->thread[tid].m_waiting_at_barrier=1;
            shader->thread[tid].m_reached_barrier=1; // not reset at barrier release, but at the issue after that
            shader->warp[wid_from_hw_tid(tid,warp_size)].n_waiting_at_barrier++;
            shader->waiting_at_barrier++;
            int cta_uid = ptx_thread_get_cta_uid( shader->thread[tid].ptx_thd_info );
            dwf_hit_barrier( shader->sid, cta_uid );
         
            int release = ptx_thread_all_at_barrier( shader->thread[tid].ptx_thd_info ); //test if all threads arrived at the barrier
            if ( release ) { //All threads arrived at barrier...releasing
               int cta_uid = ptx_thread_get_cta_uid( shader->thread[tid].ptx_thd_info );
               for ( unsigned t=0; t < gpu_n_thread_per_shader; ++t ) {
                  if ( !ptx_thread_at_barrier( shader->thread[t].ptx_thd_info ) )
                     continue;
                  int other_cta_uid = ptx_thread_get_cta_uid( shader->thread[t].ptx_thd_info );
                  if ( other_cta_uid == cta_uid ) { //reseting @barrier tracking info
                     shader->warp[wid_from_hw_tid(t,warp_size)].n_waiting_at_barrier=0;
                     shader->thread[t].m_waiting_at_barrier=0;
                     ptx_thread_reset_barrier( shader->thread[t].ptx_thd_info );
                     shader->waiting_at_barrier--;
                  }
               }
               if (shader->model == DWF) {
                  dwf_release_barrier( shader->sid, cta_uid );
               }
               ptx_thread_release_barrier( shader->thread[tid].ptx_thd_info );
            }
         }
      } else {
         assert( !shader->thread[tid].m_waiting_at_barrier );
      }

      // put the info into the shader instruction structure 
      // - useful in tracking instruction dependency (not needed for now)
      shader->pipeline_reg[IF_ID][i].in[0] = i1;
      shader->pipeline_reg[IF_ID][i].in[1] = i2;
      shader->pipeline_reg[IF_ID][i].in[2] = i3;
      shader->pipeline_reg[IF_ID][i].in[3] = i4;
      shader->pipeline_reg[IF_ID][i].out[0] = o1;
      shader->pipeline_reg[IF_ID][i].out[1] = o2;
      shader->pipeline_reg[IF_ID][i].out[2] = o3;
      shader->pipeline_reg[IF_ID][i].out[3] = o4;

      // go to the next instruction 
      // - done implicitly in ptx_exec_inst()
      
      // branch divergence detection
      if (warp_next_pc != regs_regs_PC) {
         if (warp_next_pc == 0x600DBEEF) {
            warp_next_pc = regs_regs_PC;
         } else {
            warp_diverging = 1;
         }
      }

      // direct the instruction to the appropriate next stage (config dependent)
      shader->pipeline_reg[nextstage][i] = shader->pipeline_reg[IF_ID][i];
      shader->pipeline_reg[nextstage][i].id_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      shader->pipeline_reg[IF_ID][i] = nop_inst;
   }

   if( op == BARRIER_OP ) {
      shader->set_at_barrier(cta_id,warp_id);
   }

   if ( shader->model == NO_RECONVERGE && touched_priority ) {
      update_max_branch_priority(shader,warp_tid,grid_num);
   }
   shader->n_diverge += warp_diverging;
   if (warp_diverging == 1) {
       assert(warp_current_pc != 0x600DBEEF); // guard against empty warp causing warp divergence
       ptx_file_line_stats_add_warp_divergence(warp_current_pc, 1);
   }
}

unsigned int n_regconflict_stall = 0;


int regfile_hash(signed thread_number, unsigned simd_size, unsigned n_banks) {
   if (gpgpu_thread_swizzling) {
      signed warp_ID = thread_number / simd_size;
      return((thread_number + warp_ID) % n_banks);
   } else {
      return(thread_number % n_banks);
   }
}

int gpgpu_n_reg_banks = 8;
void shader_preexecute( shader_core_ctx_t *shader, 
                        unsigned int shader_number ) {
   int i;
   static int *thread_warp = NULL;
   int n_access_per_cycle = pipe_simd_width / gpgpu_n_reg_banks;

   if (!thread_warp) {
      thread_warp = (int*) malloc(sizeof(int) * pipe_simd_width);
   }

   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[RR_EX][i].hw_thread_id != -1 ) {
         //stalled, but can still service a register read
         if (shader->RR_k) {
            shader->RR_k--;  
         }
         return;  // stalled 
      }
   }

   // if there is still register read to service, stall
   if (shader->RR_k > 1) {
      shader->RR_k--;
      return; 
   }

   // if RR_k == 1, it was stalled previously and the register read is now done
   if (!shader->RR_k && gpgpu_reg_bankconflict) {
      int max_reg_bank_acc = 0;
      for (i=0; i<pipe_simd_width; i++) {
         thread_warp[i] = 0;
      }
      for (i=0; i<pipe_simd_width; i++) {
         if (shader->pipeline_reg[ID_RR][i].hw_thread_id != -1 )
            thread_warp[regfile_hash(shader->pipeline_reg[ID_RR][i].hw_thread_id, 
                                     warp_size, gpgpu_n_reg_banks)]++;
      }
      for (i=0; i<pipe_simd_width; i++) {
         if (thread_warp[i] > max_reg_bank_acc ) {
            max_reg_bank_acc = thread_warp[i];
         }
      }
      // calculate the number of cycles needed for each register bank to fulfill all accesses
      shader->RR_k = (max_reg_bank_acc / n_access_per_cycle) + ((max_reg_bank_acc % n_access_per_cycle)? 1 : 0);
   }

   // if there are more than one access cycle needed at a bank, stall
   if (shader->RR_k > 1) {
      n_regconflict_stall++;
      shader->RR_k--;
      return; 
   }

   check_stage_pcs(shader,ID_RR);

   shader->RR_k = 0; //setting RR_k to 0 to indicate RF conflict check next cycle
   for (i=0; i<pipe_simd_width;i++) {
      if (shader->pipeline_reg[ID_RR][i].hw_thread_id == -1 )
         continue; //bubble 
      shader->pipeline_reg[ID_EX][i] = shader->pipeline_reg[ID_RR][i];
      shader->pipeline_reg[ID_RR][i] = nop_inst;
   }

}


void shader_execute_pipe( shader_core_ctx_t *shader, unsigned int shader_number, unsigned pipeline, unsigned next_stage ) 
{
   int i;
   for (i=0; i<pipe_simd_width; i++) {
      if (gpgpu_pre_mem_stages) {
         if (shader->pre_mem_pipeline[0][i].hw_thread_id != -1 ) {
            return;  // stalled 
         }
      } else {
         if (shader->pipeline_reg[next_stage][i].hw_thread_id != -1 )
            return;  // stalled 
      }
   }

   check_stage_pcs(shader,ID_EX);

   // Check that all threads have the same delay cycles
   unsigned cycles = -1;
   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[pipeline][i].hw_thread_id == -1 )
         continue; // bubble
      if(cycles == (unsigned)-1)
         cycles = shader->pipeline_reg[pipeline][i].cycles;
      else {
         if( cycles != shader->pipeline_reg[pipeline][i].cycles ) {
            printf("Shader %d: threads do not have the same delay cycles.\n", shader->sid);
            assert(0);
         }
      }
   }

   for (i=0; i<pipe_simd_width; i++) {
      if (shader->pipeline_reg[pipeline][i].hw_thread_id == -1 )
         continue; // bubble

      // Stall based on delay cycles
      shader->pipeline_reg[pipeline][i].cycles--;
      if( shader->pipeline_reg[pipeline][i].cycles > 0 )
         continue;

      if (gpgpu_pre_mem_stages) {
         shader->pre_mem_pipeline[0][i] = shader->pipeline_reg[pipeline][i];
         shader->pre_mem_pipeline[0][i].ex_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      } else {
         shader->pipeline_reg[next_stage][i] = shader->pipeline_reg[pipeline][i];
         shader->pipeline_reg[next_stage][i].ex_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      }
      shader->pipeline_reg[pipeline][i] = nop_inst;
   }  

   if (!gpgpu_pre_mem_stages) {
      // inform memory stage that a new instruction has arrived 
      shader->shader_memory_new_instruction_processed = 0;
   }
}

void shader_execute( shader_core_ctx_t *shader, unsigned int shader_number ) 
{
   shader_execute_pipe(shader,shader_number, OC_EX_SFU, EX_MM);
   shader_execute_pipe(shader,shader_number, ID_EX, EX_MM);
}

void shader_pre_memory( shader_core_ctx_t *shader, 
                        unsigned int shader_number ) {
   int i,j;


   for (j = gpgpu_pre_mem_stages; j > 0; j--) {
      for (i=0; i<pipe_simd_width; i++) {
         if (shader->pre_mem_pipeline[j][i].hw_thread_id != -1 ) {
            return; 
         }
      }
      check_pm_stage_pcs(shader,j-1);
      for (i=0; i<pipe_simd_width; i++) {
         shader->pre_mem_pipeline[j][i] = shader->pre_mem_pipeline[j - 1][i];
         shader->pre_mem_pipeline[j - 1][i] = nop_inst;
      }
   }
   check_pm_stage_pcs(shader,gpgpu_pre_mem_stages);
   for (i=0;i<pipe_simd_width ;i++ )
      shader->pipeline_reg[EX_MM][i] = shader->pre_mem_pipeline[gpgpu_pre_mem_stages][i];

   // inform memory stage that a new instruction has arrived 
   shader->shader_memory_new_instruction_processed = 0;

   if (gpgpu_pre_mem_stages) {
      for (i=0; i<pipe_simd_width; i++)
         shader->pre_mem_pipeline[0][i] = nop_inst;
   }
}

int gpgpu_coalesce_arch;

enum memory_path {
   NO_MEM_PATH = 0,
   SHARED_MEM_PATH,
   GLOBAL_MEM_PATH,
   TEXTURE_MEM_PATH,
   CONSTANT_MEM_PATH,
   NUM_MEM_PATHS //not a mem path
};

static unsigned next_access_uid = 0;   

class mem_access_t{
public:
   mem_access_t(): uid(next_access_uid++),addr(0),req_size(0),order(0),_quarter_count_all(0),warp_indices(),space(undefined_space),path(NO_MEM_PATH),isatomic(false),cache_hit(false),cache_checked(false),recheck_cache(false),iswrite(false),need_wb(false),wb_addr(0),reserved_mshr(NULL){};
   bool operator<(const mem_access_t &other) const {return (order > other.order);}//this is reverse
   unsigned uid;
   address_type addr; //address of the segment to load.
   unsigned req_size; //bytes
   unsigned order; // order of accesses, based on banks.
   union{
   unsigned _quarter_count_all;
   char quarter_count[4]; //access counts to each quarter of segment, for compaction;
   };
   std::vector<unsigned> warp_indices; //warp indicies for this request.
   memory_space_t space;
   memory_path path;
   bool isatomic;
   bool cache_hit;
   bool cache_checked;
   bool recheck_cache;
   bool iswrite;
   bool need_wb;
   address_type wb_addr; //address to wb too if necessary.
   mshr_entry_t* reserved_mshr;
};

mshr_entry_t* mshr_shader_unit::add_mshr(mem_access_t &access, inst_t* warp)
{
   static unsigned next_request_uid = 1;
   mshr_entry_t* mshr = alloc_free_mshr(is_tex(access.space));
   //note no constructor was called, all entries must be reinitialized!
   mshr->request_uid = next_request_uid++;
   mshr->status = INITIALIZED;
   mshr->addr = access.addr;
   mshr->mf = NULL;
   mshr->merged_on_other_reqest = false;
   mshr->merged_requests =NULL;
   mshr->iswrite = access.iswrite;
   assert(access.warp_indices.size()); //code assumes at least one instruction attached to mshr.
   for (unsigned i = 0; i < access.warp_indices.size(); i++) {
      mshr->insts.push_back(warp[access.warp_indices[i]]);
   }
   mshr->islocal = is_local(access.space);
   mshr->isconst = is_const(access.space);
   mshr->istexture = is_tex(access.space);
   if (gpgpu_interwarp_mshr_merge) {
      mshr_entry_t* mergehit = m_mshr_lookup.shader_get_mergeable_mshr(mshr);
      if (mergehit) {
         //merge this request;
         mergehit->merged_requests = mshr;
         mshr->merged_on_other_reqest = true;
         if (mergehit->fetched()) mshr_return_from_mem(mshr);
      }
      m_mshr_lookup.mshr_fast_lookup_insert(mshr);
   }
   return mshr;
}


inline address_type line_size_based_tag_func(address_type address, unsigned line_size)
{
   return ((address) & (~((address_type)line_size - 1)));
}

inline address_type null_tag_func(address_type address, unsigned line_size){
   return address; //no modification: each address is its own tag. Equivalent to line_size_based_tag_func(address,1), but line_size ignored.
}

// only 1 bank
inline int null_bank_func(address_type add, unsigned line_size)
{
   return 1;
}

inline int shmem_bank_func(address_type add, unsigned line_size)
{
   return shmem_bank(add);
}

inline int dcache_bank_func(address_type add, unsigned line_size)
{
   if (gpgpu_no_dl1) return 1; //no banks
   else return (add / line_size) & (gpgpu_n_cache_bank - 1);
}

#include <bitset>
void check_accessq(  shader_core_ctx_t *shader, std::vector<mem_access_t> &accessq ){
   std::bitset<32> check = 0;
   for (unsigned i = 0; i < accessq.size(); i++) {
      if (shader) {
         std::cout << shader->sid << ":" << i << " space " << accessq[i].space.get_type() << " " << gpu_sim_cycle <<  std::endl;
         assert(accessq[i].space == shader->pipeline_reg[EX_MM][accessq[i].warp_indices[0]].space);
      }
      for (unsigned j = 0; j < accessq[i].warp_indices.size(); j++) {
         if (check[accessq[i].warp_indices[j]]) {
            std::cout << "OOOPS" << std::endl; //good line for breakpoint
         }else{check[accessq[i].warp_indices[j]] = true;}
      }
   }
}

// This speciallized function calculates the list of independant memory accesses, sorted by access order
// Acesses to same tag line are coalesced.
// will neither coalesce nor overlap bank accesses accross warp parts.
template < int (*bank_func)(address_type add, unsigned line_size), address_type (*tag_func)(address_type add, unsigned line_size) >
inline void get_memory_access_list(inst_t* insns, unsigned char* paths, memory_path path, unsigned warp_parts, unsigned line_size, bool limit_broadcast,std::vector<mem_access_t> &accessq)
{
   // calculates the memory accesses for a generic cache with banks and tags.
   // can be used for coalesescing 

   //tracks bank accesses for sorting into generations;
   static std::map<unsigned,unsigned> bank_accs;
   bank_accs.clear();
   //keep track of broadcasts with unique orders if limit_broadcast
   //the normally calculated orders will never be greater than pipe_simd_width;
   unsigned broadcast_order =  pipe_simd_width;

   unsigned qbegin = accessq.size();
   unsigned qpartbegin = qbegin;
   unsigned mem_pipe_size = pipe_simd_width / warp_parts;
   for (unsigned part = 0; part < (unsigned)pipe_simd_width; part += mem_pipe_size) {
      for (unsigned i = part; i < part + mem_pipe_size; i++) {
         if (paths[i] != path) continue; //skip instructions from other memory paths
         address_type segment = (*tag_func)(insns[i].memreqaddr, line_size);
         unsigned quarter=0;
         if ( line_size>=4 ) {
           quarter = (insns[i].memreqaddr / (line_size/4)) & 3;
         } 
         //check if we are already loading this segment. 
         bool isatomic = (insns[i].callback.function != NULL);
         unsigned match = 0;
         if (not isatomic) { //atomics must have own request
            for (unsigned j = qpartbegin; j < accessq.size(); j++) {
               if (segment == accessq[j].addr and not accessq[j].isatomic) {
                  //match
                  accessq[j].quarter_count[quarter]++;
                  accessq[j].warp_indices.push_back(i);
		  if (limit_broadcast) accessq[j].order = ++broadcast_order; //do proadcast in its own cycle.
                  match = 1;
                  break;
               }
            }
         }
         if (!match) {
            //needs its own request
            accessq.push_back(mem_access_t());
            accessq.back().addr = segment; 
            accessq.back().space = insns[i].space;
            accessq.back().path = path;
            accessq.back().isatomic = isatomic;
            accessq.back().iswrite = is_store(insns[i].op);
            accessq.back().req_size = line_size;
            accessq.back().quarter_count[quarter]++;
            accessq.back().warp_indices.push_back(i);

            //Determine Bank Conflicts.
            unsigned bank = (*bank_func)(insns[i].memreqaddr, line_size);
            //ensure no concurrent bank access accross warp parts. 
            // ie. order will be less than part for all previous loads in previous parts, so:
            if (bank_accs[bank] < part) bank_accs[bank]=part; 
            accessq.back().order = bank_accs[bank];
            bank_accs[bank]++;
         }
      }
      qpartbegin = accessq.size(); //don't coalesce accross warp parts
   }
   //sort requests into order accorting to order (orders will not necessarily be consequtive if multiple parts)
   std::stable_sort(accessq.begin()+qbegin,accessq.end()); //this is a reverse sort, least order last, but doesn't really matter where consumed.
}  


void shader_memory_shared_process_inst(shader_core_ctx_t * shader, unsigned char* paths, std::vector<mem_access_t> &accessq)
{
   get_memory_access_list<&shmem_bank_func, &null_tag_func>(shader->pipeline_reg[EX_MM], paths, SHARED_MEM_PATH, 
                          gpgpu_shmem_pipe_speedup, 
                          1, //shared memory doesn't care about line_size, needs to be at least 1;
			  true, //limit broadcasts to single cycle. 
                          accessq);
   //thats it :)
}

void shader_memory_const_process_inst(shader_core_ctx_t * shader, unsigned char* paths, std::vector<mem_access_t> &accessq)
{
   unsigned qbegin = accessq.size();
   get_memory_access_list<&null_bank_func, &line_size_based_tag_func>(shader->pipeline_reg[EX_MM], paths, CONSTANT_MEM_PATH, 
                          1, //warp parts
                          shader->L1constcache->line_sz,
			  false, //no broadcast limit.
                          accessq);
   //do cache checks here for each request, could be done later for more accurate timing of cache accesses, but probably uneccesary; 
   for (unsigned i = qbegin; i < accessq.size(); i++) {
      if (is_param(accessq[i].space)) {
         accessq[i].cache_hit = true;
      } else {
         cache_request_status status = shd_cache_access_wb(shader->L1constcache,
                                                            accessq[i].addr,
                                                            WORD_SIZE, //this field is ingored.
                                                            0, //should always be a read
                                                            shader->gpu_cycle,
							    NULL/*should never writeback*/);
         accessq[i].cache_hit = (status == HIT);
         if (gpgpu_perfect_mem) accessq[i].cache_hit = true;
	 if (accessq[i].cache_hit) L1_const_miss++;
      } 
      accessq[i].cache_checked = true;
   }
}

void shader_memory_texture_process_inst(shader_core_ctx_t * shader, unsigned char* paths, std::vector<mem_access_t> &accessq)
{
   unsigned qbegin = accessq.size();
   get_memory_access_list<&null_bank_func, &line_size_based_tag_func>(shader->pipeline_reg[EX_MM], paths, TEXTURE_MEM_PATH, 
                          1, //warp parts
                          shader->L1texcache->line_sz,
			  false, //no broadcast limit.
                          accessq);
   //do cache checks here for each request, could be done later for more accurate timing of cache accesses, but probably uneccesary; 
   for (unsigned i = qbegin; i < accessq.size(); i++) {
      cache_request_status status = shd_cache_access_wb(shader->L1texcache,
                                                         accessq[i].addr,
                                                         WORD_SIZE, //this field is ignored.
                                                         0, //should always be a read
                                                         shader->gpu_cycle,
							 NULL /*should never writeback*/);
      accessq[i].cache_hit = (status == HIT);
      if (gpgpu_perfect_mem) accessq[i].cache_hit = true;
      if (accessq[i].cache_hit) L1_texture_miss++;
      accessq[i].cache_checked = true;
   }
}

void shader_memory_global_process_inst(shader_core_ctx_t * shader, unsigned char* paths, std::vector<mem_access_t> &accessq)
{
   unsigned qbegin = accessq.size();
   unsigned warp_parts = 1;
   unsigned line_size = shader->L1cache->line_sz;
   if (gpgpu_coalesce_arch == 13) {
      warp_parts = 2;
      if(gpgpu_no_dl1) {
         int valindex = -1;
         for (int i = 0; i < pipe_simd_width; i++) {
            if (paths[i] == GLOBAL_MEM_PATH) {
               valindex = i;
               break;
            }
         }
         assert(valindex != -1);
         // line size is dependant on instruction;
         //assume first valid thread instruction is the same as the rest.
         switch (shader->pipeline_reg[EX_MM][valindex].data_size) {
         case 1:
            line_size = 32;
            break;
         case 2:
            line_size = 64;
            break;
         case 4:
         case 8:
         case 16:
            line_size = 128;
            break;
         default:
            assert(0);
         }
      }
   }          
   get_memory_access_list<&dcache_bank_func, &line_size_based_tag_func>(shader->pipeline_reg[EX_MM], paths, GLOBAL_MEM_PATH, 
                          warp_parts, //warp parts
                          line_size,
                          false, //no broadcast limit.
                          accessq);

   for (unsigned i = qbegin; i < accessq.size(); i++) {
      if (gpgpu_coalesce_arch == 13 and gpgpu_no_dl1) {
         //if there is no l1 cache it makes sense to do coalescing here.
         //reduce memory request sizes.
         char* quarter_counts = accessq[i].quarter_count;
         bool low = quarter_counts[0] or quarter_counts[1];
         bool high = quarter_counts[2] or quarter_counts[3];
         if (accessq[i].req_size == 128) {
            if (low xor high) { //can reduce size
               accessq[i].req_size = 64;
               if (high) accessq[i].addr += 64;
               low = quarter_counts[0] or quarter_counts[2]; //set low and high for next pass
               high = quarter_counts[1] or quarter_counts[3];
            }
         }
         if (accessq[i].req_size == 64) {
            if (low xor high) { //can reduce size
               accessq[i].req_size = 32;
               if (high) accessq[i].addr += 32;
            }
         }
      }
   }
}
      


mem_stage_stall_type send_mem_request(shader_core_ctx_t *shader, mem_access_t &access){
   inst_t* warp = shader->pipeline_reg[EX_MM];
   inst_t* req_head = warp + access.warp_indices[0];

   if (access.need_wb) {
      //fill out and send a writeback
      unsigned req_size = shader->L1cache->line_sz + WRITE_PACKET_SIZE;
      if (!(shader->fq_has_buffer(access.wb_addr, req_size, true, shader->sid))) {
         gpu_stall_sh2icnt++; 
         return WB_ICNT_RC_FAIL;
      }

      shader->fq_push( access.wb_addr,
                       req_size,
                       true, NO_PARTIAL_WRITE, shader->sid, -1, NULL, 
                       0, 
		       is_local(access.space)?LOCAL_ACC_W:GLOBAL_ACC_W, //space of cache line is same as new request
		        -1);
      L1_writeback++;
      access.need_wb = false; 
   }

   bool requires_mshr = (shader->model != MIMD) and (not access.iswrite);

   //this decoding here might belong elsewhere
   unsigned code;
   mem_access_type  access_type;
   switch(access.space.get_type()) {
   case const_space:
   case param_space_kernel:
      code = CONSTC;
      access_type = CONST_ACC_R;   
      break;
   case tex_space:
      code = TEXTC;
      access_type = TEXTURE_ACC_R;   
      break;
   case global_space:
      code = DCACHE;
      access_type = (access.iswrite)? GLOBAL_ACC_W: GLOBAL_ACC_R;   
      break;
   case local_space:
   case param_space_local:
      code = DCACHE;
      access_type = (access.iswrite)? LOCAL_ACC_W: LOCAL_ACC_R;   
      break;
   default:
      assert(0); // NOT A MEM SPACE;
      break; 
   }

   //reserve mshr
   if (requires_mshr and not access.reserved_mshr) {

      // can allocate mshr?
      if (not shader->mshr_unit->has_mshr(1)) {
         //no mshr available;
         return MSHR_RC_FAIL;
      }

      access.reserved_mshr = shader->mshr_unit->add_mshr(access, warp);
      access.recheck_cache = false; //we have an mshr now, so have checked cache in same cycle as checking mshrs, so have merged if necessary.
   }

   //require inct if access is this far without reserved mshr, or has and mshr but not merged with another request
   bool requires_icnt = (not access.reserved_mshr) or (not access.reserved_mshr->merged_on_other_reqest);

   if (requires_icnt) {

      //calculate request size for icnt check (and later send);
      unsigned request_size = access.req_size;
      if (access.iswrite) {
         if (requires_mshr) {
            //needs information for a load back into cache.
            request_size += READ_PACKET_SIZE + WRITE_MASK_SIZE;
         } else {
            //plain write
            request_size += WRITE_PACKET_SIZE + WRITE_MASK_SIZE;
         }
      }


      // can allocate icnt?
      //unsigned char fq_has_buffer(unsigned long long int addr, int bsize, bool write, int sid);
      if (!(shader->fq_has_buffer(access.addr, request_size, access.iswrite, shader->sid))) {
         gpu_stall_sh2icnt++;
	 //std::cout<< "failed to push " << request_size << " bytes" << std::endl;
         return ICNT_RC_FAIL;
      }

      //send over interconnect

      unsigned cache_hits_waiting = 0; //fixme do we really want to be passing this in?

      partial_write_mask_t  write_mask = NO_PARTIAL_WRITE;
      if (access.iswrite) {
         for (unsigned i=0;i < access.warp_indices.size();i++) {
            unsigned w = access.warp_indices[i];
            int data_offset = warp[w].memreqaddr & ((unsigned long long int)access.req_size - 1);
            for (unsigned b = data_offset; b < data_offset + warp[w].data_size; b++) write_mask.set(b);
         }
         if (write_mask.count() != access.req_size) {
            gpgpu_n_partial_writes++;
         }
      }

      //typedef unsigned char (*fq_push_t)(unsigned long long int addr, int bsize, unsigned char readwrite,
      //                             unsigned long long int partial_write_mask, 
      //                             int sid, int wid, mshr_entry* mshr, int cache_hits_waiting,  
      //                             enum mem_access_type mem_acc, address_type pc);
      shader->fq_push( access.addr,
                       request_size,
                       access.iswrite, write_mask, shader->sid, req_head->hw_thread_id/warp_size, access.reserved_mshr, 
                       cache_hits_waiting, access_type, req_head->pc);

   }


   //book keeping for mshr since this request is done (sent/accounted for) at this point;
   if (requires_mshr) {

      for (unsigned i = 0; i < access.warp_indices.size(); i++) {
         unsigned o = access.warp_indices[i];
         shader->pending_mem_access++;
         inflight_memory_insn_add(shader, &warp[o]);
         
#if 0    //old stats
         if (i > 0) { //maintain old stats (yes/no?)
            shader->thread[warp[o].hw_thread_id].n_l1_mrghit_ac++; 
            shd_cache_mergehit(shader->L1texcache, warp[o].memreqaddr); //fixme;
         }
#endif   
      }

      // Scoreboard addition: do not make cache miss instructions wait for memory,
      //                      let the scoreboard handle stalling of instructions.
      //                      Mark thread as a cache miss

      if (not access.iswrite) {
         // set the pipeline instructions in this request to noops, they all wait for memory;
         for (unsigned i = 0; i < access.warp_indices.size(); i++) {
            unsigned o = access.warp_indices[i];
            //shader->pipeline_reg[EX_MM][o] = nop_inst;
            shader->pipeline_reg[EX_MM][o].cache_miss = true;
         }
      }

   }

   return NO_RC_FAIL;
}     


bool shader_memory_shared_cycle( shader_core_ctx_t *shader, std::vector<mem_access_t> &accessq, 
                            mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type){
   //consume port number orders from the top of the queue;
   for (unsigned i = 0; i < (unsigned) gpgpu_shmem_port_per_bank; i++) {
      if (accessq.empty()) break;
      unsigned current_order = accessq.back().order;
      //consume all requests of the same order (concurrent bank requests)
      while ((not accessq.empty()) and accessq.back().order == current_order) accessq.pop_back();
   }
   if (not accessq.empty()) {
      rc_fail = BK_CONF;
      fail_type = S_MEM;
      gpgpu_n_shmem_bkconflict++;
   }
   return accessq.empty(); //done if empty.
}

//generic memory access queue processing, accessq must be sorted by order
//--that is, requests of similar order are expected to be contiguous in the queueu.
//if you want to use this for shared memory, make sure they are marked as cashe hits (not default)
// cycle_exec may be called multiple times if memory fails. typically used for cache checks
template < mem_stage_stall_type (*cycle_exec)(shader_core_ctx_t*, mem_access_t&) >
inline mem_stage_stall_type shader_memory_generic_process_queue( shader_core_ctx_t *shader, 
                                                         unsigned ports_per_bank, unsigned memory_send_max, 
                                                         std::vector<mem_access_t> &accessq ){
   mem_stage_stall_type rc_fail = NO_RC_FAIL; 
   // number of requests to sent to memory this cycle
   unsigned mem_req_count = 0;
   //consume port number orders from the top of the queue;
   for (unsigned i = 0; i < ports_per_bank; i++) {
      if (accessq.empty()) break;
      unsigned current_order = accessq.back().order;
      //consume all requests of the same order (concurrent bank requests)
      //stop when things that go to memory exceed a per cycle limit.
      while ((not accessq.empty()) and accessq.back().order == current_order and rc_fail == NO_RC_FAIL) {
         rc_fail = (*cycle_exec)(shader, accessq.back());
         if (rc_fail != NO_RC_FAIL) break; //can't complete this request this cycle.
         if (not accessq.back().cache_hit){
            if (mem_req_count < memory_send_max) {
               mem_req_count++;
               rc_fail = send_mem_request(shader, accessq.back()); //try to get mshr, icnt, send;
            }
            else {
                rc_fail = COAL_STALL; //not really a coal stall, its a too many memory request stall;
            }
            if (rc_fail != NO_RC_FAIL) break; //can't complete this request this cycle.
         }
         accessq.pop_back();
      }
   }
   if (not accessq.empty() and rc_fail == NO_RC_FAIL) {
      //no resource failed so must be a bank comflict.
      rc_fail = BK_CONF;
   }
   return rc_fail;
}  

mem_stage_stall_type ccache_check(shader_core_ctx_t *shader, mem_access_t& access){ /*done in process queue*/ return NO_RC_FAIL;}

bool shader_memory_constant_cycle( shader_core_ctx_t *shader, std::vector<mem_access_t> &accessq, 
                            mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type){

   mem_stage_stall_type fail = shader_memory_generic_process_queue<ccache_check>( shader, gpgpu_const_port_per_bank, 
                                                                   1, //memory send max per cycle
                                                                   accessq );
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = C_MEM;
      if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
         gpgpu_n_cmem_portconflict++; //coal stalls aren't really a bank conflict, but this maintains previous behavior.
      }
   }
   return accessq.empty(); //done if empty.
}

mem_stage_stall_type tcache_check(shader_core_ctx_t *shader, mem_access_t& access){ /*done in process queue*/ return NO_RC_FAIL;}

bool shader_memory_texture_cycle( shader_core_ctx_t *shader, std::vector<mem_access_t> &accessq, 
                            mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type){

   mem_stage_stall_type fail = shader_memory_generic_process_queue<tcache_check>(shader, 1, //how is tex memory banked? 
                                                                   1, //memory send max per cycle
                                                                   accessq );
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = T_MEM;
   }
   return accessq.empty(); //done if empty.
}


mem_stage_stall_type dcache_check(shader_core_ctx_t *shader, mem_access_t& access){
   if (access.cache_checked and not access.recheck_cache) return NO_RC_FAIL;
   if (!gpgpu_no_dl1 && !gpgpu_perfect_mem) { 
      //check cache
      cache_request_status status = shd_cache_access_wb(shader->L1cache,
                                                         access.addr,
                                                         WORD_SIZE, //this field is ignored.
                                                         access.iswrite,
                                                         shader->gpu_cycle,
                                                         &access.wb_addr);
      if (status == RESERVATION_FAIL) {
         access.cache_checked = false;
         return WB_CACHE_RSRV_FAIL;
      }
      access.cache_hit = (status == HIT); //if HIT_W_WT then still send to memory so "MISS" 
      if (status == MISS_W_WB) access.need_wb = true;
      if (status == WB_HIT_ON_MISS and access.iswrite) 
      {
         //write has hit a reserved cache line
         //it has writen its data into the cache line, so no need to go to memory
         access.cache_hit = true;
         L1_write_hit_on_miss++;
         // here we would search the MSHRs for the originating read, 
         // and mask off the writen bytes, so they are not overwritten in the cache when it comes back
         // --- don't actually do this since we are pretending.
         // MSHR will still forward the unmasked value to its dependant reads. 
         // if doing stall on use, must stall this thread after this write (otherwise, inproper values may be forwarded to future reads).
      }
      if (status == WB_HIT_ON_MISS and not access.iswrite) {
         //read has hit on a reserved cache line, 
         //we need to make sure cache check happens on same cycle as a mshr merge happens, otherwise we might miss it coming back
         access.recheck_cache = true;
      }
      access.cache_checked = true;
   } else {
      access.cache_hit = false;
   }

   if (gpgpu_perfect_mem) access.cache_hit = true;
   
   //atomics always go to memory 
   if (access.isatomic) {
      if (!gpgpu_perfect_mem) {
         access.cache_hit = false;
      } else {
         //unless perfect mem, in which case, the callback can only be done here 
         dram_callback_t &atom_exec = shader->pipeline_reg[EX_MM][access.warp_indices[0]].callback;
         atom_exec.function(atom_exec.instruction, atom_exec.thread);
      }
   }

   if (!access.cache_hit) { 
      if (access.iswrite) L1_write_miss++;
      else L1_read_miss++;
   }
   return NO_RC_FAIL;
}

bool shader_memory_global_cycle( shader_core_ctx_t *shader, std::vector<mem_access_t> &accessq, 
                            mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type){
   mem_stage_stall_type fail = shader_memory_generic_process_queue<&dcache_check>(shader, gpgpu_cache_port_per_bank, 
                                                                   1, //memory send max per cycle
                                                                   accessq );
   if (fail != NO_RC_FAIL) {
      rc_fail = fail; //keep other fails if this didn't fail.
      //need to determine load/store, local/global:
      bool iswrite = accessq.back().iswrite;
      if (is_local(accessq.back().space)) {
         fail_type = (iswrite)?L_MEM_ST:L_MEM_LD;
      } else {
         fail_type = (iswrite)?G_MEM_ST:G_MEM_LD;
      }

      if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
         gpgpu_n_cache_bkconflict++;
      }
   }
   return accessq.empty(); //done if empty.
}

inline void mem_instruction_stats(inst_t* warp){
   //there must be a better way to count these
   for (unsigned i=0; i< (unsigned) pipe_simd_width; i++) {
      if (warp[i].hw_thread_id == -1) continue; //bubble 
         //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
      bool store = is_store(warp[i].op);
      switch (warp[i].space.get_type()) {
      case undefined_space:
      case reg_space:
         break;
      case shared_space:
         gpgpu_n_shmem_insn++;
         break;
      case const_space:
         gpgpu_n_const_insn++;
         break;
      case param_space_kernel:
      case param_space_local:
         gpgpu_n_param_insn++;
         break;
      case tex_space:
         gpgpu_n_tex_insn++;
         break;
      case global_space:
      case local_space:
         if (store){ 
            gpgpu_n_store_insn++;
         } else {
            gpgpu_n_load_insn++;
         }
         break;
      default:
         abort();
      }
   }
}

struct shader_queues_t{
   std::vector<mem_access_t> shared;
   std::vector<mem_access_t> constant;
   std::vector<mem_access_t> texture;
   std::vector<mem_access_t> global;
};

void shader_memory_queue(shader_core_ctx_t *shader, shader_queues_t *accessqs)
{
      //classify memory according to type;
      static unsigned char *path = NULL;
      if (!path) path = (unsigned char*)malloc(pipe_simd_width * sizeof(unsigned char));
      memset(path, 0, pipe_simd_width * sizeof(unsigned char));
      //static std::vector<char> path;
      //path.clear(); path.resize(p, NO_MEM_PATH);

      static unsigned type_counts[NUM_MEM_PATHS];
      memset(type_counts, 0, sizeof(type_counts));
      //static std::vector<unsigned> type_counts;
      //type_counts.clear(); type_counts.resize(NUM_MEM_PATHS);

      for (unsigned i=0; i< (unsigned) pipe_simd_width; i++) {
         if (shader->pipeline_reg[EX_MM][i].hw_thread_id == -1) continue; //bubble 
         //this breaks some encapsulation: the is_[space] functions; if you change those, change this.
         switch (shader->pipeline_reg[EX_MM][i].space.get_type()) {
         case shared_space:
            path[i] = SHARED_MEM_PATH;
            type_counts[SHARED_MEM_PATH]++;
            break;
         case const_space:
         case param_space_kernel:
            path[i] = CONSTANT_MEM_PATH;
            type_counts[CONSTANT_MEM_PATH]++;   
            break;
         case tex_space:
            path[i] = TEXTURE_MEM_PATH;
            type_counts[TEXTURE_MEM_PATH]++;
            break;
         case global_space:
         case local_space:
         case param_space_local:
            path[i] = GLOBAL_MEM_PATH;
            type_counts[GLOBAL_MEM_PATH]++;
            break;
         case param_space_unclassified:
            abort(); // todo: define access details
            break;
         default:
            break;
         }
      }

      //instruction counting:
      mem_instruction_stats(shader->pipeline_reg[EX_MM]);
      

      if (type_counts[SHARED_MEM_PATH]) shader_memory_shared_process_inst(shader, path, accessqs->shared);
      if (type_counts[CONSTANT_MEM_PATH]) shader_memory_const_process_inst(shader, path, accessqs->constant);
      if (type_counts[TEXTURE_MEM_PATH]) shader_memory_texture_process_inst(shader, path, accessqs->texture);
      if (type_counts[GLOBAL_MEM_PATH]) shader_memory_global_process_inst(shader, path, accessqs->global);

}


void shader_memory( shader_core_ctx_t *shader, unsigned int shader_number )
{
   enum mem_stage_stall_type rc_fail = NO_RC_FAIL; // resource allocation

   //these should be local to the shader structure but can't because it is included in non c++ files.
   //so provide static storage for it here
   static std::vector<shader_queues_t> shader_memory_queues;
   if (shader_memory_queues.size() == 0) {
      shader_memory_queues.resize(gpu_n_shader);
      for (unsigned i = 0; i < gpu_n_shader; i++) {
         shader_memory_queues[i].shared.reserve(pipe_simd_width);
         shader_memory_queues[i].constant.reserve(pipe_simd_width);
         shader_memory_queues[i].texture.reserve(pipe_simd_width);
         shader_memory_queues[i].global.reserve(pipe_simd_width);
      }
   }
   shader_queues_t *accessqs = &(shader_memory_queues[shader->sid]);

   if (shader->shader_memory_new_instruction_processed == 0) {
      shader->shader_memory_new_instruction_processed = 1; //only do this once per pipeline occupant
      shader_memory_queue(shader, accessqs);
   }

   bool done = true;
   mem_stage_access_type type;

   done &= shader_memory_shared_cycle(shader, accessqs->shared, rc_fail, type);
   done &= shader_memory_constant_cycle(shader, accessqs->constant, rc_fail, type);
   done &= shader_memory_texture_cycle(shader, accessqs->texture, rc_fail, type);
   done &= shader_memory_global_cycle(shader, accessqs->global, rc_fail, type);

   //wb stalled?
   int wb_stalled = 0; // check if next stage is stalled
   for (unsigned i=0; i< (unsigned) pipe_simd_width; i++) {
      if (shader->pipeline_reg[MM_WB][i].hw_thread_id != -1 ) {
         wb_stalled = 1;
         break;
      }
   }

   if (!done) {
      assert(rc_fail != NO_RC_FAIL);
      //log stall types
      gpu_stall_shd_mem++;
      gpu_stall_shd_mem_breakdown[type][rc_fail]++;
   }

   if (!done or wb_stalled) return;

   // this memory stage is done and not stalled by wb
   // pipeline forward

   check_stage_pcs(shader,EX_MM);
   // and pass instruction from EX_MM to MM_WB
   for (unsigned i=0; i< (unsigned) pipe_simd_width; i++) {
      if (shader->pipeline_reg[EX_MM][i].hw_thread_id == -1 )
         continue; // bubble
      shader->pipeline_reg[MM_WB][i] = shader->pipeline_reg[EX_MM][i];
      shader->pipeline_reg[MM_WB][i].mm_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
      shader->pipeline_reg[EX_MM][i] = nop_inst;
   }    

   // reflect the change to EX|MM pipeline register to the pre_mem stage
   if (gpgpu_pre_mem_stages) {
      check_stage_pcs(shader,EX_MM);
      for (unsigned i=0;i< (unsigned)pipe_simd_width ;i++ )
         shader->pre_mem_pipeline[gpgpu_pre_mem_stages][i] = shader->pipeline_reg[EX_MM][i];
   }
}

int writeback_l1_miss =0 ;


void register_cta_thread_exit(shader_core_ctx_t *shader, int tid )
{
   if (gpgpu_cuda_sim && gpgpu_spread_blocks_across_cores) {
      unsigned padded_cta_size = ptx_sim_cta_size();
      if (padded_cta_size%warp_size) {
         padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
      }
      int cta_num = tid/padded_cta_size;
      assert( shader->cta_status[cta_num] > 0 );
      shader->cta_status[cta_num]--;
      if (!shader->cta_status[cta_num]) {
         shader->n_active_cta--;
         shader->deallocate_barrier(cta_num);
         shader_CTA_count_unlog(shader->sid, 1);
         printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld)\n", shader->sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle );
      }
   }
}

void obtain_insn_latency_info(insn_latency_info *latinfo, inst_t *insn)
{
   latinfo->pc = insn->pc;
   latinfo->latency = gpu_tot_sim_cycle + gpu_sim_cycle - insn->ts_cycle;
   latinfo->ptx_thd_info = insn->ptx_thd_info;
}

int debug_tid = 0;

unsigned   gpu_n_max_mshr_writeback=1;
void shader_writeback( shader_core_ctx_t *shader, unsigned int shader_number, int grid_num ) 
{
   std::vector<inst_t> done_insts;

   static int *mshr_tid = NULL;
   static int *pl_tid = NULL;

   std::vector<insn_latency_info> unlock_lat_infos;
   static insn_latency_info *mshr_lat_info = NULL;
   static insn_latency_info *pl_lat_info = NULL;

   mshr_entry *mshr_head = NULL;

   int tid;
   op_type op;
   int o1, o2, o3, o4;
   bool stalled_by_MSHR = false;
   bool writeback_by_MSHR = false;
   bool w2rf = false;

   if ( mshr_tid == NULL ) {
      mshr_tid = (int*) malloc(sizeof(int)*pipe_simd_width);
      pl_tid = (int*) malloc(sizeof(int)*pipe_simd_width);
      mshr_lat_info = (insn_latency_info*) malloc(sizeof(insn_latency_info) * pipe_simd_width);
      pl_lat_info = (insn_latency_info*) malloc(sizeof(insn_latency_info) * pipe_simd_width);
   }

   memset(mshr_tid,   -1, sizeof(int)*pipe_simd_width);
   memset(pl_tid,     -1, sizeof(int)*pipe_simd_width);


   check_stage_pcs(shader,MM_WB);

   /* Generate Condition for instruction writeback to register file. */
   for (int i=0; i<pipe_simd_width; i++) {
	  w2rf |= (shader->pipeline_reg[MM_WB][i].hw_thread_id >= 0);
	  pl_tid[i] = shader->pipeline_reg[MM_WB][i].hw_thread_id;
   }

   //check mshrs for commit;
   unsigned mshr_threads_unlocked = 0;
   for (unsigned i = 0; i < gpu_n_max_mshr_writeback; i++) {
      mshr_head = shader->mshr_unit->return_head();
      if (mshr_head) {
         //bail if we can't unlock anymore threads
         if (mshr_threads_unlocked + mshr_head->insts.size() > (unsigned) pipe_simd_width) break;
         assert(!gpgpu_strict_simd_wrbk);//implementation removed
         assert (mshr_head->insts.size());
         for (unsigned j = 0; j < mshr_head->insts.size(); j++) {
            inst_t &insn = mshr_head->insts[j];
            time_vector_update(insn.uid,MR_WRITEBACK,gpu_sim_cycle+gpu_tot_sim_cycle,RD_REQ);
            obtain_insn_latency_info(&mshr_lat_info[mshr_threads_unlocked], &(mshr_head->insts[j]));
            inflight_memory_insn_sub(shader, &mshr_head->insts[j]);
            assert (insn.hw_thread_id >= 0);
            shader->pending_mem_access--;
            // for ensuring that we don't unlock more than the code allows, needs to be fixed.
            mshr_threads_unlocked++;
         }
         done_insts.insert(done_insts.end(), mshr_head->insts.begin(), mshr_head->insts.end());

         shader->mshr_unit->pop_return_head();
         writeback_by_MSHR = true;
         unlock_lat_infos.resize(mshr_threads_unlocked);
         std::copy(mshr_lat_info, mshr_lat_info + mshr_threads_unlocked, unlock_lat_infos.begin());

         if (w2rf) {
            stalled_by_MSHR = true;
         }
         assert(mshr_threads_unlocked);
      }
   }
   if (stalled_by_MSHR) {
      gpu_stall_by_MSHRwb++;
   }

   if (!writeback_by_MSHR) { //!writeback_by_MSHR
	  memory_space_t warp_space = undefined_space;

      for (int i=0; i<pipe_simd_width; i++) {
         op  = shader->pipeline_reg[MM_WB][i].op;
         tid = shader->pipeline_reg[MM_WB][i].hw_thread_id;
         o1  = shader->pipeline_reg[MM_WB][i].out[0];
         o2  = shader->pipeline_reg[MM_WB][i].out[1];
         o3  = shader->pipeline_reg[MM_WB][i].out[2];
         o4  = shader->pipeline_reg[MM_WB][i].out[3];

         obtain_insn_latency_info(&pl_lat_info[i], &shader->pipeline_reg[MM_WB][i]);

         // Collect threads that are done
         // Do not include cache misses for a writeback
         if(!shader->pipeline_reg[MM_WB][i].cache_miss) {
			 if(shader->pipeline_reg[MM_WB][i].hw_thread_id > -1) {
				 done_insts.push_back(shader->pipeline_reg[MM_WB][i]);
				 unlock_lat_infos.push_back(pl_lat_info[i]);
			 }
         }

         // All threads in the warp should have the same pc and space
         if(pl_tid[i] > -1 ) {
        	 warp_space = shader->pipeline_reg[MM_WB][i].space;
         }

         if(tid > -1) {
/*
        	 if(!shader->pipeline_reg[MM_WB][i].cache_miss)
        		 printf("CACHE HIT sid=%d tid=%d pc=%d \n", shader->sid, tid, shader->pipeline_reg[MM_WB][i].pc);
        	 else
        		 printf("CACHE MISS sid=%d tid=%d pc=%d \n", shader->sid, tid, shader->pipeline_reg[MM_WB][i].pc);
*/
         }
      }

      // Unlock the warp for re-fetching (put it in the fixed delay queue)
      // Only need to unlock if warp is not empty
      if(w2rf)
    	  shader_queue_warp_unlocking(shader, pl_tid, warp_space, grid_num);
   }

   // Mark threads as done in warp tracker
   for (unsigned i=0; i<done_insts.size(); i++) {
	   inst_t done_inst = done_insts[i];

		shader_call_thread_done(shader, grid_num, done_inst);

		// Statistics
		// At any rate, a real instruction is committed
		// - don't count cache miss
		gpu_sim_insn++;
		if ( !is_const(done_inst.space) )
			gpu_sim_insn_no_ld_const++;
		gpu_sim_insn_last_update = gpu_sim_cycle;
		shader->num_sim_insn++;
		shader->thread[done_inst.hw_thread_id].n_insn++;
		shader->thread[done_inst.hw_thread_id].n_insn_ac++;

		if (enable_ptx_file_line_stats) {
		  unsigned pc = unlock_lat_infos[i].pc;
		  unsigned long latency = unlock_lat_infos[i].latency;
		  ptx_file_line_stats_add_latency(unlock_lat_infos[i].ptx_thd_info, pc, latency);
		}

  }

   /* The pipeline can be stalled by MSHR */
   if (!stalled_by_MSHR) {
      for (int i=0; i<pipe_simd_width; i++) {
         shader->pipeline_reg[WB_RT][i] = shader->pipeline_reg[MM_WB][i];
         shader->pipeline_reg[MM_WB][i] = nop_inst;
      }
   }

   // Process the delay queue for current cycle
   shader_process_delay_queue(shader);
}

/*
 * Queues a warp into fixed delay queue for unlocking
 *
 * The amount of delay to add is determined by the instruction type.
 *
 * @param *shader Pointer to shader core
 * @param *tid Array of tid in the warp to unlock
 * @param pc Program counter for the current instruction in the warp
 * @param space Address space for the current instruction in the warp
 *
 */
void shader_queue_warp_unlocking(shader_core_ctx_t *shader, int *tids, memory_space_t space, int grid_num) {

	// Create a delay queue object and add it to the queue
	shader_core_ctx_t::fixeddelay_queue_warp_t fixeddelay_queue_warp;

	fixeddelay_queue_warp.grid_num = grid_num;

	// Set ready_cycle based on instruction space
	switch(space.get_type()) {
		case shared_space:
			fixeddelay_queue_warp.ready_cycle = gpu_tot_sim_cycle + gpu_sim_cycle + 5; // Adds 5*4=20 cycles
			break;
		default:
			fixeddelay_queue_warp.ready_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
			break;
	}

	// Store threads in delay queue warp object
	fixeddelay_queue_warp.tids.resize(warp_size);
	std::copy(tids, tids+warp_size, fixeddelay_queue_warp.tids.begin());

	shader->fixeddelay_queue.insert(fixeddelay_queue_warp);
}

/*
 * Process a delay queue by unlocking warps ready this cycle
 *
 * @param *shader Pointer to shader core
 *
 */
void shader_process_delay_queue(shader_core_ctx_t *shader) {
	// Unlock warps in fixeddelay_queue_warp
	std::multiset<shader_core_ctx_t::fixeddelay_queue_warp_t, shader_core_ctx_t::fixeddelay_queue_warp_comp>::iterator it;
	std::multiset<shader_core_ctx_t::fixeddelay_queue_warp_t, shader_core_ctx_t::fixeddelay_queue_warp_comp>::iterator it_last;
	for ( it=shader->fixeddelay_queue.begin() ;
		  it != shader->fixeddelay_queue.end();
		) {
	   if(it->ready_cycle <= gpu_tot_sim_cycle + gpu_sim_cycle) {
		   if(!gpgpu_stall_on_use) {
			   // This disables stall-on-use
			   // If thread is still in warp_tracker, do not unlock yet
			   bool skip_unlock = false;
			   for(unsigned i=0; i<warp_size; i++) {
				   int tid = it->tids[i];
				   if(tid < 0) continue;
				   if(get_warp_tracker_pool().wpt_thread_in_wpt(shader,tid)) {
					   skip_unlock = true;
					   break;
				   }
			   }
			   if(skip_unlock) {
				   it_last = it++;
				   continue;
			   }
		   }

		   // Unlock warp
		   shader_unlock_warp(shader,it->tids, it->grid_num);

		   // Remove warp information from delay queue
		   it_last = it++;
		   shader->fixeddelay_queue.erase(it_last);
	   } else {
		   break;
	   }
	}
}

/*
 * Unlock a warp
 *
 * @param *shd Pointer to shader core
 * @param tids Vector of tid in the warp to unlock
 *
 */
void shader_unlock_warp(shader_core_ctx_t *shd, std::vector<int> tids, int grid_num) {
    int thd_unlocked = 0;
    int thd_exited = 0;
    int tid;
    int valid_tid = -1;
    // Unlock
    for (unsigned i=0; i<warp_size; i++) {
        tid = tids[i];
		if (tid >= 0) {
			valid_tid = tid;
			// thread completed if it is going to fetching beyond code boundary
			if ( gpgpu_cuda_sim && ptx_thread_done(shd->thread[tid].ptx_thd_info) ) {
				shd->not_completed -= 1;
				gpu_completed_thread += 1;

				int warp_id = wid_from_hw_tid(tid,warp_size);
				if (!(shd->warp[warp_id].n_completed < (unsigned)warp_size)) {
				 printf("shader[%d]->warp[%d].n_completed = %d; warp_size = %d\n",
						shd->sid,warp_id, shd->warp[warp_id].n_completed, warp_size);
				}
				assert( shd->warp[warp_id].n_completed < (unsigned)warp_size );
				shd->warp[warp_id].n_completed++;
				if ( shd->model == NO_RECONVERGE ) {
				 update_max_branch_priority(shd,warp_id,grid_num);
				}

				register_cta_thread_exit(shd, tid );
				thd_exited = 1;

				//printf("THREAD EXIT sid=%d tid=%d \n", shd->sid, tid);

			} else {
				shd->thread[tid].avail4fetch++;
				assert(shd->thread[tid].avail4fetch <= 1);
				assert( shd->warp[tid/warp_size].n_avail4fetch < warp_size );
				shd->warp[tid/warp_size].n_avail4fetch++;
				thd_unlocked = 1;

				//printf("THREAD UNLOCK sid=%d tid=%d \n", shd->sid, tid);
			}
		}
    }

    // Update warp was unlocked, update the warp active mask
    if(thd_unlocked || thd_exited) {
    	// Update the warp active mask
		shader_pdom_update_warp_mask(shd, wid_from_hw_tid(valid_tid,warp_size));
    }



	if (shd->model == POST_DOMINATOR || shd->model == NO_RECONVERGE) {
		// Do nothing
	} else {
		// For this case, submit to commit_queue
		if (shd->using_commit_queue && thd_unlocked) {
			int *tid_unlocked = alloc_commit_warp();
			std::copy(tids.begin(), tids.end(), tid_unlocked);
			dq_push(shd->thd_commit_queue,(void*)tid_unlocked);
		}
	}
}


/*
 * Signals to the warp_tracker that a thread in a warp (for a given pc/instruction) is done
 *
 * @param *shd Pointer to shader core
 * @param grid_num Grid number
 * @param done_inst Completed instruction
 *
 */
void shader_call_thread_done( shader_core_ctx_t *shader, int grid_num, inst_t &done_inst ) {

	if (gpgpu_no_divg_load) {

		//printf("THREAD RETURNED sid=%d tid=%d pc=%d \n", shader->sid, done_inst.hw_thread_id, done_inst.pc);

		// Signal to unlock the thread. If all threads are done, deregister warp
		if( get_warp_tracker_pool().wpt_signal_avail(done_inst.hw_thread_id, shader, done_inst.pc) == 1 ) {
			// Entire warp has returned
			//printf("WARP RETURNED sid=%d tid=%d pc=%d \n", shader->sid, done_inst.hw_thread_id, done_inst.pc);

			// Deregister warp
			get_warp_tracker_pool().wpt_deregister_warp(done_inst.hw_thread_id, shader, done_inst.pc);

			// Signal scoreboard to release register
			shader->scrb->releaseRegisters( wid_from_hw_tid(done_inst.hw_thread_id, warp_size), &done_inst );

		}
	}

}


void shader_print_runtime_stat( FILE *fout ) {
   unsigned i;

   fprintf(fout, "SHD_INSN: ");
   for (i=0;i<gpu_n_shader;i++) {
      fprintf(fout, "%u ",sc[i]->num_sim_insn);
   }
   fprintf(fout, "\n");
   fprintf(fout, "SHD_THDS: ");
   for (i=0;i<gpu_n_shader;i++) {
      fprintf(fout, "%u ",sc[i]->not_completed);
   }
   fprintf(fout, "\n");
   fprintf(fout, "SHD_DIVG: ");
   for (i=0;i<gpu_n_shader;i++) {
      fprintf(fout, "%u ",sc[i]->n_diverge);
   }
   fprintf(fout, "\n");

   fprintf(fout, "THD_INSN: ");
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_insn);
   }
   fprintf(fout, "\n");
}


void shader_print_l1_miss_stat( FILE *fout ) {
   unsigned i;

   fprintf(fout, "THD_INSN_AC: ");
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_insn_ac);
   }
   fprintf(fout, "\n");

   fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_l1_mis_ac);
   }
   fprintf(fout, "\n");

   fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_l1_mis_ac - sc[0]->thread[i].n_l1_mrghit_ac);
   }
   fprintf(fout, "\n");

   fprintf(fout, "T_L1_Acc: "); //l1 access per thread
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      fprintf(fout, "%d ", sc[0]->thread[i].n_l1_access_ac);
   }
   fprintf(fout, "\n");

   //per warp
   int temp =0; 
   fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      temp += sc[0]->thread[i].n_l1_mis_ac;
      if (i%warp_size == (unsigned)(warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp=0;
   fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      temp += (sc[0]->thread[i].n_l1_mis_ac - sc[0]->thread[i].n_l1_mrghit_ac);
      if (i%warp_size == (unsigned)(warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp =0;
   fprintf(fout, "W_L1_Acc: "); //l1 access per warp
   for (i=0; i<gpu_n_thread_per_shader; i++) {
      temp += sc[0]->thread[i].n_l1_access_ac;
      if (i%warp_size == (unsigned)(warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");

}

void shader_print_warp( const shader_core_ctx_t *shader, inst_t *warp, FILE *fout, int stage_width, int print_mem, int mask ) 
{
   int i, j, warp_id = -1;
   for (i=0; i<stage_width; i++) {
      if (warp[i].hw_thread_id > -1) {
         warp_id = warp[i].hw_thread_id / warp_size;
         break;
      }
   }
   i = (i>=stage_width)? 0 : i;

   fprintf(fout,"0x%04x ", warp[i].pc );

   if( mask & 2 ) {
      fprintf(fout, "(" );
      for (j=0; j<stage_width; j++)
         fprintf(fout, "%03d ", warp[j].hw_thread_id);
      fprintf(fout, "): ");
   } else {
      fprintf(fout, "w%02d[", warp_id);
      for (j=0; j<stage_width; j++) 
         fprintf(fout, "%c", ((warp[j].hw_thread_id != -1)?'1':'0') );
      fprintf(fout, "]: ");
   }

   if( warp_id != -1 && shader->model == POST_DOMINATOR ) {
      pdom_warp_ctx_t *warp=&(shader->pdom_warp[warp_id]);
      if( warp->m_recvg_pc[warp->m_stack_top] == (unsigned)-1 ) {
         fprintf(fout," rp:--- ");
      } else {
         fprintf(fout," rp:0x%03x ", warp->m_recvg_pc[warp->m_stack_top] );
      }
   }

   ptx_print_insn( warp[i].pc, fout );

   if( mask & 0x10 ) {
      if ( (warp[i].op == STORE_OP ||
            warp[i].op == LOAD_OP) && print_mem )
         fprintf(fout, "  mem: 0x%016llx", warp[i].memreqaddr);
   }
   fprintf(fout, "\n");
}

void shader_print_stage(shader_core_ctx_t *shader, unsigned int stage, 
                        FILE *fout, int stage_width, int print_mem, int mask ) 
{
   inst_t *warp = shader->pipeline_reg[stage];
   shader_print_warp(shader,warp,fout,stage_width,print_mem,mask);
}

void shader_print_pre_mem_stages(shader_core_ctx_t *shader, FILE *fout, int print_mem, int mask ) 
{
   int i, j;
   int warp_id;

   if (!gpgpu_pre_mem_stages) return;

   for (unsigned pms = 0; pms <= gpgpu_pre_mem_stages - 1; pms++) {
      fprintf(fout, "PM[%01d] = ", pms);

      warp_id = -1;

      for (i=0; i<pipe_simd_width; i++) {
         if (shader->pre_mem_pipeline[pms][i].hw_thread_id > -1) {
            warp_id = shader->pre_mem_pipeline[pms][i].hw_thread_id / warp_size;
            break;
         }
      }
      i = (i>=pipe_simd_width)? 0 : i;

      fprintf(fout,"0x%04x ", shader->pre_mem_pipeline[pms][i].pc );

      if( mask & 2 ) {
         fprintf(fout, "(" );
         for (j=0; j<pipe_simd_width; j++)
            fprintf(fout, "%03d ", shader->pre_mem_pipeline[pms][j].hw_thread_id);
         fprintf(fout, "): ");
      } else {
         fprintf(fout, "w%02d[", warp_id);
         for (j=0; j<pipe_simd_width; j++)
            fprintf(fout, "%c", ((shader->pre_mem_pipeline[pms][j].hw_thread_id != -1)?'1':'0') );
         fprintf(fout, "]: ");
      }

      if( warp_id != -1 && shader->model == POST_DOMINATOR ) {
         pdom_warp_ctx_t *warp=&(shader->pdom_warp[warp_id]);
         if( warp->m_recvg_pc[warp->m_stack_top] == (unsigned)-1 ) {
            printf(" rp:--- ");
         } else {
            printf(" rp:0x%03x ", warp->m_recvg_pc[warp->m_stack_top] );
         }
      }

      ptx_print_insn( shader->pre_mem_pipeline[pms][i].pc, fout );

      if( mask & 0x10 ) {
         if ( ( shader->pre_mem_pipeline[pms][i].op == LOAD_OP ||
                shader->pre_mem_pipeline[pms][i].op == STORE_OP ) && print_mem )
            fprintf(fout, "  mem: 0x%016llx", shader->pre_mem_pipeline[pms][i].memreqaddr);
      }
      fprintf(fout, "\n");
   }
}

const char * ptx_get_fname( unsigned PC );

void shader_display_pipeline(shader_core_ctx_t *shader, FILE *fout, int print_mem, int mask ) 
{
   // call this function from within gdb to print out status of pipeline
   // if you encounter a bug, or to visualize pipeline operation
   // (this is a good way to "verify" your pipeline model makes sense!)

   fprintf(fout, "=================================================\n");
   fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", shader->sid, 
           gpu_tot_sim_cycle, gpu_sim_cycle, shader->not_completed);
   fprintf(fout, "=================================================\n");

   if ( (mask & 4) && shader->model == POST_DOMINATOR ) {
      fprintf(fout,"warp status:\n");
      unsigned n = shader->n_threads / warp_size;
      for (unsigned i=0; i < n; i++) {
         unsigned nactive = 0;
         for (unsigned j=0; j<warp_size; j++ ) {
            unsigned tid = i*warp_size + j;
            int done = ptx_thread_done( shader->thread[tid].ptx_thd_info );
            nactive += (ptx_thread_done( shader->thread[tid].ptx_thd_info )?0:1);
            if ( done && (mask & 8) ) {
               unsigned done_cycle = ptx_thread_donecycle( shader->thread[tid].ptx_thd_info );
               if ( done_cycle ) {
                  printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle );
               }
            }
         }
         if ( nactive == 0 ) {
            continue;
         }
         pdom_warp_ctx_t *warp=&(shader->pdom_warp[i]);
         for ( int k=0; k <= warp->m_stack_top; k++ ) {
            if ( k==0 ) {
               fprintf(fout, "w%02d (%2u thds active): %2u ", i, nactive, k );
            } else {
               fprintf(fout, "                      %2u ", k );
            }
            for (unsigned m=1,j=0; j<warp_size; j++, m<<=1)
               fprintf(fout, "%c", ((warp->m_active_mask[k] & m)?'1':'0') );
            fprintf(fout, " pc: %4u", warp->m_pc[k] );
            if ( warp->m_recvg_pc[k] == (unsigned)-1 ) {
               fprintf(fout," rp: ---- cd: %2u ", warp->m_calldepth[k] );
            } else {
               fprintf(fout," rp: %4u cd: %2u ", warp->m_recvg_pc[k], warp->m_calldepth[k] );
            }
            if ( warp->m_branch_div_cycle[k] != 0 ) {
               fprintf(fout," bd@%6u ", (unsigned) warp->m_branch_div_cycle[k] );
            } else {
               fprintf(fout,"           " );
            }
            //fprintf(fout," func=\'%s\' ", ptx_get_fname( warp->m_pc[k] ) );
            ptx_print_insn( warp->m_pc[k], fout );
            fprintf(fout,"\n");
         }
      }
      fprintf(fout,"\n");
   }

   if ( mask & 0x20 ) {
      fprintf(fout, "TS/IF = ");
      shader_print_stage(shader, TS_IF, fout, warp_size, print_mem, mask);
   }

   fprintf(fout, "IF/ID = ");
   shader_print_stage(shader, IF_ID, fout, pipe_simd_width, print_mem, mask );

   if (gpgpu_operand_collector)
       shader->m_opndcoll_new.dump(fout);

   if (shader->using_rrstage) {
      fprintf(fout, "ID/RR = ");
      shader_print_stage(shader, ID_RR, fout, pipe_simd_width, print_mem, mask);
   }

   fprintf(fout, "ID/EX = ");
   shader_print_stage(shader, ID_EX, fout, pipe_simd_width, print_mem, mask);

   shader_print_pre_mem_stages(shader, fout, print_mem, mask);

   if (!gpgpu_pre_mem_stages)
      fprintf(fout, "EX/MEM= ");
   else
      fprintf(fout, "PM/MEM= ");
   shader_print_stage(shader, EX_MM, fout, pipe_simd_width, print_mem, mask);

   fprintf(fout, "MEM/WB= ");
   shader_print_stage(shader, MM_WB, fout, pipe_simd_width, print_mem, mask);

   fprintf(fout, "\n");
}

void shader_dump_thread_state(shader_core_ctx_t *shader, FILE *fout )
{
   fprintf( fout, "\n");
   for ( unsigned w = 0; w < gpu_n_thread_per_shader/warp_size; w++ ) {
      int tid = w*warp_size;
      if ( shader->warp[w].n_completed < (unsigned)warp_size ) {
         fprintf( fout, "  %u:%3u fetch state = c:%u a4f:%u bw:%u (completed: ", shader->sid, tid, 
                  shader->warp[w].n_completed,
                  shader->warp[w].n_avail4fetch,
                  shader->warp[w].n_waiting_at_barrier );

         for ( unsigned i = tid; i < (w+1)*warp_size; i++ ) {
            if ( gpgpu_cuda_sim && ptx_thread_done(shader->thread[i].ptx_thd_info) ) {
               fprintf(fout,"1");
            } else {
               fprintf(fout,"0");
            }
            if ( (((i+1)%4) == 0) && (i+1) < (w+1)*warp_size ) {
               fprintf(fout,",");
            }
         }
         fprintf(fout,")\n");
      }
   }
}

void shader_dp(shader_core_ctx_t *shader, int print_mem) {
   shader_display_pipeline(shader, stdout, print_mem, 7 );
}


unsigned int max_cta_per_shader( shader_core_ctx_t *shader)
{
   unsigned int result;
   unsigned int padded_cta_size;

   padded_cta_size = ptx_sim_cta_size();
   if (padded_cta_size%warp_size) {
      padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);
      //printf("padded_cta_size=%u\n", padded_cta_size);
   }

   //Limit by n_threads/shader
   unsigned int result_thread = shader->n_threads / padded_cta_size;

   const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info();

   //Limit by shmem/shader
   unsigned int result_shmem = (unsigned)-1;
   if (kernel_info->smem > 0)
      result_shmem = shader->shmem_size / kernel_info->smem;

   //Limit by register count, rounded up to multiple of 4.
   unsigned int result_regs = (unsigned)-1;
   if (kernel_info->regs > 0)
      result_regs = shader->n_registers / (padded_cta_size * ((kernel_info->regs+3)&~3));

   //Limit by CTA
   unsigned int result_cta = shader->n_cta;

   result = result_thread;
   result = gs_min2(result, result_shmem);
   result = gs_min2(result, result_regs);
   result = gs_min2(result, result_cta);

   static const struct gpgpu_ptx_sim_kernel_info* last_kinfo = NULL;
   if (last_kinfo != kernel_info) {   //Only print out stats if kernel_info struct changes
      last_kinfo = kernel_info;
      printf ("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
      if (result == result_thread) printf (" threads");
      if (result == result_shmem) printf (" shmem");
      if (result == result_regs) printf (" regs");
      if (result == result_cta) printf (" cta_limit");
      printf ("\n");
   }

   if (result < 1) {
      printf ("Error: max_cta_per_shader(\"%s\") returning %d. Kernel requires more resources than shader has?\n", shader->name, result);
      abort();
   }
   return result;
}

void shader_cycle( shader_core_ctx_t *shader, 
                   unsigned int shader_number,
                   int grid_num ) 
{
   if (gpgpu_operand_collector) 
      shader_opnd_collect_write(shader);
   shader_writeback(shader, shader_number, grid_num);
   shader_memory(shader, shader_number);
   if (gpgpu_pre_mem_stages) // for modeling deeper pipelines
      shader_pre_memory(shader, shader_number); 
   shader_execute(shader, shader_number);
   if (shader->using_rrstage) {
      // Model register bank conflicts as in 
      // Fung et al. MICRO'07 / ACM TACO'09 papers.
      shader_preexecute (shader, shader_number);
   }
   if (gpgpu_operand_collector) 
      shader_opnd_collect_read(shader);
   shader_decode   (shader, shader_number, grid_num);
   shader_fetch    (shader, shader_number, grid_num);
}

// performance counter that are not local to one shader
void shader_print_accstats( FILE* fout ) 
{
   fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
   fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
   fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
   fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
   fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
   fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
   fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", gpgpu_n_cache_bkconflict);   

   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n", gpgpu_n_intrawarp_mshr_merge);
   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

   fprintf(fout, "gpgpu_n_writeback_l1_miss = %d\n", writeback_l1_miss);

   fprintf(fout, "gpgpu_n_partial_writes = %d\n", gpgpu_n_partial_writes);

   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][bk_conf] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]   
           ); // coalescing stall at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n", 
           gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] + 
           gpu_stall_shd_mem_breakdown[L_MEM_ST][COAL_STALL]    
           ); // coalescing stall + bank conflict at data cache 
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]);
   fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

   fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n", gpu_reg_bank_conflict_stalls);

   if (warp_occ_detailed) {
      int n_warp = gpu_n_thread_per_shader / warp_size;

      for (unsigned s = 0; s<gpu_n_shader; s++)
         for (int w = 0; w<n_warp; w++) {
            fprintf(fout, "wod[%d][%d]=", s, w);
            for (unsigned t = 0; t<warp_size; t++) {
               fprintf(fout, "%d ", warp_occ_detailed[s * n_warp + w][t]);
            }
            fprintf(fout, "\n");
         }
   }
}

// Flushes all content of the cache to memory

void shader_cache_flush(shader_core_ctx_t* sc) 
{
   unsigned int i;
   unsigned int set;
   unsigned long long int flush_addr;

   shd_cache_t *cp = sc->L1cache;
   shd_cache_line_t *pline;

   for (i=0; i<cp->nset*cp->assoc; i++) {
      pline = &(cp->lines[i]);
      set = i / cp->assoc;
      if ((pline->status & (DIRTY|VALID)) == (DIRTY|VALID)) {
         flush_addr = pline->addr;

         sc->fq_push(flush_addr, sc->L1cache->line_sz, 1, NO_PARTIAL_WRITE, sc->sid, 0, NULL, 0, GLOBAL_ACC_W, -1);

         pline->status &= ~VALID;
         pline->status &= ~DIRTY;
      } else if (pline->status & VALID) {
         pline->status &= ~VALID;
      }
   }
}

static int *_inmatch;
static int *_outmatch;
static int **_request;

// modifiers
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() 
{
   std::list<op_t> result;  // a list of registers that (a) are in different register banks, (b) do not go to the same operand collector

   int input;
   int output;
   int _inputs = m_num_banks;
   int _outputs = m_num_collectors;
   int _square = ( _inputs > _outputs ) ? _inputs : _outputs;
   int _pri = (int)m_last_cu;

   if( _inmatch == NULL ) {
      _inmatch = new int[ _inputs ];
      _outmatch = new int[ _outputs ];
      _request = new int*[ _inputs ];
      for(int i=0; i<_inputs;i++) 
         _request[i] = new int[_outputs];
   }

   // Clear matching
   for ( int i = 0; i < _inputs; ++i ) 
      _inmatch[i] = -1;
   for ( int j = 0; j < _outputs; ++j ) 
      _outmatch[j] = -1;

   for( unsigned i=0; i<m_num_banks; i++) {
      for( unsigned j=0; j<m_num_collectors; j++) 
         _request[i][j] = 0;
      if( !m_queue[i].empty() ) {
         const op_t &op = m_queue[i].front();
         int oc_id = op.get_oc_id();
         _request[i][oc_id] = 1;
      }
      if( m_allocated_bank[i].is_write() ) 
         _inmatch[i] = 0; // write gets priority
   }

   ///// wavefront allocator from booksim... --->
   
   // Loop through diagonals of request matrix

   for ( int p = 0; p < _square; ++p ) {
      output = ( _pri + p ) % _square;

      // Step through the current diagonal
      for ( input = 0; input < _inputs; ++input ) {
         if ( ( output < _outputs ) && 
              ( _inmatch[input] == -1 ) && 
              ( _outmatch[output] == -1 ) &&
              ( _request[input][output]/*.label != -1*/ ) ) {
            // Grant!
            _inmatch[input] = output;
            _outmatch[output] = input;
         }

         output = ( output + 1 ) % _square;
      }
   }

   // Round-robin the priority diagonal
   _pri = ( _pri + 1 ) % _square;

   /// <--- end code from booksim

   m_last_cu = _pri;
   for( unsigned i=0; i < m_num_banks; i++ ) {
      if( _inmatch[i] != -1 ) {
         if( !m_allocated_bank[i].is_write() ) {
            unsigned bank = (unsigned)i;
            op_t &op = m_queue[bank].front();
            result.push_back(op);
            m_queue[bank].pop_front();
         }
      }
   }


/*
   for( unsigned c=0; c < m_num_collectors; c++ ) {
      unsigned cu = (m_last_cu+c+1)%m_num_collectors;
      for( unsigned b=0; b < m_num_banks; b++ ) {
         unsigned bank = (m_allocator_rr_head[cu]+b+1)%m_num_banks;
         if( (!m_queue[bank].empty()) && m_allocated_bank[bank].is_free() ) {
            op_t &op = m_queue[bank].front();
            result.push_back(op);
            m_allocated_bank[bank].alloc_read(op);
            m_queue[bank].pop_front();
            m_allocator_rr_head[cu] = bank;
            m_last_cu = cu;
            break; // skip to next collector unit
         }
      }
   }
*/
   return result;
}


barrier_set_t::barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core )
{
   m_max_warps_per_core = max_warps_per_core;
   m_max_cta_per_core = max_cta_per_core;
   if( max_warps_per_core > WARP_PER_CTA_MAX ) {
      printf("ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or warps per cta in gpgpusim.config\n",
             WARP_PER_CTA_MAX, max_warps_per_core );
      exit(1);
   }
   m_warp_active.reset();
   m_warp_at_barrier.reset();
}

// during cta allocation
void barrier_set_t::allocate_barrier( unsigned cta_id, warp_set_t warps )
{
   assert( cta_id < m_max_cta_per_core );
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   assert( w == m_cta_to_warps.end() ); // cta should not already be active or allocated barrier resources
   m_cta_to_warps[cta_id] = warps;
   assert( m_cta_to_warps.size() <= m_max_cta_per_core ); // catch cta's that were not properly deallocated
  
   m_warp_active |= warps;
   m_warp_at_barrier &= ~warps;
}

// during cta deallocation
void barrier_set_t::deallocate_barrier( unsigned cta_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   if( w == m_cta_to_warps.end() )
      return;
   warp_set_t warps = w->second;
   warp_set_t at_barrier = warps & m_warp_at_barrier;
   assert( at_barrier.any() == false ); // no warps stuck at barrier
   warp_set_t active = warps & m_warp_active;
   assert( active.any() == false ); // no warps in CTA still running
   m_warp_active &= ~warps;
   m_warp_at_barrier &= ~warps;
   m_cta_to_warps.erase(w);
}

// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier( unsigned cta_id, unsigned warp_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);

   if( w == m_cta_to_warps.end() ) { // cta is active
      printf("ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n", cta_id, gpu_tot_sim_cycle, gpu_sim_cycle );
      dump();
      abort();
   }
   assert( w->second.test(warp_id) == true ); // warp is in cta

   m_warp_at_barrier.set(warp_id);

   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

   if( at_barrier == active ) {
      // all warps have reached barrier, so release waiting warps...
      m_warp_at_barrier &= ~at_barrier;
   }
}

// fetching a warp
bool barrier_set_t::available_for_fetch( unsigned warp_id ) const
{
   return m_warp_active.test(warp_id) && m_warp_at_barrier.test(warp_id);
}

// warp reaches exit 
void barrier_set_t::warp_exit( unsigned warp_id )
{
   // caller needs to verify all threads in warp are done, e.g., by checking PDOM stack to 
   // see it has only one entry during exit_impl()
   m_warp_active.reset(warp_id);
}

// assertions
bool barrier_set_t::warp_waiting_at_barrier( unsigned warp_id )
{ 
   return m_warp_at_barrier.test(warp_id);
}

void barrier_set_t::dump() const
{
   printf( "barrier set information\n");
   printf( "  m_max_cta_per_core = %u\n",  m_max_cta_per_core );
   printf( "  m_max_warps_per_core = %u\n", m_max_warps_per_core );
   printf( "  cta_to_warps:\n");
   
   cta_to_warp_t::const_iterator i;
   for( i=m_cta_to_warps.begin(); i!=m_cta_to_warps.end(); i++ ) {
      unsigned cta_id = i->first;
      warp_set_t warps = i->second;
      printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str() );
   }
   printf("  warp_active: %s\n", m_warp_active.to_string().c_str() );
   printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str() );
   fflush(stdout); 
}

void shader_core_ctx::set_at_barrier( unsigned cta_id, unsigned warp_id )
{
   m_barriers.warp_reaches_barrier(cta_id,warp_id);
}

void shader_core_ctx::warp_exit( unsigned warp_id )
{
   m_barriers.warp_exit( warp_id );
}

bool shader_core_ctx::warp_waiting_at_barrier( unsigned warp_id )
{
   return m_barriers.warp_waiting_at_barrier(warp_id);
}

void shader_core_ctx::allocate_barrier( unsigned cta_id, warp_set_t warps )
{
   m_barriers.allocate_barrier(cta_id,warps);
}

void shader_core_ctx::deallocate_barrier( unsigned cta_id )
{
   m_barriers.deallocate_barrier(cta_id);
}

void opndcoll_rfu_t::init( unsigned num_collectors_alu, 
                           unsigned num_collectors_sfu, 
                           unsigned num_banks, 
                           const shader_core_ctx *shader ) 
{
   unsigned num_alu_cu = gpgpu_operand_collector_num_units;
   unsigned num_sfu_cu = gpgpu_operand_collector_num_units_sfu;
   m_num_collectors = num_alu_cu+num_sfu_cu;
    
   m_shader=shader;
   m_arbiter.init(m_num_collectors,num_banks);

   m_alu_port = shader->pipeline_reg[ID_EX];
   m_sfu_port = shader->pipeline_reg[OC_EX_SFU];

   m_dispatch_units[ m_alu_port ].init( num_alu_cu );
   m_dispatch_units[ m_sfu_port ].init( num_sfu_cu );

   m_num_banks = num_banks;
   m_cu = new collector_unit_t[m_num_collectors];

   unsigned c=0;
   for(; c<num_alu_cu; c++) {
      m_cu[c].init(c,m_alu_port);
      m_free_cu[m_alu_port].push_back(&m_cu[c]);
      m_dispatch_units[m_alu_port].add_cu(&m_cu[c]);
   }
   for(; c<m_num_collectors; c++) {
      m_cu[c].init(c,m_sfu_port);
      m_free_cu[m_sfu_port].push_back(&m_cu[c]);
      m_dispatch_units[m_sfu_port].add_cu(&m_cu[c]);
   }
}
