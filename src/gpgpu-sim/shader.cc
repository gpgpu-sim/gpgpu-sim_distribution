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


unsigned mem_access_t::next_access_uid = 0;   

/////////////////////////////////////////////////////////////////////////////
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/

static const char* MSHR_Status_str[] = {
   "INITIALIZED",
   "INVALID",
   "IN_ICNT2MEM",
   "IN_CBTOL2QUEUE",
   "IN_L2TODRAMQUEUE",
   "IN_DRAM_REQ_QUEUE",
   "IN_DRAMRETURN_Q",
   "IN_DRAMTOL2QUEUE",
   "IN_L2TOCBQUEUE_HIT",
   "IN_L2TOCBQUEUE_MISS",
   "IN_ICNT2SHADER",
   "FETCHED",
};

void mshr_lookup::insert(mshr_entry* mshr) 
{
   using namespace std;
   new_addr_type tag_addr = mshr->get_addr();
   m_lut.insert(make_pair(tag_addr, mshr));
}

mshr_entry* mshr_lookup::lookup( new_addr_type addr ) const 
{
   std::pair<mshr_lut_t::const_iterator, mshr_lut_t::const_iterator> i_range = m_lut.equal_range(addr);
   if (i_range.first == i_range.second) {
      return NULL;
   } else {
      mshr_lut_t::const_iterator i_lut = i_range.first; 
      return i_lut->second->get_last_merged();
   }
}

void mshr_lookup::remove(mshr_entry* mshr)
{
   using namespace std;
   std::pair<mshr_lut_t::iterator, mshr_lut_t::iterator> i_range = m_lut.equal_range(mshr->get_addr());

   assert(i_range.first != i_range.second);

   for (mshr_lut_t::iterator i_lut = i_range.first; i_lut != i_range.second; ++i_lut) {
      if (i_lut->second == mshr) {
         m_lut.erase(i_lut);
         break;
      }
   }
}

//checks if we should do mshr merging for this mshr
bool mshr_lookup::can_merge(mshr_entry * mshr)
{
   if (mshr->iswrite()) 
       return false; // can't merge a write
   if (mshr->isatomic()) 
       return false; // can't merge a atomic operation
   bool interwarp_mshr_merge = m_shader_config->gpgpu_interwarp_mshr_merge & GLOBAL_MSHR_MERGE;
   if (mshr->isinst())
      interwarp_mshr_merge=true; 
   else if (mshr->istexture()) 
      interwarp_mshr_merge = m_shader_config->gpgpu_interwarp_mshr_merge & TEX_MSHR_MERGE;
   else if (mshr->isconst()) 
      interwarp_mshr_merge = m_shader_config->gpgpu_interwarp_mshr_merge & CONST_MSHR_MERGE;
   return interwarp_mshr_merge;
}

void mshr_lookup::mshr_fast_lookup_insert(mshr_entry* mshr) 
{        
   if (!can_merge(mshr)) 
       return;  
   insert(mshr);
}
   
void mshr_lookup::mshr_fast_lookup_remove(mshr_entry* mshr) 
{
   if (!can_merge(mshr)) 
       return;  
   remove(mshr);
}
   
mshr_entry* mshr_lookup::shader_get_mergeable_mshr(mshr_entry* mshr)
{
   if (!can_merge(mshr)) return NULL;
   return lookup(mshr->get_addr());
}

mshr_shader_unit::mshr_shader_unit( const shader_core_config *config ): m_max_mshr_used(0), m_mshr_lookup(config)
{
    m_shader_config=config;
    m_mshrs.resize(config->n_mshr_per_shader);
    unsigned n=0;
    for (std::vector<mshr_entry>::iterator i = m_mshrs.begin(); i != m_mshrs.end(); i++) {
        mshr_entry &mshr = *i; 
        mshr.set_id(n++);
        m_free_list.push_back(&mshr);
    }
}

mshr_entry* mshr_shader_unit::return_head()
{
   if (has_return()) 
      return &(*choose_return_queue().front()); 
   else 
      return NULL;
}

void mshr_shader_unit::pop_return_head() 
{
   free_mshr(return_head());
   choose_return_queue().pop_front(); 
}

mshr_entry *mshr_shader_unit::alloc_free_mshr(bool istexture)
{
   assert(!m_free_list.empty());
   mshr_entry *mshr = m_free_list.back();
   m_free_list.pop_back();
   if (istexture)
       m_texture_mshr_pipeline.push_back(mshr);
   if (mshr_used() > m_max_mshr_used) 
       m_max_mshr_used = mshr_used();
   return mshr;
}

void mshr_shader_unit::free_mshr( mshr_entry *mshr )
{
   //clean up up for next time, since not reallocating memory.
   m_mshr_lookup.mshr_fast_lookup_remove(mshr); 
   mshr->clear();
   m_free_list.push_back(mshr);
}

unsigned mshr_shader_unit::mshr_used() const
{ 
   return m_shader_config->n_mshr_per_shader - m_free_list.size();
}

std::deque<mshr_entry*> &mshr_shader_unit::choose_return_queue() 
{
   // prioritize a ready texture over a global/const...
   if ((not m_texture_mshr_pipeline.empty()) and m_texture_mshr_pipeline.front()->fetched()) 
       return m_texture_mshr_pipeline;
   assert(!m_mshr_return_queue.empty());
   return m_mshr_return_queue;
}

void mshr_shader_unit::mshr_return_from_mem(mshr_entry *mshr)
{
   mshr->set_status( FETCHED );
   if ( not mshr->istexture() ) {
       //place in return queue
       mshr->add_to_queue( m_mshr_return_queue );
   }
}

void shader_core_ctx::mshr_print(FILE* fp, unsigned mask) 
{
   m_mshr_unit->print(fp, this, mask);
}

void mshr_shader_unit::print(FILE* fp, shader_core_ctx* shader, unsigned mask)
{
    unsigned n=0;
    unsigned num_outstanding = 0;
    for (mshr_storage_type::iterator it = m_mshrs.begin(); it != m_mshrs.end(); it++,n++) {
        mshr_entry *mshr = &(*it);
        if (find(m_free_list.begin(),m_free_list.end(), mshr) == m_free_list.end()) {
            num_outstanding++;
            mshr->print(fp,mask);
        }
    }
    fprintf(fp,"\nTotal outstanding memory requests = %u\n", num_outstanding );
}

unsigned char shader_core_ctx::fq_push(unsigned long long int addr, 
                                       int bsize, 
                                       unsigned char write, 
                                       partial_write_mask_t partial_write_mask, 
                                       int wid, 
                                       mshr_entry* mshr, 
                                       enum mem_access_type mem_acc, 
                                       address_type pc) 
{
   assert(write || (partial_write_mask == NO_PARTIAL_WRITE));
   mem_fetch *mf = new mem_fetch(addr,
                                 bsize,
                                 (write?WRITE_PACKET_SIZE:READ_PACKET_SIZE),
                                 m_sid,
                                 m_tpc,
                                 wid,
                                 mshr,
                                 write,
                                 partial_write_mask,
                                 mem_acc,
                                 (write?WT_REQ:RD_REQ),
                                 pc);
   if (mshr) mshr->set_mf(mf);

   // stats
   if (write) made_write_mfs++;
   else made_read_mfs++;
   switch (mem_acc) {
   case CONST_ACC_R: m_stats->gpgpu_n_mem_const++; break;
   case TEXTURE_ACC_R: m_stats->gpgpu_n_mem_texture++; break;
   case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++; break;
   case GLOBAL_ACC_W: m_stats->gpgpu_n_mem_write_global++; break;
   case LOCAL_ACC_R: m_stats->gpgpu_n_mem_read_local++; break;
   case LOCAL_ACC_W: m_stats->gpgpu_n_mem_write_local++; break;
   case INST_ACC_R: m_stats->gpgpu_n_mem_read_inst++; break;
   default: assert(0);
   }

   return(m_gpu->issue_mf_from_fq(mf));
}

inst_t *shader_core_ctx::first_valid_thread( inst_t *warp )
{
   for(unsigned t=0; t < m_config->warp_size; t++ ) 
      if( warp[t].hw_thread_id != -1 ) 
         return warp+t;
   return NULL;
}

inst_t *shader_core_ctx::first_valid_thread( unsigned stage )
{
    return first_valid_thread(m_pipeline_reg[stage]);
}

void shader_core_ctx::move_warp( inst_t *&dst, inst_t *&src )
{

   assert( pipeline_regster_empty(dst) );
   inst_t* temp = dst;
   dst = src;
   src = temp;
   for( unsigned t=0; t < m_config->warp_size; t++) 
      src[t] = inst_t();
}

void shader_core_ctx::clear_stage( inst_t *warp )
{
   for( unsigned t=0; t < m_config->warp_size; t++) 
      warp[t] = inst_t();
}

bool shader_core_ctx::pipeline_regster_empty( inst_t *reg )
{
   return first_valid_thread(reg) == NULL;
}

void shader_core_ctx::L1cache_print( FILE *fp, unsigned &total_accesses, unsigned &total_misses) const
{
   m_L1D->shd_cache_print(fp,total_accesses,total_misses);
}

void shader_core_ctx::L1texcache_print( FILE *fp, unsigned &total_accesses, unsigned &total_misses) const
{
   m_L1T->shd_cache_print(fp,total_accesses,total_misses);
}

void shader_core_ctx::L1constcache_print( FILE *fp, unsigned &total_accesses, unsigned &total_misses) const
{
   m_L1C->shd_cache_print(fp,total_accesses,total_misses);
}

std::list<unsigned> shader_core_ctx::get_regs_written( const inst_t &fvt ) const
{
   std::list<unsigned> result;
   for( unsigned op=0; op < 4; op++ ) {
      int reg_num = fvt.arch_reg[op]; // this math needs to match that used in function_info::ptx_decode_inst
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

shader_core_ctx::shader_core_ctx( class gpgpu_sim *gpu,
                                  const char *name, 
                                  unsigned shader_id,
                                  unsigned tpc_id,
                                  const struct shader_core_config *config,
                                  struct shader_core_stats *stats )
   : m_barriers( config->max_warps_per_shader, config->max_cta_per_core )
{
   m_gpu = gpu;
   m_config = config;
   m_stats = stats;
   unsigned warp_size=config->warp_size;
   assert( !((config->model == DWF) && m_config->gpgpu_operand_collector) );

   m_name = name;
   m_sid = shader_id;
   m_tpc = tpc_id;
   m_dwf_RR_k = 0;
   m_pipeline_reg = (inst_t**) calloc(N_PIPELINE_STAGES, sizeof(inst_t*));
   for (int j = 0; j<N_PIPELINE_STAGES; j++) {
      m_pipeline_reg[j] = (inst_t*) calloc(warp_size, sizeof(inst_t));
      for (unsigned i=0; i<warp_size; i++) 
         m_pipeline_reg[j][i] = inst_t();
   }

   if (m_config->gpgpu_pre_mem_stages) {
      pre_mem_pipeline = (inst_t**) calloc(m_config->gpgpu_pre_mem_stages+1, sizeof(inst_t*));
      for (unsigned j = 0; j<=m_config->gpgpu_pre_mem_stages; j++) {
         pre_mem_pipeline[j] = (inst_t*) calloc(warp_size, sizeof(inst_t));
         for (unsigned i=0; i<warp_size; i++) {
            pre_mem_pipeline[j][i] = inst_t();
         }
      }
   }
   m_thread = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), config->n_thread_per_shader);
   m_not_completed = 0;

   m_warp.resize(m_config->max_warps_per_shader, shd_warp_t(this, warp_size));

   m_n_active_cta = 0;
   for (unsigned i = 0; i<MAX_CTA_PER_SHADER; i++  ) 
      m_cta_status[i]=0;
   m_next_warp = 0;
   for (unsigned i = 0; i<config->n_thread_per_shader; i++) {
      m_thread[i].m_functional_model_thread_state = NULL;
      m_thread[i].m_avail4fetch = false;
      m_thread[i].m_waiting_at_barrier = false;
      m_thread[i].m_cta_id = -1;
   }

   #define STRSIZE 1024
   char L1D_name[STRSIZE];
   char L1T_name[STRSIZE];
   char L1C_name[STRSIZE];
   char L1I_name[STRSIZE];

   snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
   snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
   snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
   snprintf(L1I_name, STRSIZE, "L1I_%03d", m_sid);
   enum cache_write_policy L1D_policy = m_config->gpgpu_cache_wt_through?write_through:write_back;
   m_L1D = new cache_t(L1D_name,m_config->gpgpu_cache_dl1_opt,    0,L1D_policy,m_sid,get_shader_normal_cache_id());
   m_L1T = new cache_t(L1T_name,m_config->gpgpu_cache_texl1_opt,  0,no_writes, m_sid,get_shader_texture_cache_id());
   m_L1C = new cache_t(L1C_name,m_config->gpgpu_cache_constl1_opt,0,no_writes, m_sid,get_shader_constant_cache_id());
   m_L1I = new cache_t(L1I_name,m_config->gpgpu_cache_il1_opt,    0,no_writes, m_sid,get_shader_instruction_cache_id());
   ptx_set_tex_cache_linesize(m_L1T->get_line_sz());

   m_mshr_unit = new mshr_shader_unit(m_config);
   m_pdom_warp = new pdom_warp_ctx_t*[config->max_warps_per_shader];
   for (unsigned i = 0; i < config->max_warps_per_shader; ++i) 
       m_pdom_warp[i] = new pdom_warp_ctx_t(i,this);
   if (m_config->using_commit_queue) 
      m_thd_commit_queue = new fifo_pipeline<std::vector<int> >("thd_commit_queue", 0, 0,gpu_sim_cycle);
   m_shader_memory_new_instruction_processed = false;

   // Initialize scoreboard
   m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader);

   if( m_config->gpgpu_operand_collector ) {
      m_operand_collector.init( m_config->gpgpu_operand_collector_num_units, 
                           m_config->gpgpu_operand_collector_num_units_sfu, 
                           m_config->gpgpu_num_reg_banks, this,
                           &m_pipeline_reg[ID_EX],
                           &m_pipeline_reg[OC_EX_SFU] );
   }

   m_memory_queue.shared.reserve(warp_size);
   m_memory_queue.constant.reserve(warp_size);
   m_memory_queue.texture.reserve(warp_size);
   m_memory_queue.local_global.reserve(warp_size);

   // writeback
   m_pl_tid = (int*) malloc(sizeof(int)*warp_size);
   m_mshr_lat_info = (insn_latency_info*) malloc(sizeof(insn_latency_info) * warp_size);
   m_pl_lat_info = (insn_latency_info*) malloc(sizeof(insn_latency_info) * warp_size);

   // fetch
   m_last_warp_fetched = 0;
   m_last_warp_issued = 0;
   m_ready_warps = (int*)calloc(m_config->max_warps_per_shader,sizeof(int));
   m_tmp_ready_warps = (int*)calloc(m_config->max_warps_per_shader,sizeof(int));
   m_last_warp=0;
   m_last_issued_thread=0; // MIMD

   m_warp_tracker = NULL;
   m_thread_pc_tracker = NULL;
   if (m_config->gpgpu_no_divg_load) {
      m_warp_tracker = new warp_tracker_pool(this);
      m_thread_pc_tracker = new thread_pc_tracker(warp_size, config->n_thread_per_shader);
   }
   m_fetch_tid_out = (int*) malloc(sizeof(int) * warp_size);
   m_dwf_rrstage_bank_access_counter = (int*) malloc(sizeof(int) * m_config->gpgpu_dwf_rr_stage_n_reg_banks);
}

void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed ) 
{
   if( reset_not_completed ) 
       m_not_completed = 0;
   m_next_warp = 0;
   m_last_issued_thread=0;
   for (unsigned i = start_thread; i<end_thread; i++) {
      m_thread[i].n_insn = 0;
      m_thread[i].m_cta_id = -1;
   }
   for (unsigned i = start_thread / m_config->warp_size; i < end_thread / m_config->warp_size; ++i) {
      m_warp[i].reset();
      m_pdom_warp[i]->reset();
   }
}

void shader_core_ctx::init_warps( unsigned start_thread, unsigned end_thread )
{
    unsigned num_threads = end_thread - start_thread;
    address_type start_pc = next_pc(start_thread);
    if (m_config->model == POST_DOMINATOR) {
        unsigned start_warp = start_thread / m_config->warp_size;
        unsigned end_warp = end_thread / m_config->warp_size + ((end_thread % m_config->warp_size)? 1 : 0);
        for (unsigned i = start_warp; i < end_warp; ++i) {
            unsigned initial_active_mask = 0;
            unsigned n_active=0;
            for (unsigned t = 0; t < m_config->warp_size; t++) {
                if ( i * m_config->warp_size + t < end_thread ) {
                    initial_active_mask |= (1 << t);
                    n_active++;
                }
            }
            m_pdom_warp[i]->launch(start_pc,initial_active_mask);
            m_warp[i].init(start_pc,i,n_active);
            m_not_completed += n_active;
      }
   } else if (m_config->model == DWF) {
      dwf_init_CTA(m_sid, start_thread, num_threads, start_pc);
      for (unsigned i = start_thread; i<end_thread; i++) 
         m_thread[i].m_in_scheduler = true;
   }
   for (unsigned tid=start_thread;tid<end_thread;tid++) {
       m_thread[tid].m_avail4fetch = true;
   }
}

// register id for unused register slot in instruction
#define DNA       (0)

unsigned g_next_shader_inst_uid=1;

bool shader_core_ctx::fetch_stalled()
{
   for (unsigned i=0; i<m_config->warp_size; i++) {
      if (m_pipeline_reg[TS_IF][i].hw_thread_id != -1 ) {
         return true;  // stalled 
      }
   }
   for (unsigned i=0; i<m_config->warp_size; i++) {
      if (m_pipeline_reg[IF_ID][i].hw_thread_id != -1 ) {
         return true;  // stalled 
      }
   }

   m_new_warp_TS = true;
   return false; // not stalled
}

// initalize the pipeline stage register to nops
void shader_core_ctx::clear_stage_reg(int stage)
{
   clear_stage( m_pipeline_reg[stage] );
}

// return the next pc of a thread 
address_type shader_core_ctx::next_pc( int tid ) const
{
    if( tid == -1 ) 
        return -1;
    ptx_thread_info *the_thread = m_thread[tid].m_functional_model_thread_state;
    if ( the_thread == NULL )
        return -1;
    return the_thread->get_pc(); // PC should already be updatd to next PC at this point (was set in shader_decode() last time thread ran)
}

// issue thread to the warp 
// tid - thread id, warp_id - used by PDOM, wlane - position in warp
void shader_core_ctx::shader_issue_thread(int tid, int wlane, unsigned active_mask )
{
   m_thread[tid].m_functional_model_thread_state->ptx_fetch_inst( m_pipeline_reg[TS_IF][wlane] );
   m_pipeline_reg[TS_IF][wlane].hw_thread_id = tid;
   m_pipeline_reg[TS_IF][wlane].wlane = wlane;
   m_pipeline_reg[TS_IF][wlane].memreqaddr = 0;
   m_pipeline_reg[TS_IF][wlane].uid = g_next_shader_inst_uid++;
   m_pipeline_reg[TS_IF][wlane].warp_active_mask = active_mask;
   m_pipeline_reg[TS_IF][wlane].issue_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;

   assert( m_thread[tid].m_avail4fetch );
   m_thread[tid].m_avail4fetch = false;
   assert( m_warp[wid_from_hw_tid(tid,m_config->warp_size)].get_avail4fetch() > 0 );
   m_warp[wid_from_hw_tid(tid,m_config->warp_size)].dec_avail4fetch();
}

int shader_core_ctx::pdom_sched_find_next_warp (int ready_warp_count)
{
   bool found = false; 
   int selected_warp = m_ready_warps[0];
   switch (m_config->pdom_sched_type) {
   case 0: selected_warp = m_ready_warps[0]; found=true; break; // first ok warp found
   case 1: selected_warp = m_ready_warps[rand()%ready_warp_count]; found=true;  break; //random
   case 8: 
      // "loose" round robin:
      // execute the next available warp which is after the warp execued last time
      selected_warp = (m_last_warp + 1) % m_config->max_warps_per_shader;
      while (!found) {
         for (int i=0;i<ready_warp_count;i++) {
            if (selected_warp==m_ready_warps[i]) 
               found=true;
         }
         if( !found ) 
            selected_warp = (selected_warp + 1) % m_config->max_warps_per_shader;
      }
      break;         
   default: assert(0);
   }
   if (found) {
      if (ready_warp_count==1) 
         m_stats->n_pdom_sc_single_stat++;
      else 
         m_stats->n_pdom_sc_orig_stat++;
      return selected_warp;
   } else {
      return -1;
   }
}

void shader_core_ctx::fetch_simd_postdominator()
{
   int warp_ok = 0;
   bool complete = false;
   int tmp_warp;
   int warp_id;

   address_type check_pc = -1;

   // First, check to see if entire program is completed, 
   // if it is, then break out of loop
   for (unsigned i=0; i<m_config->n_thread_per_shader; i++) {
      if (!ptx_thread_done(i)) {
         complete = false;
         break;
      } else {
         complete = true;
      }
   }
   if (complete) 
      return;

   if (fetch_stalled()) 
      return; 
   clear_stage_reg(TS_IF);

   unsigned ready_warp_count = 0;
   for (unsigned i=0; i<m_config->max_warps_per_shader; i++) {
      m_ready_warps[i]=-1;
      m_tmp_ready_warps[i]=-1;
   }

   // Finds a warp where all threads in it are available for fetching 
   // simultaneously(all threads are not yet in pipeline, or, the ones 
   // that are not available, are completed already
   for (unsigned i=0; i<m_config->max_warps_per_shader; i++) {
       if( m_warp[m_next_warp].waiting() ) {
           // waiting for kernel launch, barrier, membar, atomic
       } else if( (m_warp[m_next_warp].get_n_completed()+m_warp[m_next_warp].get_avail4fetch()) < m_config->warp_size) {
           // waiting for instruction still in pipeline barrel processing
       } else if ( !warp_scoreboard_hazard(m_next_warp) ) {
          // this warp is ready and can be issued if selected
          m_tmp_ready_warps[ready_warp_count] = m_next_warp;
          ready_warp_count++;
      }
      m_next_warp = (m_next_warp + 1) % m_config->max_warps_per_shader;
   }
   for (unsigned i=0;i<ready_warp_count;i++) 
      m_ready_warps[i]=m_tmp_ready_warps[i];
   m_stats->num_warps_issuable[ready_warp_count]++;
   m_stats->num_warps_issuable_pershader[m_sid]+= ready_warp_count;
   if (ready_warp_count) {
      tmp_warp = pdom_sched_find_next_warp (ready_warp_count);
      if (tmp_warp != -1) {
         m_next_warp = tmp_warp;
         warp_ok=1;  
      }
   }

   if (!warp_ok) {
      // None of the instructions from inside the warp can be scheduled -> should  
      // probably just stall, ie nops into pipeline
      clear_stage_reg(TS_IF);  
      m_next_warp = (m_next_warp+1) % m_config->max_warps_per_shader;  
      return;
   }

   /************************************************************/
   // at this point we have a warp to execute which is pointed to by next_warp

   warp_id   = m_next_warp;
   m_last_warp = warp_id;
   int wtid  = m_config->warp_size*warp_id;
   pdom_warp_ctx_t *scheduled_warp = m_pdom_warp[warp_id];

   // schedule threads according to active mask on the top of pdom stack
   unsigned active_mask = scheduled_warp->get_active_mask();

   for (unsigned i = 0; i < m_config->warp_size; i++) {
      unsigned int mask = (1 << i);
      if ((active_mask & mask) == mask) {
         assert (!ptx_thread_done(wtid+i));
         shader_issue_thread(wtid+i,i,active_mask);
      }
   }
   m_next_warp = (m_next_warp+1)%m_config->max_warps_per_shader;

   // check if all issued threads have the same pc
   for (unsigned i = 0; i < m_config->warp_size; i++) {
      if ( m_pipeline_reg[TS_IF][i].hw_thread_id != -1 ) {
         if ( check_pc == (unsigned)-1 ) {
            check_pc = m_pipeline_reg[TS_IF][i].pc;
         } else {
            assert( check_pc == m_pipeline_reg[TS_IF][i].pc );
         }
      }
   }
}

/**
 * check if warp has data hazard  
 * 
 * @param warp_id 
 * 
 * @return bool : false if hazard exists
 */
bool shader_core_ctx::warp_scoreboard_hazard(int warp_id) 
{
	inst_t active_inst;

	// Get an active thread in the warp
	int wtid = m_config->warp_size*warp_id;
	pdom_warp_ctx_t *scheduled_warp = m_pdom_warp[warp_id];
	thread_ctx_t *active_thread = NULL;
    unsigned active_mask = scheduled_warp->get_active_mask();
	for (unsigned i = 0; i < m_config->warp_size; i++) {
		unsigned int mask = (1 << i);
		if ((active_mask & mask) == mask) {
			active_thread = &(m_thread[wtid+i]);
		}
	}
	if(active_thread == NULL) 
        return false;

	// Decode instruction
	active_thread->m_functional_model_thread_state->ptx_fetch_inst( active_inst );
	return m_scoreboard->checkCollision(warp_id, &active_inst);
}

void pdom_warp_ctx_t::pdom_update_warp_mask() 
{
	int wtid = m_warp_size*m_warp_id;

	pdom_warp_ctx_t *scheduled_warp = this;

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
      class ptx_thread_info *first_active_thread=NULL;
	  for (int i = m_warp_size - 1; i >= 0; i--) {
		 unsigned int mask = (1 << i);
		 if ((top_active_mask & mask) == mask) { // is this thread active?
			 if (m_shader->ptx_thread_done(wtid+i)) {
			    top_active_mask &= ~mask; // remove completed thread from active mask
			 } else if (tmp_next_pc == null_pc) {
                first_active_thread=m_shader->get_thread_state(wtid+i);
			    tmp_next_pc = first_active_thread->get_pc();
			    tmp_active_mask |= mask;
			    top_active_mask &= ~mask;
			 } else if (tmp_next_pc == m_shader->get_thread_state(wtid+i)->get_pc()) {
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
	assert(scheduled_warp->m_stack_top < m_warp_size * 2);
}

void gpgpu_sim::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
    m_sc[sid]->get_pdom_stack_top_info(tid,pc,rpc);
}

void shader_core_ctx::get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc )
{
    unsigned warp_id = tid/m_config->warp_size;
    m_pdom_warp[warp_id]->get_pdom_stack_top_info(pc,rpc);
}

void pdom_warp_ctx_t::get_pdom_stack_top_info( unsigned *pc, unsigned *rpc )
{
   *pc = m_pc[m_stack_top];
   *rpc = m_recvg_pc[m_stack_top];
}

unsigned pdom_warp_ctx_t::get_rp() const 
{ 
    return m_recvg_pc[m_stack_top]; 
}

void pdom_warp_ctx_t::print (FILE *fout) const
{
    const pdom_warp_ctx_t *warp=this;
    for ( unsigned k=0; k <= warp->m_stack_top; k++ ) {
        if ( k==0 ) {
            fprintf(fout, "w%02d %1u ", m_warp_id, k );
        } else {
            fprintf(fout, "    %1u ", k );
        }
        for (unsigned m=1,j=0; j<m_warp_size; j++, m<<=1)
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
            fprintf(fout," " );
        }
        ptx_print_insn( warp->m_pc[k], fout );
        fprintf(fout,"\n");
    }
}


void shader_core_ctx::new_cache_window()
{
    m_L1D->shd_cache_new_window();
    m_L1T->shd_cache_new_window();
    m_L1C->shd_cache_new_window();
}

void shader_core_ctx::fetch_mimd()
{
   if (fetch_stalled()) 
      return; 
   clear_stage_reg(TS_IF);

   for (unsigned i=0, j=0;i<m_config->n_thread_per_shader && j< m_config->warp_size;i++) {
      int thd_id = (i + m_last_issued_thread + 1) % m_config->n_thread_per_shader;
      if (m_thread[thd_id].m_avail4fetch && !m_thread[thd_id].m_waiting_at_barrier ) {
         shader_issue_thread(thd_id, j,(unsigned)-1);
         m_last_issued_thread = thd_id;
         j++;
      }
   }
}

// seperate the incoming warp into multiple warps with seperate pcs
int shader_core_ctx::split_warp_by_pc(int *tid_in, int **tid_split, address_type *pc) 
{
   unsigned n_pc = 0;
   static int *pc_cnt = NULL; // count the number of threads with the same pc

   assert(tid_in);
   assert(tid_split);
   assert(pc);
   memset(pc,0,sizeof(address_type)*m_config->warp_size);

   if (!pc_cnt) pc_cnt = (int*) malloc(sizeof(int)*m_config->warp_size);
   memset(pc_cnt,0,sizeof(int)*m_config->warp_size);

   // go through each thread in the given warp
   for (unsigned i=0; i< m_config->warp_size; i++) {
      if (tid_in[i] < 0) continue;
      int matched = 0;
      address_type thd_pc;
      thd_pc = next_pc(tid_in[i]);

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
         assert(n_pc < m_config->warp_size);
         tid_split[n_pc][0] = tid_in[i];
         pc[n_pc] = thd_pc;
         pc_cnt[n_pc] = 1;
         n_pc++;
      }
   }
   return n_pc;
}

// see if this warp just executed the barrier instruction 
int shader_core_ctx::warp_reached_barrier(int *tid_in)
{
   int reached_barrier = 0;
   for (unsigned i=0; i<m_config->warp_size; i++) {
      if (tid_in[i] < 0) continue;
      if (m_thread[tid_in[i]].m_reached_barrier) {
         reached_barrier = 1;
         break;
      }
   }
   return reached_barrier;
}

// seperate the incoming warp into multiple warps with seperate pcs and cta
int shader_core_ctx::split_warp_by_cta(int *tid_in, int **tid_split, address_type *pc, int *cta) 
{
   unsigned n_pc = 0;
   static int *pc_cnt = NULL; // count the number of threads with the same pc

   assert(tid_in);
   assert(tid_split);
   assert(pc);
   memset(pc,0,sizeof(address_type)*m_config->warp_size);

   if (!pc_cnt) pc_cnt = (int*) malloc(sizeof(int)*m_config->warp_size);
   memset(pc_cnt,0,sizeof(int)*m_config->warp_size);

   // go through each thread in the given warp
   for (unsigned i=0; i<m_config->warp_size; i++) {
      if (tid_in[i] < 0) continue;
      int matched = 0;
      address_type thd_pc;
      thd_pc = next_pc(tid_in[i]);

      int thd_cta = ptx_thread_get_cta_uid( m_thread[tid_in[i]].m_functional_model_thread_state );

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
         assert(n_pc < m_config->warp_size);
         tid_split[n_pc][0] = tid_in[i];
         pc[n_pc] = thd_pc;
         cta[n_pc] = thd_cta;
         pc_cnt[n_pc] = 1;
         n_pc++;
      }
   }
   return n_pc;
}

void shader_core_ctx::fetch_simd_dwf()
{
   static int *tid_in = NULL;
   static int *tid_out = NULL;

   if (!tid_in) {
      tid_in = (int*) malloc(sizeof(int)*m_config->warp_size);
      memset(tid_in, -1, sizeof(int)*m_config->warp_size);
   }
   if (!tid_out) {
      tid_out = (int*) malloc(sizeof(int)*m_config->warp_size);
      memset(tid_out, -1, sizeof(int)*m_config->warp_size);
   }


   static int **tid_split = NULL;
   if (!tid_split) {
      tid_split = (int**)malloc(sizeof(int*)*m_config->warp_size);
      tid_split[0] = (int*)malloc(sizeof(int)*m_config->warp_size*m_config->warp_size);
      for (unsigned i=1; i<m_config->warp_size; i++) {
         tid_split[i] = tid_split[0] + m_config->warp_size * i;
      }
   }

   static address_type *thd_pc = NULL;
   if (!thd_pc) thd_pc = (address_type*)malloc(sizeof(address_type)*m_config->warp_size);
   static int *thd_cta = NULL;
   if (!thd_cta) thd_cta = (int*)malloc(sizeof(int)*m_config->warp_size);

   int warpupdate_bw = 1;
   while (!m_thd_commit_queue->empty() && warpupdate_bw > 0) {
      // grab a committed warp, split it into multiple BRUs (tid_split) by PC
      std::vector<int> *tid_commit = m_thd_commit_queue->pop(gpu_sim_cycle);
      memset(tid_split[0], -1, sizeof(int)*m_config->warp_size*m_config->warp_size);
      memset(thd_pc, 0, sizeof(address_type)*m_config->warp_size);
      memset(thd_cta, -1, sizeof(int)*m_config->warp_size);

      int reached_barrier = warp_reached_barrier(tid_commit->data());

      unsigned n_warp_update;
      if (reached_barrier) {
         n_warp_update = split_warp_by_cta(tid_commit->data(), tid_split, thd_pc, thd_cta);
      } else {
         n_warp_update = split_warp_by_pc(tid_commit->data(), tid_split, thd_pc);
      }

      if (n_warp_update > 2) m_stats->gpgpu_commit_pc_beyond_two++;
      warpupdate_bw -= n_warp_update;
      // put the splitted warp updates into the DWF scheduler
      for (unsigned i=0;i<n_warp_update;i++) {
         for (unsigned j=0;j<m_config->warp_size;j++) {
            if (tid_split[i][j] < 0) continue;
            assert(m_thread[tid_split[i][j]].m_avail4fetch);
            assert(!m_thread[tid_split[i][j]].m_in_scheduler);
            m_thread[tid_split[i][j]].m_in_scheduler = true;
         }
         dwf_clear_accessed(m_sid);
         if (reached_barrier) {
            dwf_update_warp_at_barrier(m_sid, tid_split[i], thd_pc[i], thd_cta[i]);
         } else {
            dwf_update_warp(m_sid, tid_split[i], thd_pc[i]);
         }
      }

      delete tid_commit;
   }

   // Track the #PC right after the warps are input to the scheduler
   dwf_update_statistics(m_sid);
   dwf_clear_policy_access(m_sid);

   if (fetch_stalled()) {
      return; 
   }
   clear_stage_reg(TS_IF);

   address_type scheduled_pc;
   dwf_issue_warp(m_sid, tid_out, &scheduled_pc);

   for (unsigned i=0; i<m_config->warp_size; i++) {
      int issue_tid = tid_out[i];
      if (issue_tid >= 0) {
         shader_issue_thread(issue_tid, i, (unsigned)-1);
         m_thread[issue_tid].m_in_scheduler = false;
         m_thread[issue_tid].m_reached_barrier = false;
         assert(m_pipeline_reg[TS_IF][i].pc == scheduled_pc);
      }
   }   
}

void gpgpu_sim::print_shader_cycle_distro( FILE *fout ) const
{
   fprintf(fout, "Warp Occupancy Distribution:\n");
   fprintf(fout, "Stall:%d\t", m_shader_stats->shader_cycle_distro[0]);
   fprintf(fout, "W0_Idle:%d\t", m_shader_stats->shader_cycle_distro[1]);
   fprintf(fout, "W0_Mem:%d", m_shader_stats->shader_cycle_distro[2]);
   for (unsigned i = 3; i < m_shader_config->warp_size + 3; i++) {
      fprintf(fout, "\tW%d:%d", i-2, m_shader_stats->shader_cycle_distro[i]);
   }
   fprintf(fout, "\n");
}


#define PROGRAM_MEM_START 0xF0000000 /* should be distinct from other memory spaces... 
                                        check ptx_ir.h to verify this does not overlap 
                                        other memory spaces */
void shader_core_ctx::fetch_new()
{
    if( m_inst_fetch_buffer.m_valid ) {
        // decode 1 or 2 instructions and place them into ibuffer
        address_type pc = m_inst_fetch_buffer.m_pc;
        const inst_t* pI1 = ptx_fetch_inst(pc);
        assert(pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(0,pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
        const inst_t* pI2 = ptx_fetch_inst(pc+pI1->isize);
        if( pI2 ) {
            m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(1,pI2);
            m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
        }
        m_inst_fetch_buffer.m_valid = false;
    }

    if( !m_inst_fetch_buffer.m_valid ) {
        // find an active warp with space in instruction buffer that is not already waiting on a cache miss
        // and get next 1-2 instructions from i-cache...
        for( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
            unsigned warp_id = (m_last_warp_fetched+1+i) % m_config->max_warps_per_shader;
            if( m_warp[warp_id].done() && !m_scoreboard->pendingWrites(warp_id) && !m_warp[warp_id].done_exit()
                && m_warp[warp_id].stores_done() && !m_warp[warp_id].inst_in_pipeline() ) {
                bool did_exit=false;
                for( unsigned t=0; t<m_config->warp_size;t++) {
                    unsigned tid=warp_id*m_config->warp_size+t;
                    if( m_thread[tid].m_functional_model_thread_state ) {
                        register_cta_thread_exit(tid);
                        m_not_completed -= 1;
                        m_thread[tid].m_functional_model_thread_state=NULL;
                        did_exit=true;
                    }
                }
                if( did_exit ) 
                    m_warp[warp_id].set_done_exit();
            }
            if( !m_warp[warp_id].done() && !m_warp[warp_id].imiss_pending() && m_warp[warp_id].ibuffer_empty() ) {
                address_type pc  = m_warp[warp_id].get_pc();
                address_type ppc = pc + PROGRAM_MEM_START;
                address_type wb=0;
                unsigned nbytes=16; 
                unsigned offset_in_block = pc & (m_L1I->get_line_sz()-1);
                if( (offset_in_block+nbytes) > m_L1I->get_line_sz() )
                    nbytes = (m_L1I->get_line_sz()-offset_in_block);
                enum cache_request_status status = m_L1I->access( (unsigned long long)pc, 0, gpu_sim_cycle, &wb );
                if( status != HIT ) {
                    unsigned req_size = READ_PACKET_SIZE;
                    if( m_gpu->fq_has_buffer(ppc, req_size, false, m_sid) ) {
                        m_last_warp_fetched=warp_id;
                        mshr_entry *mshr = new mshr_entry();
                        mshr->init(ppc,false,instruction_space,warp_id);
                        fq_push( pc, req_size, false, 
                                 NO_PARTIAL_WRITE, 
                                 warp_id, 
                                 mshr, 
                                 INST_ACC_R, pc );
                        m_warp[warp_id].set_imiss_pending(mshr);
                        m_warp[warp_id].set_last_fetch(gpu_sim_cycle);
                    }
                } else {
                    m_last_warp_fetched=warp_id;
                    m_inst_fetch_buffer = ifetch_buffer_t(pc,nbytes,warp_id);
                    m_warp[warp_id].set_last_fetch(gpu_sim_cycle);
                }
                break;
            }
        }
    }
}

int is_load ( const inst_t &inst )
{
   return (inst.op == LOAD_OP || inst.memory_op == memory_load);
}

int is_store ( const inst_t &inst )
{
   return (inst.op == STORE_OP || inst.memory_op == memory_store);
}

int is_const ( memory_space_t space ) 
{
   return((space.get_type() == const_space) || (space == param_space_kernel));
}

int is_local ( memory_space_t space ) 
{
   return (space == local_space) || (space == param_space_local);
}


void shader_core_ctx::ptx_exec_inst( inst_t &inst )
{
    m_thread[inst.hw_thread_id].m_functional_model_thread_state->ptx_exec_inst(inst);
    if( inst.callback.function != NULL ) 
       m_warp[inst.hw_thread_id/m_config->warp_size].inc_n_atomic();
    if (is_local(inst.space.get_type()) && (is_load(inst) || is_store(inst)))
       inst.memreqaddr = translate_local_memaddr(inst.memreqaddr, inst.hw_thread_id, m_gpu->num_shader());
}

void shader_core_ctx::issue_warp( const inst_t *pI, unsigned active_mask, inst_t *&warp, unsigned warp_id )
{
    m_warp[warp_id].ibuffer_free();
    assert(pI->valid());
    unsigned cta_id = (unsigned)-1;
    for ( unsigned t=0; t < m_config->warp_size; t++ ) {
        unsigned tid=m_config->warp_size*warp_id+t;
        warp[t] = *pI;
        warp[t].warp_active_mask = active_mask;
        if( active_mask & (1<<t) ) {
            cta_id = m_thread[tid].m_cta_id;
            warp[t].hw_thread_id = tid;
            warp[t].wlane = t;
            warp[t].uid = g_next_shader_inst_uid++;
            warp[t].issue_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
            ptx_exec_inst( warp[t] );
            if ( ptx_thread_done(tid) ) {
                m_warp[warp_id].inc_n_completed();
                m_warp[warp_id].ibuffer_flush();
            }
        }
    }
    assert( cta_id != (unsigned)-1 );
    if( pI->op == BARRIER_OP ) 
       set_at_barrier(cta_id,warp_id);
    else if( pI->op == MEMORY_BARRIER_OP ) 
       set_at_memory_barrier(warp_id);
    m_pdom_warp[warp_id]->pdom_update_warp_mask();
    m_scoreboard->reserveRegisters(warp_id, pI);
    m_warp[warp_id].set_next_pc(pI->pc + pI->isize);

    /////
    memset(m_fetch_tid_out, -1, sizeof(int)*m_config->warp_size);
    int n_thd_in_warp = 0;
    for (unsigned i=0; i<m_config->warp_size; i++) {
       m_fetch_tid_out[i] = warp[i].hw_thread_id;
       if (m_fetch_tid_out[i] >= 0) 
          n_thd_in_warp += 1;
    }

    m_new_warp_TS = false;

    // warp tracker keeps track of warps in the pipeline, let it know we are going to issue this warp
    assert( n_thd_in_warp > 0 );
    m_warp_tracker->wpt_register_warp(m_fetch_tid_out, pI->pc, n_thd_in_warp,m_config->warp_size);
}

void shader_core_ctx::decode_new()
{
    for ( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
        unsigned warp_id = (m_last_warp_issued+1+i) % m_config->max_warps_per_shader;
        unsigned checked=0;
        unsigned issued=0;
        while( !m_warp[warp_id].waiting() && !m_warp[warp_id].ibuffer_empty() && (checked < 2) && (issued < 2) ) {
            unsigned active_mask = m_pdom_warp[warp_id]->get_active_mask();
            const inst_t *pI = m_warp[warp_id].ibuffer_next();
            unsigned pc,rpc;
            m_pdom_warp[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
            if( pI ) {
                if( pc != pI->pc ) {
                    // control hazard
                    m_warp[warp_id].set_next_pc(pc);
                    m_warp[warp_id].ibuffer_flush();
                } else if ( !m_scoreboard->checkCollision(warp_id, pI) ) {
                    assert( m_warp[warp_id].inst_in_pipeline() );
                    if ( (pI->op != SFU_OP) && pipeline_regster_empty(m_pipeline_reg[ID_OC]) ) {
                        issue_warp(pI, active_mask, m_pipeline_reg[ID_OC], warp_id);
                        issued++;
                    } else if ( (pI->op == SFU_OP || pI->op == ALU_SFU_OP) && pipeline_regster_empty(m_pipeline_reg[ID_OC_SFU]) ) {
                        issue_warp(pI, active_mask, m_pipeline_reg[ID_OC_SFU], warp_id);
                        issued++;
                    } 
                }
            }
            m_warp[warp_id].ibuffer_step();
            checked++;
        }
        if ( issued ) {
            m_last_warp_issued=warp_id;
            break;
        }
    }
}

void shader_core_ctx::fetch()
{
   // check if decode stage is stalled
   bool decode_stalled = !pipeline_regster_empty( m_pipeline_reg[IF_ID] );

   // find a ready warp and put it in the TS_IF pipeline register
   switch (m_config->model) {
   case POST_DOMINATOR: fetch_simd_postdominator(); break;
   case DWF:  fetch_simd_dwf(); break;
   case MIMD: fetch_mimd(); break;
   default: fprintf(stderr, "Unknown scheduler: %d\n", m_config->model); assert(0); break;
   }

   memset(m_fetch_tid_out, -1, sizeof(int)*m_config->warp_size);

   if (m_config->gpgpu_no_divg_load && m_new_warp_TS && !decode_stalled) {

      // count number of active threads in this warp, determine PC value
      // record active threads in tid_out
      int n_thd_in_warp = 0;
      address_type pc_out = 0xDEADBEEF;
      for (unsigned i=0; i<m_config->warp_size; i++) {
         m_fetch_tid_out[i] = m_pipeline_reg[TS_IF][i].hw_thread_id;
         if (m_fetch_tid_out[i] >= 0) {
            n_thd_in_warp += 1;
            pc_out = m_pipeline_reg[TS_IF][i].pc;
         }
      }

      m_new_warp_TS = false;

      // warp tracker keeps track of warps in the pipeline, let it know we are going to issue this warp
      if( n_thd_in_warp > 0 ) 
         m_warp_tracker->wpt_register_warp(m_fetch_tid_out, pc_out, n_thd_in_warp,m_config->warp_size);

      // some statistics collection
      if (gpu_runtime_stat_flag & GPU_RSTAT_DWF_MAP) 
         m_thread_pc_tracker->set_threads_pc( m_fetch_tid_out, pc_out );
      if (gpgpu_cflog_interval != 0) {
         insn_warp_occ_log( m_sid, pc_out, n_thd_in_warp);
         shader_warp_occ_log( m_sid, n_thd_in_warp);
      }
      if ( m_config->gpgpu_warpdistro_shader < 0 || m_sid == (unsigned)m_config->gpgpu_warpdistro_shader ) {
         m_stats->shader_cycle_distro[n_thd_in_warp + 2] += 1;
         if (n_thd_in_warp == 0) 
            if (m_pending_mem_access == 0) 
               m_stats->shader_cycle_distro[1]++;
      }
   } else {
      if ( m_config->gpgpu_warpdistro_shader < 0 || m_sid == (unsigned)m_config->gpgpu_warpdistro_shader ) {
         m_stats->shader_cycle_distro[0] += 1;
      }
   }

   if (!decode_stalled) {
      for (unsigned i = 0; i < m_config->warp_size; i++) {
         int tid_tsif = m_pipeline_reg[TS_IF][i].hw_thread_id;
         address_type pc_out = m_pipeline_reg[TS_IF][i].pc;
         cflog_update_thread_pc(m_sid, tid_tsif, pc_out);
      }
   }

   if (enable_ptx_file_line_stats && !decode_stalled) {
      int TS_stage_empty = 1;
      for (unsigned i = 0; i < m_config->warp_size; i++) {
         if (m_pipeline_reg[TS_IF][i].hw_thread_id >= 0) {
            TS_stage_empty = 0;
            break;
         }
      }
      if (TS_stage_empty) {
         if (enable_ptx_file_line_stats) 
            ptx_file_line_stats_commit_exposed_latency(m_sid, 1);
      }
   }

   // if not, send the warp part to decode stage
   if (!decode_stalled) {
      check_stage_pcs(TS_IF);
      inst_t *fvi = first_valid_thread(m_pipeline_reg[TS_IF]);
      if( fvi ) 
         m_warp[fvi->hw_thread_id/m_config->warp_size].set_last_fetch(gpu_sim_cycle);
      move_warp(m_pipeline_reg[IF_ID],m_pipeline_reg[TS_IF]);
   }
}

address_type coalesced_segment(address_type addr, unsigned segment_size_lg2bytes)
{
   return  (addr >> segment_size_lg2bytes);
}

address_type shader_core_ctx::translate_local_memaddr(address_type localaddr, int tid, unsigned num_shader )
{
   // During functional execution, each thread sees its own memory space for local memory, but these
   // need to be mapped to a shared address space for timing simulation.  We do that mapping here.
   localaddr /=4;
   if (m_config->gpgpu_local_mem_map) {
      // Dnew = D*nTpC*nCpS*nS + nTpC*C + T%nTpC
      // C = S + nS*(T/nTpC)
      // D = data index; T = thread; C = CTA; S = shader core; p = per
      // keep threads in a warp contiguous
      // then distribute across memory space by CTAs from successive shader cores first, 
      // then by successive CTA in same shader core
      localaddr *= m_config->gpu_padded_cta_size * m_config->gpu_max_cta_per_shader * num_shader;
      localaddr += m_config->gpu_padded_cta_size * (m_sid + num_shader * (tid / m_config->gpu_padded_cta_size));
      localaddr += tid % m_config->gpu_padded_cta_size; 
   } else {
      // legacy mapping that maps the same address in the local memory space of all threads 
      // to a single contiguous address region 
      localaddr *= num_shader * m_config->n_thread_per_shader;
      localaddr += (m_config->n_thread_per_shader *m_sid) + tid;
   }
   localaddr *= 4;

   return localaddr;
}

/////////////////////////////////////////////////////////////////////////////////////////

void shader_core_ctx::decode()
{
   op_type op = NO_OP;
   unsigned warp_id = -1;
   unsigned cta_id = -1;

   address_type regs_regs_PC = 0xDEADBEEF;
   address_type warp_current_pc = 0x600DBEEF;
   address_type warp_next_pc = 0x600DBEEF;
   int       warp_diverging = 0;
   const int nextstage = (m_config->gpgpu_operand_collector) ? ID_OC : \
      (m_config->m_using_dwf_rrstage ? ID_RR : ID_EX);

   if( !pipeline_regster_empty(m_pipeline_reg[nextstage]) ) 
      return;

   check_stage_pcs(IF_ID);

   // decode the instruction 
   int first_valid_thread = -1;
   for (unsigned i=0; i<m_config->warp_size;i++) {
      if (m_pipeline_reg[IF_ID][i].hw_thread_id == -1 )
         continue; /* bubble or masked off */
      if (first_valid_thread == -1) {
         first_valid_thread = i;
         op = m_pipeline_reg[IF_ID][i].op; 
         int tid = m_pipeline_reg[IF_ID][i].hw_thread_id;
         warp_id = tid/m_config->warp_size;
         assert( !warp_waiting_at_barrier(warp_id) );
         cta_id = m_thread[tid].m_cta_id;
      } 
   }

   // execute the instruction functionally
   short last_hw_thread_id = -1;
   bool first_thread_in_warp = true;
   for (unsigned i=0; i<m_config->warp_size;i++) {
      if (m_pipeline_reg[IF_ID][i].hw_thread_id == -1 )
         continue; /* bubble or masked off */

      if(last_hw_thread_id > -1)
    	  first_thread_in_warp = false;
      last_hw_thread_id = m_pipeline_reg[IF_ID][i].hw_thread_id;

      /* get the next instruction to execute from fetch stage */
      int tid = m_pipeline_reg[IF_ID][i].hw_thread_id;

      // Functionally execute instruction
      m_thread[tid].m_functional_model_thread_state->ptx_exec_inst( m_pipeline_reg[IF_ID][i] );
      if( m_pipeline_reg[IF_ID][i].callback.function != NULL ) 
         m_warp[warp_id].inc_n_atomic();
      if (is_local(m_pipeline_reg[IF_ID][i].space) && (is_load(m_pipeline_reg[IF_ID][i]) || is_store(m_pipeline_reg[IF_ID][i])))
         m_pipeline_reg[IF_ID][i].memreqaddr = translate_local_memaddr(m_pipeline_reg[IF_ID][i].memreqaddr, tid, m_gpu->num_shader());

      // Mark destination registers as write-pending in scoreboard
      // Only do this for the first thread in warp
      if(first_thread_in_warp) 
           m_scoreboard->reserveRegisters(warp_id, &(m_pipeline_reg[IF_ID][i]));
      warp_current_pc = m_pipeline_reg[IF_ID][i].pc;
      regs_regs_PC = next_pc( tid );

      if ( ptx_thread_at_barrier( m_thread[tid].m_functional_model_thread_state ) ) {
         if (m_config->model == DWF) {
            m_thread[tid].m_waiting_at_barrier=true;
            m_thread[tid].m_reached_barrier=true; // not reset at barrier release, but at the issue after that
            m_warp[wid_from_hw_tid(tid,m_config->warp_size)].inc_waiting_at_barrier();
            int cta_uid = ptx_thread_get_cta_uid( m_thread[tid].m_functional_model_thread_state );
            dwf_hit_barrier( m_sid, cta_uid );
         
            int release = ptx_thread_all_at_barrier( m_thread[tid].m_functional_model_thread_state ); //test if all threads arrived at the barrier
            if ( release ) { //All threads arrived at barrier...releasing
               int cta_uid = ptx_thread_get_cta_uid( m_thread[tid].m_functional_model_thread_state );
               for ( unsigned t=0; t < m_config->n_thread_per_shader; ++t ) {
                  if ( !ptx_thread_at_barrier( m_thread[t].m_functional_model_thread_state ) )
                     continue;
                  int other_cta_uid = ptx_thread_get_cta_uid( m_thread[t].m_functional_model_thread_state );
                  if ( other_cta_uid == cta_uid ) { //reseting @barrier tracking info
                     m_warp[wid_from_hw_tid(t,m_config->warp_size)].clear_waiting_at_barrier();
                     m_thread[t].m_waiting_at_barrier=false;
                     ptx_thread_reset_barrier( m_thread[t].m_functional_model_thread_state );
                  }
               }
               if (m_config->model == DWF) {
                  dwf_release_barrier( m_sid, cta_uid );
               }
               ptx_thread_release_barrier( m_thread[tid].m_functional_model_thread_state );
            }
         }
      } else {
         assert( !m_thread[tid].m_waiting_at_barrier );
      }

      // branch divergence detection
      if (warp_next_pc != regs_regs_PC) {
         if (warp_next_pc == 0x600DBEEF) {
            warp_next_pc = regs_regs_PC;
         } else {
            warp_diverging = 1;
         }
      }
   }

   move_warp(m_pipeline_reg[nextstage],m_pipeline_reg[IF_ID]);

   if( op == BARRIER_OP ) 
      set_at_barrier(cta_id,warp_id);
   else if( op == MEMORY_BARRIER_OP ) 
      set_at_memory_barrier(warp_id);

   m_n_diverge += warp_diverging;
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

void shader_core_ctx::preexecute()
{
   if( m_config->gpgpu_dwf_reg_bankconflict) {
      // Model register bank conflicts as in 
      // Fung et al. MICRO'07 / ACM TACO'09 papers.
      // 
      // This models conflicts due to moving threads to different SIMD lanes
      // (which occur if not using "lane aware" dynamic warp formation).

      inst_t *fvi = first_valid_thread(m_pipeline_reg[RR_EX]);
      if( fvi ) {
         if (m_dwf_RR_k) {
            //stalled due to register access conflict, but can still service a register read
            m_dwf_RR_k--;  
            return;
         }
      
         int n_access_per_cycle = m_config->warp_size / m_config->gpgpu_dwf_rr_stage_n_reg_banks;
         int max_reg_bank_acc = 0;
         for (unsigned i=0; i<m_config->gpgpu_dwf_rr_stage_n_reg_banks; i++) 
            m_dwf_rrstage_bank_access_counter[i] = 0;
         for (unsigned i=0; i<m_config->warp_size; i++) {
            if (m_pipeline_reg[ID_RR][i].hw_thread_id != -1 )
               m_dwf_rrstage_bank_access_counter[regfile_hash(m_pipeline_reg[ID_RR][i].hw_thread_id, 
                                                              m_config->warp_size, 
                                                              m_config->gpgpu_dwf_rr_stage_n_reg_banks)]++;
         }
         for (unsigned i=0; i<m_config->gpgpu_dwf_rr_stage_n_reg_banks; i++) {
            if (m_dwf_rrstage_bank_access_counter[i] > max_reg_bank_acc ) 
               max_reg_bank_acc = m_dwf_rrstage_bank_access_counter[i];
         }
         // calculate the number of cycles needed for each register bank to fulfill all accesses
         m_dwf_RR_k = (max_reg_bank_acc / n_access_per_cycle) + ((max_reg_bank_acc % n_access_per_cycle)? 1 : 0);
   
         // if there is more than one access cycle needed at a bank, stall
         if (m_dwf_RR_k > 1) {
            n_regconflict_stall++;
            m_dwf_RR_k--;
            return; 
         }
      }
   
      check_stage_pcs(ID_RR);
      m_dwf_RR_k = 0;
   }

   if( pipeline_regster_empty(m_pipeline_reg[ID_EX]) ) 
      move_warp(m_pipeline_reg[ID_EX],m_pipeline_reg[ID_RR]);
}


void shader_core_ctx::execute_pipe( unsigned pipeline, unsigned next_stage ) 
{
   if (m_config->gpgpu_pre_mem_stages) {
      if( !pipeline_regster_empty(pre_mem_pipeline[0]) ) 
         return; // stalled
   } else {
      if( !pipeline_regster_empty(m_pipeline_reg[next_stage]) ) 
         return; // stalled
   }
         
   check_stage_pcs(ID_EX);

   // Check that all threads have the same delay cycles
   unsigned cycles = -1;
   for (unsigned i=0; i<m_config->warp_size; i++) {
      if (m_pipeline_reg[pipeline][i].hw_thread_id == -1 )
         continue; // bubble
      if(cycles == (unsigned)-1)
         cycles = m_pipeline_reg[pipeline][i].cycles;
      else {
         if( cycles != m_pipeline_reg[pipeline][i].cycles ) {
            printf("Shader %d: threads do not have the same delay cycles.\n", m_sid);
            assert(0);
         }
      }
   }

   bool stall_inst_not_done = false;
   for (unsigned i=0; i<m_config->warp_size; i++) {
      if (m_pipeline_reg[pipeline][i].hw_thread_id == -1 )
         continue;
      m_pipeline_reg[pipeline][i].cycles--;
      if( m_pipeline_reg[pipeline][i].cycles > 0 ) {
         // Stall here to model instruction throughput for different types of instructions
         stall_inst_not_done=true;
         continue;
      }
   } 
   if( stall_inst_not_done ) 
      return;
   if (m_config->gpgpu_pre_mem_stages) {
      move_warp(pre_mem_pipeline[0], m_pipeline_reg[pipeline]);
   } else {
      move_warp(m_pipeline_reg[next_stage],m_pipeline_reg[pipeline]);
      // inform memory stage that a new instruction has arrived 
      m_shader_memory_new_instruction_processed = false;
   }
}

void shader_core_ctx::execute()
{
   execute_pipe(OC_EX_SFU, EX_MM);
   execute_pipe(ID_EX, EX_MM);
}

void shader_core_ctx::pre_memory() 
{
   // This stage can be used to approximately model a deeper pipeline. 
   // The main effect this models is the register read-after-write delay.
   // We walk through pre-memory stages in reverse order 
   // (highest number = stage closest to writeback, 0 = stage closest to fetch
   if( pipeline_regster_empty(m_pipeline_reg[EX_MM]) ) {
      move_warp( m_pipeline_reg[EX_MM], pre_mem_pipeline[m_config->gpgpu_pre_mem_stages] );
      // inform memory stage that a new instruction has arrived 
      m_shader_memory_new_instruction_processed = false;
   }
   for (unsigned j = m_config->gpgpu_pre_mem_stages; j > 0; j--) {
      if( pipeline_regster_empty(pre_mem_pipeline[j]) )
         move_warp( pre_mem_pipeline[j], pre_mem_pipeline[j-1]);
   }
}

mshr_entry* mshr_shader_unit::add_mshr(mem_access_t &access, inst_t* warp)
{
    //creates an mshr based on the access struct information
    mshr_entry* mshr = alloc_free_mshr(access.space == tex_space);
    mshr->init(access.addr,access.iswrite,access.space,warp->hw_thread_id/m_shader_config->warp_size);
    assert(access.warp_indices.size()); //code assumes at least one instruction attached to mshr.
    for (unsigned i = 0; i < access.warp_indices.size(); i++)
        mshr->add_inst(warp[access.warp_indices[i]]);
    if (m_shader_config->gpgpu_interwarp_mshr_merge) {
        mshr_entry* mergehit = m_mshr_lookup.shader_get_mergeable_mshr(mshr);
        if (mergehit) {
            mergehit->merge(mshr);
            if (mergehit->fetched())
                mshr_return_from_mem(mshr);
        }
        m_mshr_lookup.mshr_fast_lookup_insert(mshr);
    }
    return mshr;
}

address_type line_size_based_tag_func(address_type address, unsigned line_size)
{
   //gives the tag for an address based on a given line size
   return ((address) & (~((address_type)line_size - 1)));
}

address_type null_tag_func(address_type address, unsigned line_size)
{
   return address; //no modification: each address is its own tag. Equivalent to line_size_based_tag_func(address,1), but line_size ignored.
}

// only 1 bank
int shader_core_ctx::null_bank_func(address_type add, unsigned line_size)
{
   return 1;
}

int shader_core_ctx::shmem_bank_func(address_type addr, unsigned line_size)
{
   //returns the integer number of the physical bank addr would go in.
   return ((int)(addr/((address_type)WORD_SIZE)) % m_config->gpgpu_n_shmem_bank);
}

int shader_core_ctx::dcache_bank_func(address_type add, unsigned line_size)
{
   //returns the integer number of the physical bank addr would go in.
   if (m_config->gpgpu_no_dl1) return 1; //no banks
   else return (add / line_size) & (m_config->gpgpu_n_cache_bank - 1);
}

typedef int (shader_core_ctx::*bank_func_t)(address_type add, unsigned line_size);
typedef address_type (*tag_func_t)(address_type add, unsigned line_size);

void shader_core_ctx::get_memory_access_list(
                               shader_core_ctx::bank_func_t bank_func,
                               tag_func_t tag_func,
                               memory_pipe_t mem_pipe, 
                               unsigned warp_parts, 
                               unsigned line_size, 
                               bool limit_broadcast, 
                               std::vector<mem_access_t> &accessq )
{
   const inst_t* insns = m_pipeline_reg[EX_MM];
   // Calculates memory accesses generated by this warp
   // Returns acesses which are "coalesced" 
   // Does not coalesce nor overlap bank accesses across warp "parts".


   // This is called once per warp when it enters the memory stage.
   // It takes the warp and produces a queue of accesses that can be peformed.
   // These are then performed over multiple cycles (stalling the pipeline) if the accessses cannot be 
   // performed all at once.
   // It is a convenience for simulation; in hardware the warp would be processed each cycle
   // until it was done. Each cycle would do the first accesses available to it and mark off the
   // those threads served by those accesses. 

   // Because it calculates all the accesses at once, what follows is largely not as the hw would do it.
   // Accesses are assigned an order number based on when that access may be issued. 
   // Accesses with the same order number may occur at the same time: they are to different banks. 
   // Later when the queue is processed it will evaluate accesses of 
   // as many orders as ports on that cache/shmem. 
   // These accesses are placed into a queue and sorted so that accesses of the same order are next to each other.


   // tracks bank accesses for sorting into generations;
   // each entry is (bank #, number of accesses)
   // the idea is that you can only access a bank a number of times each cycle equal to 
   // its number of ports in one cycle. 
   std::map<unsigned,unsigned> bank_accs;

   //keep track of broadcasts with unique orders if limit_broadcast
   //the normally calculated orders will never be greater than warp_size
   unsigned broadcast_order =  m_config->warp_size;
   unsigned qbegin = accessq.size();
   unsigned qpartbegin = qbegin;
   unsigned mem_pipe_size = m_config->warp_size / warp_parts;
   for (unsigned part = 0; part < m_config->warp_size; part += mem_pipe_size) {
      for (unsigned i = part; i < part + mem_pipe_size; i++) {
         if ( insns[i].hw_thread_id == -1 ) 
            continue;

         if( insns[i].space == undefined_space ) {
        	 // Instruction must have been predicated off
        	 continue;
         }

         address_type lane_segment_address = tag_func(insns[i].memreqaddr, line_size);
         unsigned quarter = 0;
         if( line_size>=4 )
            quarter = (insns[i].memreqaddr / (line_size/4)) & 3;
         bool isatomic = (insns[i].callback.function != NULL);
         bool match = false;
         if (not isatomic) { //atomics must have own request
            for (unsigned j = qpartbegin; j < accessq.size(); j++) {
               if (lane_segment_address == accessq[j].addr) {
                  assert( not accessq[j].isatomic );
                  accessq[j].quarter_count[quarter]++;
                  accessq[j].warp_indices.push_back(i);
                  if (limit_broadcast) 
		     // two threads access this address, so its a broadcast. 
                     accessq[j].order = ++broadcast_order; //do broadcast in its own cycle.
                  match = true;
                  break;
               }
            }
         }
         if (!match) { // does not match an previous request by another thread, so need a new request
            assert( insns[i].space != undefined_space );
            accessq.push_back( mem_access_t( lane_segment_address, 
                                             insns[i].space, 
                                             mem_pipe, 
                                             isatomic, 
                                             is_store(insns[i]),
                                             line_size, quarter, i) );
            // Determine Bank Conflicts:
            unsigned bank = (this->*bank_func)(insns[i].memreqaddr, line_size);
            // ensure no concurrent bank access accross warp parts. 
            // ie. order will be less than part for all previous loads in previous parts, so:
            if (bank_accs[bank] < part) 
               bank_accs[bank]=part; 
            accessq.back().order = bank_accs[bank];
            bank_accs[bank]++;
         }
      }
      qpartbegin = accessq.size(); //don't coalesce accross warp parts
   }
   //sort requests into order according to order (orders will not necessarily be consequtive if multiple parts)
   std::stable_sort(accessq.begin()+qbegin,accessq.end());
} 


void shader_core_ctx::memory_shared_process_warp()
{
   // initial processing of shared memory warps 
   get_memory_access_list(&shader_core_ctx::shmem_bank_func, 
                          null_tag_func,
                          SHARED_MEM_PATH, 
                          m_config->gpgpu_shmem_pipe_speedup, 
                          1, //shared memory doesn't care about line_size, needs to be at least 1;
                          true, // limit broadcasts to single cycle. 
                          m_memory_queue.shared);
}

void shader_core_ctx::memory_const_process_warp()
{
   // initial processing of const memory warps 
   std::vector<mem_access_t> &accessq = m_memory_queue.constant;
   unsigned qbegin = accessq.size();
   get_memory_access_list(
      &shader_core_ctx::null_bank_func, 
      line_size_based_tag_func,
      CONSTANT_MEM_PATH, 
      1, //warp parts 
      m_L1C->get_line_sz(), false, //no broadcast limit.
      accessq);
   //do cache checks here for each request (non-physical), could be done later for more accurate timing of cache accesses, but probably uneccesary; 
   for (unsigned i = qbegin; i < accessq.size(); i++) {
      if ( accessq[i].space == param_space_kernel ) {
         accessq[i].cache_hit = true;
      } else {
         cache_request_status status = m_L1C->access( accessq[i].addr,
                                                      0, //should always be a read
                                                      gpu_sim_cycle+gpu_tot_sim_cycle, 
                                                      NULL/*should never writeback*/);
         accessq[i].cache_hit = (status == HIT);
         if (m_config->gpgpu_perfect_mem) accessq[i].cache_hit = true;
	 if (accessq[i].cache_hit) m_stats->L1_const_miss++;
      } 
      accessq[i].cache_checked = true;
   }
}

void shader_core_ctx::memory_texture_process_warp()
{
   // initial processing of shared texture warps 
   std::vector<mem_access_t> &accessq = m_memory_queue.texture;
   unsigned qbegin = accessq.size();
   get_memory_access_list(&shader_core_ctx::null_bank_func, 
                          &line_size_based_tag_func,
                          TEXTURE_MEM_PATH, 
                          1, //warp parts
                          m_L1T->get_line_sz(),
			  false, //no broadcast limit.
                          accessq);
   //do cache checks here for each request (non-hardware), could be done later for more accurate timing of cache accesses, but probably uneccesary; 
   for (unsigned i = qbegin; i < accessq.size(); i++) {
      cache_request_status status = m_L1T->access( accessq[i].addr,
                                                   0, //should always be a read
                                                   gpu_sim_cycle+gpu_tot_sim_cycle, 
                                                   NULL /*should never writeback*/);
      accessq[i].cache_hit = (status == HIT);
      if (m_config->gpgpu_perfect_mem) accessq[i].cache_hit = true;
      if (accessq[i].cache_hit) m_stats->L1_texture_miss++;
      accessq[i].cache_checked = true;
   }
}

void shader_core_ctx::memory_global_process_warp()
{
   std::vector<mem_access_t> &accessq = m_memory_queue.local_global;
   unsigned qbegin = accessq.size();
   unsigned warp_parts = 1;
   unsigned line_size = m_L1D->get_line_sz();
   if (m_config->gpgpu_coalesce_arch == 13) {
      warp_parts = 2;
      if(m_config->gpgpu_no_dl1) {
         unsigned data_size = first_valid_thread( m_pipeline_reg[EX_MM] )->data_size;
         // line size is dependant on instruction;
         switch (data_size) {
         case 1: line_size = 32; break;
         case 2: line_size = 64; break;
         case 4: case 8: case 16: line_size = 128; break;
         default: assert(0);
         }
      }
   }          
   get_memory_access_list( &shader_core_ctx::dcache_bank_func,
                           &line_size_based_tag_func, 
                           GLOBAL_MEM_PATH, 
                           warp_parts,
                           line_size,
                           false,
                           accessq);

   // Now that we have the accesses, if we don't have a cache we can adjust request sizes to 
   // include only the data referenced by the threads 
   for (unsigned i = qbegin; i < accessq.size(); i++) {
      if (m_config->gpgpu_coalesce_arch == 13 and m_config->gpgpu_no_dl1) {
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
      


mem_stage_stall_type shader_core_ctx::send_mem_request(mem_access_t &access)
{
   //Atempt to send an request/write to memory based on information in access.  

   inst_t* warp = m_pipeline_reg[EX_MM];
   inst_t* req_head = warp + access.warp_indices[0];

   // If the cache told us it needed to write back a dirty line, do this now
   // It is possible to do this writeback in the same cycle as the access request, this may not be realistic.
   if (access.need_wb) {
      unsigned req_size = m_L1D->get_line_sz() + WRITE_PACKET_SIZE;
      if ( ! m_gpu->fq_has_buffer(access.wb_addr, req_size, true, m_sid) ) {
         m_stats->gpu_stall_sh2icnt++; 
         return WB_ICNT_RC_FAIL;
      }
      fq_push( access.wb_addr, req_size, true, NO_PARTIAL_WRITE, -1, NULL, 
               is_local(access.space)?LOCAL_ACC_W:GLOBAL_ACC_W, //space of cache line is same as new request
		        -1);
      m_stats->L1_writeback++;
      access.need_wb = false; 
   }

   mem_access_type access_type;
   switch(access.space.get_type()) {
   case const_space:
   case param_space_kernel: access_type = CONST_ACC_R;   break;
   case tex_space:          access_type = TEXTURE_ACC_R;   break;
   case global_space:       access_type = (access.iswrite)? GLOBAL_ACC_W: GLOBAL_ACC_R;   break;
   case local_space:
   case param_space_local:  access_type = (access.iswrite)? LOCAL_ACC_W: LOCAL_ACC_R;   break;
   default: assert(0); break; 
   }
   //reserve mshr
   bool requires_mshr = (m_config->model != MIMD) and (not access.iswrite);
   if (requires_mshr and not access.reserved_mshr) {
      if (not m_mshr_unit->has_mshr(1)) 
         return MSHR_RC_FAIL;
      access.reserved_mshr = m_mshr_unit->add_mshr(access, warp);
      access.recheck_cache = false; //we have an mshr now, so have checked cache in same cycle as checking mshrs, so have merged if necessary.
   }
   //require inct if access is this far without reserved mshr, or has and mshr but not merged with another request
   bool requires_icnt = (not access.reserved_mshr) or (not access.reserved_mshr->ismerged() );
   if (requires_icnt) {
      //calculate request size for icnt check (and later send);
      unsigned request_size = access.req_size;
      if (access.iswrite) {
         if (requires_mshr) 
            request_size += READ_PACKET_SIZE + WRITE_MASK_SIZE; // needs information for a load back into cache.
         else 
            request_size += WRITE_PACKET_SIZE + WRITE_MASK_SIZE; //plain write
      }
      if ( !m_gpu->fq_has_buffer(access.addr, request_size, access.iswrite, m_sid) ) {
         // can't allocate icnt
         m_stats->gpu_stall_sh2icnt++;
         return ICNT_RC_FAIL;
      }
      //send over interconnect
      partial_write_mask_t  write_mask = NO_PARTIAL_WRITE;
      unsigned warp_id = req_head->hw_thread_id/m_config->warp_size;
      if (access.iswrite) {
         if (!strcmp("GT200",m_config->pipeline_model) ) 
            m_warp[warp_id].inc_store_req();
         for (unsigned i=0;i < access.warp_indices.size();i++) {
            unsigned w = access.warp_indices[i];
            int data_offset = warp[w].memreqaddr & ((unsigned long long int)access.req_size - 1);
            for (unsigned b = data_offset; b < data_offset + warp[w].data_size; b++) write_mask.set(b);
         }
         if (write_mask.count() != access.req_size) 
            m_stats->gpgpu_n_partial_writes++;
      }
      fq_push( access.addr, request_size,
               access.iswrite, write_mask, warp_id , access.reserved_mshr, 
               access_type, req_head->pc);
   }

   // book keeping for mshr : this request is done (sent/accounted for)
   if (requires_mshr) {
      for (unsigned i = 0; i < access.warp_indices.size(); i++) {
         unsigned o = access.warp_indices[i];
         m_pending_mem_access++;
         if (enable_ptx_file_line_stats) 
             ptx_file_line_stats_add_inflight_memory_insn(m_sid, warp[o].pc);
      }

      // Scoreboard addition: do not make cache miss instructions wait for memory,
      //                      let the scoreboard handle stalling of instructions.
      //                      Mark thread as a cache miss
      if (not access.iswrite) {
         // set the pipeline instructions in this request to noops, they all wait for memory;
         for (unsigned i = 0; i < access.warp_indices.size(); i++) {
            unsigned o = access.warp_indices[i];
            m_pipeline_reg[EX_MM][o].cache_miss = true;
         }
      }
   }
   return NO_RC_FAIL;
}     


bool shader_core_ctx::memory_shared_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   // Process a single cycle of activity from the shared memory queue.

   std::vector<mem_access_t> &accessq  = m_memory_queue.shared;
   //consume port number orders from the top of the queue;
   for (int i = 0; i < m_config->gpgpu_shmem_port_per_bank; i++) {
      if (accessq.empty()) 
         break;
      unsigned current_order = accessq.back().order;
      //consume all requests of the same order (concurrent bank requests)
      while ((not accessq.empty()) and accessq.back().order == current_order) 
         accessq.pop_back();
   }
   if (not accessq.empty()) {
      rc_fail = BK_CONF;
      fail_type = S_MEM;
      m_stats->gpgpu_n_shmem_bkconflict++;
   }
   return accessq.empty(); //done if empty.
}

mem_stage_stall_type shader_core_ctx::process_memory_access_queue( shader_core_ctx::cache_check_t cache_check,
                                                                   unsigned ports_per_bank, 
                                                                   unsigned memory_send_max, 
                                                                   std::vector<mem_access_t> &accessq )
{
   // Generic algorithm for processing a single cycle of accesses for the memory space types that go to L2 or DRAM.

   // precondition: accessq sorted by mem_access_t::order
   mem_stage_stall_type hazard_cond = NO_RC_FAIL; 
   unsigned mem_req_count = 0; // number of requests to sent to memory this cycle
   for (unsigned i = 0; i < ports_per_bank; i++) {
      if (accessq.empty()) 
         break;
      unsigned current_order = accessq.back().order;
      // consume all requests of the same "order" but stop if we hit a structural hazard
      while ((not accessq.empty()) and accessq.back().order == current_order and hazard_cond == NO_RC_FAIL) {
         hazard_cond = (this->*cache_check)(accessq.back());
         if (hazard_cond != NO_RC_FAIL) 
            break; // can't complete this request this cycle.
         if (not accessq.back().cache_hit){
            if (mem_req_count < memory_send_max) {
               mem_req_count++;
               hazard_cond = send_mem_request(accessq.back()); // attemp to get mshr, icnt, send;
            }
            else hazard_cond = COAL_STALL; // not really a coal stall, its a too many memory request stall;
            if ( hazard_cond != NO_RC_FAIL) break; //can't complete this request this cycle.
         }
         accessq.pop_back();
      }
   }
   if (not accessq.empty() and hazard_cond == NO_RC_FAIL) {
      //no resource failed so must be a bank comflict.
      hazard_cond = BK_CONF;
   }
   return hazard_cond;
}  

bool shader_core_ctx::memory_constant_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   // Process a single cycle of activity from the the constant memory queue.

   std::vector<mem_access_t> &accessq = m_memory_queue.constant; 

   mem_stage_stall_type fail = process_memory_access_queue(&shader_core_ctx::ccache_check, 
                                                           m_config->gpgpu_const_port_per_bank, 
                                                           1, //memory send max per cycle
                                                           accessq );
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = C_MEM;
      if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
         m_stats->gpgpu_n_cmem_portconflict++; //coal stalls aren't really a bank conflict, but this maintains previous behavior.
      }
   }
   return accessq.empty(); //done if empty.
}

bool shader_core_ctx::memory_texture_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   // Process a single cycle of activity from the the texture memory queue.

   std::vector<mem_access_t> &accessq = m_memory_queue.texture; 
   mem_stage_stall_type fail = process_memory_access_queue(&shader_core_ctx::tcache_check, 
                                                           1, //how is tex memory banked? 
                                                           1, //memory send max per cycle
                                                           accessq );
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = T_MEM;
   }
   return accessq.empty(); //done if empty.
}


mem_stage_stall_type shader_core_ctx::dcache_check(mem_access_t& access)
{
   // Global memory (data cache) checks the cache for each access at the time it is processed.
   // This is more accurate to hardware, and necessary for proper action of the writeback cache.

   if (access.cache_checked and not access.recheck_cache) 
      return NO_RC_FAIL;
   if (!m_config->gpgpu_no_dl1 && !m_config->gpgpu_perfect_mem) { 
      //check cache
      cache_request_status status = m_L1D->access( access.addr,
                                                   access.iswrite,
                                                   gpu_sim_cycle+gpu_tot_sim_cycle,
                                                   &access.wb_addr );
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
         m_stats->L1_write_hit_on_miss++;
         // here we would search the MSHRs for the originating read, 
         // and mask off the writen bytes, so they are not overwritten in the cache when it comes back
         // --- don't actually do this since we don't functionally execute based upon values in cache
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

   if (m_config->gpgpu_perfect_mem) access.cache_hit = true;
   
   if (access.isatomic) {
      if (m_config->gpgpu_perfect_mem) {
         // complete functional execution of atomic here
         dram_callback_t &atom_exec = m_pipeline_reg[EX_MM][access.warp_indices[0]].callback;
         atom_exec.function(atom_exec.instruction, atom_exec.thread);
      } else {
         // atomics always go to memory 
         access.cache_hit = false;
      }
   }

   if (!access.cache_hit) { 
      if (access.iswrite) m_stats->L1_write_miss++;
      else m_stats->L1_read_miss++;
   }
   return NO_RC_FAIL;
}

bool shader_core_ctx::memory_cycle( mem_stage_stall_type &stall_reason, mem_stage_access_type &access_type)
{
   // Process a single cycle of activity from the the global/local memory queue.

   std::vector<mem_access_t> &accessq = m_memory_queue.local_global; 
   mem_stage_stall_type stall_cond = process_memory_access_queue(&shader_core_ctx::dcache_check, m_config->gpgpu_cache_port_per_bank, 1, accessq);

   if (stall_cond != NO_RC_FAIL) {
      stall_reason = stall_cond;
      bool iswrite = accessq.back().iswrite;
      if (is_local(accessq.back().space)) 
         access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
      else 
         access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
      if (stall_cond == BK_CONF or stall_cond == COAL_STALL) 
         m_stats->gpgpu_n_cache_bkconflict++;
   }
   return accessq.empty(); //done if empty.
}

void shader_core_ctx::memory_queue()
{
   // Called once per warp when warp enters memory stage.
   // Generates a list of memory accesses, but does not perform the memory access.

   if( pipeline_regster_empty(m_pipeline_reg[EX_MM]) )
       return;
   m_gpu->mem_instruction_stats(m_pipeline_reg[EX_MM]);
   inst_t *inst = first_valid_thread(m_pipeline_reg[EX_MM]);
   switch (inst->space.get_type()) {
      case shared_space: memory_shared_process_warp(); break;
      case tex_space:    memory_texture_process_warp(); break;
      case const_space:  case param_space_kernel: memory_const_process_warp(); break;
      case global_space: case local_space: case param_space_local: memory_global_process_warp(); break;
      case param_space_unclassified: abort(); break;
      default: break; // non-memory operations
   }
}

void shader_core_ctx::memory()
{
   if (!m_shader_memory_new_instruction_processed) {
      m_shader_memory_new_instruction_processed = true; // do once per warp instruction
      memory_queue();
   }
   bool done = true;
   enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
   mem_stage_access_type type;
   done &= memory_shared_cycle(rc_fail, type);
   done &= memory_constant_cycle(rc_fail, type);
   done &= memory_texture_cycle(rc_fail, type);
   done &= memory_cycle(rc_fail, type);

   if (!done) { // log stall types and return
      assert(rc_fail != NO_RC_FAIL);
      m_stats->gpu_stall_shd_mem++;
      m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
      return;
   }
   if( not pipeline_regster_empty( m_pipeline_reg[MM_WB] ) )
      return; // writeback stalled
   check_stage_pcs(EX_MM);
   move_warp(m_pipeline_reg[MM_WB],m_pipeline_reg[EX_MM]);
}

void shader_core_ctx::register_cta_thread_exit(int tid )
{
   shader_core_ctx *shader = this;
   unsigned padded_cta_size = m_gpu->the_kernel().threads_per_cta();
   if (padded_cta_size%m_config->warp_size) 
      padded_cta_size = ((padded_cta_size/m_config->warp_size)+1)*(m_config->warp_size);
   int cta_num = tid/padded_cta_size;
   assert( shader->m_cta_status[cta_num] > 0 );
   shader->m_cta_status[cta_num]--;
   if (!shader->m_cta_status[cta_num]) {
      shader->m_n_active_cta--;
      shader->deallocate_barrier(cta_num);
      shader_CTA_count_unlog(shader->m_sid, 1);
      printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld)\n", shader->m_sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle );
   }
}

void obtain_insn_latency_info(insn_latency_info *latinfo, const inst_t *insn)
{
   latinfo->pc = insn->pc;
   latinfo->latency = gpu_tot_sim_cycle + gpu_sim_cycle - insn->issue_cycle;
}

int debug_tid = 0;

void shader_core_ctx::writeback() 
{
   std::vector<inst_t> done_insts;
   std::vector<insn_latency_info> unlock_lat_infos;
   bool w2rf = false;
   memset(m_pl_tid,-1, sizeof(int)*m_config->warp_size);
   check_stage_pcs(MM_WB);

   // detect if a valid instruction is in MM_WB
   for (unsigned i=0; i<m_config->warp_size; i++) {
	  w2rf |= (m_pipeline_reg[MM_WB][i].hw_thread_id >= 0);
	  m_pl_tid[i] = m_pipeline_reg[MM_WB][i].hw_thread_id;
   }

   //check mshrs for commit;
   unsigned mshr_threads_unlocked = 0;
   bool stalled_by_MSHR = false;
   
    mshr_entry *mshr_head = m_mshr_unit->return_head();
    if (mshr_head && (mshr_threads_unlocked + mshr_head->num_inst() <= m_config->warp_size) ) {
        assert (mshr_head->num_inst());
        for (unsigned j = 0; j < mshr_head->num_inst(); j++) {
            const inst_t &insn = mshr_head->get_inst(j);
            time_vector_update(insn.uid,MR_WRITEBACK,gpu_sim_cycle+gpu_tot_sim_cycle,RD_REQ);
            obtain_insn_latency_info(&m_mshr_lat_info[mshr_threads_unlocked], &insn);
            if (enable_ptx_file_line_stats)
                ptx_file_line_stats_sub_inflight_memory_insn(m_sid, insn.pc);
            assert (insn.hw_thread_id >= 0);
            m_pending_mem_access--;
            mshr_threads_unlocked++;
            if (m_config->gpgpu_operand_collector) {
                if ( j== 0 )
                    m_operand_collector.writeback(insn);
            } else
                stalled_by_MSHR = true;
        }
        mshr_head->get_insts(done_insts);

        m_mshr_unit->pop_return_head();
        unlock_lat_infos.resize(mshr_threads_unlocked);
        std::copy(m_mshr_lat_info, m_mshr_lat_info + mshr_threads_unlocked, unlock_lat_infos.begin());
        assert(mshr_threads_unlocked);
    }

   if ( m_config->gpgpu_operand_collector  ) 
       stalled_by_MSHR = !m_operand_collector.writeback( m_pipeline_reg[MM_WB] );

   if (!stalled_by_MSHR) {
        inst_t inst;
        for (unsigned i=0; i<m_config->warp_size; i++) {
            op_type op;
            if (m_pipeline_reg[MM_WB][i].hw_thread_id > -1)
                op = m_pipeline_reg[MM_WB][i].op;
            obtain_insn_latency_info(&m_pl_lat_info[i], &m_pipeline_reg[MM_WB][i]);
            if (!m_pipeline_reg[MM_WB][i].cache_miss) { // Do not include cache misses for a writeback
                if (m_pipeline_reg[MM_WB][i].hw_thread_id > -1) {
                    done_insts.push_back(m_pipeline_reg[MM_WB][i]);
                    unlock_lat_infos.push_back(m_pl_lat_info[i]);
                }
            }
            if (m_pl_tid[i] > -1 ) 
                inst = m_pipeline_reg[MM_WB][i];
        }

        // Unlock the warp for re-fetching (put it in the fixed delay queue)
        if (w2rf) // Only need to unlock if this is a valid instruction
            queue_warp_unlocking(m_pl_tid, inst);
    } else
        m_stats->gpu_stall_by_MSHRwb++;

   for (unsigned i=0; i<done_insts.size(); i++) {
	   inst_t done_inst = done_insts[i];
       call_thread_done(done_inst);
            
		gpu_sim_insn++; // a (scalar) instruction is done
		if ( !is_const(done_inst.space) )
			m_stats->gpu_sim_insn_no_ld_const++;
		m_gpu->gpu_sim_insn_last_update = gpu_sim_cycle;
        m_gpu->gpu_sim_insn_last_update_sid = m_sid;
		m_num_sim_insn++;
		m_thread[done_inst.hw_thread_id].n_insn++;
		m_thread[done_inst.hw_thread_id].n_insn_ac++;

		if (enable_ptx_file_line_stats) {
		  unsigned pc = unlock_lat_infos[i].pc;
		  unsigned long latency = unlock_lat_infos[i].latency;
		  ptx_file_line_stats_add_latency(pc, latency);
		}
  }

  if (!stalled_by_MSHR)  {
     if (!strcmp("GT200",m_config->pipeline_model) ) {
      inst_t *fvt=first_valid_thread(m_pipeline_reg[MM_WB]);
      if( fvt ) {
          unsigned warp_id = fvt->hw_thread_id/m_config->warp_size;
             m_warp[warp_id].dec_inst_in_pipeline();
      }
     }
     move_warp(m_pipeline_reg[WB_RT], m_pipeline_reg[MM_WB]);
  }

  process_delay_queue();
}

/*
 * Queues a warp into fixed delay queue for unlocking
 *
 * The amount of delay to add is determined by the instruction type.
 *
 * @param *tid Array of tid in the warp to unlock
 * @param space Address space for the current instruction in the warp
 *
 */
void shader_core_ctx::queue_warp_unlocking(int *tids, const inst_t &inst ) 
{
	// Create a delay queue object and add it to the queue
    fixeddelay_queue_warp_t fixeddelay_queue_warp;

	// Set ready_cycle based on instruction space
    fixeddelay_queue_warp.inst = inst;
	switch(inst.space.get_type()) {
		case shared_space:
			fixeddelay_queue_warp.ready_cycle = gpu_tot_sim_cycle + gpu_sim_cycle + 5; // Adds 5*4=20 cycles
			break;
		default:
			fixeddelay_queue_warp.ready_cycle = gpu_tot_sim_cycle + gpu_sim_cycle;
			break;
	}

	// Store threads in delay queue warp object
	fixeddelay_queue_warp.tids.resize(m_config->warp_size);
	std::copy(tids, tids+m_config->warp_size, fixeddelay_queue_warp.tids.begin());
	m_fixeddelay_queue.insert(fixeddelay_queue_warp);
}

/*
 * Process a delay queue by unlocking warps ready this cycle
 *
 * @param *shader Pointer to shader core
 *
 */
void shader_core_ctx::process_delay_queue() {
   shader_core_ctx *shader=this;
	// Unlock warps in fixeddelay_queue_warp
	std::multiset<fixeddelay_queue_warp_t, fixeddelay_queue_warp_comp>::iterator it;
	std::multiset<fixeddelay_queue_warp_t, fixeddelay_queue_warp_comp>::iterator it_last;
	for ( it=shader->m_fixeddelay_queue.begin() ;
		  it != shader->m_fixeddelay_queue.end();
		) {
	   if(it->ready_cycle <= gpu_tot_sim_cycle + gpu_sim_cycle) {
		   if(!m_config->gpgpu_stall_on_use) {
			   // This disables stall-on-use
			   // If thread is still in warp_tracker, do not unlock yet
			   bool skip_unlock = false;
			   for(unsigned i=0; i<m_config->warp_size; i++) {
				   int tid = it->tids[i];
				   if(tid < 0) continue;
				   if(m_warp_tracker->wpt_thread_in_wpt(tid)) {
					   skip_unlock = true;
					   break;
				   }
			   }
			   if(skip_unlock) {
				   it_last = it++;
				   continue;
			   }
		   }

         if (!strcmp("GT200",m_config->pipeline_model) ) {
            if( it->inst.space == shared_space ) {
                for(unsigned i=0; i < m_config->warp_size; i++ ) {
                    if( it->tids[i]>= 0 ) {
                        unsigned warp_id = it->tids[i]/m_config->warp_size;
                        m_scoreboard->releaseRegisters(warp_id,&it->inst);
                        break;
                    }
                }
            }
         }
            
   		// Unlock warp
   		unlock_warp(it->tids);

		   // Remove warp information from delay queue
		   it_last = it++;
		   shader->m_fixeddelay_queue.erase(it_last);
	   } else {
		   break;
	   }
	}
}

/*
 * Unlock a warp
 *
 * @param tids Vector of tid in the warp to unlock
 *
 */
void shader_core_ctx::unlock_warp( std::vector<int> tids ) 
{
   assert( tids.size() == m_config->warp_size ); // required by thd_commit_queue usage in fetch_simd_dwf()
   int thd_unlocked = 0;
   int thd_exited = 0;
   int tid;
   int valid_tid = -1;

   if (!strcmp("GPGPUSIM_ORIG",m_config->pipeline_model) ) {
   // Unlock
   for (unsigned i=0; i<m_config->warp_size; i++) {
      tid = tids[i];
	   if (tid >= 0) {
   	  	valid_tid = tid;
   	  	// thread completed if it is going to fetching beyond code boundary
   	  	if ( ptx_thread_done(tid) ) {
   	  		m_not_completed -= 1;
   	  		m_stats->gpu_completed_thread += 1;
   	  		int warp_id = wid_from_hw_tid(tid,m_config->warp_size);
   	  		if (!(m_warp[warp_id].get_n_completed() < m_config->warp_size)) 
                printf("GPGPU-Sim uArch: shader[%d]->warp[%d].n_completed = %d; warp_size = %d\n",
   	  				m_sid,warp_id, m_warp[warp_id].get_n_completed(), m_config->warp_size);
   	  		assert( m_warp[warp_id].get_n_completed() < m_config->warp_size );
   	  		m_warp[warp_id].inc_n_completed();
   	  		register_cta_thread_exit( tid );
   	  		thd_exited = 1;
   	  	} else {
            if (!strcmp("GPGPUSIM_ORIG",m_config->pipeline_model) ) {
                assert(!m_thread[tid].m_avail4fetch);
                m_thread[tid].m_avail4fetch=true;
                assert( m_warp[tid/m_config->warp_size].get_avail4fetch() < m_config->warp_size );
                m_warp[tid/m_config->warp_size].inc_avail4fetch();
            }
            thd_unlocked = 1;
        }
      }
   }
   }

   if (!strcmp("GPGPUSIM_ORIG",m_config->pipeline_model) ) {
       if(thd_unlocked || thd_exited) {
          // Update the warp active mask
           m_pdom_warp[wid_from_hw_tid(valid_tid,m_config->warp_size)]->pdom_update_warp_mask();
       }
   }

	if (m_config->model == POST_DOMINATOR) {
		// Do nothing
	} else {
		// For this case, submit to commit_queue
		if (m_config->using_commit_queue && thd_unlocked) 
			m_thd_commit_queue->push( new std::vector<int>(tids), gpu_sim_cycle );
	}
}


/*
 * Signals to the warp_tracker that a thread in a warp (for a given pc/instruction) is done
 *
 * @param *shd Pointer to shader core
 * @param done_inst Completed instruction
 *
 */
void shader_core_ctx::call_thread_done( inst_t &done_inst ) 
{
	if (m_config->gpgpu_no_divg_load) {
		// Signal to unlock the thread. If all threads are done, deregister warp
		if( m_warp_tracker->wpt_signal_avail(done_inst.hw_thread_id, done_inst.pc) == 1 ) {
			// Entire warp has returned
			// Deregister warp
			m_warp_tracker->wpt_deregister_warp(done_inst.hw_thread_id, done_inst.pc);

         if (! (!strcmp("GT200",m_config->pipeline_model) && (done_inst.space == shared_space)) ) 
            // Signal scoreboard to release register
            m_scoreboard->releaseRegisters( wid_from_hw_tid(done_inst.hw_thread_id, m_config->warp_size), &done_inst );
		}
	}
}


void gpgpu_sim::shader_print_runtime_stat( FILE *fout ) 
{
   fprintf(fout, "SHD_INSN: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_THDS: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_not_completed());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_DIVG: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
   fprintf(fout, "\n");

   fprintf(fout, "THD_INSN: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
   fprintf(fout, "\n");
}


void gpgpu_sim::shader_print_l1_miss_stat( FILE *fout ) 
{
   fprintf(fout, "THD_INSN_AC: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Acc: "); //l1 access per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_access_ac(i));
   fprintf(fout, "\n");

   //per warp
   int temp =0; 
   fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp=0;
   fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i) );
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp =0;
   fprintf(fout, "W_L1_Acc: "); //l1 access per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_access_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
}

void shader_core_ctx::print_warp( inst_t *warp, FILE *fout, int print_mem, int mask ) const
{
   unsigned i, j, warp_id = (unsigned)-1;
   for (i=0; i<m_config->warp_size; i++) {
      if (warp[i].hw_thread_id > -1) {
         warp_id = warp[i].hw_thread_id / m_config->warp_size;
         break;
      }
   }
   i = (i>=m_config->warp_size)? 0 : i;

   if( warp[i].pc != (address_type)-1 ) 
      fprintf(fout,"0x%04x ", warp[i].pc );
   else
      fprintf(fout,"bubble " );

   if( mask & 2 ) {
      fprintf(fout, "(" );
      for (j=0; j<m_config->warp_size; j++)
         fprintf(fout, "%03d ", warp[j].hw_thread_id);
      fprintf(fout, "): ");
   } else {
      fprintf(fout, "w%02d[", warp_id);
      for (j=0; j<m_config->warp_size; j++) 
         fprintf(fout, "%c", ((warp[j].hw_thread_id != -1)?'1':'0') );
      fprintf(fout, "]: ");
   }

   if( warp_id != (unsigned)-1 && m_config->model == POST_DOMINATOR ) {
       unsigned rp = m_pdom_warp[warp_id]->get_rp();
       if( rp == (unsigned)-1 ) {
          fprintf(fout," rp:--- ");
       } else {
          fprintf(fout," rp:0x%03x ", rp );
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

void shader_core_ctx::print_stage(unsigned int stage, FILE *fout, int print_mem, int mask ) 
{
   inst_t *warp = m_pipeline_reg[stage];
   print_warp(warp,fout,print_mem,mask);
}

void shader_core_ctx::print_pre_mem_stages( FILE *fout, int print_mem, int mask ) 
{
   unsigned i, j;
   int warp_id;

   if (!m_config->gpgpu_pre_mem_stages) return;

   for (unsigned pms = 0; pms <= m_config->gpgpu_pre_mem_stages; pms++) {
      fprintf(fout, "PM[%01d]       = ", pms);

      warp_id = -1;

      for (i=0; i<m_config->warp_size; i++) {
         if (pre_mem_pipeline[pms][i].hw_thread_id > -1) {
            warp_id = pre_mem_pipeline[pms][i].hw_thread_id / m_config->warp_size;
            break;
         }
      }
      i = (i>=m_config->warp_size)? 0 : i;

      if( pre_mem_pipeline[pms][i].pc != (address_type)-1 ) 
         fprintf(fout,"0x%04x ", pre_mem_pipeline[pms][i].pc );
      else
         fprintf(fout,"bubble " );

      if( mask & 2 ) {
         fprintf(fout, "(" );
         for (j=0; j<m_config->warp_size; j++)
            fprintf(fout, "%03d ", pre_mem_pipeline[pms][j].hw_thread_id);
         fprintf(fout, "): ");
      } else {
         fprintf(fout, "w%02d[", warp_id);
         for (j=0; j<m_config->warp_size; j++)
            fprintf(fout, "%c", ((pre_mem_pipeline[pms][j].hw_thread_id != -1)?'1':'0') );
         fprintf(fout, "]: ");
      }

      if( warp_id != -1 && m_config->model == POST_DOMINATOR ) {
          unsigned rp = m_pdom_warp[warp_id]->get_rp();
          if( rp == (unsigned)-1 ) {
             fprintf(fout," rp:--- ");
          } else {
             fprintf(fout," rp:0x%03x ", rp );
          }
      }

      ptx_print_insn( pre_mem_pipeline[pms][i].pc, fout );

      if( mask & 0x10 ) {
         if ( ( pre_mem_pipeline[pms][i].op == LOAD_OP ||
                pre_mem_pipeline[pms][i].op == STORE_OP ) && print_mem )
            fprintf(fout, "  mem: 0x%016llx", pre_mem_pipeline[pms][i].memreqaddr);
      }
      fprintf(fout, "\n");
   }
}

const char * ptx_get_fname( unsigned PC );

void shader_core_ctx::display_pdom_state(FILE *fout, int mask )
{
    if ( (mask & 4) && m_config->model == POST_DOMINATOR ) {
       fprintf(fout,"warp status:\n");
       unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
       for (unsigned i=0; i < n; i++) {
          unsigned nactive = 0;
          for (unsigned j=0; j<m_config->warp_size; j++ ) {
             unsigned tid = i*m_config->warp_size + j;
             int done = ptx_thread_done(tid);
             nactive += (ptx_thread_done(tid)?0:1);
             if ( done && (mask & 8) ) {
                unsigned done_cycle = ptx_thread_donecycle( m_thread[tid].m_functional_model_thread_state );
                if ( done_cycle ) {
                   printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle );
                }
             }
          }
          if ( nactive == 0 ) {
             continue;
          }
          m_pdom_warp[i]->print(fout);
       }
       fprintf(fout,"\n");
    }
}

void shader_core_ctx::display_pipeline(FILE *fout, int print_mem, int mask ) 
{
   fprintf(fout, "=================================================\n");
   fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid, 
           gpu_tot_sim_cycle, gpu_sim_cycle, m_not_completed);
   fprintf(fout, "=================================================\n");

   if (!strcmp("GPGPUSIM_ORIG",m_config->pipeline_model) ) 
       display_pdom_state(fout,mask);

   if (!strcmp("GT200",m_config->pipeline_model) ) {
       dump_istream_state(fout);
       fprintf(fout,"\n");

       fprintf(fout, "IF/ID       = ");
       if( !m_inst_fetch_buffer.m_valid )
           fprintf(fout,"bubble\n");
       else {
           fprintf(fout,"w%2u : pc = 0x%x, nbytes = %u\n", 
                   m_inst_fetch_buffer.m_warp_id,
                   m_inst_fetch_buffer.m_pc, 
                   m_inst_fetch_buffer.m_nbytes );
       }
       fprintf(fout,"\nibuffer status:\n");
       for( unsigned i=0; i<m_config->max_warps_per_shader; i++) {
           if( !m_warp[i].ibuffer_empty() ) 
               m_warp[i].print_ibuffer(fout);
       }
       fprintf(fout,"\n");
       display_pdom_state(fout,mask);
   }

   m_scoreboard->printContents();

   if (!strcmp("GPGPUSIM_ORIG",m_config->pipeline_model) ) {
       if ( mask & 0x20 ) {
          fprintf(fout, "TS/IF       = ");
          print_stage(TS_IF, fout, print_mem, mask);
       }
       fprintf(fout, "IF/ID       = ");
       print_stage(IF_ID, fout, print_mem, mask );
   }
   if (m_config->gpgpu_operand_collector) {
      fprintf(fout,"ID/OC (SP)  = ");
      print_stage(ID_OC, fout, print_mem, mask);
      fprintf(fout,"ID/OC (SFU) = ");
      print_stage(ID_OC_SFU, fout, print_mem, mask);
      m_operand_collector.dump(fout);
   }
   if (m_config->m_using_dwf_rrstage) {
      fprintf(fout, "ID/RR       = ");
      print_stage(ID_RR, fout, print_mem, mask);
   }
   if (!strcmp("GT200",m_config->pipeline_model) ) 
      fprintf(fout, "ID/EX (SP)  = ");
   else
      fprintf(fout, "ID/EX       = ");
   print_stage(ID_EX, fout, print_mem, mask);
   if (!strcmp("GT200",m_config->pipeline_model) ) {
      fprintf(fout, "ID/EX (SFU) = ");
      print_stage(OC_EX_SFU, fout, print_mem, mask);
   }
   print_pre_mem_stages(fout, print_mem, mask);
   if (!m_config->gpgpu_pre_mem_stages)
      fprintf(fout, "EX/MEM      = ");
   else
      fprintf(fout, "PM/MEM      = ");
   print_stage(EX_MM, fout, print_mem, mask);
   fprintf(fout, "MEM/WB      = ");
   print_stage(MM_WB, fout, print_mem, mask);
   fprintf(fout, "\n");
   mshr_print(fout,0);
}

unsigned int shader_core_ctx::max_cta( class function_info *kernel )
{
   unsigned int padded_cta_size;

   padded_cta_size = m_gpu->the_kernel().threads_per_cta();
   if (padded_cta_size%m_config->warp_size) 
      padded_cta_size = ((padded_cta_size/m_config->warp_size)+1)*(m_config->warp_size);

   //Limit by n_threads/shader
   unsigned int result_thread = m_config->n_thread_per_shader / padded_cta_size;

   const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info(kernel);

   //Limit by shmem/shader
   unsigned int result_shmem = (unsigned)-1;
   if (kernel_info->smem > 0)
      result_shmem = m_config->gpgpu_shmem_size / kernel_info->smem;

   //Limit by register count, rounded up to multiple of 4.
   unsigned int result_regs = (unsigned)-1;
   if (kernel_info->regs > 0)
      result_regs = m_config->gpgpu_shader_registers / (padded_cta_size * ((kernel_info->regs+3)&~3));

   //Limit by CTA
   unsigned int result_cta = m_config->max_cta_per_core;

   unsigned result = result_thread;
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
      printf ("Error: max_cta_per_shader(\"%s\") returning %d. Kernel requires more resources than shader has?\n", m_name, result);
      abort();
   }
   return result;
}

void shader_core_ctx::cycle_gt200()
{
    clear_stage(m_pipeline_reg[WB_RT]);
    writeback();
    memory();
    execute();
    m_operand_collector.step(m_pipeline_reg[ID_OC],m_pipeline_reg[ID_OC_SFU]);
    decode_new();
    fetch_new();
}

void shader_core_ctx::cycle()
{
   clear_stage(m_pipeline_reg[WB_RT]);
   writeback();
   memory();
   if (m_config->gpgpu_pre_mem_stages) // for modeling deeper pipelines
      pre_memory();
   execute();
   if (m_config->m_using_dwf_rrstage) {
      preexecute();
   }
   if (m_config->gpgpu_operand_collector) 
      m_operand_collector.step(m_pipeline_reg[ID_OC]);
   decode();
   fetch();
}

// Flushes all content of the cache to memory

void shader_core_ctx::cache_flush()
{
   m_L1D->flush();
   // TODO: add flush 'interface' object to provide functionality commented out below
/* 
 
   unsigned int i;
   unsigned int set;
   unsigned long long int flush_addr;
   cache_t *cp = m_L1D;
   cache_block_t *pline;
   for (i=0; i<cp->m_nset*cp->m_assoc; i++) {
      pline = &(cp->m_lines[i]);
      set = i / cp->m_assoc;
      if ((pline->status & (DIRTY|VALID)) == (DIRTY|VALID)) {
         flush_addr = pline->addr;
         fq_push(flush_addr, m_L1D->get_line_sz(), 1, NO_PARTIAL_WRITE, 0, NULL, 0, GLOBAL_ACC_W, -1);
         pline->status &= ~VALID;
         pline->status &= ~DIRTY;
      } else if (pline->status & VALID) {
         pline->status &= ~VALID;
      }
   }
*/
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
      for( unsigned j=0; j<m_num_collectors; j++) {
         assert( i < (unsigned)_inputs );
         assert( j < (unsigned)_outputs );
         _request[i][j] = 0;
      }
      if( !m_queue[i].empty() ) {
         const op_t &op = m_queue[i].front();
         int oc_id = op.get_oc_id();
         assert( i < (unsigned)_inputs );
         assert( oc_id < _outputs );
         _request[i][oc_id] = 1;
      }
      if( m_allocated_bank[i].is_write() ) {
         assert( i < (unsigned)_inputs );
         _inmatch[i] = 0; // write gets priority
      }
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
            assert( input < _inputs );
            _inmatch[input] = output;
            assert( output < _outputs );
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
bool barrier_set_t::warp_waiting_at_barrier( unsigned warp_id ) const
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

bool shader_core_ctx::warp_waiting_at_barrier( unsigned warp_id ) const
{
   return m_barriers.warp_waiting_at_barrier(warp_id);
}

void shader_core_ctx::set_at_memory_barrier( unsigned warp_id ) 
{
   m_warp[warp_id].set_membar();
}

bool shader_core_ctx::warp_waiting_at_mem_barrier( unsigned warp_id ) 
{
   if( !m_warp[warp_id].get_membar() ) 
      return false;
   if( !m_scoreboard->pendingWrites(warp_id) ) {
      m_warp[warp_id].clear_membar();
      return false;
   }
   return true;
}

bool shader_core_ctx::warp_waiting_for_atomics( unsigned warp_id ) const
{
   return m_warp[warp_id].get_n_atomic()>0;
}

gpgpu_sim *shader_core_ctx::get_gpu()
{
   return m_gpu;
}

void shader_core_ctx::allocate_barrier( unsigned cta_id, warp_set_t warps )
{
   m_barriers.allocate_barrier(cta_id,warps);
}

void shader_core_ctx::deallocate_barrier( unsigned cta_id )
{
   m_barriers.deallocate_barrier(cta_id);
}

void shader_core_ctx::decrement_atomic_count( unsigned wid )
{
   assert( m_warp[wid].get_n_atomic() > 0 );
   m_warp[wid].dec_n_atomic();
}

bool shd_warp_t::done()
{
    return get_n_completed() == m_warp_size;
}

bool shd_warp_t::waiting() 
{
    if ( done() ) {
        // waiting to be initialized with a kernel
        return true;
    } else if ( m_shader->warp_waiting_at_barrier(m_warp_id) ) {
        // waiting for other warps in CTA to reach barrier
        return true;
    } else if ( m_shader->warp_waiting_at_mem_barrier(m_warp_id) ) {
        // waiting for memory barrier
        return true;
    } else if ( m_shader->warp_waiting_for_atomics(m_warp_id) ) {
        // waiting for atomic operation to complete at memory:
        // this stall is not required for accurate timing model, but rather we
        // stall here since if a call/return instruction occurs in the meantime
        // the functional execution of the atomic when it hits DRAM can cause
        // the wrong register to be read.
        return true;
    }
    return false;
}

void shd_warp_t::print( FILE *fout ) const
{
    if ( n_completed < m_warp_size ) {
        fprintf( fout, "w%02u npc: 0x%04x, done:%2u a4f:%2u, i:%u s:%u a:%u b:%2u, (done: ", 
                m_warp_id,
                m_next_pc,
                n_completed,
                n_avail4fetch,
                m_inst_in_pipeline, 
                m_stores_outstanding,
                m_n_atomic,
                n_waiting_at_barrier );
        for (unsigned i = m_warp_id*m_warp_size; i < (m_warp_id+1)*m_warp_size; i++ ) {
          if ( m_shader->ptx_thread_done(i) ) fprintf(fout,"1");
          else fprintf(fout,"0");
          if ( (((i+1)%4) == 0) && (i+1) < (m_warp_id+1)*m_warp_size ) 
             fprintf(fout,",");
        }
        fprintf(fout,") ");
        fprintf(fout," last fetched @ %5llu", m_last_fetch);
        if( m_imiss_pending ) 
            fprintf(fout," i-miss pending");
        fprintf(fout,"\n");
    }
}

void shd_warp_t::print_ibuffer( FILE *fout ) const
{
    fprintf(fout,"  ibuffer[%2u] : ", m_warp_id );
    for( unsigned i=0; i < IBUFFER_SIZE; i++) {
        const inst_t *inst = m_ibuffer[i];
        if( inst ) inst->print_insn(fout);
        else fprintf(fout," <empty> ");
    }
    fprintf(fout,"\n");
}

pdom_warp_ctx_t::pdom_warp_ctx_t( unsigned wid, class shader_core_ctx *shdr )
{
    m_warp_id=wid;
    m_shader=shdr;
    m_warp_size=m_shader->get_config()->warp_size;
    m_stack_top = 0;
    m_pc = (address_type*)calloc(m_warp_size * 2, sizeof(address_type));
    m_calldepth = (unsigned int*)calloc(m_warp_size * 2, sizeof(unsigned int));
    m_active_mask = (unsigned int*)calloc(m_warp_size * 2, sizeof(unsigned int));
    m_recvg_pc = (address_type*)calloc(m_warp_size * 2, sizeof(address_type));
    m_branch_div_cycle = (unsigned long long *)calloc(m_warp_size * 2, sizeof(unsigned long long ));
    reset();
}

void pdom_warp_ctx_t::reset()
{
    m_stack_top = 0;
    memset(m_pc, -1, m_warp_size * 2 * sizeof(address_type));
    memset(m_calldepth, 0, m_warp_size * 2 * sizeof(unsigned int));
    memset(m_active_mask, 0, m_warp_size * 2 * sizeof(unsigned int));
    memset(m_recvg_pc, -1, m_warp_size * 2 * sizeof(address_type));
    memset(m_branch_div_cycle, 0, m_warp_size * 2 * sizeof(unsigned long long ));
}

void pdom_warp_ctx_t::launch( address_type start_pc, unsigned active_mask )
{
    reset();
    m_pc[0] = start_pc;
    m_calldepth[0] = 1;
    m_active_mask[0] = active_mask;
}

unsigned pdom_warp_ctx_t::get_active_mask() const
{
    return m_active_mask[m_stack_top];
}

void mshr_entry::init( new_addr_type address, bool wr, memory_space_t space, unsigned warp_id )
{
   static unsigned next_request_uid = 1;
   m_request_uid = next_request_uid++;
   m_status = INITIALIZED;
   m_addr = address;
   m_mf = NULL;
   m_merged_on_other_reqest = false;
   m_merged_requests =NULL;
   m_iswrite = wr;
   m_isinst = space==instruction_space;
   m_islocal = is_local(space);
   m_isconst = is_const(space);
   m_istexture = space==tex_space;
   m_insts.clear();
   m_warp_id = warp_id;
}

void mshr_entry::set_status( enum mshr_status status ) 
{ 
   mshr_entry * req = this;
   while (req) {
      req->m_status = status;
      req = req->m_merged_requests;
   }
#if DEBUGL1MISS 
#define CACHE_TAG_OF_64(x) ((x) & (~((unsigned long long int)64 - 1)))
   printf("cycle %d Addr %x  %d \n",gpu_sim_cycle,CACHE_TAG_OF_64(m_addr),status);
#endif
}

void mshr_entry::print(FILE *fp, unsigned mask) const
{
    if ( mask & 0x100 ) {
        fprintf(fp, "MSHR(%u): w%2u req uid=%5u, %s (0x%llx) merged:%d status:%s ", 
                m_id,
                m_warp_id,
                m_request_uid,
                (m_iswrite)? "store" : "load ",
                m_addr, 
                (m_merged_requests != NULL || m_merged_on_other_reqest), 
                MSHR_Status_str[m_status]);
        if ( m_mf )
            ptx_print_insn( m_mf->get_pc(), fp );
        fprintf(fp,"\n");
        if ( mask & 0x200 ) {
            for (unsigned i = 0; i < m_insts.size(); i++) {
                fprintf(fp,"\tthread: UID:%d HW:%d ReqAddr:0x%llx\n", 
                        m_insts[i].uid, m_insts[i].hw_thread_id, m_insts[i].memreqaddr);
            }
        }
    }
}

void opndcoll_rfu_t::init( unsigned num_collectors_alu, 
                           unsigned num_collectors_sfu, 
                           unsigned num_banks, 
                           shader_core_ctx *shader,
                           inst_t **alu_port,
                           inst_t **sfu_port )
{
   m_num_collectors = num_collectors_alu+num_collectors_sfu;
    
   m_shader=shader;
   m_arbiter.init(m_num_collectors,num_banks);

   m_alu_port = alu_port;
   m_sfu_port = sfu_port;

   m_dispatch_units[ m_alu_port ].init( num_collectors_alu );
   m_dispatch_units[ m_sfu_port ].init( num_collectors_sfu );

   m_num_banks = num_banks;

   m_bank_warp_shift = 0; 
   m_warp_size = shader->get_config()->warp_size;
   m_bank_warp_shift = (unsigned)(int) (log(m_warp_size+0.5) / log(2.0));
   assert( (m_bank_warp_shift == 5) || (m_warp_size != 32) );

   m_cu = new collector_unit_t[m_num_collectors];

   unsigned c=0;
   for(; c<num_collectors_alu; c++) {
      m_cu[c].init(c,m_alu_port,num_banks,m_bank_warp_shift,m_warp_size,this);
      m_free_cu[m_alu_port].push_back(&m_cu[c]);
      m_dispatch_units[m_alu_port].add_cu(&m_cu[c]);
   }
   for(; c<m_num_collectors; c++) {
      m_cu[c].init(c,m_sfu_port,num_banks,m_bank_warp_shift,m_warp_size,this);
      m_free_cu[m_sfu_port].push_back(&m_cu[c]);
      m_dispatch_units[m_sfu_port].add_cu(&m_cu[c]);
   }
}

bool opndcoll_rfu_t::writeback( inst_t *warp )
{
   // prefer not to stall writeback
   inst_t *fvt=m_shader->first_valid_thread(warp);
   if (!fvt) 
       return true; // nothing to do
   return writeback(*fvt);
}

int register_bank(int regnum, int tid, unsigned num_banks, unsigned bank_warp_shift)
{
   int bank = regnum;
   if (bank_warp_shift)
      bank += tid >> bank_warp_shift;
   return bank % num_banks;
}

bool opndcoll_rfu_t::writeback( const inst_t &fvt )
{
   int tid = fvt.hw_thread_id;
   assert( tid >= 0 ); // must be a valid instruction
   std::list<unsigned> regs = m_shader->get_regs_written(fvt);
   std::list<unsigned>::iterator r;
   unsigned last_reg = -1;
   unsigned n=0;
   for( r=regs.begin(); r!=regs.end();r++,n++ ) {
      unsigned reg = *r;
      unsigned bank = register_bank(reg,tid,m_num_banks,m_bank_warp_shift);
      if( m_arbiter.bank_idle(bank) ) {
          m_arbiter.allocate_bank_for_write(bank,op_t(&fvt,reg,m_num_banks,m_bank_warp_shift));
      } else {
          return false;
      }
      last_reg=reg;
   }
   return true;
}

void opndcoll_rfu_t::dispatch_ready_cu()
{
   port_to_du_t::iterator p;
   for( p=m_dispatch_units.begin(); p!=m_dispatch_units.end(); ++p ) {
      inst_t **port = p->first;
      if( !m_shader->pipeline_regster_empty(*port) ) 
         continue;
      dispatch_unit_t &du = p->second;
      collector_unit_t *cu = du.find_ready();
      if( cu ) {
         cu->dispatch();
         m_free_cu[port].push_back(cu);
      }
   }
}

void opndcoll_rfu_t::allocate_cu( inst_t *&id_oc_reg )
{
   inst_t *fvi = m_shader->first_valid_thread(id_oc_reg);
   if( fvi ) {
      inst_t **port = NULL;
      if( fvi->op == SFU_OP ) 
         port = m_sfu_port;
      else
         port = m_alu_port;
      if( !m_free_cu[port].empty() ) {
         collector_unit_t *cu = m_free_cu[port].back();
         m_free_cu[port].pop_back();
         cu->allocate(id_oc_reg);
         m_arbiter.add_read_requests(cu);
      }
   }
}

void opndcoll_rfu_t::allocate_reads()
{
   // process read requests that do not have conflicts
   std::list<op_t> allocated = m_arbiter.allocate_reads();
   std::map<unsigned,op_t> read_ops;
   for( std::list<op_t>::iterator r=allocated.begin(); r!=allocated.end(); r++ ) {
      const op_t &rr = *r;
      unsigned reg = rr.get_reg();
      unsigned tid = rr.get_tid();
      unsigned bank = register_bank(reg,tid,m_num_banks,m_bank_warp_shift);
      m_arbiter.allocate_for_read(bank,rr);
      read_ops[bank] = rr;
   }
   std::map<unsigned,op_t>::iterator r;
   for(r=read_ops.begin();r!=read_ops.end();++r ) {
      op_t &op = r->second;
      unsigned cu = op.get_oc_id();
      unsigned operand = op.get_operand();
      assert( cu < m_num_collectors );
      m_cu[cu].collect_operand(operand);
   }
} 


void gpgpu_sim::decrement_atomic_count( unsigned sid, unsigned wid )
{
   m_sc[sid]->decrement_atomic_count(wid);
}


bool opndcoll_rfu_t::collector_unit_t::ready() const 
{ 
   return (!m_free) && m_not_ready.none() && m_rfu->shader_core()->pipeline_regster_empty(*m_port); 
}

void opndcoll_rfu_t::collector_unit_t::dump(FILE *fp, const shader_core_ctx *shader ) const
{
   if( m_free ) {
      fprintf(fp,"    <free>\n");
   } else {
      shader->print_warp(m_warp,fp,0,0);
      for( unsigned i=0; i < MAX_REG_OPERANDS; i++ ) {
         if( m_not_ready.test(i) ) {
            std::string r = m_src_op[i].get_reg_string();
            fprintf(fp,"    '%s' not ready\n", r.c_str() );
         }
      }
   }
}

void opndcoll_rfu_t::collector_unit_t::init( unsigned n, 
                                             inst_t **port, 
                                             unsigned num_banks, 
                                             unsigned log2_warp_size,
                                             unsigned warp_size,
                                             opndcoll_rfu_t *rfu ) 
{ 
   m_rfu=rfu;
   m_cuid=n; 
   m_port=port; 
   m_num_banks=num_banks;
   assert(m_warp==NULL); 
   m_warp = (inst_t*)calloc(sizeof(inst_t),warp_size);
   m_rfu->shader_core()->clear_stage(m_warp);
   m_bank_warp_shift=log2_warp_size;
}

void opndcoll_rfu_t::collector_unit_t::allocate( inst_t *&pipeline_reg ) 
{
   assert(m_free);
   assert(m_not_ready.none());
   m_free = false;
   inst_t *fvi = m_rfu->shader_core()->first_valid_thread(pipeline_reg);
   if( fvi ) {
      m_tid = fvi->hw_thread_id;
      m_warp_id = m_tid/m_rfu->shader_core()->get_config()->warp_size;
      for( unsigned op=0; op < 4; op++ ) {
         int reg_num = fvi->arch_reg[4+op]; // this math needs to match that used in function_info::ptx_decode_inst
         if( reg_num >= 0 ) { // valid register
            m_src_op[op] = op_t( this, op, reg_num, m_num_banks, m_bank_warp_shift );
            m_not_ready.set(op);
         } else 
            m_src_op[op] = op_t();
      }
      m_rfu->shader_core()->move_warp(m_warp,pipeline_reg);
   }
}

void opndcoll_rfu_t::collector_unit_t::dispatch()
{
   assert( m_not_ready.none() );
   m_rfu->shader_core()->move_warp(*m_port,m_warp);
   m_free=true;
   for( unsigned i=0; i<MAX_REG_OPERANDS;i++) 
      m_src_op[i].reset();
}

bool shader_core_ctx::ptx_thread_done( unsigned hw_thread_id ) const
{
    assert( hw_thread_id < m_config->n_thread_per_shader );
    ptx_thread_info *thd = m_thread[ hw_thread_id ].m_functional_model_thread_state;
    return (thd==NULL) || thd->is_done();
}

class ptx_thread_info *shader_core_ctx::get_thread_state( unsigned hw_thread_id )
{
    assert( hw_thread_id < m_config->n_thread_per_shader );
    return m_thread[ hw_thread_id ].m_functional_model_thread_state;
}
