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

unsigned shader_core_ctx::first_valid_thread( unsigned stage )
{
    abort();
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

   m_name = name;
   m_sid = shader_id;
   m_tpc = tpc_id;
   m_pipeline_reg = new warp_inst_t*[N_PIPELINE_STAGES];
   for (int j = 0; j<N_PIPELINE_STAGES; j++) 
      m_pipeline_reg[j] = new warp_inst_t(warp_size);

   m_thread = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), config->n_thread_per_shader);
   m_not_completed = 0;

   m_warp.resize(m_config->max_warps_per_shader, shd_warp_t(this, warp_size));

   m_n_active_cta = 0;
   for (unsigned i = 0; i<MAX_CTA_PER_SHADER; i++  ) 
      m_cta_status[i]=0;
   for (unsigned i = 0; i<config->n_thread_per_shader; i++) {
      m_thread[i].m_functional_model_thread_state = NULL;
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
}

void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed ) 
{
   if( reset_not_completed ) 
       m_not_completed = 0;
   for (unsigned i = start_thread; i<end_thread; i++) {
      m_thread[i].n_insn = 0;
      m_thread[i].m_cta_id = -1;
   }
   for (unsigned i = start_thread / m_config->warp_size; i < end_thread / m_config->warp_size; ++i) {
      m_warp[i].reset();
      m_pdom_warp[i]->reset();
   }
}

void shader_core_ctx::init_warps( unsigned cta_id, unsigned start_thread, unsigned end_thread )
{
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
            m_warp[i].init(start_pc,cta_id,i,n_active);
            m_not_completed += n_active;
      }
   }
}

// initalize the pipeline stage register to nops
void shader_core_ctx::clear_stage_reg(int stage)
{
   m_pipeline_reg[stage]->clear();
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
        const warp_inst_t* pI1 = ptx_fetch_inst(pc);
        assert(pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(0,pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
        const warp_inst_t* pI2 = ptx_fetch_inst(pc+pI1->isize);
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


void shader_core_ctx::func_exec_inst( warp_inst_t &inst )
{
    for ( unsigned t=0; t < m_config->warp_size; t++ ) {
        if( inst.active(t) ) {
            unsigned tid=m_config->warp_size*inst.warp_id()+t;
            m_thread[tid].m_functional_model_thread_state->ptx_exec_inst(inst,t);
            if( inst.has_callback(t) ) 
               m_warp[inst.warp_id()].inc_n_atomic();
            if (is_local(inst.space.get_type()) && (is_load(inst) || is_store(inst)))
               inst.set_addr(t, translate_local_memaddr(inst.get_addr(t), tid, m_gpu->num_shader()) );
            if ( ptx_thread_done(tid) ) {
                m_warp[inst.warp_id()].inc_n_completed();
                m_warp[inst.warp_id()].ibuffer_flush();
            }
        }
    }
}

void shader_core_ctx::issue_warp( warp_inst_t *&pipe_reg, const warp_inst_t *next_inst, unsigned active_mask, unsigned warp_id )
{
    m_warp[warp_id].ibuffer_free();
    assert(next_inst->valid());
    *pipe_reg = *next_inst; // static instruction information
    pipe_reg->issue( active_mask, warp_id, gpu_tot_sim_cycle + gpu_sim_cycle ); // dynamic instruction information
    func_exec_inst( *pipe_reg );
    if( next_inst->op == BARRIER_OP ) 
       set_at_barrier(m_warp[warp_id].get_cta_id(),warp_id);
    else if( next_inst->op == MEMORY_BARRIER_OP ) 
       set_at_memory_barrier(warp_id);
    m_pdom_warp[warp_id]->pdom_update_warp_mask();
    m_scoreboard->reserveRegisters(warp_id, next_inst);
    m_warp[warp_id].set_next_pc(next_inst->pc + next_inst->isize);
}

void shader_core_ctx::decode_new()
{
    for ( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
        unsigned warp_id = (m_last_warp_issued+1+i) % m_config->max_warps_per_shader;
        unsigned checked=0;
        unsigned issued=0;
        while( !m_warp[warp_id].waiting() && !m_warp[warp_id].ibuffer_empty() && (checked < 2) && (issued < 2) ) {
            unsigned active_mask = m_pdom_warp[warp_id]->get_active_mask();
            const warp_inst_t *pI = m_warp[warp_id].ibuffer_next();
            unsigned pc,rpc;
            m_pdom_warp[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
            if( pI ) {
                if( pc != pI->pc ) {
                    // control hazard
                    m_warp[warp_id].set_next_pc(pc);
                    m_warp[warp_id].ibuffer_flush();
                } else if ( !m_scoreboard->checkCollision(warp_id, pI) ) {
                    assert( m_warp[warp_id].inst_in_pipeline() );
                    if ( (pI->op != SFU_OP) && m_pipeline_reg[ID_OC]->empty() ) {
                        issue_warp(m_pipeline_reg[ID_OC],pI,active_mask,warp_id);
                        issued++;
                    } else if ( (pI->op == SFU_OP || pI->op == ALU_SFU_OP) && m_pipeline_reg[ID_OC_SFU]->empty() ) {
                        issue_warp(m_pipeline_reg[ID_OC_SFU],pI,active_mask,warp_id);
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

void shader_core_ctx::execute_pipe( unsigned pipeline, unsigned next_stage ) 
{
    if( !m_pipeline_reg[next_stage]->empty() )
        return;
    if( m_pipeline_reg[pipeline]->cycles ) {
        m_pipeline_reg[pipeline]->cycles--;
        return;
    }
    move_warp(m_pipeline_reg[next_stage],m_pipeline_reg[pipeline]);
    m_shader_memory_new_instruction_processed = false;
}

void shader_core_ctx::execute()
{
   execute_pipe(OC_EX_SFU, EX_MM);
   execute_pipe(ID_EX, EX_MM);
}

mshr_entry* mshr_shader_unit::add_mshr(mem_access_t &access, warp_inst_t* warp)
{
    // creates an mshr based on the access struct information
    mshr_entry* mshr = alloc_free_mshr(access.space == tex_space);
    mshr->init(access.addr,access.iswrite,access.space,warp->warp_id());
    mshr->add_inst(*warp);
    if( m_shader_config->gpgpu_interwarp_mshr_merge ) {
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
   const warp_inst_t* insns = m_pipeline_reg[EX_MM];
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
         if ( !insns->active(i) ) 
            continue;
         if( insns->space == undefined_space ) 
        	 continue; // this happens when thread predicated off
         new_addr_type addr = insns->get_addr(i);
         address_type lane_segment_address = tag_func(addr, line_size);
         unsigned quarter = 0;
         if( line_size>=4 )
            quarter = (addr / (line_size/4)) & 3;
         bool match = false;
         if( !insns->isatomic() ) { //atomics must have own request
            for (unsigned j = qpartbegin; j < accessq.size(); j++) {
               if (lane_segment_address == accessq[j].addr) {
                  assert( not accessq[j].isatomic );
                  accessq[j].quarter_count[quarter]++;
                  accessq[j].warp_indices.push_back(i);
                  if (limit_broadcast) // two threads access this address, so its a broadcast. 
                     accessq[j].order = ++broadcast_order; //do broadcast in its own cycle.
                  match = true;
                  break;
               }
            }
         }
         if (!match) { // does not match an previous request by another thread, so need a new request
            assert( insns->space != undefined_space );
            accessq.push_back( mem_access_t( lane_segment_address, 
                                             insns->space, 
                                             mem_pipe, 
                                             insns->isatomic(), 
                                             is_store(*insns),
                                             line_size, quarter, i) );
            // Determine Bank Conflicts:
            unsigned bank = (this->*bank_func)(insns->get_addr(i), line_size);
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
    get_memory_access_list( &shader_core_ctx::null_bank_func, 
                          line_size_based_tag_func,
                          CONSTANT_MEM_PATH, 
                          1, //warp parts 
                          m_L1C->get_line_sz(), false, //no broadcast limit.
                          accessq);
    // do cache checks here for each request (non-physical), could be 
    // done later for more accurate timing of cache accesses, but probably uneccesary; 
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
   // do cache checks here for each request (non-hardware), could be done later 
   // for more accurate timing of cache accesses, but probably uneccesary; 
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
         unsigned data_size = m_pipeline_reg[EX_MM]->data_size;
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
   // Attempt to send an request/write to memory based on information in access.  
   warp_inst_t* warp = m_pipeline_reg[EX_MM];

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
   bool requires_mshr = (not access.iswrite);
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
      unsigned warp_id = warp->warp_id();
      if (access.iswrite) {
         if (!strcmp("GT200",m_config->pipeline_model) ) 
            m_warp[warp_id].inc_store_req();
         for (unsigned i=0;i < access.warp_indices.size();i++) {
            unsigned w = access.warp_indices[i];
            int data_offset = warp->get_addr(w) & ((unsigned long long int)access.req_size - 1);
            for (unsigned b = data_offset; b < data_offset + warp->data_size; b++) write_mask.set(b);
         }
         if (write_mask.count() != access.req_size) 
            m_stats->gpgpu_n_partial_writes++;
      }
      fq_push( access.addr, request_size,
               access.iswrite, write_mask, warp_id , access.reserved_mshr, 
               access_type, warp->pc );
   }
   return NO_RC_FAIL;
}     

void shader_core_ctx::writeback()
{
    mshr_entry *m = m_mshr_unit->return_head();
    if( m ) 
        m_mshr_unit->pop_return_head();
    if( !m_pipeline_reg[MM_WB]->empty() ) {
        m_scoreboard->releaseRegisters( m_pipeline_reg[MM_WB] );
        m_warp[m_pipeline_reg[MM_WB]->warp_id()].dec_inst_in_pipeline();
        m_gpu->gpu_sim_insn_last_update_sid = m_sid;
        m_gpu->gpu_sim_insn_last_update = gpu_sim_cycle;
        m_gpu->gpu_sim_insn += m_pipeline_reg[MM_WB]->active_count();
    }
    move_warp(m_pipeline_reg[WB_RT],m_pipeline_reg[MM_WB]);
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
         m_pipeline_reg[EX_MM]->do_atomic();
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

   if( m_pipeline_reg[EX_MM]->empty() )
       return;
   m_gpu->mem_instruction_stats(m_pipeline_reg[EX_MM]);
   switch( m_pipeline_reg[EX_MM]->space.get_type() ) {
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
   if( not m_pipeline_reg[MM_WB]->empty() )
      return; // writeback stalled
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

void shader_core_ctx::print_warp( warp_inst_t *warp, FILE *fout, int print_mem, int mask ) const
{
    if ( warp->empty() ) {
        fprintf(fout,"bubble\n" );
        return;
    } else 
        fprintf(fout,"0x%04x ", warp->pc );
    unsigned warp_id = warp->warp_id();
    fprintf(fout, "w%02d[", warp_id);
    for (unsigned j=0; j<m_config->warp_size; j++)
        fprintf(fout, "%c", (warp->active(j)?'1':'0') );
    fprintf(fout, "]: ");
    if ( m_config->model == POST_DOMINATOR ) {
        unsigned rp = m_pdom_warp[warp_id]->get_rp();
        if ( rp == (unsigned)-1 ) {
            fprintf(fout," rp:--- ");
        } else {
            fprintf(fout," rp:0x%03x ", rp );
        }
    }
    ptx_print_insn( warp->pc, fout );
    fprintf(fout, "\n");
}

void shader_core_ctx::print_stage(unsigned int stage, FILE *fout, int print_mem, int mask ) 
{
   print_warp(m_pipeline_reg[stage],fout,print_mem,mask);
}

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
                unsigned done_cycle = m_thread[tid].m_functional_model_thread_state->donecycle();
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
   if (!strcmp("GT200",m_config->pipeline_model) ) 
      fprintf(fout, "ID/EX (SP)  = ");
   print_stage(ID_EX, fout, print_mem, mask);
   if (!strcmp("GT200",m_config->pipeline_model) ) {
      fprintf(fout, "ID/EX (SFU) = ");
      print_stage(OC_EX_SFU, fout, print_mem, mask);
   }
   fprintf(fout, "EX/MEM      = ");
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
    m_pipeline_reg[WB_RT]->clear();
    writeback();
    memory();
    execute();
    m_operand_collector.step(m_pipeline_reg[ID_OC],m_pipeline_reg[ID_OC_SFU]);
    decode_new();
    fetch_new();
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
        fprintf( fout, "w%02u npc: 0x%04x, done:%2u i:%u s:%u a:%u (done: ", 
                m_warp_id,
                m_next_pc,
                n_completed,
                m_inst_in_pipeline, 
                m_stores_outstanding,
                m_n_atomic );
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
    }
}

void opndcoll_rfu_t::init( unsigned num_collectors_alu, 
                           unsigned num_collectors_sfu, 
                           unsigned num_banks, 
                           shader_core_ctx *shader,
                           warp_inst_t **alu_port,
                           warp_inst_t **sfu_port )
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

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift)
{
   int bank = regnum;
   if (bank_warp_shift)
      bank += wid;
   return bank % num_banks;
}

bool opndcoll_rfu_t::writeback( const warp_inst_t &warp )
{
   assert( !warp.empty() );
   std::list<unsigned> regs = m_shader->get_regs_written(warp);
   std::list<unsigned>::iterator r;
   unsigned last_reg = -1;
   unsigned n=0;
   for( r=regs.begin(); r!=regs.end();r++,n++ ) {
      unsigned reg = *r;
      unsigned bank = register_bank(reg,warp.warp_id(),m_num_banks,m_bank_warp_shift);
      if( m_arbiter.bank_idle(bank) ) {
          m_arbiter.allocate_bank_for_write(bank,op_t(&warp,reg,m_num_banks,m_bank_warp_shift));
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
      warp_inst_t **port = p->first;
      if( !(*port)->empty() ) 
         continue;
      dispatch_unit_t &du = p->second;
      collector_unit_t *cu = du.find_ready();
      if( cu ) {
         cu->dispatch();
         m_free_cu[port].push_back(cu);
      }
   }
}

void opndcoll_rfu_t::allocate_cu( warp_inst_t *&id_oc_reg )
{
   if( !id_oc_reg->empty() ) {
      warp_inst_t **port = NULL;
      if( id_oc_reg->op == SFU_OP ) 
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
      unsigned wid = rr.get_wid();
      unsigned bank = register_bank(reg,wid,m_num_banks,m_bank_warp_shift);
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
   return (!m_free) && m_not_ready.none() && (*m_port)->empty(); 
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
                                             warp_inst_t **port, 
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
   m_warp = new warp_inst_t(warp_size);
   m_bank_warp_shift=log2_warp_size;
}

void opndcoll_rfu_t::collector_unit_t::allocate( warp_inst_t *&pipeline_reg ) 
{
   assert(m_free);
   assert(m_not_ready.none());
   m_free = false;
   if( !pipeline_reg->empty() ) {
      m_warp_id = pipeline_reg->warp_id();
      for( unsigned op=0; op < 4; op++ ) {
         int reg_num = pipeline_reg->arch_reg[4+op]; // this math needs to match that used in function_info::ptx_decode_inst
         if( reg_num >= 0 ) { // valid register
            m_src_op[op] = op_t( this, op, reg_num, m_num_banks, m_bank_warp_shift );
            m_not_ready.set(op);
         } else 
            m_src_op[op] = op_t();
      }
      move_warp(m_warp,pipeline_reg);
   }
}

void opndcoll_rfu_t::collector_unit_t::dispatch()
{
   assert( m_not_ready.none() );
   move_warp(*m_port,m_warp);
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
