/* 
 * shader.cc
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
#include "../intersim/statwraper.h"
#include "../intersim/interconnect_interface.h"
#include "icnt_wrapper.h"
#include <string.h>
#include <limits.h>

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a,b) (((a)>(b))?(a):(b))


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
   "IN_CLUSTER2SHADER",
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
   if (mshr->istexture()) 
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

std::list<mshr_entry*> &mshr_shader_unit::choose_return_queue() 
{
   // prioritize a ready texture over a global/const...
   if ((not m_texture_mshr_pipeline.empty()) and m_texture_mshr_pipeline.front()->fetched()) 
       return m_texture_mshr_pipeline;
   assert(!m_mshr_return_queue.empty());
   return m_mshr_return_queue;
}

void mshr_shader_unit::mshr_return_from_mem(unsigned mshr_id)
{
    if(mshr_id == -1) {
        return;
    }
   mshr_entry *mshr = &m_mshrs[mshr_id];
   mshr->set_fetched();
   mem_fetch *mf = mshr->get_mf();
   if( mf ) mf->set_status( FETCHED, MR_RETURN_Q, gpu_sim_cycle+gpu_tot_sim_cycle );
   if ( not mshr->istexture() ) {
       //place in return queue
       mshr->add_to_queue( m_mshr_return_queue );
   }
}

void mshr_shader_unit::print(FILE* fp)
{
    unsigned n=0;
    unsigned num_outstanding = 0;
    for (mshr_storage_type::iterator it = m_mshrs.begin(); it != m_mshrs.end(); it++,n++) {
        mshr_entry *mshr = &(*it);
        if (find(m_free_list.begin(),m_free_list.end(), mshr) == m_free_list.end()) {
            num_outstanding++;
            mshr->print(fp);
        }
    }
    fprintf(fp,"ready texture mshrs:\n");
    std::list<mshr_entry*>::iterator m;
    for( m=m_texture_mshr_pipeline.begin(); m!=m_texture_mshr_pipeline.end(); m++ ) {
        mshr_entry *mshr = *m;
        mshr->print(fp);
    }
    fprintf(fp,"ready non-texture mshrs:\n");
    for( m=m_mshr_return_queue.begin(); m!=m_mshr_return_queue.end(); m++ ) {
        mshr_entry *mshr = *m;
        mshr->print(fp);
    }
    fprintf(fp,"outstanding memory requests = %u\n", num_outstanding );
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

shader_core_ctx::shader_core_ctx( class gpgpu_sim *gpu, 
                                  class simt_core_cluster *cluster,
                                  unsigned shader_id,
                                  unsigned tpc_id,
                                  struct shader_core_config *config,
                                  struct shader_core_stats *stats )
   : m_barriers( config->max_warps_per_shader, config->max_cta_per_core )
{
    m_gpu = gpu;
   m_cluster = cluster;
   m_config = config;
   m_stats = stats;
   unsigned warp_size=config->warp_size;
   config->max_sfu_latency = 32;
   config->max_sp_latency = 32;

   m_sid = shader_id;
   m_tpc = tpc_id;

   m_pipeline_reg = new warp_inst_t*[N_PIPELINE_STAGES];
   for (int j = 0; j<N_PIPELINE_STAGES; j++) 
      m_pipeline_reg[j] = new warp_inst_t(config);

   m_thread = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), config->n_thread_per_shader);

   m_not_completed = 0;
   m_n_active_cta = 0;
   for (unsigned i = 0; i<MAX_CTA_PER_SHADER; i++  ) 
      m_cta_status[i]=0;
   for (unsigned i = 0; i<config->n_thread_per_shader; i++) {
      m_thread[i].m_functional_model_thread_state = NULL;
      m_thread[i].m_cta_id = -1;
   }

   // fetch
   m_last_warp_fetched = 0;
   m_last_warp_issued = 0;

   #define STRSIZE 1024
   char L1I_name[STRSIZE];
   snprintf(L1I_name, STRSIZE, "L1I_%03d", m_sid);
   m_L1I = new cache_t(L1I_name,m_config->gpgpu_cache_il1_opt,    0,no_writes, m_sid,get_shader_instruction_cache_id());

   m_warp.resize(m_config->max_warps_per_shader, shd_warp_t(this, warp_size));
   m_pdom_warp = new pdom_warp_ctx_t*[config->max_warps_per_shader];
   for (unsigned i = 0; i < config->max_warps_per_shader; ++i) 
       m_pdom_warp[i] = new pdom_warp_ctx_t(i,this);
   m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader);

   m_operand_collector.add_port( m_config->gpgpu_operand_collector_num_units_sp, 
                                 &m_pipeline_reg[ID_OC_SP],
                                 &m_pipeline_reg[OC_EX_SP] );
   m_operand_collector.add_port( m_config->gpgpu_operand_collector_num_units_sfu, 
                                 &m_pipeline_reg[ID_OC_SFU],
                                 &m_pipeline_reg[OC_EX_SFU] );
   m_operand_collector.add_port( m_config->gpgpu_operand_collector_num_units_mem, 
                                 &m_pipeline_reg[ID_OC_MEM],
                                 &m_pipeline_reg[OC_EX_MEM] );
   m_operand_collector.init( m_config->gpgpu_num_reg_banks, this );

   // execute
   m_num_function_units = 3; // sp_unit, sfu, ldst_unit
   m_dispatch_port = new enum pipeline_stage_name_t[ m_num_function_units ];
   m_issue_port = new enum pipeline_stage_name_t[ m_num_function_units ];

   m_fu = new simd_function_unit*[m_num_function_units];

   m_fu[0] = new sp_unit( &m_pipeline_reg[EX_WB], m_config );
   m_dispatch_port[0] = ID_OC_SP;
   m_issue_port[0] = OC_EX_SP;

   m_fu[1] = new sfu( &m_pipeline_reg[EX_WB], m_config );
   m_dispatch_port[1] = ID_OC_SFU;
   m_issue_port[1] = OC_EX_SFU;

   m_ldst_unit = new ldst_unit( m_cluster, this, &m_operand_collector, m_scoreboard, config, m_stats, m_sid, m_tpc );
   m_fu[2] = m_ldst_unit;
   m_dispatch_port[2] = ID_OC_MEM;
   m_issue_port[2] = OC_EX_MEM;
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

unsigned gpgpu_sim::sid_to_cluster( unsigned sid ) const
{
    return sid / m_shader_config->n_simt_cores_per_cluster;
}

void gpgpu_sim::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
    unsigned cluster_id = sid_to_cluster(sid);
    m_cluster[cluster_id]->get_pdom_stack_top_info(sid,tid,pc,rpc);
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
        fprintf(fout, " pc: 0x%03x", warp->m_pc[k] );
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
void shader_core_ctx::fetch()
{
    if( m_inst_fetch_buffer.m_valid ) {
        // decode 1 or 2 instructions and place them into ibuffer
        address_type pc = m_inst_fetch_buffer.m_pc;
        const warp_inst_t* pI1 = ptx_fetch_inst(pc);
        m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(0,pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
        if( pI1 ) {
           const warp_inst_t* pI2 = ptx_fetch_inst(pc+pI1->isize);
           if( pI2 ) {
               m_warp[m_inst_fetch_buffer.m_warp_id].ibuffer_fill(1,pI2);
               m_warp[m_inst_fetch_buffer.m_warp_id].inc_inst_in_pipeline();
           }
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
                    if( !m_cluster->icnt_injection_buffer_full(ppc, req_size, false) ) {
                        m_last_warp_fetched=warp_id;
                        mem_fetch *mf = new mem_fetch(pc,
                                                      req_size,
                                                      READ_PACKET_SIZE,
                                                      m_sid,
                                                      m_tpc,
                                                      warp_id,
                                                      (unsigned)-1/*mshr_id*/,
                                                      NULL/*we don't have an instruction yet*/,
                                                      false,
                                                      NO_PARTIAL_WRITE,
                                                      INST_ACC_R,
                                                      RD_REQ );
                        m_cluster->icnt_inject_request_packet(mf);

                        m_warp[warp_id].set_imiss_pending();
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

void shader_core_ctx::func_exec_inst( warp_inst_t &inst )
{
    for ( unsigned t=0; t < m_config->warp_size; t++ ) {
        if( inst.active(t) ) {
            unsigned tid=m_config->warp_size*inst.warp_id()+t;
            m_thread[tid].m_functional_model_thread_state->ptx_exec_inst(inst,t);
            if( inst.has_callback(t) ) 
               m_warp[inst.warp_id()].inc_n_atomic();
            if (inst.space.is_local() && (inst.is_load() || inst.is_store()))
               inst.set_addr(t, translate_local_memaddr(inst.get_addr(t), tid, 
                                 m_config->n_simt_clusters*m_config->n_simt_cores_per_cluster) );
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
    m_scoreboard->reserveRegisters(pipe_reg);
    m_warp[warp_id].set_next_pc(next_inst->pc + next_inst->isize);
}

void shader_core_ctx::decode()
{
    for ( unsigned i=0; i < m_config->max_warps_per_shader; i++ ) {
        unsigned warp_id = (m_last_warp_issued+1+i) % m_config->max_warps_per_shader;
        unsigned checked=0;
        unsigned issued=0;
        while( !m_warp[warp_id].waiting() && !m_warp[warp_id].ibuffer_empty() && (checked < 2) && (issued < 2) ) {
            const warp_inst_t *pI = m_warp[warp_id].ibuffer_next_inst();
            bool valid = m_warp[warp_id].ibuffer_next_valid();
            unsigned pc,rpc;
            m_pdom_warp[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
            if( pI ) {
                assert(valid);
                if( pc != pI->pc ) {
                    // control hazard
                    m_warp[warp_id].set_next_pc(pc);
                    m_warp[warp_id].ibuffer_flush();
                } else if ( !m_scoreboard->checkCollision(warp_id, pI) ) {
                    unsigned active_mask = m_pdom_warp[warp_id]->get_active_mask();
                    assert( m_warp[warp_id].inst_in_pipeline() );
                    if ( (pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP) ) {
                        if( m_pipeline_reg[ID_OC_MEM]->empty() ) {
                            issue_warp(m_pipeline_reg[ID_OC_MEM],pI,active_mask,warp_id);
                            issued++;
                        }
                    } else {
                        bool sp_pipe_avail = m_pipeline_reg[ID_OC_SP]->empty();
                        bool sfu_pipe_avail = m_pipeline_reg[ID_OC_SFU]->empty();
                        if( sp_pipe_avail && (pI->op != SFU_OP) ) {
                            // always prefer SP pipe for operations that can use both SP and SFU pipelines
                            issue_warp(m_pipeline_reg[ID_OC_SP],pI,active_mask,warp_id);
                            issued++;
                        } else if ( (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP) ) {
                            if( sfu_pipe_avail ) {
                                issue_warp(m_pipeline_reg[ID_OC_SFU],pI,active_mask,warp_id);
                                issued++;
                            }
                        } 
                    }
                }
            } else if( valid ) {
               // this case can happen after a return instruction in diverged warp
               m_warp[warp_id].set_next_pc(pc);
               m_warp[warp_id].ibuffer_flush();
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

address_type shader_core_ctx::translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader )
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

void shader_core_ctx::execute()
{
    m_result_bus >>= 1;
    for( unsigned n=0; n < m_num_function_units; n++ ) {
        m_fu[n]->cycle();
        enum pipeline_stage_name_t issue_port = m_issue_port[n];
        warp_inst_t *& issue_inst = m_pipeline_reg[ issue_port ];
        if( !issue_inst->empty() && m_fu[n]->can_issue( *issue_inst ) ) {
            bool schedule_wb_now = !m_fu[n]->stallable();
            if( schedule_wb_now && !m_result_bus.test( issue_inst->latency ) ) {
                assert( issue_inst->latency < MAX_ALU_LATENCY );
                m_result_bus.set( issue_inst->latency );
                m_fu[n]->issue( issue_inst );
            } else if( !schedule_wb_now ) {
                m_fu[n]->issue( issue_inst );
            } else {
                // stall issue (cannot reserve result bus)
            }
        }
    }
}

mshr_entry* mshr_shader_unit::add_mshr(mem_access_t &access, warp_inst_t* pinst)
{
    // creates an mshr based on the access struct information
    mshr_entry* mshr = alloc_free_mshr(pinst->space == tex_space);
    mshr->init(access.addr,pinst->is_store(),pinst->space,pinst->warp_id());
    warp_inst_t inst = *pinst;
    inst.set_active(access.warp_indices);
    mshr->add_inst(inst);
    if( m_shader_config->gpgpu_interwarp_mshr_merge ) {
        mshr_entry* mergehit = m_mshr_lookup.shader_get_mergeable_mshr(mshr);
        if (mergehit) {
            mergehit->merge(mshr);
            if (mergehit->fetched())
                mshr_return_from_mem(mshr->get_id());
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
   return address; //no modification: each address is its own tag.
}

unsigned shader_core_config::shmem_bank_func(address_type addr, unsigned) const
{
   return ((addr/WORD_SIZE) % gpgpu_n_shmem_bank);
}

unsigned shader_core_config::dcache_bank_func(address_type add, unsigned line_size) const
{
   if (gpgpu_no_dl1) return 1; //no banks
   else return (add / line_size) & (gpgpu_n_cache_bank - 1);
}

void warp_inst_t::get_memory_access_list()
{
   // Calculates memory accesses generated by this warp
   // Returns acesses which are "coalesced" 
   // Does not coalesce nor overlap bank accesses across warp "parts".

   // This is called once per warp_inst_t when the warp_inst_t enters the memory stage.
   // It produces the set of distinct memory accesses that need to be peformed.
   // These accessess are then performed over multiple cycles (stalling the pipeline) 
   // if the accessses cannot be performed all at once.
    
   // In hardware, these accesses would be created at the specific unit handling the type 
   // of memory access. We centralize the logic simply to reduce code duplication. 

   // Below, accesses are assigned an "order" based on when that access may be issued. 
   // Accesses with the same order number may occur at the same time: they are to different banks. 
   // Later, when the queue is processed it will evaluate accesses of as many orders as 
   // ports on that cache/shmem.
   //  
   // Accesses are placed in accessq sorted so that accesses of the same order are adjacent.

    bank_func_t bank_func = NULL;
    tag_func_t tag_func = NULL;
    unsigned warp_parts = 0;
    unsigned line_size = 0;
    bool limit_broadcast = 0; 
    bool global_mem_access = false;

    switch( space.get_type() ) {
    case shared_space: 
        bank_func = &shader_core_config::shmem_bank_func; 
        tag_func = null_tag_func;
        warp_parts = m_config->gpgpu_shmem_pipe_speedup; 
        line_size = 1; //shared memory doesn't care about line_size, needs to be at least 1;
        limit_broadcast = true; // limit broadcasts to single cycle. 
        break;
    case tex_space: 
        bank_func = &shader_core_config::null_bank_func;
        tag_func = line_size_based_tag_func;
        warp_parts = 1;
        line_size = m_config->gpgpu_cache_texl1_linesize;
        limit_broadcast = false;
        break;
    case const_space:  case param_space_kernel: 
        bank_func = &shader_core_config::null_bank_func; 
        tag_func = line_size_based_tag_func;
        warp_parts = 1;
        line_size = m_config->gpgpu_cache_constl1_linesize;
        limit_broadcast = false;
        break;
    case global_space: case local_space: case param_space_local: 
        global_mem_access=true;
        warp_parts = 1;
        line_size = m_config->gpgpu_cache_dl1_linesize;
        if( m_config->gpgpu_coalesce_arch == 13 ){
           warp_parts = 2;
           if( m_config->gpgpu_no_dl1 ) {
              // line size is dependant on instruction;
              switch (data_size) {
              case 1: line_size = 32; break;
              case 2: line_size = 64; break;
              case 4: case 8: case 16: line_size = 128; break;
              default: assert(0);
              }
           }
        }          
        bank_func = &shader_core_config::dcache_bank_func;
        tag_func = line_size_based_tag_func;
        limit_broadcast = false;
        break;
    default:
        abort();
    }

   // bank_accs tracks bank accesses for sorting into generations;
   // each entry is (bank #, number of accesses)
   // the idea is that you can only access a bank a number of times each cycle equal to 
   // its number of ports in one cycle. 
   std::map<unsigned,unsigned> bank_accs;

   // keep track of broadcasts with unique orders if limit_broadcast
   // the normally calculated orders will never be greater than warp_size
   unsigned broadcast_order =  warp_size();
   unsigned qbegin = get_accessq_size();
   unsigned qpartbegin = qbegin;
   unsigned mem_pipe_size = warp_size() / warp_parts;
   for (unsigned part = 0; part < warp_size(); part += mem_pipe_size) {
      for (unsigned i = part; i < part + mem_pipe_size; i++) {
         if ( !active(i) ) 
            continue;
         new_addr_type addr = get_addr(i);
         address_type lane_segment_address = tag_func(addr, line_size);
         unsigned quarter = 0;
         if( line_size>=4 )
            quarter = (addr / (line_size/4)) & 3;
         bool match = false;
         if( !isatomic() ) { //atomics must have own request
            for( unsigned j = qpartbegin; j <get_accessq_size(); j++ ) {
               if (lane_segment_address == accessq(j).addr) {
                  accessq(j).quarter_count[quarter]++;
                  accessq(j).warp_indices.push_back(i);
                  if (limit_broadcast) // two threads access this address, so its a broadcast. 
                     accessq(j).order = ++broadcast_order; //do broadcast in its own cycle.
                  match = true;
                  break;
               }
            }
         }
         if (!match) { // does not match a previous request by another thread, so need a new request
            assert( space != undefined_space );
            m_accessq.push_back( mem_access_t( lane_segment_address, line_size, quarter, i) );
            // Determine Bank Conflicts:
            unsigned bank = (m_config->*bank_func)(get_addr(i), line_size);
            // ensure no concurrent bank access accross warp parts. 
            // ie. order will be less than part for all previous loads in previous parts, so:
            if (bank_accs[bank] < part) 
               bank_accs[bank]=part; 
            accessq_back().order = bank_accs[bank];
            bank_accs[bank]++;
         }
      }
      qpartbegin = get_accessq_size(); //don't coalesce accross warp parts
   }
   //sort requests by order they will be processed in
   std::stable_sort( m_accessq.begin()+qbegin,m_accessq.end());

   if( global_mem_access ) {
       // Now that we have the accesses, if we don't have a cache we can adjust request sizes to 
       // include only the data referenced by the threads 
       for (unsigned i = 0; i < get_accessq_size(); i++) {
          if (m_config->gpgpu_coalesce_arch == 13 && m_config->gpgpu_no_dl1) {
             // do coalescing here.
             char* quarter_counts = accessq(i).quarter_count;
             bool low = quarter_counts[0] or quarter_counts[1];
             bool high = quarter_counts[2] or quarter_counts[3];
             if (accessq(i).req_size == 128) {
                if (low xor high) { //can reduce size
                   accessq(i).req_size = 64;
                   if (high) accessq(i).addr += 64;
                   low = quarter_counts[0] or quarter_counts[2]; //set low and high for next pass
                   high = quarter_counts[1] or quarter_counts[3];
                }
             }
             if (accessq(i).req_size == 64) {
                if (low xor high) { //can reduce size
                   accessq(i).req_size = 32;
                   if (high) accessq(i).addr += 32;
                }
             }
          }
       }
   }
} 

void ldst_unit::const_cache_access(warp_inst_t &inst)
{
    // do cache checks here for each request (non-physical), could be 
    // done later for more accurate timing of cache accesses, but probably uneccesary; 
    for (unsigned i = 0; i < inst.get_accessq_size(); i++) {
        mem_access_t &req = inst.accessq(i);
        if ( inst.space == param_space_kernel ) {
            req.cache_hit = true;
        } else {
            cache_request_status status = m_L1C->access( req.addr,
                                                         0, //should always be a read
                                                         gpu_sim_cycle+gpu_tot_sim_cycle, 
                                                         NULL/*should never writeback*/);
            req.cache_hit = (status == HIT);
            if (m_config->gpgpu_perfect_mem) req.cache_hit = true;
            if (req.cache_hit) m_stats->L1_const_miss++;
        } 
        req.cache_checked = true;
    }
}

void ldst_unit::tex_cache_access(warp_inst_t &inst)
{
   // do cache checks here for each request (non-hardware), could be done later 
   // for more accurate timing of cache accesses, but probably uneccesary; 
   for (unsigned i = 0; i < inst.get_accessq_size(); i++) {
       mem_access_t &req = inst.accessq(i);
      cache_request_status status = m_L1T->access( req.addr,
                                                   0, //should always be a read
                                                   gpu_sim_cycle+gpu_tot_sim_cycle, 
                                                   NULL /*should never writeback*/);
      req.cache_hit = (status == HIT);
      if (m_config->gpgpu_perfect_mem) req.cache_hit = true;
      if (req.cache_hit) m_stats->L1_texture_miss++;
      req.cache_checked = true;
   }
}

mem_stage_stall_type ldst_unit::send_mem_request(warp_inst_t &inst, mem_access_t &access)
{
   // Attempt to send an request/write to memory based on information in access.  

   // If the cache told us it needed to write back a dirty line, do this now
   // It is possible to do this writeback in the same cycle as the access request, this may not be realistic.
   if (access.need_wb) {
      unsigned req_size = m_config->gpgpu_cache_dl1_linesize + WRITE_PACKET_SIZE;
      if ( m_cluster->icnt_injection_buffer_full(access.wb_addr, req_size, true) ) {
         m_stats->gpu_stall_sh2icnt++; 
         return WB_ICNT_RC_FAIL;
      }
      mem_fetch *mf = new mem_fetch(access.wb_addr,
                                    req_size,
                                    READ_PACKET_SIZE,
                                    m_sid,
                                    m_tpc,
                                    -1/*wid*/,
                                    -1/*mshr id*/,
                                    NULL,
                                    true,
                                    NO_PARTIAL_WRITE,
                                    inst.space.is_local()?LOCAL_ACC_W:GLOBAL_ACC_W, //space of cache line is same as new request
                                    WT_REQ );
      m_cluster->icnt_inject_request_packet(mf);
      inst.clear_active(access.warp_indices);
      m_stats->L1_writeback++;
      access.need_wb = false; 
   }

   bool is_write = inst.is_store();
   mem_access_type access_type;
   bool requires_mshr = false;
   switch(inst.space.get_type()) {
   case const_space:
   case param_space_kernel: 
       access_type = CONST_ACC_R; 
       break;
   case tex_space: 
       access_type = TEXTURE_ACC_R;   
       break;
   case global_space:       
       access_type = is_write? GLOBAL_ACC_W: GLOBAL_ACC_R;   
       break;
   case local_space:
   case param_space_local:  
       access_type = is_write? LOCAL_ACC_W: LOCAL_ACC_R;   
       break;
   default: assert(0); break; 
   }
   //reserve mshr
   if (requires_mshr && !access.reserved_mshr) {
      if (not m_mshr_unit->has_mshr(1)) 
         return MSHR_RC_FAIL;
      access.reserved_mshr = m_mshr_unit->add_mshr(access, &inst);
      access.recheck_cache = false; 
      //we have an mshr now, so have checked cache in same cycle as checking mshrs, so have merged if necessary.
   }
   //require inct if access is this far without reserved mshr, or has and mshr but not merged with another request
   bool requires_icnt = (!access.reserved_mshr) || (!access.reserved_mshr->ismerged());
   if (requires_icnt) {
      //calculate request size for icnt check (and later send);
      unsigned request_size = access.req_size;
      if (is_write) {
         if (requires_mshr) 
            request_size += READ_PACKET_SIZE + WRITE_MASK_SIZE; // needs information for a load back into cache.
         else 
            request_size += WRITE_PACKET_SIZE + WRITE_MASK_SIZE; //plain write
      }
      if ( m_cluster->icnt_injection_buffer_full(access.addr, request_size, is_write) ) {
         // can't allocate icnt
         m_stats->gpu_stall_sh2icnt++;
         return ICNT_RC_FAIL;
      }
      //send over interconnect
      partial_write_mask_t  write_mask = NO_PARTIAL_WRITE;
      unsigned warp_id = inst.warp_id();
      if (is_write) {
         m_core->inc_store_req(warp_id);
         for (unsigned i=0;i < access.warp_indices.size();i++) {
            unsigned w = access.warp_indices[i];
            int data_offset = inst.get_addr(w) & ((unsigned long long int)access.req_size - 1);
            for (unsigned b = data_offset; b < data_offset + inst.data_size; b++) write_mask.set(b);
         }
         if (write_mask.count() != access.req_size) 
            m_stats->gpgpu_n_partial_writes++;
      }
      unsigned mshr_id = access.reserved_mshr?access.reserved_mshr->get_id():-1;
      warp_inst_t inst_copy = inst;
      inst_copy.set_active(access.warp_indices);
      mem_fetch *mf = new mem_fetch(access.addr,
                                    request_size,
                                    is_write?WRITE_PACKET_SIZE:READ_PACKET_SIZE,
                                    m_sid,
                                    m_tpc,
                                    warp_id,
                                    mshr_id,
                                    &inst_copy,
                                    is_write,
                                    write_mask,
                                    access_type,
                                    is_write?WT_REQ:RD_REQ);
      if( access.reserved_mshr ) 
          access.reserved_mshr->set_mf(mf);
      m_cluster->icnt_inject_request_packet(mf);
      if( inst.is_load() ) { 
          for( unsigned r=0; r < 4; r++) 
              if(inst.out[r] > 0) 
                  m_pending_writes[inst.warp_id()][inst.out[r]]++;
      }
   }
   return NO_RC_FAIL;
}     

void shader_core_ctx::writeback()
{
    warp_inst_t *&pipe_reg = m_pipeline_reg[EX_WB];
    if( !pipe_reg->empty() ) {
        unsigned warp_id = pipe_reg->warp_id();
        m_scoreboard->releaseRegisters( pipe_reg );
        m_warp[warp_id].dec_inst_in_pipeline();
        m_gpu->gpu_sim_insn_last_update_sid = m_sid;
        m_gpu->gpu_sim_insn_last_update = gpu_sim_cycle;
        m_gpu->gpu_sim_insn += pipe_reg->active_count();
        pipe_reg->clear();
    }
    m_ldst_unit->writeback();
}

bool ldst_unit::shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   // Process a single cycle of activity from the shared memory queue.
   if( inst.space.get_type() != shared_space )
       return true;

   //consume port number orders from the top of the queue;
   for( int i = 0; i < m_config->gpgpu_shmem_port_per_bank; i++ ) {
      if (inst.accessq_empty()) 
         break;
      unsigned current_order = inst.accessq_back().order;
      //consume all requests of the same order (concurrent bank requests)
      while ((!inst.accessq_empty()) && inst.accessq_back().order == current_order) 
         inst.accessq_pop_back();
   }
   if( !inst.accessq_empty() ) {
      rc_fail = BK_CONF;
      fail_type = S_MEM;
      m_stats->gpgpu_n_shmem_bkconflict++;
   }
   return inst.accessq_empty(); //done if empty.
}

mem_stage_stall_type ldst_unit::process_memory_access_queue( ldst_unit::cache_check_t cache_check,
                                                             unsigned ports_per_bank, 
                                                             unsigned memory_send_max, 
                                                             warp_inst_t &inst )
{
   // Generic algorithm for processing a single cycle of accesses for the memory space types that go to L2 or DRAM.

   // precondition: accessq sorted by mem_access_t::order
   mem_stage_stall_type hazard_cond = NO_RC_FAIL; 
   unsigned mem_req_count = 0; // number of requests to sent to memory this cycle
   for (unsigned i = 0; i < ports_per_bank; i++) {
      if (inst.accessq_empty()) 
         break;
      unsigned current_order = inst.accessq_back().order;
      // consume all requests of the same "order" but stop if we hit a structural hazard
      while( !inst.accessq_empty() && inst.accessq_back().order == current_order && hazard_cond == NO_RC_FAIL) {
         hazard_cond = (this->*cache_check)(inst,inst.accessq_back());
         if (hazard_cond != NO_RC_FAIL) 
            break; // can't complete this request this cycle.
         if ( !inst.accessq_back().cache_hit ){
            if (mem_req_count < memory_send_max) {
               mem_req_count++;
               hazard_cond = send_mem_request(inst,inst.accessq_back());
            }
            else hazard_cond = COAL_STALL; // not really a coal stall, its a too many memory request stall;
            if ( hazard_cond != NO_RC_FAIL) 
                break; //can't complete this request this cycle.
         } 
         inst.accessq_pop_back();
      }
   }
   if (!inst.accessq_empty() && hazard_cond == NO_RC_FAIL) {
      //no resource failed so must be a bank comflict.
      hazard_cond = BK_CONF;
   }
   return hazard_cond;
}  

bool ldst_unit::constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( (inst.space.get_type() != const_space) && (inst.space.get_type() != param_space_kernel) )
       return true;

   // Process a single cycle of activity from the the constant memory queue.
   mem_stage_stall_type fail = process_memory_access_queue(&ldst_unit::ccache_check, 
                                                           m_config->gpgpu_const_port_per_bank, 
                                                           1, //memory send max per cycle
                                                           inst );
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = C_MEM;
      if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
         m_stats->gpgpu_n_cmem_portconflict++; //coal stalls aren't really a bank conflict, but this maintains previous behavior.
      }
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.space.get_type() != tex_space )
       return true;

   // Process a single cycle of activity from the the texture memory queue.

   mem_stage_stall_type fail = process_memory_access_queue(&ldst_unit::tcache_check, 
                                                           1, //how is tex memory banked? 
                                                           1, //memory send max per cycle
                                                           inst );
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = T_MEM;
   }
   return inst.accessq_empty(); //done if empty.
}

mem_stage_stall_type ldst_unit::dcache_check(warp_inst_t &inst, mem_access_t& access)
{
   // Global memory (data cache) checks the cache for each access at the time it is processed.
   // This is more accurate to hardware, and necessary for proper action of the writeback cache.

   if (access.cache_checked && !access.recheck_cache) 
      return NO_RC_FAIL;
   if (!m_config->gpgpu_no_dl1 && !m_config->gpgpu_perfect_mem) { 
      //check cache
      cache_request_status status = m_L1D->access( access.addr,
                                                   inst.is_store(),
                                                   gpu_sim_cycle+gpu_tot_sim_cycle,
                                                   &access.wb_addr );
      if (status == RESERVATION_FAIL) {
         access.cache_checked = false;
         return WB_CACHE_RSRV_FAIL;
      }
      access.cache_hit = (status == HIT); //if HIT_W_WT then still send to memory so "MISS" 
      if (status == MISS_W_WB) 
          access.need_wb = true;
      if (status == WB_HIT_ON_MISS && inst.is_store()) 
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
      if (status == WB_HIT_ON_MISS && !inst.is_store() ) {
         //read has hit on a reserved cache line, 
         //we need to make sure cache check happens on same cycle as a mshr merge happens, otherwise we might miss it coming back
         access.recheck_cache = true;
      }
      access.cache_checked = true;
   } else {
      access.cache_hit = false;
   }

   if (m_config->gpgpu_perfect_mem) 
       access.cache_hit = true;
   
   if (inst.isatomic()) {
      if (m_config->gpgpu_perfect_mem) {
         // complete functional execution of atomic here
         inst.do_atomic();
      } else {
         // atomics always go to memory 
         access.cache_hit = false;
      }
   }

   if (!access.cache_hit) { 
      if (inst.is_store()) m_stats->L1_write_miss++;
      else m_stats->L1_read_miss++;
   }
   return NO_RC_FAIL;
}

bool ldst_unit::memory_cycle( warp_inst_t &inst, mem_stage_stall_type &stall_reason, mem_stage_access_type &access_type )
{
   if( (inst.space.get_type() != global_space) &&
       (inst.space.get_type() != local_space) &&
       (inst.space.get_type() != param_space_local) ) 
       return true;

   // Process a single cycle of activity from the the global/local memory queue.
   mem_stage_stall_type stall_cond = process_memory_access_queue(&ldst_unit::dcache_check,m_config->gpgpu_cache_port_per_bank,1,inst);
   if (stall_cond != NO_RC_FAIL) {
      stall_reason = stall_cond;
      bool iswrite = inst.is_store();
      if (inst.space.is_local()) 
         access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
      else 
         access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
      if (stall_cond == BK_CONF || stall_cond == COAL_STALL) 
         m_stats->gpgpu_n_cache_bkconflict++;
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::response_buffer_full() const
{
    return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

void ldst_unit::fill( mem_fetch *mf )
{
    m_response_fifo.push_back(mf);
}

void ldst_unit::flush()
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

void ldst_unit::generate_mem_accesses(warp_inst_t &inst)
{
    // Called once per warp when warp enters ld/st unit.
    // Generates a list of memory accesses, but does not perform the memory access.
   if( inst.empty() ) 
       return;
   if( inst.op == MEMORY_BARRIER_OP ) 
       return;
   if( inst.mem_accesses_created() )
       return;
   inst.get_memory_access_list();
   switch( inst.space.get_type() ) {
      case shared_space: break;
      case tex_space: tex_cache_access(inst); break;
      case const_space:  case param_space_kernel: const_cache_access(inst); break;
      case global_space: case local_space: case param_space_local: break;
      case param_space_unclassified: abort(); break;
      default: break; // non-memory operations
   }
   m_cluster->mem_instruction_stats(inst);
   inst.set_mem_accesses_created();
}

ldst_unit::ldst_unit( simt_core_cluster *cluster, 
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      shader_core_config *config, 
                      shader_core_stats *stats, 
                      unsigned sid,
                      unsigned tpc ) : pipelined_simd_unit(NULL,config,6) 
{
    m_cluster = cluster;
    m_core = core;
    m_operand_collector = operand_collector;
    m_scoreboard = scoreboard;
    m_stats = stats;
    m_sid = sid;
    m_tpc = tpc;
    #define STRSIZE 1024
    char L1D_name[STRSIZE];
    char L1T_name[STRSIZE];
    char L1C_name[STRSIZE];
    snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
    snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
    snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
    enum cache_write_policy L1D_policy = m_config->gpgpu_cache_wt_through?write_through:write_back;
    m_L1D = new cache_t(L1D_name,m_config->gpgpu_cache_dl1_opt,0,L1D_policy,m_sid,get_shader_normal_cache_id());
    m_L1T = new cache_t(L1T_name,m_config->gpgpu_cache_texl1_opt,0,no_writes, m_sid,get_shader_texture_cache_id());
    m_L1C = new cache_t(L1C_name,m_config->gpgpu_cache_constl1_opt,0,no_writes, m_sid,get_shader_constant_cache_id());
    config->gpgpu_cache_dl1_linesize = m_L1D->get_line_sz();
    config->gpgpu_cache_texl1_linesize = m_L1T->get_line_sz();
    config->gpgpu_cache_constl1_linesize = m_L1C->get_line_sz();
    m_cluster->get_gpu()->ptx_set_tex_cache_linesize(m_L1T->get_line_sz());
    m_mshr_unit = new mshr_shader_unit(m_config);
    m_mem_rc = NO_RC_FAIL;
}

void ldst_unit::writeback()
{
    mshr_entry *m = m_mshr_unit->return_head();
    if( m ) m_mshr_unit->pop_return_head();

    if( !m_pipeline_reg[0]->empty() ) {
        // shared memory has priority
        if( m_operand_collector->writeback(*m_pipeline_reg[0]) ) {
            m_scoreboard->releaseRegisters(m_pipeline_reg[0]);
            m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
            m_pipeline_reg[0]->clear();
        }
    }

    if( !m_response_fifo.empty() ) {
        mem_fetch *mf = m_response_fifo.front();
        if( mf->get_is_write() ) {
            m_core->store_ack(mf);
            m_response_fifo.pop_front();
        } else {
            const warp_inst_t &inst = mf->get_inst();
            if( m_operand_collector->writeback(inst) ) {
                m_response_fifo.pop_front();
                if( mf->isatomic() ) 
                    m_core->decrement_atomic_count(mf->get_wid(),inst.active_count());
                for( unsigned r=0; r < 4; r++ ) {
                    if( inst.out[r] > 0 ) {
                        assert( m_pending_writes[inst.warp_id()][inst.out[r]] > 0 );
                        unsigned still_pending = --m_pending_writes[inst.warp_id()][inst.out[r]];
                        if( !still_pending ) {
                            m_pending_writes[inst.warp_id()].erase(inst.out[r]);
                            m_scoreboard->releaseRegister( inst.warp_id(), inst.out[r] );
                        }
                    }
                }
                m_mshr_unit->mshr_return_from_mem(mf->get_mshr());
                if (mf->istexture()) 
                    m_L1T->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
                else if (mf->isconst()) 
                    m_L1C->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
                else if (!m_config->gpgpu_no_dl1) 
                    m_L1D->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
                delete mf;
            }
        }
    }
}

void ldst_unit::cycle()
{
   for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
       if( m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage+1]->empty() )
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);

   // process new memory requests
   warp_inst_t &pipe_reg = *m_dispatch_reg;
   generate_mem_accesses(pipe_reg);

   enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
   mem_stage_access_type type;
   bool done = true;
   done &= shared_cycle(pipe_reg, rc_fail, type);
   done &= constant_cycle(pipe_reg, rc_fail, type);
   done &= texture_cycle(pipe_reg, rc_fail, type);
   done &= memory_cycle(pipe_reg, rc_fail, type);
   m_mem_rc = rc_fail;
   if (!done) { // log stall types and return
      assert(rc_fail != NO_RC_FAIL);
      m_stats->gpu_stall_shd_mem++;
      m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
      return;
   }

   if( !pipe_reg.empty() ) {
       unsigned warp_id = pipe_reg.warp_id();
       if( pipe_reg.is_load() ) {
           if( pipe_reg.space.get_type() == shared_space ) {
               if( m_pipeline_reg[5]->empty() ) {
                   // new shared memory request
                   move_warp(m_pipeline_reg[5],m_dispatch_reg);
                   m_dispatch_reg->clear();
               }
           } else {
               if( pipe_reg.active_count() > 0 ) {
                   if( !m_operand_collector->writeback(pipe_reg) ) 
                       return;
               } 

               bool pending_requests=false;
               for( unsigned r=0; r<4; r++ ) {
                   unsigned reg_id = pipe_reg.out[r];
                   if( reg_id > 0 ) {
                       if( m_pending_writes[warp_id].find(reg_id) != m_pending_writes[warp_id].end() ) {
                           assert( m_pending_writes[warp_id][reg_id] > 0 );
                           pending_requests=true;
                           break;
                       }
                   }
               }
               if( !pending_requests )
                   m_scoreboard->releaseRegisters(m_dispatch_reg);
               m_core->dec_inst_in_pipeline(warp_id);
               m_dispatch_reg->clear();
           }
       } else {
           // stores exit pipeline here
           m_core->dec_inst_in_pipeline(warp_id);
           m_dispatch_reg->clear();
       }
   }
}

void shader_core_ctx::register_cta_thread_exit(int tid )
{
   unsigned padded_cta_size = m_gpu->the_kernel().threads_per_cta();
   if (padded_cta_size%m_config->warp_size) 
      padded_cta_size = ((padded_cta_size/m_config->warp_size)+1)*(m_config->warp_size);
   int cta_num = tid/padded_cta_size;
   assert( m_cta_status[cta_num] > 0 );
   m_cta_status[cta_num]--;
   if (!m_cta_status[cta_num]) {
      m_n_active_cta--;
      deallocate_barrier(cta_num);
      shader_CTA_count_unlog(m_sid, 1);
      printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld)\n", m_sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle );
   }
}

void gpgpu_sim::shader_print_runtime_stat( FILE *fout ) 
{
    /*
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
   */
}


void gpgpu_sim::shader_print_l1_miss_stat( FILE *fout ) 
{
    /*
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
   */
}

void warp_inst_t::print( FILE *fout ) const
{
    if (empty() ) {
        fprintf(fout,"bubble\n" );
        return;
    } else 
        fprintf(fout,"0x%04x ", pc );
    fprintf(fout, "w%02d[", m_warp_id);
    for (unsigned j=0; j<m_config->warp_size; j++)
        fprintf(fout, "%c", (active(j)?'1':'0') );
    fprintf(fout, "]: ");
    ptx_print_insn( pc, fout );
    fprintf(fout, "\n");
}

void shader_core_ctx::print_stage(unsigned int stage, FILE *fout ) const
{
   m_pipeline_reg[stage]->print(fout);
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

void ldst_unit::print(FILE *fout) const
{
    fprintf(fout,"LD/ST unit  = ");
    m_dispatch_reg->print(fout);
    if ( m_mem_rc != NO_RC_FAIL ) {
        fprintf(fout,"              LD/ST stall condition: ");
        switch ( m_mem_rc ) {
        case BK_CONF:        fprintf(fout,"BK_CONF"); break;
        case MSHR_RC_FAIL:   fprintf(fout,"MSHR_RC_FAIL"); break;
        case ICNT_RC_FAIL:   fprintf(fout,"ICNT_RC_FAIL"); break;
        case COAL_STALL:     fprintf(fout,"COAL_STALL"); break;
        case WB_ICNT_RC_FAIL: fprintf(fout,"WB_ICNT_RC_FAIL"); break;
        case WB_CACHE_RSRV_FAIL: fprintf(fout,"WB_CACHE_RSRV_FAIL"); break;
        case N_MEM_STAGE_STALL_TYPE: fprintf(fout,"N_MEM_STAGE_STALL_TYPE"); break;
        default: abort();
        }
        fprintf(fout,"\n");
    }
    m_mshr_unit->print(fout);
    fprintf(fout,"Pending register writes:\n");
    std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> >::const_iterator w;
    for( w=m_pending_writes.begin(); w!=m_pending_writes.end(); w++ ) {
        unsigned warp_id = w->first;
        const std::map<unsigned/*regnum*/,unsigned/*count*/> &warp_info = w->second;
        if( warp_info.empty() ) 
            continue;
        fprintf(fout,"  w%2u : ", warp_id );
        std::map<unsigned/*regnum*/,unsigned/*count*/>::const_iterator r;
        for( r=warp_info.begin(); r!=warp_info.end(); ++r ) {
            fprintf(fout,"  %u(%u)", r->first, r->second );
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

   m_scoreboard->printContents();

   fprintf(fout,"ID/OC (SP)  = ");
   print_stage(ID_OC_SP, fout);
   fprintf(fout,"ID/OC (SFU) = ");
   print_stage(ID_OC_SFU, fout);
   fprintf(fout,"ID/OC (MEM) = ");
   print_stage(ID_OC_MEM, fout);

   m_operand_collector.dump(fout);

   fprintf(fout, "OC/EX (SP)  = ");
   print_stage(OC_EX_SP, fout);
   fprintf(fout, "OC/EX (SFU) = ");
   print_stage(OC_EX_SFU, fout);
   fprintf(fout, "OC/EX (MEM) = ");
   print_stage(OC_EX_MEM, fout);
   for( unsigned n=0; n < m_num_function_units; n++ ) 
       m_fu[n]->print(fout);
   std::string bits = m_result_bus.to_string();
   fprintf(fout, "EX/WB sched= %s\n", bits.c_str() );
   fprintf(fout, "EX/WB      = ");
   print_stage(EX_WB, fout);
   fprintf(fout, "\n");
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
      printf ("Error: max_cta_per_shader(\"sid=%u\") returning %d. Kernel requires more resources than shader has?\n", m_sid, result);
      abort();
   }
   return result;
}

void shader_core_ctx::cycle()
{
    writeback();
    execute();
    m_operand_collector.step();
    decode();
    fetch();
}

// Flushes all content of the cache to memory

void shader_core_ctx::cache_flush()
{
   m_ldst_unit->flush();
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

void shader_core_ctx::decrement_atomic_count( unsigned wid, unsigned n )
{
   assert( m_warp[wid].get_n_atomic() >= n );
   m_warp[wid].dec_n_atomic(n);
}


bool shader_core_ctx::fetch_unit_response_buffer_full() const
{
    return false;
}

void shader_core_ctx::accept_fetch_response( mem_fetch *mf )
{
    m_L1I->shd_cache_fill(mf->get_addr(),gpu_sim_cycle+gpu_tot_sim_cycle);
    m_warp[mf->get_wid()].clear_imiss_pending();
    delete mf;
}

bool shader_core_ctx::ldst_unit_response_buffer_full() 
{
    return m_ldst_unit->response_buffer_full();
}

void shader_core_ctx::accept_ldst_unit_response(mem_fetch * mf) 
{
   m_ldst_unit->fill(mf);
   //freed_read_mfs++;
}

void shader_core_ctx::store_ack( class mem_fetch *mf )
{
    unsigned warp_id = mf->get_wid();
    m_warp[warp_id].dec_store_req();
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
        const inst_t *inst = m_ibuffer[i].m_inst;
        if( inst ) inst->print_insn(fout);
        else if( m_ibuffer[i].m_valid ) 
           fprintf(fout," <invalid instruction> ");
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
   m_addr = address;
   m_mf = NULL;
   m_merged_on_other_reqest = false;
   m_merged_requests =NULL;
   m_iswrite = wr;
   m_isinst = space==instruction_space;
   m_islocal = space.is_local();
   m_isconst = space.is_const();
   m_istexture = space==tex_space;
   m_isatomic = false;
   m_insts.clear();
   m_warp_id = warp_id;
}

void mshr_entry::print(FILE *fp) const
{
    fprintf(fp, "MSHR(%u): w%2u req uid=%5u, %s (0x%llx) merged:%d status:%s ", 
            m_id,
            m_warp_id,
            m_request_uid,
            (m_iswrite)? "store" : "load ",
            m_addr, 
            (m_merged_requests != NULL || m_merged_on_other_reqest), 
            m_mf?MSHR_Status_str[ m_mf->get_status() ]:"???");
    if ( m_mf )
        ptx_print_insn( m_mf->get_pc(), fp );
    fprintf(fp,"\n");
}

void opndcoll_rfu_t::add_port( unsigned num_collector_units,
                               warp_inst_t **input_port,
                               warp_inst_t **output_port )
{
    m_num_ports++;
    m_num_collectors += num_collector_units;
    m_input.resize(m_num_ports);
    m_output.resize(m_num_ports);
    m_num_collector_units.resize(m_num_ports);
    m_input[m_num_ports-1]=input_port;
    m_output[m_num_ports-1]=output_port;
    m_num_collector_units[m_num_ports-1]=num_collector_units;
}

void opndcoll_rfu_t::init( unsigned num_banks, shader_core_ctx *shader )
{
   m_shader=shader;
   m_arbiter.init(m_num_collectors,num_banks);
   for( unsigned n=0; n<m_num_ports;n++ ) 
       m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
   m_num_banks = num_banks;
   m_bank_warp_shift = 0; 
   m_warp_size = shader->get_config()->warp_size;
   m_bank_warp_shift = (unsigned)(int) (log(m_warp_size+0.5) / log(2.0));
   assert( (m_bank_warp_shift == 5) || (m_warp_size != 32) );

   m_cu = new collector_unit_t[m_num_collectors];

   unsigned c=0;
   for( unsigned n=0; n<m_num_ports;n++ ) {
       for( unsigned j=0; j<m_num_collector_units[n]; j++, c++) {
          m_cu[c].init(c,m_output[n],num_banks,m_bank_warp_shift,shader->get_config(),this);
          m_free_cu[m_output[n]].push_back(&m_cu[c]);
          m_dispatch_units[m_output[n]].add_cu(&m_cu[c]);
       }
   }
}

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift)
{
   int bank = regnum;
   if (bank_warp_shift)
      bank += wid;
   return bank % num_banks;
}

bool opndcoll_rfu_t::writeback( const warp_inst_t &inst )
{
   assert( !inst.empty() );
   std::list<unsigned> regs = m_shader->get_regs_written(inst);
   std::list<unsigned>::iterator r;
   unsigned n=0;
   for( r=regs.begin(); r!=regs.end();r++,n++ ) {
      unsigned reg = *r;
      unsigned bank = register_bank(reg,inst.warp_id(),m_num_banks,m_bank_warp_shift);
      if( m_arbiter.bank_idle(bank) ) {
          m_arbiter.allocate_bank_for_write(bank,op_t(&inst,reg,m_num_banks,m_bank_warp_shift));
      } else {
          return false;
      }
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

void opndcoll_rfu_t::allocate_cu( unsigned port_num )
{
   if( !(*m_input[port_num])->empty() ) {
      warp_inst_t **port = m_output[port_num];
      if( !m_free_cu[port].empty() ) {
         collector_unit_t *cu = m_free_cu[port].back();
         m_free_cu[port].pop_back();
         cu->allocate(*m_input[port_num]);
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

bool opndcoll_rfu_t::collector_unit_t::ready() const 
{ 
   return (!m_free) && m_not_ready.none() && (*m_port)->empty(); 
}

void opndcoll_rfu_t::collector_unit_t::dump(FILE *fp, const shader_core_ctx *shader ) const
{
   if( m_free ) {
      fprintf(fp,"    <free>\n");
   } else {
      m_warp->print(fp);
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
                                             const shader_core_config *config,
                                             opndcoll_rfu_t *rfu ) 
{ 
   m_rfu=rfu;
   m_cuid=n; 
   m_port=port; 
   m_num_banks=num_banks;
   assert(m_warp==NULL); 
   m_warp = new warp_inst_t(config);
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

simt_core_cluster::simt_core_cluster( class gpgpu_sim *gpu, 
                                      unsigned cluster_id, 
                                      struct shader_core_config *config, 
                                      struct shader_core_stats *stats )
{
    m_cta_issue_next_core=0;
    m_cluster_id=cluster_id;
    m_gpu = gpu;
    m_config = config;
    m_stats = stats;
    m_core = new shader_core_ctx*[ config->n_simt_cores_per_cluster ];
    for( unsigned i=0; i < config->n_simt_cores_per_cluster; i++ ) 
        m_core[i] = new shader_core_ctx(gpu,this,cid_to_sid(i),m_cluster_id,config,stats);
}

void simt_core_cluster::core_cycle()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->cycle();
}

void simt_core_cluster::reinit()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->reinit(0,m_config->n_thread_per_shader,true);
}

unsigned simt_core_cluster::max_cta( class function_info *kernel )
{
    return m_config->n_simt_cores_per_cluster * m_core[0]->max_cta(kernel);
}

int simt_core_cluster::get_not_completed() const
{
    unsigned not_completed=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        not_completed += m_core[i]->get_not_completed();
    return not_completed;
}

unsigned simt_core_cluster::get_n_active_cta() const
{
    unsigned n=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        n += m_core[i]->get_n_active_cta();
    return n;
}

void simt_core_cluster::issue_block2core( class kernel_info_t &kernel )
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned core = (i+m_cta_issue_next_core)%m_config->n_simt_cores_per_cluster;
        if( m_core[core]->get_n_active_cta() < m_core[core]->max_cta(kernel.entry()) ) {
            m_core[core]->issue_block2core(kernel);
            break;
        }
    }
}

void simt_core_cluster::cache_flush()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->cache_flush();
}

bool simt_core_cluster::icnt_injection_buffer_full(new_addr_type addr, int bsize, bool write )
{
    //requests should be single always now
    int rsize = bsize;
    //maintain similar functionality with fq_push, if its a read, bsize is the load size, not the request's size
    if (!write) 
        rsize = READ_PACKET_SIZE;
    return ! ::icnt_has_buffer(m_cluster_id, rsize);
}

void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf)
{
    // stats
    if (mf->get_is_write()) m_stats->made_write_mfs++;
    else m_stats->made_read_mfs++;
    switch (mf->get_mem_acc()) {
    case CONST_ACC_R: m_stats->gpgpu_n_mem_const++; break;
    case TEXTURE_ACC_R: m_stats->gpgpu_n_mem_texture++; break;
    case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++; break;
    case GLOBAL_ACC_W: m_stats->gpgpu_n_mem_write_global++; break;
    case LOCAL_ACC_R: m_stats->gpgpu_n_mem_read_local++; break;
    case LOCAL_ACC_W: m_stats->gpgpu_n_mem_write_local++; break;
    case INST_ACC_R: m_stats->gpgpu_n_mem_read_inst++; break;
    default: assert(0);
    }

   unsigned destination = mf->get_tlx_addr().chip;
   mf->set_status(IN_ICNT2MEM,MR_ICNT_PUSHED,gpu_sim_cycle+gpu_tot_sim_cycle);
   if (!mf->get_is_write()) {
      mf->set_type(RD_REQ);
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->get_ctrl_size() );
   } else {
      mf->set_type(WT_REQ);
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->size());
      //gpgpu_n_sent_writes++;
   }
}

void simt_core_cluster::icnt_eject_response_packet(class mem_fetch * mf)
{
    assert( mf->get_tpc() == m_cluster_id );
    m_response_fifo.push_back(mf);
}

void simt_core_cluster::icnt_cycle()
{
    if( !m_response_fifo.empty() ) {
        mem_fetch *mf = m_response_fifo.front();
        unsigned cid = sid_to_cid(mf->get_sid());
        if( mf->get_mem_acc() == INST_ACC_R ) {
            // instruction fetch response
            if( !m_core[cid]->fetch_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_core[cid]->accept_fetch_response(mf);
            }
        } else {
            // data response
            if( !m_core[cid]->ldst_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_core[cid]->accept_ldst_unit_response(mf);
            }
        }
    }
    if( m_response_fifo.size() < m_config->n_simt_ejection_buffer_size ) {
        mem_fetch *mf = (mem_fetch*) ::icnt_pop(m_cluster_id);
        if (!mf) 
            return;
        assert(mf->get_tpc() == m_cluster_id);
        assert(mf->get_type() == REPLY_DATA);
        mf->set_status(IN_CLUSTER2SHADER,MR_2SH_FQ_POP,gpu_sim_cycle+gpu_tot_sim_cycle);
        //m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
        m_response_fifo.push_back(mf);
    }
}

void simt_core_cluster::mem_instruction_stats(class warp_inst_t &inst)
{
    m_gpu->mem_instruction_stats(inst);
}

void simt_core_cluster::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
    unsigned cid = sid_to_cid(sid);
    m_core[cid]->get_pdom_stack_top_info(tid,pc,rpc);
}

void simt_core_cluster::display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask )
{
    m_core[sid_to_cid(sid)]->display_pipeline(fout,print_mem,mask);
}
