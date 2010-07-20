/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Ali Bakhoda, George L. Yuan, 
 * Dan O'Connor, and the University of British Columbia
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

#include "ptx_sim.h"
#include <string>
#include "ptx_ir.h"

void feature_not_implemented( const char *f );

std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;
unsigned long long g_ptx_cta_info_uid = 1;
extern int gpgpu_option_spread_blocks_across_cores;

ptx_cta_info::ptx_cta_info( unsigned sm_idx )
{
   assert( g_ptx_cta_info_sm_idx_used.find(sm_idx) == g_ptx_cta_info_sm_idx_used.end() );
   g_ptx_cta_info_sm_idx_used.insert(sm_idx);

   m_sm_idx = sm_idx;
   m_uid = g_ptx_cta_info_uid++;
}

void ptx_cta_info::add_thread( ptx_thread_info *thd )
{
   m_threads_in_cta.insert(thd);
}

void ptx_cta_info::add_to_barrier( ptx_thread_info *thd )
{
   m_threads_waiting_at_barrier.insert(thd);
}

void ptx_cta_info::release_barrier()
{
   assert( m_threads_waiting_at_barrier.size() == m_threads_in_cta.size() );
   m_threads_waiting_at_barrier.clear();
}

bool ptx_cta_info::all_at_barrier() const
{
   return(m_threads_waiting_at_barrier.size() == m_threads_in_cta.size());
}

unsigned ptx_cta_info::num_threads() const
{
   return m_threads_in_cta.size();
}

void ptx_cta_info::check_cta_thread_status_and_reset()
{
   bool fail = false;
   if ( m_threads_that_have_exited.size() != m_threads_in_cta.size() ) {
      printf("\n\n");
      printf("Execution error: Some threads still running in CTA during CTA reallocation! (1)\n");
      printf("   CTA uid = %Lu (sm_idx = %u) : %lu running out of %zu total\n", 
             m_uid, 
             m_sm_idx,
             (m_threads_in_cta.size() - m_threads_that_have_exited.size()), m_threads_in_cta.size() );
      printf("   These are the threads that are still running:\n");
      std::set<ptx_thread_info*>::iterator t_iter;
      for ( t_iter=m_threads_in_cta.begin(); t_iter != m_threads_in_cta.end(); ++t_iter ) {
         ptx_thread_info *t = *t_iter;
         if ( m_threads_that_have_exited.find(t) == m_threads_that_have_exited.end() ) {
            if ( m_dangling_pointers.find(t) != m_dangling_pointers.end() ) {
               printf("       <thread deleted>\n");
            } else {
               printf("       [done=%c] : ", (t->is_done()?'Y':'N') );
               t->print_insn( t->get_pc(), stdout );
               printf("\n");
            }
         }
      }
      printf("\n\n");
      fail = true;
   }
   assert_barrier_empty(true);
   if ( fail ) {
      abort();
   }

   bool fail2 = false;
   std::set<ptx_thread_info*>::iterator t_iter;
   for ( t_iter=m_threads_in_cta.begin(); t_iter != m_threads_in_cta.end(); ++t_iter ) {
      ptx_thread_info *t = *t_iter;
      if ( m_dangling_pointers.find(t) == m_dangling_pointers.end() ) {
         if ( !t->is_done() ) {
            if ( !fail2 ) {
               printf("Execution error: Some threads still running in CTA during CTA reallocation! (2)\n");
               printf("   CTA uid = %Lu (sm_idx = %u) :\n", m_uid, m_sm_idx );
               fail2 = true;
            }
            printf("       ");
            t->print_insn( t->get_pc(), stdout );
            printf("\n");
         }
      }
   }
   if ( fail2 ) {
      abort();
   }
   m_threads_in_cta.clear();
   m_threads_that_have_exited.clear();
   m_dangling_pointers.clear();
}

void ptx_cta_info::assert_barrier_empty( bool called_from_delete_threads ) const
{
   if ( !m_threads_waiting_at_barrier.empty() ) {
      printf( "\n\n" );
      printf( "Execution error: Some threads at barrier at %s:\n", 
              (called_from_delete_threads?"CTA re-init":"thread exit") );
      printf( "   CTA uid = %Lu (sm_idx = %u)\n", m_uid, m_sm_idx );
      std::set<ptx_thread_info*>::iterator b;
      for ( b=m_threads_waiting_at_barrier.begin(); b != m_threads_waiting_at_barrier.end(); ++b ) {
         ptx_thread_info *t = *b;
         if ( m_dangling_pointers.find(t) != m_dangling_pointers.end() ) {
            printf("       <thread deleted>\n");
         } else {
            printf("       ");
            t->print_insn( t->get_pc(), stdout );
            printf("\n");
         }
      }
      printf("\n\n");
      abort();
   }
}

void ptx_cta_info::register_thread_exit( ptx_thread_info *thd )
{
   assert( m_threads_that_have_exited.find(thd) == m_threads_that_have_exited.end() );
   m_threads_that_have_exited.insert(thd);
}

void ptx_cta_info::register_deleted_thread( ptx_thread_info *thd )
{
   m_dangling_pointers.insert(thd);
}

unsigned ptx_cta_info::get_sm_idx() const
{
   return m_sm_idx;
}

unsigned g_ptx_thread_info_uid_next=1;
unsigned g_ptx_thread_info_delete_count=0;

ptx_thread_info::~ptx_thread_info()
{
   g_ptx_thread_info_delete_count++;
}

ptx_thread_info::ptx_thread_info()
{
   m_uid = g_ptx_thread_info_uid_next++;
   m_core = NULL;
   m_barrier_num = -1;
   m_at_barrier = false;
   m_valid = false;
   m_gridid = 0;
   m_thread_done = false;
   m_cycle_done = 0;
   m_PC=0;
   m_icount = 0;
   m_last_effective_address = 0;
   m_last_memory_space = undefined_space; 
   m_branch_taken = 0;
   m_shared_mem = NULL;
   m_cta_info = NULL;
   m_local_mem = NULL;
   m_symbol_table = NULL;
   m_func_info = NULL;
   m_hw_tid = -1;
   m_hw_wid = -1;
   m_hw_sid = -1;
   m_last_dram_callback.function = NULL;
   m_last_dram_callback.instruction = NULL;
   m_regs.push_back( reg_map_t() );
   m_callstack.push_back( stack_entry() );
   m_RPC = -1;
   m_RPC_updated = false;
   m_last_was_call = false;
   m_enable_debug_trace = false;
   m_local_mem_stack_pointer = 0;
}

const ptx_version &ptx_thread_info::get_ptx_version() const 
{ 
   return m_func_info->get_ptx_version(); 
}

extern unsigned long long  gpu_sim_cycle;

void ptx_thread_info::set_done() 
{
   assert( !m_at_barrier );
   m_thread_done = true;
   m_cycle_done = gpu_sim_cycle; 
}

extern unsigned long long  gpu_sim_cycle;
extern signed long long gpu_tot_sim_cycle;

unsigned ptx_thread_info::get_builtin( int builtin_id, unsigned dim_mod ) 
{
   assert( m_valid );
   switch ((builtin_id&0xFFFF)) {
   case CLOCK_REG:
      return (unsigned)(gpu_sim_cycle + gpu_tot_sim_cycle);
   case CLOCK64_REG:
      abort(); // change return value to unsigned long long?
      return gpu_sim_cycle + gpu_tot_sim_cycle;
   case CTAID_REG:
      assert( dim_mod < 3 );
      return m_ctaid[dim_mod];
   case ENVREG_REG: feature_not_implemented( "%envreg" ); return 0;
   case GRIDID_REG:
      return m_gridid;
   case LANEID_REG: feature_not_implemented( "%laneid" ); return 0;
   case LANEMASK_EQ_REG: feature_not_implemented( "%lanemask_eq" ); return 0;
   case LANEMASK_LE_REG: feature_not_implemented( "%lanemask_le" ); return 0;
   case LANEMASK_LT_REG: feature_not_implemented( "%lanemask_lt" ); return 0;
   case LANEMASK_GE_REG: feature_not_implemented( "%lanemask_ge" ); return 0;
   case LANEMASK_GT_REG: feature_not_implemented( "%lanemask_gt" ); return 0;
   case NCTAID_REG:
      assert( dim_mod < 3 );
      return m_nctaid[dim_mod];
   case NTID_REG:
      assert( dim_mod < 3 );
      return m_ntid[dim_mod];
   case NWARPID_REG: feature_not_implemented( "%nwarpid" ); return 0;
   case PM_REG: feature_not_implemented( "%pm" ); return 0;
   case SMID_REG: feature_not_implemented( "%smid" ); return 0;
   case TID_REG:
      assert( dim_mod < 3 );
      return m_tid[dim_mod];
   case WARPSZ_REG: feature_not_implemented( "WARP_SZ" ); return 0;
   default:
      assert(0);
   }
   return 0;
}

void ptx_thread_info::set_info( symbol_table *symtab, function_info *func ) 
{
  m_symbol_table = symtab;
  m_func_info = func;
  m_PC = func->get_start_PC();
}

void ptx_thread_info::print_insn( unsigned pc, FILE * fp ) const
{
   m_func_info->print_insn(pc,fp);
}

static void print_reg( std::string name, ptx_reg_t value )
{
   const symbol *sym = g_current_symbol_table->lookup(name.c_str());
   printf("  %8s   ", name.c_str() );
   if( sym == NULL ) {
      printf("<unknown type> 0x%llx\n", (unsigned long long ) value.u64 );
      return;
   }
   const type_info *t = sym->type();
   if( t == NULL ) {
      printf("<unknown type> 0x%llx\n", (unsigned long long ) value.u64 );
      return;
   }
   type_info_key ti = t->get_key();

   switch ( ti.scalar_type() ) {
   case S8_TYPE:  printf(".s8  %d\n", value.s8 );  break;
   case S16_TYPE: printf(".s16 %d\n", value.s16 ); break;
   case S32_TYPE: printf(".s32 %d\n", value.s32 ); break;
   case S64_TYPE: printf(".s64 %Ld\n", value.s64 ); break;
   case U8_TYPE:  printf(".u8  0x%02x\n", (unsigned) value.u8 );  break;
   case U16_TYPE: printf(".u16 0x%04x\n", (unsigned) value.u16 ); break;
   case U32_TYPE: printf(".u32 0x%08x\n", (unsigned) value.u32 ); break;
   case U64_TYPE: printf(".u64 0x%llx\n", value.u64 ); break;
   case F16_TYPE: printf(".f16 %f [0x%04x]\n",  value.f16, (unsigned) value.u16 ); break;
   case F32_TYPE: printf(".f32 %.15lf [0x%08x]\n",  value.f32, value.u32 ); break;
   case F64_TYPE: printf(".f64 %.15le [0x%016llx]\n", value.f64, value.u64 ); break;
   case B8_TYPE:  printf(".b8  0x%02x\n",   (unsigned) value.u8 );  break;
   case B16_TYPE: printf(".b16 0x%04x\n",   (unsigned) value.u16 ); break;
   case B32_TYPE: printf(".b32 0x%08x\n", (unsigned) value.u32 ); break;
   case B64_TYPE: printf(".b64 0x%llx\n",    (unsigned long long ) value.u64 ); break;
   case PRED_TYPE: printf(".pred %u\n",     (unsigned) value.pred ); break;
   default: 
      printf( "non-scalar type\n" );
      break;
   }
}

void ptx_thread_info::callstack_push( unsigned pc, unsigned rpc, const symbol *return_var_src, const symbol *return_var_dst, unsigned call_uid )
{
   m_RPC = -1;
   m_RPC_updated = true;
   m_last_was_call = true;
   assert( m_func_info != NULL );
   m_callstack.push_back( stack_entry(m_symbol_table,m_func_info,pc,rpc,return_var_src,return_var_dst,call_uid) );
   m_regs.push_back( reg_map_t() );
   m_local_mem_stack_pointer += m_func_info->local_mem_framesize(); 
}

#define POST_DOMINATOR 1 /* must match definition in shader.h */
extern int gpgpu_simd_model;
extern ptx_reg_t get_operand_value( const symbol *reg );
extern void set_operand_value( const symbol *dst, const ptx_reg_t &data );

bool ptx_thread_info::callstack_pop()
{
   const symbol *rv_src = m_callstack.back().m_return_var_src;
   const symbol *rv_dst = m_callstack.back().m_return_var_dst;
   assert( !((rv_src != NULL) ^ (rv_dst != NULL)) ); // ensure caller and callee agree on whether there is a return value

   // read return value from callee frame
   arg_buffer_t buffer;
   if( rv_src != NULL ) 
      buffer = copy_arg_to_buffer(this, operand_info(rv_src), rv_dst );

   m_symbol_table = m_callstack.back().m_symbol_table;
   m_NPC = m_callstack.back().m_PC;
   m_RPC_updated = true;
   m_last_was_call = false;
   m_RPC = m_callstack.back().m_RPC;
   if( m_callstack.back().m_func_info ) {
      assert( m_local_mem_stack_pointer >= m_callstack.back().m_func_info->local_mem_framesize() );
      m_local_mem_stack_pointer -= m_func_info->local_mem_framesize(); 
   }
   m_func_info = m_callstack.back().m_func_info;
   m_callstack.pop_back();
   m_regs.pop_back();

   // write return value into caller frame
   if( rv_dst != NULL ) 
      copy_buffer_to_frame(this, buffer);

   return m_callstack.empty();
}

void ptx_thread_info::dump_callstack() const
{
   std::list<stack_entry>::const_iterator c=m_callstack.begin();
   std::list<reg_map_t>::const_iterator r=m_regs.begin();

   printf("\n\n");
   printf("Call stack for thread uid = %u (sc=%u, hwtid=%u)\n", m_uid, m_hw_sid, m_hw_tid );
   while( c != m_callstack.end() && r != m_regs.end() ) {
      const stack_entry &c_e = *c;
      const reg_map_t &regs = *r;
      if( !c_e.m_valid ) {
         printf("  <entry>                              #regs = %zu\n", regs.size() );
      } else {
         printf("  %20s  PC=%3u RV= (callee=\'%s\',caller=\'%s\') #regs = %zu\n", 
                c_e.m_func_info->get_name().c_str(), c_e.m_PC, 
                c_e.m_return_var_src->name().c_str(), 
                c_e.m_return_var_dst->name().c_str(), 
                regs.size() );
      }
      c++;
      r++;
   }
   if( c != m_callstack.end() || r != m_regs.end() ) {
      printf("  *** mismatch in m_regs and m_callstack sizes ***\n" );
   }
   printf("\n\n");
}

std::string ptx_thread_info::get_location() const
{
   const ptx_instruction *pI = m_func_info->get_instruction(m_PC);
   char buf[1024];
   snprintf(buf,1024,"%s:%u", pI->source_file(), pI->source_line() );
   return std::string(buf);
}

void ptx_thread_info::dump_regs()
{
   printf("Register File Contents:\n");
   reg_map_t::const_iterator r;
   for ( r=m_regs.back().begin(); r != m_regs.back().end(); ++r ) {
      std::string name = r->first->name();
      ptx_reg_t value = r->second;
      print_reg(name,value);

   }
}

void ptx_thread_info::dump_modifiedregs()
{
   if( m_debug_trace_regs_modified.empty() ) 
      return;
   printf("Modified Registers:\n");
   reg_map_t::const_iterator r;
   for ( r=m_debug_trace_regs_modified.begin(); r != m_debug_trace_regs_modified.end(); ++r ) {
      std::string name = r->first->name();
      ptx_reg_t value = r->second;
      print_reg(name,value);
   }
}

void ptx_thread_info::set_npc( const function_info *f )
{
   m_NPC = f->get_start_PC();
   m_func_info = const_cast<function_info*>( f );
   m_symbol_table = m_func_info->get_symtab();
}

void feature_not_implemented( const char *f ) 
{
   printf("GPGPU-Sim: feature '%s' not supported\n", f );
   abort();
}
