/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Dan O'Connor, Henry Wong and the 
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

#ifndef ptx_sim_h_INCLUDED
#define ptx_sim_h_INCLUDED

#include <stdlib.h>

#include "dram_callback.h"
#include "../abstract_hardware_model.h"


struct dim3 {
   unsigned int x, y, z;
};

struct gpgpu_ptx_sim_arg {
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
   struct gpgpu_ptx_sim_arg *m_next;
};

//Holds properties of the kernel (Kernel's resource use). These will be zero if 
//the ptxinfo file is not present.
struct gpgpu_ptx_sim_kernel_info {
   int lmem;
   int smem;
   int cmem;
   int regs;
};

#include <assert.h>
#include "opcodes.h"

#ifdef __cplusplus

   #include <string>
   #include <map>
   #include <set>
   #include <list>
   #include <unordered_map>

#include "memory.h"

struct param_t {
   const void *pdata;
   int type;
   size_t size;
   size_t offset;
};

union ptx_reg_t {
   ptx_reg_t() {
      bits.ms = 0;
      bits.ls = 0;
   }
   ptx_reg_t(unsigned x) 
   {
      bits.ms = 0;
      bits.ls = 0;
      u32 = x;
   }
   operator unsigned int() { return u32;}
   operator unsigned short() { return u16;}
   operator unsigned char() { return u8;}
   operator unsigned long long() { return u64;}

   void mask_and( unsigned ms, unsigned ls )
   {
      bits.ms &= ms;
      bits.ls &= ls;
   }

   void mask_or( unsigned ms, unsigned ls )
   {
      bits.ms |= ms;
      bits.ls |= ls;
   }
   int get_bit( unsigned bit )
   {
      if ( bit < 32 )
         return(bits.ls >> bit) & 1;
      else
         return(bits.ms >> (bit-32)) & 1;
   }

   signed char       s8;
   signed short      s16;
   signed int        s32;
   signed long long  s64;
   unsigned char     u8;
   unsigned short    u16;
   unsigned int      u32;
   unsigned long long   u64;
   float             f16; 
   float          f32;
   double            f64;
   struct {
      unsigned ls;
      unsigned ms;
   } bits;
   unsigned       pred : 1;

};

class ptx_instruction;
class operand_info;
class symbol_table;
class function_info;
class ptx_thread_info;

class ptx_cta_info {
public:
   ptx_cta_info( unsigned sm_idx );
   void add_thread( ptx_thread_info *thd );
   void add_to_barrier( ptx_thread_info *thd );
   bool all_at_barrier() const;
   void release_barrier();
   unsigned num_threads() const;
   void check_cta_thread_status_and_reset();
   void assert_barrier_empty( bool called_from_delete_threads = false ) const;
   void register_thread_exit( ptx_thread_info *thd );
   void register_deleted_thread( ptx_thread_info *thd );
   unsigned get_sm_idx() const;

private:
   unsigned long long         m_uid;
   unsigned                m_sm_idx;
   std::set<ptx_thread_info*>    m_threads_in_cta;
   std::set<ptx_thread_info*>    m_threads_waiting_at_barrier;
   std::set<ptx_thread_info*>  m_threads_that_have_exited;
   std::set<ptx_thread_info*>  m_dangling_pointers;
};

class symbol;

struct stack_entry {
   stack_entry() {
      m_symbol_table=NULL;
      m_func_info=NULL;
      m_PC=0;
      m_RPC=-1;
      m_return_var_src = NULL;
      m_return_var_dst = NULL;
      m_call_uid = 0;
      m_valid = false;
   }
   stack_entry( symbol_table *s, function_info *f, unsigned pc, unsigned rpc, const symbol *return_var_src, const symbol *return_var_dst, unsigned call_uid )
   {
      m_symbol_table=s;
      m_func_info=f;
      m_PC=pc;
      m_RPC=rpc;
      m_return_var_src = return_var_src;
      m_return_var_dst = return_var_dst;
      m_call_uid = call_uid;
      m_valid = true;
   }

   bool m_valid;
   symbol_table  *m_symbol_table;
   function_info *m_func_info;
   unsigned       m_PC;
   unsigned       m_RPC;
   const symbol  *m_return_var_src;
   const symbol  *m_return_var_dst;
   unsigned       m_call_uid;
};

class ptx_version {
public:
      ptx_version()
      {
         m_valid = false;
         m_ptx_version = 0;
         m_ptx_extensions = 0;
      }
      ptx_version(float ver, unsigned extensions)
      {
         m_valid = true;
         m_ptx_version = ver;
         m_ptx_extensions = extensions;
      }
      float ver() const { assert(m_valid); return m_ptx_version; }
      float extensions() const { assert(m_valid); return m_ptx_extensions; }
private:
      bool     m_valid;
      float    m_ptx_version;
      unsigned m_ptx_extensions;
};

class ptx_thread_info {
public:
   ~ptx_thread_info();
   ptx_thread_info();

   const ptx_version &get_ptx_version() const;
   ptx_reg_t get_operand_value( const symbol *reg );
   ptx_reg_t get_operand_value( const operand_info &op );
   void set_operand_value( const operand_info &dst, const ptx_reg_t &data );
   void set_operand_value( const symbol *dst, const ptx_reg_t &data );
   void get_vector_operand_values( const operand_info &op, ptx_reg_t* ptx_regs, unsigned num_elements );
   void set_vector_operand_values( const operand_info &dst, 
                                   const ptx_reg_t &data1, 
                                   const ptx_reg_t &data2, 
                                   const ptx_reg_t &data3, 
                                   const ptx_reg_t &data4, 
                                   unsigned num_elements );

   function_info *func_info()
   {
      return m_func_info; 
   }
   void print_insn( unsigned pc, FILE * fp ) const;
   void set_info( symbol_table *symtab, function_info *func );
   unsigned get_uid() const
   {
      return m_uid;
   }

   dim3 get_ctaid() const
   {
      dim3 r;
      r.x = m_ctaid[0];
      r.y = m_ctaid[1];
      r.z = m_ctaid[2];
      return r;
   }
   dim3 get_tid() const
   {
      dim3 r;
      r.x = m_tid[0];
      r.y = m_tid[1];
      r.z = m_tid[2];
      return r;
   }
   unsigned get_hw_tid() const { return m_hw_tid;}
   unsigned get_hw_ctaid() const { return m_hw_ctaid;}
   unsigned get_hw_wid() const { return m_hw_wid;}
   unsigned get_hw_sid() const { return m_hw_sid;}
   void set_hw_tid(unsigned tid) { m_hw_tid=tid;}
   void set_hw_wid(unsigned wid) { m_hw_wid=wid;}
   void set_hw_sid(unsigned sid) { m_hw_sid=sid;}
   void set_hw_ctaid(unsigned cta_id) { m_hw_ctaid=cta_id;}
   void set_core(core_t *core) { m_core = core; }
   core_t *get_core() { return m_core; }

   unsigned get_icount() const { return m_icount;}
   void set_valid() { m_valid = true;}
   addr_t last_eaddr() const { return m_last_effective_address;}
   unsigned last_space() const { return m_last_memory_space;}
   dram_callback_t last_callback() const { return m_last_dram_callback;}
   void set_at_barrier( int barrier_num ) 
   { 
      m_barrier_num = barrier_num;
      m_at_barrier = true; 
      m_cta_info->add_to_barrier(this);
   }
   bool is_at_barrier() const { return m_at_barrier;}
   bool all_at_barrier() const { return m_cta_info->all_at_barrier();}
   unsigned long long get_cta_uid() { return m_cta_info->get_sm_idx();}
   void clear_barrier( ) 
   { 
      m_barrier_num = -1;
      m_at_barrier = false; 
   }
   void release_barrier() { m_cta_info->release_barrier();}

   void set_single_thread_single_block()
   {
      m_ntid[0] = 1;
      m_ntid[1] = 1;
      m_ntid[2] = 1;
      m_ctaid[0] = 0;
      m_ctaid[1] = 0;
      m_ctaid[2] = 0;
      m_tid[0] = 0;
      m_tid[1] = 0;
      m_tid[2] = 0;
      m_nctaid[0] = 1;
      m_nctaid[1] = 1;
      m_nctaid[2] = 1;
      m_gridid = 0;
      m_valid = true;
   }
   void set_tid( int x, int y, int z)
   {
      m_tid[0] = x;
      m_tid[1] = y;
      m_tid[2] = z;
   }
   void set_ctaid( int x, int y, int z)
   {
      m_ctaid[0] = x;
      m_ctaid[1] = y;
      m_ctaid[2] = z;
   }
   void set_ntid( int x, int y, int z)
   {
      m_ntid[0] = x;
      m_ntid[1] = y;
      m_ntid[2] = z;
   }
   void set_nctaid( int x, int y, int z)
   {
      m_nctaid[0] = x;
      m_nctaid[1] = y;
      m_nctaid[2] = z;
   }

   unsigned get_builtin( int builtin_id, unsigned dim_mod ); 

   void set_done();
   bool is_done() { return m_thread_done;}
   unsigned donecycle() const { return m_cycle_done; }

   unsigned next_instr()
   {
      m_NPC = m_PC+1;   // increment to next instruction in case of no branch
      m_icount++;
      m_branch_taken = false;
      return m_PC;
   }
   bool branch_taken() const
   {
      return m_branch_taken;
   }
   unsigned get_pc() const
   {
      return m_PC;
   }
   void set_npc( unsigned npc )
   {
      m_NPC = npc;
   }
   void set_npc( const function_info *f );
   void callstack_push( unsigned npc, unsigned rpc, const symbol *return_var_src, const symbol *return_var_dst, unsigned call_uid );
   bool callstack_pop();
   void dump_callstack() const;
   std::string get_location() const;
   bool rpc_updated() const { return m_RPC_updated; }
   bool last_was_call() const { return m_last_was_call; }
   unsigned get_rpc() const { return m_RPC; }
   void clearRPC()
   {
      m_RPC = -1;
      m_RPC_updated = false;
      m_last_was_call = false;
   }
   unsigned get_return_PC()
   {
       return m_callstack.back().m_PC;
   }
   void update_pc()
   {
      m_PC = m_NPC;
   }
   void dump_regs();
   void dump_modifiedregs();
   void clear_modifiedregs() { m_debug_trace_regs_modified.clear();}
   function_info *get_finfo() { return m_func_info;   }

   void enable_debug_trace() { m_enable_debug_trace = true; }

public:
   addr_t         m_last_effective_address;
   bool        m_branch_taken;
   unsigned       m_last_memory_space;
   dram_callback_t   m_last_dram_callback; 
   memory_space   *m_shared_mem;
   memory_space   *m_local_mem;
   ptx_cta_info   *m_cta_info;
   ptx_reg_t m_last_set_operand_value;

private:
   unsigned m_uid;
   core_t *m_core;
   bool   m_valid;
   unsigned m_ntid[3];
   unsigned m_tid[3];
   unsigned m_nctaid[3];
   unsigned m_ctaid[3];
   unsigned m_gridid;
   bool m_thread_done;
   unsigned m_hw_sid;
   unsigned m_hw_tid;
   unsigned m_hw_wid;
   unsigned m_hw_ctaid;

   unsigned m_icount;
   unsigned m_PC;
   unsigned m_NPC;
   unsigned m_RPC;
   bool m_RPC_updated;
   bool m_last_was_call;
   unsigned m_cycle_done;

   int m_barrier_num;
   bool m_at_barrier;

   symbol_table  *m_symbol_table;
   function_info *m_func_info;

   std::list<stack_entry> m_callstack;

   // typedef std::unordered_map<std::string,ptx_reg_t> reg_map_t;
   typedef std::unordered_map<const symbol*,ptx_reg_t> reg_map_t;
   std::list<reg_map_t> m_regs;

   bool m_enable_debug_trace;
   reg_map_t m_debug_trace_regs_modified; // track the modified register for each executed insn
};

unsigned type_decode( unsigned type, size_t &size, int &t );

addr_t generic_to_local( unsigned smid, unsigned hwtid, addr_t addr );
addr_t generic_to_shared( unsigned smid, addr_t addr );
addr_t generic_to_global( addr_t addr );
addr_t local_to_generic( unsigned smid, unsigned hwtid, addr_t addr );
addr_t shared_to_generic( unsigned smid, addr_t addr );
addr_t global_to_generic( addr_t addr );
bool isspace_local( unsigned smid, unsigned hwtid, addr_t addr );
bool isspace_shared( unsigned smid, addr_t addr );
bool isspace_global( addr_t addr );
unsigned whichspace( addr_t addr );

#endif

#define MAX_REG_OPERANDS 8

#endif
