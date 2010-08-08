/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Ali Bakhoda, Joey Ting, Dan O'Connor, 
 * Clive Lin, George L. Yuan, Wilson W. L. Fung and the 
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

#include "instructions.h"
#include "ptx_ir.h"
#include "opcodes.h"
#include "ptx_sim.h"
#include "ptx.tab.h"
#include "dram_callback.h"
#include <stdlib.h>
#include <math.h>
#include <fenv.h>

#include "cuda-math.h"
#include "../abstract_hardware_model.h"
#include "ptx_loader.h"

#include <stdarg.h>

unsigned g_num_ptx_inst_uid=0;
unsigned cudasim_n_tex_insn=0;

const char *g_opcode_string[NUM_OPCODES] = {
#define OP_DEF(OP,FUNC,STR,DST,CLASSIFICATION) STR,
#include "opcodes.def"
#undef OP_DEF
};

extern std::map<unsigned,std::string> g_ptx_token_decode;
extern std::map<struct textureReference*,struct cudaArray*> TextureToArrayMap; // texture bindings
extern std::map<struct textureReference*,struct textureInfo*> TextureToInfoMap; // texture bindings
extern std::map<std::string, struct textureReference*> NameToTextureMap;

void inst_not_implemented( const ptx_instruction * pI ) ;
unsigned unfound_register_warned = 0;

ptx_reg_t ptx_thread_info::get_operand_value( const symbol *reg )
{
   // assume that the given symbol is a register and try to find it in the register hash map
   reg_map_t::iterator regs_iter = m_regs.back().find(reg);
   if (regs_iter == m_regs.back().end()) {
      assert( reg->type()->get_key().is_reg() );
      const std::string &name = reg->name();
      unsigned call_uid = m_callstack.back().m_call_uid;
      ptx_reg_t uninit_reg;
      uninit_reg.u32 = 0xDEADBEEF;
      set_operand_value(reg, uninit_reg); // give it a value since we are going to warn the user anyway
      std::string file_loc = get_location();
      if( !unfound_register_warned ) {
          printf("GPGPU-Sim PTX: WARNING (%s) ** reading undefined register \'%s\' (cuid:%u). Setting to 0XDEADBEEF.\n",
                 file_loc.c_str(), name.c_str(), call_uid );
          unfound_register_warned = 1;
      }
      regs_iter = m_regs.back().find(reg);
   }
   return regs_iter->second;
}

ptx_reg_t ptx_thread_info::get_operand_value( const operand_info &op )
{
   ptx_reg_t result;
   const char *name = NULL;
   if ( op.is_reg() ) {
      result = get_operand_value( op.get_symbol() );
   } else if ( op.is_builtin()) {
      result = get_builtin( op.get_int(), op.get_addr_offset() );
   } else if ( op.is_memory_operand() ) {
      // a few options here...
      const symbol *sym = op.get_symbol();
      const type_info *type = sym->type();
      const type_info_key &info = type->get_key();

      if ( info.is_reg() ) {
         name = op.name().c_str();
         reg_map_t::iterator regs_iter = m_regs.back().find(sym);
         assert( regs_iter != m_regs.back().end() );
         ptx_reg_t baseaddr = regs_iter->second;
         result.u64 = baseaddr.u64 + op.get_addr_offset(); 
      } else if ( info.is_param_kernel() ) {
         result = sym->get_address() + op.get_addr_offset();
      } else if ( info.is_param_local() ) {
         result = sym->get_address() + op.get_addr_offset();
      } else if ( info.is_global() ) {
         assert( op.get_addr_offset() == 0 );
         result = sym->get_address();
      } else if ( info.is_local() ) {
         result = sym->get_address() + op.get_addr_offset();
      } else if ( info.is_const() ) {
         result = sym->get_address() + op.get_addr_offset();
      } else if ( op.is_shared() ) {
         result = op.get_symbol()->get_address() + op.get_addr_offset();
      } else {
         name = op.name().c_str();
         printf("GPGPU-Sim PTX: ERROR ** get_operand_value : unknown memory operand type for %s\n", name );
         abort();
      }

   } else if ( op.is_literal() ) {
      result = op.get_literal_value();
   } else if ( op.is_label() ) {
      result = op.get_symbol()->get_address();
   } else if ( op.is_shared() ) {
      result = op.get_symbol()->get_address();
   } else if ( op.is_const() ) {
      result = op.get_symbol()->get_address();
   } else if ( op.is_global() ) {
      result = op.get_symbol()->get_address();
   } else if ( op.is_local() ) {
      result = op.get_symbol()->get_address();
   } else {
      name = op.name().c_str();
      printf("GPGPU-Sim PTX: ERROR ** get_operand_value : unknown operand type for %s\n", name );
      assert(0);
   }

   return result;
}

unsigned get_operand_nbits( const operand_info &op )
{
   if ( op.is_reg() ) {
      const symbol *sym = op.get_symbol();
      const type_info *typ = sym->type();
      type_info_key t = typ->get_key();
      switch( t.scalar_type() ) {
      case PRED_TYPE: 
         return 1;
      case B8_TYPE: case S8_TYPE: case U8_TYPE:
         return 8;
      case S16_TYPE: case U16_TYPE: case F16_TYPE: case B16_TYPE:
         return 16;
      case S32_TYPE: case U32_TYPE: case F32_TYPE: case B32_TYPE:
         return 32;
      case S64_TYPE: case U64_TYPE: case F64_TYPE: case B64_TYPE:
         return 64;
      default:
         printf("ERROR: unknown register type\n");
         fflush(stdout);
         abort();
      }
   } else {
      printf("ERROR: Need to implement get_operand_nbits() for currently unsupported operand_info type\n");
      fflush(stdout);
      abort();
   }

   return 0;
}

void ptx_thread_info::get_vector_operand_values( const operand_info &op, ptx_reg_t* ptx_regs, unsigned num_elements )
{
   assert( op.is_vector() );
   assert( num_elements <= 4 ); // max 4 elements in a vector

   for (int idx = num_elements - 1; idx >= 0; --idx) {
      const symbol *sym = NULL;
      sym = op.vec_symbol(idx);
      reg_map_t::iterator reg_iter = m_regs.back().find(sym);
      assert( reg_iter != m_regs.back().end() );
      ptx_regs[idx] = reg_iter->second;
   }
}

void sign_extend( ptx_reg_t &data, unsigned src_size, const operand_info &dst )
{
   if( !dst.is_reg() )
      return;
   unsigned dst_size = get_operand_nbits( dst );
   if( src_size >= dst_size ) 
      return;
   // src_size < dst_size
   unsigned long long mask = 1;
   mask <<= (src_size-1);
   if( (mask & data.u64) == 0 ) {
      // no need to sign extend
      return;
   }
   // need to sign extend
   mask = 1;
   mask <<= dst_size-src_size;
   mask -= 1;
   mask <<= src_size;
   data.u64 |= mask;
}

void ptx_thread_info::set_operand_value( const operand_info &dst, const ptx_reg_t &data )
{
   m_regs.back()[ dst.get_symbol() ] = data;
   if (m_enable_debug_trace ) {
      m_debug_trace_regs_modified[ dst.get_symbol() ] = data;
   }
   m_last_set_operand_value = data;
}

void ptx_thread_info::set_operand_value( const symbol *dst, const ptx_reg_t &data )
{
   m_regs.back()[ dst ] = data;
   if (m_enable_debug_trace ) {
      m_debug_trace_regs_modified[ dst ] = data;
   }
   m_last_set_operand_value = data;
}

void ptx_thread_info::set_vector_operand_values( const operand_info &dst, 
                                                 const ptx_reg_t &data1, 
                                                 const ptx_reg_t &data2, 
                                                 const ptx_reg_t &data3, 
                                                 const ptx_reg_t &data4, 
                                                 unsigned num_elements )
{
   set_operand_value(dst.vec_symbol(0), data1);
   set_operand_value(dst.vec_symbol(1), data2);
   if (num_elements > 2) {
      set_operand_value(dst.vec_symbol(2), data3);
      if (num_elements > 3) {
         set_operand_value(dst.vec_symbol(3), data4);
      }
   }

   m_last_set_operand_value = data1;
}

#define my_abs(a) (((a)<0)?(-a):(a))

#define MY_MAX_I(a,b) (a > b) ? a : b
#define MY_MAX_F(a,b) isNaN(a) ? b : isNaN(b) ? a : (a > b) ? a : b

#define MY_MIN_I(a,b) (a < b) ? a : b
#define MY_MIN_F(a,b) isNaN(a) ? b : isNaN(b) ? a : (a < b) ? a : b

#define MY_INC_I(a,b) (a >= b) ? 0 : a+1
#define MY_DEC_I(a,b) ((a == 0) || (a > b)) ? b : a-1

#define MY_CAS_I(a,b,c) (a == b) ? c : a

#define MY_EXCH(a,b) b

void abs_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case S16_TYPE: d.s16 = my_abs(a.s16); break;
   case S32_TYPE: d.s32 = my_abs(a.s32); break;
   case S64_TYPE: d.s64 = my_abs(a.s64); break;
   case F32_TYPE: d.f32 = my_abs(a.f32); break;
   case F64_TYPE: d.f64 = my_abs(a.f64); break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
   }

   thread->set_operand_value(dst,d);
}

void add_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;

   const operand_info &dst  = pI->dst();  //get operand info of sources and destination 
   const operand_info &src1 = pI->src1(); //use them to determine that they are of type 'register'
   const operand_info &src2 = pI->src2();
   src1_data = thread->get_operand_value(src1); //get values from the operand infos
   src2_data = thread->get_operand_value(src2);

   unsigned rounding_mode = pI->rounding_mode();
   int orig_rm = fegetround();
   switch ( rounding_mode ) {
   case RN_OPTION: break;
   case RZ_OPTION: fesetround( FE_TOWARDZERO ); break;
   default: assert(0); break;
   }

   unsigned to_type = pI->get_type();

   switch ( to_type ) {
   case S8_TYPE:
   case S16_TYPE:
   case S32_TYPE:
   case S64_TYPE:
   case U8_TYPE:
   case U16_TYPE:
   case U32_TYPE:
   case U64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64; break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE: data.f32 = src1_data.f32 + src2_data.f32; break;
   case F64_TYPE: data.f64 = src1_data.f64 + src2_data.f64; break;
   default: assert(0); break;
   }
   fesetround( orig_rm );
   thread->set_operand_value(dst,data);
}

void addc_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

void and_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   src1_data = thread->get_operand_value(src1);
   src2_data = thread->get_operand_value(src2);

   data.u64 = src1_data.u64 & src2_data.u64;

   thread->set_operand_value(dst,data);
}

void atom_callback( void* ptx_inst, void* thd )
{
   ptx_thread_info *thread = (ptx_thread_info*)thd;
   ptx_instruction *pI = (ptx_instruction*)ptx_inst;

   // Check state space
   assert( pI->get_space()==global_space );

   // "Decode" the output type
   unsigned to_type = pI->get_type();
   size_t size;
   int t;
   type_info_key::type_decode(to_type, size, t);

   // Set up operand variables
   ptx_reg_t data,      // d
   src1_data,   // a
   src2_data,   // b
   op_result;   // temp variable to hold operation result

   bool data_ready = false;

   // Get operand info of sources and destination
   const operand_info &dst  = pI->dst();     // d
   const operand_info &src1 = pI->src1();    // a
   const operand_info &src2 = pI->src2();    // b

   // Get operand values
   src1_data = thread->get_operand_value(src1);      // a
   src2_data = thread->get_operand_value(src2);      // b

   // Copy value pointed to in operand 'a' into register 'd'
   // (i.e. copy src1_data to dst)
   g_global_mem->read(src1_data.u32,size/8,&data.s64);
   thread->set_operand_value(dst, data);                         // Write value into register 'd'

   // Get the atomic operation to be performed
   unsigned m_atomic_spec = pI->get_atomic();

   switch ( m_atomic_spec ) {
   // AND
   case ATOMIC_AND:
      {

         switch ( to_type ) {
         case B32_TYPE:
         case U32_TYPE:
            op_result.u32 = data.u32 & src2_data.u32;
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = data.s32 & src2_data.s32;
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch (%x) with instruction\natom.AND only accepts b32\n", to_type);
            assert(0);
            break;
         }

         break;
      }
      // OR
   case ATOMIC_OR:
      {

         switch ( to_type ) {
         case B32_TYPE:
         case U32_TYPE:
            op_result.u32 = data.u32 | src2_data.u32;
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = data.s32 | src2_data.s32;
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch (%x) with instruction\natom.OR only accepts b32\n", to_type);
            assert(0);
            break;
         }

         break;
      }
      // XOR
   case ATOMIC_XOR:
      {

         switch ( to_type ) {
         case B32_TYPE:
         case U32_TYPE:
            op_result.u32 = data.u32 ^ src2_data.u32;
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = data.s32 ^ src2_data.s32;
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch (%x) with instruction\natom.XOR only accepts b32\n", to_type);
            assert(0);
            break;
         }

         break;
      }
      // CAS
   case ATOMIC_CAS:
      {

         ptx_reg_t src3_data;
         const operand_info &src3 = pI->src3();
         src3_data = thread->get_operand_value(src3);

         switch ( to_type ) {
         case B32_TYPE:
         case U32_TYPE:
            op_result.u32 = MY_CAS_I(data.u32, src2_data.u32, src3_data.u32);
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = MY_CAS_I(data.s32, src2_data.s32, src3_data.s32);
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch (%x) with instruction\natom.CAS only accepts b32\n", to_type);
            assert(0);
            break;
         }

         break;
      }
      // EXCH
   case ATOMIC_EXCH:
      {
         switch ( to_type ) {
         case B32_TYPE:
         case U32_TYPE:
            op_result.u32 = MY_EXCH(data.u32, src2_data.u32);
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = MY_EXCH(data.s32, src2_data.s32);
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch (%x) with instruction\natom.EXCH only accepts b32\n", to_type);
            assert(0);
            break;
         }

         break;
      }
      // ADD
   case ATOMIC_ADD:
      {

         switch ( to_type ) {
         case U32_TYPE:
            op_result.u32 = data.u32 + src2_data.u32;
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = data.s32 + src2_data.s32;
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch with instruction\natom.ADD only accepts u32 and s32\n");
            assert(0);
            break;
         }

         break;
      }
      // INC
   case ATOMIC_INC:
      {
         switch ( to_type ) {
         case U32_TYPE: 
            op_result.u32 = MY_INC_I(data.u32, src2_data.u32);
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch with instruction\natom.INC only accepts u32 and s32\n");
            assert(0);
            break;
         }

         break;
      }
      // DEC
   case ATOMIC_DEC:
      {
         switch ( to_type ) {
         case U32_TYPE: 
            op_result.u32 = MY_DEC_I(data.u32, src2_data.u32);
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch with instruction\natom.DEC only accepts u32 and s32\n");
            assert(0);
            break;
         }

         break;
      }
      // MIN
   case ATOMIC_MIN:
      {
         switch ( to_type ) {
         case U32_TYPE: 
            op_result.u32 = MY_MIN_I(data.u32, src2_data.u32);
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = MY_MIN_I(data.s32, src2_data.s32);
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch with instruction\natom.MIN only accepts u32 and s32\n");
            assert(0);
            break;
         }

         break;
      }
      // MAX
   case ATOMIC_MAX:
      {
         switch ( to_type ) {
         case U32_TYPE:
            op_result.u32 = MY_MAX_I(data.u32, src2_data.u32);
            data_ready = true;
            break;
         case S32_TYPE:
            op_result.s32 = MY_MAX_I(data.s32, src2_data.s32);
            data_ready = true;
            break;
         default:
            printf("Execution error: type mismatch with instruction\natom.MAX only accepts u32 and s32\n");
            assert(0);
            break;
         }

         break;
      }
      // DEFAULT
   default:
      {
         assert(0);
         break;
      }
   }

   // Write operation result into global memory
   // (i.e. copy src1_data to dst)
   g_global_mem->write(src1_data.u32,size/8,&op_result.s64,thread,pI);
}

// atom_impl will now result in a callback being called in mem_ctrl_pop (gpu-sim.c)
void atom_impl( const ptx_instruction *pI, ptx_thread_info *thread )
{   
   // SYNTAX
   // atom.space.operation.type d, a, b[, c]; (now read in callback)

   // Check state space
   assert( pI->get_space()== global_space );

   // get the memory address
   const operand_info &src1 = pI->src1();
   ptx_reg_t src1_data = thread->get_operand_value(src1);

   memory_space_t space = pI->get_space();

   thread->m_last_effective_address = src1_data.u32;
   thread->m_last_memory_space = space;
   thread->m_last_dram_callback.function = atom_callback;
   thread->m_last_dram_callback.instruction = (void*)pI; 
}

void bar_sync_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   const operand_info &dst  = pI->dst();
   ptx_reg_t b = thread->get_operand_value(dst);
   assert( b.u32 == 0 ); // not clear what should happen if this is not zero
}

void bfe_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void bfi_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void bfind_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

void bra_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   const operand_info &target  = pI->dst();
   ptx_reg_t target_pc = thread->get_operand_value(target);

   thread->m_branch_taken = true;
   thread->set_npc(target_pc);
}

void brev_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void brkpt_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

extern int gpgpu_simd_model;
#define POST_DOMINATOR 1 /* must match enum value in shader.h */
void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *npc, unsigned *rpc );
void gpgpusim_cuda_vprintf(const ptx_instruction * pI, const ptx_thread_info * thread, const function_info * target_func ); 

void call_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   static unsigned call_uid_next = 1;
    
   const operand_info &target  = pI->func_addr();
   assert( target.is_function_address() );
   const symbol *func_addr = target.get_symbol();
   const function_info *target_func = func_addr->get_pc();

   // check that number of args and return match function requirements
   if( pI->has_return() ^ target_func->has_return() ) {
      printf("GPGPU-Sim PTX: Execution error - mismatch in number of return values between\n"
             "               call instruction and function declaration\n");
      abort(); 
   }
   unsigned n_return = target_func->has_return();
   unsigned n_args = target_func->num_args();
   unsigned n_operands = pI->get_num_operands();

   if( n_operands != (n_return+1+n_args) ) {
      printf("GPGPU-Sim PTX: Execution error - mismatch in number of arguements between\n"
             "               call instruction and function declaration\n");
      abort(); 
   }

   // handle intrinsic functions
   std::string fname = target_func->get_name();
   if( fname == "vprintf" ) {
      gpgpusim_cuda_vprintf(pI, thread, target_func);
      return;
   } 

   // read source arguements into register specified in declaration of function
   arg_buffer_list_t arg_values;
   copy_args_into_buffer_list(pI, thread, target_func, arg_values);

   // record local for return value (we only support a single return value)
   const symbol *return_var_src = NULL;
   const symbol *return_var_dst = NULL;
   if( target_func->has_return() ) {
      return_var_dst = pI->dst().get_symbol();
      return_var_src = target_func->get_return_var();
   }

   unsigned sid = thread->get_hw_sid();
   unsigned tid = thread->get_hw_tid();
   unsigned callee_pc=0, callee_rpc=0;
   if( gpgpu_simd_model == POST_DOMINATOR ) {
      get_pdom_stack_top_info(sid,tid,&callee_pc,&callee_rpc);
      assert( callee_pc == thread->get_pc() );
   }

   thread->callstack_push(callee_pc+1,callee_rpc,return_var_src,return_var_dst,call_uid_next++);

   copy_buffer_list_into_frame(thread, arg_values);

   thread->set_npc(target_func);
}

void clz_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

void cnot_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, b, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case PRED_TYPE: d.pred = (a.pred == 0)?1:0; break;
   case B16_TYPE:  d.u16  = (a.u16  == 0)?1:0; break;
   case B32_TYPE:  d.u32  = (a.u32  == 0)?1:0; break;
   case B64_TYPE:  d.u64  = (a.u64  == 0)?1:0; break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); // TODO: add more typechecking like this
      break;
   }

   thread->set_operand_value(dst,d);
}

void cos_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t a, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE: 
      d.f32 = cos(a.f32);
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

ptx_reg_t chop( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   switch ( to_width ) {
   case 8:  x.mask_and(0,0xFF);  break;
   case 16: x.mask_and(0,0xFFFF);      break;
   case 32: x.mask_and(0,0xFFFFFFFF);  break;
   case 64: break;
   default: assert(0);
   }
   return x;
}

ptx_reg_t sext( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   x=chop(x,0,from_width,0,rounding_mode,saturation_mode);
   switch ( from_width ) {
   case 8: if ( x.get_bit(7) ) x.mask_or(0xFFFFFFFF,0xFFFFFF00);break;
   case 16:if ( x.get_bit(15) ) x.mask_or(0xFFFFFFFF,0xFFFF0000);break;
   case 32: if ( x.get_bit(31) ) x.mask_or(0xFFFFFFFF,0x00000000);break;
   case 64: break;
   default: assert(0);
   }
   return x;
}

ptx_reg_t zext( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   return chop(x,0,from_width,0,rounding_mode,saturation_mode);
}

int saturatei(int a, int max, int min) 
{
   if (a > max) a = max;
   else if (a < min) a = min;
   return a;
}

unsigned int saturatei(unsigned int a, unsigned int max) 
{
   if (a > max) a = max;
   return a;
}

ptx_reg_t f2x( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   assert( from_width == 32); 

   enum cuda_math::cudaRoundMode mode = cuda_math::cudaRoundZero;
   switch (rounding_mode) {
   case RZI_OPTION: mode = cuda_math::cudaRoundZero;    break;
   case RNI_OPTION: mode = cuda_math::cudaRoundNearest; break;
   case RMI_OPTION: mode = cuda_math::cudaRoundMinInf;  break;
   case RPI_OPTION: mode = cuda_math::cudaRoundPosInf;  break;
   default: break; 
   }

   ptx_reg_t y;
   if ( to_sign == 1 ) { // convert to 64-bit number first?
      int tmp = cuda_math::__internal_float2int(x.f32, mode);
      if ((x.u32 & 0x7f800000) == 0)
         tmp = 0; // round denorm. FP to 0
      if (saturation_mode && to_width < 32) {
         tmp = saturatei(tmp, (1<<to_width) - 1, -(1<<to_width));
      }
      switch ( to_width ) {
      case 8:  y.s8  = (char)tmp; break;
      case 16: y.s16 = (short)tmp; break;
      case 32: y.s32 = (int)tmp; break;
      case 64: y.s64 = (long long)tmp; break;
      default: assert(0); break;
      }
   } else if ( to_sign == 0 ) {
      unsigned int tmp = cuda_math::__internal_float2uint(x.f32, mode);
      if ((x.u32 & 0x7f800000) == 0)
         tmp = 0; // round denorm. FP to 0
      if (saturation_mode && to_width < 32) {
         tmp = saturatei(tmp, (1<<to_width) - 1);
      }
      switch ( to_width ) {
      case 8:  y.u8  = (unsigned char)tmp; break;
      case 16: y.u16 = (unsigned short)tmp; break;
      case 32: y.u32 = (unsigned int)tmp; break;
      case 64: y.u64 = (unsigned long long)tmp; break;
      default: assert(0); break;
      }
   } else {
      switch ( to_width ) {
      case 16: assert(0); break;
      case 32: assert(0); break; // handled by f2f
      case 64: 
         y.f64 = x.f32; 
         break;
      default: assert(0); break;
      }
   }
   return y;
}

double saturated2i (double a, double max, double min) {
   if (a > max) a = max;
   else if (a < min) a = min;
   return a;
}

ptx_reg_t d2x( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   assert( from_width == 64); 

   double tmp;
   switch (rounding_mode) {
   case RZI_OPTION: tmp = trunc(x.f64);     break;
   case RNI_OPTION: tmp = nearbyint(x.f64); break;
   case RMI_OPTION: tmp = floor(x.f64);     break;
   case RPI_OPTION: tmp = ceil(x.f64);      break;
   default: tmp = x.f64; break; 
   }

   ptx_reg_t y;
   if ( to_sign == 1 ) {
      tmp = saturated2i(tmp, ((1<<(to_width - 1)) - 1), (1<<(to_width - 1)) );
      switch ( to_width ) {
      case 8:  y.s8  = (char)tmp; break;
      case 16: y.s16 = (short)tmp; break;
      case 32: y.s32 = (int)tmp; break;
      case 64: y.s64 = (long long)tmp; break;
      default: assert(0); break;
      }
   } else if ( to_sign == 0 ) {
      tmp = saturated2i(tmp, ((1<<(to_width - 1)) - 1), 0);
      switch ( to_width ) {
      case 8:  y.u8  = (unsigned char)tmp; break;
      case 16: y.u16 = (unsigned short)tmp; break;
      case 32: y.u32 = (unsigned int)tmp; break;
      case 64: y.u64 = (unsigned long long)tmp; break;
      default: assert(0); break;
      }
   } else {
      switch ( to_width ) {
      case 16: assert(0); break;
      case 32:
         y.f32 = x.f64;  
         break;
      case 64: 
         y.f64 = x.f64; // should be handled by d2d
         break;
      default: assert(0); break;
      }
   }
   return y;
}

ptx_reg_t s2f( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   ptx_reg_t y;

   if (from_width < 64) { // 32-bit conversion
      y = sext(x,from_width,32,0,rounding_mode,saturation_mode);

      switch ( to_width ) {
      case 16: assert(0); break;
      case 32: 
         switch (rounding_mode) {
         case RZ_OPTION: y.f32 = cuda_math::__int2float_rz(y.s32); break;
         case RN_OPTION: y.f32 = cuda_math::__int2float_rn(y.s32); break;
         case RM_OPTION: y.f32 = cuda_math::__int2float_rd(y.s32); break;
         case RP_OPTION: y.f32 = cuda_math::__int2float_ru(y.s32); break;
         default: break; 
         }
         break;
      case 64: y.f64 = y.s32; break; // no rounding needed
      default: assert(0); break;
      }
   } else {
      switch ( to_width ) {
      case 16: assert(0); break;
      case 32: 
         switch (rounding_mode) {
         case RZ_OPTION: y.f32 = cuda_math::__ll2float_rn(y.s64); break; 
         case RN_OPTION: y.f32 = cuda_math::__ll2float_rn(y.s64); break;
         case RM_OPTION: y.f32 = cuda_math::__ll2float_rn(y.s64); break; 
         case RP_OPTION: y.f32 = cuda_math::__ll2float_rn(y.s64); break;
         default: break; 
         }
         break;
      case 64: y.f64 = y.s64; break; // no internal implementation found
      default: assert(0); break;
      }
   }

   // saturating an integer to 1 or 0?
   return y;
}

ptx_reg_t u2f( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   ptx_reg_t y;

   if (from_width < 64) { // 32-bit conversion
      y = zext(x,from_width,32,0,rounding_mode,saturation_mode);

      switch ( to_width ) {
      case 16: assert(0); break;
      case 32: 
         switch (rounding_mode) {
         case RZ_OPTION: y.f32 = cuda_math::__uint2float_rz(y.u32); break;
         case RN_OPTION: y.f32 = cuda_math::__uint2float_rn(y.u32); break;
         case RM_OPTION: y.f32 = cuda_math::__uint2float_rd(y.u32); break;
         case RP_OPTION: y.f32 = cuda_math::__uint2float_ru(y.u32); break;
         default: break; 
         }
         break;
      case 64: y.f64 = y.u32; break; // no rounding needed
      default: assert(0); break;
      }
   } else {
      switch ( to_width ) {
      case 16: assert(0); break;
      case 32: 
         switch (rounding_mode) {
         case RZ_OPTION: y.f32 = cuda_math::__ull2float_rn(y.u64); break; 
         case RN_OPTION: y.f32 = cuda_math::__ull2float_rn(y.u64); break;
         case RM_OPTION: y.f32 = cuda_math::__ull2float_rn(y.u64); break; 
         case RP_OPTION: y.f32 = cuda_math::__ull2float_rn(y.u64); break; 
         default: break; 
         }
         break;
      case 64: y.f64 = y.u64; break; // no internal implementation found
      default: assert(0); break;
      }
   }

   // saturating an integer to 1 or 0?
   return y;
}

ptx_reg_t f2f( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   ptx_reg_t y;
   switch ( rounding_mode ) {
   case RZI_OPTION: 
      y.f32 = truncf(x.f32); 
      break;          
   case RNI_OPTION: 
#if CUDART_VERSION >= 3000
      y.f32 = nearbyintf(x.f32); 
#else
      y.f32 = cuda_math::__internal_nearbyintf(x.f32); 
#endif
      break;          
   case RMI_OPTION: 
      if ((x.u32 & 0x7f800000) == 0) {
         y.u32 = x.u32 & 0x80000000; // round denorm. FP to 0, keeping sign
      } else {
         y.f32 = floorf(x.f32); 
      }
      break;          
   case RPI_OPTION: 
      if ((x.u32 & 0x7f800000) == 0) {
         y.u32 = x.u32 & 0x80000000; // round denorm. FP to 0, keeping sign
      } else {
         y.f32 = ceilf(x.f32); 
      }
      break;          
   default: 
      if ((x.u32 & 0x7f800000) == 0) {
         y.u32 = x.u32 & 0x80000000; // round denorm. FP to 0, keeping sign
      } else {
         y.f32 = x.f32;
      }
      break; 
   }
#if CUDART_VERSION >= 3000
   if (isnanf(y.f32)) 
#else
   if (cuda_math::__cuda___isnanf(y.f32)) 
#endif
   {
      y.u32 = 0x7fffffff;
   } else if (saturation_mode) {
      y.f32 = cuda_math::__saturatef(y.f32);
   }

   return y;
}

ptx_reg_t d2d( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, int rounding_mode, int saturation_mode )
{
   ptx_reg_t y;
   switch ( rounding_mode ) {
   case RZI_OPTION: 
      y.f64 = trunc(x.f64); 
      break;          
   case RNI_OPTION: 
#if CUDART_VERSION >= 3000
      y.f64 = nearbyint(x.f32); 
#else
      y.f64 = cuda_math::__internal_nearbyint(x.f64); 
#endif
      break;          
   case RMI_OPTION: 
      y.f64 = floor(x.f64); 
      break;          
   case RPI_OPTION: 
      y.f64 = ceil(x.f64); 
      break;          
   default: 
      y.f64 = x.f64;
      break; 
   }
   if (isnan(y.f64)) {
      y.u64 = 0xfff8000000000000ull;
   } else if (saturation_mode) {
      y.f64 = cuda_math::__saturatef(y.f64); 
   }
   return y;
}

ptx_reg_t (*g_cvt_fn[11][11])( ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign, 
                               int rounding_mode, int saturation_mode ) = {
   { NULL, sext, sext, sext, NULL, sext, sext, sext, s2f, s2f, s2f}, 
   { chop, NULL, sext, sext, chop, NULL, sext, sext, s2f, s2f, s2f}, 
   { chop, chop, NULL, sext, chop, chop, NULL, sext, s2f, s2f, s2f}, 
   { chop, chop, chop, NULL, chop, chop, chop, NULL, s2f, s2f, s2f}, 
   { NULL, zext, zext, zext, NULL, zext, zext, zext, u2f, u2f, u2f}, 
   { chop, NULL, zext, zext, chop, NULL, zext, zext, u2f, u2f, u2f}, 
   { chop, chop, NULL, zext, chop, chop, NULL, zext, u2f, u2f, u2f}, 
   { chop, chop, chop, NULL, chop, chop, chop, NULL, u2f, u2f, u2f}, 
   { f2x , f2x , f2x , f2x , f2x , f2x , f2x , f2x , NULL,f2x, f2x}, 
   { f2x , f2x , f2x , f2x , f2x , f2x , f2x , f2x , f2x, f2f, f2x},
   { d2x , d2x , d2x , d2x , d2x , d2x , d2x , d2x , d2x, d2x, d2d} 
};

void ptx_round(ptx_reg_t& data, int rounding_mode, int type)
{
   if (rounding_mode == RN_OPTION) {
      return;
   }
   switch ( rounding_mode ) {
   case RZI_OPTION: 
      switch ( type ) {
      case S8_TYPE:
      case S16_TYPE:
      case S32_TYPE:
      case S64_TYPE:
      case U8_TYPE:
      case U16_TYPE:
      case U32_TYPE:
      case U64_TYPE:
         printf("Trying to round an integer??\n"); assert(0); break;
      case F16_TYPE: assert(0); break;
      case F32_TYPE:
         data.f32 = truncf(data.f32); 
         break;          
      case F64_TYPE:
         if (data.f64 < 0) data.f64 = ceil(data.f64); //negative
         else data.f64 = floor(data.f64); //positive
         break; 
      default: assert(0); break;
      }
      break;
   case RNI_OPTION: 
      switch ( type ) {
      case S8_TYPE:
      case S16_TYPE:
      case S32_TYPE:
      case S64_TYPE:
      case U8_TYPE:
      case U16_TYPE:
      case U32_TYPE:
      case U64_TYPE:
         printf("Trying to round an integer??\n"); assert(0); break;
      case F16_TYPE: assert(0); break;
      case F32_TYPE: 
#if CUDART_VERSION >= 3000
         data.f32 = nearbyintf(data.f32); 
#else
         data.f32 = cuda_math::__cuda_nearbyintf(data.f32); 
#endif
         break;          
      case F64_TYPE: data.f64 = round(data.f64); break; 
      default: assert(0); break;
      }
      break;
   case RMI_OPTION: 
      switch ( type ) {
      case S8_TYPE:
      case S16_TYPE:
      case S32_TYPE:
      case S64_TYPE:
      case U8_TYPE:
      case U16_TYPE:
      case U32_TYPE:
      case U64_TYPE:
         printf("Trying to round an integer??\n"); assert(0); break;
      case F16_TYPE: assert(0); break;
      case F32_TYPE: 
         data.f32 = floorf(data.f32); 
         break;          
      case F64_TYPE: data.f64 = floor(data.f64); break; 
      default: assert(0); break;
      }
      break;
   case RPI_OPTION: 
      switch ( type ) {
      case S8_TYPE:
      case S16_TYPE:
      case S32_TYPE:
      case S64_TYPE:
      case U8_TYPE:
      case U16_TYPE:
      case U32_TYPE:
      case U64_TYPE:
         printf("Trying to round an integer??\n"); assert(0); break;
      case F16_TYPE: assert(0); break;
      case F32_TYPE: data.f32 = ceilf(data.f32); break;          
      case F64_TYPE: data.f64 = ceil(data.f64); break; 
      default: assert(0); break;
      }
      break;
   default:  break; 
   }

   if (type == F32_TYPE) {
#if CUDART_VERSION >= 3000
      if (isnanf(data.f32)) 
#else
      if (cuda_math::__cuda___isnanf(data.f32)) 
#endif
      {
         data.u32 = 0x7fffffff;
      }
   }
   if (type == F64_TYPE) {
      if (isnan(data.f64)) {
         data.u64 = 0xfff8000000000000ull;
      }
   }
}

void ptx_saturate(ptx_reg_t& data, int saturation_mode, int type)
{
   if (!saturation_mode) {
      return;
   }
   switch ( type ) {
   case S8_TYPE:
   case S16_TYPE:
   case S32_TYPE:
   case S64_TYPE:
   case U8_TYPE:
   case U16_TYPE:
   case U32_TYPE:
   case U64_TYPE:
      printf("Trying to clamp an integer to 1??\n"); assert(0); break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE:
      if (data.f32 > 1.0f) data.f32 = 1.0f; //negative
      if (data.f32 < 0.0f) data.f32 = 0.0f; //positive
      break;          
   case F64_TYPE:
      if (data.f64 > 1.0f) data.f64 = 1.0f; //negative
      if (data.f64 < 0.0f) data.f64 = 0.0f; //positive
      break; 
   default: assert(0); break;
   }

}

void cvt_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   unsigned to_type = pI->get_type();
   unsigned from_type = pI->get_type2();
   unsigned rounding_mode = pI->rounding_mode();
   unsigned saturation_mode = pI->saturation_mode();

   if ( to_type == F16_TYPE || from_type == F16_TYPE )
      abort();

   int to_sign, from_sign;
   size_t from_width, to_width;
   unsigned src_fmt = type_info_key::type_decode(from_type, from_width, from_sign);
   unsigned dst_fmt = type_info_key::type_decode(to_type, to_width, to_sign);

   ptx_reg_t data = thread->get_operand_value(src1);
   if ( g_cvt_fn[src_fmt][dst_fmt] != NULL ) {
      ptx_reg_t result = g_cvt_fn[src_fmt][dst_fmt](data,from_width,to_width,to_sign, rounding_mode, saturation_mode);
      data = result;
   }

   thread->set_operand_value(dst,data);
}

void cvta_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   memory_space_t space = pI->get_space();
   bool to_non_generic = pI->is_to();

   ptx_reg_t from_addr = thread->get_operand_value(src1);
   addr_t from_addr_hw = (addr_t)from_addr.u64;
   addr_t to_addr_hw = 0;
   unsigned smid = thread->get_hw_sid();
   unsigned hwtid = thread->get_hw_tid();

   if( to_non_generic ) {
      switch( space.get_type() ) {
      case shared_space: to_addr_hw = generic_to_shared( smid, from_addr_hw ); break;
      case local_space:  to_addr_hw = generic_to_local( smid, hwtid, from_addr_hw ); break;
      case global_space: to_addr_hw = generic_to_global(from_addr_hw ); break;
      default: abort();
      }
   } else {
      switch( space.get_type() ) {
      case shared_space: to_addr_hw = shared_to_generic( smid, from_addr_hw ); break;
      case local_space:  to_addr_hw =  local_to_generic( smid, hwtid, from_addr_hw ); break;
      case global_space: to_addr_hw = global_to_generic( from_addr_hw ); break;
      default: abort();
      }
   }
   
   ptx_reg_t to_addr;
   to_addr.u64 = to_addr_hw;
   thread->set_operand_value(dst,to_addr);
}

void div_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   ptx_reg_t src1_data = thread->get_operand_value(src1);
   ptx_reg_t src2_data = thread->get_operand_value(src2);

   unsigned i_type = pI->get_type();

   switch ( i_type ) {
   case S8_TYPE:
   case S16_TYPE:
   case S32_TYPE:
   case S64_TYPE: 
      data.s64 = src1_data.s64 / src2_data.s64; break;
   case U8_TYPE:
   case U16_TYPE:
   case U32_TYPE:
   case U64_TYPE: 
   case B8_TYPE:
   case B16_TYPE:
   case B32_TYPE:
   case B64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64; break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE: data.f32 = src1_data.f32 / src2_data.f32; break;
   case F64_TYPE: data.f64 = src1_data.f64 / src2_data.f64; break;
   default: assert(0); break;
   }
   thread->set_operand_value(dst,data);
}

void ex2_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   src1_data = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE: 
      data.f32 = cuda_math::__powf(2.0, src1_data.f32);
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,data);
}

void exit_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   core_t *sc = thread->get_core();
   unsigned warp_id = thread->get_hw_wid();
   sc->warp_exit(warp_id);

   thread->m_cta_info->register_thread_exit(thread);
   thread->set_done();
}

void mad_def( const ptx_instruction *pI, ptx_thread_info *thread );

void fma_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   mad_def(pI,thread);
}

void isspacep_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a;
   bool t=false;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   memory_space_t space = pI->get_space();

   a = thread->get_operand_value(src1);
   addr_t addr = (addr_t)a.u64;
   unsigned smid = thread->get_hw_sid();
   unsigned hwtid = thread->get_hw_tid();

   switch( space.get_type() ) {
   case shared_space: t = isspace_shared( smid, addr );
   case local_space:  t = isspace_local( smid, hwtid, addr );
   case global_space: t = isspace_global( addr );
   default: abort();
   }

   ptx_reg_t p;
   p.pred = t?1:0;

   thread->set_operand_value(dst,p);
}

void decode_space( memory_space_t &space, const ptx_thread_info *thread, const operand_info &op, memory_space *&mem, addr_t &addr)
{
   unsigned smid = thread->get_hw_sid();
   unsigned hwtid = thread->get_hw_tid();

   if( space == param_space_unclassified ) {
      // need to op to determine whether it refers to a kernel param or local param
      const symbol *s = op.get_symbol();
      const type_info *t = s->type();
      type_info_key ti = t->get_key();
      if( ti.is_param_kernel() )
         space = param_space_kernel;
      else if( ti.is_param_local() ) {
         space = param_space_local;
      } else {
         printf("GPGPU-Sim PTX: ERROR ** cannot resolve .param space for '%s'\n", s->name().c_str() );
         abort(); 
      }
   }
   switch ( space.get_type() ) {
   case global_space: mem = g_global_mem; break;
   case param_space_local:
   case local_space:
      mem = thread->m_local_mem; 
      addr += thread->get_local_mem_stack_pointer();
      break; 
   case tex_space:    mem = g_tex_mem; break; 
   case surf_space:   mem = g_surf_mem; break; 
   case param_space_kernel:  mem = g_param_mem; break;
   case shared_space:  mem = thread->m_shared_mem; break; 
   case const_space:  mem = g_global_mem; break;
   case generic_space:
      if( thread->get_ptx_version().ver() >= 2.0 ) {
         // convert generic address to memory space address
         space = whichspace(addr);
         switch ( space.get_type() ) {
         case global_space: mem = g_global_mem; addr = generic_to_global(addr); break;
         case local_space:  mem = thread->m_local_mem; addr = generic_to_local(smid,hwtid,addr); break; 
         case shared_space: mem = thread->m_shared_mem; addr = generic_to_shared(smid,addr); break; 
         default: abort();
         }
      } else {
         abort();
      }
      break;
   case param_space_unclassified:
   case undefined_space:
   default:
      abort();
   }
}

void ld_exec( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   const operand_info &dst = pI->dst();
   const operand_info &src1 = pI->src1();
   ptx_reg_t src1_data = thread->get_operand_value(src1);
   ptx_reg_t data;
   memory_space_t space = pI->get_space();
   unsigned vector_spec = pI->get_vector();
   unsigned type = pI->get_type();
   memory_space *mem = NULL;
   addr_t addr = src1_data.u32;

   decode_space(space,thread,src1,mem,addr);

   size_t size;
   int t;
   data.u64=0;
   type_info_key::type_decode(type,size,t);
   if (!vector_spec) {
      mem->read(addr,size/8,&data.s64);
      if( type == S16_TYPE || type == S32_TYPE ) 
         sign_extend(data,size,dst);
      thread->set_operand_value(dst,data);
   } else {
      ptx_reg_t data1, data2, data3, data4;
      mem->read(addr,size/8,&data1.s64);
      mem->read(addr+size/8,size/8,&data2.s64);
      if (vector_spec != V2_TYPE) { //either V3 or V4
         mem->read(addr+2*size/8,size/8,&data3.s64);
         if (vector_spec != V3_TYPE) { //v4
            mem->read(addr+3*size/8,size/8,&data4.s64);
            thread->set_vector_operand_values(dst,data1,data2,data3,data4, 4);
         } else //v3
            thread->set_vector_operand_values(dst,data1,data2,data3,data3,3);
      } else //v2
         thread->set_vector_operand_values(dst,data1,data2,data2,data2,2);
   }
   thread->m_last_effective_address = addr;
   thread->m_last_memory_space = space; 
}

void ld_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ld_exec(pI,thread);
}
void ldu_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ld_exec(pI,thread);
}

void lg2_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE: 
      d.f32 = log(a.f32)/log(2);
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
   }

   thread->set_operand_value(dst,d);
}

void mad24_impl( const ptx_instruction *pI, ptx_thread_info *thread )
{
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   const operand_info &src3 = pI->src3();
   ptx_reg_t d, t;
   ptx_reg_t a = thread->get_operand_value(src1);
   ptx_reg_t b = thread->get_operand_value(src2);
   ptx_reg_t c = thread->get_operand_value(src3);

   unsigned i_type = pI->get_type();
   unsigned sat_mode = pI->saturation_mode();

   assert( !pI->is_wide() );

   switch ( i_type ) {
   case S32_TYPE: 
      t.s64 = a.s32 * b.s32;
      if ( pI->is_hi() ) {
         d.s64 = (t.s64>>16) + c.s32;
         if ( sat_mode ) {
            if ( d.s64 > (int)0x7FFFFFFF )
               d.s64 = (int)0x7FFFFFFF;
            else if ( d.s64 < (int)0x80000000 )
               d.s64 = (int)0x80000000;
         }
      } else if ( pI->is_lo() ) d.s64 = t.s32 + c.s32;
      else assert(0);
      break;
   case U32_TYPE: 
      t.u64 = a.u32 * b.u32;
      if ( pI->is_hi() ) d.u64 = (t.u64>>16) + c.u32;
      else if ( pI->is_lo() ) d.u64 = t.u32 + c.u32;
      else assert(0);
      break;
   default: 
      assert(0);
      break;
   }
   thread->set_operand_value(dst,d);
}

void mad_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   mad_def(pI,thread);
}

void mad_def( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   const operand_info &src3 = pI->src3();
   ptx_reg_t d, t;
   ptx_reg_t a = thread->get_operand_value(src1);
   ptx_reg_t b = thread->get_operand_value(src2);
   ptx_reg_t c = thread->get_operand_value(src3);

   unsigned i_type = pI->get_type();
   unsigned rounding_mode = pI->rounding_mode();

   switch ( i_type ) {
   case S16_TYPE: 
      t.s32 = a.s16 * b.s16;
      if ( pI->is_wide() ) d.s32 = t.s32 + c.s32;
      else if ( pI->is_hi() ) d.s16 = (t.s32>>16) + c.s16;
      else if ( pI->is_lo() ) d.s16 = t.s16 + c.s16;
      else assert(0);
      break;
   case S32_TYPE: 
      t.s64 = a.s32 * b.s32;
      if ( pI->is_wide() ) d.s64 = t.s64 + c.s64;
      else if ( pI->is_hi() ) d.s32 = (t.s64>>32) + c.s32;
      else if ( pI->is_lo() ) d.s32 = t.s32 + c.s32;
      else assert(0);
      break;
   case S64_TYPE: 
      t.s64 = a.s64 * b.s64;
      assert( !pI->is_wide() );
      assert( !pI->is_hi() );
      if ( pI->is_lo() ) d.s64 = t.s64 + c.s64;
      else assert(0);
      break;
   case U16_TYPE: 
      t.u32 = a.u16 * b.u16;
      if ( pI->is_wide() ) d.u32 = t.u32 + c.u32;
      else if ( pI->is_hi() ) d.u16 = (t.u32>>16) + c.u16;
      else if ( pI->is_lo() ) d.u16 = t.u16 + c.u16;
      else assert(0);
      break;
   case U32_TYPE: 
      t.u64 = a.u32 * b.u32;
      if ( pI->is_wide() ) d.u64 = t.u64 + c.u64;
      else if ( pI->is_hi() ) d.u32 = (t.u64>>32) + c.u32;
      else if ( pI->is_lo() ) d.u32 = t.u32 + c.u32;
      else assert(0);
      break;
   case U64_TYPE: 
      t.u64 = a.u64 * b.u64;
      assert( !pI->is_wide() );
      assert( !pI->is_hi() );
      if ( pI->is_lo() ) d.u64 = t.u64 + c.u64;
      else assert(0);
      break;
   case F16_TYPE: 
      assert(0); 
      break;
   case F32_TYPE: {
         int orig_rm = fegetround();
         switch ( rounding_mode ) {
         case RN_OPTION: break;
         case RZ_OPTION: fesetround( FE_TOWARDZERO ); break;
         default: assert(0); break;
         }
         d.f32 = a.f32 * b.f32 + c.f32;
         if ( pI->saturation_mode() ) {
            if ( d.f32 < 0 ) d.f32 = 0;
            else if ( d.f32 > 1.0f ) d.f32 = 1.0f;
         }
         fesetround( orig_rm );
         break;
      }  
   case F64_TYPE: {
         int orig_rm = fegetround();
         switch ( rounding_mode ) {
         case RN_OPTION: break;
         case RZ_OPTION: fesetround( FE_TOWARDZERO ); break;
         default: assert(0); break;
         }
         d.f64 = a.f64 * b.f64 + c.f64;
         if ( pI->saturation_mode() ) {
            if ( d.f64 < 0 ) d.f64 = 0;
            else if ( d.f64 > 1.0f ) d.f64 = 1.0;
         }
         fesetround( orig_rm );
         break;
      }
   default: 
      assert(0);
      break;
   }
   thread->set_operand_value(dst,d);
}

bool isNaN(float x)
{
   return isnan(x);
}

bool isNaN(double x)
{
   return isnan(x);
}

void max_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, b, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case U16_TYPE: d.u16 = MY_MAX_I(a.u16,b.u16); break;
   case U32_TYPE: d.u32 = MY_MAX_I(a.u32,b.u32); break;
   case U64_TYPE: d.u64 = MY_MAX_I(a.u64,b.u64); break;
   case S16_TYPE: d.s16 = MY_MAX_I(a.s16,b.s16); break;
   case S32_TYPE: d.s32 = MY_MAX_I(a.s32,b.s32); break;
   case S64_TYPE: d.s64 = MY_MAX_I(a.s64,b.s64); break;
   case F32_TYPE: d.f32 = MY_MAX_F(a.f32,b.f32); break;
   case F64_TYPE: d.f64 = MY_MAX_F(a.f64,b.f64); break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

void membar_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

void min_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, b, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case U16_TYPE: d.u16 = MY_MIN_I(a.u16,b.u16); break;
   case U32_TYPE: d.u32 = MY_MIN_I(a.u32,b.u32); break;
   case U64_TYPE: d.u64 = MY_MIN_I(a.u64,b.u64); break;
   case S16_TYPE: d.s16 = MY_MIN_I(a.s16,b.s16); break;
   case S32_TYPE: d.s32 = MY_MIN_I(a.s32,b.s32); break;
   case S64_TYPE: d.s64 = MY_MIN_I(a.s64,b.s64); break;
   case F32_TYPE: d.f32 = MY_MIN_F(a.f32,b.f32); break;
   case F64_TYPE: d.f64 = MY_MIN_F(a.f64,b.f64); break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
   }

   thread->set_operand_value(dst,d);
}

void mov_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();

   if( src1.is_vector() || dst.is_vector() ) {
      // pack or unpack operation
      unsigned nbits_to_move;
      ptx_reg_t tmp_bits;

      switch( pI->get_type() ) {
      case B16_TYPE: nbits_to_move = 16; break;
      case B32_TYPE: nbits_to_move = 32; break;
      case B64_TYPE: nbits_to_move = 64; break;
      default: printf("Execution error: mov pack/unpack with unsupported type qualifier\n"); assert(0); break;
      }

      if( src1.is_vector() ) {
         unsigned nelem = src1.get_vect_nelem();
         ptx_reg_t v[4];
         thread->get_vector_operand_values(src1, v, nelem );

         unsigned bits_per_src_elem = nbits_to_move / nelem;
         for( unsigned i=0; i < nelem; i++ ) {
            switch(bits_per_src_elem) {
            case 8:   tmp_bits.u64 |= ((unsigned long long)(v[i].u8)  << (8*i));  break;
            case 16:  tmp_bits.u64 |= ((unsigned long long)(v[i].u16) << (16*i)); break;
            case 32:  tmp_bits.u64 |= ((unsigned long long)(v[i].u32) << (32*i)); break;
            default: printf("Execution error: mov pack/unpack with unsupported source/dst size ratio (src)\n"); assert(0); break;
            }
         }
      } else {
         data = thread->get_operand_value(src1);

         switch( pI->get_type() ) {
         case B16_TYPE: tmp_bits.u16 = data.u16; break;
         case B32_TYPE: tmp_bits.u32 = data.u32; break;
         case B64_TYPE: tmp_bits.u64 = data.u64; break;
         default: assert(0); break;
         }
      }

      if( dst.is_vector() ) {
         unsigned nelem = dst.get_vect_nelem();
         ptx_reg_t v[4];
         unsigned bits_per_dst_elem = nbits_to_move / nelem;
         for( unsigned i=0; i < nelem; i++ ) {
            switch(bits_per_dst_elem) {
            case 8:  v[i].u8  = tmp_bits.u64  & (((unsigned long long) 0xFF) << (8*i)); break;
            case 16: v[i].u16 = tmp_bits.u64  & (((unsigned long long) 0xFFFF) << (16*i)); break;
            case 32: v[i].u32 = tmp_bits.u64  & (((unsigned long long) 0xFFFFFFFF) << (32*i)); break;
            default:
               printf("Execution error: mov pack/unpack with unsupported source/dst size ratio (dst)\n");
               assert(0);
               break;
            }
         }
         thread->set_vector_operand_values(dst,v[0],v[1],v[2],v[3],nelem);
      } else {
         thread->set_operand_value(dst,tmp_bits);
      }
   } else {
      data = thread->get_operand_value(src1);
      thread->set_operand_value(dst,data);
   }
}

void mul24_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t src1_data, src2_data, data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   src1_data = thread->get_operand_value(src1);
   src2_data = thread->get_operand_value(src2);
   src1_data.mask_and(0,0x00FFFFFF);
   src2_data.mask_and(0,0x00FFFFFF);

   unsigned i_type = pI->get_type();

   switch ( i_type ) {
   case S32_TYPE: 
      if( src1_data.get_bit(23) ) 
         src1_data.mask_or(0xFFFFFFFF,0xFF000000);
      if( src2_data.get_bit(23) ) 
         src2_data.mask_or(0xFFFFFFFF,0xFF000000);
      data.s64 = src1_data.s64 * src2_data.s64;
      break;
   case U32_TYPE:
      data.u64 = src1_data.u64 * src2_data.u64;
      break;
   default:
      printf("GPGPU-Sim PTX: Execution error - type mismatch with instruction\n");
      assert(0);
      break;
   }

   if ( pI->is_hi() ) {
      data.u64 = data.u64 >> 16;
      data.mask_and(0,0xFFFFFFFF);
   } else if (pI->is_lo()) {
      data.mask_and(0,0xFFFFFFFF);
   }

   thread->set_operand_value(dst,data);
}

void mul_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   ptx_reg_t d, t;
   ptx_reg_t a = thread->get_operand_value(src1);
   ptx_reg_t b = thread->get_operand_value(src2);

   unsigned i_type = pI->get_type();
   unsigned rounding_mode = pI->rounding_mode();

   switch ( i_type ) {
   case S16_TYPE: 
      t.s32 = ((int)a.s16) * ((int)b.s16);
      if ( pI->is_wide() ) d.s32 = t.s32;
      else if ( pI->is_hi() ) d.s16 = (t.s32>>16);
      else if ( pI->is_lo() ) d.s16 = t.s16;
      else assert(0);
      break;
   case S32_TYPE: 
      t.s64 = ((long long)a.s32) * ((long long)b.s32);
      if ( pI->is_wide() ) d.s64 = t.s64;
      else if ( pI->is_hi() ) d.s32 = (t.s64>>32);
      else if ( pI->is_lo() ) d.s32 = t.s32;
      else assert(0);
      break;
   case S64_TYPE: 
      t.s64 = a.s64 * b.s64;
      assert( !pI->is_wide() );
      assert( !pI->is_hi() );
      if ( pI->is_lo() ) d.s64 = t.s64;
      else assert(0);
      break;
   case U16_TYPE: 
      t.u32 = ((unsigned)a.u16) * ((unsigned)b.u16);
      if ( pI->is_wide() ) d.u32 = t.u32;
      else if ( pI->is_lo() ) d.u16 = t.u16;
      else if ( pI->is_hi() ) d.u16 = (t.u32>>16);
      else assert(0);
      break;
   case U32_TYPE: 
      t.u64 = ((unsigned long long)a.u32) * ((unsigned long long)b.u32);
      if ( pI->is_wide() ) d.u64 = t.u64;
      else if ( pI->is_lo() ) d.u32 = t.u32;
      else if ( pI->is_hi() ) d.u32 = (t.u64>>32);
      else assert(0);
      break;
   case U64_TYPE: 
      t.u64 = a.u64 * b.u64;
      assert( !pI->is_wide() );
      assert( !pI->is_hi() );
      if ( pI->is_lo() ) d.u64 = t.u64;
      else assert(0);
      break;
   case F16_TYPE: 
      assert(0); 
      break;
   case F32_TYPE: {
         int orig_rm = fegetround();
         switch ( rounding_mode ) {
         case RN_OPTION: break;
         case RZ_OPTION: fesetround( FE_TOWARDZERO ); break;
         default: assert(0); break;
         }
         d.f32 = a.f32 * b.f32;
         if ( pI->saturation_mode() ) {
            if ( d.f32 < 0 ) d.f32 = 0;
            else if ( d.f32 > 1.0f ) d.f32 = 1.0f;
         }
         fesetround( orig_rm );
         break;
      }  
   case F64_TYPE: {
         int orig_rm = fegetround();
         switch ( rounding_mode ) {
         case RN_OPTION: break;
         case RZ_OPTION: fesetround( FE_TOWARDZERO ); break;
         default: assert(0); break;
         }
         d.f64 = a.f64 * b.f64;
         if ( pI->saturation_mode() ) {
            if ( d.f64 < 0 ) d.f64 = 0;
            else if ( d.f64 > 1.0f ) d.f64 = 1.0;
         }
         fesetround( orig_rm );
         break;
      }
   default: 
      assert(0); 
      break;
   }
   thread->set_operand_value(dst,d);
}

void neg_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();

   src1_data = thread->get_operand_value(src1);

   unsigned to_type = pI->get_type();
   switch ( to_type ) {
   case S8_TYPE:
   case S16_TYPE:
   case S32_TYPE:
   case S64_TYPE: 
      data.s64 = 0 - src1_data.s64; break; // seems buggy, but not (just ignore higher bits)
   case U8_TYPE:
   case U16_TYPE:
   case U32_TYPE:
   case U64_TYPE: 
      assert(0); break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE: data.f32 = 0.0f - src1_data.f32; break;
   case F64_TYPE: data.f64 = 0.0f - src1_data.f64; break;
   default: assert(0); break;
   }

   thread->set_operand_value(dst,data);
}

void not_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t a, b, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case PRED_TYPE: d.pred = ~a.pred; break;
   case B16_TYPE:  d.u16  = ~a.u16; break;
   case B32_TYPE:  d.u32  = ~a.u32; break;
   case B64_TYPE:  d.u64  = ~a.u64; break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

void or_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   src1_data = thread->get_operand_value(src1);
   src2_data = thread->get_operand_value(src2);

   data.u64 = src1_data.u64 | src2_data.u64;

   thread->set_operand_value(dst,data);
}

void pmevent_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void popc_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void prefetch_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void prefetchu_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void prmt_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

void rcp_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   src1_data = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE: 
      data.f32 = 1.0f / src1_data.f32;
      break;
   case F64_TYPE:
      data.f64 = 1.0f / src1_data.f64;
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,data);
}

void red_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

void rem_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   src1_data = thread->get_operand_value(src1);
   src2_data = thread->get_operand_value(src2);

   data.u64 = src1_data.u64 % src2_data.u64;

   thread->set_operand_value(dst,data);
}

void ret_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   bool empty = thread->callstack_pop();
   if( empty ) {
      core_t *sc = thread->get_core();
      unsigned warp_id = thread->get_hw_wid();
      sc->warp_exit(warp_id);
      thread->m_cta_info->register_thread_exit(thread);
      thread->set_done();
   }
}

void rsqrt_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE:
      if ( a.f32 < 0 ) {
         d.u64 = 0;
         d.u64 = 0x7fc00000; // NaN
      } else if ( a.f32 == 0 ) {
         d.u64 = 0;
         d.u32 = 0x7f800000; // Inf
      } else
         d.f32 = cuda_math::__internal_accurate_fdividef(1.0f, sqrtf(a.f32));
      break;
   case F64_TYPE: 
      if ( a.f32 < 0 ) {
         d.u64 = 0;
	      d.u32 = 0x7fc00000; // NaN
         float x = d.f32; 
         d.f64 = (double)x;
      } else if ( a.f32 == 0 ) {
         d.u64 = 0;
	      d.u32 = 0x7f800000; // Inf
         float x = d.f32; 
         d.f64 = (double)x;
      } else
         d.f64 = 1.0 / sqrt(a.f64); 
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
   }

   thread->set_operand_value(dst,d);
}

#define SAD(d,a,b,c) d = c + ((a<b) ? (b-a) : (a-b))

void sad_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, b, c, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src1();
   const operand_info &src3 = pI->src1();
   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);
   c = thread->get_operand_value(src3);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case U16_TYPE: SAD(d.u16,a.u16,b.u16,c.u16); break;
   case U32_TYPE: SAD(d.u32,a.u32,b.u32,c.u32); break;
   case U64_TYPE: SAD(d.u64,a.u64,b.u64,c.u64); break;
   case S16_TYPE: SAD(d.s16,a.s16,b.s16,c.s16); break;
   case S32_TYPE: SAD(d.s32,a.s32,b.s32,c.s32); break;
   case S64_TYPE: SAD(d.s64,a.s64,b.s64,c.s64); break;
   case F32_TYPE: SAD(d.f32,a.f32,b.f32,c.f32); break;
   case F64_TYPE: SAD(d.f64,a.f64,b.f64,c.f64); break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

void selp_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   const operand_info &src3 = pI->src3();

   ptx_reg_t a, b, c, d;

   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);
   c = thread->get_operand_value(src3);

   d = (c.pred)?a:b;

   thread->set_operand_value(dst,d);
}

bool isFloat(int type) 
{
   switch ( type ) {
   case F16_TYPE:
   case F32_TYPE:
   case F64_TYPE:
      return true;
   default:
      return false;
   }
}

bool CmpOp( int type, ptx_reg_t a, ptx_reg_t b, unsigned cmpop )
{
   bool t = false;

   switch ( type ) {
   case B16_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.u16 == b.u16); break;
      case NE_OPTION: t = (a.u16 != b.u16); break;
      default:
         assert(0);
      }

   case B32_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.u32 == b.u32); break;
      case NE_OPTION: t = (a.u32 != b.u32); break;
      default:
         assert(0);
      }
   case B64_TYPE:
      switch (cmpop) {
      case EQ_OPTION: t = (a.u64 == b.u64); break;
      case NE_OPTION: t = (a.u64 != b.u64); break;
      default:
         assert(0);
      }
      break;
   case S8_TYPE: 
   case S16_TYPE:
      switch (cmpop) {
      case EQ_OPTION: t = (a.s16 == b.s16); break;
      case NE_OPTION: t = (a.s16 != b.s16); break;
      case LT_OPTION: t = (a.s16 < b.s16); break;
      case LE_OPTION: t = (a.s16 <= b.s16); break;
      case GT_OPTION: t = (a.s16 > b.s16); break;
      case GE_OPTION: t = (a.s16 >= b.s16); break;
      default:
         assert(0);
      }
      break;
   case S32_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.s32 == b.s32); break;
      case NE_OPTION: t = (a.s32 != b.s32); break;
      case LT_OPTION: t = (a.s32 < b.s32); break;
      case LE_OPTION: t = (a.s32 <= b.s32); break;
      case GT_OPTION: t = (a.s32 > b.s32); break;
      case GE_OPTION: t = (a.s32 >= b.s32); break;
      default:
         assert(0);
      }
      break;
   case S64_TYPE:
      switch (cmpop) {
      case EQ_OPTION: t = (a.s64 == b.s64); break;
      case NE_OPTION: t = (a.s64 != b.s64); break;
      case LT_OPTION: t = (a.s64 < b.s64); break;
      case LE_OPTION: t = (a.s64 <= b.s64); break;
      case GT_OPTION: t = (a.s64 > b.s64); break;
      case GE_OPTION: t = (a.s64 >= b.s64); break;
      default:
         assert(0);
      }
      break;
   case U8_TYPE: 
   case U16_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.u16 == b.u16); break;
      case NE_OPTION: t = (a.u16 != b.u16); break;
      case LT_OPTION: t = (a.u16 < b.u16); break;
      case LE_OPTION: t = (a.u16 <= b.u16); break;
      case GT_OPTION: t = (a.u16 > b.u16); break;
      case GE_OPTION: t = (a.u16 >= b.u16); break;
      case LO_OPTION: t = (a.u16 < b.u16); break;
      case LS_OPTION: t = (a.u16 <= b.u16); break;
      case HI_OPTION: t = (a.u16 > b.u16); break;
      case HS_OPTION: t = (a.u16 >= b.u16); break;
      default:
         assert(0);
      }
      break;
   case U32_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.u32 == b.u32); break;
      case NE_OPTION: t = (a.u32 != b.u32); break;
      case LT_OPTION: t = (a.u32 < b.u32); break;
      case LE_OPTION: t = (a.u32 <= b.u32); break;
      case GT_OPTION: t = (a.u32 > b.u32); break;
      case GE_OPTION: t = (a.u32 >= b.u32); break;
      case LO_OPTION: t = (a.u32 < b.u32); break;
      case LS_OPTION: t = (a.u32 <= b.u32); break;
      case HI_OPTION: t = (a.u32 > b.u32); break;
      case HS_OPTION: t = (a.u32 >= b.u32); break;
      default:
         assert(0);
      }
      break;
   case U64_TYPE:
      switch (cmpop) {
      case EQ_OPTION: t = (a.u64 == b.u64); break;
      case NE_OPTION: t = (a.u64 != b.u64); break;
      case LT_OPTION: t = (a.u64 < b.u64); break;
      case LE_OPTION: t = (a.u64 <= b.u64); break;
      case GT_OPTION: t = (a.u64 > b.u64); break;
      case GE_OPTION: t = (a.u64 >= b.u64); break;
      case LO_OPTION: t = (a.u64 < b.u64); break;
      case LS_OPTION: t = (a.u64 <= b.u64); break;
      case HI_OPTION: t = (a.u64 > b.u64); break;
      case HS_OPTION: t = (a.u64 >= b.u64); break;
      default:
         assert(0);
      }
      break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.f32 == b.f32) && !isNaN(a.f32) && !isNaN(b.f32); break;
      case NE_OPTION: t = (a.f32 != b.f32) && !isNaN(a.f32) && !isNaN(b.f32); break;
      case LT_OPTION: t = (a.f32 < b.f32 ) && !isNaN(a.f32) && !isNaN(b.f32); break;
      case LE_OPTION: t = (a.f32 <= b.f32) && !isNaN(a.f32) && !isNaN(b.f32); break;
      case GT_OPTION: t = (a.f32 > b.f32 ) && !isNaN(a.f32) && !isNaN(b.f32); break;
      case GE_OPTION: t = (a.f32 >= b.f32) && !isNaN(a.f32) && !isNaN(b.f32); break;
      case EQU_OPTION: t = (a.f32 == b.f32) || isNaN(a.f32) || isNaN(b.f32); break;
      case NEU_OPTION: t = (a.f32 != b.f32) || isNaN(a.f32) || isNaN(b.f32); break;
      case LTU_OPTION: t = (a.f32 < b.f32 ) || isNaN(a.f32) || isNaN(b.f32); break;
      case LEU_OPTION: t = (a.f32 <= b.f32) || isNaN(a.f32) || isNaN(b.f32); break;
      case GTU_OPTION: t = (a.f32 > b.f32 ) || isNaN(a.f32) || isNaN(b.f32); break;
      case GEU_OPTION: t = (a.f32 >= b.f32) || isNaN(a.f32) || isNaN(b.f32); break;
      case NUM_OPTION: t = !isNaN(a.f32) && !isNaN(b.f32); break;
      case NAN_OPTION: t = isNaN(a.f32) || isNaN(b.f32); break;
      default:
         assert(0);
      }
      break;
   case F64_TYPE: 
      switch (cmpop) {
      case EQ_OPTION: t = (a.f64 == b.f64) && !isNaN(a.f64) && !isNaN(b.f64); break;
      case NE_OPTION: t = (a.f64 != b.f64) && !isNaN(a.f64) && !isNaN(b.f64); break;
      case LT_OPTION: t = (a.f64 < b.f64 ) && !isNaN(a.f64) && !isNaN(b.f64); break;
      case LE_OPTION: t = (a.f64 <= b.f64) && !isNaN(a.f64) && !isNaN(b.f64); break;
      case GT_OPTION: t = (a.f64 > b.f64 ) && !isNaN(a.f64) && !isNaN(b.f64); break;
      case GE_OPTION: t = (a.f64 >= b.f64) && !isNaN(a.f64) && !isNaN(b.f64); break;
      case EQU_OPTION: t = (a.f64 == b.f64) || isNaN(a.f64) || isNaN(b.f64); break;
      case NEU_OPTION: t = (a.f64 != b.f64) || isNaN(a.f64) || isNaN(b.f64); break;
      case LTU_OPTION: t = (a.f64 < b.f64 ) || isNaN(a.f64) || isNaN(b.f64); break;
      case LEU_OPTION: t = (a.f64 <= b.f64) || isNaN(a.f64) || isNaN(b.f64); break;
      case GTU_OPTION: t = (a.f64 > b.f64 ) || isNaN(a.f64) || isNaN(b.f64); break;
      case GEU_OPTION: t = (a.f64 >= b.f64) || isNaN(a.f64) || isNaN(b.f64); break;
      case NUM_OPTION: t = !isNaN(a.f64) && !isNaN(b.f64); break;
      case NAN_OPTION: t = isNaN(a.f64) || isNaN(b.f64); break;
      default:
         assert(0);
      }
      break;
   default: assert(0); break;
   }

   return t;
}

void setp_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t a, b;

   int t=0;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   assert( pI->get_num_operands() < 4 ); // or need to deal with "c" operand / boolOp

   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);

   unsigned type = pI->get_type();
   unsigned cmpop = pI->get_cmpop();

   t = CmpOp(type,a,b,cmpop);

   ptx_reg_t data;
   data.pred = (t!=0);

   thread->set_operand_value(dst,data);
}

void set_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, b;

   int t=0;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   assert( pI->get_num_operands() < 4 ); // or need to deal with "c" operand / boolOp

   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);

   unsigned src_type = pI->get_type2();
   unsigned cmpop = pI->get_cmpop();

   t = CmpOp(src_type,a,b,cmpop);

   ptx_reg_t data;
   if ( isFloat(pI->get_type()) ) {
      data.f32 = (t!=0)?1.0f:0.0f;
   } else {
      data.u32 = (t!=0)?0xFFFFFFFF:0;
   }

   thread->set_operand_value(dst,data);

}

void shl_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t a, b, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);



   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case B16_TYPE:
      if ( b.u16 >= 16 )
         d.u16 = 0;
      else
         d.u16 = (unsigned short) ((a.u16 << b.u16) & 0xFFFF); 
      break;
   case B32_TYPE: 
      if ( b.u32 >= 32 )
         d.u32 = 0;
      else
         d.u32 = (unsigned) ((a.u32 << b.u32) & 0xFFFFFFFF); 
      break;
   case B64_TYPE: 
      if ( b.u32 >= 64 )
         d.u64 = 0;
      else
         d.u64 = (a.u64 << b.u64); 
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

void shr_impl( const ptx_instruction *pI, ptx_thread_info *thread )
{
   ptx_reg_t a, b, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case U16_TYPE:
   case B16_TYPE: 
      if ( b.u16 < 16 )
         d.u16 = (unsigned short) ((a.u16 >> b.u16) & 0xFFFF);
      else
         d.u16 = 0;
      break;
   case U32_TYPE:
   case B32_TYPE: 
      if ( b.u32 < 32 )
         d.u32 = (unsigned) ((a.u32 >> b.u32) & 0xFFFFFFFF);
      else
         d.u32 = 0;
      break;
   case U64_TYPE:
   case B64_TYPE: 
      if ( b.u32 < 64 )
         d.u64 = (a.u64 >> b.u64);
      else
         d.u64 = 0;
      break;
   case S16_TYPE: 
      if ( b.u16 < 16 )
         d.s64 = (a.s16 >> b.s16);
      else {
         if ( a.s16 < 0 ) {
            d.s64 = -1;
         } else {
            d.s64 = 0;
         }
      }
      break;
   case S32_TYPE: 
      if ( b.u32 < 32 )
         d.s64 = (a.s32 >> b.s32);
      else {
         if ( a.s32 < 0 ) {
            d.s64 = -1;
         } else {
            d.s64 = 0;
         }
      }
      break;
   case S64_TYPE: 
      if ( b.u64 < 64 )
         d.s64 = (a.s64 >> b.u64);
      else {
         if ( a.s64 < 0 ) {
            if ( b.s32 < 0 ) {
               d.u64 = -1;
               d.s32 = 0;
            } else {
               d.s64 = -1;
            }
         } else {
            d.s64 = 0;
         }
      }
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

void sin_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t a, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE: 
      d.f32 = sin(a.f32);
      break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0); 
      break;
   }

   thread->set_operand_value(dst,d);
}

void slct_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();
   const operand_info &src3 = pI->src3();

   ptx_reg_t a, b, c, d;

   a = thread->get_operand_value(src1);
   b = thread->get_operand_value(src2);
   c = thread->get_operand_value(src3);

   bool t = false;
   unsigned c_type = pI->get_type2();
   switch ( c_type ) {
   case S32_TYPE: t = c.s32 >= 0; break;
   case F32_TYPE: t = c.f32 >= 0; break;
   default: assert(0);
   }

   unsigned i_type = pI->get_type();

   switch ( i_type ) {
   case B16_TYPE:
   case U16_TYPE:              d.u16 = t?a.u16:b.u16; break;
   case F32_TYPE:
   case B32_TYPE:
   case U32_TYPE: d.u32 = t?a.u32:b.u32; break;
   case F64_TYPE:
   case B64_TYPE:
   case U64_TYPE: d.u64 = t?a.u64:b.u64; break;
   default: assert(0);
   }

   thread->set_operand_value(dst,d);
}

void sqrt_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t a, d;
   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   a = thread->get_operand_value(src1);

   unsigned i_type = pI->get_type();
   switch ( i_type ) {
   case F32_TYPE: 
      if ( a.f32 < 0 )
         d.f32 = nanf("");
      else
         d.f32 = sqrt(a.f32); break;
   case F64_TYPE: 
      if ( a.f64 < 0 )
         d.f64 = nan("");
      else
         d.f64 = sqrt(a.f64); break;
   default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
   }

   thread->set_operand_value(dst,d);
}

void st_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   const operand_info &dst = pI->dst();
   const operand_info &src1 = pI->src1(); //may be scalar or vector of regs
   ptx_reg_t addr_reg = thread->get_operand_value(dst);
   ptx_reg_t data;
   memory_space_t space = pI->get_space();
   unsigned vector_spec = pI->get_vector();
   unsigned type = pI->get_type();
   memory_space *mem = NULL;
   addr_t addr = addr_reg.u32;

   decode_space(space,thread,dst,mem,addr);

   size_t size;
   int t;
   type_info_key::type_decode(type,size,t);

   if (!vector_spec) {
      data = thread->get_operand_value(src1);
      mem->write(addr,size/8,&data.s64,thread,pI);
   } else {
      if (vector_spec == V2_TYPE) {
         ptx_reg_t* ptx_regs = new ptx_reg_t[2]; 
         thread->get_vector_operand_values(src1, ptx_regs, 2); 
         mem->write(addr,size/8,&ptx_regs[0].s64,thread,pI);
         mem->write(addr+size/8,size/8,&ptx_regs[1].s64,thread,pI);
         free(ptx_regs);
      }
      if (vector_spec == V3_TYPE) {
         ptx_reg_t* ptx_regs = new ptx_reg_t[3]; 
         thread->get_vector_operand_values(src1, ptx_regs, 3); 
         mem->write(addr,size/8,&ptx_regs[0].s64,thread,pI);
         mem->write(addr+size/8,size/8,&ptx_regs[1].s64,thread,pI);
         mem->write(addr+2*size/8,size/8,&ptx_regs[2].s64,thread,pI);
         free(ptx_regs);
      }
      if (vector_spec == V4_TYPE) {
         ptx_reg_t* ptx_regs = new ptx_reg_t[4]; 
         thread->get_vector_operand_values(src1, ptx_regs, 4); 
         mem->write(addr,size/8,&ptx_regs[0].s64,thread,pI);
         mem->write(addr+size/8,size/8,&ptx_regs[1].s64,thread,pI);
         mem->write(addr+2*size/8,size/8,&ptx_regs[2].s64,thread,pI);
         mem->write(addr+3*size/8,size/8,&ptx_regs[3].s64,thread,pI);
         free(ptx_regs);
      }
   }
   thread->m_last_effective_address = addr;
   thread->m_last_memory_space = space; 
}

void sub_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   ptx_reg_t data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   ptx_reg_t src1_data = thread->get_operand_value(src1);
   ptx_reg_t src2_data = thread->get_operand_value(src2);

   unsigned i_type = pI->get_type();

   switch ( i_type ) {
   case S8_TYPE:
   case S16_TYPE:
   case S32_TYPE:
   case S64_TYPE: 
      data.s64 = src1_data.s64 - src2_data.s64; break;
   case U8_TYPE:
   case U16_TYPE:
   case U32_TYPE:
   case U64_TYPE: 
   case B8_TYPE:
   case B16_TYPE:
   case B32_TYPE:
   case B64_TYPE:
      data.u64 = src1_data.u64 - src2_data.u64; break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE: data.f32 = src1_data.f32 - src2_data.f32; break;
   case F64_TYPE: data.f64 = src1_data.f64 - src2_data.f64; break;
   default: assert(0); break;
   }

   thread->set_operand_value(dst,data);
}

void subc_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void suld_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void sured_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void sust_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void suq_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

ptx_reg_t* ptx_tex_regs = NULL;

union intfloat {
   int a;
   float b;
};

float reduce_precision( float x, unsigned bits )
{
   intfloat tmp;
   tmp.b = x;
   int v = tmp.a;
   int man = v & ((1<<23)-1);
   int mask =  ((1<<bits)-1) << (23-bits);
   int nv = (v & ((-1)-((1<<23)-1))) | (mask&man);
   tmp.a = nv;
   float result = tmp.b;
   return result;
}

unsigned wrap( unsigned x, unsigned y, unsigned mx, unsigned my, size_t elem_size )
{
   unsigned nx = (mx+x)%mx;
   unsigned ny = (my+y)%my;
   return nx + mx*ny;
}

unsigned clamp( unsigned x, unsigned y, unsigned mx, unsigned my, size_t elem_size )
{
   unsigned nx = x;
   while (nx >= mx) nx -= elem_size;
   unsigned ny = (y >= my)? my - 1 : y;
   return nx + mx*ny;
}

typedef unsigned (*texAddr_t) (unsigned x, unsigned y, unsigned mx, unsigned my, size_t elem_size);
float tex_linf_sampling(memory_space* mem, unsigned tex_array_base, 
                        int x, int y, unsigned int width, unsigned int height, size_t elem_size,
                        float alpha, float beta, texAddr_t b_lim)
{
   float Tij;
   float Ti1j;
   float Tij1;
   float Ti1j1;

   mem->read(tex_array_base + b_lim(x,y,width,height,elem_size), 4, &Tij);
   mem->read(tex_array_base + b_lim(x+elem_size,y,width,height,elem_size), 4, &Ti1j);
   mem->read(tex_array_base + b_lim(x,y+1,width,height,elem_size), 4, &Tij1);
   mem->read(tex_array_base + b_lim(x+elem_size,y+1,width,height,elem_size), 4, &Ti1j1);

   float sample = (1-alpha)*(1-beta)*Tij + 
                   alpha*(1-beta)*Ti1j +
                   (1-alpha)*beta*Tij1 +
                   alpha*beta*Ti1j1;
   
   return sample;
}

void tex_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   cudasim_n_tex_insn++;
   unsigned dimension = pI->dimension();
   const operand_info &dst = pI->dst(); //the registers to which fetched texel will be placed
   const operand_info &src1 = pI->src1(); //the name of the texture
   const operand_info &src2 = pI->src2(); //the vector registers containing coordinates of the texel to be fetched

   std::string texname = src1.name();
   unsigned to_type = pI->get_type();
   fflush(stdout);
   ptx_reg_t data1, data2, data3, data4;
   if (!ptx_tex_regs) ptx_tex_regs = new ptx_reg_t[4];
   thread->get_vector_operand_values(src2, ptx_tex_regs, 4); //ptx_reg should be 4 entry vector type...coordinates into texture

   assert(NameToTextureMap.find(texname) != NameToTextureMap.end());//use map to find texturerefence, then use map to find pointer to array
   struct textureReference* texref = NameToTextureMap[texname];
   assert(TextureToArrayMap.find(texref) != TextureToArrayMap.end());
   struct cudaArray* cuArray = TextureToArrayMap[texref];
   assert(TextureToInfoMap.find(texref) != TextureToInfoMap.end());
   struct textureInfo* texInfo = TextureToInfoMap[texref];

   //assume always 2D f32 input
   //access array with src2 coordinates
   memory_space *mem = g_global_mem;
   float x_f32,  y_f32;
   size_t size;
   int t;
   unsigned tex_array_base;
   unsigned int width = 0, height = 0;
   int x = 0;
   int y = 0;
   unsigned tex_array_index;
   float alpha=0, beta=0;

   type_info_key::type_decode(to_type,size,t);
   tex_array_base = cuArray->devPtr32;

   switch (dimension) {
   case GEOM_MODIFIER_1D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
         x_f32 = ptx_tex_regs[0].f32;
         if (texref->addressMode[0] == cudaAddressModeClamp) {
            x_f32 = (x_f32 > 1.0)? 1.0 : x_f32;
            x_f32 = (x_f32 < 0.0)? 0.0 : x_f32;
         } else if (texref->addressMode[0] == cudaAddressModeWrap) {
            x_f32 = x_f32 - floor(x_f32);
         }

         if( texref->filterMode == cudaFilterModeLinear ) {
            float xb = x_f32 * width - 0.5;
            alpha = xb - floor(xb);
            alpha = reduce_precision(alpha,9);
            beta = 0.0;

            x = (int)floor(xb);
            y = 0;
         } else {
            x = (int) floor(x_f32 * width);
            y = 0;
         }
      } else {
         x = ptx_tex_regs[0].u64;
      }
      width *= (cuArray->desc.w+cuArray->desc.x+cuArray->desc.y+cuArray->desc.z)/8;
      x *= (cuArray->desc.w+cuArray->desc.x+cuArray->desc.y+cuArray->desc.z)/8;
      tex_array_index = tex_array_base + x;

      break;
   case GEOM_MODIFIER_2D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
         x_f32 = reduce_precision(ptx_tex_regs[0].f32,16);
         y_f32 = reduce_precision(ptx_tex_regs[1].f32,15);

         if (texref->addressMode[0]) {//clamp
            if (x_f32<0) x_f32 = 0;
            if (x_f32>=1) x_f32 = 1 - 1/x_f32;
         } else {//wrap
            x_f32 = x_f32 - floor(x_f32);
         }
         if (texref->addressMode[1]) {//clamp
            if (y_f32<0) y_f32 = 0;
            if (y_f32>=1) y_f32 = 1 - 1/y_f32;
         } else {//wrap
            y_f32 = y_f32 - floor(y_f32);
         }

         if( texref->filterMode == cudaFilterModeLinear ) {
            float xb = x_f32 * width - 0.5;
            float yb = y_f32 * height - 0.5;
            alpha = xb - floor(xb);
            beta = yb - floor(yb);
            alpha = reduce_precision(alpha,9);
            beta = reduce_precision(beta,9);

            x = (int)floor(xb);
            y = (int)floor(yb);
         } else {
            x = (int) floor(x_f32 * width);
            y = (int) floor(y_f32 * height);
         }
      } else {
         x_f32 = ptx_tex_regs[0].f32;
         y_f32 = ptx_tex_regs[1].f32;

         alpha = x_f32 - floor(x_f32);
         beta = y_f32 - floor(y_f32);

         x = (int) x_f32;
         y = (int) y_f32;
         if (texref->addressMode[0]) {//clamp
            if (x<0) x = 0;
            if (x>= (int)width) x = width-1;
         } else {//wrap
            x = x % width;
            if (x < 0) x*= -1;
         }
         if (texref->addressMode[1]) {//clamp
            if (y<0) y = 0;
            if (y>= (int)height) y = height -1;
         } else {//wrap
            y = y % height;
            if (y < 0) y *= -1;
         }
      }

      width *= (cuArray->desc.w+cuArray->desc.x+cuArray->desc.y+cuArray->desc.z)/8;
      x *= (cuArray->desc.w+cuArray->desc.x+cuArray->desc.y+cuArray->desc.z)/8;
      tex_array_index = tex_array_base + (x + width*y);
      break;
   default:
      assert(0); break;
   }
   switch ( to_type ) {
   case U8_TYPE:
   case U16_TYPE:
   case U32_TYPE: 
   case S8_TYPE:
   case S16_TYPE:
   case S32_TYPE:
      mem->read( tex_array_index, cuArray->desc.x/8, &data1.u32);
      if (cuArray->desc.y) {
         mem->read( tex_array_index+4, cuArray->desc.y/8, &data2.u32);
         if (cuArray->desc.z) {
            mem->read( tex_array_index+8, cuArray->desc.z/8, &data3.u32);
            if (cuArray->desc.w) 
               mem->read( tex_array_index+12, cuArray->desc.w/8, &data4.u32);
         }
      }
      break;
   case U64_TYPE:
   case S64_TYPE:
      mem->read( tex_array_index, 8, &data1.u64);
      if (cuArray->desc.y) {
         mem->read( tex_array_index+4, 8, &data2.u64);
         if (cuArray->desc.z) {
            mem->read( tex_array_index+8, 8, &data3.u64);
            if (cuArray->desc.w) 
               mem->read( tex_array_index+12, 8, &data4.u64);
         }
      }
      break;
   case F16_TYPE: assert(0); break;
   case F32_TYPE:  {
      if( texref->filterMode == cudaFilterModeLinear ) {
         texAddr_t b_lim = wrap;
         if ( texref->addressMode[0] == cudaAddressModeClamp ) {
            b_lim = clamp;
         }
         size_t elem_size = (cuArray->desc.x + cuArray->desc.y + cuArray->desc.z + cuArray->desc.w) / 8;
         size_t elem_ofst = 0;

         data1.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width, height, elem_size, alpha, beta, b_lim);
         elem_ofst += cuArray->desc.x / 8; 
         if (cuArray->desc.y) {
            data2.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width, height, elem_size, alpha, beta, b_lim);
            elem_ofst += cuArray->desc.y / 8; 
            if (cuArray->desc.z) {
               data3.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width, height, elem_size, alpha, beta, b_lim);
               elem_ofst += cuArray->desc.z / 8; 
               if (cuArray->desc.w) 
                  data4.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width, height, elem_size, alpha, beta, b_lim);
            }
         }
      } else {
         mem->read( tex_array_index, cuArray->desc.x/8, &data1.f32);
         if (cuArray->desc.y) {
            mem->read( tex_array_index+4, cuArray->desc.y/8, &data2.f32);
            if (cuArray->desc.z) {
               mem->read( tex_array_index+8, cuArray->desc.z/8, &data3.f32);
               if (cuArray->desc.w) 
                  mem->read( tex_array_index+12, cuArray->desc.w/8, &data4.f32);
            }
         }
      }
   } break;
   case F64_TYPE: 
      mem->read( tex_array_index, 8, &data1.f64);
      if (cuArray->desc.y) {
         mem->read( tex_array_index+8, 8, &data2.f64);
         if (cuArray->desc.z) {
            mem->read( tex_array_index+16, 8, &data3.f64);
            if (cuArray->desc.w) 
               mem->read( tex_array_index+24, 8, &data4.f64);
         }
      }
      break;
   default: assert(0); break;
   }
   int x_block_coord, y_block_coord, memreqindex, blockoffset;

   switch (dimension) {
   case GEOM_MODIFIER_1D:
      thread->m_last_effective_address = tex_array_index;
      break;
   case GEOM_MODIFIER_2D: 
      x_block_coord = x;
      x_block_coord = x_block_coord >> (texInfo->Tx_numbits + texInfo->texel_size_numbits);

      y_block_coord = y;
      y_block_coord = y_block_coord >> texInfo->Ty_numbits;

      memreqindex = ((y_block_coord*cuArray->width/texInfo->Tx)+x_block_coord)<<6;

      blockoffset = (x%(texInfo->Tx*texInfo->texel_size) + (y%(texInfo->Ty)<<(texInfo->Tx_numbits + texInfo->texel_size_numbits)));
      memreqindex += blockoffset;
      thread->m_last_effective_address = tex_array_base + memreqindex;//tex_array_index;
      break;
   default:
      assert(0);
   }
   thread->m_last_memory_space = tex_space; 
   thread->set_vector_operand_values(dst,data1,data2,data3,data4,4);
}

void txq_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void trap_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vabsdiff_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vadd_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vmad_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vmax_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vmin_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vset_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vshl_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vshr_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }
void vsub_impl( const ptx_instruction *pI, ptx_thread_info *thread ) { inst_not_implemented(pI); }

extern unsigned g_warp_active_mask;

void vote_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{
   static bool first_in_warp = true;
   static bool and_all;
   static bool or_all;
   static std::list<ptx_thread_info*> threads_in_warp;
   static unsigned last_tid;

   if( first_in_warp ) {
      first_in_warp = false;
      threads_in_warp.clear();
      and_all = true;
      or_all = false;
      unsigned mask=0x80000000;
      unsigned offset=31;
      while( mask && ((mask & g_warp_active_mask)==0) ) {
         mask = mask>>1;
         offset--;
      }
      last_tid = (thread->get_hw_tid() - (thread->get_hw_tid()%pI->warp_size())) + offset;
   }

   ptx_reg_t src1_data;
   const operand_info &src1 = pI->src1();
   src1_data = thread->get_operand_value(src1);

   bool pred_value = src1_data.pred;
   bool invert = src1.is_neg_pred();

   threads_in_warp.push_back(thread);
   and_all &= (invert ^ pred_value);
   or_all |= (invert ^ pred_value);

   // TODO: determine last active thread in warp...
   if( thread->get_hw_tid() == last_tid ) {
      bool pred_value = false; 

      switch( pI->vote_mode() ) {
      case ptx_instruction::vote_any: pred_value = or_all; break;
      case ptx_instruction::vote_all: pred_value = and_all; break;
      case ptx_instruction::vote_uni: pred_value = (or_all ^ and_all); break;
      default:
         abort();
      }
      ptx_reg_t data;
      data.pred = pred_value?1:0;

      for( std::list<ptx_thread_info*>::iterator t=threads_in_warp.begin(); t!=threads_in_warp.end(); ++t ) {
         const operand_info &dst = pI->dst();
         (*t)->set_operand_value(dst,data);
      }
      first_in_warp = true;
   }
}

void xor_impl( const ptx_instruction *pI, ptx_thread_info *thread ) 
{ 
   ptx_reg_t src1_data, src2_data, data;

   const operand_info &dst  = pI->dst();
   const operand_info &src1 = pI->src1();
   const operand_info &src2 = pI->src2();

   src1_data = thread->get_operand_value(src1);
   src2_data = thread->get_operand_value(src2);

   data.u64 = src1_data.u64 ^ src2_data.u64;

   thread->set_operand_value(dst,data);
}

void inst_not_implemented( const ptx_instruction * pI ) 
{
   printf("GPGPU-Sim PTX: ERROR (%s:%u) instruction \"%s\" not (yet) implemented\n",
          pI->source_file(), 
          pI->source_line(), 
          pI->get_opcode_cstr() );
   abort();
}

void print_instruction(const ptx_instruction *instruction)
{
}

