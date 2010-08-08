/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Dan O'Connor, Joey Ting, Henry Wong and the 
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

#ifndef ptx_ir_INCLUDED
#define ptx_ir_INCLUDED

#include "../abstract_hardware_model.h"

#ifdef __cplusplus

   #include <cstdlib>
   #include <cstring>
   #include <string>
   #include <list>
   #include <map>
   #include <vector>
   #include <assert.h>

   #include "ptx.tab.h"
   #include "ptx_sim.h"
   #include "dram_callback.h"

   #include "memory.h"

class type_info_key {
public:
   type_info_key()
   {
      m_init = false;
   }
   type_info_key( memory_space_t space_spec, int scalar_type_spec, int vector_spec, int alignment_spec, int extern_spec, int array_dim )
   {
      m_init = true;
      m_space_spec = space_spec; 
      m_scalar_type_spec = scalar_type_spec;
      m_vector_spec = vector_spec;
      m_alignment_spec = alignment_spec;
      m_extern_spec = extern_spec;
      m_array_dim = array_dim;
      m_is_function = 0;
   } 
   void set_is_func()
   { 
      assert(!m_init);
      m_init = true;
      m_space_spec = undefined_space; 
      m_scalar_type_spec = 0;
      m_vector_spec = 0;
      m_alignment_spec = 0;
      m_extern_spec = 0;
      m_array_dim = 0; 
      m_is_function = 1;
   }

   void set_array_dim( int array_dim ) { m_array_dim = array_dim; }
   int get_array_dim() const { assert(m_init); return m_array_dim; }

   bool is_reg() const { return m_space_spec == reg_space;} 
   bool is_param_kernel() const { return m_space_spec == param_space_kernel;}
   bool is_param_local() const { return m_space_spec == param_space_local; }
   bool is_param_unclassified() const { return m_space_spec == param_space_unclassified; }
   bool is_global() const { return m_space_spec == global_space;}
   bool is_local() const { return m_space_spec == local_space;}
   bool is_shared() const { return m_space_spec == shared_space;}
   bool is_const() const { return m_space_spec.get_type() == const_space;}
   bool is_tex() const { return m_space_spec == tex_space;}
   bool is_func_addr() const { return m_is_function?true:false; }
   int  scalar_type() const { return m_scalar_type_spec;}
   unsigned type_decode( size_t &size, int &t ) const;
   static unsigned type_decode( int type, size_t &size, int &t );
   memory_space_t get_memory_space() const { return m_space_spec; }
private:
   bool m_init;
   memory_space_t m_space_spec; 
   int m_scalar_type_spec;
   int m_vector_spec;
   int m_alignment_spec;
   int m_extern_spec;
   int m_array_dim;
   int m_is_function;

   friend class type_info_key_compare;
};

class symbol_table;

struct type_info_key_compare {
   bool operator()( const type_info_key &a, const type_info_key &b ) const
   {
      assert( a.m_init && b.m_init );
      if ( a.m_space_spec < b.m_space_spec ) return true;
      if ( a.m_scalar_type_spec < b.m_scalar_type_spec ) return true;
      if ( a.m_vector_spec < b.m_vector_spec ) return true;
      if ( a.m_alignment_spec < b.m_alignment_spec ) return true;
      if ( a.m_extern_spec < b.m_extern_spec ) return true;
      if ( a.m_array_dim < b.m_array_dim ) return true;
      if ( a.m_is_function < b.m_is_function ) return true;

      return false;
   }
};

class type_info {
public:
   type_info( symbol_table *scope, type_info_key t )
   {
      m_type_info = t;
   }
   const type_info_key &get_key() const { return m_type_info;}

private:
   symbol_table *m_scope;
   type_info_key m_type_info;
};

enum operand_type {
   reg_t, vector_t, builtin_t, address_t, memory_t, float_op_t, double_op_t, int_t, 
   unsigned_t, symbolic_t, label_t, v_reg_t, v_float_op_t, v_double_op_t,
   v_int_t, v_unsigned_t
};

class operand_info;

class symbol {
public:
   symbol( const char *name, const type_info *type, const char *location, unsigned size ) 
   {
      m_uid = get_uid();
      m_name = name;
      m_decl_location = location;
      m_type = type;
      m_size = size;
      m_address_valid = false;
      m_is_label = false;
      m_is_shared = false;
      m_is_const = false;
      m_is_global = false;
      m_is_local = false;
      m_is_param_local = false;
      m_is_tex = false;
      m_is_func_addr = false;
      m_reg_num_valid = false;
      m_function = NULL;
      if ( type ) m_is_shared = type->get_key().is_shared();
      if ( type ) m_is_const = type->get_key().is_const();
      if ( type ) m_is_global = type->get_key().is_global();
      if ( type ) m_is_local = type->get_key().is_local();
      if ( type ) m_is_param_local = type->get_key().is_param_local();
      if ( type ) m_is_tex = type->get_key().is_tex();
      if ( type ) m_is_func_addr = type->get_key().is_func_addr();
   }
   unsigned get_size_in_bytes() const
   {
      return m_size;
   }
   const std::string &name() const { return m_name;}
   const std::string &decl_location() const { return m_decl_location;} 
   const type_info *type() const { return m_type;}
   addr_t get_address() const 
   { 
      assert( m_is_label || !m_type->get_key().is_reg() ); // todo : other assertions
      assert( m_address_valid );
      return m_address;
   }
   function_info *get_pc() const
   {
      return m_function;
   }
   void set_regno( unsigned regno, unsigned arch_regno )
   {
      m_reg_num_valid = true;
      m_reg_num = regno;
      m_arch_reg_num = arch_regno;
   }

   void set_address( addr_t addr )
   {
      m_address_valid = true;
      m_address = addr;
   }
   void set_label_address( addr_t addr)
   {
      m_address_valid = true;
      m_address = addr;
      m_is_label = true;
   }
   void set_function( function_info *func )
   {
      m_function = func;
      m_is_func_addr = true; 
   }

   bool is_label() const { return m_is_label;}
   bool is_shared() const { return m_is_shared;}
   bool is_const() const { return m_is_const;}
   bool is_global() const { return m_is_global;}
   bool is_local() const { return m_is_local;}
   bool is_param_local() const { return m_is_param_local; }
   bool is_tex() const { return m_is_tex;}
   bool is_func_addr() const { return m_is_func_addr; }
   bool is_reg() const {
      if( m_type == NULL ) 
         return false;
      return m_type->get_key().is_reg(); 
   }

   void add_initializer( const std::list<operand_info> &init );
   bool has_initializer() const 
   {
      return m_initializer.size() > 0; 
   }
   std::list<operand_info> get_initializer() const
   {
      return m_initializer;
   }
   unsigned reg_num() const
   {
      assert( m_reg_num_valid );
      return m_reg_num; 
   }
   unsigned arch_reg_num() const
   {
      assert( m_reg_num_valid );
      return m_arch_reg_num; 
   }
   void print_info(FILE *fp) const;

private:
   unsigned get_uid();
   unsigned m_uid;
   const type_info *m_type;
   unsigned m_size; // in bytes
   std::string m_name;
   std::string m_decl_location;

   unsigned m_address;
   function_info *m_function; // used for function symbols

   bool m_address_valid;
   bool m_is_label;
   bool m_is_shared;
   bool m_is_const;
   bool m_is_global;
   bool m_is_local;
   bool m_is_param_local;
   bool m_is_tex;
   bool m_is_func_addr;
   unsigned m_reg_num; 
   unsigned m_arch_reg_num; 
   bool m_reg_num_valid; 

   std::list<operand_info> m_initializer;
   static unsigned sm_next_uid;
};

class symbol_table {
public:
   symbol_table();
   symbol_table( const char *scope_name, unsigned entry_point, symbol_table *parent );
   void set_name( const char *name );
   const ptx_version &get_ptx_version() const;
   void set_ptx_version( float ver, unsigned ext );
   symbol* lookup( const char *identifier );
   std::string get_scope_name() const { return m_scope_name; }
   symbol *add_variable( const char *identifier, const type_info *type, unsigned size, const char *filename, unsigned line );
   void add_function( function_info *func );
   bool add_function_decl( const char *name, int entry_point, function_info **func_info, symbol_table **symbol_table );
   type_info *add_type( memory_space_t space_spec, int scalar_type_spec, int vector_spec, int alignment_spec, int extern_spec );
   type_info *add_type( function_info *func );
   type_info *get_array_type( type_info *base_type, unsigned array_dim ); 
   void set_label_address( const symbol *label, unsigned addr );
   unsigned next_reg_num() { return ++m_reg_allocator;}
   addr_t get_shared_next() { return m_shared_next;}
   addr_t get_global_next() { return m_global_next;}
   addr_t get_local_next() { return m_local_next;}
   addr_t get_tex_next() { return m_tex_next;}
   void  alloc_shared( unsigned num_bytes ) { m_shared_next += num_bytes;}
   void  alloc_global( unsigned num_bytes ) { m_global_next += num_bytes;}
   void  alloc_local( unsigned num_bytes ) { m_local_next += num_bytes;}
   void  alloc_tex( unsigned num_bytes ) { m_tex_next += num_bytes;}

   typedef std::list<symbol*>::iterator iterator;

   iterator global_iterator_begin() { return m_globals.begin();}
   iterator global_iterator_end() { return m_globals.end();}

   iterator const_iterator_begin() { return m_consts.begin();}
   iterator const_iterator_end() { return m_consts.end();}

   void dump();
private:
   unsigned m_reg_allocator;
   unsigned m_shared_next;
   unsigned m_const_next;
   unsigned m_global_next;
   unsigned m_local_next;
   unsigned m_tex_next;

   symbol_table *m_parent;
   ptx_version m_ptx_version;
   std::string m_scope_name;
   std::map<std::string, symbol *> m_symbols; //map from name of register to pointers to the registers
   std::map<type_info_key,type_info*,type_info_key_compare>  m_types;
   std::list<symbol*> m_globals;
   std::list<symbol*> m_consts;
   std::map<std::string,function_info*> m_function_info_lookup;
   std::map<std::string,symbol_table*> m_function_symtab_lookup;
};

class operand_info {
public:
   operand_info()
   {
      m_uid = get_uid();
      m_valid = false;
   }
   operand_info( const symbol *addr )
   {
      m_uid = get_uid();
      m_valid = true;
      if ( addr->is_label() ) {
         m_type = label_t;
      } else if ( addr->is_shared() ) {
         m_type = symbolic_t;
      } else if ( addr->is_const() ) {
         m_type = symbolic_t;
      } else if ( addr->is_global() ) {
         m_type = symbolic_t;
      } else if ( addr->is_local() ) {
         m_type = symbolic_t;
      } else if ( addr->is_param_local() ) {
         m_type = symbolic_t;
      } else if ( addr->is_tex() ) {
         m_type = symbolic_t;
      } else if ( addr->is_func_addr() ) {
         m_type = symbolic_t;
      } else {
         m_type = reg_t;
      }
      m_value.m_symbolic = addr;
      m_addr_offset = 0;
      m_vector = false;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( int builtin_id, int dim_mod )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = false;
      m_type = builtin_t;
      m_value.m_int = builtin_id;
      m_addr_offset = dim_mod;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( const symbol *addr, int offset )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = false;
      m_type = address_t;
      m_value.m_symbolic = addr;
      m_addr_offset = offset;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( unsigned x )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = false;
      m_type = unsigned_t;
      m_value.m_unsigned = x;
      m_addr_offset = 0;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( int x )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = false;
      m_type = int_t;
      m_value.m_int = x;
      m_addr_offset = 0;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( float x )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = false;
      m_type = float_op_t;
      m_value.m_float = x;
      m_addr_offset = 0;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( double x )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = false;
      m_type = double_op_t;
      m_value.m_double = x;
      m_addr_offset = 0;
      m_neg_pred = false;
      m_is_return_var = false;
   }
   operand_info( const symbol *s1, const symbol *s2, const symbol *s3, const symbol *s4 )
   {
      m_uid = get_uid();
      m_valid = true;
      m_vector = true;
      m_type = vector_t;
      m_value.m_vector_symbolic = new const symbol*[4];
      m_value.m_vector_symbolic[0] = s1;
      m_value.m_vector_symbolic[1] = s2;
      m_value.m_vector_symbolic[2] = s3;
      m_value.m_vector_symbolic[3] = s4;
      m_addr_offset = 0;
      m_neg_pred = false;
      m_is_return_var = false;
   }

   void make_memory_operand() { m_type = memory_t;}
   void set_return() { m_is_return_var = true; }

   const std::string &name() const
   {
      assert( m_type == symbolic_t || m_type == reg_t || m_type == address_t || m_type == memory_t || m_type == label_t);
      return m_value.m_symbolic->name();
   }

   unsigned get_vect_nelem() const
   {
      assert( is_vector() );
      if( !m_value.m_vector_symbolic[0] ) return 0;
      if( !m_value.m_vector_symbolic[1] ) return 1;
      if( !m_value.m_vector_symbolic[2] ) return 2;
      if( !m_value.m_vector_symbolic[3] ) return 3;
      return 4;
   }

   const symbol* vec_symbol(int idx) const 
   {
      assert(idx < 4);
      return m_value.m_vector_symbolic[idx];
   }

   const std::string &vec_name1() const
   {
      assert( m_type == vector_t);
      return m_value.m_vector_symbolic[0]->name();
   }

   const std::string &vec_name2() const
   {
      assert( m_type == vector_t);
      return m_value.m_vector_symbolic[1]->name();
   }

   const std::string &vec_name3() const
   {
      assert( m_type == vector_t);
      return m_value.m_vector_symbolic[2]->name();
   }

   const std::string &vec_name4() const
   {
      assert( m_type == vector_t);
      return m_value.m_vector_symbolic[3]->name();
   }

   bool is_reg() const
   {
      if ( m_type == reg_t ) {
         return true;
      }
      if ( m_type != symbolic_t ) {
         return false;
      }
      return m_value.m_symbolic->type()->get_key().is_reg();
   }
   bool is_param_local() const
   {
      if ( m_type != symbolic_t ) 
         return false;
      return m_value.m_symbolic->type()->get_key().is_param_local();
   }

   bool is_vector() const
   {
      if ( m_vector) return true;
      return false;
   }
   int reg_num() const { return m_value.m_symbolic->reg_num();}
   int reg1_num() const { return m_value.m_vector_symbolic[0]->reg_num();}
   int reg2_num() const { return m_value.m_vector_symbolic[1]->reg_num();}
   int reg3_num() const { return m_value.m_vector_symbolic[2]?m_value.m_vector_symbolic[2]->reg_num():0; }
   int reg4_num() const { return m_value.m_vector_symbolic[3]?m_value.m_vector_symbolic[3]->reg_num():0; }
   int arch_reg_num() const { return m_value.m_symbolic->arch_reg_num(); }
   int arch_reg_num(unsigned n) const { return (m_value.m_vector_symbolic[n])? m_value.m_vector_symbolic[n]->arch_reg_num() : -1; }
   bool is_label() const { return m_type == label_t;}
   bool is_builtin() const { return m_type == builtin_t;}
   bool is_memory_operand() const { return m_type == memory_t;}
   bool is_literal() const { return m_type == int_t ||
      m_type == float_op_t ||
      m_type == double_op_t ||
      m_type == unsigned_t;} 
   bool is_shared() const {
      if ( !(m_type == symbolic_t || m_type == address_t || m_type == memory_t) ) {
         return false;
      }
      return  m_value.m_symbolic->is_shared();
   }
   bool is_const() const { return m_value.m_symbolic->is_const();}
   bool is_global() const { return m_value.m_symbolic->is_global();}
   bool is_local() const { return m_value.m_symbolic->is_local();}
   bool is_tex() const { return m_value.m_symbolic->is_tex();}
   bool is_return_var() const { return m_is_return_var; }

   bool is_function_address() const
   {
      if( m_type != symbolic_t ) {
         return false;
      }
      return m_value.m_symbolic->is_func_addr();
   }

   ptx_reg_t get_literal_value() const
   {
      ptx_reg_t result;
      switch ( m_type ) {
      case int_t:         result.s32 = m_value.m_int; break;
      case float_op_t:    result.f32 = m_value.m_float; break;
      case double_op_t:   result.f64 = m_value.m_double; break; 
      case unsigned_t:    result.u32 = m_value.m_unsigned; break;
      default:
         assert(0);
         break;
      } 
      return result;
   }
   int get_int() const { return m_value.m_int;}
   int get_addr_offset() const { return m_addr_offset;}
   const symbol *get_symbol() const { return m_value.m_symbolic;}
   void set_type( enum operand_type type ) 
   {
      m_type = type;
   }
   enum operand_type get_type() const {
      return m_type;
   }
   void set_neg_pred()
   {
      assert( m_valid );
      m_neg_pred = true;
   }
   bool is_neg_pred() const { return m_neg_pred; }
   bool is_valid() const { return m_valid; }

private:
   unsigned m_uid;
   bool m_valid;
   bool m_vector;
   enum operand_type m_type;

   union {
      int             m_int;
      unsigned int    m_unsigned;
      float           m_float;
      double          m_double;
      int             m_vint[4];
      unsigned int    m_vunsigned[4];
      float           m_vfloat[4];
      double          m_vdouble[4];
      const symbol*   m_symbolic;
      const symbol**  m_vector_symbolic;
   } m_value;

   int m_addr_offset;

   bool m_neg_pred;
   bool m_is_return_var;

   static unsigned sm_next_uid;
   unsigned get_uid();
};

extern const char *g_opcode_string[];
extern unsigned g_num_ptx_inst_uid;
struct basic_block_t {
   basic_block_t( unsigned ID, ptx_instruction *begin, ptx_instruction *end, bool entry, bool ex)
   {
      bb_id = ID;
      ptx_begin = begin;
      ptx_end = end;
      is_entry=entry;
      is_exit=ex;
      immediatepostdominator_id = -1;
   }

   ptx_instruction* ptx_begin;
   ptx_instruction* ptx_end;
   std::set<int> predecessor_ids; //indices of other basic blocks in m_basic_blocks array
   std::set<int> successor_ids;
   std::set<int> postdominator_ids;
   std::set<int> dominator_ids;
   std::set<int> Tmp_ids;
   int immediatepostdominator_id;
   bool is_entry;
   bool is_exit;
   unsigned bb_id;
};

struct gpgpu_recon_t {
   address_type source_pc;
   address_type target_pc;
};

class ptx_instruction {
public:
   ptx_instruction( int opcode, 
                    const symbol *pred, 
                    int neg_pred, 
                    symbol *label,
                    const std::list<operand_info> &operands, 
                    const operand_info &return_var,
                    const std::list<int> &options, 
                    const std::list<int> &scalar_type,
                    memory_space_t space_spec,
                    const char *file, 
                    unsigned line,
                    const char *source );
   void print_insn() const;
   void print_insn( FILE *fp ) const;
   unsigned uid() const { return m_uid;}
   int get_opcode() { return m_opcode;}
   const char *get_opcode_cstr() const 
   {
      if ( m_opcode != -1 ) {
         return g_opcode_string[m_opcode]; 
      } else {
         return "label";
      }
   }
   const char *source_file() const { return m_source_file.c_str();} 
   unsigned source_line() const { return m_source_line;}
   unsigned get_num_operands() const { return m_operands.size();}
   bool has_pred() const { return m_pred != NULL;}
   operand_info get_pred() const { return operand_info( m_pred );}
   bool get_pred_neg() const { return m_neg_pred;}
   const char *get_source() const { return m_source.c_str();}

   typedef std::vector<operand_info>::const_iterator const_iterator;

   const_iterator op_iter_begin() const 
   { 
      return m_operands.begin();
   }

   const_iterator op_iter_end() const 
   { 
      return m_operands.end();
   }

   const operand_info &dst() const 
   { 
      assert( !m_operands.empty() );
      return m_operands[0];
   }

   const operand_info &func_addr() const
   {
      assert( !m_operands.empty() );
      if( !m_operands[0].is_return_var() ) {
         return m_operands[0];
      } else {
         assert( m_operands.size() >= 2 );
         return m_operands[1];
      }
   }

   operand_info &dst() 
   { 
      assert( !m_operands.empty() );
      return m_operands[0];
   }

   const operand_info &src1() const 
   { 
      assert( m_operands.size() > 1 );
      return m_operands[1];
   }

   const operand_info &src2() const 
   { 
      assert( m_operands.size() > 2 );
      return m_operands[2];
   }

   const operand_info &src3() const 
   { 
      assert( m_operands.size() > 3 );
      return m_operands[3];
   }

   const operand_info &operand_lookup( unsigned n ) const
   {
      assert( n < m_operands.size() );
      return m_operands[n];
   }
   bool has_return() const
   {
      return m_return_var.is_valid();
   }

   memory_space_t get_space() const { return m_space_spec;}
   unsigned get_vector() const { return m_vector_spec;}
   unsigned get_atomic() const { return m_atomic_spec;}

   int get_type() const 
   {
      assert( !m_scalar_type.empty() );
      return m_scalar_type.front();
   }

   int get_type2() const 
   {
      assert( m_scalar_type.size()==2 );
      return m_scalar_type.back();
   }

   void assign_bb(basic_block_t* basic_block) //assign instruction to a basic block
   {
      m_basic_block = basic_block;
   }
   basic_block_t* get_bb() { return m_basic_block;}
   void set_m_instr_mem_index(unsigned index) {
      m_instr_mem_index = index; 
   }
   void set_PC( addr_t PC )
   {
       m_PC = PC;
   }
   addr_t get_PC() const
   {
       return m_PC;
   }

   unsigned get_m_instr_mem_index() { return m_instr_mem_index;}
   unsigned get_cmpop() const { return m_compare_op;}
   const symbol *get_label() const { return m_label;}
   bool is_label() const { if(m_label){ assert(m_opcode==-1);return true;} return false;}
   bool is_hi() const { return m_hi;}
   bool is_lo() const { return m_lo;}
   bool is_wide() const { return m_wide;}
   bool is_uni() const { return m_uni;}
   bool is_to() const { return m_to_option; }
   unsigned cache_option() const { return m_cache_option; }
   unsigned rounding_mode() const { return m_rounding_mode;}
   unsigned saturation_mode() const { return m_saturation_mode;}
   unsigned dimension() const { return m_geom_spec;}
   enum vote_mode_t { vote_any, vote_all, vote_uni };
   enum vote_mode_t vote_mode() const { return m_vote_mode; }

   unsigned warp_size() const { return m_warp_size; }
   int membar_level() const { return m_membar_level; }
private:
   basic_block_t        *m_basic_block;
   unsigned          m_uid;
   addr_t            m_PC;
   std::string             m_source_file;
   unsigned                m_source_line;
   std::string          m_source;
   unsigned             m_warp_size;

   const symbol           *m_pred;
   bool                    m_neg_pred;
   int                     m_opcode;
   const symbol           *m_label;
   std::vector<operand_info> m_operands;
   operand_info m_return_var;

   std::list<int>          m_options;
   bool                m_wide;
   bool                m_hi;
   bool                m_lo;
   bool           m_uni; //if branch instruction, this evaluates to true for uniform branches (ie jumps)
   bool                m_to_option;
   unsigned            m_cache_option;
   unsigned            m_rounding_mode;
   unsigned            m_compare_op;
   unsigned            m_saturation_mode;

   std::list<int>          m_scalar_type;
   memory_space_t m_space_spec;
   int m_geom_spec;
   int m_vector_spec;
   int m_atomic_spec;
   enum vote_mode_t m_vote_mode;
   int m_membar_level;
   int m_instr_mem_index; //index into m_instr_mem array
};

class param_info {
public:
   param_info() { m_valid = false; m_value_set=false; m_size = 0; }
   param_info( std::string name, int type, size_t size ) 
   {
      m_valid = true;
      m_value_set = false;
      m_name = name;
      m_type = type;
      m_size = size;
   }
   void add_data( param_t v ) { 
      m_value_set = true;
      m_value = v;
   }
   std::string get_name() const { assert(m_valid); return m_name; }
   int get_type() const { assert(m_valid);  return m_type; }
   param_t get_value() const { assert(m_value_set); return m_value; }
   size_t get_size() const { assert(m_valid); return m_size; }
private:
   bool m_valid;
   std::string m_name;
   int m_type;
   size_t m_size;
   bool m_value_set;
   param_t m_value;
};

class function_info {
public:
   function_info(int entry_point );
   const ptx_version &get_ptx_version() const { return m_symtab->get_ptx_version(); }
   bool is_extern() const { return m_extern; }
   void set_name(const char *name)
   {
      m_name = name;
   }
   void set_symtab(symbol_table *symtab )
   {
      m_symtab = symtab;
   }
   std::string get_name() const
   {
      return m_name;
   }
   void print_insn( unsigned pc, FILE * fp ) const;
   void add_inst( const std::list<ptx_instruction*> &instructions )
   {
      m_instructions = instructions;
   }
   std::list<ptx_instruction*>::iterator find_next_real_instruction( std::list<ptx_instruction*>::iterator i );
   void create_basic_blocks( );

   void print_basic_blocks();

   void print_basic_block_links();
   void print_basic_block_dot();

   void connect_basic_blocks( ); //iterate across m_basic_blocks of function, connecting basic blocks together

   //iterate across m_basic_blocks of function, 
   //finding postdominator blocks, using algorithm of
   //Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14 
   void find_postdominators( );

   //iterate across m_basic_blocks of function, 
   //finding immediate postdominator blocks, using algorithm of
   //Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15 
   void find_ipostdominators( );

   void print_postdominators();

   void print_ipostdominators();

   unsigned get_num_reconvergence_pairs();

   void get_reconvergence_pairs(gpgpu_recon_t* recon_points);

   unsigned get_function_size() { return m_instructions.size();}

   void ptx_assemble();
   void ptx_decode_inst( ptx_thread_info *thd, 
                         unsigned *op_type, 
                         int *i1, 
                         int *i2, 
                         int *i3,
                         int *i4,
                         int *o1,
                         int *o2,
                         int *o3,
                         int *o4,
                         int *vectorin,
                         int *vectorout,
                         int *arch_reg );
   void ptx_exec_inst( ptx_thread_info *thd, addr_t *addr, memory_space_t *space, unsigned *data_size, dram_callback_t* callback, unsigned warp_active_mask  );
   void add_param( const char *name, struct param_t value )
   {
      m_kernel_params[ name ] = value;
   }
   void add_param_name_type_size( unsigned index, std::string name, int type, size_t size );
   void add_param_data( unsigned argn, struct gpgpu_ptx_sim_arg *args );
   void add_return_var( const symbol *rv )
   {
      m_return_var_sym = rv;
   }
   void add_arg( const symbol *arg )
   {
      assert( arg != NULL );
      m_args.push_back(arg);
   }
   void remove_args()
   {
      m_args.clear();
   }
   unsigned num_args() const
   {
      return m_args.size();
   }
   const symbol* get_arg( unsigned n ) const
   {
      assert( n < m_args.size() );
      return m_args[n];
   }
   bool has_return() const
   {
      return m_return_var_sym != NULL;
   }
   const symbol *get_return_var() const
   {
      return m_return_var_sym;
   }
   const ptx_instruction *get_instruction( unsigned PC ) const
   {
      unsigned index = PC - m_start_PC;
      if( index < m_instr_mem_size ) 
         return m_instr_mem[index];
      return NULL;
   }
   addr_t get_start_PC() const
   {
       return m_start_PC;
   }

   void finalize( memory_space *param_mem );
   void list_param( FILE *fout ) const;

   const struct gpgpu_ptx_sim_kernel_info* get_kernel_info () {
      return &m_kernel_info;
   }

   const void set_kernel_info (const struct gpgpu_ptx_sim_kernel_info *info) {
      m_kernel_info = *info;
   }
   symbol_table *get_symtab()
   {
      return m_symtab;
   }

   static const ptx_instruction* pc_to_instruction(unsigned pc) 
   {
      assert(pc > 0);
      assert(pc <= s_g_pc_to_insn.size());
      return s_g_pc_to_insn[pc - 1];
   }
   unsigned local_mem_framesize() const 
   { 
      return m_local_mem_framesize; 
   }
   void set_framesize( unsigned sz )
   {
      m_local_mem_framesize = sz;
   }
   bool is_entry_point() const { return m_entry_point; }

private:
   unsigned m_uid;
   unsigned m_local_mem_framesize;
   bool m_entry_point;
   bool m_extern;
   bool m_assembled;
   std::string m_name;
   ptx_instruction **m_instr_mem;
   unsigned m_start_PC;
   unsigned m_instr_mem_size;
   std::map<std::string,param_t> m_kernel_params;
   std::map<unsigned,param_info> m_ptx_kernel_param_info;
   const symbol *m_return_var_sym;
   std::vector<const symbol*> m_args;
   std::list<ptx_instruction*> m_instructions;
   std::vector<basic_block_t*> m_basic_blocks;
   std::list<std::pair<unsigned, unsigned> > m_back_edges;
   std::map<std::string,unsigned> labels;
   unsigned num_reconvergence_pairs;

   //Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along with ___.ptx
   struct gpgpu_ptx_sim_kernel_info m_kernel_info;

   symbol_table *m_symtab;

   static std::vector<ptx_instruction*> s_g_pc_to_insn; // a direct mapping from PC to instruction
   static unsigned sm_next_uid;
};

class arg_buffer_t {
public:
   arg_buffer_t()
   {
      m_is_reg=false;
      m_is_param=false;
      m_param_value=NULL;
   }
   arg_buffer_t( const arg_buffer_t &another )
   {
      make_copy(another);
   }
   void make_copy( const arg_buffer_t &another )
   {
      m_dst = another.m_dst;
      m_src_op = another.m_src_op;
      m_is_reg = another.m_is_reg;
      m_is_param = another.m_is_param;
      m_reg_value = another.m_reg_value;
      m_param_bytes = another.m_param_bytes;
      if( m_is_param ) {
         m_param_value = malloc(m_param_bytes);
         memcpy(m_param_value,another.m_param_value,m_param_bytes);
      }
   }
   void operator=( const arg_buffer_t &another )
   {
      make_copy(another);
   }
   ~arg_buffer_t()
   {
      if( m_is_param ) 
         free(m_param_value);
   }
   arg_buffer_t( const symbol *dst_sym, const operand_info &src_op, ptx_reg_t source_value ) : m_src_op(src_op)
   {
      m_dst = dst_sym;
      if( dst_sym->is_reg() ) {
         m_is_reg = true;
         m_is_param = false;
         assert( src_op.is_reg() );
         m_reg_value = source_value;
      } else {
         m_is_param = true;
         m_is_reg = false;
         m_param_value = calloc(sizeof(ptx_reg_t),1);
         //new (m_param_value) ptx_reg_t(source_value);
         memcpy(m_param_value,&source_value,sizeof(ptx_reg_t));
         m_param_bytes = sizeof(ptx_reg_t);
      }
   }
   arg_buffer_t( const symbol *dst_sym, const operand_info &src_op, void *source_param_value_array, unsigned array_size ) : m_src_op(src_op)
   {
      m_dst = dst_sym;
      if( dst_sym->is_reg() ) {
         m_is_reg = true;
         m_is_param = false;
         assert( src_op.is_param_local() );
         assert( dst_sym->get_size_in_bytes() == array_size );
         switch( array_size ) {
         case 1: m_reg_value.u8 = *(unsigned char*)source_param_value_array; break;
         case 2: m_reg_value.u16 = *(unsigned short*)source_param_value_array; break;
         case 4: m_reg_value.u32 = *(unsigned int*)source_param_value_array; break;
         case 8: m_reg_value.u64 = *(unsigned long long*)source_param_value_array; break;
         default:
            printf("GPGPU-Sim PTX: ERROR ** source param size does not match known register sizes\n");
            break;
         }
      } else {
         // param
         m_is_param = true;
         m_is_reg = false;
         m_param_value = calloc(array_size,1);
         m_param_bytes = array_size;
         memcpy(m_param_value,source_param_value_array,array_size);
      }
   }

   bool is_reg() const { return m_is_reg; }
   ptx_reg_t get_reg() const 
   { 
      assert(m_is_reg); 
      return m_reg_value; 
   }

   const void *get_param_buffer() const
   {
      assert(m_is_param);
      return m_param_value;
   }
   size_t get_param_buffer_size() const
   {
      assert(m_is_param);
      return m_param_bytes;
   }

   const symbol *get_dst() const { return m_dst; }

private:
   // destination of copy
   const symbol *m_dst;

   // source operand
   operand_info m_src_op;

   // source information
   bool m_is_reg;
   bool m_is_param;

   // source is register
   ptx_reg_t m_reg_value;

   // source is param
   void     *m_param_value;
   unsigned  m_param_bytes;
};

typedef std::list< arg_buffer_t > arg_buffer_list_t;
arg_buffer_t copy_arg_to_buffer(ptx_thread_info * thread, operand_info actual_param_op, const symbol * formal_param);
void copy_args_into_buffer_list( const ptx_instruction * pI, 
                                 ptx_thread_info * thread, 
                                 const function_info * target_func, 
                                 arg_buffer_list_t &arg_values ); 
void copy_buffer_list_into_frame(ptx_thread_info * thread, arg_buffer_list_t &arg_values);
void copy_buffer_to_frame(ptx_thread_info * thread, const arg_buffer_t &a);

/*******************************/
// These declarations should be identical to those in ./../../cuda-sim-dev/libcuda/texture_types.h
enum cudaChannelFormatKind {
   cudaChannelFormatKindSigned,
   cudaChannelFormatKindUnsigned,
   cudaChannelFormatKindFloat
};

struct cudaChannelFormatDesc {
   int                        x;
   int                        y;
   int                        z;
   int                        w;
   enum cudaChannelFormatKind f;
};

struct cudaArray {
   void *devPtr;
   int devPtr32;
   struct cudaChannelFormatDesc desc;
   int width;
   int height;
   int size; //in bytes
   unsigned dimensions;
};

enum cudaTextureAddressMode {
   cudaAddressModeWrap,
   cudaAddressModeClamp
};

enum cudaTextureFilterMode {
   cudaFilterModePoint,
   cudaFilterModeLinear
};

enum cudaTextureReadMode {
   cudaReadModeElementType,
   cudaReadModeNormalizedFloat
};

struct textureReference {
   int                           normalized;
   enum cudaTextureFilterMode    filterMode;
   enum cudaTextureAddressMode   addressMode[2];
   struct cudaChannelFormatDesc  channelDesc;
};

/**********************************/

struct textureInfo {
   unsigned int texel_size; //size in bytes, e.g. (channelDesc.x+y+z+w)/8
   unsigned int Tx,Ty; //tiling factor dimensions of layout of texels per 64B cache block
   unsigned int Tx_numbits,Ty_numbits; //log2(T)
   unsigned int texel_size_numbits; //log2(texel_size)
};


extern function_info *g_func_info;

extern int g_error_detected;
extern symbol_table *g_entrypoint_symbol_table;
extern function_info *g_entrypoint_func_info;
extern symbol_table *g_global_symbol_table;
extern std::map<std::string,symbol_table*> g_sym_name_to_symbol_table;
void init_parser();
#define GLOBAL_HEAP_START 0x10000000
   // start allocating from this address (lower values used for allocating globals in .ptx file)

#define SHARED_MEM_SIZE_MAX (64*1024)
#define LOCAL_MEM_SIZE_MAX 1024
#define MAX_STREAMING_MULTIPROCESSORS 64
#define MAX_THREAD_PER_SM 1024
#define TOTAL_LOCAL_MEM_PER_SM (MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define TOTAL_SHARED_MEM (MAX_STREAMING_MULTIPROCESSORS*SHARED_MEM_SIZE_MAX)
#define TOTAL_LOCAL_MEM (MAX_STREAMING_MULTIPROCESSORS*MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define SHARED_GENERIC_START (GLOBAL_HEAP_START-TOTAL_SHARED_MEM)
#define LOCAL_GENERIC_START (SHARED_GENERIC_START-TOTAL_LOCAL_MEM)

#define STATIC_ALLOC_LIMIT (GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM+TOTAL_SHARED_MEM))

void gpgpu_ptx_sim_load_ptx_from_string( const char *p, unsigned source_num );
void gpgpu_ptx_assemble( std::string kname, void *kinfo );
#include "../option_parser.h"
void ptx_reg_options(option_parser_t opp);


extern "C" {
#endif 

   void start_function( int entry_point );
   void add_function_name( const char *fname );
   void init_directive_state();
   void add_directive(); 
   void end_function();
   void add_identifier( const char *s, int array_dim, unsigned array_ident );
   void add_function_arg();
   void add_scalar_type_spec( int type_spec );
   void add_scalar_operand( const char *identifier );
   void add_neg_pred_operand( const char *identifier );
   void add_variables();
   void set_variable_type();
   void add_opcode( int opcode );
   void add_pred( const char *identifier, int negate );
   void add_2vector_operand( const char *d1, const char *d2 );
   void add_3vector_operand( const char *d1, const char *d2, const char *d3 );
   void add_4vector_operand( const char *d1, const char *d2, const char *d3, const char *d4 );
   void add_option(int option );
   void add_builtin_operand( int builtin, int dim_modifier );
   void add_memory_operand( );
   void add_literal_int( int value );
   void add_literal_float( float value );
   void add_literal_double( double value );
   void add_address_operand( const char *identifier, int offset );
   void add_label( const char *idenfiier );
   void add_vector_spec(int spec );
   void add_space_spec( enum _memory_space_t spec, int value );
   void add_extern_spec();
   void add_instruction();
   void set_return();
   void add_alignment_spec( int spec );
   void add_array_initializer();
   void add_file( unsigned num, const char *filename );
   void add_version_info( float ver );
   void *reset_symtab();
   void set_symtab(void*);
   void add_pragma( const char *str );


#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3

#ifdef __cplusplus
}
#endif 

#endif
