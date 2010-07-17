/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the University of British Columbia
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

#include "ptx_ir.h"
#include "ptx.tab.h"
#include "opcodes.h"
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <assert.h>
#include <algorithm>
#include <stdarg.h>

extern unsigned g_max_regs_per_thread;
extern "C" int ptx_error( const char *s );
void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, struct cudaArray* array); //texture functions
struct cudaArray* gpgpu_ptx_sim_accessArrayofTexture(struct textureReference* texref);
void gpgpu_ptx_sim_bindNameToTexture(const char* name, struct textureReference* texref);
struct textureReference* gpgpu_ptx_sim_accessTextureofName(const char* name);
const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);
int gpgpu_ptx_sim_sizeofTexture(const char* name);

// the program intermediate representation...
symbol_table *g_global_symbol_table = NULL;
symbol_table *g_current_symbol_table = NULL;
std::list<ptx_instruction*> g_instructions;
symbol *g_last_symbol = NULL;
std::map<std::string,symbol_table*> g_sym_name_to_symbol_table;

int g_error_detected = 0;

// type specifier stuff:
int g_space_spec = -1;
int g_scalar_type_spec = -1;
int g_vector_spec = -1;
int g_alignment_spec = -1;
int g_extern_spec = 0;

// variable declaration stuff:
type_info *g_var_type = NULL;

// instruction definition stuff:
const symbol *g_pred;
int g_neg_pred;
symbol *g_label;
int g_opcode = -1;
std::list<operand_info> g_operands;
std::list<int> g_options;
std::list<int> g_scalar_type;

extern bool g_debug_ir_generation;
extern const char *g_filename;
extern int ptx_lineno;
extern int g_debug_execution;
extern int g_debug_thread_uid;


#define DPRINTF(...) \
   if( g_debug_ir_generation ) { \
      printf(" %s:%u => ",g_filename,ptx_lineno); \
      printf("   (%s:%u) ", __FILE__, __LINE__); \
      printf(__VA_ARGS__); \
      printf("\n"); \
      fflush(stdout); \
   }

unsigned g_entry_func_param_index=0;
function_info *g_func_info = NULL;
function_info *g_entrypoint_func_info = NULL;
symbol_table *g_entrypoint_symbol_table = NULL;
std::map<unsigned,std::string> g_ptx_token_decode;
operand_info g_return_var;

void init_parser()
{
   g_global_symbol_table = g_current_symbol_table = new symbol_table("global",0,NULL);
   ptx_lineno = 1;

#define DEF(X,Y) g_ptx_token_decode[X] = Y;
#include "ptx_parser_decode.def"
#undef DEF
}

void init_directive_state()
{
   DPRINTF("init_directive_state");
   g_space_spec=-1;
   g_scalar_type_spec=-1;
   g_vector_spec=-1;
   g_opcode=-1;
   g_alignment_spec = -1;
   g_extern_spec = 0;
   g_scalar_type.clear();
   g_operands.clear();
   g_last_symbol = NULL;
}

void init_instruction_state()
{
   DPRINTF("init_instruction_state");
   g_pred = NULL;
   g_neg_pred = 0;
   g_label = NULL;
   g_opcode = -1;
   g_options.clear();
   g_return_var = operand_info();
   init_directive_state();
}

extern void register_ptx_function( const char *name, function_info *impl, symbol_table *symtab );

static int g_entry_point;

void start_function( int entry_point ) 
{
   DPRINTF("start_function");
   init_directive_state();
   init_instruction_state();
   g_entry_point = entry_point;
   g_func_info = NULL;
   g_entry_func_param_index=0;
}

static bool g_in_function_definition = false;

void start_function_definition()
{
   DPRINTF("start_function_definition");
   g_in_function_definition = true;
}

char *g_add_identifier_cached__identifier = NULL;
int g_add_identifier_cached__array_dim;
int g_add_identifier_cached__array_ident;

void add_function_name( const char *name ) 
{
   DPRINTF("add_function_name %s %s", name,  ((g_entry_point==1)?"(entrypoint)":((g_entry_point==2)?"(extern)":"")));
   bool prior_decl = g_global_symbol_table->add_function_decl( name, g_entry_point, &g_func_info, &g_current_symbol_table );
   if( g_entry_point ) {
      g_entrypoint_func_info = g_func_info;
      g_entrypoint_symbol_table = g_current_symbol_table;
   }
   if( g_add_identifier_cached__identifier ) {
      add_identifier( g_add_identifier_cached__identifier,
                      g_add_identifier_cached__array_dim,
                      g_add_identifier_cached__array_ident );
      free( g_add_identifier_cached__identifier );
      g_add_identifier_cached__identifier = NULL;
      g_func_info->add_return_var( g_last_symbol );
      init_directive_state();
   }
   if( prior_decl ) {
      g_func_info->remove_args();
   }
   g_global_symbol_table->add_function( g_func_info );
}

void add_directive() 
{
   DPRINTF("add_directive");
   init_directive_state();
}

#define mymax(a,b) ((a)>(b)?(a):(b))

void gpgpu_ptx_assemble( std::string kname, void *kinfo );

void end_function() 
{
   DPRINTF("end_function");

   g_in_function_definition = false;
   init_directive_state();
   init_instruction_state();
   g_max_regs_per_thread = mymax( g_max_regs_per_thread, (g_current_symbol_table->next_reg_num()-1)); 
   g_func_info->add_inst( g_instructions );
   g_instructions.clear();
   gpgpu_ptx_assemble( g_func_info->get_name(), g_func_info );
   g_current_symbol_table = g_global_symbol_table;

   DPRINTF("function %s, PC = %d\n", g_func_info->get_name().c_str(), g_func_info->get_start_PC());
}

extern int ptx_lineno;
extern const char *g_filename;

#define parse_error(msg, ...) parse_error_impl(__FILE__,__LINE__, msg, ##__VA_ARGS__)
#define parse_assert(cond,msg, ...) parse_assert_impl((cond),__FILE__,__LINE__, msg, ##__VA_ARGS__)

void parse_error_impl( const char *file, unsigned line, const char *msg, ... )
{
   va_list ap;
   char buf[1024];
   va_start(ap,msg);
   vsnprintf(buf,1024,msg,ap);
   va_end(ap);

   g_error_detected = 1;
   printf("%s:%u: Parse error: %s (%s:%u)\n\n", g_filename, ptx_lineno, buf, file, line);
   ptx_error(NULL);
   abort();
   exit(1);
}

void parse_assert_impl( int test_value, const char *file, unsigned line, const char *msg, ... )
{
   va_list ap;
   char buf[1024];
   va_start(ap,msg);
   vsnprintf(buf,1024,msg,ap);
   va_end(ap);

   if ( test_value == 0 )
      parse_error_impl(file,line, msg);
}

extern "C" char linebuf[1024];


void set_return()
{
   parse_assert( (g_opcode == CALL_OP), "only call can have return value");
   g_operands.front().set_return();
   g_return_var = g_operands.front();
}

std::map<std::string,std::map<unsigned,const ptx_instruction*> > g_inst_lookup;

const ptx_instruction *ptx_instruction_lookup( const char *filename, unsigned linenumber )
{
   std::map<std::string,std::map<unsigned,const ptx_instruction*> >::iterator f=g_inst_lookup.find(filename);
   if( f == g_inst_lookup.end() ) 
      return NULL;
   std::map<unsigned,const ptx_instruction*>::iterator l=f->second.find(linenumber);
   if( l == f->second.end() ) 
      return NULL;
   return l->second; 
}

void add_instruction() 
{
   DPRINTF("add_instruction: %s", ((g_opcode>0)?g_opcode_string[g_opcode]:"<label>") );
   ptx_instruction *i = new ptx_instruction( g_opcode, 
                                             g_pred, 
                                             g_neg_pred, 
                                             g_label, 
                                             g_operands,
                                             g_return_var,
                                             g_options, 
                                             g_scalar_type,
                                             g_space_spec,
                                             g_filename,
                                             ptx_lineno,
                                             linebuf );
   g_instructions.push_back(i);
   g_inst_lookup[g_filename][ptx_lineno] = i;
   init_instruction_state();
}

void add_variables() 
{
   DPRINTF("add_variables");
   if ( !g_operands.empty() ) {
      assert( g_last_symbol != NULL ); 
      g_last_symbol->add_initializer(g_operands);
   }
   init_directive_state();
}

void set_variable_type()
{
   DPRINTF("set_variable_type space_spec=%s scalar_type_spec=%s", 
           g_ptx_token_decode[g_space_spec].c_str(), 
           g_ptx_token_decode[g_scalar_type_spec].c_str() );
   parse_assert( g_space_spec != -1, "variable has no space specification" );
   parse_assert( g_scalar_type_spec != -1, "variable has no type information" ); // need to extend for structs?
   g_var_type = g_current_symbol_table->add_type( g_space_spec, 
                                                  g_scalar_type_spec, 
                                                  g_vector_spec, 
                                                  g_alignment_spec, 
                                                  g_extern_spec );
}

bool check_for_duplicates( const char *identifier )
{
   const symbol *s = g_current_symbol_table->lookup(identifier);
   return ( s != NULL );
}

extern std::set<std::string>   g_globals;
extern std::set<std::string>   g_constants;

int g_func_decl = 0;
int g_ident_add_uid = 0;
unsigned g_const_alloc = 1;

void add_identifier( const char *identifier, int array_dim, unsigned array_ident ) 
{
   if( g_func_decl && (g_func_info == NULL) ) {
      // return variable decl...
      assert( g_add_identifier_cached__identifier == NULL );
      g_add_identifier_cached__identifier = strdup(identifier);
      g_add_identifier_cached__array_dim = array_dim;
      g_add_identifier_cached__array_ident = array_ident;
      return;
   }
   DPRINTF("add_identifier \"%s\" (%u)", identifier, g_ident_add_uid);
   g_ident_add_uid++;
   type_info *type = g_var_type;
   type_info_key ti = type->get_key();
   int basic_type;
   int regnum;
   size_t num_bits;
   unsigned addr, addr_pad;
   type_decode(ti.scalar_type(),num_bits,basic_type);

   bool duplicates = check_for_duplicates( identifier );
   if( duplicates ) {
      symbol *s = g_current_symbol_table->lookup(identifier);
      g_last_symbol = s;
      if( g_func_decl ) 
         return;
      std::string msg = std::string(identifier) + " was delcared previous at " + s->decl_location(); 
      parse_error(msg.c_str());
   }

   assert( g_var_type != NULL );
   switch ( array_ident ) {
   case ARRAY_IDENTIFIER:
      type = g_current_symbol_table->get_array_type(type,array_dim);
      num_bits = array_dim * num_bits;
      break;
   case ARRAY_IDENTIFIER_NO_DIM:
      type = g_current_symbol_table->get_array_type(type,(unsigned)-1);
      num_bits = 0;
      break;
   default:
      break;
   }
   g_last_symbol = g_current_symbol_table->add_variable(identifier,type,g_filename,ptx_lineno);
   switch ( g_space_spec ) {
   case REG_DIRECTIVE: {
      regnum = g_current_symbol_table->next_reg_num();
      int arch_regnum = -1;
      for (int d = 0; d < strlen(identifier); d++) {
         if (isdigit(identifier[d])) {
            sscanf(identifier + d, "%d", &arch_regnum);
            break;
         }
      }
      if (strcmp(identifier, "%sp") == 0) {
         arch_regnum = 0;
      }
      g_last_symbol->set_regno(regnum, arch_regnum);
      } break;
   case SHARED_DIRECTIVE:
      printf("GPGPU-Sim PTX: allocating shared region for \"%s\" from 0x%x to 0x%lx (shared memory space)\n",
             identifier,
             g_current_symbol_table->get_shared_next(),
             g_current_symbol_table->get_shared_next() + num_bits/8 );
      fflush(stdout);
      assert( (num_bits%8) == 0  );
      addr = g_current_symbol_table->get_shared_next();
      addr_pad = num_bits ? (((num_bits/8) - (addr % (num_bits/8))) % (num_bits/8)) : 0;
      g_last_symbol->set_address( addr+addr_pad );
      g_current_symbol_table->alloc_shared( num_bits/8 + addr_pad );
      break;
   case CONST_DIRECTIVE:
      if( array_ident == ARRAY_IDENTIFIER_NO_DIM ) {
         printf("GPGPU-Sim PTX: deferring allocation of constant region for \"%s\" (need size information)\n", identifier );
      } else {
         printf("GPGPU-Sim PTX: allocating constant region for \"%s\" from 0x%x to 0x%lx (global memory space) %u\n",
                identifier,
                g_current_symbol_table->get_global_next(),
                g_current_symbol_table->get_global_next() + num_bits/8,
                g_const_alloc++ );
         fflush(stdout);
         assert( (num_bits%8) == 0  ); 
         addr = g_current_symbol_table->get_global_next();
         addr_pad = num_bits ? (((num_bits/8) - (addr % (num_bits/8))) % (num_bits/8)) : 0;
         g_last_symbol->set_address( addr + addr_pad );
         g_current_symbol_table->alloc_global( num_bits/8 + addr_pad ); 
      }
      if( g_current_symbol_table == g_global_symbol_table ) { 
         g_constants.insert( identifier ); 
      }
      assert( g_current_symbol_table != NULL );
      g_sym_name_to_symbol_table[ identifier ] = g_current_symbol_table;
      break;
   case GLOBAL_DIRECTIVE:
      printf("GPGPU-Sim PTX: allocating global region for \"%s\" from 0x%x to 0x%lx (global memory space)\n",
             identifier,
             g_current_symbol_table->get_global_next(),
             g_current_symbol_table->get_global_next() + num_bits/8 );
      fflush(stdout);
      assert( (num_bits%8) == 0  );
      addr = g_current_symbol_table->get_global_next();
      addr_pad = num_bits ? (((num_bits/8) - (addr % (num_bits/8))) % (num_bits/8)) : 0;
      g_last_symbol->set_address( addr+addr_pad );
      g_current_symbol_table->alloc_global( num_bits/8 + addr_pad );
      g_globals.insert( identifier );
      assert( g_current_symbol_table != NULL );
      g_sym_name_to_symbol_table[ identifier ] = g_current_symbol_table;
      break;
   case LOCAL_DIRECTIVE:
      printf("GPGPU-Sim PTX: allocating local region for \"%s\" from 0x%x to 0x%lx (local memory space)\n",
             identifier,
             g_current_symbol_table->get_local_next(),
             g_current_symbol_table->get_local_next() + num_bits/8 );
      fflush(stdout);
      assert( (num_bits%8) == 0  );
      g_last_symbol->set_address( g_current_symbol_table->get_local_next() );
      g_current_symbol_table->alloc_local( num_bits/8 );
      break;
   case TEX_DIRECTIVE:
      printf("GPGPU-Sim PTX: encountered texture directive %s.\n", identifier);
      break;
   default:
      break;
   }


   if ( ti.is_param() ) {
      if( !g_in_function_definition ) {
         g_func_info->add_param_name_type_size(g_entry_func_param_index,identifier, ti.scalar_type(), num_bits );
         g_entry_func_param_index++;
      }
   }
}

void add_function_arg()
{
   if( g_func_info ) {
      DPRINTF("add_function_arg \"%s\"", g_last_symbol->name().c_str() );
      g_func_info->add_arg(g_last_symbol);
   }
}

void add_extern_spec() 
{
   DPRINTF("add_extern_spec");
   g_extern_spec = 1;
}

void add_alignment_spec( int spec )
{
   DPRINTF("add_alignment_spec");
   parse_assert( g_alignment_spec == -1, "multiple .align specifiers per variable declaration not allowed." );
   g_alignment_spec = spec;
}

void add_space_spec( int spec ) 
{
   DPRINTF("add_space_spec \"%s\"", g_ptx_token_decode[spec].c_str() );
   parse_assert( g_space_spec == -1, "multiple space specifiers not allowed." );
   g_space_spec = spec;
}

void add_vector_spec(int spec ) 
{
   DPRINTF("add_vector_spec");
   parse_assert( g_vector_spec == -1, "multiple vector specifiers not allowed." );
   g_vector_spec = spec;
}

void add_scalar_type_spec( int type_spec ) 
{
   DPRINTF("add_scalar_type_spec \"%s\"", g_ptx_token_decode[type_spec].c_str());
   g_scalar_type.push_back( type_spec );
   if ( g_scalar_type.size() > 1 ) {
      parse_assert( (g_opcode == -1) || (g_opcode == CVT_OP) || (g_opcode == SET_OP) || (g_opcode == SLCT_OP)
                    || (g_opcode == TEX_OP), 
                    "only cvt, set, slct, and tex can have more than one type specifier.");
   }
   g_scalar_type_spec = type_spec;
}

void add_label( const char *identifier ) 
{
   DPRINTF("add_label");
   symbol *s = g_current_symbol_table->lookup(identifier);
   if ( s != NULL ) {
      g_label = s;
   } else {
      g_label = g_current_symbol_table->add_variable(identifier,NULL,g_filename,ptx_lineno);
   }
}

void add_opcode( int opcode ) 
{
   g_opcode = opcode;
}

void add_pred( const char *identifier, int neg ) 
{
   DPRINTF("add_pred");
   const symbol *s = g_current_symbol_table->lookup(identifier);
   if ( s == NULL ) {
      std::string msg = std::string("predicate \"") + identifier + "\" has no declaration.";
      parse_error( msg.c_str() );
   }
   g_pred = s;
   g_neg_pred = neg;
}

void add_option( int option ) 
{
   DPRINTF("add_option");
   g_options.push_back( option );
}

void add_2vector_operand( const char *d1, const char *d2 ) 
{
   DPRINTF("add_2vector_operand");
   const symbol *s1 = g_current_symbol_table->lookup(d1);
   const symbol *s2 = g_current_symbol_table->lookup(d2);
   parse_assert( s1 != NULL && s2 != NULL, "v2 component(s) missing declarations.");
   g_operands.push_back( operand_info(s1,s2,NULL,NULL) );
}

void add_3vector_operand( const char *d1, const char *d2, const char *d3 ) 
{
   DPRINTF("add_3vector_operand");
   const symbol *s1 = g_current_symbol_table->lookup(d1);
   const symbol *s2 = g_current_symbol_table->lookup(d2);
   const symbol *s3 = g_current_symbol_table->lookup(d3);
   parse_assert( s1 != NULL && s2 != NULL && s3 != NULL, "v3 component(s) missing declarations.");
   g_operands.push_back( operand_info(s1,s2,s3,NULL) );
}

void add_4vector_operand( const char *d1, const char *d2, const char *d3, const char *d4 ) 
{
   DPRINTF("add_4vector_operand");
   const symbol *s1 = g_current_symbol_table->lookup(d1);
   const symbol *s2 = g_current_symbol_table->lookup(d2);
   const symbol *s3 = g_current_symbol_table->lookup(d3);
   const symbol *s4 = g_current_symbol_table->lookup(d4);
   parse_assert( s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL, "v4 component(s) missing declarations.");
   g_operands.push_back( operand_info(s1,s2,s3,s4) );
}

void add_builtin_operand( int builtin, int dim_modifier ) 
{
   DPRINTF("add_builtin_operand");
   g_operands.push_back( operand_info(builtin,dim_modifier) );
}

void add_memory_operand() 
{
   DPRINTF("add_memory_operand");
   assert( !g_operands.empty() );
   g_operands.back().make_memory_operand();
}

void add_literal_int( int value ) 
{
   DPRINTF("add_literal_int");
   g_operands.push_back( operand_info(value) );
}

void add_literal_float( float value ) 
{
   DPRINTF("add_literal_float");
   g_operands.push_back( operand_info(value) );
}

void add_literal_double( double value ) 
{
   DPRINTF("add_literal_double");
   g_operands.push_back( operand_info(value) );
}

void add_scalar_operand( const char *identifier ) 
{
   DPRINTF("add_scalar_operand");
   const symbol *s = g_current_symbol_table->lookup(identifier);
   if ( s == NULL ) {
      if ( g_opcode == BRA_OP ) {
         // forward branch target...
         s = g_current_symbol_table->add_variable(identifier,NULL,g_filename,ptx_lineno);
      } else {
         std::string msg = std::string("operand \"") + identifier + "\" has no declaration.";
         parse_error( msg.c_str() );
      }
   }
   g_operands.push_back( operand_info(s) );
}

void add_neg_pred_operand( const char *identifier ) 
{
   DPRINTF("add_neg_pred_operand");
   const symbol *s = g_current_symbol_table->lookup(identifier);
   if ( s == NULL ) {
       s = g_current_symbol_table->add_variable(identifier,NULL,g_filename,ptx_lineno);
   }
   operand_info op(s);
   op.set_neg_pred();
   g_operands.push_back( op );
}

void add_address_operand( const char *identifier, int offset ) 
{
   DPRINTF("add_address_operand");
   const symbol *s = g_current_symbol_table->lookup(identifier);
   if ( s == NULL ) {
      std::string msg = std::string("operand \"") + identifier + "\" has no declaration.";
      parse_error( msg.c_str() );
   }
   g_operands.push_back( operand_info(s,offset) );
}

unsigned symbol::sm_next_uid = 1;

unsigned symbol::get_uid()
{
   unsigned result = sm_next_uid++;
   return result;
}

void symbol::add_initializer( const std::list<operand_info> &init )
{
   m_initializer = init;
}

void symbol::print_info(FILE *fp) const
{
   fprintf(fp,"uid:%u, decl:%s, type:%p, ", m_uid, m_decl_location.c_str(), m_type );
   if( m_address_valid ) 
      fprintf(fp,"<address valid>, ");
   if( m_is_label )
      fprintf(fp," is_label ");
   if( m_is_shared )
      fprintf(fp," is_shared ");
   if( m_is_const )
      fprintf(fp," is_const ");
   if( m_is_global )
      fprintf(fp," is_global ");
   if( m_is_local )
      fprintf(fp," is_local ");
   if( m_is_tex )
      fprintf(fp," is_tex ");
   if( m_is_func_addr )
      fprintf(fp," is_func_addr ");
   if( m_function ) 
      fprintf(fp," %p ", m_function );
}

symbol_table::symbol_table() 
{ 
   assert(0); 
}

symbol_table::symbol_table( const char *scope_name, unsigned entry_point, symbol_table *parent )
{
   m_scope_name = std::string(scope_name);
   m_reg_allocator=0;
   m_shared_next = 0x100; // for debug with valgrind: make zero imply undefined address
   m_const_next  = 0x100; // for debug with valgrind: make zero imply undefined address
   m_global_next = 0x100; // for debug with valgrind: make zero imply undefined address
   m_local_next  = 0x100; // for debug with valgrind: make zero imply undefined address
   m_parent = parent;
   if ( m_parent ) {
      m_shared_next = m_parent->m_shared_next;
      m_global_next = m_parent->m_global_next;
   }
}

void symbol_table::set_name( const char *name )
{
   m_scope_name = std::string(name);
}

const ptx_version &symbol_table::get_ptx_version() const 
{ 
   if( m_parent == NULL ) return m_ptx_version;
   else return m_parent->get_ptx_version(); 
}

void symbol_table::set_ptx_version( float ver, unsigned ext ) 
{ 
   m_ptx_version = ptx_version(ver,ext); 
   assert( m_ptx_version.ver() < 3 );
}

symbol *symbol_table::lookup( const char *identifier ) 
{
   std::string key(identifier);
   std::map<std::string, symbol *>::iterator i = m_symbols.find(key);
   if (  i != m_symbols.end() ) {
      return i->second;
   }
   if ( m_parent ) {
      return m_parent->lookup(identifier);
   }
   return NULL;
}

symbol *symbol_table::add_variable( const char *identifier, const type_info *type, const char *filename, unsigned line )
{
   char buf[1024];
   std::string key(identifier);
   assert( m_symbols.find(key) == m_symbols.end() );
   snprintf(buf,1024,"%s:%u",filename,line);
   symbol *s = new symbol(identifier,type,buf);
   m_symbols[ key ] = s;

   if ( type != NULL && type->get_key().is_param()  ) {
      m_params.push_back(s);
   }
   if ( type != NULL && type->get_key().is_global()  ) {
      m_globals.push_back(s);
   }
   if ( type != NULL && type->get_key().is_const()  ) {
      m_consts.push_back(s);
   }

   return s;
}

void symbol_table::add_function( function_info *func )
{
   std::map<std::string, symbol *>::iterator i = m_symbols.find( func->get_name() );
   if( i != m_symbols.end() )
      return;
   char buf[1024];
   snprintf(buf,1024,"%s:%u",g_filename,ptx_lineno);
   type_info *type = add_type( func );
   symbol *s = new symbol(func->get_name().c_str(),type,buf);
   s->set_function(func);
   m_symbols[ func->get_name() ] = s;
}

bool symbol_table::add_function_decl( const char *name, int entry_point, function_info **func_info, symbol_table **sym_table )
{
   std::string key = std::string(name);
   bool prior_decl = false;
   if( m_function_info_lookup.find(key) != m_function_info_lookup.end() ) {
      *func_info = m_function_info_lookup[key];
      prior_decl = true;
   } else {
      *func_info = new function_info(entry_point);
      (*func_info)->set_name(name);
      m_function_info_lookup[key] = *func_info;
   }

   if( m_function_symtab_lookup.find(key) != m_function_symtab_lookup.end() ) {
      assert( prior_decl );
      *sym_table = m_function_symtab_lookup[key];
   } else {
      assert( !prior_decl );
      *sym_table = new symbol_table( "", entry_point, g_global_symbol_table );
      symbol *null_reg = (*sym_table)->add_variable("_",NULL,"",0); 
      null_reg->set_regno(0, 0);
      (*sym_table)->set_name(name);
      (*func_info)->set_symtab(*sym_table);
      m_function_symtab_lookup[key] = *sym_table;
      register_ptx_function(name,*func_info,*sym_table);
   }
   return prior_decl;
}

type_info *symbol_table::add_type( int space_spec, int scalar_type_spec, int vector_spec, int alignment_spec, int extern_spec )
{
   type_info_key t(space_spec,scalar_type_spec,vector_spec,alignment_spec,extern_spec,0);
   type_info *pt;
   pt = new type_info(this,t);
   return pt;
}

type_info *symbol_table::add_type( function_info *func )
{
   type_info_key t;
   type_info *pt;
   t.set_is_func();
   pt = new type_info(this,t);
   return pt;
}

type_info *symbol_table::get_array_type( type_info *base_type, unsigned array_dim ) 
{
   type_info_key t = base_type->get_key();
   t.set_array_dim(array_dim);
   type_info *pt;
   pt = m_types[t] = new type_info(this,t);
   return pt;
}

void symbol_table::set_label_address( const symbol *label, unsigned addr )
{
   std::map<std::string, symbol *>::iterator i=m_symbols.find(label->name());
   assert( i != m_symbols.end() );
   symbol *s = i->second;
   s->set_label_address(addr);
}

void symbol_table::dump()
{
   printf("\n\n");
   printf("Symbol table for \"%s\":\n", m_scope_name.c_str() );
   std::map<std::string, symbol *>::iterator i;
   for( i=m_symbols.begin(); i!=m_symbols.end(); i++ ) {
      printf("%30s : ", i->first.c_str() );
      if( i->second ) 
         i->second->print_info(stdout);
      else
         printf(" <no symbol object> ");
      printf("\n");
   }
   printf("\n");
}

unsigned operand_info::sm_next_uid=1;

unsigned operand_info::get_uid()
{
   unsigned result = sm_next_uid++;
   return result;
}

void add_array_initializer()
{
   g_last_symbol->add_initializer(g_operands);
}


std::list<ptx_instruction*>::iterator function_info::find_next_real_instruction( std::list<ptx_instruction*>::iterator i)
{
   while( (i != m_instructions.end()) && (*i)->is_label() ) 
      i++;
   return i;
}

void function_info::create_basic_blocks()
{
   std::list<ptx_instruction*> leaders;
   std::list<ptx_instruction*>::iterator i, l;

   // first instruction is a leader
   i=m_instructions.begin();
   leaders.push_back(*i);
   i++;
   while( i!=m_instructions.end() ) {
      ptx_instruction *pI = *i;
      if( pI->is_label() ) {
         leaders.push_back(pI);
         i = find_next_real_instruction(++i);
      } else {
         switch( pI->get_opcode() ) {
         case BRA_OP: case RET_OP: case EXIT_OP:
            i++;
            if( i != m_instructions.end() ) 
               leaders.push_back(*i);
            i = find_next_real_instruction(i);
            break;
         case CALL_OP:
            if( pI->has_pred() ) {
               printf("GPGPU-Sim PTX: Warning found predicated call\n");
               i++;
               if( i != m_instructions.end() ) 
                  leaders.push_back(*i);
               i = find_next_real_instruction(i);
            } else i++;
            break;
         default:
            i++;
         }
      } 
   }

   if( leaders.empty() ) {
      printf("GPGPU-Sim PTX: Function \'%s\' has no basic blocks\n", m_name.c_str());
      return;
   }

   unsigned bb_id = 0;
   l=leaders.begin();
   i=m_instructions.begin();
   m_basic_blocks.push_back( new basic_block_t(bb_id++,*find_next_real_instruction(i),NULL,1,0) );
   ptx_instruction *last_real_inst=*(l++);

   for( ; i!=m_instructions.end(); i++ ) {
      ptx_instruction *pI = *i;
      if( l != leaders.end() && *i == *l ) {
         // found start of next basic block
         m_basic_blocks.back()->ptx_end = last_real_inst;
         if( find_next_real_instruction(i) != m_instructions.end() ) { // if not bogus trailing label
            m_basic_blocks.push_back( new basic_block_t(bb_id++,*find_next_real_instruction(i),NULL,0,0) );
            last_real_inst = *find_next_real_instruction(i);
         }
         // start search for next leader
         l++;
      }
      pI->assign_bb( m_basic_blocks.back() );
      if( !pI->is_label() ) last_real_inst = pI;
   }
   m_basic_blocks.back()->ptx_end = last_real_inst;
   m_basic_blocks.push_back( /*exit basic block*/ new basic_block_t(bb_id,NULL,NULL,0,1) );
}

void function_info::print_basic_blocks()
{
   printf("Printing basic blocks for function \'%s\':\n", m_name.c_str() );
   std::list<ptx_instruction*>::iterator ptx_itr;
   unsigned last_bb=0;
   for (ptx_itr = m_instructions.begin();ptx_itr != m_instructions.end(); ptx_itr++) {
      if( (*ptx_itr)->get_bb() ) {
         if( (*ptx_itr)->get_bb()->bb_id != last_bb ) {
            printf("\n");
            last_bb = (*ptx_itr)->get_bb()->bb_id;
         }
         printf("bb_%02u\t: ", (*ptx_itr)->get_bb()->bb_id);
         (*ptx_itr)->print_insn();
         printf("\n");
      }
   }
   printf("\nSummary of basic blocks for \'%s\':\n", m_name.c_str() );
   std::vector<basic_block_t*>::iterator bb_itr;
   for (bb_itr = m_basic_blocks.begin();bb_itr != m_basic_blocks.end(); bb_itr++) {
      printf("bb_%02u\t:", (*bb_itr)->bb_id);
      if ((*bb_itr)->ptx_begin)
         printf(" first: %s\t", ((*bb_itr)->ptx_begin)->get_opcode_cstr());
      else printf(" first: NULL\t");
      if ((*bb_itr)->ptx_end) {
         printf(" last: %s\t", ((*bb_itr)->ptx_end)->get_opcode_cstr());
      } else printf(" last: NULL\t");
      printf("\n");
   }
   printf("\n");
}

void function_info::print_basic_block_links()
{
   printf("Printing basic blocks links for function \'%s\':\n", m_name.c_str() );
   std::vector<basic_block_t*>::iterator bb_itr;
   for (bb_itr = m_basic_blocks.begin();bb_itr != m_basic_blocks.end(); bb_itr++) {
      printf("ID: %d\t:", (*bb_itr)->bb_id);
      if ( !(*bb_itr)->predecessor_ids.empty() ) {
         printf("Predecessors:");
         std::set<int>::iterator p;
         for (p= (*bb_itr)->predecessor_ids.begin();p != (*bb_itr)->predecessor_ids.end();p++) {
            printf(" %d", *p);
         }
         printf("\t");
      }
      if ( !(*bb_itr)->successor_ids.empty() ) {
         printf("Successors:");
         std::set<int>::iterator s;
         for (s= (*bb_itr)->successor_ids.begin();s != (*bb_itr)->successor_ids.end();s++) {
            printf(" %d", *s);
         }
      }
      printf("\n");
   }
}
void function_info::connect_basic_blocks( ) //iterate across m_basic_blocks of function, connecting basic blocks together
{
   std::vector<basic_block_t*>::iterator bb_itr;
   std::vector<basic_block_t*>::iterator bb_target_itr;
   basic_block_t* exit_bb = m_basic_blocks.back();

   //start from first basic block, which we know is the entry point
   bb_itr = m_basic_blocks.begin(); 
   for (bb_itr = m_basic_blocks.begin();bb_itr != m_basic_blocks.end(); bb_itr++) {
      ptx_instruction *pI = (*bb_itr)->ptx_end;
      if ((*bb_itr)->is_exit) //reached last basic block, no successors to link 
         continue;
      if (pI->get_opcode() == RET_OP || pI->get_opcode() == EXIT_OP ) {
         (*bb_itr)->successor_ids.insert(exit_bb->bb_id);
         exit_bb->predecessor_ids.insert((*bb_itr)->bb_id);
         if( pI->has_pred() ) {
            printf("GPGPU-Sim PTX: Warning detected predicated return/exit.\n");
            // if predicated, add link to next block
            unsigned next_addr = pI->get_m_instr_mem_index() + 1;
            if( next_addr < m_instr_mem_size && m_instr_mem[next_addr] ) {
               basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
               (*bb_itr)->successor_ids.insert(next_bb->bb_id);
               next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
            }
         }
         continue;
      } else if (pI->get_opcode() == BRA_OP) {
         //find successor and link that basic_block to this one
         operand_info &target = pI->dst(); //get operand, e.g. target name
         unsigned addr = labels[ target.name() ];
         ptx_instruction *target_pI = m_instr_mem[addr];
         basic_block_t *target_bb = target_pI->get_bb();
         (*bb_itr)->successor_ids.insert(target_bb->bb_id);
         target_bb->predecessor_ids.insert((*bb_itr)->bb_id);
      } 
      if ( !(pI->get_opcode()==BRA_OP && (!pI->has_pred())) ) { 
         // if basic block does not end in an unpredicated branch, 
         // then next basic block is also successor
         // (this is better than testing for .uni)
         unsigned next_addr = pI->get_m_instr_mem_index() + 1;
         basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
         (*bb_itr)->successor_ids.insert(next_bb->bb_id);
         next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
      } else
         assert(pI->get_opcode() == BRA_OP);
   }
}

void intersect( std::set<int> &A, const std::set<int> &B )
{
   // return intersection of A and B in A
   for( std::set<int>::iterator a=A.begin(); a!=A.end(); ) {    
      std::set<int>::iterator a_next = a;
      a_next++;
      if( B.find(*a) == B.end() ) {
         A.erase(*a);
         a = a_next;
      } else 
         a++;
   }
}

bool is_equal( const std::set<int> &A, const std::set<int> &B )
{
   if( A.size() != B.size() ) 
      return false;
   for( std::set<int>::iterator b=B.begin(); b!=B.end(); b++ ) 
      if( A.find(*b) == A.end() ) 
         return false;
   return true;
}

void print_set(const std::set<int> &A)
{
   std::set<int>::iterator a;
   for (a= A.begin(); a != A.end(); a++) {
      printf("%d ", (*a));
   }
   printf("\n");
}

void function_info::find_postdominators( )
{  
   // find postdominators using algorithm of Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14 
   printf("GPGPU-Sim PTX: Finding postdominators for \'%s\'...\n", m_name.c_str() );
   fflush(stdout);
   assert( m_basic_blocks.size() >= 2 ); // must have a distinquished exit block
   std::vector<basic_block_t*>::reverse_iterator bb_itr = m_basic_blocks.rbegin();
   (*bb_itr)->postdominator_ids.insert((*bb_itr)->bb_id);  // the only postdominator of the exit block is the exit
   for (++bb_itr;bb_itr != m_basic_blocks.rend();bb_itr++) { //copy all basic blocks to all postdominator lists EXCEPT for the exit block
      for (unsigned i=0; i<m_basic_blocks.size(); i++) 
         (*bb_itr)->postdominator_ids.insert(i);
   }
   bool change = true;
   while (change) {
      change = false;
      for ( int h = m_basic_blocks.size()-2/*skip exit*/; h >= 0 ; --h ) {
         assert( m_basic_blocks[h]->bb_id == (unsigned)h );
         std::set<int> T;
         for (unsigned i=0;i< m_basic_blocks.size();i++) 
            T.insert(i);
         for ( std::set<int>::iterator s = m_basic_blocks[h]->successor_ids.begin();s != m_basic_blocks[h]->successor_ids.end();s++) 
            intersect(T, m_basic_blocks[*s]->postdominator_ids);
         T.insert(h);
         if (!is_equal(T,m_basic_blocks[h]->postdominator_ids)) {
            change = true;
            m_basic_blocks[h]->postdominator_ids = T;
         }
      }
   }
}

void function_info::find_ipostdominators( )
{  
   // find immediate postdominator blocks, using algorithm of
   // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15 
   printf("GPGPU-Sim PTX: Finding immediate postdominators for \'%s\'...\n", m_name.c_str() );
   fflush(stdout);
   assert( m_basic_blocks.size() >= 2 ); // must have a distinquished exit block
   for (unsigned i=0; i<m_basic_blocks.size(); i++) { //initialize Tmp(n) to all pdoms of n except for n
      m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->postdominator_ids;
      assert( m_basic_blocks[i]->bb_id == i );
      m_basic_blocks[i]->Tmp_ids.erase(i);
   }
   for ( int n = m_basic_blocks.size()-2; n >=0;--n) {
      // point iterator to basic block before the exit
      for( std::set<int>::iterator s=m_basic_blocks[n]->Tmp_ids.begin(); s != m_basic_blocks[n]->Tmp_ids.end(); s++ ) {
         int bb_s = *s;
         for( std::set<int>::iterator t=m_basic_blocks[n]->Tmp_ids.begin(); t != m_basic_blocks[n]->Tmp_ids.end(); ) {
            std::set<int>::iterator t_next = t; t_next++; // might erase thing pointed to be t, invalidating iterator t
            if( *s == *t ) {
               t = t_next;
               continue;
            }
            int bb_t = *t;
            if( m_basic_blocks[bb_s]->postdominator_ids.find(bb_t) != m_basic_blocks[bb_s]->postdominator_ids.end() ) 
                m_basic_blocks[n]->Tmp_ids.erase(bb_t);
            t = t_next;
         }
      }
   }
   unsigned num_ipdoms=0;
   for ( int n = m_basic_blocks.size()-1; n >=0;--n) {
      assert( m_basic_blocks[n]->Tmp_ids.size() <= 1 ); 
         // if the above assert fails we have an error in either postdominator 
         // computation, the flow graph does not have a unique exit, or some other error
      if( !m_basic_blocks[n]->Tmp_ids.empty() ) {
         m_basic_blocks[n]->immediatepostdominator_id = *m_basic_blocks[n]->Tmp_ids.begin();
         num_ipdoms++;
      }
   }
   assert( num_ipdoms == m_basic_blocks.size()-1 ); 
      // the exit node does not have an immediate post dominator, but everyone else should
}

void function_info::print_postdominators()
{
   printf("Printing postdominators for function \'%s\':\n", m_name.c_str() );
   std::vector<int>::iterator bb_itr;
   for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
      printf("ID: %d\t:", i);
      for( std::set<int>::iterator j=m_basic_blocks[i]->postdominator_ids.begin(); j!=m_basic_blocks[i]->postdominator_ids.end(); j++) 
         printf(" %d", *j );
      printf("\n");
   }
}

void function_info::print_ipostdominators()
{
   printf("Printing immediate postdominators for function \'%s\':\n", m_name.c_str() );
   std::vector<int>::iterator bb_itr;
   for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
      printf("ID: %d\t:", i);
      printf("%d\n", m_basic_blocks[i]->immediatepostdominator_id);
   }
}

unsigned function_info::get_num_reconvergence_pairs()
{
   if (!num_reconvergence_pairs) {
      for (unsigned i=0; i< (m_basic_blocks.size()-1); i++) { //last basic block containing exit obviously won't have a pair
         if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
            num_reconvergence_pairs++;
         }
      }
   }
   return num_reconvergence_pairs;
}

void function_info::get_reconvergence_pairs(gpgpu_recon_t* recon_points)
{
   unsigned idx=0; //array index
   for (unsigned i=0; i< (m_basic_blocks.size()-1); i++) { //last basic block containing exit obviously won't have a pair
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("i=%d\n", i); fflush(stdout);
#endif
      if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
#ifdef DEBUG_GET_RECONVERG_PAIRS
         printf("\tbranch!\n");
         printf("\tbb_id=%d; ipdom=%d\n", m_basic_blocks[i]->bb_id, m_basic_blocks[i]->immediatepostdominator_id);
         printf("\tm_instr_mem index=%d\n", m_basic_blocks[i]->ptx_end->get_m_instr_mem_index());
         fflush(stdout);
#endif
         recon_points[idx].source_pc = m_basic_blocks[i]->ptx_end->get_PC();
#ifdef DEBUG_GET_RECONVERG_PAIRS
         printf("\trecon_points[idx].source_pc=%d\n", recon_points[idx].source_pc);
#endif
         if( m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]->ptx_begin ) {
            recon_points[idx].target_pc = m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]->ptx_begin->get_PC();
         } else {
            // reconverge after function return
            recon_points[idx].target_pc = -2;
         }
#ifdef DEBUG_GET_RECONVERG_PAIRS
         m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]->ptx_begin->print_insn();
         printf("\trecon_points[idx].target_pc=%d\n", recon_points[idx].target_pc); fflush(stdout);
#endif
         idx++;
      }
   }
}

// interface with graphviz (print the graph in DOT language) for plotting
void function_info::print_basic_block_dot()
{
   printf("Basic Block in DOT\n");
   printf("digraph %s {\n", m_name.c_str());
   std::vector<basic_block_t*>::iterator bb_itr;
   for (bb_itr = m_basic_blocks.begin();bb_itr != m_basic_blocks.end(); bb_itr++) {
      printf("\t");
      std::set<int>::iterator s;
      for (s = (*bb_itr)->successor_ids.begin();s != (*bb_itr)->successor_ids.end();s++) {
         unsigned succ_bb = *s;
         printf("%d -> %d; ", (*bb_itr)->bb_id, succ_bb );
      }
      printf("\n");
   }
   printf("}\n");
}

extern "C" void add_version_info( float ver )
{
   g_global_symbol_table->set_ptx_version(ver,0);
}


extern "C" void add_file( unsigned num, const char *filename )
{
   if( g_filename == NULL ) {
      char *b = strdup(filename);
      char *l=b;
      char *n=b;
      while( *n != '\0' ) {
          if( *n == '/' ) 
              l = n+1;
          n++;
      }

      char *p = strtok(l,".");
      char buf[1024];
      snprintf(buf,1024,"%s.ptx",p);

      char *q = strtok(NULL,".");
      if( q && !strcmp(q,"cu") ) {
          g_filename = strdup(buf);
      }

      free( b );
   }

   g_current_symbol_table = g_global_symbol_table;
}

extern "C" void *reset_symtab()
{
   void *result = g_current_symbol_table;
   g_current_symbol_table = g_global_symbol_table;
   return result;
}

extern "C" void set_symtab(void*symtab)
{
   g_current_symbol_table = (symbol_table*)symtab;
}

extern "C" void add_pragma( const char *str )
{
   printf("GPGPU-Sim: Warning -- ignoring pragma '%s'\n", str );
}

unsigned ptx_kernel_shmem_size( void *kernel_impl )
{
   function_info *f = (function_info*)kernel_impl;
   const struct gpgpu_ptx_sim_kernel_info *kernel_info = f->get_kernel_info();
   return kernel_info->smem;
}

unsigned ptx_kernel_nregs( void *kernel_impl )
{
   function_info *f = (function_info*)kernel_impl;
   const struct gpgpu_ptx_sim_kernel_info *kernel_info = f->get_kernel_info();
   return kernel_info->regs;
}
