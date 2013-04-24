// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ptx_parser_INCLUDED
#define ptx_parser_INCLUDED

#include "../abstract_hardware_model.h"
extern const char *g_filename;
extern int g_error_detected;

#ifdef __cplusplus 
class symbol_table* init_parser(const char*);
const class ptx_instruction *ptx_instruction_lookup( const char *filename, unsigned linenumber );
#endif

const char *decode_token( int type );
void read_parser_environment_variables();
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
void add_pred( const char *identifier, int negate, int predModifier );
void add_1vector_operand( const char *d1 );
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
void add_address_operand2( int offset );
void add_label( const char *idenfiier );
void add_vector_spec(int spec );
void add_space_spec( enum _memory_space_t spec, int value );
void add_ptr_spec( enum _memory_space_t spec ); 
void add_extern_spec();
void add_instruction();
void set_return();
void add_alignment_spec( int spec );
void add_array_initializer();
void add_file( unsigned num, const char *filename );
void add_version_info( float ver, unsigned ext);
void *reset_symtab();
void set_symtab(void*);
void add_pragma( const char *str );
void func_header(const char* a);
void func_header_info(const char* a);
void func_header_info_int(const char* a, int b);
void add_constptr(const char* identifier1, const char* identifier2, int offset);
void target_header(char* a);
void target_header2(char* a, char* b);
void target_header3(char* a, char* b, char* c);
void add_double_operand( const char *d1, const char *d2 );
void change_memory_addr_space( const char *identifier );
void change_operand_lohi( int lohi );
void change_double_operand_type( int addr_type );
void change_operand_neg( );
void set_immediate_operand_type( );
void version_header(double a);

#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3

#endif
