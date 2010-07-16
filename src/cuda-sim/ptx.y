/* 
 * Copyright (c) 2009 by Tor M. Aamodt, George L. Yuan and the 
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

%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token <string_value> STRING
%token <int_value>  OPCODE
%token  ALIGN_DIRECTIVE
%token  BYTE_DIRECTIVE
%token  CONST_DIRECTIVE
%token  ENTRY_DIRECTIVE
%token  EXTERN_DIRECTIVE
%token  FILE_DIRECTIVE
%token  FUNC_DIRECTIVE
%token  GLOBAL_DIRECTIVE
%token  LOCAL_DIRECTIVE
%token  LOC_DIRECTIVE
%token  PARAM_DIRECTIVE
%token  REG_DIRECTIVE
%token  SECTION_DIRECTIVE
%token  SHARED_DIRECTIVE
%token  SREG_DIRECTIVE
%token  STRUCT_DIRECTIVE
%token  SURF_DIRECTIVE
%token  TARGET_DIRECTIVE
%token  TEX_DIRECTIVE
%token  UNION_DIRECTIVE
%token  VERSION_DIRECTIVE
%token  VISIBLE_DIRECTIVE
%token  MAXNTID_DIRECTIVE
%token  <string_value> IDENTIFIER
%token  <int_value> INT_OPERAND
%token  <float_value> FLOAT_OPERAND
%token  <double_value> DOUBLE_OPERAND
%token  S8_TYPE
%token  S16_TYPE
%token  S32_TYPE
%token  S64_TYPE
%token  U8_TYPE
%token  U16_TYPE
%token  U32_TYPE
%token  U64_TYPE
%token  F16_TYPE
%token  F32_TYPE
%token  F64_TYPE
%token  B8_TYPE
%token  B16_TYPE
%token  B32_TYPE
%token  B64_TYPE
%token  PRED_TYPE
%token  V2_TYPE
%token  V3_TYPE
%token  V4_TYPE
%token  COMMA
%token  PRED
%token  EQ_OPTION
%token  NE_OPTION
%token  LT_OPTION
%token  LE_OPTION
%token  GT_OPTION
%token  GE_OPTION
%token  LO_OPTION
%token  LS_OPTION
%token  HI_OPTION
%token  HS_OPTION
%token  EQU_OPTION
%token  NEU_OPTION
%token  LTU_OPTION
%token  LEU_OPTION
%token  GTU_OPTION
%token  GEU_OPTION
%token  NUM_OPTION
%token  NAN_OPTION
%token  LEFT_SQUARE_BRACKET
%token  RIGHT_SQUARE_BRACKET
%token  WIDE_OPTION
%token  <int_value> SPECIAL_REGISTER
%token  PLUS
%token  COLON
%token  SEMI_COLON
%token  EXCLAMATION
%token	RIGHT_BRACE
%token	LEFT_BRACE
%token	EQUALS
%token  PERIOD
%token <int_value> DIMENSION_MODIFIER
%token RN_OPTION
%token RZ_OPTION
%token RM_OPTION
%token RP_OPTION
%token RNI_OPTION
%token RZI_OPTION
%token RMI_OPTION
%token RPI_OPTION
%token UNI_OPTION
%token GEOM_MODIFIER_1D
%token GEOM_MODIFIER_2D
%token GEOM_MODIFIER_3D
%token SAT_OPTION
%token FTZ_OPTION
%token ATOMIC_AND
%token ATOMIC_OR
%token ATOMIC_XOR
%token ATOMIC_CAS
%token ATOMIC_EXCH
%token ATOMIC_ADD
%token ATOMIC_INC
%token ATOMIC_DEC
%token ATOMIC_MIN
%token ATOMIC_MAX
%token  LEFT_ANGLE_BRACKET
%token  RIGHT_ANGLE_BRACKET
%token  LEFT_PAREN
%token  RIGHT_PAREN
%token  APPROX_OPTION
%token  FULL_OPTION
%token  ANY_OPTION
%token  ALL_OPTION
%token  GLOBAL_OPTION
%token  CTA_OPTION
%type <int_value> function_decl_header
%type <ptr_value> function_decl

%{
  	#include "ptx_ir.h"
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented();
	extern int g_func_decl;
	int ptx_lex(void);
	int ptx_error(const char *);
%}

%%

input:	/* empty */
	| input directive_statement
	| input function_defn
	| input function_decl
	;

function_defn: function_decl { set_symtab($1); } LEFT_BRACE statement_list RIGHT_BRACE { end_function(); }
	| function_decl { set_symtab($1); } block_spec LEFT_BRACE statement_list RIGHT_BRACE { end_function(); }
	;

block_spec: MAXNTID_DIRECTIVE INT_OPERAND COMMA INT_OPERAND COMMA INT_OPERAND
	;

function_decl: function_decl_header LEFT_PAREN { start_function($1); } param_entry RIGHT_PAREN function_ident_param { $$ = reset_symtab(); }
	| function_decl_header { start_function($1); } function_ident_param { $$ = reset_symtab(); }
	| function_decl_header { start_function($1); add_function_name(""); g_func_decl=0; $$ = reset_symtab(); }
	;

function_ident_param: IDENTIFIER { add_function_name($1); } LEFT_PAREN param_list RIGHT_PAREN { g_func_decl=0; } 
	| IDENTIFIER { add_function_name($1); g_func_decl=0; } 
	;

function_decl_header: ENTRY_DIRECTIVE { $$ = 1; g_func_decl=1; }
	| FUNC_DIRECTIVE { $$ = 0; g_func_decl=1; }
	;

param_list: param_entry { add_directive(); }
	| param_list COMMA param_entry { add_directive(); }

param_entry: PARAM_DIRECTIVE { add_space_spec(PARAM_DIRECTIVE); } variable_spec identifier_spec { add_function_arg(); }
	| REG_DIRECTIVE { add_space_spec(REG_DIRECTIVE); } variable_spec identifier_spec { add_function_arg(); }

statement_list: directive_statement { add_directive(); }
	| instruction_statement { add_instruction(); }
	| statement_list directive_statement { add_directive(); }
	| statement_list instruction_statement { add_instruction(); } 
	;

directive_statement: variable_declaration SEMI_COLON
	| VERSION_DIRECTIVE DOUBLE_OPERAND 
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER 
	| TARGET_DIRECTIVE IDENTIFIER 
	| FILE_DIRECTIVE INT_OPERAND STRING { add_file($2,$3); } 
	| LOC_DIRECTIVE INT_OPERAND INT_OPERAND INT_OPERAND 
	;

variable_declaration: variable_spec identifier_list { add_variables(); }
	| variable_spec identifier_spec EQUALS initializer_list { add_variables(); }
	| variable_spec identifier_spec EQUALS literal_operand { add_variables(); }
	;

variable_spec: var_spec_list { set_variable_type(); }

identifier_list: identifier_spec
	| identifier_list COMMA identifier_spec;

identifier_spec: IDENTIFIER { add_identifier($1,0,NON_ARRAY_IDENTIFIER); }
	| IDENTIFIER LEFT_ANGLE_BRACKET INT_OPERAND RIGHT_ANGLE_BRACKET {
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen($1);
		for( i=0; i < $3; i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = malloc(l);
			snprintf(id,l,"%s%u",$1,i);
			add_identifier(id,0,NON_ARRAY_IDENTIFIER); 
		}
		free($1);
	}
	| IDENTIFIER LEFT_SQUARE_BRACKET RIGHT_SQUARE_BRACKET { add_identifier($1,0,ARRAY_IDENTIFIER_NO_DIM); }
	| IDENTIFIER LEFT_SQUARE_BRACKET INT_OPERAND RIGHT_SQUARE_BRACKET { add_identifier($1,$3,ARRAY_IDENTIFIER); }
	;

var_spec_list: var_spec 
	 | var_spec_list var_spec;

var_spec: space_spec 
	| type_spec
	| align_spec
	| EXTERN_DIRECTIVE { add_extern_spec(); }
	;

align_spec: ALIGN_DIRECTIVE INT_OPERAND { add_alignment_spec($2); }

space_spec: REG_DIRECTIVE {  add_space_spec(REG_DIRECTIVE); }
	| SREG_DIRECTIVE  {  add_space_spec(SREG_DIRECTIVE); }
	| addressable_spec
	;

addressable_spec: CONST_DIRECTIVE {  add_space_spec(CONST_DIRECTIVE); }
	| GLOBAL_DIRECTIVE 	  {  add_space_spec(GLOBAL_DIRECTIVE); }
	| LOCAL_DIRECTIVE 	  {  add_space_spec(LOCAL_DIRECTIVE); }
	| PARAM_DIRECTIVE 	  {  add_space_spec(PARAM_DIRECTIVE); }
	| SHARED_DIRECTIVE 	  {  add_space_spec(SHARED_DIRECTIVE); }
	| SURF_DIRECTIVE 	  {  add_space_spec(SURF_DIRECTIVE); }
	| TEX_DIRECTIVE 	  {  add_space_spec(TEX_DIRECTIVE); }
	;

type_spec: scalar_type 
	|  vector_spec scalar_type 
	;

vector_spec:  V2_TYPE {  add_option(V2_TYPE); }
	| V3_TYPE     {  add_option(V3_TYPE); }
	| V4_TYPE     {  add_option(V4_TYPE); }
	;

scalar_type: S8_TYPE { add_scalar_type_spec( S8_TYPE );  }
	| S16_TYPE   { add_scalar_type_spec( S16_TYPE ); }
	| S32_TYPE   { add_scalar_type_spec( S32_TYPE ); }
	| S64_TYPE   { add_scalar_type_spec( S64_TYPE ); }
	| U8_TYPE    { add_scalar_type_spec( U8_TYPE );  }
	| U16_TYPE   { add_scalar_type_spec( U16_TYPE ); }
	| U32_TYPE   { add_scalar_type_spec( U32_TYPE ); }
	| U64_TYPE   { add_scalar_type_spec( U64_TYPE ); }
	| F16_TYPE   { add_scalar_type_spec( F16_TYPE ); }
	| F32_TYPE   { add_scalar_type_spec( F32_TYPE ); }
	| F64_TYPE   { add_scalar_type_spec( F64_TYPE ); }
	| B8_TYPE    { add_scalar_type_spec( B8_TYPE );  }
	| B16_TYPE   { add_scalar_type_spec( B16_TYPE ); }
	| B32_TYPE   { add_scalar_type_spec( B32_TYPE ); }
	| B64_TYPE   { add_scalar_type_spec( B64_TYPE ); }
	| PRED_TYPE  { add_scalar_type_spec( PRED_TYPE ); }
	;

initializer_list: LEFT_BRACE literal_list RIGHT_BRACE { add_array_initializer(); } 
	| LEFT_BRACE initializer_list RIGHT_BRACE { syntax_not_implemented(); }

literal_list: literal_operand
	| literal_operand COMMA literal_list;

instruction_statement:  instruction SEMI_COLON
	| IDENTIFIER COLON { add_label($1); }    
	| pred_spec instruction SEMI_COLON;

instruction: opcode_spec LEFT_PAREN operand RIGHT_PAREN { set_return(); } COMMA operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand_list 
	| opcode_spec
	;

opcode_spec: OPCODE { add_opcode($1); } option_list
	| OPCODE { add_opcode($1); }

pred_spec: PRED IDENTIFIER  { add_pred($2,0); }
	| PRED EXCLAMATION IDENTIFIER { add_pred($3,1); } 
	;

option_list: option
	| option option_list ;

option: type_spec
	| compare_spec
	| addressable_spec
	| rounding_mode
	| UNI_OPTION { add_option(UNI_OPTION); }
	| WIDE_OPTION { add_option(WIDE_OPTION); }
	| ANY_OPTION { add_option(ANY_OPTION); }
	| ALL_OPTION { add_option(ALL_OPTION); }
	| GLOBAL_OPTION { add_option(GLOBAL_OPTION); }
	| CTA_OPTION { add_option(CTA_OPTION); }
	| GEOM_MODIFIER_1D { add_option(GEOM_MODIFIER_1D); }
	| GEOM_MODIFIER_2D { add_option(GEOM_MODIFIER_2D); }
	| GEOM_MODIFIER_3D { add_option(GEOM_MODIFIER_3D); }
	| SAT_OPTION { add_option(SAT_OPTION); }
 	| FTZ_OPTION { add_option(FTZ_OPTION); } 
	| APPROX_OPTION { add_option(APPROX_OPTION); }
	| FULL_OPTION { add_option(FULL_OPTION); }
	| atomic_operation_spec ;

atomic_operation_spec: ATOMIC_AND { add_option(ATOMIC_AND); } 
	| ATOMIC_OR { add_option(ATOMIC_OR); } 
	| ATOMIC_XOR { add_option(ATOMIC_XOR); } 
	| ATOMIC_CAS { add_option(ATOMIC_CAS); } 
	| ATOMIC_EXCH { add_option(ATOMIC_EXCH); } 
	| ATOMIC_ADD { add_option(ATOMIC_ADD); } 
	| ATOMIC_INC { add_option(ATOMIC_INC); } 
	| ATOMIC_DEC { add_option(ATOMIC_DEC); } 
	| ATOMIC_MIN { add_option(ATOMIC_MIN); } 
	| ATOMIC_MAX { add_option(ATOMIC_MAX); } 
	;

rounding_mode: floating_point_rounding_mode
	| integer_rounding_mode;

floating_point_rounding_mode: RN_OPTION { add_option(RN_OPTION); } 
 	| RZ_OPTION { add_option(RZ_OPTION); } 
 	| RM_OPTION { add_option(RM_OPTION); } 
 	| RP_OPTION { add_option(RP_OPTION); } 
	;

integer_rounding_mode: RNI_OPTION { add_option(RNI_OPTION); } 
	| RZI_OPTION { add_option(RZI_OPTION); } 
 	| RMI_OPTION { add_option(RMI_OPTION); } 
 	| RPI_OPTION { add_option(RPI_OPTION); } 
	;

compare_spec:EQ_OPTION { add_option(EQ_OPTION); } 
	| NE_OPTION { add_option(NE_OPTION); } 
	| LT_OPTION { add_option(LT_OPTION); } 
	| LE_OPTION { add_option(LE_OPTION); } 
	| GT_OPTION { add_option(GT_OPTION); } 
	| GE_OPTION { add_option(GE_OPTION); } 
	| LO_OPTION { add_option(LO_OPTION); } 
	| LS_OPTION { add_option(LS_OPTION); } 
	| HI_OPTION { add_option(HI_OPTION); } 
	| HS_OPTION  { add_option(HS_OPTION); } 
	| EQU_OPTION { add_option(EQU_OPTION); } 
	| NEU_OPTION { add_option(NEU_OPTION); } 
	| LTU_OPTION { add_option(LTU_OPTION); } 
	| LEU_OPTION { add_option(LEU_OPTION); } 
	| GTU_OPTION { add_option(GTU_OPTION); } 
	| GEU_OPTION { add_option(GEU_OPTION); } 
	| NUM_OPTION { add_option(NUM_OPTION); } 
	| NAN_OPTION { add_option(NAN_OPTION); } 
	;

operand_list: operand
	| operand COMMA operand_list;

operand: IDENTIFIER  { add_scalar_operand( $1 ); }
	| EXCLAMATION IDENTIFIER { add_neg_pred_operand( $2 ); }
	| memory_operand
	| literal_operand
	| builtin_operand
	| vector_operand
	| tex_operand
	| IDENTIFIER PLUS INT_OPERAND { add_address_operand($1,$3); }
	;

vector_operand: LEFT_BRACE IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { add_2vector_operand($2,$4); }
	| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { add_3vector_operand($2,$4,$6); }
	| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { add_4vector_operand($2,$4,$6,$8); }
	;

tex_operand: LEFT_SQUARE_BRACKET IDENTIFIER COMMA 
		LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE 
	     RIGHT_SQUARE_BRACKET { 
		add_scalar_operand($2);
		add_4vector_operand($5,$7,$9,$11); 
	}

builtin_operand: SPECIAL_REGISTER DIMENSION_MODIFIER { add_builtin_operand($1,$2); }
        | SPECIAL_REGISTER { add_builtin_operand($1,-1); }
	;

memory_operand : LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { add_memory_operand(); }

literal_operand : INT_OPERAND { add_literal_int($1); }
	| FLOAT_OPERAND { add_literal_float($1); }
	| DOUBLE_OPERAND { add_literal_double($1); }
	;

address_expression: IDENTIFIER { add_address_operand($1,0); }
	| IDENTIFIER PLUS INT_OPERAND { add_address_operand($1,$3); }
	;

%%

extern int ptx_lineno;
extern const char *g_filename;

void syntax_not_implemented()
{
	printf("Parse error (%s:%u): this syntax is not (yet) implemented:\n",g_filename,ptx_lineno);
	ptx_error(NULL);
	abort();
}
