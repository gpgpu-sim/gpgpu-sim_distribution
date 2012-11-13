/*
Copyright (c) 2009-2011, Tor M. Aamodt
The University of British Columbia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
Neither the name of The University of British Columbia nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
%token  BRANCHTARGETS_DIRECTIVE
%token  BYTE_DIRECTIVE
%token  CALLPROTOTYPE_DIRECTIVE
%token  CALLTARGETS_DIRECTIVE
%token  <int_value> CONST_DIRECTIVE
%token  CONSTPTR_DIRECTIVE
%token  PTR_DIRECTIVE
%token  ENTRY_DIRECTIVE
%token  EXTERN_DIRECTIVE
%token  FILE_DIRECTIVE
%token  FUNC_DIRECTIVE
%token  GLOBAL_DIRECTIVE
%token  LOCAL_DIRECTIVE
%token  LOC_DIRECTIVE
%token  MAXNCTAPERSM_DIRECTIVE
%token  MAXNNREG_DIRECTIVE
%token  MAXNTID_DIRECTIVE
%token  MINNCTAPERSM_DIRECTIVE
%token  PARAM_DIRECTIVE
%token  PRAGMA_DIRECTIVE
%token  REG_DIRECTIVE
%token  REQNTID_DIRECTIVE
%token  SECTION_DIRECTIVE
%token  SHARED_DIRECTIVE
%token  SREG_DIRECTIVE
%token  STRUCT_DIRECTIVE
%token  SURF_DIRECTIVE
%token  TARGET_DIRECTIVE
%token  TEX_DIRECTIVE
%token  UNION_DIRECTIVE
%token  VERSION_DIRECTIVE
%token  ADDRESS_SIZE_DIRECTIVE
%token  VISIBLE_DIRECTIVE
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
%token  FF64_TYPE
%token  B8_TYPE
%token  B16_TYPE
%token  B32_TYPE
%token  B64_TYPE
%token  BB64_TYPE
%token  BB128_TYPE
%token  PRED_TYPE
%token  TEXREF_TYPE
%token  SAMPLERREF_TYPE
%token  SURFREF_TYPE
%token  V2_TYPE
%token  V3_TYPE
%token  V4_TYPE
%token  COMMA
%token  PRED
%token  HALF_OPTION
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
%token  CF_OPTION
%token  SF_OPTION
%token  NSF_OPTION
%token  LEFT_SQUARE_BRACKET
%token  RIGHT_SQUARE_BRACKET
%token  WIDE_OPTION
%token  <int_value> SPECIAL_REGISTER
%token  MINUS
%token  PLUS
%token  COLON
%token  SEMI_COLON
%token  EXCLAMATION
%token  PIPE
%token	RIGHT_BRACE
%token	LEFT_BRACE
%token	EQUALS
%token  PERIOD
%token  BACKSLASH
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
%token NEG_OPTION
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
%token  BALLOT_OPTION
%token  GLOBAL_OPTION
%token  CTA_OPTION
%token  SYS_OPTION
%token  EXIT_OPTION
%token  ABS_OPTION
%token  TO_OPTION
%token  CA_OPTION;
%token  CG_OPTION;
%token  CS_OPTION;
%token  LU_OPTION;
%token  CV_OPTION;
%token  WB_OPTION;
%token  WT_OPTION;

%type <int_value> function_decl_header
%type <ptr_value> function_decl

%{
  	#include "ptx_parser.h"
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

function_defn: function_decl { set_symtab($1); func_header(".skip"); } statement_block { end_function(); }
	| function_decl { set_symtab($1); } block_spec_list { func_header(".skip"); } statement_block { end_function(); }
	;

block_spec: MAXNTID_DIRECTIVE INT_OPERAND COMMA INT_OPERAND COMMA INT_OPERAND {func_header_info_int(".maxntid", $2);
										func_header_info_int(",", $4);
										func_header_info_int(",", $6); }
	| MINNCTAPERSM_DIRECTIVE INT_OPERAND { func_header_info_int(".minnctapersm", $2); printf("GPGPU-Sim: Warning: .minnctapersm ignored. \n"); }
	| MAXNCTAPERSM_DIRECTIVE INT_OPERAND { func_header_info_int(".maxnctapersm", $2); printf("GPGPU-Sim: Warning: .maxnctapersm ignored. \n"); }
	;

block_spec_list: block_spec
	| block_spec_list block_spec
	;

function_decl: function_decl_header LEFT_PAREN { start_function($1); func_header_info("(");} param_entry RIGHT_PAREN {func_header_info(")");} function_ident_param { $$ = reset_symtab(); }
	| function_decl_header { start_function($1); } function_ident_param { $$ = reset_symtab(); }
	| function_decl_header { start_function($1); add_function_name(""); g_func_decl=0; $$ = reset_symtab(); }
	;

function_ident_param: IDENTIFIER { add_function_name($1); } LEFT_PAREN {func_header_info("(");} param_list RIGHT_PAREN { g_func_decl=0; func_header_info(")"); } 
	| IDENTIFIER { add_function_name($1); g_func_decl=0; } 
	;

function_decl_header: ENTRY_DIRECTIVE { $$ = 1; g_func_decl=1; func_header(".entry"); }
	| FUNC_DIRECTIVE { $$ = 0; g_func_decl=1; func_header(".func"); }
	| VISIBLE_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; g_func_decl=1; func_header(".func"); }
	| EXTERN_DIRECTIVE FUNC_DIRECTIVE { $$ = 2; g_func_decl=1; func_header(".func"); }
	;

param_list: /*empty*/
	| param_entry { add_directive(); }
	| param_list COMMA {func_header_info(",");} param_entry { add_directive(); }

param_entry: PARAM_DIRECTIVE { add_space_spec(param_space_unclassified,0); } variable_spec ptr_spec identifier_spec { add_function_arg(); }
	| REG_DIRECTIVE { add_space_spec(reg_space,0); } variable_spec identifier_spec { add_function_arg(); }

ptr_spec: /*empty*/
        | PTR_DIRECTIVE ptr_space_spec ptr_align_spec
        | PTR_DIRECTIVE ptr_align_spec

ptr_space_spec: GLOBAL_DIRECTIVE { add_ptr_spec(global_space); }
              | LOCAL_DIRECTIVE  { add_ptr_spec(local_space); }
              | SHARED_DIRECTIVE { add_ptr_spec(shared_space); }

ptr_align_spec: ALIGN_DIRECTIVE INT_OPERAND

statement_block: LEFT_BRACE statement_list RIGHT_BRACE 

statement_list: directive_statement { add_directive(); }
	| instruction_statement { add_instruction(); }
	| statement_list directive_statement { add_directive(); }
	| statement_list instruction_statement { add_instruction(); }
	| statement_list statement_block
	| statement_block
	;

directive_statement: variable_declaration SEMI_COLON
	| VERSION_DIRECTIVE DOUBLE_OPERAND { add_version_info($2, 0); }
	| VERSION_DIRECTIVE DOUBLE_OPERAND PLUS { add_version_info($2,1); }
	| ADDRESS_SIZE_DIRECTIVE INT_OPERAND {/*Do nothing*/}
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER { target_header2($2,$4); }
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER { target_header3($2,$4,$6); }
	| TARGET_DIRECTIVE IDENTIFIER { target_header($2); }
	| FILE_DIRECTIVE INT_OPERAND STRING { add_file($2,$3); } 
	| LOC_DIRECTIVE INT_OPERAND INT_OPERAND INT_OPERAND 
	| PRAGMA_DIRECTIVE STRING SEMI_COLON { add_pragma($2); }
	| function_decl SEMI_COLON {/*Do nothing*/}
	;

variable_declaration: variable_spec identifier_list { add_variables(); }
	| variable_spec identifier_spec EQUALS initializer_list { add_variables(); }
	| variable_spec identifier_spec EQUALS literal_operand { add_variables(); }
	| CONSTPTR_DIRECTIVE IDENTIFIER COMMA IDENTIFIER COMMA INT_OPERAND { add_constptr($2, $4, $6); }
	;

variable_spec: var_spec_list { set_variable_type(); }

identifier_list: identifier_spec
	| identifier_list COMMA identifier_spec;

identifier_spec: IDENTIFIER { add_identifier($1,0,NON_ARRAY_IDENTIFIER); func_header_info($1);}
	| IDENTIFIER LEFT_ANGLE_BRACKET INT_OPERAND RIGHT_ANGLE_BRACKET { func_header_info($1); func_header_info_int("<", $3); func_header_info(">");
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen($1);
		for( i=0; i < $3; i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = (char*) malloc(l);
			snprintf(id,l,"%s%u",$1,i);
			add_identifier(id,0,NON_ARRAY_IDENTIFIER); 
		}
		free($1);
	}
	| IDENTIFIER LEFT_SQUARE_BRACKET RIGHT_SQUARE_BRACKET { add_identifier($1,0,ARRAY_IDENTIFIER_NO_DIM); func_header_info($1); func_header_info("["); func_header_info("]");}
	| IDENTIFIER LEFT_SQUARE_BRACKET INT_OPERAND RIGHT_SQUARE_BRACKET { add_identifier($1,$3,ARRAY_IDENTIFIER); func_header_info($1); func_header_info_int("[",$3); func_header_info("]");}
	;

var_spec_list: var_spec 
	 | var_spec_list var_spec;

var_spec: space_spec 
	| type_spec
	| align_spec
	| EXTERN_DIRECTIVE { add_extern_spec(); }
	;

align_spec: ALIGN_DIRECTIVE INT_OPERAND { add_alignment_spec($2); }

space_spec: REG_DIRECTIVE {  add_space_spec(reg_space,0); }
	| SREG_DIRECTIVE  {  add_space_spec(reg_space,0); }
	| addressable_spec
	;

addressable_spec: CONST_DIRECTIVE {  add_space_spec(const_space,$1); }
	| GLOBAL_DIRECTIVE 	  {  add_space_spec(global_space,0); }
	| LOCAL_DIRECTIVE 	  {  add_space_spec(local_space,0); }
	| PARAM_DIRECTIVE 	  {  add_space_spec(param_space_unclassified,0); }
	| SHARED_DIRECTIVE 	  {  add_space_spec(shared_space,0); }
	| SURF_DIRECTIVE 	  {  add_space_spec(surf_space,0); }
	| TEX_DIRECTIVE 	  {  add_space_spec(tex_space,0); }
	;

type_spec: scalar_type 
	|  vector_spec scalar_type 
	;

vector_spec:  V2_TYPE {  add_option(V2_TYPE); func_header_info(".v2");}
	| V3_TYPE     {  add_option(V3_TYPE); func_header_info(".v3");}
	| V4_TYPE     {  add_option(V4_TYPE); func_header_info(".v4");}
	;

scalar_type: S8_TYPE { add_scalar_type_spec( S8_TYPE ); }
	| S16_TYPE   { add_scalar_type_spec( S16_TYPE ); }
	| S32_TYPE   { add_scalar_type_spec( S32_TYPE ); }
	| S64_TYPE   { add_scalar_type_spec( S64_TYPE ); }
	| U8_TYPE    { add_scalar_type_spec( U8_TYPE ); }
	| U16_TYPE   { add_scalar_type_spec( U16_TYPE ); }
	| U32_TYPE   { add_scalar_type_spec( U32_TYPE ); }
	| U64_TYPE   { add_scalar_type_spec( U64_TYPE ); }
	| F16_TYPE   { add_scalar_type_spec( F16_TYPE ); }
	| F32_TYPE   { add_scalar_type_spec( F32_TYPE ); }
	| F64_TYPE   { add_scalar_type_spec( F64_TYPE ); }
	| FF64_TYPE   { add_scalar_type_spec( FF64_TYPE ); }
	| B8_TYPE    { add_scalar_type_spec( B8_TYPE );  }
	| B16_TYPE   { add_scalar_type_spec( B16_TYPE ); }
	| B32_TYPE   { add_scalar_type_spec( B32_TYPE ); }
	| B64_TYPE   { add_scalar_type_spec( B64_TYPE ); }
	| BB64_TYPE   { add_scalar_type_spec( BB64_TYPE ); }
	| BB128_TYPE   { add_scalar_type_spec( BB128_TYPE ); }
	| PRED_TYPE  { add_scalar_type_spec( PRED_TYPE ); }
	| TEXREF_TYPE  { add_scalar_type_spec( TEXREF_TYPE ); }
	| SAMPLERREF_TYPE  { add_scalar_type_spec( SAMPLERREF_TYPE ); }
	| SURFREF_TYPE  { add_scalar_type_spec( SURFREF_TYPE ); }
	;

initializer_list: LEFT_BRACE literal_list RIGHT_BRACE { add_array_initializer(); } 
	| LEFT_BRACE initializer_list RIGHT_BRACE { syntax_not_implemented(); }

literal_list: literal_operand
	| literal_list COMMA literal_operand;

instruction_statement:  instruction SEMI_COLON
	| IDENTIFIER COLON { add_label($1); }    
	| pred_spec instruction SEMI_COLON;

instruction: opcode_spec LEFT_PAREN operand RIGHT_PAREN { set_return(); } COMMA operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand COMMA LEFT_PAREN RIGHT_PAREN
	| opcode_spec operand_list 
	| opcode_spec
	;

opcode_spec: OPCODE { add_opcode($1); } option_list
	| OPCODE { add_opcode($1); }

pred_spec: PRED IDENTIFIER  { add_pred($2,0, -1); }
	| PRED EXCLAMATION IDENTIFIER { add_pred($3,1, -1); } 
	| PRED IDENTIFIER LT_OPTION  { add_pred($2,0,1); }
	| PRED IDENTIFIER EQ_OPTION  { add_pred($2,0,2); }
	| PRED IDENTIFIER LE_OPTION  { add_pred($2,0,3); }
	| PRED IDENTIFIER NE_OPTION  { add_pred($2,0,5); }
	| PRED IDENTIFIER GE_OPTION  { add_pred($2,0,6); }
	| PRED IDENTIFIER EQU_OPTION  { add_pred($2,0,10); }
	| PRED IDENTIFIER GTU_OPTION  { add_pred($2,0,12); }
	| PRED IDENTIFIER NEU_OPTION  { add_pred($2,0,13); }
	| PRED IDENTIFIER CF_OPTION  { add_pred($2,0,17); }
	| PRED IDENTIFIER SF_OPTION  { add_pred($2,0,19); }
	| PRED IDENTIFIER NSF_OPTION  { add_pred($2,0,28); }
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
	| BALLOT_OPTION { add_option(BALLOT_OPTION); }
	| GLOBAL_OPTION { add_option(GLOBAL_OPTION); }
	| CTA_OPTION { add_option(CTA_OPTION); }
	| SYS_OPTION { add_option(SYS_OPTION); }
	| GEOM_MODIFIER_1D { add_option(GEOM_MODIFIER_1D); }
	| GEOM_MODIFIER_2D { add_option(GEOM_MODIFIER_2D); }
	| GEOM_MODIFIER_3D { add_option(GEOM_MODIFIER_3D); }
	| SAT_OPTION { add_option(SAT_OPTION); }
 	| FTZ_OPTION { add_option(FTZ_OPTION); } 
 	| NEG_OPTION { add_option(NEG_OPTION); } 
	| APPROX_OPTION { add_option(APPROX_OPTION); }
	| FULL_OPTION { add_option(FULL_OPTION); }
	| EXIT_OPTION { add_option(EXIT_OPTION); }
	| ABS_OPTION { add_option(ABS_OPTION); }
	| atomic_operation_spec ;
	| TO_OPTION { add_option(TO_OPTION); }
	| HALF_OPTION { add_option(HALF_OPTION); }
	| CA_OPTION { add_option(CA_OPTION); }
	| CG_OPTION { add_option(CG_OPTION); }
	| CS_OPTION { add_option(CS_OPTION); }
	| LU_OPTION { add_option(LU_OPTION); }
	| CV_OPTION { add_option(CV_OPTION); }
	| WB_OPTION { add_option(WB_OPTION); }
	| WT_OPTION { add_option(WT_OPTION); }
	;

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
	| MINUS IDENTIFIER  { add_scalar_operand( $2 ); change_operand_neg(); }
	| memory_operand
	| literal_operand
	| builtin_operand
	| vector_operand
	| MINUS vector_operand { change_operand_neg(); }
	| tex_operand
	| IDENTIFIER PLUS INT_OPERAND { add_address_operand($1,$3); }
	| IDENTIFIER LO_OPTION { add_scalar_operand( $1 ); change_operand_lohi(1);}
	| MINUS IDENTIFIER LO_OPTION { add_scalar_operand( $2 ); change_operand_lohi(1); change_operand_neg();}
	| IDENTIFIER HI_OPTION { add_scalar_operand( $1 ); change_operand_lohi(2);}
	| MINUS IDENTIFIER HI_OPTION { add_scalar_operand( $2 ); change_operand_lohi(2); change_operand_neg();}
	| IDENTIFIER PIPE IDENTIFIER { add_2vector_operand($1,$3); change_double_operand_type(-1);}
	| IDENTIFIER PIPE IDENTIFIER LO_OPTION { add_2vector_operand($1,$3); change_double_operand_type(-1); change_operand_lohi(1);}
	| IDENTIFIER PIPE IDENTIFIER HI_OPTION { add_2vector_operand($1,$3); change_double_operand_type(-1); change_operand_lohi(2);}
	| IDENTIFIER BACKSLASH IDENTIFIER { add_2vector_operand($1,$3); change_double_operand_type(-3);}
	| IDENTIFIER BACKSLASH IDENTIFIER LO_OPTION { add_2vector_operand($1,$3); change_double_operand_type(-3); change_operand_lohi(1);}
	| IDENTIFIER BACKSLASH IDENTIFIER HI_OPTION { add_2vector_operand($1,$3); change_double_operand_type(-3); change_operand_lohi(2);}
	;

vector_operand: LEFT_BRACE IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { add_2vector_operand($2,$4); }
		| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { add_3vector_operand($2,$4,$6); }
		| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { add_4vector_operand($2,$4,$6,$8); }
		| LEFT_BRACE IDENTIFIER RIGHT_BRACE { add_1vector_operand($2); }
	;

tex_operand: LEFT_SQUARE_BRACKET IDENTIFIER COMMA { add_scalar_operand($2); }
		vector_operand 
	     RIGHT_SQUARE_BRACKET
	;

builtin_operand: SPECIAL_REGISTER DIMENSION_MODIFIER { add_builtin_operand($1,$2); }
        | SPECIAL_REGISTER { add_builtin_operand($1,-1); }
	;

memory_operand : LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { add_memory_operand(); }
	| IDENTIFIER LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { add_memory_operand(); change_memory_addr_space($1); }
	| IDENTIFIER LEFT_SQUARE_BRACKET literal_operand RIGHT_SQUARE_BRACKET { change_memory_addr_space($1); }
	| IDENTIFIER LEFT_SQUARE_BRACKET twin_operand RIGHT_SQUARE_BRACKET { change_memory_addr_space($1); add_memory_operand();}
        | MINUS memory_operand { change_operand_neg(); }
	;

twin_operand : IDENTIFIER PLUS IDENTIFIER { add_double_operand($1,$3); change_double_operand_type(1); }
	| IDENTIFIER PLUS IDENTIFIER LO_OPTION { add_double_operand($1,$3); change_double_operand_type(1); change_operand_lohi(1); }
	| IDENTIFIER PLUS IDENTIFIER HI_OPTION { add_double_operand($1,$3); change_double_operand_type(1); change_operand_lohi(2); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER  { add_double_operand($1,$4); change_double_operand_type(2); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER LO_OPTION { add_double_operand($1,$4); change_double_operand_type(2); change_operand_lohi(1); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER HI_OPTION { add_double_operand($1,$4); change_double_operand_type(2); change_operand_lohi(2); }
	| IDENTIFIER PLUS EQUALS INT_OPERAND  { add_address_operand($1,$4); change_double_operand_type(3); }
	;

literal_operand : INT_OPERAND { add_literal_int($1); }
	| FLOAT_OPERAND { add_literal_float($1); }
	| DOUBLE_OPERAND { add_literal_double($1); }
	;

address_expression: IDENTIFIER { add_address_operand($1,0); }
	| IDENTIFIER LO_OPTION { add_address_operand($1,0); change_operand_lohi(1);}
	| IDENTIFIER HI_OPTION { add_address_operand($1,0); change_operand_lohi(2); }
	| IDENTIFIER PLUS INT_OPERAND { add_address_operand($1,$3); }
	| INT_OPERAND { add_address_operand2($1); }
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
