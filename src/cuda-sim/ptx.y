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

%{
typedef void * yyscan_t;
class ptx_recognizer;
#include "../../libcuda/gpgpu_context.h"
%}

%define api.pure full
%parse-param {yyscan_t scanner}
%parse-param {ptx_recognizer* recognizer}
%lex-param {yyscan_t scanner}
%lex-param {ptx_recognizer* recognizer}

%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token <string_value> STRING
%token <int_value>  OPCODE
%token <int_value>  WMMA_DIRECTIVE
%token <int_value>  LAYOUT 
%token <int_value>  CONFIGURATION 
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
%token	SSTARR_DIRECTIVE
%token  STRUCT_DIRECTIVE
%token  SURF_DIRECTIVE
%token  TARGET_DIRECTIVE
%token  TEX_DIRECTIVE
%token  UNION_DIRECTIVE
%token  VERSION_DIRECTIVE
%token  ADDRESS_SIZE_DIRECTIVE
%token  VISIBLE_DIRECTIVE
%token  WEAK_DIRECTIVE
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
%token  EXTP_OPTION
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
%token SYNC_OPTION
%token RED_OPTION
%token ARRIVE_OPTION
%token ATOMIC_POPC
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
%token	NC_OPTION;
%token	UP_OPTION;
%token	DOWN_OPTION;
%token	BFLY_OPTION;
%token	IDX_OPTION;
%token	PRMT_F4E_MODE;
%token	PRMT_B4E_MODE;
%token	PRMT_RC8_MODE;
%token	PRMT_RC16_MODE;
%token	PRMT_ECL_MODE;
%token	PRMT_ECR_MODE;

%type <int_value> function_decl_header
%type <ptr_value> function_decl

%{
  	#include "ptx_parser.h"
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented(yyscan_t yyscanner, ptx_recognizer* recognizer);
	int ptx_lex(YYSTYPE * yylval_param, yyscan_t yyscanner, ptx_recognizer* recognizer);
	int ptx_error( yyscan_t yyscanner, ptx_recognizer* recognizer, const char *s );
%}

%%

input:	/* empty */
	| input directive_statement
	| input function_defn
	| input function_decl
	;

function_defn: function_decl { recognizer->set_symtab($1); recognizer->func_header(".skip"); } statement_block { recognizer->end_function(); }
	| function_decl { recognizer->set_symtab($1); } block_spec_list { recognizer->func_header(".skip"); } statement_block { recognizer->end_function(); }
	;

block_spec: MAXNTID_DIRECTIVE INT_OPERAND COMMA INT_OPERAND COMMA INT_OPERAND {recognizer->func_header_info_int(".maxntid", $2);
										recognizer->func_header_info_int(",", $4);
										recognizer->func_header_info_int(",", $6);
                                                                                recognizer->maxnt_id($2, $4, $6);}
	| MINNCTAPERSM_DIRECTIVE INT_OPERAND { recognizer->func_header_info_int(".minnctapersm", $2); printf("GPGPU-Sim: Warning: .minnctapersm ignored. \n"); }
	| MAXNCTAPERSM_DIRECTIVE INT_OPERAND { recognizer->func_header_info_int(".maxnctapersm", $2); printf("GPGPU-Sim: Warning: .maxnctapersm ignored. \n"); }
	;

block_spec_list: block_spec
	| block_spec_list block_spec
	;

function_decl: function_decl_header LEFT_PAREN { recognizer->start_function($1); recognizer->func_header_info("(");} param_entry RIGHT_PAREN {recognizer->func_header_info(")");} function_ident_param { $$ = recognizer->reset_symtab(); }
	| function_decl_header { recognizer->start_function($1); } function_ident_param { $$ = recognizer->reset_symtab(); }
	| function_decl_header { recognizer->start_function($1); recognizer->add_function_name(""); recognizer->g_func_decl=0; $$ = recognizer->reset_symtab(); }
	;

function_ident_param: IDENTIFIER { recognizer->add_function_name($1); } LEFT_PAREN {recognizer->func_header_info("(");} param_list RIGHT_PAREN { recognizer->g_func_decl=0; recognizer->func_header_info(")"); }
	| IDENTIFIER { recognizer->add_function_name($1); recognizer->g_func_decl=0; }
	;

function_decl_header: ENTRY_DIRECTIVE { $$ = 1; recognizer->g_func_decl=1; recognizer->func_header(".entry"); }
	| VISIBLE_DIRECTIVE ENTRY_DIRECTIVE { $$ = 1; recognizer->g_func_decl=1; recognizer->func_header(".entry"); }
	| WEAK_DIRECTIVE ENTRY_DIRECTIVE { $$ = 1; recognizer->g_func_decl=1; recognizer->func_header(".entry"); }
	| FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	| VISIBLE_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	| WEAK_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	| EXTERN_DIRECTIVE FUNC_DIRECTIVE { $$ = 2; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	| WEAK_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	;

param_list: /*empty*/
	| param_entry { recognizer->add_directive(); }
	| param_list COMMA {recognizer->func_header_info(",");} param_entry { recognizer->add_directive(); }

param_entry: PARAM_DIRECTIVE { recognizer->add_space_spec(param_space_unclassified,0); } variable_spec ptr_spec identifier_spec { recognizer->add_function_arg(); }
	| REG_DIRECTIVE { recognizer->add_space_spec(reg_space,0); } variable_spec identifier_spec { recognizer->add_function_arg(); }

ptr_spec: /*empty*/
        | PTR_DIRECTIVE ptr_space_spec ptr_align_spec
        | PTR_DIRECTIVE ptr_align_spec

ptr_space_spec: GLOBAL_DIRECTIVE { recognizer->add_ptr_spec(global_space); }
              | LOCAL_DIRECTIVE  { recognizer->add_ptr_spec(local_space); }
              | SHARED_DIRECTIVE { recognizer->add_ptr_spec(shared_space); }
			  | CONST_DIRECTIVE { recognizer->add_ptr_spec(global_space); }

ptr_align_spec: ALIGN_DIRECTIVE INT_OPERAND

statement_block: LEFT_BRACE statement_list RIGHT_BRACE 

statement_list: directive_statement { recognizer->add_directive(); }
    | statement_list prototype_block {printf("Prototype statement detected. WARNING: this is not supported yet on GPGPU-SIM\n"); }
	| instruction_statement { recognizer->add_instruction(); }
	| statement_list directive_statement { recognizer->add_directive(); }
	| statement_list instruction_statement { recognizer->add_instruction(); }
	| statement_list {recognizer->start_inst_group();} statement_block {recognizer->end_inst_group();}
	| {recognizer->start_inst_group();} statement_block {recognizer->end_inst_group();}
	;

directive_statement: variable_declaration SEMI_COLON
	| VERSION_DIRECTIVE DOUBLE_OPERAND { recognizer->add_version_info($2, 0); }
	| VERSION_DIRECTIVE DOUBLE_OPERAND PLUS { recognizer->add_version_info($2,1); }
	| ADDRESS_SIZE_DIRECTIVE INT_OPERAND {/*Do nothing*/}
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER { recognizer->target_header2($2,$4); }
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER { recognizer->target_header3($2,$4,$6); }
	| TARGET_DIRECTIVE IDENTIFIER { recognizer->target_header($2); }
	| FILE_DIRECTIVE INT_OPERAND STRING { recognizer->add_file($2,$3); }
	| FILE_DIRECTIVE INT_OPERAND STRING COMMA INT_OPERAND COMMA INT_OPERAND { recognizer->add_file($2,$3); }
	| LOC_DIRECTIVE INT_OPERAND INT_OPERAND INT_OPERAND 
	| PRAGMA_DIRECTIVE STRING SEMI_COLON { recognizer->add_pragma($2); }
	| function_decl SEMI_COLON {/*Do nothing*/}
	;

variable_declaration: variable_spec identifier_list { recognizer->add_variables(); }
	| variable_spec identifier_spec EQUALS initializer_list { recognizer->add_variables(); }
	| variable_spec identifier_spec EQUALS literal_operand { recognizer->add_variables(); }
	| CONSTPTR_DIRECTIVE IDENTIFIER COMMA IDENTIFIER COMMA INT_OPERAND { recognizer->add_constptr($2, $4, $6); }
	;

variable_spec: var_spec_list { recognizer->set_variable_type(); }

identifier_list: identifier_spec
	| identifier_list COMMA identifier_spec;

identifier_spec: IDENTIFIER { recognizer->add_identifier($1,0,NON_ARRAY_IDENTIFIER); recognizer->func_header_info($1);}
	| IDENTIFIER LEFT_ANGLE_BRACKET INT_OPERAND RIGHT_ANGLE_BRACKET { recognizer->func_header_info($1); recognizer->func_header_info_int("<", $3); recognizer->func_header_info(">");
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen($1);
		for( i=0; i < $3; i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = (char*) malloc(l);
			snprintf(id,l,"%s%u",$1,i);
			recognizer->add_identifier(id,0,NON_ARRAY_IDENTIFIER);
		}
		free($1);
	}
	| IDENTIFIER LEFT_SQUARE_BRACKET RIGHT_SQUARE_BRACKET { recognizer->add_identifier($1,0,ARRAY_IDENTIFIER_NO_DIM); recognizer->func_header_info($1); recognizer->func_header_info("["); recognizer->func_header_info("]");}
	| IDENTIFIER LEFT_SQUARE_BRACKET INT_OPERAND RIGHT_SQUARE_BRACKET { recognizer->add_identifier($1,$3,ARRAY_IDENTIFIER); recognizer->func_header_info($1); recognizer->func_header_info_int("[",$3); recognizer->func_header_info("]");}
	;

var_spec_list: var_spec 
	 | var_spec_list var_spec;

var_spec: space_spec 
	| type_spec
	| align_spec
	| VISIBLE_DIRECTIVE
	| EXTERN_DIRECTIVE { recognizer->add_extern_spec(); }
    | WEAK_DIRECTIVE
	;

align_spec: ALIGN_DIRECTIVE INT_OPERAND { recognizer->add_alignment_spec($2); }

space_spec: REG_DIRECTIVE {  recognizer->add_space_spec(reg_space,0); }
	| SREG_DIRECTIVE  {  recognizer->add_space_spec(reg_space,0); }
	| addressable_spec
	;

addressable_spec: CONST_DIRECTIVE {  recognizer->add_space_spec(const_space,$1); }
	| GLOBAL_DIRECTIVE 	  {  recognizer->add_space_spec(global_space,0); }
	| LOCAL_DIRECTIVE 	  {  recognizer->add_space_spec(local_space,0); }
	| PARAM_DIRECTIVE 	  {  recognizer->add_space_spec(param_space_unclassified,0); }
	| SHARED_DIRECTIVE 	  {  recognizer->add_space_spec(shared_space,0); }
	| SSTARR_DIRECTIVE    {  recognizer->add_space_spec(sstarr_space,0); }
	| SURF_DIRECTIVE 	  {  recognizer->add_space_spec(surf_space,0); }
	| TEX_DIRECTIVE 	  {  recognizer->add_space_spec(tex_space,0); }
	;

type_spec: scalar_type 
	|  vector_spec scalar_type 
	;

vector_spec:  V2_TYPE {  recognizer->add_option(V2_TYPE); recognizer->func_header_info(".v2");}
	| V3_TYPE     {  recognizer->add_option(V3_TYPE); recognizer->func_header_info(".v3");}
	| V4_TYPE     {  recognizer->add_option(V4_TYPE); recognizer->func_header_info(".v4");}
	;

scalar_type: S8_TYPE { recognizer->add_scalar_type_spec( S8_TYPE ); }
	| S16_TYPE   { recognizer->add_scalar_type_spec( S16_TYPE ); }
	| S32_TYPE   { recognizer->add_scalar_type_spec( S32_TYPE ); }
	| S64_TYPE   { recognizer->add_scalar_type_spec( S64_TYPE ); }
	| U8_TYPE    { recognizer->add_scalar_type_spec( U8_TYPE ); }
	| U16_TYPE   { recognizer->add_scalar_type_spec( U16_TYPE ); }
	| U32_TYPE   { recognizer->add_scalar_type_spec( U32_TYPE ); }
	| U64_TYPE   { recognizer->add_scalar_type_spec( U64_TYPE ); }
	| F16_TYPE   { recognizer->add_scalar_type_spec( F16_TYPE ); }
	| F32_TYPE   { recognizer->add_scalar_type_spec( F32_TYPE ); }
	| F64_TYPE   { recognizer->add_scalar_type_spec( F64_TYPE ); }
	| FF64_TYPE   { recognizer->add_scalar_type_spec( FF64_TYPE ); }
	| B8_TYPE    { recognizer->add_scalar_type_spec( B8_TYPE );  }
	| B16_TYPE   { recognizer->add_scalar_type_spec( B16_TYPE ); }
	| B32_TYPE   { recognizer->add_scalar_type_spec( B32_TYPE ); }
	| B64_TYPE   { recognizer->add_scalar_type_spec( B64_TYPE ); }
	| BB64_TYPE   { recognizer->add_scalar_type_spec( BB64_TYPE ); }
	| BB128_TYPE   { recognizer->add_scalar_type_spec( BB128_TYPE ); }
	| PRED_TYPE  { recognizer->add_scalar_type_spec( PRED_TYPE ); }
	| TEXREF_TYPE  { recognizer->add_scalar_type_spec( TEXREF_TYPE ); }
	| SAMPLERREF_TYPE  { recognizer->add_scalar_type_spec( SAMPLERREF_TYPE ); }
	| SURFREF_TYPE  { recognizer->add_scalar_type_spec( SURFREF_TYPE ); }
	;

initializer_list: LEFT_BRACE literal_list RIGHT_BRACE { recognizer->add_array_initializer(); }
	| LEFT_BRACE initializer_list RIGHT_BRACE { syntax_not_implemented(scanner, recognizer); }

literal_list: literal_operand
	| literal_list COMMA literal_operand;

// TODO: This is currently hardcoded to handle and ignore one specific case
// that all prototype statements follow in the PTX from Pytorch. As a
// workaround, this parses and ignores both the prototype declaration 
// and calling of the prototype (which conveniently comes right after the 
// declaration for all cases.) This should be changed to handle both 
// declaring the prototype, and actually calling it.
prototype_block: prototype_decl prototype_call

prototype_decl: IDENTIFIER COLON CALLPROTOTYPE_DIRECTIVE LEFT_PAREN prototype_param RIGHT_PAREN IDENTIFIER LEFT_PAREN prototype_param RIGHT_PAREN SEMI_COLON 
	      
prototype_call: OPCODE LEFT_PAREN IDENTIFIER RIGHT_PAREN COMMA operand COMMA LEFT_PAREN IDENTIFIER RIGHT_PAREN COMMA IDENTIFIER SEMI_COLON
	      | OPCODE IDENTIFIER COMMA LEFT_PAREN IDENTIFIER RIGHT_PAREN COMMA IDENTIFIER SEMI_COLON

prototype_param: /* empty */
	       | PARAM_DIRECTIVE B64_TYPE IDENTIFIER
	       | PARAM_DIRECTIVE B32_TYPE IDENTIFIER

instruction_statement:  instruction SEMI_COLON
	| IDENTIFIER COLON { recognizer->add_label($1); }
	| pred_spec instruction SEMI_COLON;

instruction: opcode_spec LEFT_PAREN operand RIGHT_PAREN { recognizer->set_return(); } COMMA operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand COMMA LEFT_PAREN RIGHT_PAREN
	| opcode_spec operand_list 
	| opcode_spec
	;

opcode_spec: OPCODE { recognizer->add_opcode($1); } option_list
	| OPCODE { recognizer->add_opcode($1); }

pred_spec: PRED IDENTIFIER  { recognizer->add_pred($2,0, -1); }
	| PRED EXCLAMATION IDENTIFIER { recognizer->add_pred($3,1, -1); }
	| PRED IDENTIFIER LT_OPTION  { recognizer->add_pred($2,0,1); }
	| PRED IDENTIFIER EQ_OPTION  { recognizer->add_pred($2,0,2); }
	| PRED IDENTIFIER LE_OPTION  { recognizer->add_pred($2,0,3); }
	| PRED IDENTIFIER NE_OPTION  { recognizer->add_pred($2,0,5); }
	| PRED IDENTIFIER GE_OPTION  { recognizer->add_pred($2,0,6); }
	| PRED IDENTIFIER EQU_OPTION  { recognizer->add_pred($2,0,10); }
	| PRED IDENTIFIER GTU_OPTION  { recognizer->add_pred($2,0,12); }
	| PRED IDENTIFIER NEU_OPTION  { recognizer->add_pred($2,0,13); }
	| PRED IDENTIFIER CF_OPTION  { recognizer->add_pred($2,0,17); }
	| PRED IDENTIFIER SF_OPTION  { recognizer->add_pred($2,0,19); }
	| PRED IDENTIFIER NSF_OPTION  { recognizer->add_pred($2,0,28); }
	;

option_list: option
	| option option_list ;

option: type_spec
	| compare_spec
	| addressable_spec
	| rounding_mode
	| wmma_spec 
	| prmt_spec 
	| SYNC_OPTION { recognizer->add_option(SYNC_OPTION); }
	| ARRIVE_OPTION { recognizer->add_option(ARRIVE_OPTION); }
	| RED_OPTION { recognizer->add_option(RED_OPTION); }
	| UNI_OPTION { recognizer->add_option(UNI_OPTION); }
	| WIDE_OPTION { recognizer->add_option(WIDE_OPTION); }
	| ANY_OPTION { recognizer->add_option(ANY_OPTION); }
	| ALL_OPTION { recognizer->add_option(ALL_OPTION); }
	| BALLOT_OPTION { recognizer->add_option(BALLOT_OPTION); }
	| GLOBAL_OPTION { recognizer->add_option(GLOBAL_OPTION); }
	| CTA_OPTION { recognizer->add_option(CTA_OPTION); }
	| SYS_OPTION { recognizer->add_option(SYS_OPTION); }
	| GEOM_MODIFIER_1D { recognizer->add_option(GEOM_MODIFIER_1D); }
	| GEOM_MODIFIER_2D { recognizer->add_option(GEOM_MODIFIER_2D); }
	| GEOM_MODIFIER_3D { recognizer->add_option(GEOM_MODIFIER_3D); }
	| SAT_OPTION { recognizer->add_option(SAT_OPTION); }
	| FTZ_OPTION { recognizer->add_option(FTZ_OPTION); }
	| NEG_OPTION { recognizer->add_option(NEG_OPTION); }
	| APPROX_OPTION { recognizer->add_option(APPROX_OPTION); }
	| FULL_OPTION { recognizer->add_option(FULL_OPTION); }
	| EXIT_OPTION { recognizer->add_option(EXIT_OPTION); }
	| ABS_OPTION { recognizer->add_option(ABS_OPTION); }
	| atomic_operation_spec ;
	| TO_OPTION { recognizer->add_option(TO_OPTION); }
	| HALF_OPTION { recognizer->add_option(HALF_OPTION); }
	| EXTP_OPTION { recognizer->add_option(EXTP_OPTION); }
	| CA_OPTION { recognizer->add_option(CA_OPTION); }
	| CG_OPTION { recognizer->add_option(CG_OPTION); }
	| CS_OPTION { recognizer->add_option(CS_OPTION); }
	| LU_OPTION { recognizer->add_option(LU_OPTION); }
	| CV_OPTION { recognizer->add_option(CV_OPTION); }
	| WB_OPTION { recognizer->add_option(WB_OPTION); }
	| WT_OPTION { recognizer->add_option(WT_OPTION); }
	| NC_OPTION { recognizer->add_option(NC_OPTION); }
	| UP_OPTION { recognizer->add_option(UP_OPTION); }
	| DOWN_OPTION { recognizer->add_option(DOWN_OPTION); }
	| BFLY_OPTION { recognizer->add_option(BFLY_OPTION); }
	| IDX_OPTION { recognizer->add_option(IDX_OPTION); }
	;

atomic_operation_spec: ATOMIC_AND { recognizer->add_option(ATOMIC_AND); }
	| ATOMIC_POPC { recognizer->add_option(ATOMIC_POPC); }
	| ATOMIC_OR { recognizer->add_option(ATOMIC_OR); }
	| ATOMIC_XOR { recognizer->add_option(ATOMIC_XOR); }
	| ATOMIC_CAS { recognizer->add_option(ATOMIC_CAS); }
	| ATOMIC_EXCH { recognizer->add_option(ATOMIC_EXCH); }
	| ATOMIC_ADD { recognizer->add_option(ATOMIC_ADD); }
	| ATOMIC_INC { recognizer->add_option(ATOMIC_INC); }
	| ATOMIC_DEC { recognizer->add_option(ATOMIC_DEC); }
	| ATOMIC_MIN { recognizer->add_option(ATOMIC_MIN); }
	| ATOMIC_MAX { recognizer->add_option(ATOMIC_MAX); }
	;

rounding_mode: floating_point_rounding_mode
	| integer_rounding_mode;


floating_point_rounding_mode: RN_OPTION { recognizer->add_option(RN_OPTION); }
	| RZ_OPTION { recognizer->add_option(RZ_OPTION); }
	| RM_OPTION { recognizer->add_option(RM_OPTION); }
	| RP_OPTION { recognizer->add_option(RP_OPTION); }
	;

integer_rounding_mode: RNI_OPTION { recognizer->add_option(RNI_OPTION); }
	| RZI_OPTION { recognizer->add_option(RZI_OPTION); }
	| RMI_OPTION { recognizer->add_option(RMI_OPTION); }
	| RPI_OPTION { recognizer->add_option(RPI_OPTION); }
	;

compare_spec:EQ_OPTION { recognizer->add_option(EQ_OPTION); }
	| NE_OPTION { recognizer->add_option(NE_OPTION); }
	| LT_OPTION { recognizer->add_option(LT_OPTION); }
	| LE_OPTION { recognizer->add_option(LE_OPTION); }
	| GT_OPTION { recognizer->add_option(GT_OPTION); }
	| GE_OPTION { recognizer->add_option(GE_OPTION); }
	| LO_OPTION { recognizer->add_option(LO_OPTION); }
	| LS_OPTION { recognizer->add_option(LS_OPTION); }
	| HI_OPTION { recognizer->add_option(HI_OPTION); }
	| HS_OPTION  { recognizer->add_option(HS_OPTION); }
	| EQU_OPTION { recognizer->add_option(EQU_OPTION); }
	| NEU_OPTION { recognizer->add_option(NEU_OPTION); }
	| LTU_OPTION { recognizer->add_option(LTU_OPTION); }
	| LEU_OPTION { recognizer->add_option(LEU_OPTION); }
	| GTU_OPTION { recognizer->add_option(GTU_OPTION); }
	| GEU_OPTION { recognizer->add_option(GEU_OPTION); }
	| NUM_OPTION { recognizer->add_option(NUM_OPTION); }
	| NAN_OPTION { recognizer->add_option(NAN_OPTION); }
	;

prmt_spec: PRMT_F4E_MODE { recognizer->add_option( PRMT_F4E_MODE); }
	|  PRMT_B4E_MODE { recognizer->add_option( PRMT_B4E_MODE); }
	|  PRMT_RC8_MODE { recognizer->add_option( PRMT_RC8_MODE); }
	|  PRMT_RC16_MODE{ recognizer->add_option( PRMT_RC16_MODE);}
	|  PRMT_ECL_MODE { recognizer->add_option( PRMT_ECL_MODE); }
	|  PRMT_ECR_MODE { recognizer->add_option( PRMT_ECR_MODE); }
	;

wmma_spec: WMMA_DIRECTIVE LAYOUT CONFIGURATION{recognizer->add_space_spec(global_space,0);recognizer->add_ptr_spec(global_space); recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);}
	| WMMA_DIRECTIVE LAYOUT LAYOUT CONFIGURATION{recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);recognizer->add_wmma_option($4);}
	;

vp_spec: WMMA_DIRECTIVE LAYOUT CONFIGURATION{recognizer->add_space_spec(global_space,0);recognizer->add_ptr_spec(global_space);recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);}
	| WMMA_DIRECTIVE LAYOUT LAYOUT CONFIGURATION{recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);recognizer->add_wmma_option($4);}
	;



operand_list: operand
	| operand COMMA operand_list;

operand: IDENTIFIER  { recognizer->add_scalar_operand( $1 ); }
	| EXCLAMATION IDENTIFIER { recognizer->add_neg_pred_operand( $2 ); }
	| MINUS IDENTIFIER  { recognizer->add_scalar_operand( $2 ); recognizer->change_operand_neg(); }
	| memory_operand
	| literal_operand
	| builtin_operand
	| vector_operand
	| MINUS vector_operand { recognizer->change_operand_neg(); }
	| tex_operand
	| IDENTIFIER PLUS INT_OPERAND { recognizer->add_address_operand($1,$3); }
	| IDENTIFIER LO_OPTION { recognizer->add_scalar_operand( $1 ); recognizer->change_operand_lohi(1);}
	| MINUS IDENTIFIER LO_OPTION { recognizer->add_scalar_operand( $2 ); recognizer->change_operand_lohi(1); recognizer->change_operand_neg();}
	| IDENTIFIER HI_OPTION { recognizer->add_scalar_operand( $1 ); recognizer->change_operand_lohi(2);}
	| MINUS IDENTIFIER HI_OPTION { recognizer->add_scalar_operand( $2 ); recognizer->change_operand_lohi(2); recognizer->change_operand_neg();}
	| IDENTIFIER PIPE IDENTIFIER { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-1);}
	| IDENTIFIER PIPE IDENTIFIER LO_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-1); recognizer->change_operand_lohi(1);}
	| IDENTIFIER PIPE IDENTIFIER HI_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-1); recognizer->change_operand_lohi(2);}
	| IDENTIFIER BACKSLASH IDENTIFIER { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-3);}
	| IDENTIFIER BACKSLASH IDENTIFIER LO_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-3); recognizer->change_operand_lohi(1);}
	| IDENTIFIER BACKSLASH IDENTIFIER HI_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-3); recognizer->change_operand_lohi(2);}
	;

vector_operand: LEFT_BRACE IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_2vector_operand($2,$4); }
		| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_3vector_operand($2,$4,$6); }
		| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_4vector_operand($2,$4,$6,$8); }
		| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_8vector_operand($2,$4,$6,$8,$10,$12,$14,$16); }
		| LEFT_BRACE IDENTIFIER RIGHT_BRACE { recognizer->add_1vector_operand($2); }
	;

tex_operand: LEFT_SQUARE_BRACKET IDENTIFIER COMMA { recognizer->add_scalar_operand($2); }
		vector_operand 
	     RIGHT_SQUARE_BRACKET
	;

builtin_operand: SPECIAL_REGISTER DIMENSION_MODIFIER { recognizer->add_builtin_operand($1,$2); }
        | SPECIAL_REGISTER { recognizer->add_builtin_operand($1,-1); }
	;

memory_operand : LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { recognizer->add_memory_operand(); }
	| IDENTIFIER LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { recognizer->add_memory_operand(); recognizer->change_memory_addr_space($1); }
	| IDENTIFIER LEFT_SQUARE_BRACKET literal_operand RIGHT_SQUARE_BRACKET { recognizer->change_memory_addr_space($1); }
	| IDENTIFIER LEFT_SQUARE_BRACKET twin_operand RIGHT_SQUARE_BRACKET { recognizer->change_memory_addr_space($1); recognizer->add_memory_operand();}
        | MINUS memory_operand { recognizer->change_operand_neg(); }
	;

twin_operand : IDENTIFIER PLUS IDENTIFIER { recognizer->add_double_operand($1,$3); recognizer->change_double_operand_type(1); }
	| IDENTIFIER PLUS IDENTIFIER LO_OPTION { recognizer->add_double_operand($1,$3); recognizer->change_double_operand_type(1); recognizer->change_operand_lohi(1); }
	| IDENTIFIER PLUS IDENTIFIER HI_OPTION { recognizer->add_double_operand($1,$3); recognizer->change_double_operand_type(1); recognizer->change_operand_lohi(2); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER  { recognizer->add_double_operand($1,$4); recognizer->change_double_operand_type(2); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER LO_OPTION { recognizer->add_double_operand($1,$4); recognizer->change_double_operand_type(2); recognizer->change_operand_lohi(1); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER HI_OPTION { recognizer->add_double_operand($1,$4); recognizer->change_double_operand_type(2); recognizer->change_operand_lohi(2); }
	| IDENTIFIER PLUS EQUALS INT_OPERAND  { recognizer->add_address_operand($1,$4); recognizer->change_double_operand_type(3); }
	;

literal_operand : INT_OPERAND { recognizer->add_literal_int($1); }
	| FLOAT_OPERAND { recognizer->add_literal_float($1); }
	| DOUBLE_OPERAND { recognizer->add_literal_double($1); }
	;

address_expression: IDENTIFIER { recognizer->add_address_operand($1,0); }
	| IDENTIFIER LO_OPTION { recognizer->add_address_operand($1,0); recognizer->change_operand_lohi(1);}
	| IDENTIFIER HI_OPTION { recognizer->add_address_operand($1,0); recognizer->change_operand_lohi(2); }
	| IDENTIFIER PLUS INT_OPERAND { recognizer->add_address_operand($1,$3); }
	| INT_OPERAND { recognizer->add_address_operand2($1); }
	;

%%

void syntax_not_implemented(yyscan_t yyscanner, ptx_recognizer* recognizer)
{
	printf("Parse error (%s): this syntax is not (yet) implemented:\n", recognizer->gpgpu_ctx->g_filename);
	ptx_error(yyscanner, recognizer, NULL);
	abort();
}
