/* 
 * Copyright (c) 2009 by Tor M. Aamodt and the University of British Columbia
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

%option noyywrap
%option yylineno
%option prefix="ptx_"
%{
#include "opcodes.h"
#include "ptx.tab.h"
#include <string.h>

char linebuf[1024];
unsigned col = 0;
#define TC col+=strlen(ptx_text); 
int ptx_error( const char *s );
%}

%s IN_STRING
%s IN_COMMENT
%x NOT_OPCODE
%%

abs	TC; ptx_lval.int_value = ABS_OP; return OPCODE;
add	TC; ptx_lval.int_value = ADD_OP; return OPCODE;
and	TC; ptx_lval.int_value = AND_OP; return OPCODE;
atom	TC; ptx_lval.int_value = ATOM_OP; return OPCODE;
bar.sync TC; ptx_lval.int_value = BAR_OP; return OPCODE;
bra     TC; ptx_lval.int_value = BRA_OP; return OPCODE;
call	TC; ptx_lval.int_value = CALL_OP; return OPCODE;
cnot	TC; ptx_lval.int_value = CNOT_OP; return OPCODE;
cos	TC; ptx_lval.int_value = COS_OP; return OPCODE;
cvt	TC; ptx_lval.int_value = CVT_OP; return OPCODE;
div	TC; ptx_lval.int_value = DIV_OP; return OPCODE;
ex2	TC; ptx_lval.int_value = EX2_OP; return OPCODE;
exit	TC; ptx_lval.int_value = EXIT_OP; return OPCODE;
ld      TC; ptx_lval.int_value = LD_OP; return OPCODE;
ld.volatile TC; ptx_lval.int_value = LD_OP; return OPCODE;
lg2	TC; ptx_lval.int_value = LG2_OP; return OPCODE;
mad24   TC; ptx_lval.int_value = MAD24_OP; return OPCODE;
mad     TC; ptx_lval.int_value = MAD_OP; return OPCODE;
max     TC; ptx_lval.int_value = MAX_OP; return OPCODE;
min     TC; ptx_lval.int_value = MIN_OP; return OPCODE;
mov     TC; ptx_lval.int_value = MOV_OP; return OPCODE;
mul24   TC; ptx_lval.int_value = MUL24_OP; return OPCODE;
mul     TC; ptx_lval.int_value = MUL_OP; return OPCODE;
neg     TC; ptx_lval.int_value = NEG_OP; return OPCODE;
not     TC; ptx_lval.int_value = NOT_OP; return OPCODE;
or      TC; ptx_lval.int_value = OR_OP; return OPCODE;
rcp	TC; ptx_lval.int_value = RCP_OP; return OPCODE;
rem	TC; ptx_lval.int_value = REM_OP; return OPCODE;
ret	TC; ptx_lval.int_value = RET_OP; return OPCODE;
rsqrt	TC; ptx_lval.int_value = RSQRT_OP; return OPCODE;
sad     TC; ptx_lval.int_value = SAD_OP; return OPCODE;
selp	TC; ptx_lval.int_value = SELP_OP; return OPCODE;
setp    TC; ptx_lval.int_value = SETP_OP; return OPCODE;
set	TC; ptx_lval.int_value = SET_OP; return OPCODE;
shl     TC; ptx_lval.int_value = SHL_OP; return OPCODE;
shr     TC; ptx_lval.int_value = SHR_OP; return OPCODE;
sin	TC; ptx_lval.int_value = SIN_OP; return OPCODE;
slct	TC; ptx_lval.int_value = SLCT_OP; return OPCODE;
sqrt	TC; ptx_lval.int_value = SQRT_OP; return OPCODE;
st      TC; ptx_lval.int_value = ST_OP; return OPCODE;
st.volatile TC; ptx_lval.int_value = ST_OP; return OPCODE;
sub	TC; ptx_lval.int_value = SUB_OP; return OPCODE;
tex	TC; BEGIN(NOT_OPCODE); ptx_lval.int_value = TEX_OP; return OPCODE;
trap	TC; ptx_lval.int_value = TRAP_OP; return OPCODE;
vote	TC; ptx_lval.int_value = VOTE_OP; return OPCODE;
xor     TC; ptx_lval.int_value = XOR_OP; return OPCODE;
membar  TC; ptx_lval.int_value = MEMBAR_OP; return OPCODE;
red     TC; ptx_lval.int_value = RED_OP; return OPCODE;
brkpt   TC; ptx_lval.int_value = BRKPT_OP; return OPCODE;
addc    TC; ptx_lval.int_value = ADDC_OP; return OPCODE;

<INITIAL,NOT_OPCODE>{

\.align TC; return ALIGN_DIRECTIVE;
\.byte	TC; return BYTE_DIRECTIVE;
\.const\[[0-9]+\] TC; return CONST_DIRECTIVE;
\.const TC; return CONST_DIRECTIVE;
\.entry TC; return ENTRY_DIRECTIVE;
\.extern TC; return EXTERN_DIRECTIVE;
\.file	 TC; return FILE_DIRECTIVE;
\.func   TC; return FUNC_DIRECTIVE;
\.global TC; return GLOBAL_DIRECTIVE;
\.local  TC; return LOCAL_DIRECTIVE;
\.loc    TC; return LOC_DIRECTIVE;
\.param  TC; return PARAM_DIRECTIVE;
\.reg    TC; return REG_DIRECTIVE;
\.section TC; return SECTION_DIRECTIVE;
\.shared  TC; return SHARED_DIRECTIVE;
\.sreg   TC; return SREG_DIRECTIVE;
\.struct TC; return STRUCT_DIRECTIVE;
\.surf   TC; return SURF_DIRECTIVE;
\.target  TC; return TARGET_DIRECTIVE;
\.tex   TC; BEGIN(NOT_OPCODE); return TEX_DIRECTIVE;
\.union  TC; return UNION_DIRECTIVE;
\.version TC; return VERSION_DIRECTIVE;
\.visible TC; return VISIBLE_DIRECTIVE;

\.maxntid TC; return MAXNTID_DIRECTIVE;

"%clock" TC; ptx_lval.int_value = CLOCK_ID; return SPECIAL_REGISTER;
"%ctaid" TC; ptx_lval.int_value = CTA_ID; return SPECIAL_REGISTER;
"%ntid"  TC; ptx_lval.int_value = NTID_ID; return SPECIAL_REGISTER;
"%gridid" TC; ptx_lval.int_value = GRIDID_ID; return SPECIAL_REGISTER;
"%nctaid" TC; ptx_lval.int_value = NCTAID_ID; return SPECIAL_REGISTER;
"%tid"  TC; ptx_lval.int_value = TID_ID; return SPECIAL_REGISTER;

[_A-Za-z$%][_0-9A-Za-z$]*  TC; ptx_lval.string_value = strdup(yytext); /*printf("\n<<identifier=\"%s\">>\n", yytext); fflush(stdout);*/ return IDENTIFIER;

[0-9]+\.[0-9]+ 	 TC; sscanf(yytext,"%lf", &ptx_lval.double_value); return DOUBLE_OPERAND;
[-]{0,1}[0-9]+		 TC; ptx_lval.int_value =  atoi(yytext); return INT_OPERAND;
0[xX][0-9]+	 TC;  sscanf(yytext,"%x", &ptx_lval.int_value); return INT_OPERAND;
0[fF][0-9a-fA-F]{8}  TC; sscanf(yytext+2,"%x", (unsigned*)(void*)&ptx_lval.float_value); return FLOAT_OPERAND;
0[dD][0-9a-fA-F]{16}  TC; sscanf(yytext+2,"%Lx", (unsigned long long*)(void*)&ptx_lval.double_value); return DOUBLE_OPERAND;

\.s8   TC;  return S8_TYPE;
\.s16  TC;  return S16_TYPE;
\.s32  TC;  return S32_TYPE;
\.s64  TC;  return S64_TYPE;
\.u8   TC;  return U8_TYPE;
\.u16  TC;  return U16_TYPE;
\.u32  TC;  return U32_TYPE;
\.u64  TC;  return U64_TYPE;
\.f16  TC;  return F16_TYPE;
\.f32  TC;  return F32_TYPE;
\.f64  TC;  return F64_TYPE;
\.b8   TC;  return B8_TYPE;
\.b16  TC;  return B16_TYPE;
\.b32  TC;  return B32_TYPE;
\.b64  TC;  return B64_TYPE;
\.pred TC;  return PRED_TYPE;

\.v2	TC; return V2_TYPE;
\.v3	TC; return V3_TYPE;
\.v4	TC; return V4_TYPE;

\.equ	TC; return EQU_OPTION;
\.neu	TC; return NEU_OPTION;
\.ltu	TC; return LTU_OPTION;
\.leu	TC; return LEU_OPTION;
\.gtu	TC; return GTU_OPTION;
\.geu	TC; return GEU_OPTION;
\.num	TC; return NUM_OPTION;
\.nan	TC; return NAN_OPTION;

\.sat	TC; return SAT_OPTION;

\.eq    TC; return EQ_OPTION;
\.ne    TC; return NE_OPTION;
\.lt    TC; return LT_OPTION;
\.le    TC; return LE_OPTION;
\.gt    TC; return GT_OPTION;
\.ge    TC; return GE_OPTION;

\.lo    TC; return LO_OPTION;
\.ls    TC; return LS_OPTION;
\.hi    TC; return HI_OPTION;
\.hs    TC; return HS_OPTION;


\.rni	TC; return RNI_OPTION;
\.rzi	TC; return RZI_OPTION;
\.rmi	TC; return RMI_OPTION;
\.rpi	TC; return RPI_OPTION;

\.rn	TC; return RN_OPTION;
\.rz	TC; return RZ_OPTION;
\.rm	TC; return RM_OPTION;
\.rp	TC; return RP_OPTION;

\.ftz	TC; return FTZ_OPTION;

\.wide  TC; return WIDE_OPTION;
\.uni	TC; return UNI_OPTION;

\.approx TC; return APPROX_OPTION;
\.full  TC; return FULL_OPTION;

\.any	TC; return ANY_OPTION;
\.all	TC; return ALL_OPTION;
\.gl    TC; return GLOBAL_OPTION;
\.cta   TC; return CTA_OPTION;

\.and	TC; return ATOMIC_AND;
\.or	TC; return ATOMIC_OR;
\.xor	TC; return ATOMIC_XOR;
\.cas	TC; return ATOMIC_CAS;
\.exch	TC; return ATOMIC_EXCH;
\.add	TC; return ATOMIC_ADD;
\.inc	TC; return ATOMIC_INC;
\.dec	TC; return ATOMIC_DEC;
\.min	TC; return ATOMIC_MIN;
\.max	TC; return ATOMIC_MAX;

\.1d	TC; return GEOM_MODIFIER_1D;
\.2d	TC; return GEOM_MODIFIER_2D;
\.3d	TC; return GEOM_MODIFIER_3D;

\.0	TC; ptx_lval.int_value = 0; return DIMENSION_MODIFIER;
\.1	TC; ptx_lval.int_value = 1; return DIMENSION_MODIFIER;
\.2	TC; ptx_lval.int_value = 2; return DIMENSION_MODIFIER;
\.x	TC; ptx_lval.int_value = 0; return DIMENSION_MODIFIER;
\.y	TC; ptx_lval.int_value = 1; return DIMENSION_MODIFIER;
\.z	TC; ptx_lval.int_value = 2; return DIMENSION_MODIFIER;

"+"	TC; return PLUS;
","     TC; return COMMA;
"@" 	TC; return PRED;
"["     TC; return LEFT_SQUARE_BRACKET;
"]"	TC; return RIGHT_SQUARE_BRACKET;
"<"     TC; return LEFT_ANGLE_BRACKET;
">"	TC; return RIGHT_ANGLE_BRACKET;
"("	TC; return LEFT_PAREN;
")"	TC; return RIGHT_PAREN;
":"	TC; return COLON;
";"	TC; BEGIN(INITIAL); return SEMI_COLON;
"!"	TC; return EXCLAMATION;
"=" 	TC; return EQUALS;
"{"	TC; return LEFT_BRACE;
"}"	TC; return RIGHT_BRACE;
\.	TC; return PERIOD;

"//"[^\n]* TC;	// eat single

\n.*  col=0; strncpy(linebuf, yytext + 1, 1024); yyless( 1 );

" " TC;
"\t" TC;


}

<INITIAL>{
"/*"  BEGIN(IN_COMMENT);
}
<IN_COMMENT>{
"*/"      BEGIN(INITIAL);
[^*\n]+   // eat comment in chunks
"*"       // eat the lone star
\n        TC; 
}

<INITIAL>{
"\""  BEGIN(IN_STRING);
}
<IN_STRING>{
"\"" 	TC; BEGIN(INITIAL); return STRING;
[^\"]*	TC; ptx_lval.string_value = strdup(yytext); 
}

<INITIAL,NOT_OPCODE>.  TC; ptx_error((const char*)NULL);

%%

extern int g_error_detected;
extern const char *g_filename;

int ptx_error( const char *s )
{
	int i;
	g_error_detected = 1;
	fflush(stdout);
	if( s != NULL )
		printf("%s:%u: Syntax error:\n\n", g_filename, ptx_lineno );
	printf("   %s\n", linebuf );
	printf("   ");
	for( i=0; i < col-1; i++ ) {
		if( linebuf[i] == '\t' ) printf("\t");
		else printf(" ");
	}
			
	printf("^\n\n");
	fflush(stdout);
	//exit(1);
	return 0;
}
