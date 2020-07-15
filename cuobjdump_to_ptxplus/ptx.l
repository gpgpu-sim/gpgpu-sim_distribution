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

%option nounput
%option noyywrap
%option yylineno
%option prefix="ptx_"
%{
#include "opcodes.h"
#include "ptx.tab.h"
#include <string.h>

#define LINEBUF_SIZE (64*1024)
char linebuf[LINEBUF_SIZE];
unsigned col = 0;
#define TC col+=strlen(ptx_text); 
#define CHECK_UNSIGNED \
	if( yytext[strlen(yytext)-1]=='U' ) { \
		printf("GPGPU-Sim: ERROR ** U modifier not implemented\n"); \
		abort(); \
	}
int ptx_error( const char *s );
%}

%s IN_STRING
%s IN_COMMENT
%x IN_FUNC_DECL
%x NOT_OPCODE
%%

abs	TC; ptx_lval.int_value = ABS_OP; return OPCODE;
add	TC; ptx_lval.int_value = ADD_OP; return OPCODE;
addp	TC; ptx_lval.int_value = ADDP_OP; return OPCODE;
addc    TC; ptx_lval.int_value = ADDC_OP; return OPCODE;
and	TC; ptx_lval.int_value = AND_OP; return OPCODE;
andn	TC; ptx_lval.int_value = ANDN_OP; return OPCODE;
atom	TC; ptx_lval.int_value = ATOM_OP; return OPCODE;
bar.warp 	TC; ptx_lval.int_value = NOP_OP; return OPCODE;
bar 	TC; ptx_lval.int_value = BAR_OP; return OPCODE;
bfe     TC; ptx_lval.int_value = BFE_OP; return OPCODE;
bfi     TC; ptx_lval.int_value = BFI_OP; return OPCODE;
bfind   TC; ptx_lval.int_value = BFIND_OP; return OPCODE;
bra     TC; ptx_lval.int_value = BRA_OP; return OPCODE;
brx     TC; ptx_lval.int_value = BRX_OP; return OPCODE;
brev    TC; ptx_lval.int_value = BREV_OP; return OPCODE;
brkpt   TC; ptx_lval.int_value = BRKPT_OP; return OPCODE;

wmma	TC; ptx_lval.int_value = MMA_OP; return OPCODE;
wmma\.load	TC; ptx_lval.int_value = MMA_LD_OP; return OPCODE;
wmma\.store 	TC; ptx_lval.int_value = MMA_ST_OP; return OPCODE;

call	TC; BEGIN(NOT_OPCODE); ptx_lval.int_value = CALL_OP; return OPCODE; // blocking opcode token in case the callee has the same name as an opcode
callp   TC; BEGIN(NOT_OPCODE); ptx_lval.int_value = CALLP_OP; return OPCODE;
clz	TC; ptx_lval.int_value = CLZ_OP; return OPCODE;
cnot	TC; ptx_lval.int_value = CNOT_OP; return OPCODE;
cos	TC; ptx_lval.int_value = COS_OP; return OPCODE;
cvt	TC; ptx_lval.int_value = CVT_OP; return OPCODE;
cvta	TC; ptx_lval.int_value = CVTA_OP; return OPCODE;
div	TC; ptx_lval.int_value = DIV_OP; return OPCODE;
dp4a	TC; ptx_lval.int_value = DP4A_OP; return OPCODE;
ex2	TC; ptx_lval.int_value = EX2_OP; return OPCODE;
exit	TC; ptx_lval.int_value = EXIT_OP; return OPCODE;
fma     TC; ptx_lval.int_value = FMA_OP; return OPCODE;
isspacep TC; ptx_lval.int_value = ISSPACEP_OP; return OPCODE;
ld      TC; ptx_lval.int_value = LD_OP; return OPCODE;
ld.volatile TC; ptx_lval.int_value = LD_OP; return OPCODE;
ldu     TC; ptx_lval.int_value = LDU_OP; return OPCODE;
lg2	TC; ptx_lval.int_value = LG2_OP; return OPCODE;
mad24   TC; ptx_lval.int_value = MAD24_OP; return OPCODE;
mad     TC; ptx_lval.int_value = MAD_OP; return OPCODE;
madc	TC; ptx_lval.int_value = MADC_OP; return OPCODE;
madp	TC; ptx_lval.int_value = MADP_OP; return OPCODE;
max     TC; ptx_lval.int_value = MAX_OP; return OPCODE;
membar  TC; ptx_lval.int_value = MEMBAR_OP; return OPCODE;
min     TC; ptx_lval.int_value = MIN_OP; return OPCODE;
mov     TC; ptx_lval.int_value = MOV_OP; return OPCODE;
mul24   TC; ptx_lval.int_value = MUL24_OP; return OPCODE;
mul     TC; ptx_lval.int_value = MUL_OP; return OPCODE;
neg     TC; ptx_lval.int_value = NEG_OP; return OPCODE;
nandn   TC; ptx_lval.int_value = NANDN_OP; return OPCODE;
norn    TC; ptx_lval.int_value = NORN_OP; return OPCODE;
not     TC; ptx_lval.int_value = NOT_OP; return OPCODE;
or      TC; ptx_lval.int_value = OR_OP; return OPCODE;
orn     TC; ptx_lval.int_value = ORN_OP; return OPCODE;
pmevent TC; ptx_lval.int_value = PMEVENT_OP; return OPCODE;
popc    TC; ptx_lval.int_value = POPC_OP; return OPCODE;
prefetch TC; ptx_lval.int_value = PREFETCH_OP; return OPCODE;
prefetchu TC; ptx_lval.int_value = PREFETCHU_OP; return OPCODE;
prmt    TC; ptx_lval.int_value = PRMT_OP; return OPCODE;
rcp	TC; ptx_lval.int_value = RCP_OP; return OPCODE;
red     TC; ptx_lval.int_value = RED_OP; return OPCODE;
rem	TC; ptx_lval.int_value = REM_OP; return OPCODE;
ret	TC; ptx_lval.int_value = RET_OP; return OPCODE;
retp     TC; ptx_lval.int_value = RETP_OP; return OPCODE;
rsqrt	TC; ptx_lval.int_value = RSQRT_OP; return OPCODE;
sad     TC; ptx_lval.int_value = SAD_OP; return OPCODE;
selp	TC; ptx_lval.int_value = SELP_OP; return OPCODE;
setp    TC; ptx_lval.int_value = SETP_OP; return OPCODE;
set	TC; ptx_lval.int_value = SET_OP; return OPCODE;
shfl	TC; ptx_lval.int_value = SHFL_OP; return OPCODE;
shl     TC; ptx_lval.int_value = SHL_OP; return OPCODE;
shr     TC; ptx_lval.int_value = SHR_OP; return OPCODE;
sin	TC; ptx_lval.int_value = SIN_OP; return OPCODE;
slct	TC; ptx_lval.int_value = SLCT_OP; return OPCODE;
sqrt	TC; ptx_lval.int_value = SQRT_OP; return OPCODE;
sst	TC; ptx_lval.int_value = SST_OP; return OPCODE;
ssy     TC; ptx_lval.int_value = SSY_OP; return OPCODE;
st      TC; ptx_lval.int_value = ST_OP; return OPCODE;
st.volatile TC; ptx_lval.int_value = ST_OP; return OPCODE;
sub	TC; ptx_lval.int_value = SUB_OP; return OPCODE;
subc	TC; ptx_lval.int_value = SUBC_OP; return OPCODE;
suld	TC; ptx_lval.int_value = SULD_OP; return OPCODE;
sured	TC; ptx_lval.int_value = SURED_OP; return OPCODE;
surst	TC; ptx_lval.int_value = SUST_OP; return OPCODE;
suq	TC; ptx_lval.int_value = SUQ_OP; return OPCODE;
tex	TC; BEGIN(NOT_OPCODE); ptx_lval.int_value = TEX_OP; return OPCODE;
txq	TC; ptx_lval.int_value = TEX_OP; return OPCODE;
trap	TC; ptx_lval.int_value = TRAP_OP; return OPCODE;
vabsdiff TC; ptx_lval.int_value = VABSDIFF_OP; return OPCODE;
vadd    TC; ptx_lval.int_value = VADD_OP; return OPCODE;
vmad    TC; ptx_lval.int_value = VMAD_OP; return OPCODE;
vmax    TC; ptx_lval.int_value = VMAX_OP; return OPCODE;
vmin    TC; ptx_lval.int_value = VMIN_OP; return OPCODE;
vset    TC; ptx_lval.int_value = VSET_OP; return OPCODE;
vshl    TC; ptx_lval.int_value = VSHL_OP; return OPCODE;
vshr    TC; ptx_lval.int_value = VSHR_OP; return OPCODE;
vsub    TC; ptx_lval.int_value = VSUB_OP; return OPCODE;
vote	TC; ptx_lval.int_value = VOTE_OP; return OPCODE;
xor     TC; ptx_lval.int_value = XOR_OP; return OPCODE;
nop     TC; ptx_lval.int_value = NOP_OP; return OPCODE;
break  TC; ptx_lval.int_value = BREAK_OP; return OPCODE;
breakaddr  TC; ptx_lval.int_value = BREAKADDR_OP; return OPCODE;

"CPTX_END"	printf("ENDING CUSTOM PTX.\n"); BEGIN(IN_COMMENT);

<INITIAL,NOT_OPCODE,IN_INST,IN_FUNC_DECL>{
\.a\.sync TC; ptx_lval.int_value = LOAD_A; return WMMA_DIRECTIVE;
\.b\.sync TC; ptx_lval.int_value = LOAD_B; return WMMA_DIRECTIVE;
\.c\.sync TC; ptx_lval.int_value = LOAD_C; return WMMA_DIRECTIVE;
\.d\.sync TC; ptx_lval.int_value = STORE_D; return WMMA_DIRECTIVE;
\.mma\.sync TC;ptx_lval.int_value=MMA; return WMMA_DIRECTIVE;

\.row TC; ptx_lval.int_value = ROW; return LAYOUT;
\.col TC; ptx_lval.int_value = COL; return LAYOUT;
\.m16n16k16 TC; ptx_lval.int_value = M16N16K16; return CONFIGURATION;
\.f4e TC;  return PRMT_F4E_MODE;
\.b4e TC;  return PRMT_B4E_MODE;
\.rc8 TC;  return PRMT_RC8_MODE;
\.ecl TC;  return PRMT_ECL_MODE;
\.ecr TC;  return PRMT_ECR_MODE;
\.rc16 TC; return PRMT_RC16_MODE;

\.align TC; return ALIGN_DIRECTIVE;
\.branchtargets TC; return BRANCHTARGETS_DIRECTIVE;
\.byte	TC; return BYTE_DIRECTIVE; /* not in PTX 2.1 */
\.callprototype TC; return CALLPROTOTYPE_DIRECTIVE;
\.calltargets TC; return CALLTARGETS_DIRECTIVE;
\.const\[[0-9]+\] TC; ptx_lval.int_value = atoi(yytext+7); return CONST_DIRECTIVE;
\.const TC; ptx_lval.int_value = 0; return CONST_DIRECTIVE;
\.entry TC; return ENTRY_DIRECTIVE;
\.extern TC; return EXTERN_DIRECTIVE;
\.file	 TC; BEGIN(INITIAL); return FILE_DIRECTIVE;
\.func   TC; BEGIN(IN_FUNC_DECL); return FUNC_DIRECTIVE; // blocking opcode parsing in case the function has the same name as an opcode (e.g. sin(), cos())
\.global TC; return GLOBAL_DIRECTIVE;
\.global.volatile TC; return GLOBAL_DIRECTIVE; //TODO: fix this!
\.local  TC; return LOCAL_DIRECTIVE;
\.loc    TC; return LOC_DIRECTIVE;
\.maxnctapersm TC; return MAXNCTAPERSM_DIRECTIVE;
\.maxnreg TC; return MAXNNREG_DIRECTIVE;
\.maxntid TC; return MAXNTID_DIRECTIVE;
\.minnctapersm TC; return MINNCTAPERSM_DIRECTIVE;
\.param  TC; return PARAM_DIRECTIVE;
\.pragma TC; return PRAGMA_DIRECTIVE;
\.reg    TC; return REG_DIRECTIVE;
\.reqntid TC; return REQNTID_DIRECTIVE;
\.section TC; return SECTION_DIRECTIVE;
\.shared  TC; return SHARED_DIRECTIVE;
\.sreg   TC; return SREG_DIRECTIVE;
\.sstarr TC; return SSTARR_DIRECTIVE;
\.struct TC; return STRUCT_DIRECTIVE;
\.surf   TC; return SURF_DIRECTIVE;   /* not in PTX 2.1 */
\.target  TC; return TARGET_DIRECTIVE;
\.tex   TC; BEGIN(NOT_OPCODE); return TEX_DIRECTIVE;
\.union  TC; return UNION_DIRECTIVE; /* not in PTX 2.1 */
\.version TC; return VERSION_DIRECTIVE;
\.visible TC; return VISIBLE_DIRECTIVE;
\.weak TC; return WEAK_DIRECTIVE;
\.address_size TC; return ADDRESS_SIZE_DIRECTIVE;
\.weak	TC; return WEAK_DIRECTIVE;

\.constptr TC; return CONSTPTR_DIRECTIVE; /* Ptx plus directive for pointer to constant memory */
\.ptr TC; return PTR_DIRECTIVE; /* Added for new OpenCL genrated code */

"%clock" TC; ptx_lval.int_value = CLOCK_REG; return SPECIAL_REGISTER;
"%halfclock" TC; ptx_lval.int_value = HALFCLOCK_ID; return SPECIAL_REGISTER;
"%clock64" TC; ptx_lval.int_value = CLOCK64_REG; return SPECIAL_REGISTER;
"%ctaid" TC; ptx_lval.int_value = CTAID_REG; return SPECIAL_REGISTER;
"%envreg"[0-9]+ TC; sscanf(yytext+7,"%u",&ptx_lval.int_value); ptx_lval.int_value<<=16; ptx_lval.int_value += ENVREG_REG; return SPECIAL_REGISTER;
"%gridid" TC; ptx_lval.int_value = GRIDID_REG; return SPECIAL_REGISTER;
"%laneid"  TC; ptx_lval.int_value = LANEID_REG; return SPECIAL_REGISTER;
"%lanemask_eq"  TC; ptx_lval.int_value = LANEMASK_EQ_REG; return SPECIAL_REGISTER;
"%lanemask_le"  TC; ptx_lval.int_value = LANEMASK_LE_REG; return SPECIAL_REGISTER;
"%lanemask_lt"  TC; ptx_lval.int_value = LANEMASK_LT_REG; return SPECIAL_REGISTER;
"%lanemask_ge"  TC; ptx_lval.int_value = LANEMASK_GE_REG; return SPECIAL_REGISTER;
"%lanemask_gt"  TC; ptx_lval.int_value = LANEMASK_GT_REG; return SPECIAL_REGISTER;
"%nctaid" TC; ptx_lval.int_value = NCTAID_REG; return SPECIAL_REGISTER;
"%ntid"  TC; ptx_lval.int_value = NTID_REG; return SPECIAL_REGISTER;
"%nsmid"  TC; ptx_lval.int_value = NSMID_REG; return SPECIAL_REGISTER;
"%nwarpid"  TC; ptx_lval.int_value = NWARPID_REG; return SPECIAL_REGISTER;
"%pm"[0-3]  TC; sscanf(yytext+3,"%u",&ptx_lval.int_value); ptx_lval.int_value<<=16; ptx_lval.int_value += PM_REG; return SPECIAL_REGISTER;
"%smid"  TC; ptx_lval.int_value = SMID_REG; return SPECIAL_REGISTER;
"%tid"  TC; ptx_lval.int_value = TID_REG; return SPECIAL_REGISTER;
"%warpid"  TC; ptx_lval.int_value = WARPID_REG; return SPECIAL_REGISTER;
"WARP_SZ"  TC; ptx_lval.int_value = WARPSZ_REG; return SPECIAL_REGISTER;

[a-zA-Z_][a-zA-Z0-9_$]*  TC; ptx_lval.string_value = strdup(yytext); return IDENTIFIER;
[$%][a-zA-Z0-9_$]+  TC; ptx_lval.string_value = strdup(yytext); return IDENTIFIER;

[0-9]+\.[0-9]+ 	 TC; sscanf(yytext,"%lf", &ptx_lval.double_value); return DOUBLE_OPERAND;
	
0[xX][0-9a-fA-F]+U? TC; CHECK_UNSIGNED; sscanf(yytext,"%x", &ptx_lval.int_value); return INT_OPERAND;
0[0-7]+U?   	TC; printf("GPGPU-Sim: ERROR ** parsing octal not (yet) implemented\n"); abort(); return INT_OPERAND;
0[bB][01]+U?  	TC; printf("GPGPU-Sim: ERROR ** parsing binary not (yet) implemented\n"); abort(); return INT_OPERAND;
[-]?[0-9]+U?    TC; CHECK_UNSIGNED; ptx_lval.int_value =  atoi(yytext); return INT_OPERAND;

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
\.f16x2  TC;  return F16_TYPE; /* TODO: figure out what this should really be */
\.f32  TC;  return F32_TYPE;
\.f64  TC;  return F64_TYPE;
\.ff64 TC;  return FF64_TYPE;
\.b8   TC;  return B8_TYPE;
\.b16  TC;  return B16_TYPE;
\.b32  TC;  return B32_TYPE;
\.b64  TC;  return B64_TYPE;
\.bb64  TC;  return BB64_TYPE;
\.bb128  TC;  return BB128_TYPE;
\.pred TC;  return PRED_TYPE;

\.texref TC; BEGIN(NOT_OPCODE); return TEXREF_TYPE;
\.samplerref TC;  return SAMPLERREF_TYPE;
\.surfref TC;  return SURFREF_TYPE;

\.v2	TC; return V2_TYPE;
\.v3	TC; return V3_TYPE;
\.v4	TC; return V4_TYPE;

\.half  TC; return HALF_OPTION; /* ptxplus */
\.cc	TC; return EXTP_OPTION; /* extended precision option */

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
\.cf    TC; return CF_OPTION;
\.sf    TC; return SF_OPTION;
\.nsf    TC; return NSF_OPTION;

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

\.neg	TC; return NEG_OPTION;

\.wide  TC; return WIDE_OPTION;
\.uni	TC; return UNI_OPTION;

\.sync TC; return SYNC_OPTION;
\.arrive TC; return ARRIVE_OPTION;
\.red TC; return RED_OPTION;


\.approx TC; return APPROX_OPTION;
\.full  TC; return FULL_OPTION;

\.any	TC; return ANY_OPTION;
\.all	TC; return ALL_OPTION;
\.ballot	TC; return BALLOT_OPTION;
\.gl    TC; return GLOBAL_OPTION;
\.cta   TC; return CTA_OPTION;
\.sys   TC; return SYS_OPTION;

\.exit   TC; return EXIT_OPTION;

\.abs   TC; return ABS_OPTION;

\.to	TC; return TO_OPTION;

\.ca    TC; return CA_OPTION;
\.cg    TC; return CG_OPTION;
\.cs    TC; return CS_OPTION;
\.lu    TC; return LU_OPTION;
\.cv    TC; return CV_OPTION;

\.wb	TC; return WB_OPTION;
\.wt	TC; return WT_OPTION;

\.nc	TC; return NC_OPTION;

\.up	TC; return UP_OPTION;
\.down	TC; return DOWN_OPTION;
\.bfly	TC; return BFLY_OPTION;
\.idx	TC; return IDX_OPTION;

\.popc	TC; return ATOMIC_POPC;
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

"-"	TC; return MINUS;
"+"	TC; return PLUS;
","     TC; return COMMA;
"@" 	TC; return PRED;
"|" 	TC; return PIPE;
"["     TC; return LEFT_SQUARE_BRACKET;
"]"	TC; return RIGHT_SQUARE_BRACKET;
"<"     TC; return LEFT_ANGLE_BRACKET;
">"	TC; return RIGHT_ANGLE_BRACKET;
"("	TC; return LEFT_PAREN;
")"	TC; return RIGHT_PAREN;
":"	TC; BEGIN(INITIAL); return COLON;
";"	TC; BEGIN(INITIAL); return SEMI_COLON;
"!"	TC; return EXCLAMATION;
"=" 	TC; return EQUALS;
"}"	TC; return RIGHT_BRACE;
\.	TC; return PERIOD;
"/"	TC; return BACKSLASH;

"//"[^\n]* TC;	// eat single

\n.*  col=0; strncpy(linebuf, yytext + 1, LINEBUF_SIZE); yyless( 1 );

" " TC;
"\t" TC;


}

<IN_FUNC_DECL>"{"	TC; BEGIN(INITIAL); return LEFT_BRACE; // starting a statement block (allow next token to be parsed as an opcode)
<INITIAL,NOT_OPCODE>"{"	TC; return LEFT_BRACE; // starting a vector operand (next token cannot be opcode)

<INITIAL>{
"/*"  BEGIN(IN_COMMENT);
}
<IN_COMMENT>{
"*/"      BEGIN(INITIAL);
"CPTX_BEGIN" printf("BEGINNING CUSTOM PTX.\n"); BEGIN(INITIAL);
[^C*\n]+	// eat comment in chunks
"C"			// eat the lone C 
"*"			// eat the lone star
\n        TC; 
}

<INITIAL>{
"\""  BEGIN(IN_STRING);
}
<IN_STRING>{
"\"" 	TC; BEGIN(INITIAL); return STRING;
[^\"]*	TC; ptx_lval.string_value = strdup(yytext); 
}

<*>\t@@DWARF.*\n

<INITIAL,NOT_OPCODE,IN_FUNC_DECL>.  TC; ptx_error((const char*)NULL);
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
