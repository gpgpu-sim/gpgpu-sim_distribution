/*
// Copyright (c) 2009-2011, Jimmy Kwa,
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
*/

%{
#include <iostream>
#include "decudaParser.hpp"
#include <string.h>

/* not sure if this is needed*/
#define	YY_NEVER_INTERACTIVE	1

void yyerror(const char*);
%}
%option noyywrap
%option yylineno

/*Regular expresions go here*/
alpha		[a-zA-Z]
numeric		[0-9]
hexalpha	[A-Fa-f]
alphanum	({alpha}|{numeric})
hexdigit	({numeric}|{hexalpha})
whitespace	[ \t]
newline		[\n]
anycharbutnewline [^\n]
anychar		({newline}|{anycharbutnewline})
pipe		\|

identifier	({alpha}({alphanum}|["_$"])*)|([_$%]({alphanum}|["_$"])+)
hexliteral	0[xX]{hexdigit}+U?
octliteral	0[0-7]+U?
binliteral	0[bB][01]+U?
decliteral	([1-9][0-9]*U?)|0U?
fltliteral	[0-9]+\.[0-9]+

%%
	/*Translation rules*/

	/*Compiler directives*/
\.entry	yylval.string_value = strdup(yytext); return DOTENTRY;
\.lmem	yylval.string_value = strdup(yytext); return DOTLMEM;
\.smem	yylval.string_value = strdup(yytext); return DOTSMEM;
\.reg	yylval.string_value = strdup(yytext); return DOTREG;
\.bar	yylval.string_value = strdup(yytext); return DOTBAR;

	/*constant memory directives*/
\.constseg	return DOTCONSTSEG;
const		return CONST;
d\.u32		return DDOTU32;
d\.f32		return DDOTF32;

	/*guard predicate*/
@!?\$p{numeric}+	yylval.string_value = strdup(yytext); return GUARDPRED;

	/*label*/
label{numeric}+		yylval.string_value = strdup(yytext); return LABEL;

	/*base instructions*/
cvt	yylval.string_value = strdup(yytext); return CVT;
shl	yylval.string_value = strdup(yytext); return SHL;
shr	yylval.string_value = strdup(yytext); return SHR;
add	yylval.string_value = strdup(yytext); return ADD;
addc	yylval.string_value = strdup(yytext); return ADDC;
sub	yylval.string_value = strdup(yytext); return SUB;
subr	yylval.string_value = strdup(yytext); return SUBR;
movsh	yylval.string_value = strdup(yytext); return MOVSH;
mov	yylval.string_value = strdup(yytext); return MOV;
bar\.sync	yylval.string_value = strdup(yytext); return BARSYNC;
mul	yylval.string_value = strdup(yytext); return MUL;
mad24	yylval.string_value = strdup(yytext); return MAD24;
mad24c1	yylval.string_value = strdup(yytext); return MAD24C1;
set	yylval.string_value = strdup(yytext); return SET;
mad	yylval.string_value = strdup(yytext); return MAD;
mul24	yylval.string_value = strdup(yytext); return MUL24;
set\?{numeric}+\?	yylval.string_value = strdup(yytext); return SETUNKNOWN;
rcp	yylval.string_value = strdup(yytext); return RCP;
pre\.ex2	yylval.string_value = strdup(yytext); return PREEX2;
pre\.sin	yylval.string_value = strdup(yytext); return PRESIN;
ex2	yylval.string_value = strdup(yytext); return EX2;
sin	yylval.string_value = strdup(yytext); return SIN;
cos	yylval.string_value = strdup(yytext); return COS;
lg2	yylval.string_value = strdup(yytext); return LG2;
rsqrt	yylval.string_value = strdup(yytext); return RSQRT;
mac	yylval.string_value = strdup(yytext); return MAC;
and	yylval.string_value = strdup(yytext); return AND;
andn	yylval.string_value = strdup(yytext); return ANDN;
nandn	yylval.string_value = strdup(yytext); return NANDN;
xor	yylval.string_value = strdup(yytext); return XOR;
or	yylval.string_value = strdup(yytext); return OR;
orn	yylval.string_value = strdup(yytext); return ORN;
min	yylval.string_value = strdup(yytext); return MIN;
tex	yylval.string_value = strdup(yytext); return TEX;
bra\.label	yylval.string_value = strdup(yytext); return BRA;
join\.label	yylval.string_value = strdup(yytext); return JOIN;
breakaddr.label	yylval.string_value = strdup(yytext); return BREAKADDR;
nop\.end	yylval.string_value = strdup(yytext); return NOPEND;
nop\.join	yylval.string_value = strdup(yytext); return NOPJOIN;
nop	yylval.string_value = strdup(yytext); return NOP;
call.label	yylval.string_value = strdup(yytext); return CALL;
not	yylval.string_value = strdup(yytext); return NOT;
max	yylval.string_value = strdup(yytext); return MAX;
delta	yylval.string_value = strdup(yytext); return DELTA;
break	yylval.string_value = strdup(yytext); return BREAK;
inc	yylval.string_value = strdup(yytext); return INC;
exch	yylval.string_value = strdup(yytext); return EXCH;
cas	yylval.string_value = strdup(yytext); return CAS;
norn	yylval.string_value = strdup(yytext); return NORN;
nxor	yylval.string_value = strdup(yytext); return NXOR;
sad		yylval.string_value = strdup(yytext); return SAD;
op\.13	yylval.string_value = strdup(yytext); return OP13;
op\.e5	yylval.string_value = strdup(yytext); return OPE5;
op\.e6	yylval.string_value = strdup(yytext); return OPE6;
op\.d0	yylval.string_value = strdup(yytext); return OPD0;

return	yylval.string_value = strdup(yytext); return RETURN;

	/*instruction modifiers*/
\.rp	yylval.string_value = strdup(yytext); return DOTRP;
\.rm	yylval.string_value = strdup(yytext); return DOTRM;
\.rn	yylval.string_value = strdup(yytext); return DOTRN;
\.rni	yylval.string_value = strdup(yytext); return DOTRNI;
\.rz	yylval.string_value = strdup(yytext); return DOTRZ;
\.rzi	yylval.string_value = strdup(yytext); return DOTRZI;
\.half	yylval.string_value = strdup(yytext); return DOTHALF;
\.end	yylval.string_value = strdup(yytext); return DOTEND;
\.lo	yylval.string_value = strdup(yytext); return DOTLO;
\.abs	yylval.string_value = strdup(yytext); return DOTABS;
\.join	yylval.string_value = strdup(yytext); return DOTJOIN;
\.neg	yylval.string_value = strdup(yytext); return DOTNEG;
\.sat	yylval.string_value = strdup(yytext); return DOTSAT;

	/* comparison operators */
\.eq	yylval.string_value = strdup(yytext); return DOTEQ;
\.ne	yylval.string_value = strdup(yytext); return DOTNE;
\.lt	yylval.string_value = strdup(yytext); return DOTLT;
\.le	yylval.string_value = strdup(yytext); return DOTLE;
\.gt	yylval.string_value = strdup(yytext); return DOTGT;
\.ge	yylval.string_value = strdup(yytext); return DOTGE;
\.ls	yylval.string_value = strdup(yytext); return DOTLS;
\.hi	yylval.string_value = strdup(yytext); return DOTHI;
\.hs	yylval.string_value = strdup(yytext); return DOTHS;
\.equ	yylval.string_value = strdup(yytext); return DOTEQU;
\.neu	yylval.string_value = strdup(yytext); return DOTNEU;
\.ltu	yylval.string_value = strdup(yytext); return DOTLTU;
\.leu	yylval.string_value = strdup(yytext); return DOTLEU;
\.gtu	yylval.string_value = strdup(yytext); return DOTGTU;
\.geu	yylval.string_value = strdup(yytext); return DOTGEU;
\.num	yylval.string_value = strdup(yytext); return DOTNUM;
\.nan	yylval.string_value = strdup(yytext); return DOTNAN;
\.nsf	yylval.string_value = strdup(yytext); return DOTNSF;
\.sf	yylval.string_value = strdup(yytext); return DOTSF;
\.cf	yylval.string_value = strdup(yytext); return DOTCF;

	/*operand types*/
\.s8	yylval.string_value = strdup(yytext); return DOTS8;
\.s16	yylval.string_value = strdup(yytext); return DOTS16;
\.s32	yylval.string_value = strdup(yytext); return DOTS32;
\.s64	yylval.string_value = strdup(yytext); return DOTS64;
\.u8	yylval.string_value = strdup(yytext); return DOTU8;
\.u16	yylval.string_value = strdup(yytext); return DOTU16;
\.u32	yylval.string_value = strdup(yytext); return DOTU32;
\.u64	yylval.string_value = strdup(yytext); return DOTU64;
\.b8	yylval.string_value = strdup(yytext); return DOTB8;
\.b16	yylval.string_value = strdup(yytext); return DOTB16;
\.b32	yylval.string_value = strdup(yytext); return DOTB32;
\.b64	yylval.string_value = strdup(yytext); return DOTB64;
\.b128	yylval.string_value = strdup(yytext); return DOTB128;
\.f16	yylval.string_value = strdup(yytext); return DOTF16;
\.f32	yylval.string_value = strdup(yytext); return DOTF32;
\.f64	yylval.string_value = strdup(yytext); return DOTF64;

	/*geometries*/
\.1d	yylval.string_value = strdup(yytext); return DOT1D;
\.2d	yylval.string_value = strdup(yytext); return DOT2D;
\.3d	yylval.string_value = strdup(yytext); return DOT3D;

	/*register operands*/
[\-]?\$r{numeric}+	yylval.string_value = strdup(yytext); return REGISTER;
[\-]?\$r{numeric}+\.lo	yylval.string_value = strdup(yytext); return REGISTERLO;
[\-]?\$r{numeric}+\.hi	yylval.string_value = strdup(yytext); return REGISTERHI;
[\-]?\$ofs{numeric}+	yylval.string_value = strdup(yytext); return OFFSETREGISTER;
\$p{numeric}+		yylval.string_value = strdup(yytext); return PREDREGISTER;
\$o{numeric}+		yylval.string_value = strdup(yytext); return PREDREGISTER2;

	/*tex operand*/
[\-]?\$tex{numeric}+	yylval.string_value = strdup(yytext); return TEXOP;

	/*vector operands*/
\{($r{numeric}+|_)\}	yylval.string_value = strdup(yytext); return VECTOR1D;
\{($r{numeric}+|_)\,($r{numeric}+|_)\}	yylval.string_value = strdup(yytext); return VECTOR2D;
\{($r{numeric}+|_)\,($r{numeric}+|_)\,($r{numeric}+|_)\}	yylval.string_value = strdup(yytext); return VECTOR3D;
\{($r{numeric}+|_)\,($r{numeric}+|_)\,($r{numeric}+|_)\,($r{numeric}+|_)\}	yylval.string_value = strdup(yytext); return VECTOR4D;

	/*special registers*/
"%ntid.x"	return NTIDXREGISTER;
"%ntid.y"	return NTIDYREGISTER;
"%ctaid.x"	return CTAIDXREGISTER;
"%nctaid.x"	return NCTAIDXREGISTER;
"%ctaid.y"	return CTAIDYREGISTER;
"%nctaid.y"	return NCTAIDYREGISTER;
"%clock"	return CLOCKREGISTER;

	/*memory location operands*/
[\-]?l\[{hexliteral}\]				yylval.string_value = strdup(yytext); return LMEMLOCATION;
[\-]?l\[\$r{numeric}+\]				yylval.string_value = strdup(yytext); return LMEMLOCATION;
[\-]?l\[\$ofs{numeric}+\+{hexliteral}\]		yylval.string_value = strdup(yytext); return LMEMLOCATION;
[\-]?l\[\$ofs{numeric}+\+={hexliteral}\]	yylval.string_value = strdup(yytext); return LMEMLOCATION;

[\-]?s\[{hexliteral}\]				yylval.string_value = strdup(yytext); return SMEMLOCATION;
[\-]?s\[\$r{numeric}+\]				yylval.string_value = strdup(yytext); return SMEMLOCATION;
[\-]?s\[\$ofs{numeric}+\+{hexliteral}\]		yylval.string_value = strdup(yytext); return SMEMLOCATION;
[\-]?s\[\$ofs{numeric}+\+={hexliteral}\]	yylval.string_value = strdup(yytext); return SMEMLOCATION;

[\-]?g\[{hexliteral}\]				yylval.string_value = strdup(yytext); return GMEMLOCATION;
[\-]?g\[\$r{numeric}+\]				yylval.string_value = strdup(yytext); return GMEMLOCATION;
[\-]?g\[\$ofs{numeric}+\+{hexliteral}\]		yylval.string_value = strdup(yytext); return GMEMLOCATION;
[\-]?g\[\$ofs{numeric}+\+={hexliteral}\]	yylval.string_value = strdup(yytext); return GMEMLOCATION;


[\-]?c{numeric}+\[{hexliteral}\]			yylval.string_value = strdup(yytext); return CMEMLOCATION;
[\-]?c{numeric}+\[\$r{numeric}+\]			yylval.string_value = strdup(yytext); return CMEMLOCATION;
[\-]?c{numeric}+\[\$ofs{numeric}+\+{hexliteral}\]	yylval.string_value = strdup(yytext); return CMEMLOCATION;
[\-]?c{numeric}+\[\$ofs{numeric}+\+={hexliteral}\]	yylval.string_value = strdup(yytext); return CMEMLOCATION;


	/*identifier + literals*/
{identifier}	yylval.string_value = strdup(yytext); return IDENTIFER;
{hexliteral}	yylval.string_value = strdup(yytext); return HEXLITERAL; //change to int later?
{octliteral}	yylval.string_value = strdup(yytext); return OCTLITERAL; //change to int later?
{binliteral}	yylval.string_value = strdup(yytext); return BINLITERAL; //change to int later?
{decliteral}	yylval.string_value = strdup(yytext); return DECLITERAL; //change to int later?
{fltliteral}	yylval.string_value = strdup(yytext); return FLTLITERAL;

\:	return COLON;

{pipe}	return PIPE;

	/*braces*/
\{	yylval.string_value = strdup(yytext); return LEFTBRACE;
\}	yylval.string_value = strdup(yytext); return RIGHTBRACE;

	/* pound sign*/
"#"	yylval.string_value = strdup(yytext); return POUND;

	/* comma*/
","	yylval.string_value = strdup(yytext); return COMMA;

	/* underscore sign*/
"_"	yylval.string_value = strdup(yytext); return UNDERSCORE;

	/*Need to create to newline tokens*/
{newline}+	return NEWLINE;


	/*discard whitespace*/
{whitespace}+
	/*discard comments*/
"//"{anycharbutnewline}*


	/*reached end of file*/
<<EOF>>	return 0;

	/*None of the rules matched. Throw error*/
.			yyerror("Invalid token");

%%

void yyerror(const char* message)
{
	printf("   "); printf(message); printf(" near \""); printf(yytext); printf("\"");
	printf(" on line "); 
	char line[5]; sprintf(line, "%i", yylineno); printf(line);
	printf("\n");
}
