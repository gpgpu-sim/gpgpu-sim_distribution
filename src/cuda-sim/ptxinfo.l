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


%option noyywrap
%option yylineno
%option prefix="ptxinfo_"
%{
#include "ptxinfo.tab.h"
#include <string.h>

#define LINEBUF_SIZE 1024
char ptxinfo_linebuf[LINEBUF_SIZE];
unsigned ptxinfo_col = 0;
#define TC if( (ptxinfo_lineno == 1) && ((ptxinfo_col + strlen(ptxinfo_text)) < LINEBUF_SIZE) ) { \
		strncpy(ptxinfo_linebuf+ptxinfo_col,ptxinfo_text,strlen(ptxinfo_text)); \
	   } \
	   ptxinfo_col+=strlen(ptxinfo_text); 
%}

%%
"ptxas info    : Function properties for"[^\n]*\n[^\n]*
"warning"[^\n]* TC; return WARNING;
"ptxas"	TC; 	return HEADER;
"info"	TC; 	return INFO;
"Compiling entry function"	TC; return FUNC;
"Used"		TC; return USED;
"registers"	TC; return REGS;
"bytes"		TC; return BYTES;
"lmem"		TC; return LMEM;
"smem"		TC; return SMEM;
"cmem"		TC; return CMEM;
"line"		TC; return LINE;
"for"		TC; return FOR;

[_A-Za-z$%][_0-9A-Za-z$]*  TC; ptxinfo_lval.string_value = strdup(yytext); return IDENTIFIER;
[-]{0,1}[0-9]+	 TC; ptxinfo_lval.int_value =  atoi(yytext); return INT_OPERAND;

"+"	TC; return PLUS;
","     TC; return COMMA;
"["     TC; return LEFT_SQUARE_BRACKET;
"]"	TC; return RIGHT_SQUARE_BRACKET;
":"	TC; return COLON;
";"	TC; return SEMICOLON;
"'"	TC; return QUOTE;
" " TC;
"\t" TC;

\n.*  ptxinfo_col=0; strncpy(ptxinfo_linebuf, yytext + 1, 1024); yyless( 1 );

%%

extern int g_ptxinfo_error_detected;
extern const char *g_filename;
extern const char *g_ptxinfo_filename;

int ptxinfo_error( const char *s )
{
	int i;
	g_ptxinfo_error_detected = 1;
	fflush(stdout);
	printf("GPGPU-Sim: ERROR while parsing output of ptxas (used to capture resource usage information)\n");
	if( s != NULL )
		printf("GPGPU-Sim:     %s (%s:%u) Syntax error:\n\n", g_filename, g_ptxinfo_filename, ptxinfo_lineno );
	printf("   %s\n", ptxinfo_linebuf );
	printf("   ");
	for( i=0; i < ptxinfo_col-1; i++ ) {
		if( ptxinfo_linebuf[i] == '\t' ) printf("\t");
		else printf(" ");
	}
			
	printf("^\n\n");
	fflush(stdout);
	exit(43);
	return 0;
}
