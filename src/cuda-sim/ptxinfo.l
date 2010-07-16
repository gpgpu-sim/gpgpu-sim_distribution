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
%option prefix="ptxinfo_"
%{
#include "ptxinfo.tab.h"
#include <string.h>

char ptxinfo_linebuf[1024];
unsigned ptxinfo_col = 0;
#define TC ptxinfo_col+=strlen(ptxinfo_text); 
%}

%%

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
	if( s != NULL )
		printf("ptxas -v %s (%s:%u) Syntax error:\n\n", g_filename, g_ptxinfo_filename, ptxinfo_lineno );
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
