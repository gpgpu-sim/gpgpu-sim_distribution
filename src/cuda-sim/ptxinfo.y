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
  int    int_value;
  char * string_value;
}

%token <int_value> INT_OPERAND
%token HEADER
%token INFO
%token FUNC
%token USED
%token REGS
%token BYTES
%token LMEM
%token SMEM
%token CMEM
%token <string_value> IDENTIFIER
%token PLUS
%token COMMA
%token LEFT_SQUARE_BRACKET
%token RIGHT_SQUARE_BRACKET
%token COLON
%token SEMICOLON
%token QUOTE
%token LINE
%token WARNING

%{
	#include <stdlib.h>
	#include <string.h>

	static unsigned g_declared;
	static unsigned g_system;
	int ptxinfo_lex(void);
	void ptxinfo_addinfo();
	void ptxinfo_function(const char *fname );
	void ptxinfo_regs( unsigned nregs );
	void ptxinfo_lmem( unsigned declared, unsigned system );
	void ptxinfo_smem( unsigned declared, unsigned system );
	void ptxinfo_cmem( unsigned nbytes, unsigned bank );
	int ptxinfo_error(const char*);
%}

%%

input:	/* empty */
	| input line
	;

line: 	HEADER INFO COLON line_info
	| HEADER IDENTIFIER COMMA LINE INT_OPERAND SEMICOLON WARNING
	;

line_info: function_name
	| function_info { ptxinfo_addinfo(); }
	;

function_name: FUNC QUOTE IDENTIFIER QUOTE { ptxinfo_function($3); }

function_info: info
	| function_info COMMA info
	;

info: 	  USED INT_OPERAND REGS { ptxinfo_regs($2); }
	| tuple LMEM { ptxinfo_lmem(g_declared,g_system); }
	| tuple SMEM { ptxinfo_smem(g_declared,g_system); }
	| INT_OPERAND BYTES CMEM LEFT_SQUARE_BRACKET INT_OPERAND RIGHT_SQUARE_BRACKET { ptxinfo_cmem($1,$5); }
	| INT_OPERAND BYTES LMEM { ptxinfo_lmem($1,0); }
	| INT_OPERAND BYTES SMEM { ptxinfo_smem($1,0); }
	| INT_OPERAND BYTES CMEM { ptxinfo_cmem($1,0); }
	| INT_OPERAND REGS { ptxinfo_regs($1); }
	;

tuple: INT_OPERAND PLUS INT_OPERAND BYTES { g_declared=$1; g_system=$3; }

%%


