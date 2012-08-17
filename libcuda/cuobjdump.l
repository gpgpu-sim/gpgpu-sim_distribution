/*
// Copyright (c) 2011-2012, Andrew Boktor
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
	/*Lex file for output of cuobjdump*/
%{
#include <stdio.h>
#include <string.h>
#include "cuobjdump_parser.h"

#define YY_NEVER_INTERACTIVE 1
#define YY_NO_INPUT
#define YY_NO_TOP_STATE

#define YYDEBUG 1

#define yylval cuobjdump_lval

void cuobjdump_error(const char*);
%}

%option stack
%option noyywrap
%option yylineno
%option nounput


%s ptxcode
%s sasscode
%s elfcode
%s header
%s endheader

alpha		[a-zA-Z]
numeric		[0-9]
whitespace	[ \t]
newline		[\n]
notnewline	[^\n]
newlines	{newline}+
%%


"ptxasOptions"{notnewline}*{newline}
[1-9]{numeric}* yylval.string_value = strdup(yytext); return DECIMAL;


"Fatbin ptx code:"{newline}	{
	yy_push_state(ptxcode);
	yy_push_state(header);
	yylval.string_value = strdup(yytext);
	return PTXHEADER;
}
"Fatbin elf code:"{newline}	{
	yy_push_state(elfcode);
	yy_push_state(header);
	yylval.string_value = strdup(yytext);
	return ELFHEADER;
}
	/*PTX code tokens*/
<ptxcode>{notnewline}*{newline}	yylval.string_value = strdup(yytext); return PTXLINE;

	/*ELF code tokens*/
<elfcode>{whitespace}*"code for sm_"{numeric}+{newline}	{
	BEGIN(sasscode); 
	yylval.string_value = strdup(yytext); 
	return SASSLINE;
}
<elfcode>{notnewline}*{newline} yylval.string_value = strdup(yytext); return ELFLINE;

	/*SASS code tokens*/
<sasscode>{notnewline}*{newline} yylval.string_value = strdup(yytext); return SASSLINE;

	/*Header tokens*/
<header>[[:alnum:]_]+ yylval.string_value = strdup(yytext); return IDENTIFIER;
<header>"================"	return H_SEPARATOR;
<header>"arch = "		return H_ARCH;
<header>"code version = "	return H_CODEVERSION;
<header>\[{numeric},{numeric}\]	return CODEVERSION;
<header>"producer = "		return H_PRODUCER;
<header>"host = "			return H_HOST;
<header>"compile_size = "	return H_COMPILESIZE;
<header>"identifier = "		BEGIN(endheader); return H_IDENTIFIER;
<header>"has debug info"{newline}

	/* Looking for the identifier (filename) then the header is done */
	/* <endheader>[[:alnum:]_\./]+	yylval.string_value = strdup(yytext); yy_pop_state(); return FILENAME; */
<endheader>{notnewline}+	yylval.string_value = strdup(yytext); yy_pop_state(); return FILENAME;


{newline}	return NEWLINE;

	/*Reached end of file*/
<<EOF>> BEGIN(INITIAL);return 0;

	/*No other rule matched. Throw an error*/
.	cuobjdump_error("Invalid token");

%%
void cuobjdump_error(const char* message)
{
	printf("   "); printf(message); printf(" near \""); printf(yytext); printf("\"");
	printf(" on line "); 
	char line[5]; sprintf(line, "%i", yylineno); printf(line);
	printf("\n");
}
