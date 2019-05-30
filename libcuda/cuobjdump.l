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
#include "cuobjdump.h"
#include "cuobjdump_parser.h"

#define YY_NEVER_INTERACTIVE 1
#define YY_NO_INPUT
#define YY_NO_TOP_STATE

#define YYDEBUG 1

void cuobjdump_error(yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList, const char* msg);
#define YY_DECL int cuobjdump_lex \
	       (YYSTYPE * yylval_param , yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList)
%}

%option stack
%option noyywrap
%option yylineno
%option nounput

%option bison-bridge
%option reentrant

%s ptxcode
%s sasscode
%s elfcode
%s elfheader
%s ptxheader
%s endheader
%x identifier
%x conidentifier
%s endidentifier

alpha		[a-zA-Z]
numeric		[0-9]
whitespace	[ \t]
newline		[\n]
notnewline	[^\n]
newlines	{newline}+
%%


"ptxasOptions"{notnewline}*{newline}
[1-9]{numeric}* yylval->string_value = strdup(yytext); return DECIMAL;

"has debug info"{newline}


"Fatbin ptx code:"{newline}	{
	yy_push_state(ptxcode, yyscanner);
	yy_push_state(identifier, yyscanner);
	yy_push_state(ptxheader, yyscanner);
	yylval->string_value = strdup(yytext);
	return PTXHEADER;
}
"Fatbin elf code:"{newline}	{
	yy_push_state(elfcode, yyscanner);
	yy_push_state(identifier, yyscanner);
	yy_push_state(elfheader, yyscanner);
	yylval->string_value = strdup(yytext);
	return ELFHEADER;
}
	/*PTX code tokens*/
<ptxcode>{notnewline}*{newline}	yylval->string_value = strdup(yytext); return PTXLINE;

	/*ELF code tokens*/
<elfcode>{whitespace}*"code for sm_"{numeric}+{newline}	{
	BEGIN(sasscode); 
	yylval->string_value = strdup(yytext);
	return SASSLINE;
}
<elfcode>{notnewline}*{newline} yylval->string_value = strdup(yytext); return ELFLINE;

	/*SASS code tokens*/
<sasscode>{notnewline}*{newline} yylval->string_value = strdup(yytext); return SASSLINE;

<identifier>{newline}"compressed"{newline}	BEGIN(conidentifier); return H_COMPRESSED;
<identifier>{newline}"identifier = "	BEGIN(endidentifier); return H_IDENTIFIER;
<identifier>{newline}{newline} yy_pop_state(yyscanner);


<conidentifier>"identifier = "	BEGIN(endidentifier); return H_IDENTIFIER;
<conidentifier>{newline} yy_pop_state(yyscanner);


<endidentifier>{notnewline}+	yylval->string_value = strdup(yytext); yy_pop_state(yyscanner); return FILENAME;



	/*Header tokens*/
<elfheader>[[:alnum:]_]+ yylval->string_value = strdup(yytext); return IDENTIFIER;
<elfheader>"================"	return H_SEPARATOR;
<elfheader>"arch = "		return H_ARCH;
<elfheader>"code version = "	return H_CODEVERSION;
<elfheader>\[{numeric},{numeric}\]	return CODEVERSION;
<elfheader>"producer = "		return H_PRODUCER;
<elfheader>"<unknown>"			return H_UNKNOWN;
<elfheader>"host = "			return H_HOST;
<elfheader>"compile_size = "	BEGIN(endheader); return H_COMPILESIZE;


	/*Header tokens*/
<ptxheader>[[:alnum:]_]+ yylval->string_value = strdup(yytext); return IDENTIFIER;
<ptxheader>"================"	return H_SEPARATOR;
<ptxheader>"arch = "		return H_ARCH;
<ptxheader>"code version = "	return H_CODEVERSION;
<ptxheader>\[{numeric},{numeric}\]	return CODEVERSION;
<ptxheader>"producer = "		return H_PRODUCER;
<ptxheader>"host = "			return H_HOST;
<ptxheader>"compile_size = "	BEGIN(endheader); return H_COMPILESIZE;






	/* Looking for the identifier (filename) then the header is done */
<endheader>{notnewline}+	yylval->string_value = strdup(yytext); yy_pop_state(yyscanner); return IDENTIFIER;



{newline}	return NEWLINE;

	/*Reached end of file*/
<<EOF>> BEGIN(INITIAL);return 0;

	/*No other rule matched. Throw an error*/
.	cuobjdump_error(yyscanner, parser, cuobjdumpSectionList, "Invalid token");

%%
void cuobjdump_error(yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList, const char* msg)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	printf("   %s near \"%s\"",msg, yytext);
	printf(" on line %i\n",yylineno);
}
