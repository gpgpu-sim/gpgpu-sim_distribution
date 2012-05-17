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

%option noyywrap
%option yylineno
%option prefix="header_"
%option nounput
%{
#include <iostream>
#include "header_parser.hh"
#include <string.h>

/* not sure if this is needed*/
#define	YY_NEVER_INTERACTIVE	1

void header_error(const char*);
extern void output(const char * text);
%}


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
anycharbutrightbrace [^"}"]

identifier	({alpha}({alphanum}|["_$"])*)|([_$%]({alphanum}|["_$"])+)
hexliteral	0[xX]{hexdigit}+U?
octliteral	0[0-7]+U?
binliteral	0[bB][01]+U?
decliteral	([1-9][0-9]*U?)|0U?

%%
	/*Translation rules*/

	/*Compiler directives*/
\.version	return DOTVERSION;
\.target	return DOTTARGET;
\.entry	return DOTENTRY;

	/*State spaces*/
\.param	return DOTPARAM;

	/*operand types*/
\.u64	return DOTU64;
\.u32	return DOTU32;
\.u16	return DOTU16;
\.b32	return DOTB32;
\.f32	return DOTF32;


	/*identifier + literals*/
{identifier}	header_lval.string_value = strdup(yytext); return IDENTIFER;
{decliteral}	header_lval.string_value = strdup(yytext); return DECLITERAL; //change to int later?

	/*braces and parenthesis*/
\(	return LEFTPAREN;
\)	return RIGHTPAREN;

	/*discard newline*/
{newline}+


	/*discard commas and periods*/
\,
\.
	/*discard whitespace*/
{whitespace}+
	/*discard comments*/
"//"{anycharbutnewline}*{newline}

	/*discard other header stuff*/
\.file{anycharbutnewline}*{newline}
\.extern{anycharbutnewline}*{newline}

	/*discard kernel body*/
\{{anycharbutrightbrace}*\}

	/*reached end of file*/
<<EOF>>	return 0;

	/*None of the rules matched. Throw error*/
.			header_error("Invalid token");

%%

void header_error(const char* message)
{
	output("   "); output(message); output(" near \""); output(header_text); output("\"");
	output(" on line "); 
	char line[5]; sprintf(line, "%i", header_lineno); output(line);
	output("\n");
}
