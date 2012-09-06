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
	/*Lex file for elf files output by cuobjdump*/
%{
#include <stdio.h>
#include <string.h>
#include "elf_parser.hh"

#define yylval elf_lval

void elf_error(const char*);
%}

%option noyywrap
%option yylineno
%option nounput

%x cmem
%x symtab
%x cmem14

whitespace	[ \t]
newline		[\n]
notnewline	[^\n]
newlines	{newline}+
ident		[[:alnum:]_\.]
hexdigit	[[:digit:]abcdef]
%%

<cmem,cmem14,INITIAL>\.nv\.constant1\.{notnewline}*{newline}	{
	BEGIN(cmem);
	yylval.string_value = strdup(yytext); 
	return C1BEGIN;
}

<cmem,cmem14,INITIAL>\.nv\.local{notnewline}*{newline}	{
	BEGIN(INITIAL);
	yylval.string_value = strdup(yytext); 
	return LOCALMEM;
}

<cmem,cmem14,INITIAL>\.nv\.constant0{notnewline}*{newline}	{
	BEGIN(cmem);
	return C0BEGIN;
}

<cmem,cmem14,INITIAL>\.section" "\.rel\.nv\.constant14{notnewline}*{newline}	{
	BEGIN(cmem14);
	return C14BEGIN;
}

\.section" "\.symtab\tSYMTAB{newline}	{
	BEGIN(symtab);
	return STBEGIN;
}

\.section{notnewline}*{newline}	{
	//These are the lines before the symtab
}

<cmem,cmem14,INITIAL>\.{notnewline}*{newline}	{
	BEGIN(INITIAL);
}

{notnewline}*{newline} {
	//printf("Discarding line %s", yytext);
}

<cmem>"  "{newline}?	{
	//printf("#SPACE#");
	return SPACE2;
}

<cmem>0x[0-9a-fA-F]{8}	{
	//printf("CMEM VAL: %s \n", yytext);
	yylval.string_value = strdup(yytext); 
	return CMEMVAL;
}

<symtab>{whitespace}*[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+{newline} {
	return STHEADER;
}

<symtab,cmem14>{whitespace}+ {}

<symtab,cmem14>{newline} {}

<symtab,cmem14>{hexdigit}+ {
	yylval.string_value = strdup(yytext);
	return NUMBER;
}

<symtab,cmem14>{ident}+ {
	yylval.string_value = strdup(yytext);
	return IDENTIFIER;
}

<symtab>{newline}{whitespace}*{newline}{whitespace}* {
	BEGIN(INITIAL);
}
	/*Reached end of file*/
<<EOF>> return 0;

	/*No other rule matched. Throw an error*/
.	elf_error("Invalid token");

%%
void elf_error(const char* message)
{
	printf("   ");
	printf("%s",message);
	printf(" near \"");
	printf("%s",yytext);
	printf("\"");
	printf(" on line "); 
	char line[5];
	sprintf(line, "%i", yylineno);
	printf("%s",line);
	printf("\n");
	abort();
}
