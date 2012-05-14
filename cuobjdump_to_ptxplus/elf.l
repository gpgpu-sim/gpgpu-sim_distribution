	/*Lex file for elf files output by cuobjdump*/
%{
#include <stdio.h>
#include <string.h>
#include "elf_parser.hh"

#define yylval elf_lval

void elf_error(const char*);
%}

%option stack
%option noyywrap
%option yylineno

%x cmem
%x symtab

whitespace	[ \t]
newline		[\n]
notnewline	[^\n]
newlines	{newline}+
ident		[[:alnum:]_\.]
hexdigit	[[:digit:]abcdef]
%%

<cmem,INITIAL>\.nv\.constant1\.{notnewline}*{newline}	{
	BEGIN(cmem);
	yylval.string_value = strdup(yytext); 
	return C1BEGIN;
}

<cmem,INITIAL>\.nv\.local{notnewline}*{newline}	{
	BEGIN(INITIAL);
	yylval.string_value = strdup(yytext); 
	return LOCALMEM;
}

<cmem,INITIAL>\.nv\.constant0{notnewline}*{newline}	{
	BEGIN(cmem);
	return C0BEGIN;
}


\.section" "\.symtab\tSYMTAB{newline}	{
	BEGIN(symtab);
	return STBEGIN;
}

\.section{notnewline}*{newline}	{
	//These are the lines before the symtab
}

<cmem,INITIAL>\.{notnewline}*{newline}	{
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

<symtab>{whitespace}+ {}

<symtab>{newline} {}

<symtab>{hexdigit}+ {
	yylval.string_value = strdup(yytext);
	return NUMBER;
}

<symtab>{ident}+ {
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
	printf(message);
	printf(" near \"");
	printf(yytext);
	printf("\"");
	printf(" on line "); 
	char line[5];
	sprintf(line, "%i", yylineno);
	printf(line);
	printf("\n");
	abort();
}
