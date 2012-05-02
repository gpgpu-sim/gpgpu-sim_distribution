	/*Lex file for output of cuobjdump*/
%{
#include <stdio.h>
#include <string.h>
#include "cuobjdump_parser.h"

#define YY_NEVER_INTERACTIVE 1

#define YYDEBUG 1

#define yylval cuobjdump_lval

void cuobjdump_error(const char*);
%}

%option stack
%option noyywrap
%option yylineno

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
<endheader>[[:alnum:]_\.]+{newline}	yylval.string_value = strdup(yytext); yy_pop_state(); return FILENAME;


{newline}	return NEWLINE;

	/*Reached end of file*/
<<EOF>> return 0;

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
