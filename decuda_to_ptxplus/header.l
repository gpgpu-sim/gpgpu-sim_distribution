%option noyywrap
%option yylineno
%option prefix="header_"
%{
#include <iostream>
#include "headerParser.hpp"
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
