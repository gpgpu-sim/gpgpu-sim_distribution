
%{

#include "booksim.hpp"
#include <stdlib.h>
#include <string>
#include <cstring>

#include "config_tab.hpp"
#include "config_utils.hpp"

void configerror(string msg);
extern "C" int configwrap() { return 1; }

extern int config_input(char *, int);
#undef YY_INPUT
#define YY_INPUT(b, r, ms) (r = config_input(b, ms))

int configlineno = 1;

%}

%option nounput

%%

   /* Ignore comments and all spaces */

\/\/[^\n]* ;
[ \t\r]*   ;

\n         { configlineno++; }

   /* Commands */

[A-Za-z_][A-Za-z0-9_]* { configlval.name = strdup( yytext ); return STR; }

[0-9]+     { configlval.num = strtoul( yytext, 0, 10 ); return NUM; }

[0-9]+\.[0-9]+ { configlval.fnum = strtod( yytext, 0 ); return FNUM; }

.          { return yytext[0]; }

%%

void configerror( string msg )
{
  Configuration::GetTheConfig( )->ParseError( msg, configlineno );
}


