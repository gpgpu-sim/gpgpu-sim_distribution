%{

#include "booksim.hpp"
#include <string>
#include "config_utils.hpp"

int  configlex(void);
void configerror(string msg);

%}

%union {
  char         *name;
  unsigned int num;
  double       fnum;
}

%token <name> STR
%token <num>  NUM
%token <fnum> FNUM

%%

commands : commands command
         | command
;

command : STR '=' STR ';'   { Configuration::GetTheConfig()->Assign( $1, $3 ); free( $1 ); free( $3 ); }
        | STR '=' NUM ';'   { Configuration::GetTheConfig()->Assign( $1, $3 ); free( $1 ); }
        | STR '=' FNUM ';'  { Configuration::GetTheConfig()->Assign( $1, $3 ); free( $1 ); }
;

%%
