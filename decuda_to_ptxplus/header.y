%{
#include <iostream>
#include "decudaInstList.h"

int yylex(void);
void yyerror(const char*);

extern decudaInstList *g_headerList;
extern void output(const char * text);
%}


%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token DOTVERSION
%token DOTTARGET
%token DOTENTRY

%token DOTPARAM

%token DOTU64
%token DOTU32
%token DOTU16
%token DOTB32
%token DOTF32

%token <string_value> IDENTIFER

	/*change these 4 to int later?*/
%token <string_value> DECLITERAL

%token LEFTPAREN
%token RIGHTPAREN


%%

	/*translation rules*/
program		: statementList			{ output("No parsing errors\n");  }
		;

statementList	: statementList statement	{ output("\n"); }
		| statement			{ output("\n"); }
		;

statement	: compilerDirective literal literal	{}
		| compilerDirective identifierList		{}
		| compilerDirective identifierList LEFTPAREN parameterList RIGHTPAREN	{}
		;

compilerDirective	: DOTVERSION	{ output(".version"); decudaInst *instEntry = new decudaInst(); instEntry->setBase(".version"); g_headerList->add(instEntry);}
			| DOTTARGET	{ output(".target"); decudaInst *instEntry = new decudaInst(); instEntry->setBase(".target"); g_headerList->add(instEntry);}
			| DOTENTRY	{ output(".entry"); decudaInst *instEntry = new decudaInst(); instEntry->setBase(".entry"); g_headerList->add(instEntry);}
			;

identifierList	: identifierList IDENTIFER	{ output(" "); output($2); g_headerList->getListEnd().addOperand($2); }
		| IDENTIFER			{ output(" "); output($1); g_headerList->getListEnd().addOperand($1); }
		;

parameterList	: parameterList parameter
		| parameter
		;

parameter	: stateSpace opTypes IDENTIFER	{ output(" "); output($3); g_headerList->getListEnd().addOperand($3);}
		;

stateSpace	: DOTPARAM	{ output("\n.param"); decudaInst *instEntry = new decudaInst(); instEntry->setBase(".param"); g_headerList->add(instEntry); }
		;

opTypes		: DOTU64		{ output(".u64"); g_headerList->getListEnd().addTypeModifier(".u64");}
		| DOTU32		{ output(".u32"); g_headerList->getListEnd().addTypeModifier(".u32");}
		| DOTU16		{ output(".u16"); g_headerList->getListEnd().addTypeModifier(".u16");}
		| DOTB32		{ output(".b32"); g_headerList->getListEnd().addTypeModifier(".b32");}
		| DOTF32		{ output(".f32"); g_headerList->getListEnd().addTypeModifier(".f32");}
		;

literal		: DECLITERAL	{ output(" "); output($1); g_headerList->getListEnd().addOperand($1); }
		;

%%

/*support c++ functions go here*/
