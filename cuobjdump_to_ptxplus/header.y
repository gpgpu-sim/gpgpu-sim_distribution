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


%{
#include <iostream>
#include "cuobjdumpInstList.h"

int yylex(void);
void yyerror(const char*);

extern cuobjdumpInstList *g_headerList;
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

compilerDirective	: DOTVERSION	{ output(".version"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".version"); g_headerList->add(instEntry);}
			| DOTTARGET	{ output(".target"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".target"); g_headerList->add(instEntry);}
			| DOTENTRY	{ output(".entry"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".entry"); g_headerList->add(instEntry);}
			;

identifierList	: identifierList IDENTIFER	{ output(" "); output($2); g_headerList->getListEnd().addOperand($2); }
		| IDENTIFER			{ output(" "); output($1); g_headerList->getListEnd().addOperand($1); }
		;

parameterList	: parameterList parameter
		| parameter
		;

parameter	: stateSpace opTypes IDENTIFER	{ output(" "); output($3); g_headerList->getListEnd().addOperand($3);}
		;

stateSpace	: DOTPARAM	{ output("\n.param"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".param"); g_headerList->add(instEntry); }
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
