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
#include "decudaInstList.h"

int yylex(void);
void yyerror(const char*);

extern decudaInstList *g_instList;

decudaInst *instEntry;
%}


%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token <string_value> DOTENTRY DOTLMEM DOTSMEM DOTREG DOTBAR

%token <string_value> GUARDPRED LABEL

%token <string_value> CVT SHL SHR ADD SUB SUBR MOVSH MOV BARSYNC MUL MAD24 MAD24C1 SET RETURN MAD MUL24 SETUNKNOWN RCP PREEX2 PRESIN EX2 SIN COS LG2 RSQRT MAC BRA NOPEND JOIN NOPJOIN NOP AND ANDN XOR MIN BREAKADDR TEX CALL NOT OR MAX DELTA BREAK INC EXCH CAS NORN ADDC ORN NANDN NXOR SAD OPE5 OPE6 OP13 OPD0

%token <string_value> DOTEQ DOTNE DOTLT DOTLE DOTGT DOTGE DOTLO DOTLS DOTHI DOTHS DOTEQU DOTNEU DOTLTU DOTLEU DOTGTU DOTGEU DOTNUM DOTNAN DOTNSF DOTSF DOTCF

%token <string_value> DOTRP DOTRM DOTRN DOTRNI DOTRZ DOTRZI DOTHALF DOTEND DOTABS DOTJOIN DOTNEG DOTSAT

%token <string_value> DOT1D DOT2D DOT3D

%token <string_value> DOTS8 DOTS16 DOTS32 DOTS64
%token <string_value> DOTU8 DOTU16 DOTU32 DOTU64
%token <string_value> DOTB8 DOTB16 DOTB32 DOTB64 DOTB128
%token <string_value> DOTF16 DOTF32 DOTF64

%token <string_value> REGISTER REGISTERLO REGISTERHI OFFSETREGISTER PREDREGISTER PREDREGISTER2
%token <string_value> VECTOR1D VECTOR2D VECTOR3D VECTOR4D

%token <string_value> TEXOP

%token <string_value> NTIDXREGISTER NTIDYREGISTER CTAIDXREGISTER NCTAIDXREGISTER CTAIDYREGISTER NCTAIDYREGISTER CLOCKREGISTER

%token <string_value> LMEMLOCATION SMEMLOCATION GMEMLOCATION CMEMLOCATION

%token <string_value> IDENTIFER

%token <string_value> DOTCONSTSEG CONST DDOTU32 DDOTF32;

	/*change these 4 to int later?*/
%token <string_value> HEXLITERAL
%token <string_value> OCTLITERAL
%token <string_value> BINLITERAL
%token <string_value> DECLITERAL
%token <string_value> FLTLITERAL

%token <string_value> COLON
%token <string_value> LEFTBRACE
%token <string_value> RIGHTBRACE
%token NEWLINE
%token PIPE
%token <string_value> COMMA
%token <string_value> POUND
%token <string_value> UNDERSCORE


	/* set types for rules */
%type<string_value> registerlocation
%type<string_value> predicate predicateModifier
%type<string_value> simpleInstructions
%type<string_value> comparisonOp
%type<string_value> constMemoryTypes
%type<string_value> compilerDirective

%%

	/*translation rules*/
program		: statementList			{ printf("No parsing errors\n");  }
		;

statementList	: statementList statement newlines	{ printf("\n"); }
		| statement newlines			{ printf("\n"); }
		| newlines				{}
		;

newlines	: NEWLINE
		| newlines NEWLINE;

statement	: LEFTBRACE			{ printf("{"); }
		| RIGHTBRACE			{ printf("}"); }
		| compilerDirectiveStatement	{}
		| constMemoryDirectiveStatement	{}
		| { instEntry = new decudaInst(); }	assemblyInstruction
		;

compilerDirectiveStatement	: DOTENTRY IDENTIFER		{ printf("%s %s", $1, $2);
								  g_instList->addEntry($2);
								  instEntry = new decudaInst(); instEntry->setBase($1); 					  			  g_instList->add(instEntry);
								  g_instList->getListEnd().addOperand($2);}

				| compilerDirective 		{ }
				;

compilerDirective	: DOTLMEM DECLITERAL	{ printf(".lmem "); printf($2); g_instList->setLastEntryLMemSize(atoi($2));}
			| DOTSMEM DECLITERAL	{ printf(".smem "); printf($2); }
			| DOTREG DECLITERAL	{ printf(".reg "); printf($2); }
			| DOTBAR DECLITERAL	{ printf(".bar "); printf($2); }
			;

assemblyInstruction	: baseInstruction modifierList operandList	{ }
			| otherInstruction	{ }
			| predicate assemblyInstruction	{ }
			| instructionLabel assemblyInstruction	{ }
			;

instructionLabel	: LABEL COLON	{ printf($1); printf(": "); instEntry->setLabel($1); };

predicate		: GUARDPRED	{ printf($1); printf(" "); instEntry->setPredicate($1); }
			| GUARDPRED predicateModifier	{ printf($1); printf($2); printf(" "); instEntry->setPredicate($1); instEntry->addPredicateModifier($2);}
			;

predicateModifier	: DOTNE 	{ }
			| DOTNEU	{ }
			| DOTEQ		{ }
			| DOTEQU	{ }
			| DOTGTU	{ }
			| DOTLTU	{ }
			| DOTLE		{ }
			| DOTGE		{ }
			| DOTNSF	{ }
			| DOTSF		{ }
			| DOTCF		{ }
			;

baseInstruction		: simpleInstructions	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry);}
			| SET comparisonOp { printf($1); instEntry->setBase($1); g_instList->add(instEntry); printf($2); instEntry->addBaseModifier($2); }
			| SET DOTJOIN comparisonOp { printf($1); instEntry->setBase($1); g_instList->add(instEntry); printf($2); instEntry->addBaseModifier($2); instEntry->addBaseModifier($3); }
			;

simpleInstructions	: CVT | SHL | SHR | ADD | SUB | SUBR | MOVSH | MOV | BARSYNC | MUL | MAD24 | MAD24C1 | MAD | MUL24 | SETUNKNOWN | RCP | PREEX2 | PRESIN | SIN | COS | EX2 | LG2 | RSQRT | MAC | AND | ANDN | XOR | MIN | TEX | NOT | OR | MAX | DELTA | INC | EXCH | CAS | NORN | ADDC | ORN | NANDN | NXOR | SAD | OPD0
			;

otherInstruction	: RETURN	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry);}
			| BRA LABEL 	{ printf($1); printf(" "); printf($2); instEntry->setBase($1); g_instList->add(instEntry); instEntry->addOperand($2);  }
			| JOIN LABEL 	{ printf($1); printf(" "); printf($2); instEntry->setBase($1); g_instList->add(instEntry); instEntry->addOperand($2);  }
			| BREAKADDR LABEL 	{ printf($1); printf(" "); printf($2); instEntry->setBase($1); g_instList->add(instEntry); instEntry->addOperand($2);  }
			| CALL LABEL 	{ printf($1); printf(" "); printf($2); instEntry->setBase($1); g_instList->add(instEntry); instEntry->addOperand($2);  }
			| NOPEND	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); }
			| NOPJOIN	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); }
			| NOP		{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); }
			| BREAK		{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); }
			| OP13 PREDREGISTER		{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); g_instList->addPredicate($2);}
			| OPE6 { printf($1); instEntry->setBase($1); g_instList->add(instEntry); }
			| OPE5 { printf($1); instEntry->setBase($1); g_instList->add(instEntry); }

comparisonOp	: DOTEQ			{ }
		| DOTNE			{ }
		| DOTLE			{ }
		| DOTLT			{ }
		| DOTGE			{ }
		| DOTGT			{ }
		| DOTLO			{ }
		| DOTLS			{ }
		| DOTHI			{ }
		| DOTHS			{ }
		| DOTEQU		{ }
		| DOTNEU		{ }
		| DOTLTU		{ }
		| DOTLEU		{ }
		| DOTGTU		{ }
		| DOTGEU		{ }
		| DOTNUM		{ }
		| DOTNAN		{ }
		;

modifierList	: modifierList modifier	{}
		| modifier		{}
		;

modifier	: opTypes		{}
		| geometries		{}
		| DOTRP			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTRM			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTRN			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTRNI		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTRZ			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTRZI		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTHALF		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTEND		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTLO			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTABS		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTJOIN		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTNEG		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOTSAT		{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		;


geometries	: DOT1D			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOT2D			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| DOT3D			{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		;

opTypes		: DOTS8			{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS16		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS32		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS64		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU8			{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU16		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU32		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU64		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTB8			{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTB16		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTB32		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTB64		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTB128		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTF16		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTF32		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTF64		{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		;

operandList	: operandList COMMA operand	{}
		| operand
		;

operand		: { printf(" "); } registerlocation
		| { printf(" "); } specialRegister
		| { printf(" "); } memorylocation
		| { printf(" "); } immediateValue	
		;

registerlocation	: REGISTER		{ printf($1); g_instList->addRegister($1);}
			| vector		{ }
			| REGISTERLO		{ printf($1); g_instList->addRegister($1,true);}
			| REGISTERHI		{ printf($1); g_instList->addRegister($1,true);}
			| OFFSETREGISTER	{ printf($1); g_instList->addRegister($1);}
			| PREDREGISTER 		{ printf($1); g_instList->addPredicate($1);}
			| PREDREGISTER PIPE PREDREGISTER2
						{ printf($1); printf("|"); printf($3); g_instList->addDoublePredReg($1,$3);}
			| PREDREGISTER PIPE REGISTER
						{ printf($1); printf("|"); printf($3); g_instList->addDoublePredReg($1,$3);}
			| TEXOP			{ printf($1); g_instList->addTex($1);}
			;

vector		: VECTOR1D	{ printf($1); g_instList->addVector($1,1); }
		| VECTOR2D	{ printf($1); g_instList->addVector($1,2); }
		| VECTOR3D	{ printf($1); g_instList->addVector($1,3); }
		| VECTOR4D	{ printf($1); g_instList->addVector($1,4); }
		;

specialRegister	: NTIDXREGISTER		{ printf("%%ntid.x"); g_instList->getListEnd().addOperand("%%ntid.x");}
		| NTIDYREGISTER		{ printf("%%ntid.y"); g_instList->getListEnd().addOperand("%%ntid.y");}
		| CTAIDXREGISTER	{ printf("%%ctaid.x"); g_instList->getListEnd().addOperand("%%ctaid.x");}
		| CTAIDYREGISTER	{ printf("%%ctaid.y"); g_instList->getListEnd().addOperand("%%ctaid.y");}
		| NCTAIDXREGISTER	{ printf("%%nctaid.x"); g_instList->getListEnd().addOperand("%%nctaid.x");}
		| NCTAIDYREGISTER	{ printf("%%nctaid.y"); g_instList->getListEnd().addOperand("%%nctaid.y");}
		| CLOCKREGISTER		{ printf("%%clock"); g_instList->getListEnd().addOperand("%%clock");}
		;

memorylocation	: LMEMLOCATION		{ printf($1); g_instList->addMemoryOperand($1,3);}
		| SMEMLOCATION		{ printf($1); g_instList->addMemoryOperand($1,1);}
		| GMEMLOCATION		{ printf($1); g_instList->addMemoryOperand($1,2);}
		| CMEMLOCATION		{ printf($1); g_instList->addMemoryOperand($1,0);}
		;

immediateValue	: IDENTIFER	{ printf($1); g_instList->getListEnd().addOperand($1);}
		| HEXLITERAL	{ printf($1); g_instList->getListEnd().addOperand($1);}
		| OCTLITERAL	{ printf($1); g_instList->getListEnd().addOperand($1);}
		| BINLITERAL	{ printf($1); g_instList->getListEnd().addOperand($1);}
		| DECLITERAL	{ printf($1); g_instList->getListEnd().addOperand($1);}
		;


constMemoryDirectiveStatement 	: POUND DOTCONSTSEG DECLITERAL COLON HEXLITERAL CONST NEWLINE
				   { g_instList->addEntryConstMemory(atoi($3)); printf("//c"); printf($3); printf(" = "); }
				  POUND LEFTBRACE NEWLINE
				  constMemoryStatements
				  POUND RIGHTBRACE
				;

constMemoryStatements	: constMemoryStatement
			| constMemoryStatements constMemoryStatement
			;

constMemoryStatement	: POUND constMemoryTypes constMemoryList { g_instList->setConstMemoryType($2); } NEWLINE

constMemoryTypes	: DDOTU32 { char* tempString = new char[5];
					strcpy(tempString, ".u32");
					$$ = tempString; }
			| DDOTF32 { char* tempString = new char[5];
					strcpy(tempString, ".f32");
					$$ = tempString; }
			;

constMemoryList		: constMemory
			| constMemoryList COMMA constMemory
			;

constMemory		: HEXLITERAL	{printf($1); printf(" "); g_instList->addConstMemoryValue($1); }
			| OCTLITERAL	{printf($1); printf(" "); g_instList->addConstMemoryValue($1); }
			| BINLITERAL	{printf($1); printf(" "); g_instList->addConstMemoryValue($1); }
			| DECLITERAL	{printf($1); printf(" "); g_instList->addConstMemoryValue($1); }
			| FLTLITERAL	{printf($1); printf(" "); g_instList->addConstMemoryValue($1); }
			;
                       


%%

/*support c++ functions go here*/
