// Copyright (c) 2009-2012, Jimmy Kwa, Andrew Boktor
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
#include <stdio.h>
#include "cuobjdumpInstList.h"

int yylex(void);
void yyerror(const char*);

extern cuobjdumpInstList *g_instList;

cuobjdumpInst *instEntry;
%}


%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token <string_value> BAR
%token <string_value> ADA AND ANDS BRA CAL COS DADD DMIN DMAX DFMA DMUL EX2 F2F F2I FADD
%token <string_value> FADD32 FADD32I FMAD FMAD32I FMUL FMUL32 FMUL32I FSET DSET G2R
%token <string_value> GLD GST I2F I2I IADD IADD32 IADD32I IMAD ISAD IMAD24 IMAD32I IMAD32 IADDCARRY
%token <string_value> IMUL IMUL24 IMULS24 IMUL32 IMUL32S24 IMUL32U24 IMUL32I IMUL32I24 IMUL32IS24
%token <string_value> ISET LG2 LLD LST MOV MOV32 MVC MVI NOP NOT NOTS OR ORS
%token <string_value> R2A R2G R2GU16U8 RCP RCP32 RET RRO RSQ SIN SHL SHR SSY XOR XORS
%token <string_value> S2R SASS_LD STS LDS SASS_ST IMIN IMAX A2R FMAX FMIN TEX TEX32 C2R EXIT
%token <string_value> GRED PBK BRK R2C GATOM VOTE

%token <string_value> EQ EQU GE GEU GT GTU LE LEU LT LTU NE NEU
%token <string_value> DOTBEXT DOTS DOTSFU
%token <string_value> DOTTRUNC DOTCEIL DOTFLOOR DOTIR DOTUN DOTNODEP DOTSAT DOTANY DOTALL
%token <string_value> DOTF16 DOTF32 DOTF64 DOTS8 DOTS16 DOTS32 DOTS64 DOTS128 DOTU8 DOTU16 DOTU32 DOTU24 DOTU64
%token <string_value> DOTHI DOTNOINC
%token <string_value> DOTEQ DOTEQU DOTGE DOTGEU DOTGT DOTGTU DOTLE DOTLEU DOTLT DOTLTU DOTNE DOTNEU DOTNSF DOTSF DOTCARRY
%token <string_value> DOTCC DOTX DOTE DOTRED DOTPOPC
%token <string_value> REGISTER REGISTERLO REGISTERHI OFFSETREGISTER
%token <string_value> PREDREGISTER PREDREGISTER2 PREDREGISTER3 SREGISTER
%token <string_value> VERSIONHEADER FUNCTIONHEADER
%token <string_value> SMEMLOCATION ABSSMEMLOCATION GMEMLOCATION CMEMLOCATION LMEMLOCATION
%token <string_value> IDENTIFIER
%token <string_value> HEXLITERAL
%token <string_value> LEFTBRACKET RIGHTBRACKET
%token <string_value> PIPE TILDE
%token <string_value> NEWLINE SEMICOLON /*COMMA*/
%token <string_value> LABEL LABELSTART LABELEND
%token <string_value> PTXHEADER ELFHEADER
%token <string_value> INFOARCHVERSION
%token <string_value> INFOCODEVERSION_HEADER INFOCODEVERSION
%token <string_value> INFOPRODUCER
%token <string_value> INFOHOST
%token <string_value> INFOCOMPILESIZE_HEADER INFOCOMPILESIZE
%token <string_value> INFOIDENTIFIER DOT
%token <string_value> INSTHEX
%token <string_value> OSQBRACKET CSQBRACKET
	/* set types for rules */
%type<string_value> simpleInstructions
%type<string_value> predicateModifier
%type<string_value> opTypes

%%

	/*translation rules*/
program		: program sassCode
			| sassCode;

sassCode	: VERSIONHEADER IDENTIFIER NEWLINE functionList			{ printf($1); printf($2); printf(" No parsing errors\n\n");  }
		| NEWLINE VERSIONHEADER IDENTIFIER NEWLINE functionList	{ printf($2); printf($3); printf(" No parsing errors\n\n");  }
		| VERSIONHEADER IDENTIFIER NEWLINE;

functionList	: functionList function
				| function
				;
				
function	:	FUNCTIONHEADER IDENTIFIER {
					printf($1); 
					printf($2);
					printf("\n");
					g_instList->addEntry($2);
					instEntry = new cuobjdumpInst();
					instEntry->setBase(".entry");
					g_instList->add(instEntry);
					g_instList->getListEnd().addOperand($2);} statementList NEWLINE
					;


statementList	: statementList statement NEWLINE	{ printf("\n"); }
		| statementList statement SEMICOLON NEWLINE	{ printf(";\n"); }
		| statement NEWLINE			{ printf("\n"); }
		| statement SEMICOLON NEWLINE			{ printf(";\n"); }
		| NEWLINE	{}
		;

statement	: { instEntry = new cuobjdumpInst(); } instructionLabel instructionHex assemblyInstruction
			;

instructionHex	: INSTHEX
				;

instructionLabel	: LABELSTART LABEL LABELEND	{ char* tempInput = $2;
							  char* tempLabel = new char[12];
							  tempLabel[0] = 'l';
							  tempLabel[1] = '0';
							  tempLabel[2] = 'x';
							  for(int i=0; i<(8-strlen(tempInput)); i++)
							  {
								tempLabel[3+i] = '0';
							  }
							  for(int i=(11-strlen(tempInput)); i<11; i++)
							  {
								tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
							  }
							  tempLabel[11] = '\0';
							  instEntry->setLabel(tempLabel); }
			;

assemblyInstruction	: baseInstruction modifierList operandList	{ }
					/*| baseInstruction operandList			{ }*/
					/*| baseInstruction modifierList			{ }*/
					/*| baseInstruction				{ }*/
					;

baseInstruction : simpleInstructions	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry);}
		| branchInstructions
		| GRED DOT simpleInstructions	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); g_instList->getListEnd().addBaseModifier($3);}
		| GATOM DOT simpleInstructions	{ printf($1); instEntry->setBase($1); g_instList->add(instEntry); g_instList->getListEnd().addBaseModifier($3);}
		| pbkInstruction
		;

simpleInstructions	: ADA | AND | ANDS | COS | DADD | DMIN | DMAX | DFMA | DMUL | EX2 | F2F 
					| F2I | FADD | FADD32 | FADD32I | FMAD | FMAD32I | FMUL 
					| FMUL32 | FMUL32I | FSET | DSET | G2R | GLD | GST | I2F | I2I 
					| IADD | IADD32 | IADD32I | IMAD | ISAD | IMAD24 | IMAD32I | IMAD32 | IMUL 
					| IMUL24 | IMULS24 | IMUL32 | IMUL32S24 | IMUL32I | IMUL32I24 | IMUL32IS24
					| IMUL32U24
					| ISET | LG2 | LLD | LST | MOV | MOV32 | MVC | MVI | NOP 
					| NOT | NOTS | OR | ORS | R2A | R2G | R2GU16U8 | RCP | RCP32 | RET | RRO 
					| RSQ | SHL | SHR | SIN | SSY | XOR | XORS | S2R | SASS_LD | STS 
					| LDS | SASS_ST | EXIT | BAR | IMIN | IMAX | A2R | FMAX | FMIN 
					| TEX | TEX32 | C2R | BRK | R2C | IADDCARRY | VOTE
					;

pbkInstruction	:	PBK {
						printf($1); instEntry->setBase($1); g_instList->add(instEntry);
					} HEXLITERAL {
						char* tempInput = $3;
						char* tempLabel = new char[12];
						tempLabel[0] = 'l';
						tempLabel[1] = '0';
						tempLabel[2] = 'x';
						for(int i=0; i<(10-strlen(tempInput)); i++)
						{
							tempLabel[3+i] = '0';
						}
						for(int i=(13-strlen(tempInput)); i<11; i++)
						{
							tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
						}
						tempLabel[11] = '\0';
						g_instList->getListEnd().addOperand(tempLabel);
						g_instList->addCubojdumpLabel(tempLabel);
					}
				;

branchInstructions	: BRA {printf($1); instEntry->setBase($1); g_instList->add(instEntry);} instructionPredicate HEXLITERAL
				{ printf($4);
				  char* tempInput = $4;
				  char* tempLabel = new char[12];
				  tempLabel[0] = 'l';
				  tempLabel[1] = '0';
				  tempLabel[2] = 'x';
				  for(int i=0; i<(10-strlen(tempInput)); i++)
				  {
					tempLabel[3+i] = '0';
				  }
				  for(int i=(13-strlen(tempInput)); i<11; i++)
				  {
					tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
				  }
				  tempLabel[11] = '\0';
				  g_instList->getListEnd().addOperand(tempLabel);
				  g_instList->addCubojdumpLabel(tempLabel);}
			| BRA {printf($1); instEntry->setBase($1); g_instList->add(instEntry);} HEXLITERAL
				{ printf($3);
				  char* tempInput = $3;
				  char* tempLabel = new char[12];
				  tempLabel[0] = 'l';
				  tempLabel[1] = '0';
				  tempLabel[2] = 'x';
				  for(int i=0; i<(10-strlen(tempInput)); i++)
				  {
					tempLabel[3+i] = '0';
				  }
				  for(int i=(13-strlen(tempInput)); i<11; i++)
				  {
					tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
				  }
				  tempLabel[11] = '\0';
				  g_instList->getListEnd().addOperand(tempLabel);
				  g_instList->addCubojdumpLabel(tempLabel);}
			| CAL {printf($1); instEntry->setBase($1); g_instList->add(instEntry);} HEXLITERAL
				{ printf($3);
				  char* tempInput = $3;
				  char* tempLabel = new char[12];
				  tempLabel[0] = 'l';
				  tempLabel[1] = '0';
				  tempLabel[2] = 'x';
				  for(int i=0; i<(10-strlen(tempInput)); i++)
				  {
					tempLabel[3+i] = '0';
				  }
				  for(int i=(13-strlen(tempInput)); i<11; i++)
				  {
					tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
				  }
				  tempLabel[11] = '\0';
				  g_instList->getListEnd().addOperand(tempLabel);
				  g_instList->addCubojdumpLabel(tempLabel);}
			
			| CAL {printf($1); instEntry->setBase($1); g_instList->add(instEntry);} DOTNOINC HEXLITERAL
				{ printf($4);
				  char* tempInput = $4;
				  char* tempLabel = new char[12];
				  tempLabel[0] = 'l';
				  tempLabel[1] = '0';
				  tempLabel[2] = 'x';
				  for(int i=0; i<(10-strlen(tempInput)); i++)
				  {
					tempLabel[3+i] = '0';
				  }
				  for(int i=(13-strlen(tempInput)); i<11; i++)
				  {
					tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
				  }
				  tempLabel[11] = '\0';
				  g_instList->getListEnd().addOperand(tempLabel);
				  g_instList->addCubojdumpLabel(tempLabel);}

			;

modifierList	: modifier modifierList
				/*| modifier */
				|
				;

modifier	: opTypes	{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTBEXT		{ g_instList->getListEnd().addBaseModifier(".bext"); }
		| DOTS			{ g_instList->getListEnd().addBaseModifier(".s"); }
		| DOTSFU		{ g_instList->getListEnd().addBaseModifier(".sfu"); }
		| DOTTRUNC		{ g_instList->getListEnd().addBaseModifier(".rz"); }
		| DOTCEIL		{ g_instList->getListEnd().addBaseModifier(".rp"); }
		| DOTFLOOR		{ g_instList->getListEnd().addBaseModifier(".rm"); }
		| DOTX			{ g_instList->getListEnd().addBaseModifier(".x"); }
		| DOTE			{ g_instList->getListEnd().addBaseModifier(".e"); }
		| DOTRED		{ g_instList->getListEnd().addBaseModifier(".red"); }
		| DOTPOPC		{ g_instList->getListEnd().addBaseModifier(".popc"); }
		| DOTIR			{ g_instList->getListEnd().addBaseModifier(".ir"); }
		| DOTUN			{ /*g_instList->getListEnd().addBaseModifier(".un"); */}
		| DOTNODEP		{ /*g_instList->getListEnd().addBaseModifier(".nodep"); */}
		| DOTANY		{ g_instList->getListEnd().addBaseModifier(".any"); }
		| DOTALL		{ g_instList->getListEnd().addBaseModifier(".all"); }
		;

opTypes		: DOTF16	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTF32	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTF64	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS8		//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS16	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS32	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS64	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTS128	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU8		//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU16	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU32	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU24	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTU64	//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		| DOTHI		//{ printf($1); g_instList->getListEnd().addTypeModifier($1);}
		;

operandList	: operandList { printf(" "); } /*COMMA*/ operand	{}
			/*| { printf(" "); } operand		{}*/
			|
			;

operand		: registerlocation
		| PIPE registerlocation PIPE	{ g_instList->getListEnd().addBaseModifier(".abs"); }
		| TILDE registerlocation
		| LEFTBRACKET instructionPredicate RIGHTBRACKET
		| memorylocation opTypes { printf($2); g_instList->getListEnd().addTypeModifier($2);}
		| memorylocation
		| immediateValue
		| extraModifier
		| operandPredicate
		| preOperand
		;
/* Register of the format [R0] will be converted to R0 */
/* regMod will be also ignored */
registerlocation	: REGISTER regMod	{ printf($1); g_instList->addCuobjdumpRegister($1);}
			| OSQBRACKET REGISTER CSQBRACKET	{ printf($1); printf($2); printf($3); g_instList->addCuobjdumpRegister($2);}
			| REGISTERLO	{ printf($1); g_instList->addCuobjdumpRegister($1,true);}
			| REGISTERHI	{ printf($1); g_instList->addCuobjdumpRegister($1,true);}
			| SREGISTER		{ printf($1); g_instList->addCuobjdumpRegister($1,false);}
			| OFFSETREGISTER	{ printf($1); g_instList->addCuobjdumpRegister($1);}
			| PREDREGISTER PREDREGISTER2	{ printf($1); printf(" "); printf($2); g_instList->addCuobjdumpDoublePredReg($1, $2);}
			| PREDREGISTER REGISTER	{ printf($1); printf(" "); printf($2); g_instList->addCuobjdumpDoublePredReg($1, $2);}
			/*| REGISTER PREDREGISTER3 { printf($1); printf(" "); printf($2); g_instList->addCuobjdumpRegister($1); printf("WEIRD CASE\n");}*/
			;

regMod		: DOTCC
			|
			;


memorylocation	: SMEMLOCATION	{ printf($1); g_instList->addCuobjdumpMemoryOperand($1,1);}
		|	ABSSMEMLOCATION {
				printf($1);
				char* input = $1;
				char* temp = new char[99];
				temp[0] = input[1];
				unsigned i=1;
				while (i < strlen(input)-2) {
					temp[i] = input[i+2];
					i++;
				}
				g_instList->addCuobjdumpMemoryOperand(temp,1);
				g_instList->getListEnd().addBaseModifier(".abs");
			}
		| GMEMLOCATION	{ printf($1); g_instList->addCuobjdumpMemoryOperand($1,2);}
		| CMEMLOCATION	{ printf($1); g_instList->addCuobjdumpMemoryOperand($1,0);}
		| LMEMLOCATION	{ printf($1); g_instList->addCuobjdumpMemoryOperand($1,3);}
		;

immediateValue	: IDENTIFIER { printf($1); g_instList->getListEnd().addOperand($1);}
		| HEXLITERAL { printf($1); g_instList->getListEnd().addOperand($1);}
		;

extraModifier	: EQ	{ printf($1); g_instList->getListEnd().addBaseModifier($1);} 
		| EQU	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| GE	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| GEU	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| GT	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| GTU	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| LE	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| LEU	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| LT	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| LTU	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| NE	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		| NEU	{ printf($1); g_instList->getListEnd().addBaseModifier($1);}
		;

instructionPredicate	: PREDREGISTER3	predicateModifier {printf($1); printf($2);
								g_instList->getListEnd().setPredicate($1);
								g_instList->getListEnd().addPredicateModifier($2);}
						| PREDREGISTER3 {printf($1); g_instList->getListEnd().setPredicate($1);}
			;

operandPredicate	:	PREDREGISTER3	predicateModifier {
							printf($1); 
							printf($2);
							//g_instList->getListEnd().addOperand($1);
							g_instList->getListEnd().setPredicate($1);
							g_instList->getListEnd().addPredicateModifier($2);
							/*May be the modifier needs to be added too*/
						}
					|	PREDREGISTER3 {
							printf("HELLO: "); 
							printf($1); 
							g_instList->getListEnd().addOperand($1);
						}
					;


preOperand	: EX2	{ printf($1); g_instList->getListEnd().addBaseModifier("ex2");}
		| SIN	{ printf($1); g_instList->getListEnd().addBaseModifier("sin");}
		| COS	{ printf($1); g_instList->getListEnd().addBaseModifier("cos");}
		;

predicateModifier	: DOTEQ	{ }
			| DOTEQU	{ }
			| DOTGE	{ }
			| DOTGEU	{ }
			| DOTGT	{ }
			| DOTGTU	{ }
			| DOTLE	{ }
			| DOTLEU	{ }
			| DOTLT	{ }
			| DOTLTU	{ }
			| DOTNE	{ }
			| DOTNEU	{ }
			| DOTNSF	{ }
			| DOTSF	{ }
			| DOTCARRY	{ }
			;

%%

/*support c++ functions go here*/

