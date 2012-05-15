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

	/*Yacc file for elf files output by cuobjdump*/
%{
#include <stdio.h>
#include "cuobjdumpInstList.h"

int yylex(void);
void yyerror(const char*);
//void addEntryConstMemory(int, int);
//void setConstMemoryType(const char*);
//void addConstMemoryValue(const char*);
extern cuobjdumpInstList *g_instList;
int cmemcount=1;
int lmemcount=1;
bool lastcmem = false;// false = constrant0, true = constant1
%}
%union {
	char* string_value;
}
%token <string_value> C1BEGIN CMEMVAL SPACE2 C0BEGIN STBEGIN STHEADER
%token <string_value> NUMBER HEXNUMBER IDENTIFIER LOCALMEM

%%

elffile	:	symtab program
		;

symtab	:	STBEGIN STHEADER stcontent
		;

stcontent	:	stcontent stline
			|	stline;

stline	:	NUMBER NUMBER NUMBER NUMBER NUMBER NUMBER IDENTIFIER {
				if (strcmp($4, "11")==0) {
					g_instList->addConstMemoryPtr($2, $3, $7);
				}
			}
		|	NUMBER NUMBER NUMBER NUMBER NUMBER NUMBER {}
		;

program	:	program cmemsection
		|	program localmemsection
		|	{
				g_instList->setKernelCount(cmemcount-1);
				//g_instList->reverseConstMemory();
			};

localmemsection	:	LOCALMEM {
						printf("Found LocalMem section number %d\n", lmemcount);
						g_instList->addEntryLocalMemory(0, lmemcount);
						g_instList->setLocalMemoryMap($1, lmemcount);
						lmemcount++;
					};

cmemsection	:	C1BEGIN {
					printf("Found ConstMem section number %d\n", cmemcount);
					//g_instList->addEntryConstMemory(1, cmemcount);
					g_instList->addEntryConstMemory2($1);
					g_instList->setConstMemoryType2(".u32");
					//g_instList->setConstMemoryType(".u32");
					//g_instList->setConstMemoryMap($1,cmemcount);
					cmemcount++;
					lastcmem = true;
				} cmemvals
			|	C0BEGIN {
					printf("Found ConstMem c0 section\n");
					g_instList->addConstMemory(0);
					g_instList->setConstMemoryType(".u32");
					lastcmem = false;
				} cmemvals;

cmemvals	:	cmemvals CMEMVAL SPACE2 {
					//printf("Found ConstMem value\n");
					printf("addConstMemoryValue %s\n", $3);
					if (lastcmem)
						g_instList->addConstMemoryValue2($3);
					else
						g_instList->addConstMemoryValue($3);
				}
			|	;
%%
