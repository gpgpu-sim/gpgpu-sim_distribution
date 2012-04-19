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
