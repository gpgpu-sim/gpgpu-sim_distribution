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

	/*Yacc file for output of cuobjdump*/
%{
#include <stdio.h>

int yylex(void);
void yyerror(const char*);
extern void addCuobjdumpSection(int sectiontype);
void setCuobjdumparch(const char* arch);
void setCuobjdumpidentifier(const char* identifier);
void setCuobjdumpptxfilename(const char* filename);
void setCuobjdumpelffilename(const char* filename);
void setCuobjdumpsassfilename(const char* filename);
int elfserial = 1;
int ptxserial = 1;
FILE *ptxfile;
FILE *elffile;
FILE *sassfile;
char filename [1024];
%}
%union {
	char* string_value;
}
%token <string_value> H_SEPARATOR H_ARCH H_CODEVERSION H_PRODUCER H_HOST H_COMPILESIZE H_IDENTIFIER
%token <string_value> CODEVERSION
%token <string_value> STRING
%token <string_value> FILENAME
%token <string_value> DECIMAL
%token <string_value> PTXHEADER ELFHEADER
%token <string_value> PTXLINE
%token <string_value> ELFLINE
%token <string_value> SASSLINE
%token <string_value> IDENTIFIER
%token <string_value> NEWLINE

%%

program :	{printf("######### cuobjdump parser ########\n");}
			emptylines section
		|	program section;

emptylines	:	emptylines NEWLINE
			|	;

section :	PTXHEADER {
				addCuobjdumpSection(0);
				snprintf(filename, 1024, "_cuobjdump_%d.ptx", ptxserial++);
				ptxfile = fopen(filename, "w");
				setCuobjdumpptxfilename(filename);
			} headerinfo ptxcode {
				fclose(ptxfile);
			}
		|	ELFHEADER {
				addCuobjdumpSection(1);
				snprintf(filename, 1024, "_cuobjdump_%d.elf", elfserial);
				elffile = fopen(filename, "w");
				setCuobjdumpelffilename(filename);
			} headerinfo elfcode { 
				fclose(elffile);
				snprintf(filename, 1024, "_cuobjdump_%d.sass", elfserial++);
				sassfile = fopen(filename, "w");
				setCuobjdumpsassfilename(filename);
			} sasscode { 
				fclose(sassfile);
			};

headerinfo :	H_SEPARATOR NEWLINE
				H_ARCH IDENTIFIER NEWLINE
				H_CODEVERSION CODEVERSION NEWLINE
				H_PRODUCER IDENTIFIER NEWLINE
				H_HOST IDENTIFIER NEWLINE
				H_COMPILESIZE IDENTIFIER NEWLINE
				H_IDENTIFIER FILENAME emptylines {setCuobjdumparch($4); setCuobjdumpidentifier($19);};

ptxcode :	ptxcode PTXLINE {fprintf(ptxfile, "%s", $2);}
		|	;

elfcode :	elfcode ELFLINE {fprintf(elffile, "%s", $2);}
		|	;

sasscode :	sasscode SASSLINE {fprintf(sassfile, "%s", $2);}
		 |	;


%%
