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

typedef void * yyscan_t;
#include "cuobjdump.h"

extern void addCuobjdumpSection(int sectiontype, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumparch(const char* arch, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpidentifier(const char* identifier, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpptxfilename(const char* filename, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpelffilename(const char* filename, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpsassfilename(const char* filename, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
%}
%define api.pure full
%parse-param {yyscan_t scanner}
%parse-param {struct cuobjdump_parser* parser}
%parse-param {std::list<cuobjdumpSection*> &cuobjdumpSectionList}
%lex-param {yyscan_t scanner}
%lex-param {struct cuobjdump_parser* parser}
%lex-param {std::list<cuobjdumpSection*> &cuobjdumpSectionList}

%union {
	char* string_value;
}
%{
int yylex(YYSTYPE * yylval_param, yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void yyerror(yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList, const char* msg);
%}
%token <string_value> H_SEPARATOR H_ARCH H_CODEVERSION H_PRODUCER H_HOST H_COMPILESIZE H_IDENTIFIER H_UNKNOWN H_COMPRESSED
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
				addCuobjdumpSection(0, cuobjdumpSectionList);
				snprintf(parser->filename, 1024, "_cuobjdump_%d.ptx", parser->ptxserial++);
				parser->ptxfile = fopen(parser->filename, "w");
				setCuobjdumpptxfilename(parser->filename, cuobjdumpSectionList);
			} headerinfo compressedkeyword identifier ptxcode {
				fclose(parser->ptxfile);
			}
		|	ELFHEADER {
				addCuobjdumpSection(1, cuobjdumpSectionList);
				snprintf(parser->filename, 1024, "_cuobjdump_%d.elf", parser->elfserial);
				parser->elffile = fopen(parser->filename, "w");
				setCuobjdumpelffilename(parser->filename, cuobjdumpSectionList);
			} headerinfo compressedkeyword identifier elfcode {
				fclose(parser->elffile);
				snprintf(parser->filename, 1024, "_cuobjdump_%d.sass", parser->elfserial++);
				parser->sassfile = fopen(parser->filename, "w");
				setCuobjdumpsassfilename(parser->filename, cuobjdumpSectionList);
			} sasscode { 
				fclose(parser->sassfile);
			};

headerinfo :	H_SEPARATOR NEWLINE
				H_ARCH IDENTIFIER NEWLINE
				H_CODEVERSION CODEVERSION NEWLINE
				H_PRODUCER H_UNKNOWN NEWLINE
				H_HOST IDENTIFIER NEWLINE
				H_COMPILESIZE IDENTIFIER  {setCuobjdumparch($4, cuobjdumpSectionList);};
			|   H_SEPARATOR NEWLINE
				H_ARCH IDENTIFIER NEWLINE
				H_CODEVERSION CODEVERSION NEWLINE
				H_PRODUCER IDENTIFIER NEWLINE
				H_HOST IDENTIFIER NEWLINE
				H_COMPILESIZE IDENTIFIER {setCuobjdumparch($4, cuobjdumpSectionList);};

identifier : H_IDENTIFIER FILENAME emptylines {setCuobjdumpidentifier($2, cuobjdumpSectionList);}
			 |	{setCuobjdumpidentifier("default", cuobjdumpSectionList);};

compressedkeyword : H_COMPRESSED emptylines
                    | ;

ptxcode :	ptxcode PTXLINE {fprintf(parser->ptxfile, "%s", $2);}
		|	;

elfcode :	elfcode ELFLINE {fprintf(parser->elffile, "%s", $2);}
		|	;

sasscode :	sasscode SASSLINE {fprintf(parser->sassfile, "%s", $2);}
		 |	;


%%
