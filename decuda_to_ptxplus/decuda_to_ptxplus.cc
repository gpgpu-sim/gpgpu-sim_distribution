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


#include <iostream>
#include "decudaInstList.h"
#include <stdio.h>
#include<fstream>

using namespace std;

decudaInstList *g_instList = new decudaInstList();
decudaInstList *g_headerList = new decudaInstList();

int yyparse();
extern "C" FILE *yyin;

int ptx_parse();
extern "C" FILE *ptx_in;

FILE *bin_in;
FILE *ptxplus_out;

void output(const char * text)
{
	fprintf(ptxplus_out, text);
}

std::string fileToString(const char * fileName) {
	ifstream fileStream(fileName, ios::in);
	string text, line;
	while(getline(fileStream,line)) {
		text += (line + "\n");
	}
	fileStream.close();
	return text;
}

int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		cout << "Usage: decuda_to_ptxplus [decuda filename] [ptx filename] [bin filename] [ptxplus output filename]\n";
		return 0;
	}

	const char *decudaFilename = argv[1];
	const char *ptxFilename = argv[2];
	const char *binFilename = argv[3];
	const char *ptxplusFilename = argv[4];

	//header_in = fopen( ptxFilename, "r" );
	ptx_in = fopen( ptxFilename, "r" );
	yyin = fopen( decudaFilename, "r" );
	bin_in = fopen( binFilename, "r" );
	ptxplus_out = fopen( ptxplusFilename, "w" );


	fileToString(binFilename);

	printf("RUNNING decuda2ptxplus ...\n");

	// Parse original ptx
	ptx_parse();

	// Copy real tex list from ptx to ptxplus instruction list
	g_instList->setRealTexList(g_headerList->getRealTexList());

	// Insert constant memory from bin file
	g_instList->readConstMemoryFromBinFile(fileToString(binFilename));

	// Insert global memory from bin file
	g_instList->readGlobalMemoryFromBinFile(fileToString(binFilename));

	// Parse decuda output
	yyparse();
	printf("END RUN\n");

	// Print ptxplus
	g_headerList->printHeaderInstList();
	g_instList->printNewPtxList(g_headerList);

	fclose(ptx_in);
	fclose(yyin);
	fclose(bin_in);
	fclose(ptxplus_out);

	printf("DONE. \n");

	return 0;
}
