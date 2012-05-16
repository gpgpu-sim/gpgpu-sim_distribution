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

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cassert>

#include "cuobjdumpInstList.h"

using namespace std;

cuobjdumpInstList *g_instList = new cuobjdumpInstList();
cuobjdumpInstList *g_headerList = new cuobjdumpInstList();

int sass_parse();
extern "C" FILE *sass_in;

int ptx_parse();
extern "C" FILE *ptx_in;

int elf_parse();
extern "C" FILE *elf_in;

extern int g_error_detected;

FILE *bin_in;
FILE *ptxplus_out;

void output(const char * text)
{
	printf(text);
	fprintf(ptxplus_out, text);
}

void output(const std::string text) {
	output(text.c_str());
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

std::string extractFilename( const std::string& path )
{
	return path.substr( path.find_last_of( '/' ) +1 );
}
int main(int argc, char* argv[])
{
	//TODO: Output to file not yet supported.
	if(argc != 5)
	{
		cout << "Usage: cuobjdump_to_ptxplus ptxfile sassfile elffile ptxplusfile(output)\n";
		return 0;
	}
	int result;
	string ptxfile = argv[1];
	string sassfile = argv[2];
	string elffile = argv[3];
	string ptxplusfile = argv[4];

	/*
	char commandline[1024];

	snprintf(commandline,1024,"cuobjdump -ptx %s > %s.ptx", exefile.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}


	snprintf(commandline,1024,"sed '/arch\\ =\\ sm_20/,/1a$/d' %s.ptx > %s.stripped2.ptx", basefilename.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}

	snprintf(commandline,1024,"grep -v \"^arch\\ =\\|Fatbin\\ \\|^code\\ version\\|===========\\|producer\\ =\\|host\\ =\\|compile_size\\ =\\|identifier\\ =\" %s.stripped2.ptx > %s.stripped.ptx", basefilename.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}

	snprintf(commandline,1024,"cuobjdump -sass %s > %s.sass", exefile.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}

	snprintf(commandline,1024,"sed '/arch\\ =\\ sm_20/,/1a$/d' %s.sass > %s.stripped.sass", basefilename.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}

	snprintf(commandline,1024,"cuobjdump -elf %s > %s.elf", exefile.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}

	snprintf(commandline,1024,"sed '/arch\\ =\\ sm_20/,/1a$/d' %s.elf > %s.stripped.elf", basefilename.c_str(), basefilename.c_str());
	fflush(stdout);
	result = system(commandline);
	if (result) {printf("ERROR: could not execute %s\n", commandline); exit(1);}
	*/

	sass_in = fopen(sassfile.c_str(), "r" );
	ptx_in = fopen(ptxfile.c_str(), "r" );
	elf_in = fopen(elffile.c_str(), "r");
	ptxplus_out = fopen(ptxplusfile.c_str(), "w" );


	std::string elf = fileToString(elffile.c_str());

	printf("RUNNING cuobjdump_to_ptxplus ...\n");


	printf("Parsing .elf file %s\n", elffile.c_str());
	elf_parse();
	printf("Finished parsing .elf file %s\n", elffile.c_str());

	//Parse original ptx
	printf("Parsing .ptx file %s\n", ptxfile.c_str());
	ptx_parse();
	if (g_error_detected){
		assert(0 && "ptx parsing failed");
	}
	printf("Finished parsing .ptx file %s\n", ptxfile.c_str());

	// Copy real tex list from ptx to ptxplus instruction list
	g_instList->setRealTexList(g_headerList->getRealTexList());
	// Insert constant memory from bin file
//	g_instList->readConstMemoryFromElfFile(elf);
//	g_instList->readOtherConstMemoryFromBinFile(fileToString(binFilename));

	// Insert global memory from bin file
//	g_instList->readGlobalMemoryFromBinFile(fileToString(binFilename));

	// Parse cuobjdump output
	printf("Parsing .sass file %s\n", sassfile.c_str());
	sass_parse();
	printf("Finished parsing .sass file %s\n", sassfile.c_str());
	/*
	printf("################################################## Instruction List dump\n");
	g_instList->printCuobjdumpInstList();
	printf("################################################## END Instruction List dump\n");
	printf("################################################## Header Instructions Dump\n");
	g_headerList->printCuobjdumpInstList();
	printf("################################################## END Header Instructions Dump\n");
	*/
	// Print ptxplus
	output("//HEADER\n");
	g_headerList->printHeaderInstList();
	output("//END HEADER\n\n\n");
	output("//INSTRUCTIONS\n");
	g_instList->printCuobjdumpPtxPlusList(g_headerList);
	output("//END INSTRUCTIONS\n");
	// TODO: remove this. Prints recorded cuobjdump output.
//	g_instList->printCuobjdumpInstList();

	fclose(sass_in);
	fclose(ptx_in);
//	fclose(bin_in);
	fclose(ptxplus_out);

	printf("DONE. \n");

	return 0;
}

