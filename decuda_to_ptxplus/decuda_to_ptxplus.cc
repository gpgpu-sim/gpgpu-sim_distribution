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
