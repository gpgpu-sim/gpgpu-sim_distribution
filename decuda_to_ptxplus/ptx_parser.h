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


   #include <cstdlib>
   #include <cstring>
   #include <string>
   #include <list>
   #include <map>
   #include <vector>
   #include <assert.h>

#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3


enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space
};

int g_error_detected;
const char *g_filename = "";
int g_func_decl;

void set_symtab( void* a ) {}
void end_function() {}
void add_directive() {}
void add_function_arg() {}
void add_instruction() {}
void add_file( unsigned a, const char *b ) {}
void add_variables() {}
void set_variable_type() {}
void add_identifier( const char *a, int b, unsigned c ) {}
void add_option(int a ) {}
void add_array_initializer() {}
void add_label( const char *a ) {}
void set_return() {}
void add_opcode( int a ) {}
void add_pred( const char *a, int b, int c ) {}
void add_scalar_operand( const char *a ) {}
void add_neg_pred_operand( const char *a ) {}
void add_address_operand( const char *a, int b ) {}
void change_operand_lohi( int a ) {}
void change_double_operand_type( int a ) {}
void change_operand_neg( ) {}
void add_double_operand( const char *a, const char *b ) {}
void add_2vector_operand( const char *a, const char *b ) {}
void add_3vector_operand( const char *a, const char *b, const char *c ) {}
void add_4vector_operand( const char *a, const char *b, const char *c, const char *d ) {}
void add_builtin_operand( int a, int b ) {}
void add_memory_operand() {}
void change_memory_addr_space( const char *a ) {}
void add_literal_int( int a ) {}
void add_literal_float( float a ) {}
void add_literal_double( double a ) {}
void func_header_info_int(const char*, int) {}
void add_extern_spec() {}
void add_alignment_spec( int ) {}
void add_pragma( const char *a ) {}

void add_constptr(const char* identifier1, const char* identifier2, int offset) {}

/*non-dummy stuff below this point*/

#include <iostream>
#include <sstream>
#include "decudaInstList.h"
#include <string.h>

extern decudaInstList *g_headerList;

// Global variable to track if we are currently inside a entry directive
bool inEntryDirective = false;
// Global variable to track is we are currently inside the parameter definitions for an entry
bool inParamDirective = false;

// Global variable to track if we are currently inside a tex directive
bool inTexDirective = false;

void add_function_name( const char *headerInput )
{
	char* headerInfo = (char*) headerInput;
	const char* compareString = g_headerList->getListEnd().getBase();

	if((strcmp(compareString, ".entry")==0)||(strcmp(compareString, ".func")==0))
	{
		g_headerList->setLastEntryName(headerInfo);
		g_headerList->getListEnd().addOperand(headerInfo);
	}
}

//void add_space_spec(int headerInput)
void add_space_spec( enum _memory_space_t spec, int value )
{
	switch(spec)
	{
		case param_space_unclassified:
			if(inEntryDirective && inParamDirective) {
				decudaInst *instEntry = new decudaInst();
				instEntry->setBase(".param");
				g_headerList->add(instEntry);
			}
			break;
		case tex_space:
			inTexDirective = true;
			decudaInst *instEntry = new decudaInst();
			instEntry->setBase(".tex");
			g_headerList->add(instEntry);			
			break;
	}
}

void add_scalar_type_spec( int headerInput )
{
	const char* compareString = g_headerList->getListEnd().getBase();

	if( (inEntryDirective && inParamDirective) || inTexDirective)
	{
		switch(headerInput)
		{
			case S8_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s8");
				break;
			case S16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s16");
				break;
			case S32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s32");
				break;
			case S64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".s64");
				break;
			case U8_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u8");
				break;
			case U16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u16");
				break;
			case U32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u32");
				break;
			case U64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".u64");
				break;
			case F16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".f16");
				break;
			case F32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".f32");
				break;
			case F64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".f64");
				break;
			case B8_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b8");
				break;
			case B16_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b16");
				break;
			case B32_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b32");
				break;
			case B64_TYPE:
				g_headerList->getListEnd().addBaseModifier(".b64");
				break;
			case PRED_TYPE:
				g_headerList->getListEnd().addBaseModifier(".pred");
				break;
			default:
				std::cout << "Unknown type spec" << "\n";
				break;
		}
	}
}

//void version_header(double versionNumber)
void add_version_info( float versionNumber, unsigned ext)
{
	decudaInst *instEntry = new decudaInst();
	instEntry->setBase(".version");
	g_headerList->add(instEntry);


	//convert double to char*
	std::ostringstream strs;
	strs << versionNumber;
	char *versionNumber2 = strdup(strs.str().c_str());

	g_headerList->getListEnd().addOperand(versionNumber2);
	//g_headerList->getListEnd().addOperand("1.4");
}

void target_header(char* firstTarget)
{

	decudaInst *instEntry = new decudaInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);	

	g_headerList->getListEnd().addOperand(firstTarget);
}

void target_header2(char* firstTarget, char* secondTarget)
{
	decudaInst *instEntry = new decudaInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);	

	g_headerList->getListEnd().addOperand(firstTarget);

	g_headerList->getListEnd().addOperand(secondTarget);
}

void target_header3(char* firstTarget, char* secondTarget, char* thirdTarget)
{
        decudaInst *instEntry = new decudaInst();
        instEntry->setBase(".target");
        g_headerList->add(instEntry);

        g_headerList->getListEnd().addOperand(firstTarget);

        g_headerList->getListEnd().addOperand(secondTarget);

	g_headerList->getListEnd().addOperand(thirdTarget);
}

void start_function( int a )
{
	inEntryDirective = true;
}

void* reset_symtab()
{
	inEntryDirective = false;
	void* a;
	return a;
}

void func_header(const char* headerBase)
{

	// If start of an entry
	if((strcmp(headerBase, ".entry")==0)||(strcmp(headerBase, ".func")==0)) {
		g_headerList->addEntry("");

		decudaInst *instEntry = new decudaInst();
		instEntry->setBase(headerBase);
		g_headerList->add(instEntry);

	}
}

void func_header_info(const char* headerInfo)
{
	const char* compareString = g_headerList->getListEnd().getBase();

	if(inEntryDirective) {
		g_headerList->getListEnd().addOperand(headerInfo);

		// If start of parameters
		if(strcmp(headerInfo,"(")==0)
			inParamDirective = true;

		// If end of parameters
		if(strcmp(headerInfo,")")==0) {
			inParamDirective = false;
		}
	}

	if(inTexDirective) {
		inTexDirective = false;
		g_headerList->addTex(headerInfo);
	}

}
