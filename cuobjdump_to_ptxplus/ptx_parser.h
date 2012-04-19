#include <cstdlib>
#include <cstring>
#include <string>
#include <list>
#include <map>
#include <vector>
#include <assert.h>
#include <iostream>
#include <sstream>
#include "decudaInstList.h"
#include <string.h>

#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3
#define P_DEBUG 1
#define DPRINTF(...) \
   if(P_DEBUG) { \
      printf("(%s:%s:%u) ", __FILE__, __FUNCTION__, __LINE__); \
      printf(__VA_ARGS__); \
      printf("\n"); \
      fflush(stdout); \
   }


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

void set_symtab( void* a ) {DPRINTF("");}
void end_function() {DPRINTF("");}
void add_directive() {DPRINTF("");}
void add_function_arg() {DPRINTF("");}
void add_instruction() {DPRINTF("");}
void add_file( unsigned a, const char *b ) {DPRINTF("");}
void add_variables() {DPRINTF("");}
void set_variable_type() {DPRINTF("");}
void add_option(int a ) {DPRINTF("");}
void add_array_initializer() {DPRINTF("");}
void add_label( const char *a ) {DPRINTF("");}
void set_return() {DPRINTF("");}
void add_opcode( int a ) {DPRINTF("");}
void add_pred( const char *a, int b, int c ) {DPRINTF("");}
void add_scalar_operand( const char *a ) {DPRINTF("%s", a);}
void add_neg_pred_operand( const char *a ) {DPRINTF("");}
void add_address_operand( const char *a, int b ) {DPRINTF("%s", a);}
void change_operand_lohi( int a ) {DPRINTF("");}
void change_double_operand_type( int a ) {DPRINTF("");}
void change_operand_neg( ) {DPRINTF("");}
void add_double_operand( const char *a, const char *b ) {DPRINTF("");}
void add_2vector_operand( const char *a, const char *b ) {DPRINTF("");}
void add_3vector_operand( const char *a, const char *b, const char *c ) {DPRINTF("");}
void add_4vector_operand( const char *a, const char *b, const char *c, const char *d ) {DPRINTF("");}
void add_builtin_operand( int a, int b ) {DPRINTF("");}
void add_memory_operand() {DPRINTF("");}
void change_memory_addr_space( const char *a ) {DPRINTF("");}
void add_literal_int( int a ) {DPRINTF("");}
void add_literal_float( float a ) {DPRINTF("");}
void add_literal_double( double a ) {DPRINTF("");}
void add_extern_spec() {DPRINTF("");}
void add_alignment_spec( int ) {DPRINTF("");}
void add_pragma( const char *a ) {DPRINTF("");}
void add_constptr(const char* identifier1, const char* identifier2, int offset) {DPRINTF("");}

/*non-dummy stuff below this point*/



extern decudaInstList *g_headerList;

// Global variable to track if we are currently inside a entry directive
bool inEntryDirective = false;
// Global variable to track is we are currently inside the parameter definitions for an entry
bool inParamDirective = false;

bool inConstDirective = false;

// Global variable to track if we are currently inside a tex directive
bool inTexDirective = false;


void add_identifier( const char *a, int b, unsigned c ) {
	DPRINTF("name=%s", a);
	if(inConstDirective){
		//g_headerList->getListEnd()
	}
}

void add_function_name( const char *headerInput )
{
	DPRINTF("name=%s", headerInput);
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
	DPRINTF("spec=%u", spec);
	decudaInst *instEntry;
	static int constmemindex=1;
	switch(spec)
	{
		case param_space_unclassified:
			if(inEntryDirective && inParamDirective) {
				instEntry = new decudaInst();
				instEntry->setBase(".param");
				g_headerList->add(instEntry);
			}
			break;
		case tex_space:
			inTexDirective = true;
			/*
			instEntry = new decudaInst();
			instEntry->setBase(".tex");
			g_headerList->add(instEntry);
			*/
			break;
		case const_space:
			if(!inEntryDirective) {
				/*
				inConstDirective = true;
				instEntry = new decudaInst();
				instEntry->setBase(".const");
				g_headerList->add(instEntry);
				*/
				//g_headerList->addConstMemory(constmemindex++);
			}
			break;
	}
}

void add_scalar_type_spec( int headerInput )
{
	DPRINTF("");
	const char* compareString = g_headerList->getListEnd().getBase();

	if( (inEntryDirective && inParamDirective) || inTexDirective || inConstDirective)
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
	DPRINTF("");
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
	DPRINTF("%s", firstTarget);
	decudaInst *instEntry = new decudaInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);	

	g_headerList->getListEnd().addOperand(firstTarget);
}

void target_header2(char* firstTarget, char* secondTarget)
{
	DPRINTF("%s, %s", firstTarget, secondTarget);
	decudaInst *instEntry = new decudaInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);	

	g_headerList->getListEnd().addOperand(firstTarget);

	g_headerList->getListEnd().addOperand(secondTarget);
}

void target_header3(char* firstTarget, char* secondTarget, char* thirdTarget)
{
	DPRINTF("%s, %s, %s", firstTarget, secondTarget, thirdTarget);
	decudaInst *instEntry = new decudaInst();
	instEntry->setBase(".target");
	g_headerList->add(instEntry);

	g_headerList->getListEnd().addOperand(firstTarget);

	g_headerList->getListEnd().addOperand(secondTarget);

	g_headerList->getListEnd().addOperand(thirdTarget);
}

void start_function( int a )
{
	DPRINTF("");
	inEntryDirective = true;
}

void* reset_symtab()
{
	DPRINTF("");
	inEntryDirective = false;
	void* a;
	return a;
}

void func_header(const char* headerBase)
{
	DPRINTF("%s", headerBase);
	// If start of an entry
	if((strcmp(headerBase, ".entry")==0)||(strcmp(headerBase, ".func")==0)) {
		inEntryDirective = true;
		g_headerList->addEntry("");
		decudaInst *instEntry = new decudaInst();
		instEntry->setBase(headerBase);
		g_headerList->add(instEntry);

	}
}

void func_header_info(const char* headerInfo)
{
	DPRINTF("%s", headerInfo);
	const char* compareString = g_headerList->getListEnd().getBase();

	if(inEntryDirective && !inTexDirective) {
		g_headerList->getListEnd().addOperand(headerInfo);
		// If start of parameters
		if(strcmp(headerInfo,"(")==0)
			inParamDirective = true;

		// If end of parameters
		if(strcmp(headerInfo,")")==0) {
			inParamDirective = false;
		}
	} else if(inTexDirective) {
		g_headerList->addTex(headerInfo);
		inTexDirective = false;
	} else if(inConstDirective){

	} else {
		printf("Unkown header info: #%s#\n", headerInfo);
	}

}

void func_header_info_int(const char* s, int i)
{
	DPRINTF("%s %d", s, i);
	if(inEntryDirective && !inTexDirective) {
		g_headerList->getListEnd().addOperand(s);
		char *buff = (char*) malloc(30*sizeof(char));
		sprintf(buff, "%d", i);
		assert (i>=0);
		g_headerList->getListEnd().addOperand(buff);
	}
}
