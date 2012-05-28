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

// External includes
#include <sstream>
#include <iostream>
#include <cassert>
#include <string>

// Local includes
#include "cuobjdumpInst.h"

extern void output(const char * text);
extern void output(const std::string text);

//Constructor
cuobjdumpInst::cuobjdumpInst() {
	//initilize everything to empty
	m_label = "";
	m_predicate = new std::list<std::string>();
	m_base = "";
	m_baseModifiers = new std::list<std::string>();
	m_typeModifiers = new std::list<std::string>();
	m_operands = new std::list<std::string>();
	m_predicateModifiers = new std::list<std::string>();
}

cuobjdumpInst::~cuobjdumpInst() {
	/*
	delete m_predicate;
	delete m_baseModifiers;
	delete m_typeModifiers;
	delete m_operands;
	delete m_predicateModifiers;
	*/
}

void cuobjdumpInst::printCuobjdumpInst()
{
	/*TODO: print label here*/
	/*TODO: print predicate here*/
	/*std::cout << "Instruction Base: " << m_base << "\n";

	std::cout << "Instruction Modifiers: ";
	m_baseModifiers->printStringList();
	std::cout << "\n";

	std::cout << "Operand types: ";
	m_typeModifiers->printStringList();
	std::cout << "\n";

	std::cout << "Operands: ";
	m_operands->printStringList();
	std::cout << "\n\n";*/

	std::cout << m_base << " ";
	cuobjdumpInst::printStringList(m_baseModifiers);
	std::cout << " ";
	cuobjdumpInst::printStringList(m_typeModifiers);
	std::cout << " ";
	cuobjdumpInst::printStringList(m_operands);
	std::cout << "\n";
}

//static
void cuobjdumpInst::printStringList(std::list<std::string>* strlist) {
	for (	std::list<std::string>::iterator iter = strlist->begin();
			iter != strlist->end();
			iter++) {
		std::cout << *iter << " ";
	}
}

// Just prints the base and operands
void cuobjdumpInst::printHeaderPtx()
{
	output(m_base);
	output(" ");
	for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
			basemod != m_baseModifiers->end();
			basemod++) {
		output(" ");
		output(*basemod);
	}

	for (	std::list<std::string>::iterator operand = m_operands->begin();
			operand != m_operands->end();
			operand++) {
		output(" ");
		output(*operand);
	}
}

//retreive instruction mnemonic
const char* cuobjdumpInst::getBase()
{
	return m_base;
}

std::list<std::string>* cuobjdumpInst::getTypeModifiers()
{
	return m_typeModifiers;
}

//print out .version and .target header lines
bool cuobjdumpInst::printHeaderInst()
{
	if(strcmp(m_base, ".version")==0)
	{
		output(m_base);
		output(" ");

		std::list<std::string>::iterator operand = m_operands->begin();
		output(*operand);
		operand ++;

		if(operand != m_operands->end()) {
			output(".");
			output(*operand);
		}
      output("+");
		output("\n");
	}
	else if(strcmp(m_base, ".target")==0)
	{
		output(m_base);
		output(" ");

		std::list<std::string>::iterator operand = m_operands->begin();
		output(*operand);
		operand ++;

		while(operand != m_operands->end()) {
			output(", ");
			output(*operand);
			operand++;
		}
		output("\n");
	}
	else if(strcmp(m_base, ".tex")==0)
	{
		output(m_base); output(" ");

		std::list<std::string>::iterator curr = m_baseModifiers->begin();
		output(*curr);
		output(" ");
		curr++;

		while(curr != m_baseModifiers->end())
		{
			output(" ");
			output(*curr);
		}

		std::list<std::string>::iterator operand = m_operands->begin();
		output(*operand);
		operand++;

		while(operand != m_operands->end()) {
			output(", ");
			output(*operand);
			operand++;
		}
		output(";\n");
	}
	else
	{
		return false;
	}
	return true;
}

void cuobjdumpInst::setBase(const char* setBaseValue)
{
	m_base = setBaseValue;
}

void cuobjdumpInst::addBaseModifier(const char* addBaseMod)
{
	m_baseModifiers->push_back(addBaseMod);
}

void cuobjdumpInst::addTypeModifier(const char* addTypeMod)
{
	//We cannot have more than two modifiers, replace the last
	//This will be the case if we have memory operand modifiers
	if (m_typeModifiers->size() == 2){
		m_typeModifiers->pop_back();
	}
	m_typeModifiers->push_back(addTypeMod);
}

void cuobjdumpInst::addOperand(const char* addOp)
{
	m_operands->push_back(addOp);
}

void cuobjdumpInst::setPredicate(const char* setPredicateValue)
{
	m_predicate->push_back(setPredicateValue);
}

void cuobjdumpInst::addPredicateModifier(const char* addPredicateMod)
{
	m_predicateModifiers->push_back(addPredicateMod);
}

void cuobjdumpInst::setLabel(const char* setLabelValue)
{
	m_label = setLabelValue;
}

bool cuobjdumpInst::checkCubojdumpLabel(std::list<std::string> labelList, std::string label)
{
	if(labelList.empty())
		return false;

	std::list<std::string>::iterator labelIterator;

	for( labelIterator=labelList.begin(); labelIterator!=labelList.end(); labelIterator++ )
	{
		if(label.compare(*labelIterator) == 0)
			return true;
	}

	return false;
}

void cuobjdumpInst::printCuobjdumpLabel(std::list<std::string> labelList)
{
	if((strcmp(m_label, "")!=0)&&(checkCubojdumpLabel(labelList, m_label))) {
		output(m_label);
		output(": ");
	}
}

void cuobjdumpInst::printCuobjdumpPredicate()
{
	std::list<std::string>::iterator pred = m_predicate->begin();
	if(pred != m_predicate->end())
	{
		output("@$p");
		output((*pred).substr(1,1));
		for (	std::list<std::string>::iterator predmod = m_predicateModifiers->begin();
				predmod != m_predicateModifiers->end();
				predmod++) {
			std::string modString3 = *predmod;

			for(unsigned i=0; i<modString3.length(); i++)
			{
				modString3[i] = tolower(modString3[i]);
			}
			if(modString3 ==".not_sign") {
				output(".nsf");
			} else if(modString3 == ".sign") {
				output(".sf");
			} else if(modString3 == ".carry") {
				output(".cf");
			} else {
				output(modString3);
			}
		}
		output(" ");
	}
}

void cuobjdumpInst::printCuobjdumpTypeModifiers()
{
	for (	std::list<std::string>::iterator typemod = m_typeModifiers->begin();
			typemod != m_typeModifiers->end();
			typemod++) {
		if (*typemod ==  ".F16")
			output(".f16");
		else if(*typemod == ".F32")
			output(".f32");
		else if(*typemod == ".F64"){
			if(		strcmp(m_base, "F2I") == 0||
					strcmp(m_base, "F2F") == 0)
				output(".f64");
			else
				output(".ff64");
		}
		else if(*typemod == ".S8")
			output(".s8");
		else if(*typemod == ".S16")
			output(".s16");
		else if(*typemod == ".S32")
			output(".s32");
		else if(*typemod == ".S64")
			output(".bb64"); //TODO: might have to change to .ss64 in the future.
		else if(*typemod == ".S128")
			output(".bb128"); //TODO: might have to change to .ss64 in the future.
		else if(*typemod == ".U8")
			output(".u8");
		else if(*typemod == ".U16")
			output(".u16");
		else if(*typemod == ".U32")
			output(".u32");
		else if(*typemod == ".U64")
			output(".bb64"); //TODO: might have to change to .ss64 in the future.
		else if(*typemod == ".HI")
			output(".hi");
		else
		{
			printf("Unknown Type: ");
			printf((*typemod).c_str());
			printf("\n");
			output("Unknown Type: ");
			output(*typemod);
			assert(0);
		}
	}
}

void cuobjdumpInst::printCuobjdumpBaseModifiers()
{
	for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
			basemod != m_baseModifiers->end();
			basemod++)
	{
		if( *basemod ==  "EQ")
			output(".eq");
		else if( *basemod == "EQU")
			output(".equ");
		else if( *basemod == "GE")
			output(".ge");
		else if( *basemod == "GEU")
			output(".geu");
		else if( *basemod == "GT")
			output(".gt");
		else if( *basemod == "GTU")
			output(".gtu");
		else if( *basemod == "LE")
			output(".le");
		else if( *basemod == "LEU")
			output(".leu");
		else if( *basemod == "LT")
			output(".lt");
		else if( *basemod == "LTU")
			output(".ltu");
		else if( *basemod == "NE")
			output(".ne");
		else if( *basemod == "NEU")
			output(".neu");
		else if( *basemod == ".abs")
		{
			if((strcmp(m_base, "F2F")!=0) && (strcmp(m_base, "I2I")!=0))
			{
				output(*basemod);
			}
		}
		else if(	(*basemod == "ex2") ||
					(*basemod == ".exit") ||
					(*basemod == "sin") ||
					(*basemod == "cos") ||
					(*basemod == ".rz") ||
					(*basemod == ".rp") ||
					(*basemod == ".rm") ||
					(*basemod == ".any") ||
					(*basemod == ".all") )
				output(*basemod);
		else if( *basemod == ".bext")
		{
			//".bext" is a modifier that indicated u8 to u16 type conversion, I think
		}
		else if( *basemod == ".s")
		{
			//".s" is the same as ".join" in cuobjdump.
		}
		else if( *basemod == ".sfu")
		{
			//".sfu" is an unknown base modifier, TODO: find out what it is
		}
		else if( *basemod == ".x")
		{
			//".x" is an unknown base modifier, TODO: find out what it is
			output(*basemod);
		}
		else if( *basemod == ".e")
		{
			//".e" is an unknown base modifier, TODO: find out what it is
			output(*basemod);
		}
		else if( *basemod == ".ir")
		{
			//".ir" is an unknown base modifier, TODO: find out what it is
			output(*basemod);
		}
		else if((*basemod == "IADD") ||
				(*basemod == "IMIN") ||
				(*basemod == "IMAX"))
		{
			/*
			 * This is the case of a GRED or GATOM operation
			 */
			output(".");
			std::string modstr = *basemod;
			modstr = modstr.substr(1);
			for (unsigned i=0; i<modstr.length(); i++){
				modstr[i] = tolower(modstr[i]);
			}
			output(modstr.c_str());
		}
		else
		{
			printf("Unknown Base Mod: ");
			printf((*basemod).c_str());
			printf("\n");
			output("Unknown Base Mod: ");
			output(*basemod);
			assert(0);
		}
	}
}

/*
 * Remove the trailing 'l' or 'h' and output the operand followed by ".lo" or ".hi" respectively
 */
void cuobjdumpInst::printCuobjdumpOperandlohi(std::string op) {
	if (op.substr(op.length()-1) == "l") {
		output(op.substr(0, op.length()-1).c_str());
		output(".lo");
	} else if (op.substr(op.length()-1) == "h"){
		output(op.substr(0, op.length()-1).c_str());
		output(".hi");
	} else {
		output(op.c_str());
	}
}

void cuobjdumpInst::printCuobjdumpOperand(std::string currentPiece, std::string operandDelimiter, const char* base)
{

	output(operandDelimiter);
	output(" ");

	//Current piece
	std::string currp = currentPiece;
	std::string mod;

	if(currp[0] == '-') {
		mod = currp.substr(1);

		if(mod.substr(0,2) == "0x") {
			unsigned immValue;
			std::stringstream hexStringConvert2;
			hexStringConvert2 << std::hex << mod;
			hexStringConvert2 >> immValue;

			if(immValue){
				immValue = ~immValue + 1;
			}

			std::stringstream outputhexstream;
			outputhexstream << std::hex << immValue;
			std::string outputhex;
			outputhex = outputhexstream.str();
			output("0x");
			for(unsigned i=8; i > outputhex.length(); i--)
			{
				output("0");
			}
			output(outputhex.c_str());
			return;
		}

		output("-");
	}
	else
	{
		mod = std::string(currp);
	}

	// Make it lower case
	if(mod.substr(0,9)!= "constant1")
	for(unsigned i=0; i<mod.length(); i++)
	{
		mod[i] = tolower(mod[i]);
	}

	//double destination
	if(mod[0]=='.') //double destination
	{
		std::string temp = mod.substr(2,2);
		output("$p");
		output(mod.substr(2,2).c_str());

		mod = mod.substr(4);
	}


	if(mod == "g [0x1].u16") { //handling special register case: %ntid.x
		output("%%ntid.x");
	} else if(mod == "g [0x2].u16") { //handling special register case: %ntid.y
		output("%%ntid.y");
	} else if(mod == "g [0x3].u16") { //handling special register case: %ntid.x
		output("%%ntid.z");
	} else if(mod =="g [0x4].u16") { //handling special register case: %nctaid.x
		output("%%nctaid.x");
	} else if(mod =="g [0x5].u16") { //handling special register case: %nctaid.y
		output("%%nctaid.y");
	} else if(mod == "g [0x6].u16") {//handling special register case: %ctaid.x
		output("%%ctaid.x");
	} else if(mod == "g [0x7].u16") {//handling special register case: %ctaid.y
		output("%%ctaid.y");
	} else if(mod == "sr1") {//handling special register case: %clock
		output("%%clock");
	} else if(mod[0]=='r') { //basic register
		if(	(strcmp(m_base, "DADD")==0) ||
			(strcmp(m_base, "DMUL")==0) ||
			(strcmp(m_base, "DFMA")==0) ||
			(	(m_typeModifiers->size()==1) &&
				(m_typeModifiers->front() == ".S64") &&
				(	(strcmp(m_base, "G2R")==0) ||
					(strcmp(m_base, "R2G")==0) ||
					(strcmp(m_base, "GLD")==0) ||
					(strcmp(m_base, "GST")==0) ||
					(strcmp(m_base, "LST")==0) ))) {
			std::string modsub = mod.substr(1);
			int regNumInt = atoi(modsub.c_str());
			std::stringstream temp;
			temp << "{$r" << (regNumInt) << ",$r"<< (regNumInt+1) << "}";
			output(temp.str().c_str());
		} else if(	(m_typeModifiers->size()==1) &&
					(m_typeModifiers->front() == ".S128")) {
			std::string modsub = mod.substr(1);
			int regNumInt = atoi(modsub.c_str());
			std::stringstream temp;
			temp << "{$r" << (regNumInt);
			temp << ",$r" << (regNumInt+1);
			temp << ",$r" << (regNumInt+2);
			temp << ",$r" << (regNumInt+3) << "}";
			output(temp.str().c_str());
		} else {
			output("$");
			printCuobjdumpOperandlohi(mod);
		}
	} else if(mod[0] == 'c' && mod.length() == 2) { //predicate register (conditional code)
		output("$p");
		output(mod.substr(1,1).c_str());
	} else if(mod[0]=='a') {//offset register
		output("$ofs");
		mod = mod.substr(1);
		printCuobjdumpOperandlohi(mod);
	} else if(mod[0]=='o') {//output register
		output("$o127");
	} else if (	mod[0]=='g' ||
				mod.substr(0,2) == "lo" ||
				mod.substr(0,8) == "constant") {
		//memory operands, global14 = global, g = shared, c = constant
		std::string modsub;
		std::string modsub2;
		std::string modsub3;
		modsub = mod.c_str();
		int localFlag = 0;

		if(mod.find("global14") != std::string::npos) {
			//Those instructions don't need the dereferencing done by g [*]
			if(	strcmp(base,"GRED")==0 ||
				strcmp(base, "GATOM")==0 ||
				strcmp(base, "GST")==0 ||
				strcmp(base, "GLD")==0)
				output("[");
			else
				output("g[");
		} else if(mod[0]=='g') {
			//Shared memory
			output("s[");
		} else if(mod.find("local") !=  std::string::npos) {
			modsub3 = modsub.substr(4, modsub.length()-10);
			output(modsub3.c_str());
			output("[");
			localFlag = 1;
		} else if(mod.substr(0,9) == "constant1") {
			output(modsub.substr(0, modsub.find_first_of("[]")+1).c_str());
		} else if(mod.substr(0,9)=="constant0"){
			output("constant0[");
		} else {
			printf("Unidentified modifier: %s\n", mod.c_str());
			assert(0);
		}

		modsub = modsub.substr(modsub.find_first_of("[]")+1);
		modsub = modsub.substr(0, modsub.length()-1);
		//Here we handle whatever was inside the []
		int plusequalFlag = 0;
		if(modsub.find("+++") !=  std::string::npos) {
			plusequalFlag = 1;
		}

		if(modsub.find("+") !=  std::string::npos)
		{
			//Handle a1+++0x1 or a1+0x1
			modsub2 = modsub.substr(modsub.find("+")+1);

			output("$ofs");
			output(modsub.substr(1,1).c_str());

			if(plusequalFlag == 1) {
				output("+=");
			} else {
				output("+");
			}


			if(modsub2[0]=='r') {
				output("$");
				printCuobjdumpOperandlohi(modsub2);
			} else {
				modsub2 = modsub2.substr(2);
				unsigned int addrValue;
				std::stringstream hexStringConvert;
				hexStringConvert << std::hex << modsub2;
				hexStringConvert >> addrValue;
				unsigned chunksize = 4;
				if (	this->m_typeModifiers->size()>0 &&
						(	m_typeModifiers->back() == ".S16" ||
							m_typeModifiers->back() == ".U16")) chunksize = 2;
				if (	this->m_typeModifiers->size()>0 &&
						(	m_typeModifiers->back() == ".U8" ||
							m_typeModifiers->back() == ".S8")) chunksize = 1;
				addrValue = addrValue*chunksize;
				char outputHex[10];
				sprintf(outputHex, "%x", addrValue);
				std::stringstream outputhex;
				outputhex << std::hex << addrValue;
				output("0x");
				for(unsigned i=4; i > outputhex.str().length(); i--)
				{
					output("0");
				}
				output(outputhex.str().c_str());
			}
		} else {
			if(modsub[0]=='r') {
				//Register
				output("$");
				printCuobjdumpOperandlohi(modsub);
			} else {
				// Immediate offset
				mod = mod.substr(2);

				unsigned int addrValue;
				std::stringstream hexStringConvert;
				hexStringConvert << std::hex << modsub;
				hexStringConvert >> addrValue;

				if(localFlag == 0)
				{
					unsigned chunksize = 4;
					if ( m_typeModifiers->size()>0 &&
							(	m_typeModifiers->back() == ".S16" ||
								m_typeModifiers->back() == ".U16")) chunksize = 2;
					if (	this->m_typeModifiers->size()>0 &&
							(	m_typeModifiers->back() == ".U8" ||
								m_typeModifiers->back() == ".S8")) chunksize = 1;
					addrValue = addrValue*chunksize;
				}
				std::stringstream outputhex;
				outputhex << std::hex << addrValue;
				output("0x");
				for(unsigned i=4; i > outputhex.str().length(); i--) {
					output("0");
				}
				output(outputhex.str().c_str());
			}
		}
		output("]");
	} else if(mod.substr(0,2) == "0x") { //immediate value
		output("0x");
		std::string outputhex;
		outputhex = mod.substr(2);

		for(unsigned i=8; i > outputhex.length(); i--) {
			output("0");
		}
		output(outputhex.c_str());
	} else if(mod.substr(0,3) == "l0x") { //label
		output(mod.c_str());
	} else {//variable name
		output(mod.c_str());
	}
}

void cuobjdumpInst::printCuobjdumpOperands()
{
	std::string delimiter = "";
	unsigned i=0;
	for (	std::list<std::string>::iterator operand = m_operands->begin();
			operand != m_operands->end();
			operand++, i++) {
		if((strcmp(m_base, "LOP.PASS_B")==0 || strcmp(m_base, "LOP.S.PASS_B")==0) && (i==1)) {
			continue;
		}
		printCuobjdumpOperand(*operand, delimiter, m_base);
		delimiter = ",";
	}
}

std::string int_default_mod () { return ".u32" ;}


std::string breaktarget;

void cuobjdumpInst::printCuobjdumpPtxPlus(std::list<std::string> labelList, std::list<std::string> texList)
{
	printCuobjdumpLabel(labelList);

	if(strcmp(m_base, "")==0)
	{
	}
	else if(strcmp(m_base, ".entry")==0)
	{
		/*do nothing here*/
	}
	else if(strcmp(m_base, "BAR.ARV.WAIT b0, 0xfff")==0)
	{
		printCuobjdumpPredicate();
		output("bar.sync 0x00000000;");
	}
	else if(strcmp(m_base, "ADA")==0)
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "BRA")==0)
	{
		printCuobjdumpPredicate();
		output("bra");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "CAL")==0)
	{
		printCuobjdumpPredicate();
		output("callp");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "COS")==0)
	{
		printCuobjdumpPredicate();
		//output("nop;");
		//output(" //cos");
		output("cos");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "DADD")==0)
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".ff64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "DMIN")==0)
	{
		printCuobjdumpPredicate();
		output("min");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "DMAX")==0)
	{
		printCuobjdumpPredicate();
		output("max");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}


	/*
	else if(strcmp(m_base, "DFMA")==0)
	{
		printCuobjdumpPredicate();
		output("mad");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".ff64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	*/
	else if(strcmp(m_base, "DMUL")==0)
	{
		printCuobjdumpPredicate();
		output("mul");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".ff64"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "EX2")==0)
	{
		printCuobjdumpPredicate();
		//output("nop;");
		//output(" //ex2");
		output("ex2");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "F2F")==0)
	{
		printCuobjdumpPredicate();

		int absFlag = 0;

		for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
				basemod != m_baseModifiers->end();
				basemod++){
			if( *basemod ==  ".abs")
			{
				output("abs");
				absFlag = 1;
				break;
			}
		}
		if(absFlag == 0)
		{
			output("cvt");
		}

		printCuobjdumpBaseModifiers();

		if(absFlag == 0)
		{
			printCuobjdumpTypeModifiers();
		}
		else
		{
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "F2I")==0)
	{
		printCuobjdumpPredicate();
		output("cvt");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		//output(".f64.s32");
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "FADD")==0 || strcmp(m_base, "FADD32I")==0)
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "FADD32")==0)
	{
		printCuobjdumpPredicate();
		output("add.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "FMAD")==0 || strcmp(m_base, "FMAD32I")==0)
	{
		printCuobjdumpPredicate();
		output("mad");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "FMUL")==0 || strcmp(m_base, "FMUL32I")==0)
	{
		printCuobjdumpPredicate();
		output("mul");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "FMUL32")==0)
	{
		printCuobjdumpPredicate();
		output("mul.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "FSET")==0)
	{
		printCuobjdumpPredicate();
		output("set");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".f32.f32");
		}
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "DSET")==0)
	{
		printCuobjdumpPredicate();
		output("set");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".f64.f64");
		}
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "G2R")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();

		if( m_typeModifiers->size() == 2 ) {
			std::string type1, type2, type;
			int type1Size, type2Size;
			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size < type2Size) ? type1 : type2;

			if( strcmp(type.c_str(), ".F16")==0 )
				output(".f16");
			else if( strcmp(type.c_str(), ".F32")==0 )
				output(".f32");
			else if( strcmp(type.c_str(), ".F64")==0 )
				output(".ff64");
			else if( strcmp(type.c_str(), ".S8")==0 )
				output(".s8");
			else if( strcmp(type.c_str(), ".S16")==0 )
				output(".s16");
			else if( strcmp(type.c_str(), ".S32")==0 )
				output(".s32");
			else if( strcmp(type.c_str(), ".S64")==0 )
				output(".bb64");
			else if( strcmp(type.c_str(), ".S128")==0 )
				output(".bb128");
			else if( strcmp(type.c_str(), ".U8")==0 )
				output(".u8");
			else if( strcmp(type.c_str(), ".U16")==0 )
				output(".u16");
			else if( strcmp(type.c_str(), ".U32")==0 )
				output(".u32");
			else if( strcmp(type.c_str(), ".U64")==0 )
				output(".bb64");
			else
				output(type.c_str());
		} else if( m_typeModifiers->size() == 1 ) {
			printCuobjdumpTypeModifiers();
		} else {
			output("Error: unsupported number of type modifiers. ");
                }
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "GLD")==0)
	{
		printCuobjdumpPredicate();
		//output("mov");
		output("ld.global");
		printCuobjdumpBaseModifiers();
		if (m_typeModifiers->front() == ".S128") {
			output(".v4.u32");
		} else if (m_typeModifiers->front() == ".S64") {
			output(".v2.u32");
		} else {
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "GST")==0)
	{
		printCuobjdumpPredicate();
		//output("mov");
		output("st.global");
		printCuobjdumpBaseModifiers();
		if (m_typeModifiers->front() == ".S128") {
			output(".v4.u32");
		} else if (m_typeModifiers->front() == ".S64") {
			output(".v2.u32");
		} else {
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "I2F")==0)
	{
		printCuobjdumpPredicate();
		output("cvt");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "I2I")==0)
	{
		printCuobjdumpPredicate();

		int absFlag = 0;

		for (	std::list<std::string>::iterator basemod = m_baseModifiers->begin();
				basemod != m_baseModifiers->end();
				basemod++) {
			if(*basemod == ".abs")
			{
				output("abs");
				absFlag = 1;
				break;
			}
		}
		if(absFlag == 0)
		{
			output("cvt");
		}

		printCuobjdumpBaseModifiers();


		if(m_typeModifiers->size() == 0 || absFlag)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();


		printCuobjdumpOperands();
		output(";");
	}
	else if(strstr(m_base, "IADD.CARRY")){
		std::string pred = "C0";
		pred[1] = m_base[10];
		this->addOperand(pred.c_str());
		printCuobjdumpPredicate();
		output("addp");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IADD")==0)
	{
		printCuobjdumpPredicate();
		output("add");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IADD32")==0  || strcmp(m_base, "IADD32I")==0 )
	{
		printCuobjdumpPredicate();
		output("add.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IMAD32I")==0  || strcmp(m_base, "IMAD32")==0)
	{
		printCuobjdumpPredicate();
		output("mad.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IMAD")==0)
	{
		printCuobjdumpPredicate();
		output("mad.wide");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "ISAD")==0)
	{
		printCuobjdumpPredicate();
		output("sad");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}

	else if(strcmp(m_base, "IMAD.U24")==0)
	{
		printCuobjdumpPredicate();
		output("mad24.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
        }
	else if(strcmp(m_base, "IMAD.S24")==0)
	{
		printCuobjdumpPredicate();
		output("mad24.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".s32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IMUL32I.U24.U24")==0 ||
			strcmp(m_base, "IMUL32I.S24.S24")==0 ||
			strcmp(m_base, "IMUL32.U24.U24")==0 ||
			strcmp(m_base, "IMUL32.S24.S24")==0 ||
			strcmp(m_base, "IMUL.U24.U24")==0 ||
			strcmp(m_base, "IMUL.S24.S24")==0)
	{
		printCuobjdumpPredicate();
		output("mul24.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
				output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);
		}	
		else
		{
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IMUL")==0 || strcmp(m_base, "IMUL32I")==0)
	{
		printCuobjdumpPredicate();
		output("mul.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;

			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();


			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
					output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);
		}	
		else
		{
			printCuobjdumpTypeModifiers();
		}
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IMUL32")==0)
	{
		printCuobjdumpPredicate();
		output("mul.half.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
				output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);

		}	
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "IMUL32.S24.S24")==0)
	{
		printCuobjdumpPredicate();
		output("mul24.half.lo");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
		{
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		}
		else if(m_typeModifiers->size() == 2)
		{
			std::string type1, type2, type;
			int type1Size, type2Size;
			char tempString[5];

			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size > type2Size) ? type1 : type2;
			strcpy(tempString, type.c_str());
			/*if(type1Size==16 && type2Size==16)
				output(".lo");*/
			if(tempString[1] >= 'A' && tempString[1] <= 'Z')
				tempString[1] += 32;			
			output(tempString);

		}	
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "ISET")==0)
	{
		printCuobjdumpPredicate();
		output("set");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32.u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else if(m_typeModifiers->size() == 1)
		{
			printCuobjdumpTypeModifiers(); printCuobjdumpTypeModifiers();
		}
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LG2")==0)
	{
		printCuobjdumpPredicate();
		output("lg2");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LST")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "MVC")==0){
		printCuobjdumpPredicate();
		//Use cvt if there is conversion involved (2 modifiers) otherwise mov
		if(m_typeModifiers->size() < 2)
			output("mov");
		else
			output("cvt");

		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "MOV")==0 || strcmp(m_base, "MVI")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "MOV32")==0)
	{
		printCuobjdumpPredicate();
		output("mov.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".u32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "NOP")==0 || strcmp(m_base, "SSY")==0)
	{
		printCuobjdumpPredicate();
		output("nop");
		printCuobjdumpBaseModifiers();
		output(";");
	}
	else if(strcmp(m_base, "LLD")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LOP.AND")==0 || strcmp(m_base, "LOP.S.AND")==0)
	{
		printCuobjdumpPredicate();
		output("and");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LOP.OR")==0 || strcmp(m_base, "LOP.S.OR")==0)
	{
		printCuobjdumpPredicate();
		output("or");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LOP.PASS_B")==0 || strcmp(m_base, "LOP.S.PASS_B")==0)
	{
		printCuobjdumpPredicate();
		output("not");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LOP.XOR")==0 || strcmp(m_base, "LOP.S.XOR")==0)
	{
		printCuobjdumpPredicate();
		output("xor");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "R2A")==0)
	{
		printCuobjdumpPredicate();
		output("shl");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".b32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		if(m_operands->size() == 2)
			output(", 0x0");
		output(";");
	}
	else if(strcmp(m_base, "R2G.U16.U8")==0){
		/*
		 * This code handles a cuobjdump bug that causes the wrong register number to be printed
		 */
		printCuobjdumpPredicate();
		output("mov.u8");
		std::list<std::string>::iterator operand = m_operands->begin();
		std::string delimiter = "";
		printCuobjdumpOperand(*operand, delimiter, m_base);
		operand++;
		output(", ");
		std::string curr = *operand;
		curr = curr.substr(1);
		int regnum;
		std::istringstream(curr)>>regnum;
		output("$r");
		std::stringstream finalregnum;
		finalregnum << (regnum/2);
		output(finalregnum.str().c_str());
		output( regnum%2==0? ".lo": ".hi");
		output(";");
	}
	else if(strcmp(m_base, "R2G")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();

		if( m_typeModifiers->size() == 2 ) {
			std::string type1, type2, type;
			int type1Size, type2Size;
			std::list<std::string>::iterator curr = m_typeModifiers->begin();
			type1 = (*curr).c_str();
			curr++;
			type2 = (*curr).c_str();


			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size < type2Size) ? type1 : type2;

			if( strcmp(type.c_str(), ".F16")==0 )
				output(".f16");
			else if( strcmp(type.c_str(), ".F32")==0 )
				output(".f32");
			else if( strcmp(type.c_str(), ".F64")==0 )
				output(".ff64");
			else if( strcmp(type.c_str(), ".S8")==0 )
				output(".s8");
			else if( strcmp(type.c_str(), ".S16")==0 )
				output(".s16");
			else if( strcmp(type.c_str(), ".S32")==0 )
				output(".s32");
			else if( strcmp(type.c_str(), ".S64")==0 )
				output(".bb64");
			else if( strcmp(type.c_str(), ".S128")==0 )
				output(".bb128");
			else if( strcmp(type.c_str(), ".U8")==0 )
				output(".u8");
			else if( strcmp(type.c_str(), ".U16")==0 )
				output(".u16");
			else if( strcmp(type.c_str(), ".U32")==0 )
				output(".u32");
			else if( strcmp(type.c_str(), ".U64")==0 )
				output(".bb64");
			else
				output(type.c_str());
		} else if( m_typeModifiers->size() == 1 ) {
			printCuobjdumpTypeModifiers();
		} else {
			output("Error: unsupported number of type modifiers. ");
                }

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "RCP")==0)
	{
		printCuobjdumpPredicate();
		output("rcp");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "RCP32")==0)
	{
		printCuobjdumpPredicate();
		output("rcp.half");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32");
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "RET")==0)
	{
		printCuobjdumpPredicate();
		output("retp;");
	}
	else if(strcmp(m_base, "RRO")==0)
	{
		output("nop; //");
		printCuobjdumpPredicate();
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "RSQ")==0)
	{
		printCuobjdumpPredicate();
		output("rsqrt");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "SHL")==0)
	{
		printCuobjdumpPredicate();
		output("shl");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "SHR")==0)
	{
		printCuobjdumpPredicate();
		output("shr");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "SIN")==0)
	{
		printCuobjdumpPredicate();
		output("sin");
		printCuobjdumpBaseModifiers();

		if(m_typeModifiers->size() == 0)
			output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "S2R")==0)
	{
		printCuobjdumpPredicate();
		output("cvt");
		printCuobjdumpBaseModifiers();
		output(".u32.u32");
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "LD")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "STS")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "BAR")==0)
	{
		printCuobjdumpPredicate();
		output("bar.sync 0;");
	}
	else if(strcmp(m_base, "LDS")==0)
	{
		// If there is not global address space that includes shared memory, then fix this.
		// Same for STS
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "ST")==0)
	{
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	} else if(strcmp(m_base, "IMIN")==0) {
		printCuobjdumpPredicate();
		output("min");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(".s32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	} else if(strcmp(m_base, "IMAX")==0) {
		printCuobjdumpPredicate();
		output("max");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(".s32"); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	} else if(strcmp(m_base, "FMIN")==0) {
			printCuobjdumpPredicate();
			output("min");
			printCuobjdumpBaseModifiers();
			if(m_typeModifiers->size() == 0)
				output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
			else
				printCuobjdumpTypeModifiers();
			printCuobjdumpOperands();
			output(";");
	}
	else if(strcmp(m_base, "FMAX")==0) {
			printCuobjdumpPredicate();
			output("max");
			printCuobjdumpBaseModifiers();
			if(m_typeModifiers->size() == 0)
				output(".f32"); //TODO: setting default type modifier but I'm not sure if this is right.
			else
				printCuobjdumpTypeModifiers();
			printCuobjdumpOperands();
			output(";");
	}
	else if(strcmp(m_base, "A2R")==0) {
		printCuobjdumpPredicate();
		output("mov");
		printCuobjdumpBaseModifiers();
		if(m_typeModifiers->size() == 0)
			output(int_default_mod()); //TODO: setting default type modifier but I'm not sure if this is right.
		else
			printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if((strcmp(m_base, "TEX")==0) ||
			(strcmp(m_base, "TEX32")==0)) {
		printCuobjdumpPredicate();
		output("tex.1d.v4.f32.s32 ");
		std::string addrReg, tex_id;
		std::list<std::string>::iterator operand = m_operands->begin();
		output("{");
		printCuobjdumpOperand(*operand, "", "");
		output(",_,_,_} , ");
		std::string reg = *operand;
		operand++;
		tex_id = *operand;
		unsigned int tex_id_int;
		std::stringstream ss;
		ss << std::hex << tex_id;
		ss >> tex_id_int;
		std::list<std::string>::iterator texIter = texList.begin();
		for (unsigned i=0; i<tex_id_int; i++) {
			assert (texIter!=texList.end());
			texIter++;
		}
		output((*texIter).c_str());
		output(",{");
		printCuobjdumpOperand(reg, "", "");
		output(",_,_,_};");
	}
	else if(strcmp(m_base, "EXIT")==0) {
		printCuobjdumpPredicate();
		output("exit");
		output(";");
	}
	else if(strcmp(m_base, "GRED")==0) {
		printCuobjdumpPredicate();
		// ptx instruction atom can be used to perform reduction using destination register '_'
		output("atom.global");
		printCuobjdumpBaseModifiers();
			printCuobjdumpTypeModifiers();
		output(" _, ");
		printCuobjdumpOperands();
		output(";");
	} else if(strcmp(m_base, "GATOM")==0) {
		printCuobjdumpPredicate();
		output("atom.global");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();

		printCuobjdumpOperands();
		output(";");
	}else if(strcmp(m_base, "PBK")==0) {
		//PKB specifies the target of later BRK (break) instructions
		// Here we convert it to nop and store the target
		output("nop;");
		breaktarget = m_operands->front();
	} else if(strcmp(m_base, "BRK")==0) {
		printCuobjdumpPredicate();
		/*
		 * Convert it **at compile time** to a branch to the break target saved earlier
		 * Correct operation not guaranteed
		 */

		output("bra ");
		output(breaktarget.c_str());
		output(";");
	} else if(strcmp(m_base, "C2R")==0) {
		printCuobjdumpPredicate();
		output("mov.u32");
		printCuobjdumpOperands();
		output(";");
	} else if(strcmp(m_base, "R2C")==0) {
		printCuobjdumpPredicate();
		output("mov.pred");
		printCuobjdumpOperands();
		output(";");
	} else if(strcmp(m_base, "VOTE")==0) {
		printCuobjdumpPredicate();
		output("vote");
		printCuobjdumpBaseModifiers();
		printCuobjdumpTypeModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else if(strcmp(m_base, "DFMA")==0)
	{
		printCuobjdumpPredicate();
		output("fma.rz.ff64");
		printCuobjdumpBaseModifiers();
		printCuobjdumpOperands();
		output(";");
	}
	else
	{
		printf("Unknown Instruction: ");
		printf(m_base);
		printf("\n");
		output("Unknown Instruction: ");
		output(m_base);
		assert(0);
	}

}

