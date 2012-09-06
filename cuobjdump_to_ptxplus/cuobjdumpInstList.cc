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

#include <sstream>
#include <iostream>
#include <cassert>
#include "cuobjdumpInstList.h"

#define P_DEBUG 0
#define DPRINTF(...) \
   if(P_DEBUG) { \
      printf("(%s:%u) ", __FILE__, __LINE__); \
      printf(__VA_ARGS__); \
      printf("\n"); \
      fflush(stdout); \
   }

extern void output(const char * text);

//Constructor
cuobjdumpInstList::cuobjdumpInstList()
{
        //initilize everything to empty
}



// add to tex list
void cuobjdumpInstList::addTex(std::string tex)
{
	std::string origTex = tex;
	DPRINTF("cuobjdumpInstList::addTex tex=%s", tex.c_str());
	// If $tex# tex from cuobjdump, then use index to get real tex name
	if(tex.substr(0, 4) == "$tex") {
		tex = tex.substr(4, tex.size()-4);
		unsigned texNum = atoi(tex.c_str());
		if(texNum >= m_realTexList.size()) {
			output("ERROR: tex does not exist in real tex list from ptx.\n.");
			assert(0);
		}

		std::list<std::string>::iterator itex = m_realTexList.begin();
		for(unsigned i=0; i<texNum; i++) itex++;
		origTex = *itex;
	}
	// Otherwise, tex from original ptx
	else {
		m_realTexList.push_back(tex);
	}

	// Add the tex to instruction operand list
	//char* texName = new char [strlen(origTex.c_str())+1];
	//strcpy(texName, origTex.c_str());
	//getListEnd().addOperand(texName);
}

void cuobjdumpInstList::setLastEntryName(std::string entryName)
{
	m_entryList.back().m_entryName = entryName;
}

// create new global constant memory "bank"
void cuobjdumpInstList::addConstMemory(int index)
{
	constMemory newConstMem;
	newConstMem.index = index;
	newConstMem.entryIndex = 0;
	m_constMemoryList.push_back(newConstMem);
}

//add cuobjdumpInst to the last entry in entry list
int cuobjdumpInstList::add(cuobjdumpInst* newCuobjdumpInst)
{
	if(m_entryList.size() == 0) {
		//output("ERROR: Adding an instruction before entry.\n");
		addEntry("");
		//assert(0);
	}

	m_entryList.back().m_instList.push_back(*newCuobjdumpInst);

	return m_entryList.size();
}

// add a new entry
int cuobjdumpInstList::addEntry(std::string entryName)
{
	cuobjdumpEntry newEntry;
	newEntry.m_largestRegIndex = -1;
	newEntry.m_largestOfsRegIndex = -1;
	newEntry.m_largestPredIndex = -1;
	newEntry.m_reg124 = false;
	newEntry.m_oreg127 = false;
	newEntry.m_lMemSize = -1;

	newEntry.m_entryName = entryName;


   // Fill opPerCycle histogram with values
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_1", 0) );
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_2", 0) );
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_8", 0) );


	m_entryList.push_back(newEntry);
	return m_entryList.size();
}

// print out .version and .target headers
void cuobjdumpInstList::printHeaderInstList()
{
	// These should be in the first entry
	cuobjdumpEntry e_first = m_entryList.front();

	std::list<cuobjdumpInst>::iterator currentInst;
	for(currentInst=e_first.m_instList.begin(); currentInst!=e_first.m_instList.end(); ++currentInst)
	{
		if(!(currentInst->printHeaderInst()))
		{
			break;
		}
	}
	for (	std::list<std::string>::iterator iter = m_realTexList.begin();
			iter != m_realTexList.end();
			iter ++) {
		output(".tex .u64 ");
		output((*iter).c_str());
		output(";\n");
	}
}

bool cuobjdumpInstList::findEntry(std::string entryName, cuobjdumpEntry& entry) {
	std::list<cuobjdumpEntry>::iterator e;

	std::string entryNameS = entryName;

	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		if( e->m_entryName == entryNameS) {
			entry = *e;
			return true;
		}
	}

	return false;
}

// get the list of real tex names
std::list<std::string> cuobjdumpInstList::getRealTexList() {
	return m_realTexList;
}

// set the list of real tex names
void cuobjdumpInstList::setRealTexList(std::list<std::string> realTexList) {
	m_realTexList = realTexList;
}

// add value to const memory
void cuobjdumpInstList::addConstMemoryValue(std::string constMemoryValue)
{
	m_constMemoryList.back().m_constMemory.push_back(constMemoryValue);
}

void cuobjdumpInstList::addConstMemoryValue2(std::string constMemoryValue)
{
	m_constMemoryList2.back().m_constMemory.push_back(constMemoryValue);
}

// set type of constant memory
void cuobjdumpInstList::setConstMemoryType(const char* type)
{
	m_constMemoryList.back().type = type;
}

void cuobjdumpInstList::setConstMemoryType2(const char* type)
{
	m_constMemoryList2.back().type = type;
}

//retrieve point to list end
cuobjdumpInst cuobjdumpInstList::getListEnd()
{
	return m_entryList.back().m_instList.back();
}

// print out predicate names
void cuobjdumpInstList::printPredNames(cuobjdumpEntry entry)
{
	if( entry.m_largestPredIndex >= 0) {
		char out[30];
		// there is at least 4 predicates for GT200, possibly more in Fermi 
		sprintf(out, "\t.reg .pred $p<%d>;", std::max(entry.m_largestPredIndex+1, 4)); 
		output(out);
		output("\n");
	}

}

// print reg124 and set its value to 0
void cuobjdumpInstList::printOutOfBoundRegisters(cuobjdumpEntry entry)
{
	if( entry.m_reg124 == true ) {
		output("\n");
		output("\t.reg .u32 $r124;\n");
	//	output("\tmov.u32 $r124, 0x00000000;\n");
	}
	if( entry.m_oreg127 == true) {
		output("\n");
		output("\t.reg .u32 $o127;\n");
	}
}

// print out register names
void cuobjdumpInstList::printRegNames(cuobjdumpEntry entry)
{
	if( entry.m_largestRegIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .u32 $r<%d>;", entry.m_largestRegIndex+1);
		output(out);
		output("\n");
	}

	if( entry.m_largestOfsRegIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .u32 $ofs<%d>;", entry.m_largestOfsRegIndex+1);
		output(out);
		output("\n");
	}
}

// print const memory directive
void cuobjdumpInstList::printMemory()
{

	// Constant memory

	for(std::list<constMemory>::iterator i=m_constMemoryList.begin(); i!=m_constMemoryList.end(); ++i) {
		char line[40];

		// Global or entry specific
		if(i->entryIndex == 0)
			sprintf(line, ".const %s constant0[%d] = {", i->type, (int)i->m_constMemory.size());
		else
			sprintf(line, ".const %s ce%dc%d[%d] = {", i->type, i->entryIndex, i->index, (int)i->m_constMemory.size());

		output(line);

		std::list<std::string>::iterator j;
		int l=0;
		for(j=i->m_constMemory.begin(); j!=i->m_constMemory.end(); ++j) {
			if(j!=i->m_constMemory.begin())
				output(", ");
			if( (l++ % 4) == 0) output("\n          ");
			output(j->c_str());
		}
		output("\n};\n\n");
	}


	for(std::list<constMemory2>::iterator i=m_constMemoryList2.begin(); i!=m_constMemoryList2.end(); ++i) {
		char line[1024];

		// Global or entry specific
		sprintf(line, ".const %s constant1%s[%d] = {", i->type, i->kernel, (int)i->m_constMemory.size());

		output(line);

		std::list<std::string>::iterator j;
		int l=0;
		for(j=i->m_constMemory.begin(); j!=i->m_constMemory.end(); ++j) {
			if(j!=i->m_constMemory.begin())
				output(", ");
			if( (l++ % 4) == 0) output("\n          ");
			output(j->c_str());
		}
		output("\n};\n\n");
	}

	// Next, print out the local memory declaration
	std::list<cuobjdumpEntry>::iterator e;
	int eIndex=1; // entry index starts from 1 from the first blank entry is missing here (only in header entry list)
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		if(e->m_lMemSize > 0) {
			std::stringstream ssout;
			ssout << ".local .b8 l" << eIndex << "[" << e->m_lMemSize << "];" << std::endl;
			output(ssout.str().c_str());
		}
		eIndex++;
	}
	output("\n");

	// Next, print out the global memory declaration
	std::list<globalMemory>::iterator g;
	for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
		std::stringstream out;
		out << ".global .b8 " << g->name << "[" << g->bytes << "];" << std::endl;
		output(out.str().c_str());
	}
	output("\n");

	// Next, print out constant memory pointers
	std::list<constMemoryPtr>::iterator cp;
	for(cp=m_constMemoryPtrList.begin(); cp!=m_constMemoryPtrList.end(); ++cp) {
		std::stringstream out;
		out << ".const .b8 " << cp->name << "[" << cp->bytes << "];" << std::endl;
		out << ".constptr " << cp->name << ", " << cp->destination << ", " << cp->offset << ";" << std::endl;
		output(out.str().c_str());
	}
	output("\n");

}


//TODO: Some register processing work is supposed to be done here.
void cuobjdumpInstList::addCuobjdumpRegister(std::string reg, bool lo)
{
	int vectorFlag = 0;
	char * regString;
	regString = new char [reg.size()+1];

	std::list<std::string>* typeModifiers = getListEnd().getTypeModifiers();
	std::string baseInst = getListEnd().getBase();

	//TODO: support for 64bit vectors and 128bit vectors
	if((baseInst == "DADD") || (baseInst == "DMUL") || (baseInst == "DFMA") ||
		((typeModifiers->size()==1) &&
		(typeModifiers->front() == ".S64") &&
		((baseInst == "G2R")||(baseInst == "R2G")||
		(baseInst == "GLD")||(baseInst == "GST")||
		(baseInst == "LST")|| (baseInst == "LLD"))))
	{
		vectorFlag = 64;
	}
	else if((typeModifiers->size()==1) && (typeModifiers->front() == ".S128"))
	{
		vectorFlag = 128;
	}

	//TODO: does the vector flag ever need to be set?
	std::string parsedReg = parseCuobjdumpRegister(reg, lo, vectorFlag);
	
	strcpy(regString, parsedReg.c_str());

	getListEnd().addOperand(regString);
}

// add memory operand
// memType: 0=constant, 1=shared, 2=global, 3=local
void cuobjdumpInstList::addCuobjdumpMemoryOperand(std::string mem, int memType) {
	std::string origMem = mem;
	bool neg = false;

	// If constant memory type, add prefix for entry specific constant memory
	if(memType == 0) {
		// Global memory c14
		// Replace this with the actual global memory name
		if(mem.substr(0,1) == "-") {
			//Remove minus sign if exists
			mem = mem.substr(1, mem.size()-1);
			neg = true;
		}

		if(mem.substr(0, 7) == "c [0xe]") {
			// Find the global memory identifier based on the offset provided
			int offset;
			sscanf(mem.substr(9,mem.size()-10).c_str(), "%x", &offset);
			// Find memory
			bool found = false;
			std::list<globalMemory>::iterator g;
			for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
				if(g->offset == offset) {
					mem = "varglobal" + g->name;
					found = true;
					break;
				}
			}

			if(!found) {
				printf("Could not find a global memory with this offset in: %s\n", mem.c_str());
				output("Could not find a global memory with this offset.\n");
				assert(0);
			}

		}
		else if(mem.substr(0, 7) == "c [0x0]"){
			mem = "constant0" + mem.substr(7, mem.length());
		}
		else if(mem.substr(0, 5) == "c [0x"){
			std::string out;
			out = "constant1" + m_entryList.back().m_entryName + mem.substr(8);
			mem = out.c_str();
		}
		else {
			output("Unrecognized memory type:");
			output(mem.c_str());
			output("\n");
			assert(0);
		}

		if (neg) {
			mem = "-"+mem;
		}
	}

	// Local memory
	/*
	if(memType == 3) {
		std::stringstream out;
		printf("Trying to find lmem for: %s\n", m_entryList.back().m_entryName.c_str());
		printf("Original memory: %s\n", mem.c_str());
		assert(kernellmemmap[m_entryList.back().m_entryName] !=0 );
		out << "l" << kernellmemmap[m_entryList.back().m_entryName];// << mem;
		mem = out.str();
	}
	*/
	// Add the memory operand to instruction operand list
	char* memName = new char [strlen(mem.c_str())+1];
	strcpy(memName, mem.c_str());
	getListEnd().addOperand(memName);
}

// increment register list and parse register
std::string cuobjdumpInstList::parseCuobjdumpRegister(std::string reg, bool lo, int vectorFlag)
{
	std::string origReg = reg;
	// Make sure entry list is not empty
	if(m_entryList.size() == 0) {
		output("ERROR: Adding a register before adding an entry.\n");
		assert(0);
	}

	// remove minus sign if exists
	if(reg.substr(0,1) == "-")
		reg = reg.substr(1, reg.size()-1);

	// if lo or hi register, get register name only (remove 'H' or 'L')
	if(lo)
		reg = reg.substr(0, reg.size()-1);

	// Increase register number if needed
	// Two types of registers, R# or A#
	if(reg.substr(0, 1) == "R") {
		reg = reg.substr(1, reg.size()-1);
		int regNum = atoi(reg.c_str());

		// Remove register overlap at 64
		// TODO: is this still needed?
		/*if(regNum > 63 && regNum < 124) {
			regNum -= 64;
			// Fix the origReg string
			std::stringstream out;
			out << ((origReg.substr(0,1)=="-") ? "-" : "")
			    << "$r" << regNum
			    << (lo ? origReg.substr(origReg.size()-3, 3) : "");
			origReg = out.str();
		}*/

		if(vectorFlag==64)
			regNum += 1;
		if(vectorFlag==128)
			regNum += 3;

		if( m_entryList.back().m_largestRegIndex < regNum && regNum < 124 )
			m_entryList.back().m_largestRegIndex = regNum;
		else if( regNum == 124 )
			m_entryList.back().m_reg124 = true;
	} else if(reg.substr(0, 1) == "A") {
		reg = reg.substr(1, reg.size()-1);
		int regNum = atoi(reg.c_str());

		if( m_entryList.back().m_largestOfsRegIndex < regNum && regNum < 124 )
			m_entryList.back().m_largestOfsRegIndex = regNum;
	} else if(reg == "o [0x7f]") {
		m_entryList.back().m_oreg127 = true;
	} else if (reg.substr(0,3) == "SR_") {
		if(reg.substr(3,3)=="Tid") {
			origReg = "%%tid";
			if(reg.substr(7,1)=="X") {
				origReg += ".x";
			}
		}
	} else {
		output("ERROR: unknown register type.\n");
		printf("\nERROR: unknown register type: ");
		printf(reg.c_str());
		printf("\n");
		assert(0);
	}
	return origReg;
}

// pred|reg double operand
void cuobjdumpInstList::addCuobjdumpDoublePredReg(std::string pred, std::string reg, bool lo)
{
	std::string parsedPred = parseCuobjdumpPredicate(pred);
	std::string parsedReg = parseCuobjdumpRegister(reg, lo, 0);

	std::string doublePredReg;
	if(
		getListEnd().getBase() == "DSET" ||
		getListEnd().getBase() == "FSET" ||
		getListEnd().getBase() == "ISET"
	)
		doublePredReg = parsedPred + "/" + parsedReg;
	else
		doublePredReg = parsedPred + "|" + parsedReg;

	char* doublePredRegName = new char [strlen(doublePredReg.c_str())+1];
	strcpy(doublePredRegName, doublePredReg.c_str());
	doublePredRegName[strlen(doublePredReg.c_str())] = '\0';
	getListEnd().addOperand(doublePredRegName);
}

std::string cuobjdumpInstList::parseCuobjdumpPredicate(std::string pred)
{
	std::string origPred = pred;

	// Make sure entry list is not empty
	if(m_entryList.size() == 0) {
		output("ERROR: Adding a predicate before adding an entry.\n");
		assert(0);
	}

	// increase predicate numbers if needed
	pred = pred.substr(2, pred.size()-2);
	int predNum = atoi(pred.c_str());
	if( m_entryList.back().m_largestPredIndex < predNum )
		m_entryList.back().m_largestPredIndex = predNum;

	return origPred;
}

void cuobjdumpInstList::addCubojdumpLabel(std::string label)
{

	if(!(m_entryList.back().m_labelList.empty()))
	{
		std::list<std::string>::iterator labelIterator;

		for( labelIterator=m_entryList.back().m_labelList.begin(); labelIterator!=m_entryList.back().m_labelList.end(); labelIterator++ )
		{
			if(label.compare(*labelIterator) == 0)
				return;
		}
	}

	m_entryList.back().m_labelList.push_back(label);
}

void cuobjdumpInstList::setConstMemoryMap(const char* kernelname, int index){
	std::string kernel = kernelname;
	kernel = kernel.substr(14, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	kernelcmemmap[kernel] = index;
}

void cuobjdumpInstList::setLocalMemoryMap(const char* kernelname, int index){
	std::string kernel = kernelname;
	kernel = kernel.substr(10, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	kernellmemmap[kernel] = index;
}

void cuobjdumpInstList::setglobalVarShndx(const char* shndx){
	m_globalVarShndx = atoi(shndx);
}

int cuobjdumpInstList::getglobalVarShndx(){
	return m_globalVarShndx;
}

void cuobjdumpInstList::addGlobalMemoryID(const char* bytes, const char* name){
	globalMemory globalMemID;
	//globalMemID.offset = atoi(index)/4;
	globalMemID.bytes = atoi(bytes);
	globalMemID.name = name;

	m_globalMemoryList.push_back(globalMemID);
}

void cuobjdumpInstList::updateGlobalMemoryID(const char* offset, const char* name){
	bool found = false;
	std::list<globalMemory>::iterator g;
	for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
		if(g->name.compare(name) == 0) {
			g->offset = atoi(offset)/4;
			found = true;
			break;
		}
	}

	if(!found) {
		printf("Could not find a global memory with this offset in: %s\n", name);
		output("Could not find a global memory with this offset.\n");
		assert(0);
	}
}

//NOT USED
void cuobjdumpInstList::reverseConstMemory() {
	int total = kernelcmemmap.size();
	for (	std::map<std::string,int>::iterator iter = kernelcmemmap.begin();
			iter != kernelcmemmap.end();
			iter++){
		(*iter).second = total - (*iter).second;
	}
}


// create new entry specific constant memory "bank"
void cuobjdumpInstList::addEntryConstMemory(int index, int entryIndex)
{
	constMemory newConstMem;
	newConstMem.index = index;
	newConstMem.entryIndex = entryIndex;
	m_constMemoryList.push_back(newConstMem);
}

void cuobjdumpInstList::addEntryConstMemory2(char* kernelname)
{
	std::string kernel = kernelname;
	kernel = kernel.substr(14, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	constMemory2 newConstMem2;
	newConstMem2.kernel = strdup(kernel.c_str());
	m_constMemoryList2.push_back(newConstMem2);
}

void cuobjdumpInstList::addEntryLocalMemory(int value, int entryIndex)
{
	localMemory newLocalMem;
	newLocalMem.value = value;
	newLocalMem.entryIndex = entryIndex;
	m_localMemoryList.push_back(newLocalMem);
}

void cuobjdumpInstList::setKernelCount(int k){
	m_kernelCount = k;
}

void cuobjdumpInstList::printCuobjdumpInstList()
{
	// Each entry
	std::list<cuobjdumpEntry>::iterator e;
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {




		for(	std::list<cuobjdumpInst>::iterator currentInst=e->m_instList.begin();
				currentInst!=e->m_instList.end();
				++currentInst) {
			// Output the instruction
			output("\t");
			currentInst->printCuobjdumpInst();
			output("\n");
		}
	}
}

void cuobjdumpInstList::printCuobjdumpLocalMemory()
{
	for(	std::list<localMemory>::iterator i=m_localMemoryList.begin();
			i!=m_localMemoryList.end();
			++i) {
		char line[40];
		//if(i->value > 0)
		{
			sprintf(line, ".local .b8 l%d[%d];\n", i->entryIndex, i->value);
			output(line);
		}
	}
}

void cuobjdumpInstList::printCuobjdumpPtxPlusList(cuobjdumpInstList* headerInfo)
{
	output("\n");
	printMemory();
	printCuobjdumpLocalMemory();
	// Each entry
	std::list<cuobjdumpEntry>::reverse_iterator e;
	for(e=m_entryList.rbegin(); e!=m_entryList.rend(); ++e) {

		output("\n");

		// Output the header information for this entry using headerInfo
		// First, find the matching entry in headerInfo
		cuobjdumpEntry headerEntry;

		if( headerInfo->findEntry(e->m_entryName, headerEntry) ) {
			// Entry for current header found, print it out
			std::list<cuobjdumpInst>::iterator headerInstIter;
			for(headerInstIter=headerEntry.m_instList.begin();
				headerInstIter!=headerEntry.m_instList.end();
				++headerInstIter) {
				if(headerInstIter!=headerEntry.m_instList.begin()) {
					output("\t");
				}
				headerInstIter->printHeaderPtx();
				output("\n");
			}
			output("{\n");
		} else {
			// Couldn't find this entry in ptx file
			// Check if it is a dummy entry
			if(e->m_entryName == "__cuda_dummy_entry__") {
				output(".entry ");
				output("__cuda_dummy_entry__");
				output("\n");
				output("{\n");
			} else {
				output("Mismatch in entry names between cuobjdump output and original ptx file.\n");
				assert(0);
			}
		}
		assert( &*e != NULL);
		printRegNames(*e);
		printPredNames(*e);
		printOutOfBoundRegisters(*e);
		output("\n");

		for(std::list<cuobjdumpInst>::iterator currentInst=e->m_instList.begin(); currentInst!=e->m_instList.end(); ++currentInst){
			// Output the instruction
			//cuobjdumpInst* outputInst = &*currentInst;
			output("\t");
			//outputInst->printCuobjdumpPtxPlus(m_entryList.back().m_labelList);
			currentInst->printCuobjdumpPtxPlus(e->m_labelList, this->m_realTexList);
			output("\n");
		}
		output("\n\tl_exit: exit;\n");
		output("}\n");
	}
}

void cuobjdumpInstList::addConstMemoryPtr(const char* offset, const char* size, const char* name){
	constMemoryPtr ptr;
	ptr.offset = atoi(offset);
	ptr.bytes = atoi(size);
	ptr.name = name;
	ptr.destination = "constant0";
	m_constMemoryPtrList.push_back(ptr);
}

