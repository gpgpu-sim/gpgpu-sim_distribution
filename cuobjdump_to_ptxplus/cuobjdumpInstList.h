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

#ifndef _CUOBJDUMPINSTLIST_H_
#define _CUOBJDUMPINSTLIST_H_

// External includes
#include <list>
#include <map>
#include <string>

// Local includes
#include "cuobjdumpInst.h"

// Used for entry specific constant memory segments (c1)
struct constMemory
{
	int index;
	int entryIndex;
	const char* type;
	std::list<std::string> m_constMemory;
};

struct constMemory2
{
	const char* kernel;
	const char* type;
	std::list<std::string> m_constMemory;
};

// Used for uninitialized constant memory (globally defined)
struct constMemoryPtr
{
	int bytes;
	std::string name;

	std::string destination;
	int offset;
};

// Used for global memory segments
struct globalMemory
{
	int offset;
	int bytes;
	std::string name;
};

struct cuobjdumpEntry
{
	//char* m_entryName;
	std::string m_entryName;
	std::list<cuobjdumpInst> m_instList;	// List of cuobjdump instructions

	// Register list
	int m_largestRegIndex;
	int m_largestOfsRegIndex;
	bool m_reg124;
	bool m_oreg127;

	// Predicate list
	int m_largestPredIndex;

	// Local memory size
	int m_lMemSize;

	//use for recording used labels
	std::list<std::string> m_labelList;

   // Histogram for operation per cycle count
   std::map<std::string, int> m_opPerCycleHistogram;
};

// Used for local memory segments
struct localMemory
{
	int value;
	int entryIndex;
};

class cuobjdumpInstList
{
protected:
	std::list<cuobjdumpEntry> m_entryList;
	std::list<constMemory> m_constMemoryList;
	std::list<constMemory2> m_constMemoryList2;
	std::list<globalMemory> m_globalMemoryList;

	int m_kernelCount;
	std::map<std::string,int>kernelcmemmap;
	std::map<std::string,int>kernellmemmap;
	std::list<localMemory> m_localMemoryList;
	std::list<std::string>  m_realTexList;	// Stores the real names of tex variables
	std::list<constMemoryPtr> m_constMemoryPtrList;
	int m_globalVarShndx; //records shndx value of global variables in the elf file SYMTAB

	// Functions:
	std::string parseCuobjdumpPredicate(std::string pred);
	void printMemory();// Print const memory directives
	// Print register names
	void printRegNames(cuobjdumpEntry entry);
	void printOutOfBoundRegisters(cuobjdumpEntry entry);

	// Print predicate names
	void printPredNames(cuobjdumpEntry entry);
public:
	//Constructor
	cuobjdumpInstList();

	cuobjdumpInst getListEnd();

	// Functions used by the parser
	int addEntry(std::string entryName); // creates a new entry
	int add(cuobjdumpInst* newInst); //add cuobjdumpInst to list
	void addConstMemory(int index); // add global const memory
	void addTex(std::string tex);	// add tex operand
	bool findEntry(std::string entryName, cuobjdumpEntry& entry); // find and return entry

	void setKernelCount(int k);
	void readConstMemoryFromElfFile(std::string elf);
	void setLastEntryName(std::string entryName); // sets name of last entry
	void addCuobjdumpRegister(std::string reg, bool lo=false); //add register
	void addCuobjdumpMemoryOperand(std::string mem, int memType);
	std::string parseCuobjdumpRegister(std::string reg, bool lo, int vectorFlag);
	void addCuobjdumpDoublePredReg(std::string pred, std::string reg, bool lo=false);

	void addCubojdumpLabel(std::string label);

	void addEntryConstMemory(int index, int entryIndex);
	void addEntryConstMemory2(char* kernel);
	void setConstMemoryType(const char* type);
	void setConstMemoryType2(const char* type); // set type of constant memory
	void addConstMemoryValue(std::string constMemoryValue); // add const memory
	void addConstMemoryValue2(std::string constMemoryValue); // add const memory
	void addConstMemoryPtr(const char* bytes, const char* offset, const char* name);
	void setConstMemoryMap(const char* kernelname, int index);
	void setLocalMemoryMap(const char* kernelname, int index);
	void setglobalVarShndx(const char* shndx);
	int getglobalVarShndx();
	void addGlobalMemoryID(const char* bytes, const char* name);
	void updateGlobalMemoryID(const char* offset, const char* name);
	void reverseConstMemory();
	void addEntryLocalMemory(int value, int entryIndex);
	void readOtherConstMemoryFromBinFile(std::string binString); // read in constant memory from bin file
	std::list<std::string> getRealTexList(); // get the list of real tex names
	void setRealTexList(std::list<std::string> realTexList); // set the list of real tex names
	void printHeaderInstList();
	void printCuobjdumpLocalMemory();
	void printCuobjdumpInstList();
	void printCuobjdumpPtxPlusList(cuobjdumpInstList* headerInfo);
};

#endif //_CUOBJDUMPINSTLIST_H_
