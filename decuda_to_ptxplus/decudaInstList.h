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


#include "decudaInst.h"
#include <list>
#include <string>
#include <cstring>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <regex.hpp>
#include <map>

// Used for entry specific constant memory segments (c1)
struct constMemory
{
	int index;
	int entryIndex;
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



struct decudaEntry
{
	//char* m_entryName;
	std::string m_entryName;
	std::list<decudaInst> m_instList;	// List of decuda instructions

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



class decudaInstList
{

protected:
	// List of decuda entries
	std::list<decudaEntry> m_entryList;

	// Const memory list
	std::list<constMemory> m_constMemoryList;

	// Const memory pointers list
	std::list<constMemoryPtr> m_constMemoryPtrList;

	// Global memory list
	std::list<globalMemory> m_globalMemoryList;

	// Tex list
	std::list<std::string>  m_realTexList;	// Stores the real names of tex variables

	// Print register names
	void printRegNames(decudaEntry entry);
	void printOutOfBoundRegisters(decudaEntry entry);

	// Print predicate names
	void printPredNames(decudaEntry entry);

	// Print const memory directives
	void printMemory();

	// Increment register or predicate offsets
	std::string parseRegister(std::string reg, bool lo=false, int vectorFlag=0);
	std::string parsePredicate(std::string pred);

public:
	//constructor
	decudaInstList();

	//accessors
	decudaInst getListEnd();

	//mutator
	int addEntry(std::string entryName); // creates a new entry
	void setLastEntryName(std::string entryName); // sets name of last entry
	void setLastEntryLMemSize(int lMemSize); // sets the local memory size of last entry
	bool findEntry(std::string entryName, decudaEntry& entry); // find and return entry
	

	int add(decudaInst* newInst); //add DecudaInst to list

	void addRegister(std::string reg, bool lo=false); //add register
	void addPredicate(std::string pred); //add predicate
	void addDoublePredReg(std::string pred, std::string reg, bool lo=false); // add pred|reg double operand

	void addTex(std::string tex);	// add tex operand

	void addVector(char* vector, int vectorSize); // add vector operand

	void addMemoryOperand(std::string mem, int memType); // add memory operand


	// Parsing constant memory segments list
	void addEntryConstMemory(int index); // add entry specific const memory
	void addConstMemory(int index); // add global const memory
	void setConstMemoryType(const char* type); // set type of constant memory
	void addConstMemoryValue(std::string constMemoryValue); // add const memory

	std::list<std::string> getRealTexList(); // get the list of real tex names
	void setRealTexList(std::list<std::string> realTexList); // set the list of real tex names

	void readConstMemoryFromBinFile(std::string binString); // read in constant memory from bin file
	void readGlobalMemoryFromBinFile(std::string binString); // read in global memory from bin file

	//print representation
	void printHeaderInstList();
	void printDecudaInstList();
	void printNewPtxList(decudaInstList* headerInfo);


	// debug helper methods
	void printEntryNames();

};
