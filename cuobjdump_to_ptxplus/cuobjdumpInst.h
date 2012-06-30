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

#ifndef _CUOBJDUMPINST_H_
#define _CUOBJDUMPINST_H_

// External includes
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>

// Local includes
//#include "cuobjdumpInstList.h"

class cuobjdumpInst
{
protected:
	//instruction data
	std::string m_label; //instruction label
	std::list<std::string>* m_predicate; //instruction predicate
	std::string m_base; //instruction mnemonic
	std::list<std::string>* m_baseModifiers; //base modifiers
	std::list<std::string>* m_typeModifiers; //operand types
	std::list<std::string>* m_operands; //operands
	std::list<std::string>* m_predicateModifiers; //predicate modifiers

public:
	//Constructor
	cuobjdumpInst();
	~cuobjdumpInst();

	//accessors
	const std::string getBase();
	std::list<std::string>* getTypeModifiers();

	//Mutators
	void setLabel(const char* setLabelValue);
	void setPredicate(const char* setPredicateValue);
	void addPredicateModifier(const char* addPredicateMod);
	void setBase(const char* setBaseValue);
	void addBaseModifier(const char* addBaseMod);
	void addTypeModifier(const char* addTypeMod);
	void addOperand(const char* addOp);

	bool checkCubojdumpLabel(std::list<std::string> labelList, std::string label);

	void printCuobjdumpLabel(std::list<std::string> labelList);
	void printCuobjdumpPredicate();
	void printCuobjdumpTypeModifiers();
	void printCuobjdumpOutputModifiers(const char* defaultMod);
	void printCuobjdumpBaseModifiers();
	void printCuobjdumpOperand(std::string currentPiece, std::string operandDelimiter, std::string base);
	void printCuobjdumpOperandlohi(std::string op);
	void printCuobjdumpOperands();

	void printCuobjdumpPtxPlus(std::list<std::string> labelList, std::list<std::string> texList);

	//print representation
	bool printHeaderInst();
	void printCuobjdumpInst();
	void printHeaderPtx();

	static void printStringList(std::list<std::string>* strlist);
};

#endif //_CUOBJDUMPINST_H_
