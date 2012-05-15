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
#include "stringList.h"

class cuobjdumpInst
{
protected:
	//instruction data
	const char* m_label; //instruction label
	stringList* m_predicate; //instruction predicate
	const char* m_base; //instruction mnemonic
	stringList* m_baseModifiers; //base modifiers
	stringList* m_typeModifiers; //operand types
	stringList* m_operands; //operands
	stringList* m_predicateModifiers; //predicate modifiers

public:
	//Constructor
	cuobjdumpInst();

	//accessors
	const char* getBase();
	stringList* getTypeModifiers();

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
	void printCuobjdumpBaseModifiers();
	void printCuobjdumpOperand(stringListPiece* currentPiece, std::string operandDelimiter, const char* base);
	void printCuobjdumpOperandlohi(std::string op);
	void printCuobjdumpOperands();

	void printCuobjdumpPtxPlus(std::list<std::string> labelList, std::list<std::string> texList);

	//print representation
	bool printHeaderInst();
	void printCuobjdumpInst();
	void printHeaderPtx();
};

#endif //_CUOBJDUMPINST_H_
