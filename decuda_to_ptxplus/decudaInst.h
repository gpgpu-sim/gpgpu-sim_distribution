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


#include "stringList.h"
#include <assert.h>

class decudaInst
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
	

	int m_opPerCycle; // stores operations per cycle count for current instruction

	//next instruction in linked list
	//direction is m_listStart to m_listEnd
	decudaInst* m_nextDecudaInst;

	// print instruction unmodified
	void printDefaultPtx();
	void printBaseModifiers();
	void printTypeModifiers();
	void printOperands();
	void printLabel();
	void printPredicate();

public:
	//constructor
	decudaInst();

	//accessors
	const char* getBase();
	stringList* getOperands();
	stringList* getBaseModifiers();
        stringList* getTypeModifiers();
	decudaInst* getNextDecudaInst();

   int getOpPerCycle() const { return m_opPerCycle; }

	bool isEntryStart();	// true if start of an entry

	//mutators
	void setBase(const char* setBaseValue);
	void addBaseModifier(const char* addBaseMod);
	void addTypeModifier(const char* addTypeMod);
	void addOperand(const char* addOp);
	void setPredicate(const char* setPredicateValue);
	void addPredicateModifier(const char* addPredicateMod);
	void setLabel(const char* setLabelValue);

	void setNextDecudaInst(decudaInst* setDecudaInstValue);

	//print representation
	bool printHeaderInst();
	bool printHeaderInst2();
	void printDecudaInst();
	void printNewPtx();
	void printHeaderPtx();
	//TODO: translate to New PTX and print

};
