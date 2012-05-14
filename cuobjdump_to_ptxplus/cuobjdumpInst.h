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
