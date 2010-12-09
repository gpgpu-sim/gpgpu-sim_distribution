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
