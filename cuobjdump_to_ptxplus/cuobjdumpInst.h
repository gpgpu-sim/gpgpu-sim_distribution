#include "decudaInstList.h"

class cuobjdumpInst: public decudaInst
{

public:
	//Constructor
	cuobjdumpInst();

	bool checkCubojdumpLabel(std::list<std::string> labelList, std::string label);

	void printCuobjdumpLabel(std::list<std::string> labelList);
	void printCuobjdumpPredicate();
	void printCuobjdumpTypeModifiers();
	void printCuobjdumpBaseModifiers();
	void printCuobjdumpOperand(stringListPiece* currentPiece, std::string operandDelimiter, const char* base);
	void printCuobjdumpOperandlohi(std::string op);
	void printCuobjdumpOperands();

	void printCuobjdumpPtxPlus(std::list<std::string> labelList, std::list<std::string> texList);
};

