#include "cuobjdumpInst.h"

// Used for local memory segments
struct localMemory
{
	int value;
	int entryIndex;
};

class cuobjdumpInstList: public decudaInstList
{
protected:
	std::string parseCuobjdumpPredicate(std::string pred);
	int m_kernelCount;
	std::map<std::string,int>kernelcmemmap;
	std::map<std::string,int>kernellmemmap;
	std::list<localMemory> m_localMemoryList;

public:
	//Constructor
	cuobjdumpInstList();

	void setKernelCount(int k);
	void readConstMemoryFromElfFile(std::string elf);

	void addCuobjdumpRegister(std::string reg, bool lo=false); //add register
	void addCuobjdumpMemoryOperand(std::string mem, int memType);
	std::string parseCuobjdumpRegister(std::string reg, bool lo, int vectorFlag);
	void addCuobjdumpDoublePredReg(std::string pred, std::string reg, bool lo=false);

	void addCubojdumpLabel(std::string label);

	void addEntryConstMemory(int index, int entryIndex);
	void addEntryConstMemory2(char* kernel);
	void addConstMemoryPtr(const char* bytes, const char* offset, const char* name);
	void setConstMemoryMap(const char* kernelname, int index);
	void setLocalMemoryMap(const char* kernelname, int index);
	void reverseConstMemory();
	void addEntryLocalMemory(int value, int entryIndex);
	void readOtherConstMemoryFromBinFile(std::string binString); // read in constant memory from bin file

	void printCuobjdumpLocalMemory();

	void printCuobjdumpInstList();
	void printCuobjdumpPtxPlusList(cuobjdumpInstList* headerInfo);
};

