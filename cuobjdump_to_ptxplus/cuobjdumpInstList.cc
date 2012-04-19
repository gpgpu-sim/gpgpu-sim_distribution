#include <iostream>
#include "cuobjdumpInstList.h"

extern void output(const char * text);

//Constructor
cuobjdumpInstList::cuobjdumpInstList()
{
        //initilize everything to empty
}




//TODO: Some register processing work is supposed to be done here.
void cuobjdumpInstList::addCuobjdumpRegister(std::string reg, bool lo)
{
	int vectorFlag = 0;
	char * regString;
	regString = new char [reg.size()+1];

	stringList* typeModifiers = getListEnd().getTypeModifiers();
	const char* baseInst = getListEnd().getBase();

	//TODO: support for 64bit vectors and 128bit vectors
	if((strcmp(baseInst, "DADD")==0) || (strcmp(baseInst, "DMUL")==0) || (strcmp(baseInst, "DFMA")==0) ||
		((typeModifiers->getSize()==1) &&
		(strcmp((typeModifiers->getListStart()->stringText), ".S64")==0) &&
		((strcmp(baseInst, "G2R")==0)||(strcmp(baseInst, "R2G")==0)||
		(strcmp(baseInst, "GLD")==0)||(strcmp(baseInst, "GST")==0)||
		(strcmp(baseInst, "LST")==0))))
	{
		vectorFlag = 64;
	}
	else if((typeModifiers->getSize()==1) && (strcmp((typeModifiers->getListStart()->stringText), ".S128")==0))
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
int currconstmem =1;
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
					mem = g->name;
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
			if(currconstmem != m_entryList.size()) currconstmem++;
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
	if(memType == 3) {
		std::stringstream out;
		printf("Trying to find lmem for: %s\n", m_entryList.back().m_entryName.c_str());
		printf("Original memory: %s\n", mem.c_str());
		assert(kernellmemmap[m_entryList.back().m_entryName] !=0 );
		out << "l" << kernellmemmap[m_entryList.back().m_entryName];// << mem;
		mem = out.str();
	}

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
		strcmp(getListEnd().getBase(), "DSET")==0 ||
		strcmp(getListEnd().getBase(), "FSET")==0 ||
		strcmp(getListEnd().getBase(), "ISET")==0
	)
		doublePredReg = parsedPred + "/" + parsedReg;
	else
		doublePredReg = parsedPred + "|" + parsedReg;

	char* doublePredRegName = new char [strlen(doublePredReg.c_str())];
	strcpy(doublePredRegName, doublePredReg.c_str());
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
	printf("Setting kernelcmemmap[%s]=%d\n", kernel.c_str(), index);
	kernelcmemmap[kernel] = index;
}

void cuobjdumpInstList::setLocalMemoryMap(const char* kernelname, int index){
	std::string kernel = kernelname;
	kernel = kernel.substr(10, kernel.length()-1);
	kernel = kernel.substr(0, kernel.find("\t"));
	printf("Setting kernellmemmap[%s]=%d\n", kernel.c_str(), index);
	kernellmemmap[kernel] = index;
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


// Read in constant memory from bin file
// Two cases of constant memory have been noticed so far
// 1 - All the constant memory is initialized in original ptx file. The assembler combines all this memory into c0
// 2 - Constant memory is declared in ptx, but not initialized (initialized by host). The assembler still calls this c0
void cuobjdumpInstList::readConstMemoryFromElfFile(std::string elf)
{
	unsigned k=1;
	printf("Trying to find constant memory in elf file:\n");

	// Get each constant segment
	const boost::regex constPattern("^\\.nv\\.constant1\\.[^\n]+\n[ x0-9a-f\t]+$");
	// Parse each constseg
	const boost::sregex_token_iterator end;
	for (
		boost::sregex_token_iterator i(elf.begin(),elf.end(), constPattern);
		i != end;
		++i
		)
	{
		std::string memseg = *i;
		boost::smatch memResult;

		const boost::regex memValuePattern("(0x[A-Fa-f0-9]{8,8})");

		bool memExists = boost::regex_search(memseg, memResult, memValuePattern);


		std::list<std::string> c1;
		std::list<std::string>::iterator it = c1.begin();

		const boost::sregex_token_iterator end2;
		for (
			boost::sregex_token_iterator j(memseg.begin(),memseg.end(), memValuePattern);
			j != end2;
			++j ){
			c1.insert(it, *j);
		}

		addEntryConstMemory(1, k);
		setConstMemoryType(".u32");

		std::list<std::string>::iterator c;
		if(c1.size() > 0) {
			for(c=c1.begin(); c!=c1.end(); ++c) {
				std::string a = *c;
				//printf("%s ", a.c_str());
				addConstMemoryValue(a);
			}
		}


		printf("Found constant memory\n");
		printf(memseg.c_str());
		printf("\n");
		k++;
	}
	m_kernelCount = k-1;
}
void cuobjdumpInstList::setKernelCount(int k){
	m_kernelCount = k;
}
void cuobjdumpInstList::readOtherConstMemoryFromBinFile(std::string binString)
{
	// Initialize a list to store memory values
	// std::list<std::string> c0;

	// Get each code segment
	//const boost::regex codePattern("(code \\{[^\\{\\}]*(const \\{[^\\{\\}]*(mem \\{[^\\{\\}]*\\}[^\\{\\}]*)+\\}[^\\{\\}]*)+bincode \\{[^\\{\\}]*\\}[^\\{\\}]*\\})");
	const boost::regex codePattern("(code \\{[^\\{\\}]*(const \\{[^\\{\\}]*(mem \\{[^\\{\\}]*\\}[^\\{\\}]*)+\\}[^\\{\\}]*)*bincode \\{[^\\{\\}]*\\}[^\\{\\}]*\\})");

	int k=1;

	// Parse each codeseg
	const boost::sregex_token_iterator end;
	for(
		boost::sregex_token_iterator i(binString.begin(),binString.end(), codePattern);
		i != end;
		++i
	)
	{
		std::list<std::string> c1;

		// For each code segment, get the seg numbers and memory values string
		std::string codeSeg_s = *i;
		std::string segnum_s, lmem_s, mem;
		int segnum;
		int lmem;

		boost::smatch segnumResult;
		boost::smatch lmemResult;
		boost::smatch memResult;

		const boost::regex segnumPattern("segnum\\s*=\\s(\\d*)");
		const boost::regex lmemPattern("lmem\\s*=\\s(\\d*)");
		const boost::regex memPattern("mem \\{([^\\}]*)\\}");

		boost::regex_search(codeSeg_s, segnumResult, segnumPattern);
		boost::regex_search(codeSeg_s, lmemResult, lmemPattern);
		bool memExists = boost::regex_search(codeSeg_s, memResult, memPattern);

		lmem_s = lmemResult[1];
		lmem = atoi(lmem_s.c_str());

		addEntryLocalMemory(lmem, k);

		if(memExists)
		{
		segnum_s = segnumResult[1];
		segnum = atoi(segnum_s.c_str());

		mem = memResult[1];
		const boost::regex memValuePattern("(0x[A-Fa-f0-9]{8,8})");

		std::list<std::string>::iterator it = c1.begin();

		const boost::sregex_token_iterator end2;
		for (
			boost::sregex_token_iterator j(mem.begin(),mem.end(), memValuePattern);
			j != end2;
			++j
			)
		{
			c1.insert(it, *j);
		}

		addEntryConstMemory(segnum, k);
		setConstMemoryType(".u32");

		std::list<std::string>::iterator c;
		if(c1.size() > 0) {
			for(c=c1.begin(); c!=c1.end(); ++c) {
				std::string a = *c;
				//printf("%s ", a.c_str());
				addConstMemoryValue(a);
			}
		}
		}
		k++;
	}
	m_kernelCount = k-1;
}

void cuobjdumpInstList::printCuobjdumpInstList()
{
	// Each entry
	std::list<decudaEntry>::iterator e;
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {




		for(	std::list<decudaInst>::iterator currentInst=e->m_instList.begin();
				currentInst!=e->m_instList.end();
				++currentInst) {
			// Output the instruction
			output("\t");
			currentInst->printDecudaInst();
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
	std::list<decudaEntry>::reverse_iterator e;
	for(e=m_entryList.rbegin(); e!=m_entryList.rend(); ++e) {

		output("\n");

		// Output the header information for this entry using headerInfo
		// First, find the matching entry in headerInfo
		decudaEntry headerEntry;

		if( headerInfo->findEntry(e->m_entryName, headerEntry) ) {
			// Entry for current header found, print it out
			std::list<decudaInst>::iterator headerInstIter;
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
				output("Mismatch in entry names between decuda output and original ptx file.\n");
				assert(0);
			}
		}
		assert( &*e != NULL);
		printRegNames(*e);
		printPredNames(*e);
		printOutOfBoundRegisters(*e);
		output("\n");

		for(std::list<decudaInst>::iterator currentInst=e->m_instList.begin(); currentInst!=e->m_instList.end(); ++currentInst){
			// Output the instruction
			//cuobjdumpInst* outputInst = &*currentInst;
			cuobjdumpInst* outputInst = static_cast<cuobjdumpInst*>(&*currentInst);
			output("\t");
			//outputInst->printCuobjdumpPtxPlus(m_entryList.back().m_labelList);
			outputInst->printCuobjdumpPtxPlus(e->m_labelList, this->m_realTexList);
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
	printf("\naddConstMemoryPtr: %s, size: %d, offset: %d\n", ptr.name.c_str(), ptr.bytes, ptr.offset);
}

