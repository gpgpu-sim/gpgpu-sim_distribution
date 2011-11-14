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


#include "decudaInstList.h"

extern void output(const char * text);

//Constructor
decudaInstList::decudaInstList()
{
	//initilize everything to empty
}

//retrieve point to list end
decudaInst decudaInstList::getListEnd()
{
	return m_entryList.back().m_instList.back();
}

//add decudaInst to the last entry in entry list
int decudaInstList::add(decudaInst* newDecudaInst)
{
	if(m_entryList.size() == 0) {
		//output("ERROR: Adding an instruction before entry.\n");
		addEntry("");
		//assert(0);
	}

	m_entryList.back().m_instList.push_back(*newDecudaInst);

	return m_entryList.size();
}

// add a new entry
int decudaInstList::addEntry(std::string entryName)
{
	decudaEntry newEntry;
	newEntry.m_largestRegIndex = -1;
	newEntry.m_largestOfsRegIndex = -1;
	newEntry.m_largestPredIndex = -1;
	newEntry.m_reg124 = false;
	newEntry.m_oreg127 = false;
	newEntry.m_lMemSize = -1;

	newEntry.m_entryName = entryName;


   // Fill opPerCycle histogram with values
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_1", 0) );
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_2", 0) );
   newEntry.m_opPerCycleHistogram.insert( std::pair<std::string,int>("OP_8", 0) );


	m_entryList.push_back(newEntry);
	return m_entryList.size();
}

void decudaInstList::setLastEntryName(std::string entryName)
{
	m_entryList.back().m_entryName = entryName;
}

void decudaInstList::setLastEntryLMemSize(int lMemSize)
{
	m_entryList.back().m_lMemSize = lMemSize;
}

bool decudaInstList::findEntry(std::string entryName, decudaEntry& entry) {
	std::list<decudaEntry>::iterator e;

	std::string entryNameS = entryName;

	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		if( e->m_entryName == entryNameS) {
			entry = *e;
			return true;
		}
	}

	return false;
}

void decudaInstList::printEntryNames() {
	printf("------------\n");
	printf("%d Entry names:\n", m_entryList.size());
	std::list<decudaEntry>::iterator e;
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		printf("existing entry=%s\n", e->m_entryName.c_str());
	}
	printf("------------\n");
}

// print out .version and .target headers
void decudaInstList::printHeaderInstList()
{
	// These should be in the first entry
	decudaEntry e_first = m_entryList.front();

	std::list<decudaInst>::iterator currentInst;
	for(currentInst=e_first.m_instList.begin(); currentInst!=e_first.m_instList.end(); ++currentInst)
	{
		if(!(currentInst->printHeaderInst()))
		{
			break;
		}
	}
}

void decudaInstList::printNewPtxList(decudaInstList* headerInfo)
{

	// Print memory segment definitions
	output("\n");
	printMemory();


	// Each entry
	std::list<decudaEntry>::iterator e;
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {

		// The first instruction will be the entry instruction
		std::list<decudaInst>::iterator currentInst;
		currentInst=e->m_instList.begin();

		// Output the header information for this entry using headerInfo
		// First, find the matching entry in headerInfo
		decudaEntry headerEntry;

		if( headerInfo->findEntry(e->m_entryName, headerEntry) ) {
			// Entry for current header found, print it out
			std::list<decudaInst>::iterator headerInst;
			for(headerInst=headerEntry.m_instList.begin(); 
			     headerInst!=headerEntry.m_instList.end(); 
			     ++headerInst) {
				if(headerInst!=headerEntry.m_instList.begin())
					output("\t");
				headerInst->printHeaderPtx();
				output("\n");
			}
		} else {
			// Couldn't find this entry in ptx file
			// Check if it is a dummy entry
			if(e->m_entryName == "__cuda_dummy_entry__") {
				output(".entry ");
				output("__cuda_dummy_entry__");
				output("\n");
			} else {
				output("Mismatch in entry names between decuda output and original ptx file.\n");
				assert(0);
			}
		}

		// Output the registers, predicates and other things
		output("{\n");
		printRegNames(*e);
		printPredNames(*e);
		printOutOfBoundRegisters(*e);
		output("\n");


		// Print the rest of the instructions in this entry
		for(++currentInst; currentInst!=e->m_instList.end(); ++currentInst){
			// Output the instruction
			output("\t");
			currentInst->printNewPtx();
			output("\n");

         // Update the opPerCycle histogram
         int opPerCycle = currentInst->getOpPerCycle();
         switch( opPerCycle ) {
            case 8:
               e->m_opPerCycleHistogram["OP_8"] += 1;
               break;
            case 2:
               e->m_opPerCycleHistogram["OP_2"] += 1;
               break;
            case 1:
               e->m_opPerCycleHistogram["OP_1"] += 1;
               break;
         }
		}

		// To prevent the 'ret' instruction deadlock bug in gpgpusim, insert a dummy exit instruction
		output("\n\t");
		output("l_exit: exit;");
		output("\n");

		output("}\n\n\n");

      // Print out histogram
      printf("Entry: %s\n", e->m_entryName.c_str());
      printf("OP_8 %d\n", e->m_opPerCycleHistogram["OP_8"]);
      printf("OP_2 %d\n", e->m_opPerCycleHistogram["OP_2"]);
      printf("OP_1 %d\n", e->m_opPerCycleHistogram["OP_1"]);
      printf("\n");
	}
}


// print out register names
void decudaInstList::printRegNames(decudaEntry entry)
{
	if( entry.m_largestRegIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .u32 $r<%d>;", entry.m_largestRegIndex+1);
		output(out);
		output("\n");
	}

	if( entry.m_largestOfsRegIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .u32 $ofs<%d>;", entry.m_largestOfsRegIndex+1);
		output(out);
		output("\n");
	}
}

// print reg124 and set its value to 0
void decudaInstList::printOutOfBoundRegisters(decudaEntry entry)
{
	if( entry.m_reg124 == true ) {
		output("\n");
		output("\t.reg .u32 $r124;\n");
		output("\tmov.u32 $r124, 0x00000000;\n");
	}
	if( entry.m_oreg127 == true) {
		output("\n");
		output("\t.reg .u32 $o127;\n");
	}
}

// increment register list and parse register
std::string decudaInstList::parseRegister(std::string reg, bool lo, int vectorFlag)
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

	// if lo or hi register, get register name only (remove '.lo' or '.hi')
	if(lo)
		reg = reg.substr(0, reg.size()-3);


	// Increase register number if needed
	// Two types of registers, $r# or $ofs#
	if(reg.substr(0, 2) == "$r") {
		reg = reg.substr(2, reg.size()-2);
		int regNum = atoi(reg.c_str());

		// Remove register overlap at 64
		if(regNum > 63 && regNum < 124) {
			regNum -= 64;
			// Fix the origReg string
			std::stringstream out;
			out << ((origReg.substr(0,1)=="-") ? "-" : "")
			    << "$r" << regNum
			    << (lo ? origReg.substr(origReg.size()-3, 3) : "");
			origReg = out.str();
		}

		if(vectorFlag==64)
			regNum += 1;
                if(vectorFlag==128)
			regNum += 3;

		if( m_entryList.back().m_largestRegIndex < regNum && regNum < 124 )
			m_entryList.back().m_largestRegIndex = regNum;
		else if( regNum == 124 )
			m_entryList.back().m_reg124 = true;
	} else if(reg.substr(0, 4) == "$ofs") {
		reg = reg.substr(4, reg.size()-4);
		int regNum = atoi(reg.c_str());

		if( m_entryList.back().m_largestOfsRegIndex < regNum && regNum < 124 )
			m_entryList.back().m_largestOfsRegIndex = regNum;
	} else if(reg == "$o127") {
		m_entryList.back().m_oreg127 = true;
	} else {
		output("ERROR: unknown register type.\n");
		assert(0);
	}
	return origReg;
}


// add to register list
void decudaInstList::addRegister(std::string reg, bool lo)
{
	//Check to see if the register is an implied vector.
	//If .b64 is a type modifier, $r0 becomes {$r0, $r1}
	//If .b128 is a type modifier, $r0 becomes {$r0, $r1, $r2, $r3}
	//This information is passed to parseRegister so the registers get declared.
	int vectorFlag = 0;
	stringList* typeModifiers = getListEnd().getTypeModifiers();
	stringListPiece* currentPiece;
	currentPiece = typeModifiers->getListStart();
        for(int i=0; (i<typeModifiers->getSize())&&(currentPiece!=NULL); i++)
        {
                const char* modString = currentPiece->stringText;

                if( (strcmp(modString, ".b64")==0) || (strcmp(modString, ".f64")==0) )
                        vectorFlag = 64;
                if( strcmp(modString, ".b128")==0 )
                        vectorFlag = 128;

                currentPiece = currentPiece->nextString;
        }


	std::string parsedReg = parseRegister(reg, lo, vectorFlag);

	// Add the register to instruction operand list
	char* regName = new char [strlen(parsedReg.c_str())+1];
	strcpy(regName, parsedReg.c_str());
	getListEnd().addOperand(regName);
}



// print out predicate names
void decudaInstList::printPredNames(decudaEntry entry)
{
	if( entry.m_largestPredIndex >= 0) {
		char out[30];
		sprintf(out, "\t.reg .pred $p<%d>;", entry.m_largestPredIndex+1);
		output(out);
		output("\n");
	}
	
}

// increment predicate list
std::string decudaInstList::parsePredicate(std::string pred)
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

// add to predicate list
void decudaInstList::addPredicate(std::string pred)
{
	std::string parsedPred = parsePredicate(pred);

	// Add the predicate to instruction operand list
	char* predName = new char [strlen(parsedPred.c_str())+1];
	strcpy(predName, parsedPred.c_str());
	getListEnd().addOperand(predName);
}


// pred|reg double operand
void decudaInstList::addDoublePredReg(std::string pred, std::string reg, bool lo)
{
	std::string parsedPred = parsePredicate(pred);
	std::string parsedReg = parseRegister(reg, lo, 0);

	// Add the double operand to instruction operand list
	// If the base instruction is "set", then both operand get same value, use '/' for separator
	// For cvt,shr,mul use '|' separator
	std::string doublePredReg;
	if( 
		strcmp(getListEnd().getBase(), "set") == 0 ||
		strcmp(getListEnd().getBase(), "setp") == 0 ||
		strcmp(getListEnd().getBase(), "set?68?") == 0 ||
		strcmp(getListEnd().getBase(), "set?65?") == 0 ||
		strcmp(getListEnd().getBase(), "set?67?") == 0 ||
		strcmp(getListEnd().getBase(), "set?13?") == 0
	)
		doublePredReg = parsedPred + "/" + parsedReg;
	else
		doublePredReg = parsedPred + "|" + parsedReg;

	char* doublePredRegName = new char [strlen(doublePredReg.c_str())+1];
	strcpy(doublePredRegName, doublePredReg.c_str());
	getListEnd().addOperand(doublePredRegName);
}


// add to tex list
void decudaInstList::addTex(std::string tex)
{
	std::string origTex = tex;

	// If $tex# tex from decuda, then use index to get real tex name
	if(tex.substr(0, 4) == "$tex") {
		tex = tex.substr(4, tex.size()-4);
		int texNum = atoi(tex.c_str());
		if(texNum >= m_realTexList.size()) {
			output("ERROR: tex does not exist in real tex list from ptx.\n.");
			assert(0);
		}
		
		std::list<std::string>::iterator itex = m_realTexList.begin();
		for(int i=0; i<texNum; i++) itex++;
		origTex = *itex;
	}
	// Otherwise, tex from original ptx
	else {
		m_realTexList.push_back(tex);
	}

	// Add the tex to instruction operand list
	char* texName = new char [strlen(origTex.c_str())+1];
	strcpy(texName, origTex.c_str());
	getListEnd().addOperand(texName);
}


// create new global constant memory "bank"
void decudaInstList::addConstMemory(int index)
{
	constMemory newConstMem;
	newConstMem.index = index;
	newConstMem.entryIndex = 0;
	m_constMemoryList.push_back(newConstMem);
}

// create new entry specific constant memory "bank"
void decudaInstList::addEntryConstMemory(int index)
{
	constMemory newConstMem;
	newConstMem.index = index;
	newConstMem.entryIndex = m_entryList.size();
	m_constMemoryList.push_back(newConstMem);
}



// add value to const memory
void decudaInstList::addConstMemoryValue(std::string constMemoryValue)
{
	m_constMemoryList.back().m_constMemory.push_back(constMemoryValue);
}

// set type of constant memory
void decudaInstList::setConstMemoryType(const char* type)
{
	m_constMemoryList.back().type = type;
}

// print const memory directive
void decudaInstList::printMemory()
{

	// Constant memory
	std::list<constMemory>::iterator i;
	for(i=m_constMemoryList.begin(); i!=m_constMemoryList.end(); ++i) {
		char line[40];

		// Global or entry specific
		if(i->entryIndex == 0)
			sprintf(line, ".const %s c%d[%d] = {", i->type, i->index, i->m_constMemory.size());
		else
			sprintf(line, ".const %s ce%dc%d[%d] = {", i->type, i->entryIndex, i->index, i->m_constMemory.size());
	
		output(line);

		std::list<std::string>::iterator j;
		int l=0;
		for(j=i->m_constMemory.begin(); j!=i->m_constMemory.end(); ++j) {
			if(j!=i->m_constMemory.begin())
				output(", ");
			if( (l++ % 4) == 0) output("\n          ");
			output(j->c_str());
		}
		output("\n};\n\n");
	}

	// Next, print out the local memory declaration
	std::list<decudaEntry>::iterator e;
	int eIndex=1; // entry index starts from 1 from the first blank entry is missing here (only in header entry list)
	for(e=m_entryList.begin(); e!=m_entryList.end(); ++e) {
		if(e->m_lMemSize > 0) {
			std::stringstream out;
			out << ".local .b8 l" << eIndex << "[" << e->m_lMemSize << "];" << std::endl;
			output(out.str().c_str());
		}
		eIndex++;
	}
	output("\n");

	// Next, print out the global memory declaration
	std::list<globalMemory>::iterator g;
	for(g=m_globalMemoryList.begin(); g!=m_globalMemoryList.end(); ++g) {
		std::stringstream out;
		out << ".global .b8 " << g->name << "[" << g->bytes << "];" << std::endl;
		output(out.str().c_str());
	}
	output("\n");

	// Next, print out constant memory pointers
	std::list<constMemoryPtr>::iterator cp;
	for(cp=m_constMemoryPtrList.begin(); cp!=m_constMemoryPtrList.end(); ++cp) {
		std::stringstream out;
		out << ".const .b8 " << cp->name << "[" << cp->bytes << "];" << std::endl;
		out << ".constptr " << cp->name << ", " << cp->destination << ", " << cp->offset << ";" << std::endl;
		output(out.str().c_str());
	}
	output("\n");

}

// add vector operand
void decudaInstList::addVector(char* vector, int vectorSize) {
	// If vector size is 1, make it 4 by adding blanks
	if(vectorSize == 1) {
		std::string vectorNew = vector;
		vectorNew = vectorNew.substr(0,vectorNew.size()-1) + ",_,_,_}";
		char* vectorNewName = new char [strlen(vectorNew.c_str())+1];
		strcpy(vectorNewName, vectorNew.c_str());
		getListEnd().addOperand(vectorNewName);
	} else {
		getListEnd().addOperand(vector);
	}
}


// add memory operand
// memType: 0=constant, 1=shared, 2=global, 3=local
void decudaInstList::addMemoryOperand(std::string mem, int memType) {
	std::string origMem = mem;

	// If constant memory type, add prefix for entry specific constant memory
	if(memType == 0) {
		// Entry-specific constant memory c1
		if(mem.substr(0, 3) == "c1[") {
			std::stringstream out;
			out << "ce" << m_entryList.size() << mem;
			mem = out.str();
		}
		// Global memory c14
		// Replace this with the actual global memory name
		else if(mem.substr(0, 3) == "c14") {
			// Find the global memory identifier based on the offset provided
			int offset;
			sscanf(mem.substr(4,mem.size()-5).c_str(), "%x", &offset);
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
				output("Could not find a global memory with this offset.\n");
				assert(0);
			}
		}
		// Global constant memory c0
		else if(mem.substr(0, 3) == "c0[") {
			// Do nothing
		}
		else {
			output("Unrecognized memory type.\n");
			assert(0);
		}
	}

	// If local memory type, fix the decuda bug where l[4] is actually outputted l[$r4]
	if(memType == 3) {
		// Remove "$r" from "l[$r#]"
		if(mem.substr(2, 2) == "$r") {
			std::stringstream out;
			out << mem.substr(0, 2) << mem.substr(4, mem.size()-4);
			mem = out.str();
		}

		// Add entry entry number after 'l' to differentiate from other entries
		std::stringstream out;
		out << mem.substr(0,1) << m_entryList.size() << mem.substr(1,mem.size()-1);
		mem = out.str();
	}

	// Add the memory operand to instruction operand list
	char* memName = new char [strlen(mem.c_str())+1];
	strcpy(memName, mem.c_str());
	getListEnd().addOperand(memName);
}



// get the list of real tex names
std::list<std::string> decudaInstList::getRealTexList() {
	return m_realTexList;
}

// set the list of real tex names
void decudaInstList::setRealTexList(std::list<std::string> realTexList) {
	m_realTexList = realTexList;
}





// Read in constant memory from bin file
// Two cases of constant memory have been noticed so far
// 1 - All the constant memory is initialized in original ptx file. The assembler combines all this memory into c0
// 2 - Constant memory is declared in ptx, but not initialized (initialized by host). The assembler still calls this c0
void decudaInstList::readConstMemoryFromBinFile(std::string binString) {
	// Initialize a list to store memory values
	std::list<std::string> c0;
	
	// Get each constant segment
	const boost::regex constPattern("(consts \\{[^\\}]*(mem \\{[^\\}]*\\})?[^\\}]*\\})");

	// Parse each constseg
	const boost::sregex_token_iterator end;
	for (
		boost::sregex_token_iterator i(binString.begin(),binString.end(), constPattern);
		i != end;
		++i
	     )
	{
		// For each const segment, get the offset, bytes and memory values string
		std::string constSeg_s = *i;
		std::string offset_s, bytes_s, name, mem;
		int offset, bytes;

		boost::smatch offsetResult;
		boost::smatch bytesResult;
		boost::smatch nameResult;
		boost::smatch memResult;

		const boost::regex offsetPattern("offset\\s*=\\s(\\d*)");
		const boost::regex bytesPattern("bytes\\s*=\\s(\\d*)");
		const boost::regex namePattern("name\\s*=\\s(\\w*)");
		const boost::regex memPattern("mem \\{([^\\}]*)\\}");

		boost::regex_search(constSeg_s, offsetResult, offsetPattern);
		boost::regex_search(constSeg_s, bytesResult, bytesPattern);
		boost::regex_search(constSeg_s, nameResult, namePattern);
		bool memExists = boost::regex_search(constSeg_s, memResult, memPattern);

		//printf("\nmemexists=%d\n", memExists);

		offset_s = offsetResult[1];
		offset = atoi(offset_s.c_str());
		bytes_s = bytesResult[1];
		bytes = atoi(bytes_s.c_str());
		name = nameResult[1];


		// Resize the c0 list if needed
		if(c0.size() < offset/4 + bytes/4) c0.resize(offset/4 + bytes/4, "0x00000000");

		// If memory is initialized, import values
		if(memExists) {
			mem = memResult[1];

			// Parse mem string, loop through each memory value and store it in the appropriate offset
			// in the c0 list
			// Before adding to the list, we increase the size of the list by inserting
			// dummy elements. Then when adding memory values, the dummy elements are removed.
			const boost::regex memValuePattern("(0x[A-Fa-f0-9]{8,8})");

			// Initialize iterator
			std::list<std::string>::iterator it = c0.begin();
			std::advance(it, offset/4);

			// Add values to memory list
			const boost::sregex_token_iterator end2;
			for (
				boost::sregex_token_iterator j(mem.begin(),mem.end(), memValuePattern);
				j != end2;
				++j
			     )
			{
				it = c0.erase(it);
				c0.insert(it, *j);
			}
		} else {
			// Uninitialized const memory - defined a const memory pointer
			constMemoryPtr cMemPtr;
			cMemPtr.bytes = bytes;
			cMemPtr.offset = offset;
			cMemPtr.name = name;
			cMemPtr.destination = "c0";

			m_constMemoryPtrList.push_back(cMemPtr);
		}
	}

	
	// Finished parsing of the file, now iterate over the list and add values to constant memory segment
	if(c0.size() > 0) {
		addConstMemory(0);
		setConstMemoryType(".u32");	
		std::list<std::string>::iterator c;
		for(c=c0.begin(); c!=c0.end(); ++c) {
			addConstMemoryValue(*c);
		}
	}		
}

// Read in global memory from bin file
void decudaInstList::readGlobalMemoryFromBinFile(std::string binString) {
	// Get each constant segment
	const boost::regex globalPattern("(reloc \\{[^\\}]*segnum  = 14[^\\}]*\\})");

	// Parse each constseg
	const boost::sregex_token_iterator end;
	for (
		boost::sregex_token_iterator i(binString.begin(),binString.end(), globalPattern);
		i != end;
		++i
	     )
	{
		// For each global segment, get the offset, bytes and name
		std::string globalSeg_s = *i;
		std::string offset_s, bytes_s, name;
		int offset, bytes;

		boost::smatch offsetResult;
		boost::smatch bytesResult;
		boost::smatch nameResult;

		const boost::regex offsetPattern("offset\\s*=\\s(\\d*)");
		const boost::regex bytesPattern("bytes\\s*=\\s(\\d*)");
		const boost::regex namePattern("name\\s*=\\s(\\w*)");

		boost::regex_search(globalSeg_s, offsetResult, offsetPattern);
		boost::regex_search(globalSeg_s, bytesResult, bytesPattern);
		boost::regex_search(globalSeg_s, nameResult, namePattern);

		offset_s = offsetResult[1];
		offset = atoi(offset_s.c_str());
		bytes_s = bytesResult[1];
		bytes = atoi(bytes_s.c_str());
		name = nameResult[1];

		// Add global memory
		globalMemory gMem;
		gMem.offset = offset;
		gMem.bytes = bytes;
		gMem.name = name;

		m_globalMemoryList.push_back(gMem);
	}
}
