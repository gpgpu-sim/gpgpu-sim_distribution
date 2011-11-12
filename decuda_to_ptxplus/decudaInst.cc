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
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>

extern void output(const char * text);

//Constructor
decudaInst::decudaInst()
{
	//initilize everything to empty
	m_label = "";
	m_predicate = new stringList();
	m_base = "";
	m_baseModifiers = new stringList();
	m_typeModifiers = new stringList();
	m_operands = new stringList();
	m_predicateModifiers = new stringList();

	m_nextDecudaInst = NULL;

   // Set operations per cycle to 8 by default (no penalty)
   m_opPerCycle = 8;
}

//retreive instruction mnemonic
const char* decudaInst::getBase()
{
	return m_base;
}

stringList* decudaInst::getOperands()
{
	return m_operands;
}

stringList* decudaInst::getBaseModifiers()
{
	return m_baseModifiers;
}

stringList* decudaInst::getTypeModifiers()
{
        return m_typeModifiers;
}

//get next instruction in linked list
//direction is m_listStart to m_listEnd
decudaInst* decudaInst::getNextDecudaInst()
{
	return m_nextDecudaInst;
}

void decudaInst::setBase(const char* setBaseValue)
{
	m_base = setBaseValue;
}

void decudaInst::addBaseModifier(const char* addBaseMod)
{
	stringListPiece* tempPiece = new stringListPiece;
	tempPiece->stringText = addBaseMod;

	m_baseModifiers->add(tempPiece);
}

void decudaInst::addTypeModifier(const char* addTypeMod)
{
	stringListPiece* tempPiece = new stringListPiece;
	tempPiece->stringText = addTypeMod;

	m_typeModifiers->add(tempPiece);
}

void decudaInst::addOperand(const char* addOp)
{
	stringListPiece* tempPiece = new stringListPiece;
	tempPiece->stringText = addOp;

	m_operands->add(tempPiece);
}

void decudaInst::setPredicate(const char* setPredicateValue)
{
	stringListPiece* tempPiece = new stringListPiece;
	tempPiece->stringText = setPredicateValue;

	m_predicate->add(tempPiece);
}

void decudaInst::addPredicateModifier(const char* addPredicateMod)
{
	stringListPiece* tempPiece = new stringListPiece;
	tempPiece->stringText = addPredicateMod;

	m_predicateModifiers->add(tempPiece);
}

void decudaInst::setLabel(const char* setLabelValue)
{
	m_label = setLabelValue;
}

//set next instruction in linked list
//direction is m_listStart to m_listEnd
void decudaInst::setNextDecudaInst(decudaInst* setDecudaInstValue)
{
	m_nextDecudaInst = setDecudaInstValue;
}

// returns true if current instruction is start of an entry (i.e. '{')
bool decudaInst::isEntryStart()
{
	return (strcmp(m_base, "{")==0);
}

//print out .version and .target header lines
bool decudaInst::printHeaderInst()
{
	if(strcmp(m_base, ".version")==0)
	{
		output(m_base); output(" ");

		stringListPiece* currentPiece = m_operands->getListStart();
		output(currentPiece->stringText);
		currentPiece = currentPiece->nextString;

		if(currentPiece!=NULL)
		{
			output("."); output(currentPiece->stringText);
		}
      output("+");
		output("\n");
	}
	else if(strcmp(m_base, ".target")==0)
	{
		output(m_base); output(" ");

		stringListPiece* currentPiece = m_operands->getListStart();
		output(currentPiece->stringText);
		currentPiece = currentPiece->nextString;

		while(currentPiece!=NULL)
		{
			output(", "); output(currentPiece->stringText);
			currentPiece = currentPiece->nextString;
		}
		output("\n");
	}
	else if(strcmp(m_base, ".tex")==0)
	{
		output(m_base); output(" ");

		stringListPiece* currentPiece;


		currentPiece = m_baseModifiers->getListStart();
		output(currentPiece->stringText); output(" ");
		currentPiece = currentPiece->nextString;

		while(currentPiece!=NULL)
		{
			output(" "); output(currentPiece->stringText);
			currentPiece = currentPiece->nextString;
		}

		currentPiece = m_operands->getListStart();
		output(currentPiece->stringText);
		currentPiece = currentPiece->nextString;

		while(currentPiece!=NULL)
		{
			output(" "); output(currentPiece->stringText);
			currentPiece = currentPiece->nextString;
		}
		output(";\n");
	}
	else
	{
		return false;
	}
	return true;
}

//print out parameters
bool decudaInst::printHeaderInst2()
{
	if(strcmp(m_base, ".param")==0)
	{
		output(m_base);

		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		
		for(int i=0; (i<m_typeModifiers->getSize())&&(currentPiece!=NULL); i++)
		{
			output(" "); output(currentPiece->stringText);

			currentPiece = currentPiece->nextString;
		}

		currentPiece = m_operands->getListStart();
		for(int i=0; (i<m_operands->getSize())&&(currentPiece!=NULL); i++)
		{
			output(" "); output(currentPiece->stringText);

			currentPiece = currentPiece->nextString;
		}
	}
	else
	{
		return false;
	}
	return true;
}
//print out the Decuda instruction
void decudaInst::printDecudaInst()
{
	/*TODO: print label here*/
	/*TODO: print predicate here*/
	/*std::cout << "Instruction Base: " << m_base << "\n";

	std::cout << "Instruction Modifiers: ";
	m_baseModifiers->printStringList();
	std::cout << "\n";

	std::cout << "Operand types: ";
	m_typeModifiers->printStringList();
	std::cout << "\n";

	std::cout << "Operands: ";
	m_operands->printStringList();
	std::cout << "\n\n";*/

	std::cout << m_base << " ";
	m_baseModifiers->printStringList();
	std::cout << " ";
	m_typeModifiers->printStringList();
	std::cout << " ";
	m_operands->printStringList();
	std::cout << "\n";


}


// Just prints the base and operands
void decudaInst::printHeaderPtx()
{
	output(m_base); output(" ");

	stringListPiece* currentPiece;

	currentPiece = m_baseModifiers->getListStart();
	for(int i=0; (i<m_baseModifiers->getSize())&&(currentPiece!=NULL); i++)
	{
		output(" "); output(currentPiece->stringText);
		currentPiece = currentPiece->nextString;
	}

	currentPiece = m_operands->getListStart();
	for(int i=0; (i<m_operands->getSize())&&(currentPiece!=NULL); i++)
	{
		output(" "); output(currentPiece->stringText);
		currentPiece = currentPiece->nextString;
	}
}



// Print unmodified instruction
void decudaInst::printDefaultPtx()
{
	printLabel();
	printPredicate();

	output(m_base);

	printBaseModifiers();
	printTypeModifiers();
	printOperands();

	output(";");
}


// Print unmodified base modifiers, operands, labels and predicates
void decudaInst::printLabel()
{
	if(m_label != "") {
		output(m_label);
		output(": ");
	}
}

void decudaInst::printPredicate()
{
	stringListPiece* currentPiece = m_predicate->getListStart();
	if(currentPiece!=NULL)
	{
		output(currentPiece->stringText);

		stringListPiece* currentPiece2 = m_predicateModifiers->getListStart();
		for(int i=0; (i<m_predicateModifiers->getSize())&&(currentPiece2!=NULL); i++)
		{
			output(currentPiece2->stringText);

			currentPiece2 = currentPiece2->nextString;
		}

		output(" ");
	}
}

void decudaInst::printBaseModifiers()
{
	stringListPiece* currentPiece = m_baseModifiers->getListStart();
	for(int i=0; (i<m_baseModifiers->getSize())&&(currentPiece!=NULL); i++)
	{
		output(currentPiece->stringText);

		currentPiece = currentPiece->nextString;
	}
}

void decudaInst::printTypeModifiers()
{
	stringListPiece* currentPiece = m_typeModifiers->getListStart();
	for(int i=0; (i<m_typeModifiers->getSize())&&(currentPiece!=NULL); i++)
	{
		output(currentPiece->stringText);

		currentPiece = currentPiece->nextString;
	}
}

void decudaInst::printOperands()
{
	stringListPiece* currentPiece = m_operands->getListStart();
	const char* operandDelimiter = "";
	for(int i=0; (i<m_operands->getSize())&&(currentPiece!=NULL); i++)
	{
		output(operandDelimiter); output(" "); output(currentPiece->stringText);
		operandDelimiter = ",";

		currentPiece = currentPiece->nextString;
	}
}

//This is where the conversion to new PTX takes place
void decudaInst::printNewPtx()
{
	//
	// Common modifications that apply to all instructions
	//
	stringListPiece* currentPiece;
        int vectorFlag[4] = {0,0,0,0}; //0=16/32type, 1=bb64/ff64 type, 2=bb128 type

	// Replace '%clock' with '%halfclock'
	currentPiece = m_operands->getListStart();
	for(int i=0; (i<m_operands->getSize())&&(currentPiece!=NULL); i++)
	{
		const char* modString = currentPiece->stringText;
		if( strcmp(modString, "%%clock")==0 ) {
			const char* newText = "%%halfclock";
			currentPiece->stringText = newText;
		}
		currentPiece = currentPiece->nextString;
	}

	// Remove .join from base modifier list
	currentPiece = m_baseModifiers->getListStart();
	for(int i=0; (i<m_baseModifiers->getSize())&&(currentPiece!=NULL); i++)
	{
		const char* modString = currentPiece->stringText;
		if( strcmp(modString, ".join")==0 ) {
			m_baseModifiers->remove(i);
		}
		currentPiece = currentPiece->nextString;
	}

	// Change .end to .exit from base modified list
	currentPiece = m_baseModifiers->getListStart();
	for(int i=0; (i<m_baseModifiers->getSize())&&(currentPiece!=NULL); i++)
	{
		const char* modString = currentPiece->stringText;
		if( strcmp(modString, ".end")==0 ) {
			const char* newText = ".exit";
			currentPiece->stringText = newText;
		}
		currentPiece = currentPiece->nextString;
	}

	// Change .b64 to .bb64 from type modifier list
	// Change .b128 to .bb128 from type modifier list
	// Change .f64 to .ff64 from type modifier list
	currentPiece = m_typeModifiers->getListStart();
	for(int i=0; (i<m_typeModifiers->getSize())&&(currentPiece!=NULL); i++)
	{
		const char* modString = currentPiece->stringText;
		if( strcmp(modString, ".b64")==0 ) {
			const char* newText = ".bb64";
			currentPiece->stringText = newText;
			vectorFlag[i] = 1;
		}
		else if( strcmp(modString, ".b128")==0 ) {
			const char* newText = ".bb128";
			currentPiece->stringText = newText;
			vectorFlag[i] = 2;
		}
		else if( strcmp(modString, ".f64")==0 ) {
			const char* newText = ".ff64";
			currentPiece->stringText = newText;
			vectorFlag[i] = 1;
		}
		currentPiece = currentPiece->nextString;
	}

	/*decuda bug workaround.
	cvt.abs should drop the first type modifier.
	cvt.rn.f32.f64 needs to have type modifiers reversed to cvt.ff64.f32*/
	int absFound = 0;
	if(strcmp(m_base, "cvt")==0 && m_typeModifiers->getSize() == 2)
	{
		stringListPiece* currentPieceCvt = m_baseModifiers->getListStart();
		for(int i=0; (i<m_baseModifiers->getSize())&&(currentPieceCvt!=NULL); i++)
		{
                        const char* modStringCvt = currentPieceCvt->stringText;
			if( strcmp(modStringCvt, ".abs")==0 ) {
				vectorFlag[0] = vectorFlag[1];
				absFound = 1;
				break;
			}
			currentPieceCvt = currentPieceCvt->nextString;
		}
		if(absFound == 0 && vectorFlag[1] == 1)
		{
			vectorFlag[0] = 1;
			vectorFlag[1] = 0;
			const char* tempCharPtr = m_typeModifiers->getListStart()->stringText;
			m_typeModifiers->getListStart()->stringText = m_typeModifiers->getListStart()->nextString->stringText;
			m_typeModifiers->getListStart()->nextString->stringText = tempCharPtr;
		}
	}

	/*decuda bug workaround, cvt.rz.f64 is really cvt.rz.f32.f64*/
	if(strcmp(m_base, "cvt")==0 && m_typeModifiers->getSize() == 1 && vectorFlag[0]==1)
	{
		vectorFlag[0] = 0;
		vectorFlag[1] = 1;
		addTypeModifier(m_typeModifiers->getListStart()->stringText);
		const char* newText = ".f32";
		m_typeModifiers->getListStart()->stringText = newText;
	}

	/*expand vector operands eg. $r0 -> {$r0, $r1}*/
	if((vectorFlag[0] != 0) || (vectorFlag[1] != 0) || (vectorFlag[2] != 0) || (vectorFlag[3] != 0) )
	{
		currentPiece = m_operands->getListStart();
		for(int i=0; (i<m_operands->getSize())&&(currentPiece!=NULL); i++)
		{
			char *newText = new char[40];
			char *regNumString;
			int regNumInt;

			const char* modString = currentPiece->stringText;

			if(strcmp(m_base, "set")==0 || strcmp(m_base, "cvt")==0 || 
				strcmp(m_base, "set?68?")==0 || strcmp(m_base, "set?65?")==0 ||
				strcmp(m_base, "set?67?")==0 || strcmp(m_base, "set?13?")==0)
			{
				if( modString[0] == '$' && modString[1] == 'r' ) {
					strcpy(newText, modString);
					strtok (newText, "r");
					regNumString = strtok (NULL, "r");
					regNumInt = atoi(regNumString);
					if(vectorFlag[i] ==0)
						strcpy(newText, modString);
					if(vectorFlag[i] ==1)
						snprintf(newText,40,"{$r%u,$r%u}", regNumInt+0, regNumInt+1);
					if(vectorFlag[i] ==2)
						snprintf(newText,40,"{$r%u,$r%u,$r%u,$r%u}", regNumInt+0, regNumInt+1, regNumInt+2, regNumInt+3);

					currentPiece->stringText = newText;
				} else if( modString[0] == '-' && modString[1] == '$' && modString[2] == 'r' ) {
					strcpy(newText, modString);
					strtok (newText, "r");
					regNumString = strtok (NULL, "r");
					regNumInt = atoi(regNumString);
					if(vectorFlag[i] ==0)
						strcpy(newText, modString);
					else if(vectorFlag[i] ==1)
						snprintf(newText,40,"-{$r%u,$r%u}", regNumInt+0, regNumInt+1);
					else if(vectorFlag[i] ==2)
						snprintf(newText,40,"-{$r%u,$r%u,$r%u,$r%u}", regNumInt+0, regNumInt+1, regNumInt+2, regNumInt+3);

					currentPiece->stringText = newText;
				}
			}
			else
			{
				if( modString[0] == '$' && modString[1] == 'r' ) {
					strcpy(newText, modString);
					strtok (newText, "r");
					regNumString = strtok (NULL, "r");
					regNumInt = atoi(regNumString);
					if(vectorFlag[0] ==0)
						strcpy(newText, modString);
					else if(vectorFlag[0] ==1)
						snprintf(newText,40,"{$r%u,$r%u}", regNumInt+0, regNumInt+1);
					else if(vectorFlag[0] ==2)
						snprintf(newText,40,"{$r%u,$r%u,$r%u,$r%u}", regNumInt+0, regNumInt+1, regNumInt+2, regNumInt+3);

					currentPiece->stringText = newText;
				} else if( modString[0] == '-' && modString[1] == '$' && modString[2] == 'r' ) {
					strcpy(newText, modString);
					strtok (newText, "r");
					regNumString = strtok (NULL, "r");
					regNumInt = atoi(regNumString);
					if(vectorFlag[0] ==0)
						strcpy(newText, modString);
					else if(vectorFlag[0] ==1)
						snprintf(newText,40,"-{$r%u,$r%u}", regNumInt+0, regNumInt+1);
					else if(vectorFlag[0] ==2)
						snprintf(newText,40,"-{$r%u,$r%u,$r%u,$r%u}", regNumInt+0, regNumInt+1, regNumInt+2, regNumInt+3);

					currentPiece->stringText = newText;
				}
			}
			currentPiece = currentPiece->nextString;
		}
	}

	//
	// Instruction specific modifications
	//
	if(strcmp(m_base, "")==0)
	{
	}
	else if(strcmp(m_base, ".entry")==0)
	{
		/*do nothing here*/
	}
	else if(strcmp(m_base, ".lmem")==0)
	{
	}
	else if(strcmp(m_base, ".smem")==0)
	{
	}
	else if(strcmp(m_base, ".reg")==0)
	{
	}
	else if(strcmp(m_base, ".bar")==0)
	{
	}


	else if(strcmp(m_base, "cvt")==0)
	{

		int cvt_inst_type = 0;	//0==cvt, 1==abs
                int cvt_neg_mod_flag = 0; //0=off, 1=on

		// Check the actual base instruction
		stringListPiece* currentPiece = m_baseModifiers->getListStart();
		for(int i=0; (i<m_baseModifiers->getSize())&&(currentPiece!=NULL); i++)
		{
                        const char* modString = currentPiece->stringText;
			if( strcmp(modString, ".abs")==0 ) {
				cvt_inst_type = 1;
			}
			if( strcmp(modString, ".neg")==0 ) {
				cvt_neg_mod_flag = 1;
			}
			currentPiece = currentPiece->nextString;
		}

		if( cvt_inst_type == 1) {
			// abs instruction
			printLabel();
			printPredicate();

			output("abs");

			if(m_typeModifiers->getSize() == 1)
			{
				const char* type = m_typeModifiers->getListStart()->stringText;
				output(type);
			}
			else
			{
				const char* type = m_typeModifiers->getListStart()->nextString->stringText;
				output(type);
			}

			printOperands();

			output(";");

		} else {
			// cvt instruction
			printLabel();
			printPredicate();

			output("cvt");

			int typeModifiers = m_typeModifiers->getSize();

			if(typeModifiers == 2) {

				stringListPiece* currentPiece = m_baseModifiers->getListStart();

				if(cvt_neg_mod_flag != 0)
				    currentPiece = m_baseModifiers->getListStart()->nextString;

				const char* dstType = m_typeModifiers->getListStart()->stringText;
				const char* srcType = m_typeModifiers->getListStart()->nextString->stringText;

				for(int i=0; (i<m_baseModifiers->getSize())&&(currentPiece!=NULL); i++)
				{
				        const char* modString = currentPiece->stringText;
					if(
						(strcmp(modString, ".rn")==0) || 
						(strcmp(modString, ".rm")==0) ||
						(strcmp(modString, ".rp")==0) ||
						(strcmp(modString, ".rz")==0)
					)
					{
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".ff64")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".ff64")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".f32")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".u8")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".u16")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".u32")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".u64")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".s8")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".s16")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".s32")==0))
							output(modString);
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".s64")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".u8")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".u16")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".u32")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".u64")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".s8")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".s16")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".s32")==0))
							output(modString);
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".s64")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".u8")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".u16")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".u32")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".u64")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".s8")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".s16")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".s32")==0))
							output(modString);
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".s64")==0))
							output(modString);
						if((strcmp(dstType, ".u8")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u16")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u32")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u64")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s8")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s16")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s32")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s64")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u8")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u16")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u32")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u64")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s8")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s16")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s32")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s64")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u8")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u16")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u32")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".u64")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s8")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s16")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s32")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".s64")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".f16")==0) && (strcmp(srcType, ".f16")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".f32")==0) && (strcmp(srcType, ".f32")==0))
							{ output(modString); output("i"); }
						if((strcmp(dstType, ".ff64")==0) && (strcmp(srcType, ".ff64")==0))
							{ output(modString); output("i"); }
					}
					else if (						
						(strcmp(modString, ".rni")==0 || strcmp(modString, ".rmi")==0 ||
						 strcmp(modString, ".rpi")==0 || strcmp(modString, ".rzi")==0)
					)
					{
						output(modString);
					}
					else
					{
						output("\nUnknown mod:"); output(currentPiece->stringText);
						assert(0);
					}

					currentPiece = currentPiece->nextString;
				}
			} else {
				printBaseModifiers();
			}

			// If one type modifier, duplicate to two;
			int numModifiers = m_typeModifiers->getSize();
			stringListPiece* currentPiece = m_typeModifiers->getListStart();
			if( numModifiers == 1 ) {
				output(currentPiece->stringText);
				output(currentPiece->stringText);
			} else {
				printTypeModifiers();
			}

			printOperands();

			output(";");

		}
		
	}
	else if(strcmp(m_base, "shl")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "add")==0)
	{
		printLabel();
		printPredicate();

		output(m_base);

		printBaseModifiers();

		currentPiece = m_typeModifiers->getListStart();
		for(int i=0; (i<m_typeModifiers->getSize())&&(currentPiece!=NULL); i++)
		{
			if(strcmp(currentPiece->stringText, ".b8")==0)
			{
				output(".u8");
			}
			else if(strcmp(currentPiece->stringText, ".b16")==0)
			{
				output(".u16");
			}
			else if(strcmp(currentPiece->stringText, ".b32")==0)
			{
				output(".u32");
			}
			else if(strcmp(currentPiece->stringText, ".b64")==0)
			{
				output(".u64");
			}
			else
			{
				output(currentPiece->stringText);
			}

			currentPiece = currentPiece->nextString;
		}

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "movsh")==0)
	{
		printLabel();
		printPredicate();

		output("shl");

		printBaseModifiers();

		printTypeModifiers();

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "mov")==0)
	{
		printLabel();
		printPredicate();

		output("mov");

		printBaseModifiers();

		// If mov has two type modifiers, pick the smaller one for proper
		// bit truncation
		if( m_typeModifiers->getSize() == 2 ) {
			std::string type1, type2, type;
			int type1Size, type2Size;
			stringListPiece* currentPiece = m_typeModifiers->getListStart();
			type1 = currentPiece->stringText;
			type2 = currentPiece->nextString->stringText;

			type1Size = atoi(type1.substr(2, type1.size()-2).c_str());
			type2Size = atoi(type2.substr(2, type2.size()-2).c_str());
			type = (type1Size < type2Size) ? type1 : type2;
			output(type.c_str());
		} else if( m_typeModifiers->getSize() == 1 ) {
			printTypeModifiers();
		} else {
			output("Error: unsupported number of type modifiers. ");
		}

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "bar.sync")==0)
	{
		printLabel();
		printPredicate();

		output(m_base);

		printBaseModifiers();

		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		currentPiece = m_typeModifiers->getListStart();
		for(int i=0; (i<m_typeModifiers->getSize())&&(currentPiece!=NULL); i++)
		{
			//output(currentPiece->stringText);

			currentPiece = currentPiece->nextString;
		}

		printOperands();

		output(";");
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "mul")==0)
	{
		printDefaultPtx();

      // opPerCycle - lower if a 32-bit integer mul
      const char* dstType = m_typeModifiers->getListStart()->stringText;
      if( strcmp(dstType, ".s32")==0 || strcmp(dstType, ".u32")==0 || strcmp(dstType, ".b32")==0 )
         m_opPerCycle = 2;
	}
	else if(strcmp(m_base, "mad24")==0)
	{
		printLabel();
		printPredicate();

		output("mad24");

		printBaseModifiers();

		// Only output the destination operand type (first type modifier only)
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		output(currentPiece->stringText);

		printOperands();
		// If 3 operands, this is a mac24 instruction, so add destination operand as 4th operand
		int numOperands = m_operands->getSize();
		if(numOperands == 3) {
			stringListPiece* currentPiece = m_operands->getListStart();
			output(", "); output(currentPiece->stringText);
		}

		output(";");
	}
	else if(strcmp(m_base, "mad24c1")==0)
	{
		output("nop; //");

		printLabel();
		printPredicate();

		output("mad24c1");

		printBaseModifiers();

		// Only output the destination operand type (first type modifier only)
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		output(currentPiece->stringText);

		printOperands();
		// If 3 operands, this is a mac24 instruction, so add destination operand as 4th operand
		int numOperands = m_operands->getSize();
		if(numOperands == 3) {
			stringListPiece* currentPiece = m_operands->getListStart();
			output(", "); output(currentPiece->stringText);
		}

		output(";");
	}
	else if(strcmp(m_base, "set")==0)
	{
		printLabel();
		printPredicate();

		output("set");

		printBaseModifiers();

		// If one type modifier, duplicate to two;
		// if three type modifiers, remove third
		int numModifiers = m_typeModifiers->getSize();
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		if( numModifiers == 1 ) {
			output(currentPiece->stringText);
			output(currentPiece->stringText);
		} else if( numModifiers == 3) {
			for(int i=0; (i<numModifiers-1)&&(currentPiece!=NULL); i++)
			{
				output(currentPiece->stringText);
				currentPiece = currentPiece->nextString;
			}
		}

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "mad")==0)
	{
		printDefaultPtx();

      // opPerCycle - lower if a 32-bit integer mad
      const char* dstType = m_typeModifiers->getListStart()->stringText;
      if( strcmp(dstType, ".s32")==0 || strcmp(dstType, ".u32")==0 || strcmp(dstType, ".b32")==0 )
         m_opPerCycle = 2;
	}
	else if(strcmp(m_base, "mul24")==0)
	{
		printLabel();
		printPredicate();

		output("mul24");

		printBaseModifiers();

		// Only output the destination operand type (first type modifier only)
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		output(currentPiece->stringText);

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "set?68?")==0)
	{
		// This actually takes absolute value of first source operand and is set.gt
		printLabel();
		printPredicate();

		output("set.gt.abs");

		printBaseModifiers();

		// If one type modifier, duplicate to two;
		// if three type modifiers, remove third
		int numModifiers = m_typeModifiers->getSize();
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		if( numModifiers == 1 ) {
			output(currentPiece->stringText);
			output(currentPiece->stringText);
		} else if( numModifiers == 3) {
			for(int i=0; (i<numModifiers-1)&&(currentPiece!=NULL); i++)
			{
				output(currentPiece->stringText);
				currentPiece = currentPiece->nextString;
			}
		}

		printOperands();

		output(";");

		output(" //set?68?");
	}
	else if(strcmp(m_base, "set?65?")==0)
	{
		// This actually takes absolute value of first source operand and is set.lt
		printLabel();
		printPredicate();

		output("set.lt.abs");

		printBaseModifiers();

		// If one type modifier, duplicate to two;
		// if three type modifiers, remove third
		int numModifiers = m_typeModifiers->getSize();
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		if( numModifiers == 1 ) {
			output(currentPiece->stringText);
			output(currentPiece->stringText);
		} else if( numModifiers == 3) {
			for(int i=0; (i<numModifiers-1)&&(currentPiece!=NULL); i++)
			{
				output(currentPiece->stringText);
				currentPiece = currentPiece->nextString;
			}
		}

		printOperands();

		output(";");

		output(" //set?65?");
	}
	else if(strcmp(m_base, "set?67?")==0)
	{
		// Change to set.gt
		printLabel();
		printPredicate();

		output("set.gt.abs");

		printBaseModifiers();

		// If one type modifier, duplicate to two;
		// if three type modifiers, remove third
		int numModifiers = m_typeModifiers->getSize();
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		if( numModifiers == 1 ) {
			output(currentPiece->stringText);
			output(currentPiece->stringText);
		} else if( numModifiers == 3) {
			for(int i=0; (i<numModifiers-1)&&(currentPiece!=NULL); i++)
			{
				output(currentPiece->stringText);
				currentPiece = currentPiece->nextString;
			}
		}

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "set?13?")==0)
	{
		// Change to set.gt
		printLabel();
		printPredicate();

		output("set.neu");

		printBaseModifiers();

		// If one type modifier, duplicate to two;
		// if three type modifiers, remove third
		int numModifiers = m_typeModifiers->getSize();
		stringListPiece* currentPiece = m_typeModifiers->getListStart();
		if( numModifiers == 1 ) {
			output(currentPiece->stringText);
			output(currentPiece->stringText);
		} else if( numModifiers == 3) {
			for(int i=0; (i<numModifiers-1)&&(currentPiece!=NULL); i++)
			{
				output(currentPiece->stringText);
				currentPiece = currentPiece->nextString;
			}
		}

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "rcp")==0)
	{
		printDefaultPtx();

      m_opPerCycle = 2;
	}
	else if(strcmp(m_base, "pre.sin")==0)
	{
		printLabel();
		printPredicate();

		output("nop;");
		output(" //");
		output("pre.sin");
		printBaseModifiers();
		printTypeModifiers();
		printOperands();

      // m_opPerCycle - Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "sin")==0)
	{
		printDefaultPtx();

      m_opPerCycle = 1;
	}
	else if(strcmp(m_base, "pre.ex2")==0)
	{
		printLabel();
		printPredicate();

		output("ex2");
		printBaseModifiers();
		printTypeModifiers();
		printOperands();
		output(";");

		output(" //");
		printDefaultPtx();

      // m_opPerCycle - Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "ex2")==0)
	{
		printLabel();
		printPredicate();

		output("nop;");
		output(" //");
		output("ex2");
		printBaseModifiers();
		printTypeModifiers();
		printOperands();

      m_opPerCycle = 1;
	}
	else if(strcmp(m_base, "cos")==0)
	{
		printDefaultPtx();

      m_opPerCycle = 1;
	}
	else if(strcmp(m_base, "lg2")==0)
	{
		printDefaultPtx();

      m_opPerCycle = 2;
	}
	else if(strcmp(m_base, "rsqrt")==0)
	{
		printDefaultPtx();

      m_opPerCycle = 2;
	}
	else if(strcmp(m_base, "mac")==0)
	{
		// Replace with mad by adding a 4th operand
		printLabel();
		printPredicate();

		output("mad");

		printBaseModifiers();

		printTypeModifiers();

		// Print operands and then include destination (1st) operand as the 4th operand
		printOperands();
		stringListPiece* currentPiece = m_operands->getListStart();
		output(","); output(" "); output(currentPiece->stringText);

		output(";");

      // opPerCycle - lower if a 32-bit integer mac
      const char* dstType = m_typeModifiers->getListStart()->stringText;
      if( strcmp(dstType, ".s32")==0 || strcmp(dstType, ".u32")==0 || strcmp(dstType, ".b32")==0 )
         m_opPerCycle = 2;
	}
	else if(strcmp(m_base, "bra.label")==0)
	{
		printLabel();
		printPredicate();

		output("bra");

		printBaseModifiers();

		printTypeModifiers();

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "join.label")==0)
	{
		printLabel();
		printPredicate();

		output("nop;");
		output(" //join.label");

      // Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "nop.end")==0)
	{
		printLabel();
		printPredicate();

		output("nop.exit;");

      // Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "nop.join")==0)
	{
		printLabel();
		printPredicate();

		output("nop;");

      // Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "nop")==0)
	{
		//printLabel();
                //printPredicate();

                output("nop;");
	}
	else if(strcmp(m_base, "return")==0)
	{
		// ret instruction causes a deadlock bug in the simulator
		// Temporary fix: branch to a dummy exit instruction that is added to the end of each
		//                entry with the label 'l_exit'
		printLabel();
		printPredicate();

		output("retp");
		//output("bra l_exit");

		printBaseModifiers();

		printTypeModifiers();

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "and")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "andn")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "tex")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "xor")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "or")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "shr")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "subr")==0)
	{

		// Replace with sub instruction with destination operands switched
		printLabel();
		printPredicate();

		output("sub");

		printBaseModifiers();

		printTypeModifiers();

		// Switch source operands before printing
		// Must be 3 operands, switch the last two;
		if( m_operands->getSize() != 3) {
			output("Error: subr instruction with number of operands other than 3.\n");
			assert(0);
		}
		const char* firstOperand = m_operands->getListStart()->stringText;
		const char* secondOperand = m_operands->getListStart()->nextString->stringText;
		const char* thirdOperand = m_operands->getListStart()->nextString->nextString->stringText;
		output(" "); output(firstOperand); output(",");
		output(" "); output(thirdOperand); output(",");
		output(" "); output(secondOperand);
		

		output(";");
	}
	else if(strcmp(m_base, "sub")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "max")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "min")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "call.label")==0)
	{
		printLabel();
		printPredicate();

		output("callp");

		printBaseModifiers();

		printTypeModifiers();

		printOperands();

		output(";");

      // Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "not")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "delta")==0)
	{
		// This is a neg instruction
		printLabel();
		printPredicate();

		output("neg");

		printBaseModifiers();

		printTypeModifiers();

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "break")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "breakaddr.label")==0)
	{
		printLabel();
		printPredicate();

		output("breakaddr");

		printBaseModifiers();

		printTypeModifiers();

		printOperands();

		output(";");
	}
	else if(strcmp(m_base, "inc")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "exch")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "cas")==0)
	{
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "norn")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "addc")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}
	else if(strcmp(m_base, "orn")==0)
	{
		printDefaultPtx();
	}
	else if(strcmp(m_base, "nandn")==0)
	{
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "nxor")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "sad")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "op.13")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "op.e5")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "op.e6")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}
   	else if(strcmp(m_base, "op.d0")==0)
	{
		output("nop; //");
		printDefaultPtx();
	}


	else if(strcmp(m_base, "{")==0)
	{
		//output(m_base);

      // Ignore
      m_opPerCycle = -1;
	}
	else if(strcmp(m_base, "}")==0)
	{
		//output(m_base);

      // Ignore
      m_opPerCycle = -1;
	}
	else
	{
		output("Unknown Instruction: "); output(m_base);
		assert(0);
	}

}
