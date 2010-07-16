
/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


/**
	@author Svetlin Manavski <svetlin@manavski.com>
 */

#include "aesCudaUtils.h"
#include <string.h>

extern unsigned Rcon[];
extern unsigned SBox[];
extern unsigned LogTable[];
extern unsigned ExpoTable[];

extern unsigned MODE;

//------------------------------------------help-functions-------------------------------------------------------
unsigned mySbox(unsigned num) { 
	return SBox[num]; 
}

unsigned myXor(unsigned num1, unsigned num2) {
	return num1 ^ num2;
}

//------------------------------------------help-functions-------------------------------------------------------


//------------------------------------------AESCudaUtils-----------------------------------------------------------
//------------------------------------------AESCudaUtils-----------------------------------------------------------
//------------------------------------------AESCudaUtils-----------------------------------------------------------

// gestisce la riga di commando
unsigned commandLineManager(int argc, char *argv[]) {

	if ( argc<5 )
		usage();

	if ( argc%2 != 1)
		usage();

	if ( (*argv[1] != 'e') && (*argv[1] != 'd') )
		usage();

	if ( ( strcmp(argv[2], "256") ) && ( strcmp(argv[2], "128") ) )
		usage();

	if (*argv[1] != 'e') 
		MODE = 0;

	//controllo esistenza file in futuro

	return ( (argc-3) / 2 );
}

void usage() {
	std::cout << "\nAES - CUDA by Svetlin Manavski" << std::endl;
	std::cout << "Version 0.90" << std::endl << std::endl;
	std::cout << "Usage:" << std::endl << std::endl;
	std::cout << "aescuda <e|d> <128|256> input1 key1 input2 key2 .... inputN keyN" << std::endl;
	std::cout << "e\t- means to encrypt the input file" << std::endl;
	std::cout << "d\t- means to decrypt the input file" << std::endl;
	std::cout << "128/256\t- selects the aes type" << std::endl << std::endl;
	std::cout << "input file maximum size: 34.603.007 bytes" << std::endl;
	std::cout << "key file must contain 32 or 16 elements in hex format separated by blanks\n\n";
	exit(0);
}

//funzione principale
unsigned initAesCuda(std::string myKeyFile, unsigned char myKeyBuffer[], const unsigned int myKeyBitsSize, std::string myInputFile, char inputArray[], const unsigned inputArraySize){
	
	path inputPath(myInputFile.c_str());
	path keyPath(myKeyFile.c_str());
	
	if ( !exists(keyPath) )
		throw std::string("file "+keyPath.string()+" doesn't exist");

	if ( !exists(inputPath) )
		throw std::string("file "+inputPath.string()+" doesn't exist");
	
	if ( myKeyBitsSize!=256 && myKeyBitsSize!=128)
		throw std::string("cannot use a key dimension different from 256 or 128");

	if ( !myKeyBuffer )
		throw std::string("key array not allocated");

	if ( !inputArray )
		throw std::string("input array not allocated");

	boost::intmax_t inputFileSize	= getFileSize(inputPath);
	boost::intmax_t keyFileSize		= getFileSize(keyPath);

	if ( keyFileSize==0 ) 
		throw std::string("cannot use an empty input file");

	if ( inputFileSize==0 ) 
		throw std::string("cannot use an empty key file");

	if ( inputFileSize > inputArraySize - 1 && MODE) 
		throw std::string("cannot encrypt a file bigger than 34.603.007 bytes");

	if ( inputFileSize > inputArraySize && !MODE) 
		throw std::string("cannot decrypt a file bigger than 33MB");

	//legge l'input
	readFromFileNotForm(inputPath, inputArray, inputFileSize);
	
	std::vector<unsigned> keyArray(myKeyBitsSize/8);
	
	unsigned ekSize = (myKeyBitsSize != 256) ? 176 : 240;

	std::vector<unsigned> expKeyArray(ekSize);
	std::vector<unsigned> invExpKeyArray(ekSize);
	
	//legge la chiave
	readFromFileForm(keyPath, keyArray);

	std::cout << "\n###############################################################\n\n";
	std::cout << "AES - CUDA by Svetlin Manavski)\n\n";
	std::cout << "AES " << myKeyBitsSize << " is running...." << std::endl << std::endl;
	std::cout << "Input file size: " << inputFileSize << " Bytes" << std::endl << std::endl;
	std::cout << "Key: ";
	for (unsigned cnt=0; cnt<keyArray.size(); ++cnt)
		std::cout << std::hex << keyArray[cnt];

	if (MODE){
		//ENCRYPTION MODE

		//PADDING MANAGEMENT FOLLOWING THE PKCS STANDARD
		unsigned mod16 = inputFileSize % 16;
		unsigned div16 = inputFileSize / 16;

		unsigned padElem;
		if ( mod16 != 0 )
			padElem =  16 - mod16;
		else 
			padElem =  16;

		for (unsigned cnt = 0; cnt < padElem; ++cnt)
				inputArray[div16*16 + mod16 + cnt] = padElem;

		inputFileSize = inputFileSize + padElem;
		
		//IN THE ENCRYPTION MODE I NEED THE EXPANDED KEY
		expFunc(keyArray, expKeyArray);
		for (unsigned cnt=0; cnt<expKeyArray.size(); ++cnt){
			unsigned val = expKeyArray[cnt];
			unsigned char *pc = reinterpret_cast<unsigned char *>(&val);
			myKeyBuffer[cnt] = *(pc);
		}
	} else {
		//DECRYPTION MODE 

		//IN THE ENCRYPTION MODE I NEED THE INVERSE EXPANDED KEY
		expFunc(keyArray, expKeyArray);
		invExpFunc(expKeyArray, invExpKeyArray);
		for (unsigned cnt=0; cnt<invExpKeyArray.size(); ++cnt){
			unsigned val = invExpKeyArray[cnt];
			unsigned char *pc = reinterpret_cast<unsigned char *>(&val);
			myKeyBuffer[cnt] = *(pc);
		}
	}
	std::cout << std::endl;

	return inputFileSize;
}


boost::intmax_t getFileSize(path &myPath){
	if ( !exists(myPath) )
		throw std::string("file "+myPath.string()+" doesn't exist");

	return file_size(myPath);
}

//legge file non formattati come l'input
void readFromFileNotForm(path &myPath, char *storingArray, unsigned dataSize){
    if ( !exists(myPath) )
		throw std::string("file "+myPath.string()+" doesn't exist");
	if ( !storingArray )
		throw std::string("readFromFileNotForm: array not allocated");

	std::ifstream inStream;
	inStream.open( myPath.string().c_str(), std::ifstream::binary );
	inStream.read(storingArray, dataSize);
	inStream.close();
}

//legge file formattati come la chiave (esadecimali separati da spazi)
void readFromFileForm(path &myPath, std::vector<unsigned> &storingArray){
	if ( !exists(myPath) )
		throw std::string("file "+myPath.string()+" doesn't exist");
	if ( storingArray.size()!=32 && storingArray.size()!=16)
		throw std::string("readFromFileForm: storing array of wrong dimension");

	std::ifstream inStream;
	inStream>>std::hex;
	inStream.open( myPath.string().c_str() );
	
	for (unsigned cnt=0; cnt<storingArray.size(); ++cnt){
		inStream >> storingArray[cnt];
		if ( ( inStream.eof() ) && ( cnt != storingArray.size()-1 ) ) 
			throw std::string("cannot use a key with less than 32 or 16 elements ");
		if ( ( !inStream.eof() ) && ( cnt == storingArray.size()-1 ) ) 
			std::cout << "WARNING: your key file has more than 32 or 16 elements. It will be cut down to this threshold\n";
		if ( inStream.fail() ) 
			throw std::string("Check that your key file elements are in hexadecimal format separeted by blanks");
	}
	
	inStream.close();
}

//espande la chiave
void expFunc(std::vector<unsigned> &keyArray, std::vector<unsigned> &expKeyArray){
	if ( keyArray.size()!=32 && keyArray.size()!=16 )
		throw std::string("expFunc: key array of wrong dimension");
	if ( expKeyArray.size()!=240 && expKeyArray.size()!=176 )
		throw std::string("expFunc: expanded key array of wrong dimension");

	copy(keyArray.begin(), keyArray.end(), expKeyArray.begin());

	unsigned cycles = (expKeyArray.size()!=240) ? 11 : 8;

	for (unsigned i=1; i<cycles; ++i){
		singleStep(expKeyArray, i);
	}
}

void singleStep(std::vector<unsigned> &expKey, unsigned stepIdx){
	if ( expKey.size()!=240 && expKey.size()!=176 )
		throw std::string("singleStep: expanded key array of wrong dimension");
	if ( stepIdx<1 && stepIdx>11 )
		throw std::string("singleStep: index out of range");

	unsigned num = (expKey.size()!=240) ? 16 : 32;
	unsigned idx = (expKey.size()!=240) ? 16*stepIdx : 32*stepIdx;

	copy(expKey.begin()+(idx)-4, expKey.begin()+(idx),expKey.begin()+(idx));
	rotate(expKey.begin()+(idx), expKey.begin()+(idx)+1, expKey.begin()+(idx)+4);

	transform(expKey.begin()+(idx), expKey.begin()+(idx)+4, expKey.begin()+(idx), mySbox);
	
	expKey[idx] = expKey[idx] ^ Rcon[stepIdx-1];

	transform(expKey.begin()+(idx), expKey.begin()+(idx)+4, expKey.begin()+(idx)-num, expKey.begin()+(idx), myXor);

	for (unsigned cnt=0; cnt<3; ++cnt){
		copy(expKey.begin()+(idx)+4*cnt, expKey.begin()+(idx)+4*(cnt+1),expKey.begin()+(idx)+(4*(cnt+1)));
		transform(expKey.begin()+(idx)+4*(cnt+1), expKey.begin()+(idx)+4*(cnt+2), expKey.begin()+(idx)-(num-4*(cnt+1)), expKey.begin()+(idx)+4*(cnt+1), myXor);
	}

	if(stepIdx!=7 && expKey.size()!=176){
		copy(expKey.begin()+(idx)+12, expKey.begin()+(idx)+16,expKey.begin()+(idx)+16);
		transform(expKey.begin()+(idx)+16, expKey.begin()+(idx)+20, expKey.begin()+(idx)+16, mySbox);
		transform(expKey.begin()+(idx)+16, expKey.begin()+(idx)+20, expKey.begin()+(idx)-(32-16), expKey.begin()+(idx)+16, myXor);

		for (unsigned cnt=4; cnt<7; ++cnt){
			copy(expKey.begin()+(idx)+4*cnt, expKey.begin()+(idx)+4*(cnt+1),expKey.begin()+(idx)+(4*(cnt+1)));
			transform(expKey.begin()+(idx)+4*(cnt+1), expKey.begin()+(idx)+4*(cnt+2), expKey.begin()+(idx)-(32-4*(cnt+1)), expKey.begin()+(idx)+4*(cnt+1), myXor);
		}
	}
}

//espande la chiave inversa per la decriptazione
void invExpFunc(std::vector<unsigned> &expKey, std::vector<unsigned> &invExpKey){
	if ( expKey.size()!=240 && expKey.size()!=176 )
		throw std::string("invExpFunc: expanded key array of wrong dimension");
	if ( invExpKey.size()!=240 && invExpKey.size()!=176 )
		throw std::string("invExpFunc: inverse expanded key array of wrong dimension");

	std::vector<unsigned> temp(16);

	copy(expKey.begin(), expKey.begin()+16,invExpKey.end()-16);
	copy(expKey.end()-16, expKey.end(),invExpKey.begin());
	
	unsigned cycles = (expKey.size()!=240) ? 10 : 14;

	for (unsigned cnt=1; cnt<cycles; ++cnt){
		copy(expKey.end()-(16*cnt+16), expKey.end()-(16*cnt), temp.begin());
		invMixColumn(temp);
		copy(temp.begin(), temp.end(), invExpKey.begin()+(16*cnt));
	}
}

void invMixColumn(std::vector<unsigned> &temp){
	if ( temp.size()!=16 )
		throw std::string("invMixColumn: array of wrong dimension");

	std::vector<unsigned> result(4);
	
	for(unsigned cnt=0; cnt<4; ++cnt){
		result[0] = galoisProd(0x0e, temp[cnt*4]) ^ galoisProd(0x0b, temp[cnt*4+1]) ^ galoisProd(0x0d, temp[cnt*4+2]) ^ galoisProd(0x09, temp[cnt*4+3]);
		result[1] = galoisProd(0x09, temp[cnt*4]) ^ galoisProd(0x0e, temp[cnt*4+1]) ^ galoisProd(0x0b, temp[cnt*4+2]) ^ galoisProd(0x0d, temp[cnt*4+3]);
		result[2] = galoisProd(0x0d, temp[cnt*4]) ^ galoisProd(0x09, temp[cnt*4+1]) ^ galoisProd(0x0e, temp[cnt*4+2]) ^ galoisProd(0x0b, temp[cnt*4+3]);
		result[3] = galoisProd(0x0b, temp[cnt*4]) ^ galoisProd(0x0d, temp[cnt*4+1]) ^ galoisProd(0x09, temp[cnt*4+2]) ^ galoisProd(0x0e, temp[cnt*4+3]);
	
		copy(result.begin(), result.end(), temp.begin()+(4*cnt));
	}
}

//prodotto di Galois di due numeri
unsigned galoisProd(unsigned a, unsigned b){
	
	if(a==0 || b==0) return 0;
	else {
		a = LogTable[a];
		b = LogTable[b];
		a = a+b;
		a = a % 255;
		a = ExpoTable[a];
		return a;
	}
}

//scrive su file il risultato
void writeToFile(const std::string &outPath, char *storingArray, boost::intmax_t dataSize, unsigned maxInputSize){
	if ( !storingArray )
		throw std::string("writeToFile: array not allocated");
	
	if ( dataSize >  maxInputSize)
		dataSize = maxInputSize;

	std::ofstream outStream;
	outStream.open( outPath.c_str() , std::ifstream::binary);

	if (!MODE)
		dataSize = dataSize - storingArray[dataSize-1];

	outStream.write(storingArray, dataSize);
	outStream.close();
}

