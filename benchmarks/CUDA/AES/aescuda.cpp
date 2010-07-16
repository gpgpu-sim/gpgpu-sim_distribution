
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

#include <iostream>
#include "aesCudaUtils.h"
#include "utilsBox.h"
#include <string.h>

using namespace std;

const int INPUTSIZE = 33*1024*1024;
bool MODE = 1;

//handler
extern "C"
int aesHost(unsigned char* result, const unsigned char* inData, int inputSize, const unsigned char* key, int keySize, bool toEncrypt);

int main(int argc, char *argv[])
{	
	try {
		unsigned numPairs = commandLineManager(argc, argv);

		unsigned aesType = static_cast<unsigned>(atoi(argv[2]));

		unsigned ekSize=240;
		if (aesType != 256)
			ekSize = 176;

		//per ogni coppia di file di input e di chiave svolge l'operazione
		for (unsigned cnt=0; cnt<numPairs; ++cnt) {

			char *h_Input = new char[INPUTSIZE];
			if ( !h_Input )
				throw string("cannot allocate memory for the input\n");	
			memset(h_Input, 0, INPUTSIZE);
	
			unsigned char myExpKey[ekSize];
			memset(myExpKey, 0, ekSize);
			
			//funzione di inizializzazione che legge i file di input ed espande le chiavi.
			unsigned usefulData = initAesCuda(argv[(cnt*2)+4], myExpKey, aesType, argv[(cnt*2)+3], h_Input, INPUTSIZE);
	
			// memory for the result
			unsigned char *h_Result = new unsigned char[INPUTSIZE];
			if ( !h_Result )
				throw string("cannot allocate memory for the result\n");	
	
			//la seguente operazione ha il seguente significato: poichè l'input viene scomposto in 256 blocchi di unsigned e cioè 1024 bytes, bisogna assicurarsi che la dimensione dell'input sia sempre multiplo di tale valore e cioè 1024 appunto.
			unsigned modUsefulData = ( usefulData - ( usefulData % 1024 ) + 1024 );

			//chiamata all'handler
			int errorReturned = aesHost(h_Result, reinterpret_cast<unsigned char*>(h_Input), modUsefulData, myExpKey, sizeof(myExpKey), MODE);
	
			if ( errorReturned == -1 )
				throw string("aesHost: cannot use an input size minor of 256\n");
	
			if ( errorReturned == -11 )
				throw string("aesHost: cannot use an input size not multiple of 256\n");
	
			if ( errorReturned == -2 )
				throw string("aesHost: cannot use an expanded key size different from 240 or 176\n");
	
			if ( errorReturned == -3 )
				throw string("aesHost: some input not allocated\n");
	
			cout << "\n###############################################################\n\n";

			char numFile[10000];
			sprintf(numFile, "%d", cnt);
			string nameOutFile("output_");
			nameOutFile.append(numFile);
			nameOutFile.append(".dat");

			//scrittura risultati in uscita
			writeToFile(nameOutFile, reinterpret_cast<char *>(h_Result), usefulData, INPUTSIZE);
	
			delete[] h_Result;
			delete[] h_Input;
		}

	} catch(string ex){
		cout << "Exception occurred: " << ex << endl;
	}

    return EXIT_SUCCESS;
}
