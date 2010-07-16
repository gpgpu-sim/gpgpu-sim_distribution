
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

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/fstream.hpp"
#include <iostream>
#include <string>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iterator>
#include <functional>
#include <fstream>

using namespace boost::filesystem;

//------------------------------------------help-functions-------------------------------------------------------
unsigned mySbox(unsigned num);

unsigned myXor(unsigned num1, unsigned num2);

//------------------------------------------help-functions-------------------------------------------------------


//------------------------------------------AESCudaUtils-----------------------------------------------------------
//------------------------------------------AESCudaUtils-----------------------------------------------------------
//------------------------------------------AESCudaUtils-----------------------------------------------------------
unsigned commandLineManager(int argc, char *argv[]);

void usage();

unsigned initAesCuda(std::string myKeyFile, unsigned char myKeyBuffer[], const unsigned int myKeyBitsSize, std::string myInputFile, char inputArray[], const unsigned inputArraySize);

boost::intmax_t getFileSize(path &myPath);

void readFromFileNotForm(path &myPath, char *storingArray, unsigned dataSize);

void readFromFileForm(path &myPath, std::vector<unsigned> &storingArray);

void expFunc(std::vector<unsigned> &keyArray, std::vector<unsigned> &expKeyArray);

void singleStep(std::vector<unsigned> &expKey, unsigned step);

void invExpFunc(std::vector<unsigned> &expKey, std::vector<unsigned> &invExpKeyArray);

void invMixColumn(std::vector<unsigned> &temp);

unsigned galoisProd(unsigned a, unsigned b);

void writeToFile(const std::string &outPath, char *storingArray, boost::intmax_t dataSize, unsigned maxInputSize);
