// $Id: pim.cpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "booksim.hpp"
#include <iostream>

#include "pim.hpp"
#include "random_utils.hpp"

//#define DEBUG_PIM

PIM::PIM( Module *parent, const string& name,
	  int inputs, int outputs, int iters ) :
  DenseAllocator( parent, name, inputs, outputs ),
  _PIM_iter(iters)
{
}

PIM::~PIM( )
{
}

void PIM::Allocate( )
{
  int input;
  int output;

  int input_offset;
  int output_offset;

  for ( int iter = 0; iter < _PIM_iter; ++iter ) {
    // Grant phase --- outputs randomly choose
    // between one of their requests

    vector<int> grants(_outputs, -1);

    for ( output = 0; output < _outputs; ++output ) {
      
      // A random arbiter between input requests
      input_offset  = RandomInt( _inputs - 1 );
      
      for ( int i = 0; i < _inputs; ++i ) {
	input = ( i + input_offset ) % _inputs;  
	
	if ( ( _request[input][output].label != -1 ) && 
	     ( _inmatch[input] == -1 ) &&
	     ( _outmatch[output] == -1 ) ) {
	  
	  // Grant
	  grants[output] = input;
	  break;
	}
      }
    }
  
    // Accept phase -- inputs randomly choose
    // between input_speedup of their grants
    
    for ( input = 0; input < _inputs; ++input ) {
      
      // A random arbiter between output grants
      output_offset  = RandomInt( _outputs - 1 );
      
      for ( int o = 0; o < _outputs; ++o ) {
	output = ( o + output_offset ) % _outputs;
	
	if ( grants[output] == input ) {
	  
	  // Accept
	  _inmatch[input]   = output;
	  _outmatch[output] = input;
	  
	  break;
	}
      }
    }
  }

#ifdef DEBUG_PIM
  if ( _outputs == 8 ) {
    cout << "input match: " << endl;
    for ( int i = 0; i < _inputs; ++i ) {
      cout << "  from " << i << " to " << _inmatch[i] << endl;
    }
    cout << endl;
  }

  cout << "output match: ";
  for ( int j = 0; j < _outputs; ++j ) {
    cout << _outmatch[j] << " ";
  }
  cout << endl;
#endif
}


