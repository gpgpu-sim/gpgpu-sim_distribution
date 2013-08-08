// $Id: islip.cpp 5188 2012-08-30 00:31:31Z dub $

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

#include "islip.hpp"
#include "random_utils.hpp"

//#define DEBUG_ISLIP

iSLIP_Sparse::iSLIP_Sparse( Module *parent, const string& name,
			    int inputs, int outputs, int iters ) :
  SparseAllocator( parent, name, inputs, outputs ),
  _iSLIP_iter(iters)
{
  _gptrs.resize(_outputs, 0);
  _aptrs.resize(_inputs, 0);
}

void iSLIP_Sparse::Allocate( )
{
  int input;
  int output;

  int input_offset;
  int output_offset;

  map<int, sRequest>::iterator p;
  bool wrapped;

  for ( int iter = 0; iter < _iSLIP_iter; ++iter ) {
    // Grant phase

    vector<int> grants(_outputs, -1);

    for ( output = 0; output < _outputs; ++output ) {

      // Skip loop if there are no requests
      // or the output is already matched
      if ( ( _out_req[output].empty( ) ) ||
	   ( _outmatch[output] != -1 ) ) {
	continue;
      }

      // A round-robin arbiter between input requests
      input_offset = _gptrs[output];

      p = _out_req[output].begin( );
      while( ( p != _out_req[output].end( ) ) &&
	     ( p->second.port < input_offset ) ) {
	p++;
      }

      wrapped = false;
      while( (!wrapped) || 
	     ( ( p != _out_req[output].end( ) ) &&
	       ( p->second.port < input_offset ) ) ) {
	if ( p == _out_req[output].end( ) ) {
	  if ( wrapped ) { break; }
	  // p is valid here because empty lists
	  // are skipped (above)
	  p = _out_req[output].begin( );
	  wrapped = true;
	}

	input = p->second.port;

	// we know the output is free (above) and
	// if the input is free, grant request
	if ( _inmatch[input] == -1 ) {
	  grants[output] = input;
	  break;
	}

	p++;
      }      
    }

#ifdef DEBUG_ISLIP
    cout << "grants: ";
    for ( int i = 0; i < _outputs; ++i ) {
      cout << grants[i] << " ";
    }
    cout << endl;

    cout << "aptrs: ";
    for ( int i = 0; i < _inputs; ++i ) {
      cout << _aptrs[i] << " ";
    }
    cout << endl;
#endif

    // Accept phase

    for ( input = 0; input < _inputs; ++input ) {

      if ( _in_req[input].empty( ) ) {
	continue;
      }

      // A round-robin arbiter between output grants
      output_offset = _aptrs[input];

      p = _in_req[input].begin( );
      while( ( p != _in_req[input].end( ) ) &&
	     ( p->second.port < output_offset ) ) {
	p++;
      }

      wrapped = false;
      while( (!wrapped) || 
	     ( ( p != _in_req[input].end( ) ) &&
	       ( p->second.port < output_offset ) ) ) {
	if ( p == _in_req[input].end( ) ) {
	  if ( wrapped ) { break; }
	  // p is valid here because empty lists
	  // are skipped (above)
	  p = _in_req[input].begin( );
	  wrapped = true;
	}

	output = p->second.port;

	// we know the output is free (above) and
	// if the input is free, grant request
	if ( grants[output] == input ) {
	  // Accept
	  _inmatch[input]   = output;
	  _outmatch[output] = input;

	  // Only update pointers if accepted during the 1st iteration
	  if ( iter == 0 ) {
	    _gptrs[output] = ( input + 1 ) % _inputs;
	    _aptrs[input]  = ( output + 1 ) % _outputs;
	  }

	  break;
	}

	p++;
      } 
    }
  }

#ifdef DEBUG_ISLIP
  cout << "input match: ";
  for ( int i = 0; i < _inputs; ++i ) {
    cout << _inmatch[i] << " ";
  }
  cout << endl;

  cout << "output match: ";
  for ( int j = 0; j < _outputs; ++j ) {
    cout << _outmatch[j] << " ";
  }
  cout << endl;
#endif
}
