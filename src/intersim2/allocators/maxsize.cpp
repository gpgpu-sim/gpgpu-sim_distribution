// $Id: maxsize.cpp 5188 2012-08-30 00:31:31Z dub $

/*
  Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
  All rights reserved.

  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this 
  list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.
  Neither the name of the Stanford University nor the names of its contributors 
  may be used to endorse or promote products derived from this software without 
  specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
  POSSIBILITY OF SUCH DAMAGE.
*/

#include "booksim.hpp"
#include <iostream>

#include "maxsize.hpp"

// shortest augmenting path:
//
// for all unmatched left nodes,
//    push node onto work stack
// end
//
// for all j,
//   from[j] = undefined
// end
//
// do,
//
//   while( !stack.empty ),
//     
//     nl = stack.pop
//     for each edge (nl,j),
//       if ( ( lmatch[nl] != j ) && ( from[j] == undefined ) ),
//         if ( rmatch[j] == undefined ),
//           stop // augmenting path found
//         else
//           from[j] = nl
//           newstack.push( rmatch[j] ) 
//         end
//       end
//     end
//   end
//
//   stack = newstack
// end
//

//#define DEBUG_MAXSIZE
//#define PRINT_MATCHING

MaxSizeMatch::MaxSizeMatch( Module *parent, const string& name,
			    int inputs, int outputs ) :
  DenseAllocator( parent, name, inputs, outputs )
{
  _from.resize(outputs);
  _s    = new int [inputs];
  _ns   = new int [inputs];
  _prio = 0;
}

MaxSizeMatch::~MaxSizeMatch( )
{
  delete [] _s;
  delete [] _ns;
}

void MaxSizeMatch::Allocate( )
{

  // augment as many times as possible 
  // (this is an O(N^3) maximum-size matching algorithm)
  while( _ShortestAugmenting( ) );

  // next time, start at next input to ensure fairness
  _prio = (_prio + 1) % _inputs;
}


bool MaxSizeMatch::_ShortestAugmenting( )
{
  int i, j, jn;
  int slen, nslen;

  // start with empty stack
  slen = 0;

  // push all unassigned inputs to the stack
  for ( i = 0; i < _inputs; ++i ) {
    j = (i + _prio) % _inputs;
    if ( _inmatch[j] == -1 ) { // start with unmatched left nodes
      _s[slen++] = j;
    }
  }

  _from.assign(_inputs, -1);

  for ( int iter = 0; iter < _inputs; iter++ ) {
    nslen = 0;

    for ( int e = 0; e < slen; ++e ) {
      i = _s[e];
      
      for ( j = 0; j < _outputs; ++j ) {
	if ( ( _request[i][j].label != -1 ) && // edge (i,j) exists
	     ( _inmatch[i] != j ) &&     // (i,j) is not contained in the current matching
	     ( _from[j] == -1 ) ) {      // no shorter path to j exists
	  
	  _from[j] = i;                  // how did we get to j?

#ifdef DEBUG_MAXSIZE
	  cout << "  got to " << j << " from " << i << endl;
#endif
	  if ( _outmatch[j] == -1 ) {   // j is unmatched -- augmenting path found
	    goto found_augmenting;
	  } else {                      // j is matched
	    _ns[nslen] = _outmatch[j];  // add the destination of this edge to the leaf nodes
	    nslen++;
	    
#ifdef DEBUG_MAXSIZE
	    cout << "  adding " << _outmatch[j] << endl;
#endif
	  }
	}
      }
    }

    // no augmenting path found yet, swap stacks
    int * t = _s;
    _s = _ns;
    _ns = t;
    slen = nslen;
  }
  
  return false; // no augmenting paths

 found_augmenting:
  
  // the augmenting path ends at node j on the right
  
#ifdef DEBUG_MAXSIZE
  cout << "Found path: " << j << "c <- ";
#endif

  i = _from[j];
  _outmatch[j] = i;

#ifdef DEBUG_MAXSIZE
  cout << i;
#endif

  while ( _inmatch[i] != -1 ) {  // loop until the end of the path
    jn = _inmatch[i];            // remove previous edge (i,jn) and add (i,j)
    _inmatch[i] = j;

#ifdef DEBUG_MAXSIZE
    cout << " <- " << j << "c <- ";
#endif

    j = jn;                    // add edge from (jn,in)
    i = _from[j];
    _outmatch[j] = i; 

#ifdef DEBUG_MAXSIZE
    cout << i;
#endif
  }

#ifdef DEBUG_MAXSIZE
  cout << endl;
#endif
  
  _inmatch[i] = j;

#ifdef PRINT_MATCHING
  cout << "left  matching: ";

  for ( i = 0; i < _inputs; i++ ) {
    cout << _inmatch[i] << " ";
  }
  cout << endl;

  cout << "right matching: ";
  for ( i = 0; i < _outputs; i++ ) {
    cout << _outmatch[i] << " ";
  }
  cout << endl;
#endif

  return true;
}
