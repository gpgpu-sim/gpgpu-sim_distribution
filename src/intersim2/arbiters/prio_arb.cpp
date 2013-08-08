// $Id: prio_arb.cpp 5188 2012-08-30 00:31:31Z dub $

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
#include <cassert>

#include "prio_arb.hpp"


PriorityArbiter::PriorityArbiter( const Configuration &config,
				  Module *parent, const string& name,
				  int inputs ) 
: Module( parent, name ), _rr_ptr(0), _inputs( inputs )
{

}

void PriorityArbiter::Clear( )
{
  _requests.clear( );
}

void PriorityArbiter::AddRequest( int in, int label, int pri )
{
  sRequest r;
  list<sRequest>::iterator insert_point;

  r.in = in; r.label = label; r.pri = pri;

  insert_point = _requests.begin( );
  while( ( insert_point != _requests.end( ) ) &&
	 ( insert_point->in < in ) ) {
    insert_point++;
  }

  bool del = false;
  bool add = true;

  // For consistant behavior, delete the existing request
  // if it is for the same input and has a higher
  // priority

  if ( ( insert_point != _requests.end( ) ) &&
       ( insert_point->in == in ) ) {
    if ( insert_point->pri < pri ) {
      del = true;
    } else {
      add = false;
    }
  }

  if ( add ) {
    _requests.insert( insert_point, r );
  }

  if ( del ) {
    _requests.erase( insert_point );
  }
}

void PriorityArbiter::RemoveRequest( int in, int label )
{
  list<sRequest>::iterator erase_point;

  erase_point = _requests.begin( );
  while( ( erase_point != _requests.end( ) ) &&
	 ( erase_point->in < in ) ) {
    erase_point++;
  }

  assert( erase_point != _requests.end( ) );
  _requests.erase( erase_point );
}

int PriorityArbiter::Match( ) const
{
  return _match;
}

void PriorityArbiter::Arbitrate( )
{
  list<sRequest>::iterator p;

  int max_index, max_pri;
  bool wrapped;

  //MERGENOTE
  //booksim does not have this if statement
  //as far as I can tell they are identical in function 
  if ( _requests.begin( ) != _requests.end( ) ) {
    // A round-robin arbiter between input requests
    p = _requests.begin( );
    while( ( p != _requests.end( ) ) &&
	   ( p->in < _rr_ptr ) ) {
      p++;
    }
    
    max_index = -1;
    max_pri   = 0;
    
    wrapped = false;
    while( (!wrapped) || ( p->in < _rr_ptr ) ) {
      if ( p == _requests.end( ) ) {
	if ( wrapped ) { break; }
	// p is valid here because empty lists
	// are skipped (above)
	p = _requests.begin( );
	wrapped = true;
      }
      
      // check if request is the highest priority so far
      if ( ( p->pri > max_pri ) || ( max_index == -1 ) ) {
	max_pri   = p->pri;
	max_index = p->in;
      }
      
      p++;
    }   

    _match = max_index; // -1 for no match
    if ( _match != -1 ) { 
      _rr_ptr = ( _match + 1 ) % _inputs;
    }
    
  } else {
    _match = -1;
  }
}

//MERGENOTE
//added update function to priorityarbiter

void PriorityArbiter::Update( )
{
  _rr_ptr = ( _rr_ptr + 1 ) % _inputs;
}
