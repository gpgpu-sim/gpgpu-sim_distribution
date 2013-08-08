// $Id: outputset.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*outputset.cpp
 *
 *output set assigns a flit which output to go to in a router
 *used by the VC class
 *the output assignment is done by the routing algorithms..
 *
 */

#include <cassert>

#include "booksim.hpp"
#include "outputset.hpp"

void OutputSet::Clear( )
{
  _outputs.clear( );
}

void OutputSet::Add( int output_port, int vc, int pri  )
{
  AddRange( output_port, vc, vc, pri );
}

void OutputSet::AddRange( int output_port, int vc_start, int vc_end, int pri )
{

  sSetElement s;

  s.vc_start = vc_start;
  s.vc_end   = vc_end;
  s.pri      = pri;
  s.output_port = output_port;
  _outputs.insert( s );
}

//legacy support, for performance, just use GetSet()
int OutputSet::NumVCs( int output_port ) const
{
  int total = 0;
  set<sSetElement>::const_iterator i = _outputs.begin( );
  while(i!=_outputs.end( )){
    if(i->output_port == output_port){
      total += (i->vc_end - i->vc_start + 1);
    }
    i++;
  }
  return total;
}

bool OutputSet::OutputEmpty( int output_port ) const
{
  set<sSetElement>::const_iterator i = _outputs.begin( );
  while(i!=_outputs.end( )){
    if(i->output_port == output_port){
      return false;
    }
    i++;
  }
  return true;
}


const set<OutputSet::sSetElement> & OutputSet::GetSet() const{
  return _outputs;
}

//legacy support, for performance, just use GetSet()
int OutputSet::GetVC( int output_port, int vc_index, int *pri ) const
{

  int range;
  int remaining = vc_index;
  int vc = -1;
  
  if ( pri ) { *pri = -1; }

  set<sSetElement>::const_iterator i = _outputs.begin( );
  while(i!=_outputs.end( )){
    if(i->output_port == output_port){
      range = i->vc_end - i->vc_start + 1;
      if ( remaining >= range ) {
	remaining -= range;
      } else {
	vc = i->vc_start + remaining;
	if ( pri ) {
	  *pri = i->pri;
	}
	break;
      }
    }
    i++;
  }
  return vc;
}

//legacy support, for performance, just use GetSet()
bool OutputSet::GetPortVC( int *out_port, int *out_vc ) const
{

  
  bool single_output = false;
  int  used_outputs  = 0;

  set<sSetElement>::const_iterator i = _outputs.begin( );
  if(i!=_outputs.end( )){
    used_outputs = i->output_port;
  }
  while(i!=_outputs.end( )){

    if ( i->vc_start == i->vc_end ) {
      *out_vc   = i->vc_start;
      *out_port = i->output_port;
      single_output = true;
    } else {
      // multiple vc's selected
      break;
    }
    if (used_outputs != i->output_port) {
      // multiple outputs selected
      single_output = false;
      break;
    }
       i++;
  }
  return single_output;
}
