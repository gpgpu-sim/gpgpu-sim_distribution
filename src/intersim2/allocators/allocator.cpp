// $Id: allocator.cpp 5188 2012-08-30 00:31:31Z dub $

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
#include <sstream>
#include <cassert>
#include "allocator.hpp"

/////////////////////////////////////////////////////////////////////////
//Allocator types
#include "maxsize.hpp"
#include "pim.hpp"
#include "islip.hpp"
#include "loa.hpp"
#include "wavefront.hpp"
#include "selalloc.hpp"
#include "separable_input_first.hpp"
#include "separable_output_first.hpp"
//
/////////////////////////////////////////////////////////////////////////

//==================================================
// Allocator base class
//==================================================

Allocator::Allocator( Module *parent, const string& name,
		      int inputs, int outputs ) :
Module( parent, name ), _inputs( inputs ), _outputs( outputs ), _dirty( false )
{
  _inmatch.resize(_inputs, -1);   
  _outmatch.resize(_outputs, -1);
}

void Allocator::Clear( )
{
  if(_dirty) {
    _inmatch.assign(_inputs, -1);
    _outmatch.assign(_outputs, -1);
    _dirty = false;
  }
}

void Allocator::AddRequest( int in, int out, int label, int in_pri,
			    int out_pri ) {

  assert( ( in >= 0 ) && ( in < _inputs ) );
  assert( ( out >= 0 ) && ( out < _outputs ) );
  assert( label >= 0 );
  _dirty = true;
}

int Allocator::OutputAssigned( int in ) const
{
  assert( ( in >= 0 ) && ( in < _inputs ) );

  return _inmatch[in];
}

int Allocator::InputAssigned( int out ) const
{
  assert( ( out >= 0 ) && ( out < _outputs ) );

  return _outmatch[out];
}

void Allocator::PrintGrants( ostream * os ) const
{
  if(!os) os = &cout;

  *os << "Input grants = [ ";
  for ( int input = 0; input < _inputs; ++input ) {
    if(_inmatch[input] >= 0) {
      *os << input << " -> " << _inmatch[input] << "  ";
    }
  }
  *os << "], output grants = [ ";
  for ( int output = 0; output < _outputs; ++output ) {
    if(_outmatch[output] >= 0) {
      *os << output << " -> " << _outmatch[output] << "  ";
    }
  }
  *os << "]." << endl;
}

//==================================================
// DenseAllocator
//==================================================

DenseAllocator::DenseAllocator( Module *parent, const string& name,
				int inputs, int outputs ) :
  Allocator( parent, name, inputs, outputs )
{
  _request.resize(_inputs);

  for ( int i = 0; i < _inputs; ++i ) {
    _request[i].resize(_outputs);  
    for ( int j = 0; j < _outputs; ++j ) {
      _request[i][j].label = -1;
    }
  }
}

void DenseAllocator::Clear( )
{
  for ( int i = 0; i < _inputs; ++i ) {
    for ( int j = 0; j < _outputs; ++j ) {
      _request[i][j].label = -1;
    }
  }
  Allocator::Clear();
}

int DenseAllocator::ReadRequest( int in, int out ) const
{
  assert( ( in >= 0 ) && ( in < _inputs ) );
  assert( ( out >= 0 ) && ( out < _outputs ) );

  return _request[in][out].label;
}

bool DenseAllocator::ReadRequest( sRequest &req, int in, int out ) const
{
  assert( ( in >= 0 ) && ( in < _inputs ) );
  assert( ( out >= 0 ) && ( out < _outputs ) );

  req = _request[in][out];

  return ( req.label >= 0 );
}

void DenseAllocator::AddRequest( int in, int out, int label, 
				 int in_pri, int out_pri )
{
  Allocator::AddRequest(in, out, label, in_pri, out_pri);
  assert( _request[in][out].label == -1 );

  _request[in][out].label   = label;
  _request[in][out].in_pri  = in_pri;
  _request[in][out].out_pri = out_pri;
}

void DenseAllocator::RemoveRequest( int in, int out, int label )
{
  assert( ( in >= 0 ) && ( in < _inputs ) );
  assert( ( out >= 0 ) && ( out < _outputs ) ); 
  
  _request[in][out].label = -1;
}

bool DenseAllocator::InputHasRequests( int in ) const
{
  for(int out = 0; out < _outputs; ++out) {
    if(_request[in][out].label >= 0) {
      return true;
    }
  }
  return false;
}

bool DenseAllocator::OutputHasRequests( int out ) const
{
  for(int in = 0; in < _inputs; ++in) {
    if(_request[in][out].label >= 0) {
      return true;
    }
  }
  return false;
}

int DenseAllocator::NumInputRequests( int in ) const
{
  int result = 0;
  for(int out = 0; out < _outputs; ++out) {
    if(_request[in][out].label >= 0) {
      ++result;
    }
  }
  return result;
}

int DenseAllocator::NumOutputRequests( int out ) const
{
  int result = 0;
  for(int in = 0; in < _inputs; ++in) {
    if(_request[in][out].label >= 0) {
      ++result;
    }
  }
  return result;
}

void DenseAllocator::PrintRequests( ostream * os ) const
{
  if(!os) os = &cout;

  *os << "Input requests = [ ";
  for ( int input = 0; input < _inputs; ++input ) {
    bool print = false;
    ostringstream ss;
    for ( int output = 0; output < _outputs; ++output ) {
      const sRequest & req = _request[input][output];
      if ( req.label >= 0 ) {
	print = true;
	ss << output << "@" << req.in_pri << " ";
      }
    }
    if(print) {
      *os << input << " -> [ " << ss.str() << "]  ";
    }
  }
  *os << "], output requests = [ ";
  for ( int output = 0; output < _outputs; ++output ) {
    bool print = false;
    ostringstream ss;
    for ( int input = 0; input < _inputs; ++input ) {
      const sRequest & req = _request[input][output];
      if ( req.label >= 0 ) {
	print = true;
	ss << input << "@" << req.out_pri << " ";
      }
    }
    if(print) {
      *os << output << " -> [ " << ss.str() << "]  ";
    }
  }
  *os << "]." << endl;
}

//==================================================
// SparseAllocator
//==================================================

SparseAllocator::SparseAllocator( Module *parent, const string& name,
				  int inputs, int outputs ) :
  Allocator( parent, name, inputs, outputs )
{
  _in_req.resize(_inputs);
  _out_req.resize(_outputs);
}


void SparseAllocator::Clear( )
{
  for ( int i = 0; i < _inputs; ++i ) {
    if(!_in_req[i].empty())
      _in_req[i].clear( );
  }

  for ( int j = 0; j < _outputs; ++j ) {
    if(!_out_req[j].empty())
      _out_req[j].clear( );
  }

  _in_occ.clear( );
  _out_occ.clear( );

  Allocator::Clear();
}

int SparseAllocator::ReadRequest( int in, int out ) const
{
  sRequest r;

  if ( ! ReadRequest( r, in, out ) ) {
    r.label = -1;
  } 

  return r.label;
}

bool SparseAllocator::ReadRequest( sRequest &req, int in, int out ) const
{
  bool found;

  assert( ( in >= 0 ) && ( in < _inputs ) );
  assert( ( out >= 0 ) && ( out < _outputs ) );

  map<int, sRequest>::const_iterator match = _in_req[in].find(out);
  if ( match != _in_req[in].end( ) ) {
    req = match->second;
    found = true;
  } else {
    found = false;
  }

  return found;
}

void SparseAllocator::AddRequest( int in, int out, int label, 
				  int in_pri, int out_pri )
{
  Allocator::AddRequest(in, out, label, in_pri, out_pri);
  assert( _in_req[in].count(out) == 0 );
  assert( _out_req[out].count(in) == 0 );

  // insert into occupied inputs set if
  // input is currently empty
  if ( _in_req[in].empty( ) ) {
    _in_occ.insert(in);
  }

  // similarly for the output
  if ( _out_req[out].empty( ) ) {
    _out_occ.insert(out);
  }

  sRequest req;
  req.port    = out;
  req.label   = label;
  req.in_pri  = in_pri;
  req.out_pri = out_pri;

  _in_req[in][out] = req;

  req.port  = in;

  _out_req[out][in] = req;
}

void SparseAllocator::RemoveRequest( int in, int out, int label )
{
  assert( ( in >= 0 ) && ( in < _inputs ) );
  assert( ( out >= 0 ) && ( out < _outputs ) ); 
  
  assert( _in_req[in].count( out ) > 0 );
  assert( _in_req[in][out].label == label );
  _in_req[in].erase( out );

  // remove from occupied inputs list if
  // input is now empty
  if ( _in_req[in].empty( ) ) {
    _in_occ.erase(in);
  }

  // similarly for the output
  assert( _out_req[out].count( in ) > 0 );
  assert( _out_req[out][in].label == label );
  _out_req[out].erase( in );

  if ( _out_req[out].empty( ) ) {
    _out_occ.erase(out);
  }
}

bool SparseAllocator::InputHasRequests( int in ) const
{
  return _in_occ.count(in) > 0;
}

bool SparseAllocator::OutputHasRequests( int out ) const
{
  return _out_occ.count(out) > 0;
}

int SparseAllocator::NumInputRequests( int in ) const
{
  return _in_occ.count(in);
}

int SparseAllocator::NumOutputRequests( int out ) const
{
  return _out_occ.count(out);
}

void SparseAllocator::PrintRequests( ostream * os ) const
{
  map<int, sRequest>::const_iterator iter;
  
  if(!os) os = &cout;
  
  *os << "Input requests = [ ";
  for ( int input = 0; input < _inputs; ++input ) {
    if(!_in_req[input].empty()) {
      *os << input << " -> [ ";
      for ( iter = _in_req[input].begin( ); 
	    iter != _in_req[input].end( ); iter++ ) {
	*os << iter->second.port << "@" << iter->second.in_pri << " ";
      }
      *os << "]  ";
    }
  }
  *os << "], output requests = [ ";
  for ( int output = 0; output < _outputs; ++output ) {
    if(!_out_req[output].empty()) {
      *os << output << " -> ";
      *os << "[ ";
      for ( iter = _out_req[output].begin( ); 
	    iter != _out_req[output].end( ); iter++ ) {
	*os << iter->second.port << "@" << iter->second.out_pri << " ";
      }
      *os << "]  ";
    }
  }
  *os << "]." << endl;
}

//==================================================
// Global allocator allocation function
//==================================================

Allocator *Allocator::NewAllocator( Module *parent, const string& name,
				    const string &alloc_type, 
				    int inputs, int outputs, 
				    Configuration const * const config )
{
  Allocator *a = 0;
  
  string alloc_name;
  string param_str;
  size_t left = alloc_type.find_first_of('(');
  if(left == string::npos) {
    alloc_name = alloc_type;
  } else {
    alloc_name = alloc_type.substr(0, left);
    size_t right = alloc_type.find_last_of(')');
    if(right == string::npos) {
      param_str = alloc_type.substr(left+1);
    } else {
      param_str = alloc_type.substr(left+1, right-left-1);
    }
  }
  if ( alloc_name == "max_size" ) {
    a = new MaxSizeMatch( parent, name, inputs, outputs );
  } else if ( alloc_name == "pim" ) {
    int iters = param_str.empty() ? (config ? config->GetInt("alloc_iters") : 1) : atoi(param_str.c_str());
    a = new PIM( parent, name, inputs, outputs, iters );
  } else if ( alloc_name == "islip" ) {
    int iters = param_str.empty() ? (config ? config->GetInt("alloc_iters") : 1) : atoi(param_str.c_str());
    a = new iSLIP_Sparse( parent, name, inputs, outputs, iters );
  } else if ( alloc_name == "loa" ) {
    a = new LOA( parent, name, inputs, outputs );
  } else if ( alloc_name == "wavefront" ) {
    a = new Wavefront( parent, name, inputs, outputs );
  } else if ( alloc_name == "rr_wavefront" ) {
    a = new Wavefront( parent, name, inputs, outputs, true );
  } else if ( alloc_name == "select" ) {
    int iters = param_str.empty() ? (config ? config->GetInt("alloc_iters") : 1) : atoi(param_str.c_str());
    a = new SelAlloc( parent, name, inputs, outputs, iters );
  } else if (alloc_name == "separable_input_first") {
    string arb_type = param_str.empty() ? (config ? config->GetStr("arb_type") : "round_robin") : param_str;
    a = new SeparableInputFirstAllocator( parent, name, inputs, outputs,
					  arb_type );
  } else if (alloc_name == "separable_output_first") {
    string arb_type = param_str.empty() ? (config ? config->GetStr("arb_type") : "round_robin") : param_str;
    a = new SeparableOutputFirstAllocator( parent, name, inputs, outputs,
					   arb_type );
  }

//==================================================
// Insert new allocators here, add another else if 
//==================================================


  return a;
}

