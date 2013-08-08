// $Id: separable.cpp 5188 2012-08-30 00:31:31Z dub $

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

// ----------------------------------------------------------------------
//
//  SeparableAllocator: Separable Allocator Base Class
//
// ----------------------------------------------------------------------

#include "separable.hpp"

#include <sstream>

#include "arbiter.hpp"

SeparableAllocator::SeparableAllocator( Module* parent, const string& name,
					int inputs, int outputs,
					const string& arb_type )
  : SparseAllocator( parent, name, inputs, outputs )
{
  
  _input_arb.resize(inputs);

  for (int i = 0; i < inputs; ++i) {
    ostringstream arb_name("arb_i");
    arb_name << i;
    _input_arb[i] = Arbiter::NewArbiter(this, arb_name.str(), arb_type, outputs);
  }

  _output_arb.resize(outputs);

  for (int i = 0; i < outputs; ++i) {
    ostringstream arb_name("arb_o");
    arb_name << i;
    _output_arb[i] = Arbiter::NewArbiter(this, arb_name.str( ), arb_type, inputs);
  }

}

SeparableAllocator::~SeparableAllocator() {

  for (int i = 0; i < _inputs; ++i) {
    delete _input_arb[i];
  }

  for (int i = 0; i < _outputs; ++i) {
    delete _output_arb[i];
  }

}

void SeparableAllocator::Clear() {
  for ( int i = 0 ; i < _inputs ; i++ ) {
    if(_input_arb[i]-> _num_reqs)
      _input_arb[i]->Clear();
  }
  for ( int o = 0; o < _outputs; o++ ) {
    if(_output_arb[o]->_num_reqs)
      _output_arb[o]->Clear();
  }
  SparseAllocator::Clear();
}
