// $Id: separable_output_first.cpp 5188 2012-08-30 00:31:31Z dub $

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
//  SeparableOutputFirstAllocator: Separable Output-First Allocator
//
// ----------------------------------------------------------------------

#include "separable_output_first.hpp"

#include "booksim.hpp"
#include "arbiter.hpp"

#include <vector>
#include <iostream>
#include <cstring>

SeparableOutputFirstAllocator::
SeparableOutputFirstAllocator( Module* parent, const string& name, int inputs,
			       int outputs, const string& arb_type )
  : SeparableAllocator( parent, name, inputs, outputs, arb_type )
{}

void SeparableOutputFirstAllocator::Allocate() {
  
  set<int>::const_iterator port_iter = _out_occ.begin();
  while(port_iter != _out_occ.end()) {
    
    const int & output = *port_iter;

    // add requests to the output arbiter

    map<int, sRequest>::const_iterator req_iter = _out_req[output].begin();
    while(req_iter != _out_req[output].end()) {
      
      const sRequest & req = req_iter->second;

      _output_arb[output]->AddRequest(req.port, req.label, req.out_pri);

      ++req_iter;
    }
    
    // Execute the output arbiter and propagate the grants to the
    // input arbiters.

    int label = -1;
    const int input = _output_arb[output]->Arbitrate(&label, NULL);
    assert(input > -1);

    const sRequest & req = _in_req[input][output];
    assert((req.port == output) && (req.label == label));

    _input_arb[input]->AddRequest(req.port, req.label, req.in_pri);

    ++port_iter;
  }
  
  port_iter = _in_occ.begin();
  while(port_iter != _in_occ.end()) {
    
    const int & input = *port_iter;

    // Execute the input arbiters.
    
    const int output = _input_arb[input]->Arbitrate(NULL, NULL);
  
    if(output > -1) {
      assert((_inmatch[input] == -1) && (_outmatch[output] == -1));

      _inmatch[input] = output;
      _outmatch[output] = input;
      _input_arb[input]->UpdateState() ;
      _output_arb[output]->UpdateState() ;
    }
    
    ++port_iter;
  }
}
