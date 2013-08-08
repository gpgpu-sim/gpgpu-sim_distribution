// $Id: tree_arb.cpp 5188 2012-08-30 00:31:31Z dub $

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
//  TreeArbiter
//
// ----------------------------------------------------------------------

#include "tree_arb.hpp"
#include <iostream>
#include <sstream>

using namespace std ;

TreeArbiter::TreeArbiter( Module *parent, const string &name,
			  int size, int groups, const string & arb_type ) 
  : Arbiter( parent, name, size ) {
  assert(size % groups == 0);
  _group_arbiters.resize(groups);
  _group_reqs.resize(groups, 0);
  _group_size = size / groups;
  for(int i = 0; i < groups; ++i) {
    ostringstream group_arb_name;
    group_arb_name << "group_arb" << i;
    _group_arbiters[i] = Arbiter::NewArbiter(this, group_arb_name.str(), arb_type, _group_size);
  }
  _global_arbiter = Arbiter::NewArbiter(this, "global_arb", arb_type, groups);
}

TreeArbiter::~TreeArbiter() {
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) {
    delete _group_arbiters[i];
  }
  delete _global_arbiter;
}

void TreeArbiter::PrintState() const  {
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) {
    cout << "Group arbiter " << i << ":" << endl;
    _group_arbiters[i]->PrintState();
  }
  cout << "Global arbiter:" << endl;
  _global_arbiter->PrintState();
}

void TreeArbiter::UpdateState() {
  if(_selected > -1) {
    int last_winner = _global_arbiter->LastWinner();
    assert(last_winner >= 0 && last_winner < (int)_group_arbiters.size());
    _group_arbiters[last_winner]->UpdateState();
    _global_arbiter->UpdateState();
  }
}

void TreeArbiter::AddRequest( int input, int id, int pri )
{
  Arbiter::AddRequest(input, id, pri);
  int group_index = input / _group_size;
  _group_arbiters[group_index]->AddRequest( input % _group_size, id, pri );
  ++_group_reqs[group_index];
}

int TreeArbiter::Arbitrate( int* id, int* pri ) {
  if(!_num_reqs) {
    return -1;
  } 
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) {
    if(_group_reqs[i]) {
      int group_id, group_pri;
      _group_arbiters[i]->Arbitrate(&group_id, &group_pri);
      _global_arbiter->AddRequest(i, group_id, group_pri);
    }
  }
  int group = _global_arbiter->Arbitrate(NULL, NULL);
  assert(group >= 0 && group < (int)_group_arbiters.size());
  int group_sel = _group_arbiters[group]->LastWinner();
  assert(group_sel >= 0 && group_sel < _group_size);
  _selected = group * _group_size + group_sel;
  assert(_selected >= 0 && _selected < _size);
  return Arbiter::Arbitrate(id, pri);
}

void TreeArbiter::Clear()
{
  if(!_num_reqs) {
    return;
  }
  for(int i = 0; i < (int)_group_arbiters.size(); ++i) {
    _group_arbiters[i]->Clear();
    _group_reqs[i] = 0;
  }
  _global_arbiter->Clear();
  Arbiter::Clear();
}
