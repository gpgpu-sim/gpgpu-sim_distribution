// $Id: channel.hpp 5188 2012-08-30 00:31:31Z dub $

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

//////////////////////////////////////////////////////////////////////
//
//  File Name: channel.hpp
//
//  The Channel models a generic channel with a multi-cycle 
//   transmission delay. The channel latency can be specified as 
//   an integer number of simulator cycles.
//
/////
#ifndef _CHANNEL_HPP
#define _CHANNEL_HPP

#include <queue>
#include <cassert>

#include "globals.hpp"
#include "module.hpp"
#include "timed_module.hpp"

using namespace std;

template<typename T>
class Channel : public TimedModule {
public:
  Channel(Module * parent, string const & name);
  virtual ~Channel() {}

  // Physical Parameters
  void SetLatency(int cycles);
  int GetLatency() const { return _delay ; }
  
  // Send data 
  virtual void Send(T * data);
  
  // Receive data
  virtual T * Receive(); 
  
  virtual void ReadInputs();
  virtual void Evaluate() {}
  virtual void WriteOutputs();

protected:
  int _delay;
  T * _input;
  T * _output;
  queue<pair<int, T *> > _wait_queue;

};

template<typename T>
Channel<T>::Channel(Module * parent, string const & name)
  : TimedModule(parent, name), _delay(1), _input(0), _output(0) {
}

template<typename T>
void Channel<T>::SetLatency(int cycles) {
  if(cycles <= 0) {
    Error("Channel must have positive delay.");
  }
  _delay = cycles ;
}

template<typename T>
void Channel<T>::Send(T * data) {
  _input = data;
}

template<typename T>
T * Channel<T>::Receive() {
  return _output;
}

template<typename T>
void Channel<T>::ReadInputs() {
  if(_input) {
    _wait_queue.push(make_pair(GetSimTime() + _delay - 1, _input));
    _input = 0;
  }
}

template<typename T>
void Channel<T>::WriteOutputs() {
  _output = 0;
  if(_wait_queue.empty()) {
    return;
  }
  pair<int, T *> const & item = _wait_queue.front();
  int const & time = item.first;
  if(GetSimTime() < time) {
    return;
  }
  assert(GetSimTime() == time);
  _output = item.second;
  assert(_output);
  _wait_queue.pop();
}

#endif
