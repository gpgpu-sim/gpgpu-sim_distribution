// $Id: chaos_router.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _CHAOS_ROUTER_HPP_
#define _CHAOS_ROUTER_HPP_

#include <string>
#include <queue>
#include <vector>

#include "module.hpp"
#include "router.hpp"
#include "allocator.hpp"
#include "routefunc.hpp"
#include "outputset.hpp"
#include "buffer_state.hpp"
#include "pipefifo.hpp"
#include "vc.hpp"

class ChaosRouter : public Router {

  tRoutingFunction   _rf;

  vector<OutputSet*> _input_route;
  vector<OutputSet*> _mq_route;

  enum eQState {
    empty,         //            input avail
    filling,       //    >**H    ready to send
    full,          //  T****H    ready to send
    leaving,       //    T***>   input avail
    cut_through,   //    >***>
    shared         // >**HT**>
  };

  PipelineFIFO<Flit>   *_crossbar_pipe;

  int _multi_queue_size;
  int _buffer_size;

  vector<queue<Flit *> > _input_frame;
  vector<queue<Flit *> > _output_frame;
  vector<queue<Flit *> > _multi_queue;

  vector<int> _next_queue_cnt;

  vector<queue<Credit *> > _credit_queue;

  vector<eQState> _input_state;
  vector<eQState> _multi_state;

  vector<int> _input_output_match;
  vector<int> _input_mq_match;
  vector<int> _multi_match;

  vector<int> _mq_age;

  vector<bool> _output_matched;
  vector<bool> _mq_matched;

  int _cur_channel;
  int _read_stall;

  bool _IsInjectionChan( int chan ) const;
  bool _IsEjectionChan( int chan ) const;

  bool _InputReady( int input ) const;
  bool _OutputFull( int out ) const;
  bool _OutputAvail( int out ) const;
  bool _MultiQueueFull( int mq ) const;

  int  _InputForOutput( int output ) const;
  int  _MultiQueueForOutput( int output ) const;
  int  _FindAvailMultiQueue( ) const;

  void _NextInterestingChannel( );
  void _OutputAdvance( );
  void _SendFlits( );
  void _SendCredits( );

  virtual void _InternalStep( );

public:
  ChaosRouter( const Configuration& config,
	    Module *parent, const string & name, int id,
	    int inputs, int outputs );

  virtual ~ChaosRouter( );

  virtual void ReadInputs( );
  virtual void WriteOutputs( );

  virtual int GetUsedCredit(int out) const {return 0;}
  virtual int GetBufferOccupancy(int i) const {return 0;}

#ifdef TRACK_BUFFERS
  virtual int GetUsedCreditForClass(int output, int cl) const {return 0;}
  virtual int GetBufferOccupancyForClass(int input, int cl) const {return 0;}
#endif

  virtual vector<int> UsedCredits() const { return vector<int>(); }
  virtual vector<int> FreeCredits() const { return vector<int>(); }
  virtual vector<int> MaxCredits() const { return vector<int>(); }

  void Display( ostream & os = cout ) const;
};

#endif
