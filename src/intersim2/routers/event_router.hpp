// $Id: event_router.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _EVENT_ROUTER_HPP_
#define _EVENT_ROUTER_HPP_

#include <string>
#include <queue>
#include <vector>

#include "module.hpp"
#include "router.hpp"
#include "buffer.hpp"
#include "vc.hpp"
#include "prio_arb.hpp"
#include "routefunc.hpp"
#include "outputset.hpp"
#include "pipefifo.hpp"

class EventNextVCState : public Module {
public:
  enum eNextVCState { idle, busy, tail_pending };

  struct tWaiting {
    int  input;
    int  vc;
    int  id;
    int  pres;
    bool watch;
  };

private:
  int _buf_size;
  int _vcs;

  vector<int> _credits;
  vector<int> _presence;
  vector<int> _input;
  vector<int> _inputVC;

  vector<list<tWaiting *> > _waiting;
 
  vector<eNextVCState> _state;

public:

  EventNextVCState( const Configuration& config, 
		    Module *parent, const string& name );

  eNextVCState GetState( int vc ) const;
  int GetPresence( int vc ) const;
  int GetCredits( int vc ) const;
  int GetInput( int vc ) const;
  int GetInputVC( int vc ) const;

  bool IsWaiting( int vc ) const;
  bool IsInputWaiting( int vc, int w_input, int w_vc ) const;

  void PushWaiting( int vc, tWaiting *w );
  void IncrWaiting( int vc, int w_input, int w_vc );
  tWaiting *PopWaiting( int vc );

  void SetState( int vc, eNextVCState state );
  void SetCredits( int vc, int value );
  void SetPresence( int vc, int value );
  void SetInput( int vc, int input );
  void SetInputVC( int vc, int in_vc );
};

class EventRouter : public Router {
  int _vcs;

  int _vct;

  vector<Buffer *> _buf;
  vector<vector<bool> > _active;

  tRoutingFunction   _rf;

  vector<EventNextVCState *> _output_state;

  PipelineFIFO<Flit>   *_crossbar_pipe;
  PipelineFIFO<Credit> *_credit_pipe;

  vector<queue<Flit *> > _input_buffer;
  vector<queue<Flit *> > _output_buffer;

  vector<queue<Credit *> > _in_cred_buffer;
  vector<queue<Credit *> > _out_cred_buffer;

  struct tArrivalEvent {
    int  input;
    int  output;
    int  src_vc;
    int  dst_vc;
    bool head;
    bool tail;
    
    int  id;    // debug
    bool watch; // debug
  };

  PipelineFIFO<tArrivalEvent> *_arrival_pipe;
  vector<queue<tArrivalEvent *> > _arrival_queue;
  vector<PriorityArbiter*> _arrival_arbiter;

  struct tTransportEvent {
    int  input;
    int  src_vc;
    int  dst_vc;

    int  id;    // debug
    bool watch; // debug
  };

  vector<queue<tTransportEvent *> > _transport_queue;
  vector<PriorityArbiter*> _transport_arbiter;

  vector<bool> _transport_free;
  vector<int> _transport_match;

  void _ReceiveFlits( );
  void _ReceiveCredits( );

  void _IncomingFlits( );
  void _ArrivalRequests( int input );
  void _ArrivalArb( int output );
  void _SendTransport( int input, int output, tArrivalEvent *aevt );
  void _ProcessWaiting( int output, int out_vc );
  void _TransportRequests( int output );
  void _TransportArb( int input );
  void _OutputQueuing( );

  void _SendFlits( );
  void _SendCredits( );

  virtual void _InternalStep( );

public:
  EventRouter( const Configuration& config,
	       Module *parent, const string & name, int id,
	       int inputs, int outputs );
  virtual ~EventRouter( );

  virtual void ReadInputs( );
  virtual void WriteOutputs( );

  virtual int GetUsedCredit(int o) const {return 0;}
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
