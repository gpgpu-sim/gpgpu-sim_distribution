// $Id: event_router.cpp 5188 2012-08-30 00:31:31Z dub $

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

#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "event_router.hpp"
#include "stats.hpp"
#include "globals.hpp"

EventRouter::EventRouter( const Configuration& config,
		    Module *parent, const string & name, int id,
		    int inputs, int outputs )
  : Router( config,
	    parent, name,
	    id,
	    inputs, outputs )
{
  ostringstream module_name;
  
  _vcs            = config.GetInt( "num_vcs" );

  // Cut-through mode --- packets are not broken
  // up and input buffers are assumed to be 
  // expressed in units of maximum size packets.
  
  _vct            = config.GetInt( "vct" );

  // Routing

  string rf = config.GetStr("routing_function") + "_" + config.GetStr("topology");
  map<string, tRoutingFunction>::iterator rf_iter = gRoutingFunctionMap.find(rf);
  if(rf_iter == gRoutingFunctionMap.end()) {
    Error("Invalid routing function: " + rf);
  }
  _rf = rf_iter->second;

  // Alloc VC's

  _buf.resize(_inputs);
  _active.resize(_inputs);

  for ( int i = 0; i < _inputs; ++i ) {
    module_name << "buf_" << i;
    _buf[i] = new Buffer( config, _outputs, this, module_name.str( ) );
    module_name.seekp( 0, ios::beg );
    _active[i].resize(_vcs, false);
  }

  // Alloc next VCs' state

  _output_state.resize(_outputs);

  for ( int o = 0; o < _outputs; ++o ) {
    module_name << "output" << o << "_vc_state";
    _output_state[o] = new EventNextVCState(config, this, module_name.str());
    module_name.seekp( 0, ios::beg );
  }

  // Alloc arbiters

  _arrival_arbiter.resize(_outputs);

  for ( int o = 0; o < _outputs; ++o ) {
    module_name << "arrival_arb_output" << o;
    _arrival_arbiter[o] = 
      new PriorityArbiter( config, this, module_name.str( ), _inputs );
    module_name.seekp( 0, ios::beg );
  }

  _transport_arbiter.resize(_inputs);

  for ( int i = 0; i < _inputs; ++i ) {
    module_name << "transport_arb_input" << i;
    _transport_arbiter[i] = 
      new PriorityArbiter( config, this, module_name.str( ), _outputs );
    module_name.seekp( 0, ios::beg );
  }

  // Alloc pipelines (to simulate processing/transmission delays)

  _crossbar_pipe = 
    new PipelineFIFO<Flit>( this, "crossbar_pipeline", _outputs, 
			    _crossbar_delay );

  _credit_pipe =
    new PipelineFIFO<Credit>( this, "credit_pipeline", _inputs,
			      _credit_delay );

  _arrival_pipe =
    new PipelineFIFO<tArrivalEvent>( this, "arrival_pipeline", _inputs,
				     0 /* FIX THIS EVENTUALLY */);

  // Queues

  _input_buffer.resize(_inputs); 
  _output_buffer.resize(_outputs); 

  _in_cred_buffer.resize(_inputs); 
  _out_cred_buffer.resize(_outputs);

  _arrival_queue.resize(_inputs);
  _transport_queue.resize(_outputs);

  // Misc.

  _transport_free.resize(_inputs, true);
  _transport_match.resize(_inputs, -1);
}

EventRouter::~EventRouter( )
{
  for ( int i = 0; i < _inputs; ++i ) {
    delete _buf[i];
  }

  for ( int o = 0; o < _outputs; ++o ) {
    delete _output_state[o];
  }

  for ( int o = 0; o < _outputs; ++o ) {
    delete _arrival_arbiter[o];
  }

  for ( int i = 0; i < _inputs; ++i ) {
    delete _transport_arbiter[i];
  }

  delete _crossbar_pipe;
  delete _credit_pipe;
  delete _arrival_pipe;
}
  
void EventRouter::ReadInputs( )
{
  _ReceiveFlits( );
  _ReceiveCredits( );
}

void EventRouter::_InternalStep( )
{
  // Receive incoming flits
  _IncomingFlits( );

  // The input pipe simulates routing delay
  _arrival_pipe->Advance( );

  // Clear output requests
  for ( int output = 0; output < _outputs; ++output ) {
    _arrival_arbiter[output]->Clear( );
  }

  // Check input arrival queues and generate
  // requests for the outputs
  for ( int input = 0; input < _inputs; ++input ) {
    _ArrivalRequests( input );
  }

  // Arbitrate between requests at outputs
  for ( int output = 0; output < _outputs; ++output ) {
    _ArrivalArb( output );
  }

  for ( int input = 0; input < _inputs; ++input ) {
    _transport_arbiter[input]->Clear( );
  }

  _crossbar_pipe->WriteAll( 0 );
  _credit_pipe->WriteAll( 0 );

  // Generate transport events and their
  // requests for the inputs
  for ( int output = 0; output < _outputs; ++output ) {
    _TransportRequests( output );
  }

  // Arbitrate between requests at inputs
  for ( int input = 0; input < _inputs; ++input ) {
    _TransportArb( input );
  }

  _crossbar_pipe->Advance( );
  _credit_pipe->Advance( );

  _OutputQueuing( );
}

void EventRouter::WriteOutputs( )
{
  _SendFlits( );
  _SendCredits( );
}

void EventRouter::_ReceiveFlits( )
{
  Flit *f;

  for ( int input = 0; input < _inputs; ++input ) { 
    f = _input_channels[input]->Receive();

    if ( f ) {
      _input_buffer[input].push( f );
    }
  }
}

void EventRouter::_ReceiveCredits( )
{
  Credit *c;

  for ( int output = 0; output < _outputs; ++output ) {  
    c = _output_credits[output]->Receive();

    if ( c ) {
      _out_cred_buffer[output].push( c );
    }
  }
}

void EventRouter::_ProcessWaiting( int output, int out_vc )
{
  // out_vc just sent the transport event for out_vc, 
  // check if any events are queued on that vc.  if so,
  // generate another transport event and set the 
  // owner of the vc, otherwise set the vc to idle.

  int credits;
  
  tTransportEvent *tevt;

  EventNextVCState::tWaiting *w;

  if ( _output_state[output]->IsWaiting( out_vc ) ) {
	    
    // State remains as busy, but the waiting VC takes over
    w = _output_state[output]->PopWaiting( out_vc );
    
    _output_state[output]->SetState( out_vc, EventNextVCState::busy );
    _output_state[output]->SetInput( out_vc, w->input );
    _output_state[output]->SetInputVC( out_vc, w->vc );

    if ( w->watch ) {
      cout << "Dequeuing waiting arrival event at " << FullName() 
	   << " for flit " << w->id << endl;
    }
    
    credits = _output_state[output]->GetCredits( out_vc );

    // Try to queue a transmit event for a waiting packet
    if ( credits > 0 ) {
      tevt         = new tTransportEvent;
      tevt->src_vc = w->vc;
      tevt->dst_vc = out_vc;
      tevt->input  = w->input;
      tevt->watch  = w->watch; // just to have something here
      tevt->id     = w->id;
      
      _transport_queue[output].push( tevt );
      
      if ( tevt->watch ) {
	cout << "Injecting transport event at " << FullName() 
	     << " for flit " << tevt->id << endl;
      }
      
      credits--;
      _output_state[output]->SetCredits( out_vc, credits );
      _output_state[output]->SetPresence( out_vc, w->pres - 1 );
      
    } else {
      // No credits available, just store presence
      _output_state[output]->SetPresence( out_vc, w->pres );
    }

    delete w;

  } else {
    // Tail sent, none waiting => VC is idle
    _output_state[output]->SetState( out_vc, EventNextVCState::idle );
  }
}

void EventRouter::_IncomingFlits( )
{
  Flit   *f;
  Buffer *cur_buf;

  tArrivalEvent *aevt;

  _arrival_pipe->WriteAll( 0 );

  for ( int input = 0; input < _inputs; ++input ) {
    if ( !_input_buffer[input].empty( ) ) {
      f = _input_buffer[input].front( );
      _input_buffer[input].pop( );

      cur_buf = _buf[input];
      int vc = f->vc;

      cur_buf->AddFlit( vc, f );

      // Head flit arriving at idle VC
      if ( !_active[input][vc] ) {
	
	if ( !f->head ) {
	  cout << "Non-head flit:" << endl;
	  cout << *f;
	  Error( "Received non-head flit at idle VC" );
	}

	const OutputSet *route_set;
	int out_vc, out_port;

	cur_buf->Route( vc, _rf, this, f, input );
	route_set = cur_buf->GetRouteSet( vc );

	if ( !route_set->GetPortVC( &out_port, &out_vc ) ) {
	  Error( "The event-driven router requires routing functions with a single (port,vc) output" );
	}

	cur_buf->SetOutput( vc, out_port, out_vc );
	_active[input][vc] = true;
      } else {
	if ( f->head ) {
	  cout << *f;
	  Error( "Received head flit at non-idle VC." );
	}
      }
      
      if ( f->watch ) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Received flit at " << FullName() << ".  Output port = " 
		    << cur_buf->GetOutputPort( vc ) << ", output VC = " 
		    << cur_buf->GetOutputVC( vc ) << endl
		    << *f;
      }

      // In cut-through mode, only head flits generate arrivals,
      // otherwise all flits generate

      if ( ( !_vct ) || ( _vct && f->head ) ) {
	// Add the arrival event to a delay pipeline to
	// account for routing/decoding time

	aevt         = new tArrivalEvent;
	
	aevt->input  = input;
	aevt->output = cur_buf->GetOutputPort( vc );
	aevt->src_vc = f->vc;
	aevt->dst_vc = cur_buf->GetOutputVC( vc );
	aevt->head   = f->head;
	aevt->tail   = f->tail;
	
	//if ( f->head && f->tail ) {
	//	Error( "Head/tail packets not supported." );
	//}
	
	aevt->watch  = f->watch;
	aevt->id     = f->id;
	
	_arrival_pipe->Write( aevt, input );

	if ( aevt->watch ) {
	  cout << "Injected arrival event at " << FullName() 
	       << " for flit " << aevt->id << endl;
	}
      }
    }
  }
}

void EventRouter::_ArrivalRequests( int input ) 
{
  tArrivalEvent *aevt;
  
  aevt = _arrival_pipe->Read( input );
  if ( aevt ) {
    _arrival_queue[input].push( aevt );
  }
  
  if ( !_arrival_queue[input].empty( ) ) {
    aevt = _arrival_queue[input].front( );
    _arrival_arbiter[aevt->output]->AddRequest( input );
  }
}

void EventRouter::_SendTransport( int input, int output, tArrivalEvent *aevt )
{
  // Try to send a transport event

  tTransportEvent *tevt;

  int credits;
  int pres;

  credits = _output_state[output]->GetCredits( aevt->dst_vc );
  
  if ( credits > 0 ) {
    // Take a credit and queue a transport event
    credits--;
    _output_state[output]->SetCredits( aevt->dst_vc, credits );
    
    tevt         = new tTransportEvent;
    tevt->src_vc = aevt->src_vc;
    tevt->dst_vc = aevt->dst_vc;
    tevt->input  = input;
    tevt->watch  = aevt->watch;
    tevt->id     = aevt->id;
	
    _transport_queue[output].push( tevt );
	
    if ( tevt->watch ) {
      cout << "Injecting transport event at " << FullName() 
	   << " for flit " << tevt->id << endl;
    }
  } else {
    if ( aevt->watch ) {
      cout << "No credits available at " << FullName() 
	   << " for flit " << aevt->id << " storing presence." << endl;
    }
    
    // No credits available, just store presence
    pres = _output_state[output]->GetPresence( aevt->dst_vc );
    _output_state[output]->SetPresence( aevt->dst_vc, pres + 1 );
  }
}

void EventRouter::_ArrivalArb( int output )
{
  tArrivalEvent   *aevt;
  tTransportEvent *tevt;
  Credit          *c;

  EventNextVCState::tWaiting *w;

  int input;
  int credits;
  int pres;

  // Incoming credits can produce or enable
  // transport events --- process them first

  if ( !_out_cred_buffer[output].empty( ) ) {
    c = _out_cred_buffer[output].front( );
    _out_cred_buffer[output].pop( );
    
    assert( c->vc.size() == 1 );
    int vc = *c->vc.begin();

    EventNextVCState::eNextVCState state = 
      _output_state[output]->GetState( vc );
    
    credits = _output_state[output]->GetCredits( vc );
    pres    = _output_state[output]->GetPresence( vc );
      
    if ( _vct ) {
      // In cut-through mode, only head credits indicate a change in 
      // channel state.

      if ( c->head ) {
	credits++;
	_output_state[output]->SetCredits( vc, credits );
	_ProcessWaiting( output, vc );
      }
    } else {
      credits++;
      _output_state[output]->SetCredits( vc, credits );

      if ( c->tail ) { // tail flit -- recycle VC
	if ( state != EventNextVCState::busy ) {
	  Error( "Received tail credit at non-busy output VC" );
	}
	
	_ProcessWaiting( output, vc );
      } else if ( ( state == EventNextVCState::busy ) && ( pres > 0 ) ) {
	// Flit is present => generate transport event
	
	tevt         = new tTransportEvent;
	tevt->input  = _output_state[output]->GetInput( vc );
	tevt->src_vc = _output_state[output]->GetInputVC( vc );
	tevt->dst_vc = vc;
	tevt->watch  = false;
	tevt->id     = -1;
	
	_transport_queue[output].push( tevt );
	
	pres--;
	credits--;
	_output_state[output]->SetPresence( vc, pres );
	_output_state[output]->SetCredits( vc, credits );
      }
    }

    c->Free();
  }

  // Now process arrival events

  _arrival_arbiter[output]->Arbitrate( );
  input = _arrival_arbiter[output]->Match( );

  if ( input != -1 ) {  
    // Winning arrival event gets access to output

    aevt = _arrival_queue[input].front( );
    _arrival_queue[input].pop( );

    if ( aevt->watch ) {
      cout << "Processing arrival event at " << FullName() 
	     << " for flit " << aevt->id << endl;
    }
      
    EventNextVCState::eNextVCState state = 
      _output_state[output]->GetState( aevt->dst_vc );

    if ( aevt->head ) { // Head flits
      if ( state == EventNextVCState::idle ) {
	// Allocate the output VC and queue a transport event
	_output_state[output]->SetState( aevt->dst_vc, EventNextVCState::busy );
	_output_state[output]->SetInput( aevt->dst_vc, input );
	_output_state[output]->SetInputVC( aevt->dst_vc, aevt->src_vc );

	_SendTransport( input, output, aevt );
      } else {
	// VC busy => queue a waiting event

	w = new EventNextVCState::tWaiting;

	w->input = input;
	w->vc    = aevt->src_vc;
	w->id    = aevt->id;
	w->watch = aevt->watch;
	w->pres  = 1;

	_output_state[output]->PushWaiting( aevt->dst_vc, w );
      }
    } else {
      if ( _vct ) {
	Error( "Received arrival event for non-head flit in cut-through mode" );
      }

      if ( state != EventNextVCState::busy ) {
	cout << "flit id = " << aevt->id << endl;
	Error( "Received a body flit at a non-busy output VC" );
      }
      
      if ( ( !_output_state[output]->IsInputWaiting( aevt->dst_vc, input, aevt->src_vc ) ) &&
	   ( input == _output_state[output]->GetInput( aevt->dst_vc ) ) &&
	   ( aevt->src_vc == _output_state[output]->GetInputVC( aevt->dst_vc ) ) ) {
	// Body flit part of the current active VC => queue transport event
	// (the weird IsInputWaiting call handles a body flit waiting in addition
	// to a head flit)

	_SendTransport( input, output, aevt );
      } else {

	// VC busy with a differnet transaction => update waiting event
	_output_state[output]->IncrWaiting( aevt->dst_vc, input, aevt->src_vc );
      } 
    }

    delete aevt;
  }
}

void EventRouter::_TransportRequests( int output )
{
  tTransportEvent *tevt;
  
  if ( !_transport_queue[output].empty( ) ) {
    tevt = _transport_queue[output].front( );
    _transport_arbiter[tevt->input]->AddRequest( output );
  }
}

void EventRouter::_TransportArb( int input ) 
{
  tTransportEvent *tevt;

  int    output;
  Buffer *cur_buf;
  Flit   *f;
  Credit *c;

  if ( _transport_free[input] ) {
    _transport_arbiter[input]->Arbitrate( );
    output = _transport_arbiter[input]->Match( );
  } else {
    output = _transport_match[input];
  }

  if ( output != -1 ) {  
    // This completes the match from input to output =>
    // one flit can be transferred

    tevt = _transport_queue[output].front( );
    
    if ( tevt->watch ) {
      cout << "Processing transport event at " << FullName() 
	   << " for flit " << tevt->id << endl;
    }

    cur_buf = _buf[input];
    int vc = tevt->src_vc;

    // Some sanity checking first

    if ( !_active[input][vc] ) {
      Error( "Non-active VC received grant." );
    }

    if ( cur_buf->Empty( vc ) ) {
      return; //Error( "Empty VC received grant." );
    }

    if ( tevt->dst_vc != cur_buf->GetOutputVC( vc ) ) {
      Error( "Transport event's VC does not match input's destination VC." );
    }

    f = cur_buf->RemoveFlit( vc );

    if ( _vct ) {
      if ( f->tail ) {
	_transport_free[input]  = true;
	_transport_match[input] = -1;

	_transport_queue[output].pop( );
	delete tevt;

	_active[input][vc] = false;
      } else {
	_transport_free[input]  = false;
	_transport_match[input] = output;
      }
    } else {
      _transport_free[input]  = true;
      _transport_match[input] = -1;

      _transport_queue[output].pop( );
      delete tevt;

      if ( f->tail ) {
	_active[input][vc] = false;
      }
    }

    c = Credit::New( );
    c->vc.insert(f->vc);
    c->head          = f->head;
    c->tail          = f->tail;
    c->id            = f->id;
    _credit_pipe->Write( c, input );
    
    if ( f->watch && c->tail ) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		  << FullName() << " sending tail credit back for flit " << f->id << endl;
    }

    // Update and forward the flit to the crossbar

    f->hops++;
    f->vc = cur_buf->GetOutputVC( vc );
    _crossbar_pipe->Write( f, output );

    if ( f->watch ) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		  << "Forwarding flit through crossbar at " << FullName() << ":" << endl
		  << *f;
    }  
  }
}

void EventRouter::_OutputQueuing( )
{
  Flit   *f;
  Credit *c;

  for ( int output = 0; output < _outputs; ++output ) {
    f = _crossbar_pipe->Read( output );

    if ( f ) {
      _output_buffer[output].push( f );
    }
  }  

  for ( int input = 0; input < _inputs; ++input ) {
    c = _credit_pipe->Read( input );

    if ( c ) {
      _in_cred_buffer[input].push( c );
    }
  }
}

void EventRouter::_SendFlits( )
{
  for ( int output = 0; output < _outputs; ++output ) {
    if ( !_output_buffer[output].empty( ) ) {
      Flit *f = _output_buffer[output].front( );
      _output_buffer[output].pop( );
      _output_channels[output]->Send( f );
    }
  }
}

void EventRouter::_SendCredits( )
{
  for ( int input = 0; input < _inputs; ++input ) {
    if ( !_in_cred_buffer[input].empty( ) ) {
      Credit *c = _in_cred_buffer[input].front( );
      _in_cred_buffer[input].pop( );
      _input_credits[input]->Send( c );
    }
  }
}

void EventRouter::Display( ostream & os ) const
{
  for ( int input = 0; input < _inputs; ++input ) {
    _buf[input]->Display( os );
  }
}

EventNextVCState::EventNextVCState( const Configuration& config, 
				    Module *parent, const string& name ) :
  Module( parent, name )
{
  _buf_size = config.GetInt( "vc_buf_size" );
  _vcs      = config.GetInt( "num_vcs" );

  _credits.resize(_vcs, _buf_size);
  _presence.resize(_vcs, 0);
  _input.resize(_vcs);
  _inputVC.resize(_vcs);
  _waiting.resize(_vcs);
  _state.resize(_vcs, idle);
}

EventNextVCState::eNextVCState EventNextVCState::GetState( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  return _state[vc];
}

int EventNextVCState::GetPresence( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  return _presence[vc];
}

int EventNextVCState::GetCredits( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  return _credits[vc];
}

int EventNextVCState::GetInput( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  return _input[vc];
}

int EventNextVCState::GetInputVC( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  return _inputVC[vc];
}

bool EventNextVCState::IsWaiting( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  return !_waiting[vc].empty( );
}

void EventNextVCState::PushWaiting( int vc, tWaiting *w )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );

  if ( w->watch ) {
    cout << FullName() << " pushing flit " << w->id
	 << " onto a waiting queue of length " << _waiting[vc].size( ) << endl;
  }

  _waiting[vc].push_back( w );
}

void EventNextVCState::IncrWaiting( int vc, int w_input, int w_vc )
{
  list<tWaiting *>::iterator match;

  // search for match
  for ( match = _waiting[vc].begin( ); match != _waiting[vc].end( ); match++ ) {
    if ( ( (*match)->input == w_input ) &&
	 ( (*match)->vc    == w_vc ) ) break;
  }

  if ( match != _waiting[vc].end( ) ) {
    (*match)->pres++;
  } else {
    Error( "Did not find match in IncrWaiting" );
  }
}

bool EventNextVCState::IsInputWaiting( int vc, int w_input, int w_vc ) const
{
  list<tWaiting *>::const_iterator match;
  bool r;

  // search for match
  for ( match = _waiting[vc].begin( ); match != _waiting[vc].end( ); match++ ) {
    if ( ( (*match)->input == w_input ) &&
	 ( (*match)->vc    == w_vc ) ) break;
  }

  if ( match != _waiting[vc].end( ) ) {
    r = true;
  } else {
    r = false;
  }

  return r;
}

EventNextVCState::tWaiting *EventNextVCState::PopWaiting( int vc )
{
  tWaiting *w;

  assert( ( vc >= 0 ) && ( vc < _vcs ) );
 
  w = _waiting[vc].front( );
  _waiting[vc].pop_front( );

  return w;
}

void EventNextVCState::SetState( int vc, eNextVCState state )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  _state[vc] = state;
}

void EventNextVCState::SetCredits( int vc, int value )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  _credits[vc] = value;
}

void EventNextVCState::SetPresence( int vc, int value )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  _presence[vc] = value;
}

void EventNextVCState::SetInput( int vc, int input )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  _input[vc] = input;
}

void EventNextVCState::SetInputVC( int vc, int in_vc )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  _inputVC[vc] = in_vc;
}
