// $Id: chaos_router.cpp 5188 2012-08-30 00:31:31Z dub $

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
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

#include "chaos_router.hpp"
#include "random_utils.hpp"
#include "globals.hpp"

ChaosRouter::ChaosRouter( const Configuration& config,
		    Module *parent, const string & name, int id,
		    int inputs, int outputs )
  : Router( config,
	    parent, name,
	    id,
	    inputs, outputs )
{
  int i;

  if ( inputs != outputs ) {
    Error( "Chaos router must have equal number of input and output ports" );
  }

  _buffer_size      = config.GetInt( "vc_buf_size" );
  assert(_buffer_size >= config.GetInt( "const_flits_per_packet" ));

  _multi_queue_size = config.GetInt( "multi_queue_size" );
  
  _cur_channel = 0;
  _read_stall  = 0;

  // Routing

  string rf = config.GetStr("routing_function") + "_" + config.GetStr("topology");
  map<string, tRoutingFunction>::iterator rf_iter = gRoutingFunctionMap.find(rf);
  if(rf_iter == gRoutingFunctionMap.end()) {
    Error("Invalid routing function: " + rf);
  }
  _rf = rf_iter->second;

  _input_route.resize(_inputs);

  for ( i = 0; i < _inputs; ++i ) {
    _input_route[i] = new OutputSet( );
  }

  _mq_route.resize(_multi_queue_size);

  for ( i = 0; i < _multi_queue_size; ++i ) {
    _mq_route[i] = new OutputSet( );
  }				       

  // Alloc pipelines (to simulate processing/transmission delays)

  _crossbar_pipe = 
    new PipelineFIFO<Flit>( this, "crossbar_pipeline", _outputs, 
			    _crossbar_delay );

  // Input and output queues

  _input_frame.resize(_inputs);
  _output_frame.resize(_outputs); 
  _multi_queue.resize(_multi_queue_size);
  
  _credit_queue.resize(_inputs);

  _input_state.resize(_inputs, empty);
  _input_output_match.resize(_inputs, -1);
  _input_mq_match.resize(_inputs, -1);

  _output_matched.resize(_outputs, false);
  _next_queue_cnt.resize(_outputs, 0);

  _multi_match.resize(_multi_queue_size, -1);
  _mq_age.resize(_multi_queue_size);
  _mq_matched.resize(_multi_queue_size, false);
  _multi_state.resize(_multi_queue_size, empty);

  for ( i = 0; i < _multi_queue_size; ++i ) {
    _multi_state[i] = empty;
    _multi_match[i] = -1;
    _mq_matched[i] = false;
  }
}

ChaosRouter::~ChaosRouter( )
{
  int i;

  delete _crossbar_pipe;

  for ( i = 0; i < _inputs; ++i ) {
    delete _input_route[i];
  }

  for ( i = 0; i < _multi_queue_size; ++i ) {
    delete _mq_route[i];
  }
}
  
void ChaosRouter::ReadInputs( )
{
  Flit   *f;
  Credit *c;

  for ( int input = 0; input < _inputs; ++input ) { 
    f = _input_channels[input]->Receive();

    if ( f ) {
      _input_frame[input].push( f );

      if ( f->watch ) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Flit arriving at " << FullName() 
		    << " on channel " << input << endl
		    << *f;
      }

      switch( _input_state[input] ) {
      case empty:
	if ( f->head ) {
	  if ( f->tail ) {
	    _input_state[input] = full;
	  } else {
	    _input_state[input] = filling;
	  }
	  _rf( this, f, input, _input_route[input], false );
	} else {
	  cout << *f;
	  Error( "Empty buffer received non-head flit!" );
	}
	break;

      case filling:
	if ( f->tail ) {
	  _input_state[input] = full;
	} else if ( f->head ) {
	  Error( "Input buffer received another head before previous tail!" );
	}
	break;
	
      case full:
	Error( "Received flit while full!" );
	break;

      case leaving:
	if ( f->head ) {
	  _input_state[input] = shared;

	  if ( f->tail ) {
	    Error( "Received single-flit packet in leaving state!" );
	  }
	} else {
	  cout << *f;
	  Error( "Received non-head flit while packet leaving!" );
	}
	break;

      case cut_through:
	if ( f->tail ) {
	  _input_state[input] = leaving;
	} 
	if ( f->head ) {
	  cout << *f;
	  Error( "Received head flit in cut through buffer!" );
	}
	break;

      case shared:
	if ( f->head ) {
	  Error( "Shared buffer received another head!" );
	} else if ( f->tail ) {
	  cout << "Input " << input << endl;
	  cout << *f;
	  Error( "Shared buffer received another tail!" );
	}
	break;
      }
    }
  }

  // Process incoming credits

  for ( int output = 0; output < _outputs; ++output ) {
    c = _output_credits[output]->Receive();
    
    if ( c ) {
      _next_queue_cnt[output]--;

      if ( _next_queue_cnt[output] < 0 ) {
	Error( "Next queue count fell below zero!" );
      }

      c->Free();
    }
  }
}

void ChaosRouter::_InternalStep( )
{
  _NextInterestingChannel( );
  _OutputAdvance( );

  _crossbar_pipe->Advance( );
}

void ChaosRouter::WriteOutputs( )
{
  _SendFlits( );
  _SendCredits( );
}

bool ChaosRouter::_IsInjectionChan( int chan ) const
{
  return ( chan == _inputs - 1 );
}

bool ChaosRouter::_IsEjectionChan( int chan ) const
{
  return ( chan == _outputs - 1 );
}

bool ChaosRouter::_InputReady( int input ) const
{
  bool ready = false;

  if ( ( _input_state[input] == filling ) ||
       ( _input_state[input] == full ) ) {
    ready = true;
  }

  return ready;
}

bool ChaosRouter::_OutputFull( int out ) const
{
  return ( _output_frame[out].size( ) >= (size_t)_buffer_size );
}

bool ChaosRouter::_OutputAvail( int out ) const
{
  return ( ( !_output_matched[out] ) &&  ( _output_frame[out].empty( ) ) );
	   //&& ( _next_queue_cnt[out] == 0 ) );
  //return ( ( !_output_matched[out] ) && !_OutputFull( out ) );
}

bool ChaosRouter::_MultiQueueFull( int mq ) const
{
  return ( _multi_queue[mq].size( ) >= (size_t)_buffer_size );
}

int ChaosRouter::_InputForOutput( int output ) const
{
  // return an input that prefers this output

  int  input;
  int  offset = RandomInt( _inputs - 1 );
  bool match  = false;

  for ( int i = 0; ( i < _inputs ) && ( !match ); ++i ) {
    input = ( i + offset ) % _inputs;

    if ( _InputReady( input ) &&
	 ( ! _input_route[input]->OutputEmpty( output ) ) ) {
      match = true;
    }
  }

  return match ? input : -1;
}

int ChaosRouter::_MultiQueueForOutput( int output ) const
{
  // return oldest multi queue that prefers the output,
  // or if none prefer and the multi queue is full,
  // return a random entry

  int mq_oldest = -1;
  int mq_age;

  int m, r;

  bool isfull = true;

  for ( int i = 0; i < _multi_queue_size; ++i ) {
    if ( ( _multi_match[i] == -1 ) &&
	 ( ( _multi_state[i] == full ) ||
	   (  _multi_state[i] == filling ) ) ) {
      
      if ( ( !_mq_route[i]->OutputEmpty( output ) ) && 
	   ( ( mq_oldest == -1 ) || ( _mq_age[i] > mq_age ) ) ) {
	mq_oldest = i;
	mq_age    = _mq_age[i];
      }
    } 

    // deroute only if all queues contain head flits ...

    if ( ( _multi_state[i] != full ) &&
	 ( _multi_state[i] != filling ) &&
	 ( _multi_state[i] != shared ) ) {
      isfull = false;
    }
  }

  // Don't deroute MQs to the ejection channel
  if ( ( mq_oldest == -1 ) && isfull && 
       ( !_IsEjectionChan( output ) ) ) {
    r = RandomInt( _multi_queue_size - 1 );

    // Find first routable multi-queue
    for ( int i = 0; i < _multi_queue_size; ++i ) {
      m = ( i + r ) % _multi_queue_size;
      if ( ( _multi_state[m] == filling ) ||
	   ( _multi_state[m] == full ) ) {
	mq_oldest = m;
	//cout << "DEROUTING at " << FullName() << endl;
	break;
      }
    }

    if ( mq_oldest == -1 ) {
      cout << "write stall" << endl;
    }
  }

  return mq_oldest;
}

int ChaosRouter::_FindAvailMultiQueue( ) const
{
  // return any empty multi queue slot

  int avail = -1;

  for ( int i = 0; i < _multi_queue_size; ++i ) {
    if ( ( !_MultiQueueFull( i ) ) && 
	 ( !_mq_matched[i] ) ) {
      avail = i;
      break;
    }
  }

  return avail;
}

void ChaosRouter::_NextInterestingChannel( ) 
{
  bool interesting;

  int  mq_index;
  int  in_index;
  int  mq_avail;

  int c;

  interesting = false;
  mq_index = -1;
  in_index = -1;

  // A channel is interesting if
  // 
  // ( output frame available and 
  //   ( ( a multiqueue packet wants output channel ) or
  //     ( an input packet wants output channel ) or 
  //     ( the multiqueue is full ) ) )
  // or
  // ( the packet at the input channel is stalled )
  
  for ( c = 0; ( c < _inputs ) && ( !interesting ); ++c ) {
    if ( _OutputAvail( _cur_channel ) ) {
      mq_index = _MultiQueueForOutput( _cur_channel );
      in_index = _InputForOutput( _cur_channel );
      
      if ( ( mq_index != -1 ) || ( in_index != -1 ) ) {
	interesting = true;
      }
    } 
    
    if ( _input_state[_cur_channel] == full ) { 
      interesting = true;
    }

    if ( !interesting ) {
      _cur_channel = ( _cur_channel + 1 ) % _inputs;
    }
  }
    
  if ( interesting ) {
    //cout << _cur_channel << " is interesting at " << FullName() << endl;
    
    if ( mq_index != -1 ) {
      //cout << "Match for multi-queue " << mq_index << " at " << FullName() 
      //    << ", output matched = " << _output_matched[_cur_channel] << endl;
      
      _output_matched[_cur_channel] = true;
      _multi_match[mq_index] = _cur_channel;
    } else if ( in_index != -1 ) {
      _output_matched[_cur_channel] = true;
      _input_output_match[in_index] = _cur_channel;
      
      //cout << "Match for input " << in_index << " at " << FullName() << endl;
      
      if ( _input_state[in_index] == full ) {
	_input_state[in_index] = leaving;
      } else if ( _input_state[in_index] == filling ) {
	_input_state[in_index] = cut_through;
      } else {
	Error( "Tried to route input through crossbar that was not full or filling!" );
      }
    }
    
    // Any non-injection channel that is routable is
    // directed to the multi-queue
    if ( ( ( _input_state[_cur_channel] == filling ) ||
	   ( _input_state[_cur_channel] == full ) ) &&
	 ( !_IsInjectionChan( _cur_channel ) ) ) {
      ++_read_stall;
    } else {
      // go to next channel for the next cycle
      _cur_channel = ( _cur_channel + 1 ) % _inputs;
      _read_stall = 0;
    }
  }

  if ( _read_stall > 0 ) {
    mq_avail = _FindAvailMultiQueue( );

    if ( mq_avail != -1 ) {
	if ( _input_state[_cur_channel] == full ) {
	  _input_state[_cur_channel] = leaving;
	} else if ( _input_state[_cur_channel] == filling ) {
	  _input_state[_cur_channel] = cut_through;
	} else {
	  cout << "Input " << _cur_channel << " state = " 
	       << _input_state[_cur_channel] << endl;
	  Error( "Tried to route input throught multi-queue that was not full or filling!" );
	}

	_input_mq_match[_cur_channel] = mq_avail;
	_mq_matched[mq_avail] = true;

	// go to next channel for the next cycle
	_cur_channel = ( _cur_channel + 1 ) % _inputs;
	_read_stall = 0;
    } else {
      ++_read_stall;
      //cout << "stalling at input " << _cur_channel << " (count = " << _read_stall << ")" << endl;
    }
  }
}

void ChaosRouter::_OutputAdvance( )
{
  Flit *f, *f2;
  Credit *c;
  bool advanced;
  int  mq;

  _crossbar_pipe->WriteAll( 0 );

  for ( int i = 0; i < _inputs; ++i ) {
    if ( ( ( _input_output_match[i] != -1 ) ||
	   ( _input_mq_match[i] != -1 ) ) &&
	 ( !_input_frame[i].empty( ) ) ) {
      
      advanced = false;
      f = _input_frame[i].front( );

      /*if ( ! ) {

      } else {
	cout << "Input = " << i 
	     << ", input_output_match = " << _input_output_match[i]
	     << ", input_mq_match = " << _input_mq_match[i] << endl;
	Error( "Input queue empty, but matched!" );
	}*/

      if ( _input_output_match[i] != -1 ) {
	if ( f->tail ) {
	  _output_matched[_input_output_match[i]] = false;
	}

	_crossbar_pipe->Write( f, _input_output_match[i] );
	
	if ( f->watch ) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		      << "Flit traversing crossbar from input queue " 
		      << i << " at " 
		      << FullName() << endl
		      << *f;
	}

	advanced = true;

      } else if ( !_MultiQueueFull( _input_mq_match[i] ) ) {

	mq = _input_mq_match[i];

	if ( f->head ) {
	  _rf( this, f, i, _mq_route[mq], false );
	  _mq_age[mq] = 0;

	  if ( _multi_state[mq] == empty ) {
	    _multi_state[mq] = filling;
	  } else if ( _multi_state[mq] == leaving ) {
	    _multi_state[mq] = shared;
	  } else {
	    Error( "Multi-queue received head while not empty or leaving!" );
	  }
	}
	
	if ( f->tail ) {
	  _mq_matched[mq] = false;

	  if ( _multi_state[mq] == filling ) {
	    _multi_state[mq] = full;
	  } else if ( _multi_state[mq] == cut_through ) {
	    _multi_state[mq] = leaving; 
	  } else {
	    Error( "Multi-queue received tail while not filling or cutting-through!" );
	  }
	}
	
	_multi_queue[mq].push( f );
	
	if ( f->watch ) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		      << "Flit stored in multiqueue at " 
		      << FullName() << endl
		      << "State = " << _multi_state[mq] << endl
		      << *f;
	}

	advanced = true;
      }

      if ( advanced ) {
	_input_frame[i].pop( );

	if ( f->tail ) { // last in packet, update state
	  if ( _input_state[i] == leaving ) {
	    _input_state[i] = empty;
	  } else if ( _input_state[i] == shared ) {
	    _input_state[i] = filling;
	    f2 = _input_frame[i].front( );
	    // update routes
	    _rf( this, f2, i, _input_route[i], false );
	  }
	  
	  _input_output_match[i] = -1;
	  _input_mq_match[i]     = -1;
	}
	
	c = Credit::New( );
	c->vc.insert(0);
	_credit_queue[i].push( c );
      }
    }
  }

  for ( int m = 0; m < _multi_queue_size; ++m ) {
    if ( _multi_match[m] != -1 ) {
      if ( !_multi_queue[m].empty( ) ) {
	f = _multi_queue[m].front( );
	_multi_queue[m].pop( );
      } else {
	cout << "State = " << _multi_state[m] << endl;
	Error( "Multi queue empty, but matched!" );
      }

      _crossbar_pipe->Write( f, _multi_match[m] );

      if ( f->watch ) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Flit traversing crossbar from multiqueue slot "
		    << m << " at " 
		    << FullName() << endl
		    << *f;
      }

      if ( f->head ) {
	if ( _multi_state[m] == filling ) {
	  _multi_state[m] = cut_through;
	} else if ( _multi_state[m] == full ) {
	  _multi_state[m] = leaving;
	} else {
	  Error( "Multi-queue sent head while not filling or full!" );
	}
      }

      if ( f->tail ) {
	_output_matched[_multi_match[m]] = false;
	_multi_match[m] = -1;

	if ( _multi_state[m] == shared ) {
	  _multi_state[m] = filling;
	} else if ( _multi_state[m] == leaving ) {
	  _multi_state[m] = empty;
	} else {
	  cout << "State = " << _multi_state[m] << endl;
	  cout << *f;
	  Error( "Multi-queue sent tail while not leaving or shared!" );
	}
      }
    }

    _mq_age[m]++;
  }
}


void ChaosRouter::_SendFlits( )
{
  for ( int output = 0; output < _outputs; ++output ) {
    Flit *f = _crossbar_pipe->Read( output );

    if ( f ) {
      _output_frame[output].push( f );
      f->hops++;
    }

    if ( ( _next_queue_cnt[output] < _buffer_size ) &&
	 ( !_output_frame[output].empty( ) ) ) {
      _output_channels[output]->Send( _output_frame[output].front( ) );
      _output_frame[output].pop( );
      ++_next_queue_cnt[output];
    }
  }
}

void ChaosRouter::_SendCredits( )
{
  for ( int input = 0; input < _inputs; ++input ) {
    if ( !_credit_queue[input].empty( ) ) {
      Credit *c = _credit_queue[input].front( );
      _credit_queue[input].pop( );
      _input_credits[input]->Send( c );
    }
  }
}

void ChaosRouter::Display( ostream & os ) const
{
}
