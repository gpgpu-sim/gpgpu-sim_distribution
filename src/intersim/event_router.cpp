#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "event_router.hpp"
#include "stats.hpp"

EventRouter::EventRouter( const Configuration& config,
                          Module *parent, string name, int id,
                          int inputs, int outputs )
: Router( config,
          parent, name,
          id,
          inputs, outputs )
{
   ostringstream module_name;

   _vcs            = config.GetInt( "num_vcs" );
   _vc_size        = config.GetInt( "vc_buf_size" );

   // Cut-through mode --- packets are not broken
   // up and input buffers are assumed to be 
   // expressed in units of maximum size packets.

   _vct            = config.GetInt( "vct" );

   // Routing

   _rf = GetRoutingFunction( config );

   // Alloc VC's

   _vc = new VC * [_inputs];

   for ( int i = 0; i < _inputs; ++i ) {
      _vc[i] = new VC [_vcs];
      for( int j=0; j < _vcs; ++j ) {
         _vc[i][j].init( config, _outputs );
      }

      for ( int v = 0; v < _vcs; ++v ) { // Name the vc modules
         module_name << "vc_i" << i << "_v" << v;
         _vc[i][v].SetName( this, module_name.str( ) );
         module_name.seekp( 0, ios::beg );
      }
   }

   // Alloc next VCs' state

   _output_state = new EventNextVCState [_outputs];
   for( int j=0; j < _outputs; ++j ) {
      _output_state[j].init( config );
   }

   for ( int o = 0; o < _outputs; ++o ) {
      module_name << "output" << o << "_vc_state";
      _output_state[o].SetName( this, module_name.str( ) );
      module_name.seekp( 0, ios::beg );
   }

   // Alloc arbiters

   _arrival_arbiter = new PriorityArbiter * [_outputs];

   for ( int o = 0; o < _outputs; ++o ) {
      module_name << "arrival_arb_output" << o;
      _arrival_arbiter[o] = 
      new PriorityArbiter( config, this, module_name.str( ), _inputs );
      module_name.seekp( 0, ios::beg );
   }

   _transport_arbiter = new PriorityArbiter * [_inputs];

   for ( int i = 0; i < _inputs; ++i ) {
      module_name << "transport_arb_input" << i;
      _transport_arbiter[i] = 
      new PriorityArbiter( config, this, module_name.str( ), _outputs );
      module_name.seekp( 0, ios::beg );
   }

   // Alloc pipelines (to simulate processing/transmission delays)

   _crossbar_pipe = 
   new PipelineFIFO<Flit>( this, "crossbar_pipeline", _outputs, 
                           _st_prepare_delay + _st_final_delay );

   _credit_pipe =
   new PipelineFIFO<Credit>( this, "credit_pipeline", _inputs,
                             _credit_delay );

   _arrival_pipe =
   new PipelineFIFO<tArrivalEvent>( this, "arrival_pipeline", _inputs,
                                    0 /* FIX THIS EVENTUALLY */);

   // Queues

   _input_buffer  = new queue<Flit *> [_inputs]; 
   _output_buffer = new queue<Flit *> [_outputs]; 

   _in_cred_buffer  = new queue<Credit *> [_inputs]; 
   _out_cred_buffer = new queue<Credit *> [_outputs];

   _arrival_queue   = new queue<tArrivalEvent *> [_inputs];
   _transport_queue = new queue<tTransportEvent *> [_outputs];

   // Misc.

   _transport_free  = new bool [_inputs];
   _transport_match = new int  [_inputs];

   for ( int i = 0; i < _inputs; ++i ) {
      _transport_free[i]  = true;
      _transport_match[i] = -1;
   }
}

EventRouter::~EventRouter( )
{
   for ( int i = 0; i < _inputs; ++i ) {
      delete [] _vc[i];
   }

   delete [] _vc;
   delete [] _output_state;

   for ( int o = 0; o < _outputs; ++o ) {
      delete _arrival_arbiter[o];
   }

   for ( int i = 0; i < _inputs; ++i ) {
      delete _transport_arbiter[i];
   }

   delete [] _arrival_arbiter;
   delete [] _transport_arbiter;

   delete _crossbar_pipe;
   delete _credit_pipe;
   delete _arrival_pipe;

   delete [] _input_buffer;
   delete [] _output_buffer;

   delete [] _in_cred_buffer;
   delete [] _out_cred_buffer;

   delete [] _arrival_queue;
   delete [] _transport_queue;

   delete [] _transport_free;
   delete [] _transport_match;
}

void EventRouter::ReadInputs( )
{
   _ReceiveFlits( );
   _ReceiveCredits( );
}

void EventRouter::InternalStep( )
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
      f = *((*_input_channels)[input]);

      if ( f ) {
         _input_buffer[input].push( f );
      }
   }
}

void EventRouter::_ReceiveCredits( )
{
   Credit *c;

   for ( int output = 0; output < _outputs; ++output ) {
      c = *((*_output_credits)[output]);

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

   if ( _output_state[output].IsWaiting( out_vc ) ) {

      // State remains as busy, but the waiting VC takes over
      w = _output_state[output].PopWaiting( out_vc );

      _output_state[output].SetState( out_vc, EventNextVCState::busy );
      _output_state[output].SetInput( out_vc, w->input );
      _output_state[output].SetInputVC( out_vc, w->vc );

      if ( w->watch ) {
         cout << "Dequeuing waiting arrival event at " << _fullname 
         << " for flit " << w->id << endl;
      }

      credits = _output_state[output].GetCredits( out_vc );

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
            cout << "Injecting transport event at " << _fullname 
            << " for flit " << tevt->id << endl;
         }

         credits--;
         _output_state[output].SetCredits( out_vc, credits );
         _output_state[output].SetPresence( out_vc, w->pres - 1 );

      } else {
         // No credits available, just store presence
         _output_state[output].SetPresence( out_vc, w->pres );
      }

      delete w;

   } else {
      // Tail sent, none waiting => VC is idle
      _output_state[output].SetState( out_vc, EventNextVCState::idle );
   }
}

void EventRouter::_IncomingFlits( )
{
   Flit   *f;
   VC     *cur_vc;

   tArrivalEvent *aevt;

   _arrival_pipe->WriteAll( 0 );

   for ( int input = 0; input < _inputs; ++input ) {
      if ( !_input_buffer[input].empty( ) ) {
         f = _input_buffer[input].front( );
         _input_buffer[input].pop( );

         cur_vc = &_vc[input][f->vc];

         if ( !cur_vc->AddFlit( f ) ) {
            cout << "Error processing flit:" << endl << *f;
            Error( "VC buffer overflow" );
         }

         // Head flit arriving at idle VC
         if ( cur_vc->GetState( ) == VC::idle ) {

            if ( !f->head ) {
               cout << "Non-head flit:" << endl;
               cout << *f;
               Error( "Received non-head flit at idle VC" );
            }

            const OutputSet *route_set;
            int out_vc, out_port;

            cur_vc->Route( _rf, this, f, input );
            route_set = cur_vc->GetRouteSet( );

            if ( !route_set->GetPortVC( &out_port, &out_vc ) ) {
               Error( "The event-driven router requires routing functions with a single (port,vc) output" );
            }

            cur_vc->SetOutput( out_port, out_vc );
            cur_vc->SetState( VC::active );
         } else {
            if ( f->head ) {
               cout << *f;
               Error( "Received head flit at non-idle VC." );
            }
         }

         if ( f->watch ) {
            cout << "Received flit at " << _fullname << ".  Output port = " 
            << cur_vc->GetOutputPort( ) << ", output VC = " 
            << cur_vc->GetOutputVC( ) << endl;
            cout << *f;
         }

         // In cut-through mode, only head flits generate arrivals,
         // otherwise all flits generate

         if ( ( !_vct ) || ( _vct && f->head ) ) {
            // Add the arrival event to a delay pipeline to
            // account for routing/decoding time

            aevt         = new tArrivalEvent;

            aevt->input  = input;
            aevt->output = cur_vc->GetOutputPort( );
            aevt->src_vc = f->vc;
            aevt->dst_vc = cur_vc->GetOutputVC( );
            aevt->head   = f->head;
            aevt->tail   = f->tail;

            //if ( f->head && f->tail ) {
            //	Error( "Head/tail packets not supported." );
            //}

            aevt->watch  = f->watch;
            aevt->id     = f->id;

            _arrival_pipe->Write( aevt, input );

            if ( aevt->watch ) {
               cout << "Injected arrival event at " << _fullname 
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

   credits = _output_state[output].GetCredits( aevt->dst_vc );

   if ( credits > 0 ) {
      // Take a credit and queue a transport event
      credits--;
      _output_state[output].SetCredits( aevt->dst_vc, credits );

      tevt         = new tTransportEvent;
      tevt->src_vc = aevt->src_vc;
      tevt->dst_vc = aevt->dst_vc;
      tevt->input  = input;
      tevt->watch  = aevt->watch;
      tevt->id     = aevt->id;

      _transport_queue[output].push( tevt );

      if ( tevt->watch ) {
         cout << "Injecting transport event at " << _fullname 
         << " for flit " << tevt->id << endl;
      }
   } else {
      if ( aevt->watch ) {
         cout << "No credits available at " << _fullname 
         << " for flit " << aevt->id << " storing presence." << endl;
      }

      // No credits available, just store presence
      pres = _output_state[output].GetPresence( aevt->dst_vc );
      _output_state[output].SetPresence( aevt->dst_vc, pres + 1 );
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

      if ( c->vc_cnt != 1 ) {
         Error( "Code can't handle credit counts not equal to 1." );
      }

      EventNextVCState::eNextVCState state = 
      _output_state[output].GetState( c->vc[0] );

      credits = _output_state[output].GetCredits( c->vc[0] );
      pres    = _output_state[output].GetPresence( c->vc[0] );

      if ( _vct ) {
         // In cut-through mode, only head credits indicate a change in 
         // channel state.

         if ( c->head ) {
            credits++;
            _output_state[output].SetCredits( c->vc[0], credits );
            _ProcessWaiting( output, c->vc[0] );
         }
      } else {
         credits++;
         _output_state[output].SetCredits( c->vc[0], credits );

         if ( c->tail ) { // tail flit -- recycle VC
            if ( state != EventNextVCState::busy ) {
               Error( "Received tail credit at non-busy output VC" );
            }

            _ProcessWaiting( output, c->vc[0] );
         } else if ( ( state == EventNextVCState::busy ) && ( pres > 0 ) ) {
            // Flit is present => generate transport event

            tevt         = new tTransportEvent;
            tevt->input  = _output_state[output].GetInput( c->vc[0] );
            tevt->src_vc = _output_state[output].GetInputVC( c->vc[0] );
            tevt->dst_vc = c->vc[0];
            tevt->watch  = false;
            tevt->id     = -1;

            _transport_queue[output].push( tevt );

            pres--;
            credits--;
            _output_state[output].SetPresence( c->vc[0], pres );
            _output_state[output].SetCredits( c->vc[0], credits );
         }
      }

      delete c;
   }

   // Now process arrival events

   _arrival_arbiter[output]->Arbitrate( );
   input = _arrival_arbiter[output]->Match( );

   if ( input != -1 ) {
      // Winning arrival event gets access to output

      aevt = _arrival_queue[input].front( );
      _arrival_queue[input].pop( );

      if ( aevt->watch ) {
         cout << "Processing arrival event at " << _fullname 
         << " for flit " << aevt->id << endl;
      }

      EventNextVCState::eNextVCState state = 
      _output_state[output].GetState( aevt->dst_vc );

      if ( aevt->head ) { // Head flits
         if ( state == EventNextVCState::idle ) {
            // Allocate the output VC and queue a transport event
            _output_state[output].SetState( aevt->dst_vc, EventNextVCState::busy );
            _output_state[output].SetInput( aevt->dst_vc, input );
            _output_state[output].SetInputVC( aevt->dst_vc, aevt->src_vc );

            _SendTransport( input, output, aevt );
         } else {
            // VC busy => queue a waiting event

            w = new EventNextVCState::tWaiting;

            w->input = input;
            w->vc    = aevt->src_vc;
            w->id    = aevt->id;
            w->watch = aevt->watch;
            w->pres  = 1;

            _output_state[output].PushWaiting( aevt->dst_vc, w );
         }
      } else {
         if ( _vct ) {
            Error( "Received arrival event for non-head flit in cut-through mode" );
         }

         if ( state != EventNextVCState::busy ) {
            cout << "flit id = " << aevt->id << endl;
            Error( "Received a body flit at a non-busy output VC" );
         }

         if ( ( !_output_state[output].IsInputWaiting( aevt->dst_vc, input, aevt->src_vc ) ) &&
              ( input == _output_state[output].GetInput( aevt->dst_vc ) ) &&
              ( aevt->src_vc == _output_state[output].GetInputVC( aevt->dst_vc ) ) ) {
            // Body flit part of the current active VC => queue transport event
            // (the weird IsInputWaiting call handles a body flit waiting in addition
            // to a head flit)

            _SendTransport( input, output, aevt );
         } else {

            // VC busy with a differnet transaction => update waiting event
            _output_state[output].IncrWaiting( aevt->dst_vc, input, aevt->src_vc );
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
   VC     *cur_vc;
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
         cout << "Processing transport event at " << _fullname 
         << " for flit " << tevt->id << endl;
      }

      cur_vc = &_vc[input][tevt->src_vc];

      // Some sanity checking first

      if ( ( cur_vc->GetState( ) != VC::active ) ) {
         Error( "Non-active VC received grant." );
      }

      if ( cur_vc->Empty( ) ) {
         return; //Error( "Empty VC received grant." );
      }

      if ( tevt->dst_vc != cur_vc->GetOutputVC( ) ) {
         Error( "Transport event's VC does not match input's destination VC." );
      }

      f = cur_vc->RemoveFlit( );

      if ( _vct ) {
         if ( f->tail ) {
            _transport_free[input]  = true;
            _transport_match[input] = -1;

            _transport_queue[output].pop( );
            delete tevt;

            cur_vc->SetState( VC::idle );
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
            cur_vc->SetState( VC::idle );
         }
      }

      c = _NewCredit( );
      c->vc[c->vc_cnt] = f->vc;
      c->head          = f->head;
      c->tail          = f->tail;
      c->vc_cnt++;
      c->id            = f->id;
      _credit_pipe->Write( c, input );

      if ( f->watch && c->tail ) {
         cout << _fullname << " sending tail credit back for flit " << f->id << endl;
      }

      // Update and forward the flit to the crossbar

      f->hops++;
      f->vc = cur_vc->GetOutputVC( );
      _crossbar_pipe->Write( f, output );

      if ( f->watch ) {
         cout << "Forwarding flit through crossbar at " << _fullname << ":" << endl;
         cout << *f;
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
   Flit *f;

   for ( int output = 0; output < _outputs; ++output ) {
      if ( !_output_buffer[output].empty( ) ) {
         f = _output_buffer[output].front( );
         _output_buffer[output].pop( );
      } else {
         f = 0;
      }

      *(*_output_channels)[output] = f;
   }
}

void EventRouter::_SendCredits( )
{
   Credit *c;

   for ( int input = 0; input < _inputs; ++input ) {
      if ( !_in_cred_buffer[input].empty( ) ) {
         c = _in_cred_buffer[input].front( );
         _in_cred_buffer[input].pop( );
      } else {
         c = 0;
      }

      *(*_input_credits)[input] = c;
   }
}

void EventRouter::Display( ) const
{
   for ( int input = 0; input < _inputs; ++input ) {
      for ( int v = 0; v < _vcs; ++v ) {
         _vc[input][v].Display( );
      }
   }
}

void EventNextVCState::init( const Configuration& config ) 
{
   _Init( config );
}

EventNextVCState::EventNextVCState( const Configuration& config, 
                                    Module *parent, const string& name ) :
Module( parent, name )
{
   _Init( config );
}

void EventNextVCState::_Init( const Configuration& config )
{
   _buf_size = config.GetInt( "vc_buf_size" );
   _vcs      = config.GetInt( "num_vcs" );

   _credits   = new int [_vcs];
   _presence  = new int [_vcs];
   _input     = new int [_vcs];
   _inputVC   = new int [_vcs];
   _waiting   = new list<tWaiting *> [_vcs];
   _state     = new eNextVCState [_vcs];

   for ( int vc = 0; vc < _vcs; ++vc ) {
      _presence[vc] = 0;
      _credits[vc]  = _buf_size;
      _state[vc]    = idle;
   }
}

EventNextVCState::~EventNextVCState( )
{
   delete [] _credits;
   delete [] _presence;
   delete [] _input;
   delete [] _inputVC;
   delete [] _waiting;
   delete [] _state;
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
      cout << _fullname << " pushing flit " << w->id
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
