#include "booksim.hpp"
#include <sstream>
#include <math.h>
#include <assert.h>

#include "trafficmanager.hpp"
#include "random_utils.hpp" 
#include "interconnect_interface.h"

//Turns on flip tracking!
//#ifndef DEBUG
#define DEBUG 0
//#endif

int MATLAB_OUTPUT        = 0;    // output data in MATLAB friendly format
int DISPLAY_LAT_DIST     = 1; // distribution of packet latencies
int DISPLAY_HOP_DIST     = 1;    // distribution of hop counts
int DISPLAY_PAIR_LATENCY = 0;    // avg. latency for each s-d pair

TrafficManager::TrafficManager( const Configuration &config, Network *net , int u_id)
: Module( 0, "traffic_manager" )
{
   int s;
   ostringstream tmp_name;
   string sim_type, priority;

   uid = u_id;
   _net    = net;
   _cur_id = 0;

   _sources = _net->NumSources( );
   _dests   = _net->NumDests( );

   // ============ Message priorities ============ 

   config.GetStr( "priority", priority );

   _classes = 1;

   if ( priority == "class" ) {
      _classes  = 2;
      _pri_type = class_based;
   } else if ( priority == "age" ) {
      _pri_type = age_based;
   } else if ( priority == "none" ) {
      _pri_type = none;
   } else {
      Error( "Unknown priority " + priority );
   }

   // ============ Injection VC states  ============ 

   _buf_states = new BufferState * [_sources];

   for ( s = 0; s < _sources; ++s ) {
      tmp_name << "buf_state_" << s;
      _buf_states[s] = new BufferState( config, this, tmp_name.str( ) );
      tmp_name.seekp( 0, ios::beg );
   }

   // ============ Injection queues ============ 

   _voqing = config.GetInt( "voq" );

   if ( _voqing ) {
      _use_lagging = false;
   } else {
      _use_lagging = true;
   }

   _time               = 0;
   _warmup_time        = -1;
   _drain_time         = -1;
   _empty_network      = false;

   _measured_in_flight = 0;
   _total_in_flight    = 0;

   if ( _use_lagging ) {
      _qtime    = new int * [_sources];
      _qdrained = new bool * [_sources];
   }

   if ( _voqing ) {
      _voq         = new list<Flit *> * [_sources];
      _active_list = new list<int> [_sources];
      _active_vc   = new bool * [_sources];
   }

   _partial_packets = new list<Flit *> * [_sources];

   for ( s = 0; s < _sources; ++s ) {
      if ( _use_lagging ) {
         _qtime[s]    = new int [_classes];
         _qdrained[s] = new bool [_classes];
      }

      if ( _voqing ) {
         _voq[s]       = new list<Flit *> [_dests];
         _active_vc[s] = new bool [_dests];
      }

      _partial_packets[s] = new list<Flit *> [_classes];
   }

   _split_packets = config.GetInt( "split_packets" );

   credit_return_queue = new queue<Flit *> [_sources];

   // ============ Reorder queues ============ 

   _reorder = config.GetInt( "reorder" ) ? true : false;

   if ( _reorder ) {
      _inject_sqn = new int * [_sources];
      _rob_sqn    = new int * [_sources];
      _rob_sqn_max = new int * [_sources];
      _rob        = new priority_queue<Flit *, vector<Flit *>, flitp_compare> * [_sources];

      for ( int i = 0; i < _sources; ++i ) {
         _inject_sqn[i] = new int [_dests];
         _rob_sqn[i]    = new int [_dests];
         _rob_sqn_max[i] = new int [_dests];
         _rob[i]        = new priority_queue<Flit *, vector<Flit *>, flitp_compare> [_dests];

         for ( int j = 0; j < _dests; ++j ) {
            _inject_sqn[i][j] = 0;
            _rob_sqn[i][j]    = 0;
            _rob_sqn_max[i][j] = 0;
         }
      }

      _rob_pri = new int [_dests];

      for ( int i = 0; i < _dests; ++i ) {
         _rob_pri[i] = 0;
      }
   }

   // ============ Statistics ============ 

   _latency_stats   = new Stats * [_classes];
   _overall_latency = new Stats * [_classes];

   for ( int c = 0; c < _classes; ++c ) {
      tmp_name << "latency_stat_" << c;
      _latency_stats[c] = new Stats( this, tmp_name.str( ), 1.0, 1000 );
      tmp_name.seekp( 0, ios::beg );

      tmp_name << "overall_latency_stat_" << c;
      _overall_latency[c] = new Stats( this, tmp_name.str( ), 1.0, 1000 );
      tmp_name.seekp( 0, ios::beg );  
   }

   _pair_latency     = new Stats * [_dests];
   _accepted_packets = new Stats * [_dests];

   for ( int i = 0; i < _dests; ++i ) {
      tmp_name << "pair_stat_" << i;
      _pair_latency[i] = new Stats( this, tmp_name.str( ), 1.0, 250 );
      tmp_name.seekp( 0, ios::beg );

      tmp_name << "accepted_stat_" << i;
      _accepted_packets[i] = new Stats( this, tmp_name.str( ) );
      tmp_name.seekp( 0, ios::beg );    
   }

   _hop_stats            = new Stats( this, "hop_stats", 1.0, 20 );;
   _overall_accepted     = new Stats( this, "overall_acceptance" );
   _overall_accepted_min = new Stats( this, "overall_min_acceptance" );

   if ( _reorder ) {
      _rob_latency = new Stats( this, "rob_latency", 1.0, 1000 );
      _rob_size    = new Stats( this, "rob_size", 1.0, 250 );
   }

   _flit_timing = config.GetInt( "flit_timing" );

   // ============ Simulation parameters ============ 

   _load = config.GetFloat( "injection_rate" ); 
   _packet_size = config.GetInt( "const_flits_per_packet" );

   _total_sims = config.GetInt( "sim_count" );

   _internal_speedup = config.GetFloat( "internal_speedup" );
   _partial_internal_cycles = 0.0;

   _traffic_function  = NULL; // GetTrafficFunction( config ); // Not used by gpgpusim
   _routing_function  = GetRoutingFunction( config );
   _injection_process = NULL; // GetInjectionProcess( config ); // Not used by gpgpusim

   config.GetStr( "sim_type", sim_type );

   if ( sim_type == "latency" ) {
      _sim_mode = latency;
   } else if ( sim_type == "throughput" ) {
      _sim_mode = throughput;
   } else {
      Error( "Unknown sim_type " + sim_type );
   }

   _sample_period   = config.GetInt( "sample_period" );
   _max_samples     = config.GetInt( "max_samples" );
   _warmup_periods  = config.GetInt( "warmup_periods" );
   _latency_thres   = config.GetFloat( "latency_thres" );
   _include_queuing = config.GetInt( "include_queuing" );
}

TrafficManager::~TrafficManager( )
{
   for ( int s = 0; s < _sources; ++s ) {
      if ( _use_lagging ) {
         delete [] _qtime[s];
         delete [] _qdrained[s];
      }
      if ( _voqing ) {
         delete [] _voq[s];
         delete [] _active_vc[s];
      }
      delete [] _partial_packets[s];
      delete _buf_states[s];
   }

   if ( _use_lagging ) {
      delete [] _qtime;
      delete [] _qdrained;
   }

   if ( _voqing ) {
      delete [] _voq;
      delete [] _active_vc;
   }

   if ( _reorder ) {
      for ( int i = 0; i < _sources; ++i ) {
         delete [] _inject_sqn[i]; 
         delete [] _rob_sqn[i];
         delete [] _rob_sqn_max[i];
         delete [] _rob[i];
      }

      delete [] _inject_sqn;
      delete [] _rob_sqn;
      delete [] _rob_sqn_max;
      delete [] _rob;
      delete [] _rob_pri;

      delete _rob_latency;
      delete _rob_size;
   }

   delete [] _buf_states;
   delete [] _partial_packets;

   for ( int c = 0; c < _classes; ++c ) {
      delete _latency_stats[c];
      delete _overall_latency[c];
   }

   delete [] _latency_stats;
   delete [] _overall_latency;

   delete _hop_stats;
   delete _overall_accepted;
   delete _overall_accepted_min;

   for ( int i = 0; i < _dests; ++i ) {
      delete _accepted_packets[i];
      delete _pair_latency[i];
   }

   delete [] _accepted_packets;
   delete [] _pair_latency;
}

Flit *TrafficManager::_NewFlit( )
{
   Flit *f;
   f = new Flit;

   f->id    = _cur_id;
   f->hops  = 0;
   f->watch = false;

   // Add specific packet watches for debugging
   if (DEBUG || f->id == -1 ) {
      f->watch = true;
   }

   _in_flight[_cur_id] = true;
   ++_cur_id;
   return f;
}

void TrafficManager::_RetireFlit( Flit *f, int dest )
{
   static int sample_num = 0;

   map<int, bool>::iterator match;

   match = _in_flight.find( f->id );

   if ( match != _in_flight.end( ) ) {
      if ( f->watch ) {
         cout << "Matched flit ID = " << f->id << endl;
      }
      _in_flight.erase( match );
   } else {
      cout << "Unmatched flit! ID = " << f->id << endl;
      Error( "" );
   }

   if ( f->watch ) {
      cout << "Ejecting flit " << f->id 
      << ",  lat = " << _time - f->time 
      << ", src = " << f->src 
      << ", dest = " << f->dest << endl;
   }

   // Only record statistics once per packet (at true tails)
   // unless flit-level timing is on 
   if ( f->tail || _flit_timing ) {
      _total_in_flight--;
      if ( _total_in_flight < 0 ) {
         Error( "Total in flight count dropped below zero!" );
      }

      if ( ( _sim_state == warming_up ) || f->record ) {
         if ( f->true_tail || _flit_timing ) {
            _hop_stats->AddSample( f->hops );
            assert( (_time - f->time)>0 );
            switch ( _pri_type ) {
            case class_based:
               _latency_stats[f->pri]->AddSample( (_time - f->time) );
               break;
            case age_based:   // fall through
            case none:
               _latency_stats[0]->AddSample( (_time - f->time) );
               break;
            }

            if ( _reorder ) {
               _rob_latency->AddSample( (_time - f->rob_time ));
            }

            if ( f->src == 0 ) {
               _pair_latency[dest]->AddSample( (_time - f->time ) );
            }
         }

         if ( f->record ) {

            _measured_in_flight--;
            if ( _measured_in_flight < 0 ) {
               Error( "Measured in flight count dropped below zero!" );
            }
         }

         ++sample_num;
      }
   }

   delete f;
}

//never called in gpgpusim
int TrafficManager::_IssuePacket( int source, int cl ) const
{ 
   float class_load;
   if ( _pri_type == class_based ) {
      if ( cl == 0 ) {
         class_load = 0.9 * _load;
      } else {
         class_load = 0.1 * _load;
      }
   } else {
      class_load = _load;
   }
   //gppgusim_injector ignores second parameter!
   return _injection_process( source, class_load );
}

void TrafficManager::_GeneratePacket( int source, int psize /*# of flits*/ , 
                                      int cl, int time, void* data, int dest )
{
   Flit *f;
   bool record;
   bool split_head;
   bool split_tail;

   if ( ( _sim_state == running ) ||
        ( ( _sim_state == draining ) && ( time < _drain_time ) ) ) {
      record = true;
   } else {
      record = false;
   }

   for ( int i = 0; i < psize; ++i ) {
      f = _NewFlit( );

      split_head = false;
      split_tail = false;

      if ( _split_packets > 0 ) {
         if ( ( i % _split_packets ) == 0 ) {
            split_head = true;
         }

         if ( ( i % _split_packets ) == ( _split_packets - 1 ) ) {
            split_tail = true;
         }
      }

      f->src    = source;
      f->time   = time;
      f->record = record;
      f->data = data;
      f->net_num = uid; 
      if ( ( i == 0 ) || ( split_head ) ) {     // Head flit
         f->head = true;
         f->dest = dest;
      } else {
         f->head = false;
         f->dest = -1;
      }

      f->true_tail = false;
      if ( ( i == ( psize - 1 ) ) || ( split_tail ) ) { // Tail flit
         f->tail = true;

         if ( i == ( psize - 1 ) ) {
            f->true_tail = true;
         }
      } else {
         f->tail = false;
      }

      if ( _reorder ) {
         f->sn = _inject_sqn[source][dest];
         _inject_sqn[source][dest]++;
      }

      switch ( _pri_type ) {
      case class_based:
         f->pri = cl; break;
      case age_based:
         f->pri = -time; break;
      case none:
         f->pri = 0; break;
      }

      f->vc  = -1;

      if ( f->watch ) {
         cout << "Generating flit at time " << time << endl;
         cout << *f;
      }

      if ( f->tail || _flit_timing ) {
         if ( record ) {
            ++_measured_in_flight;
         }
         ++_total_in_flight;
      }

      if ( _flit_timing ) {
         time++;
      }

      _partial_packets[source][cl].push_back( f );
   }
}

void TrafficManager::_FirstStep( )
{  
   // Ensure that all outputs are defined before starting simulation

   _net->WriteOutputs( );

   for ( int output = 0; output < _net->NumDests( ); ++output ) {
      _net->WriteCredit( 0, output );
   }
}

void TrafficManager::_ClassInject( )
{
   Flit   *f, *nf;
   Credit *cred;

   // Receive credits and inject new traffic
   for ( int input = 0; input < _net->NumSources( ); ++input ) {

      cred = _net->ReadCredit( input );
      if ( cred ) {
         _buf_states[input]->ProcessCredit( cred );
         delete cred;
      }

      bool write_flit    = false;
      int  highest_class = 0;
      bool generated;

      for ( int c = 0; c < _classes; ++c ) {
         // Potentially generate packets for any (input,class)
         // that is currently empty
         if ( _partial_packets[input][c].empty( ) ) {
            generated = false;

            if ( !_empty_network ) {
               if ( ( _sim_state == draining ) && 
                    ( _qtime[input][c] > _drain_time ) ) {
                  _qdrained[input][c] = true;
               }
            }

            if ( generated ) {
               highest_class = c;
            }

         } else {
            highest_class = c;
         }
      }

      // Now, check partially issued packet to
      // see if it can be issued
      if ( !_partial_packets[input][highest_class].empty( ) ) {
         f = _partial_packets[input][highest_class].front( );

         if ( f->head && ( f->vc == -1 ) ) { // Find first available VC
            f->vc = _buf_states[input]->FindAvailable( );

            if ( f->vc != -1 ) {
               _buf_states[input]->TakeBuffer( f->vc );
            }
         }

         if ( f->vc != -1 ) {
            if ( !_buf_states[input]->IsFullFor( f->vc ) ) {

               _partial_packets[input][highest_class].pop_front( );
               _buf_states[input]->SendingFlit( f );
               time_vector_update_icnt_injected(f->data, input);
               write_flit = true;

               // Pass VC "back"
               if ( !_partial_packets[input][highest_class].empty( ) && !f->tail ) {
                  nf = _partial_packets[input][highest_class].front( );
                  nf->vc = f->vc;
               }
            }
            if ( f->watch ) {
               cout << "Flit " << f->id << " written into injection port at time " << _time << endl;
            }
         } else {
            if ( f->watch ) {
               cout << "Flit " << f->id << " stalled at injection waiting for available VC at time " << _time << endl;
            }
         }
      }

      _net->WriteFlit( write_flit ? f : 0, input );
   }
}

void TrafficManager::_VOQInject( )
{
   Flit   *f;
   Credit *cred;

   int vc;
   int dest;

   for ( int input = 0; input < _net->NumSources( ); ++input ) {

      // Receive credits 
      cred = _net->ReadCredit( input );
      if ( cred ) {
         _buf_states[input]->ProcessCredit( cred );

         for ( int i = 0; i < cred->vc_cnt; i++ ) {
            vc = cred->vc[i];

            // If this credit enables a VC that has packets waiting,
            // set the VC to active (append it to the active list)

            if ( !_voq[input][vc].empty( ) && !_active_vc[input][vc] ) {
               f = _voq[input][vc].front( );

               if ( ( f->head && _buf_states[input]->IsAvailableFor( vc ) ) ||
                    ( !f->head && !_buf_states[input]->IsFullFor( vc ) ) ) {
                  _active_list[input].push_back( vc );
                  _active_vc[input][vc] = true;
               }
            }
         }

         delete cred;
      }
/*
      if ( !_empty_network ) {
         // Inject packets
         psize = _IssuePacket( input, 0 );
      } else {
         psize = 0;
      }
*/
      if ( !_partial_packets[input][0].empty( )/*was psize */) {
         //_GeneratePacket( input, psize, 0, _time ); //already generated in interconnect_push
         dest = -1;

         bool wasempty = false;

         // Move a generated packet to the appropriate VOQ
         while ( !_partial_packets[input][0].empty( ) ) {
            f = _partial_packets[input][0].front( );
            _partial_packets[input][0].pop_front( );
            time_vector_update_icnt_injected(f->data, input);

            if ( f->head ) {
               dest = f->dest; 
               wasempty = _voq[input][dest].empty( );
            }

            if ( dest == -1 ) {
               Error( "Didn't see head flit in VOQ injection" );
            }

            f->dest = dest;
            f->vc   = dest;

            _voq[input][dest].push_back( f );
         }

         // If this packet enables a VC,
         // set the VC to active (append it to the active list)
         if ( wasempty &&
              ( !_active_vc[input][dest] ) && 
              ( _buf_states[input]->IsAvailableFor( dest ) ) ) {
            _active_list[input].push_back( dest );
            _active_vc[input][dest] = true;
         }
      }

      // Write packets to the network
      if ( !_active_list[input].empty( ) ) {

         dest = _active_list[input].front( );
         _active_list[input].pop_front( );

         if ( _voq[input][dest].empty( ) ) {
            Error( "VOQ marked as active, but empty" );
         }

         f = _voq[input][dest].front( );
         _voq[input][dest].pop_front( );

         if ( f->head ) {
            _buf_states[input]->TakeBuffer( dest );
         }

         _buf_states[input]->SendingFlit( f );
         _net->WriteFlit( f, input );

         // Inactivate VC if it can't accept any more flits or
         // no more flits are available to be sent
         if ( ( f->tail && _buf_states[input]->IsAvailableFor( dest ) ) ||
              ( !f->tail && !_buf_states[input]->IsFullFor( dest ) ) ) {
            _active_list[input].push_back( dest );
         } else {
            _active_vc[input][dest] = false;
         }

      } else {
         _net->WriteFlit( 0, input );
      }
   }
}

Flit *TrafficManager::_ReadROB( int dest )
{
   int  src;
   Flit *f;

   src = _rob_pri[dest];
   f   = 0;

   for ( int i = 0; i < _sources; ++i ) {

      if ( !_rob[src][dest].empty( ) ) {
         f = _rob[src][dest].top( );

         if ( f->sn == _rob_sqn[src][dest] ) {
            _rob[src][dest].pop( );
            _rob_sqn[src][dest]++;
            _rob_pri[dest] = ( src + 1 ) % _sources;
            break;
         } else {
            f = 0;
         }
      }

      src = ( src + 1 ) % _sources;
   }

   return f;
}

void TrafficManager::_Step( )
{
   Flit   *f;
   Credit *cred;

   // Inject traffic
   if ( _voqing ) {
      _VOQInject( );
   } else {
      _ClassInject( );
   }

   // Advance network

   _net->ReadInputs( );

   _partial_internal_cycles += _internal_speedup;
   while ( _partial_internal_cycles >= 1.0 ) {
      _net->InternalStep( );
      _partial_internal_cycles -= 1.0;
   }

   _net->WriteOutputs( );

   ++_time;                                        

   // Eject traffic and send credits
   Flit   *last_valid_flit; //= new Flit;
   for ( int output = 0; output < _dests; ++output ) {
      f = _net->ReadFlit( output );

      if ( f ) {
         if (1 || f->tail) {
            write_out_buf(output, f); // it should have space!
            if ( f->watch ) {
               cout << "Sent flit " << f->id << " to output buffer " << output << endl;
               cout << " Not sending the credit yet! " <<endl;
            }
         } else {
            if ( f->watch ) {
               cout << "ejected flit " << f->id << " at output " << output << endl;
               cout << "sending credit for " << f->vc << endl;
            }

            if ( _reorder ) {
               if ( f->watch ) {
                  cout << "adding flit " << f->id << " to reorder buffer" << endl;
                  cout << "flit's SN is " << f->sn << " buffer's SN is " 
                  << _rob_sqn[f->src][f->dest] << endl;
               }

               if ( f->sn > _rob_sqn_max[f->src][f->dest] ) {
                  _rob_sqn_max[f->src][f->dest] = f->sn;
               }

               if ( f->head ) {
                  _rob_size->AddSample( f->sn - _rob_sqn[f->src][f->dest] );
               }

               f->rob_time = _time;
               _rob[f->src][output].push( f );
            } else {
               _RetireFlit( f, output );
               if ( !_empty_network ) {
                  _accepted_packets[output]->AddSample( 1 );
               }
            }
         }
      }
      transfer2boundary_buf( output );
      if (!credit_return_queue[output].empty()) {
         last_valid_flit = credit_return_queue[output].front();
         credit_return_queue[output].pop();
      } else {
         last_valid_flit=NULL;
      }
      if (last_valid_flit) {


         cred = new Credit( 1 );
         cred->vc[0] =last_valid_flit->vc;
         cred->vc_cnt = 1;
         cred->head = last_valid_flit->head;
         cred->tail =last_valid_flit->tail;

         _net->WriteCredit( cred, output );
         if (last_valid_flit->watch) {
            cout <<"WE WROTE A CREDIT for flit "<<last_valid_flit->id<<"To output "<<output<< endl;
         }
         _RetireFlit(last_valid_flit, output );
         if ( !_empty_network ) {
            _accepted_packets[output]->AddSample( 1 );
         }
      } else {
         _net->WriteCredit( 0, output );

         if ( !_reorder && !_empty_network) {
            _accepted_packets[output]->AddSample( 0 );
         }
      }

      if ( _reorder ) {
         f = _ReadROB( output );

         if ( f ) {
            if ( f->watch ) {
               cout << "flit " << f->id << " removed from ROB at output " << output << endl;
            }

            _RetireFlit( f, output );
            if ( !_empty_network ) {
               _accepted_packets[output]->AddSample( 1 );
            }
         } else {
            if ( !_empty_network ) {
               _accepted_packets[output]->AddSample( 0 );
            }
         }
      }
   }
}

bool TrafficManager::_PacketsOutstanding( ) const
{
   bool outstanding;

   if ( _measured_in_flight == 0 ) {
      outstanding = false;

      if ( _use_lagging ) {
         for ( int c = 0; c < _classes; ++c ) {
            for ( int s = 0; s < _sources; ++s ) {
               if ( !_qdrained[s][c] ) {
#ifdef DEBUG_DRAIN
                  cout << "waiting on queue " << s << " class " << c;
                  cout << ", time = " << _time << " qtime = " << _qtime[s][c] << endl;
#endif
                  outstanding = true;
                  break;
               }
            }
            if ( outstanding ) {
               break;
            }
         }
      }
   } else {
#ifdef DEBUG_DRAIN
      cout << "in flight = " << _measured_in_flight << endl;
#endif
      outstanding = true;
   }

   return outstanding;
}

void TrafficManager::_ClearStats( )
{
   for ( int c = 0; c < _classes; ++c ) {
      _latency_stats[c]->Clear( );
   }

   for ( int i = 0; i < _dests; ++i ) {
      _accepted_packets[i]->Clear( );
      _pair_latency[i]->Clear( );
   }

   if ( _reorder ) {
      _rob_latency->Clear( );
      _rob_size->Clear( );
   }
}

int TrafficManager::_ComputeAccepted( double *avg, double *min ) const 
{
   int dmin;

   *min = 1.0;
   *avg = 0.0;

   for ( int d = 0; d < _dests; ++d ) {
      if ( _accepted_packets[d]->Average( ) < *min ) {
         *min = _accepted_packets[d]->Average( );
         dmin = d;
      }
      *avg += _accepted_packets[d]->Average( );
   }

   *avg /= (double)_dests;

   return dmin;
}

void TrafficManager::_DisplayRemaining( ) const 
{
   map<int, bool>::const_iterator iter;
   int i;

   cout << "Remaining flits (" << _measured_in_flight << " measurement packets) : ";
   for ( iter = _in_flight.begin( ), i = 0;
       ( iter != _in_flight.end( ) ) && ( i < 20 );
       iter++, i++ ) {
      cout << iter->first << " ";
   }
   cout << endl;
}

//special initilization each tiem a new GPU grid is started
void TrafficManager::IcntInitPerGrid  (int time)
{     //some initialization parts of _SingleSim for gpgpgusim
   _time =  time ;
   if ( _use_lagging ) {
      for ( int s = 0; s < _sources; ++s ) {
         for ( int c = 0; c < _classes; ++c  ) {
            _qtime[s][c]    = _time; // Was Zero 
            _qdrained[s][c] = false;
         }
      }
   }

   if ( _voqing ) {
      for ( int s = 0; s < _sources; ++s ) {
         for ( int d = 0; d < _dests; ++d ) {
            _active_vc[s][d] = false;
         }
      }
   }
   _sim_state    = running;
   _ClearStats( );
}

bool TrafficManager::_SingleSim( )
{
   int  iter;
   int  total_phases;
   int  converged;
   int  max_outstanding;
   int  empty_steps;

   double cur_latency;
   double prev_latency;

   double cur_accepted;
   double prev_accepted;

   double warmup_threshold;
   double stopping_threshold;
   double acc_stopping_threshold;

   double min, avg;

   bool   clear_last;

   _time = 0;

   if ( _use_lagging ) {
      for ( int s = 0; s < _sources; ++s ) {
         for ( int c = 0; c < _classes; ++c  ) {
            _qtime[s][c]    = 0;
            _qdrained[s][c] = false;
         }
      }
   }

   if ( _voqing ) {
      for ( int s = 0; s < _sources; ++s ) {
         for ( int d = 0; d < _dests; ++d ) {
            _active_vc[s][d] = false;
         }
      }
   }

   stopping_threshold     = 0.01;
   acc_stopping_threshold = 0.01;
   warmup_threshold       = 0.05;
   iter            = 0;
   converged       = 0;
   max_outstanding = 0;
   total_phases    = 0;

   // warm-up ...
   // reset stats, all packets after warmup_time marked
   // converge
   // draing, wait until all packets finish

   _sim_state    = warming_up;
   total_phases  = 0;
   prev_latency  = 0;
   prev_accepted = 0;

   _ClearStats( );
   clear_last    = false;

   while ( ( total_phases < _max_samples ) && 
           ( ( _sim_state != running ) || 
             ( converged < 3 ) ) ) {

      if ( clear_last || ( ( _sim_state == warming_up ) && ( (total_phases & 0x1) == 0 ) ) ) {
         clear_last = false;
         _ClearStats( );
      }

      for ( iter = 0; iter < _sample_period; ++iter ) {
         _Step( );
      } 

      cout << "%=================================" << endl;

      int dmin;

      cur_latency = _latency_stats[0]->Average( );
      dmin = _ComputeAccepted( &avg, &min );
      cur_accepted = avg;

      cout << "% Average latency = " << cur_latency << endl;

      if ( _reorder ) {
         cout << "% Reorder latency = " << _rob_latency->Average( ) << endl;
         cout << "% Reorder size = " << _rob_size->Average( ) << endl;
      }

      cout << "% Accepted packets = " << min << " at node " << dmin << " (avg = " << avg << ")" << endl;

      if ( MATLAB_OUTPUT ) {
         cout << "lat(" << total_phases + 1 << ") = " << cur_latency << ";" << endl;
         cout << "thru(" << total_phases + 1 << ",:) = [ ";
         for ( int d = 0; d < _dests; ++d ) {
            cout << _accepted_packets[d]->Average( ) << " ";
         }
         cout << "];" << endl;
      }

      // Fail safe
      if ( ( _sim_mode == latency ) && ( cur_latency >_latency_thres ) ) {
         cout << "Average latency is getting huge" << endl;
         converged = 0; 
         _sim_state = warming_up;
         break;
      }

      cout << "% latency change    = " << fabs( ( cur_latency - prev_latency ) / cur_latency ) << endl;
      cout << "% throughput change = " << fabs( ( cur_accepted - prev_accepted ) / cur_accepted ) << endl;

      if ( _sim_state == warming_up ) {

         if ( _warmup_periods == 0 ) {
            if ( _sim_mode == latency ) {
               if ( ( fabs( ( cur_latency - prev_latency ) / cur_latency ) < warmup_threshold ) &&
                    ( fabs( ( cur_accepted - prev_accepted ) / cur_accepted ) < warmup_threshold ) ) {
                  cout << "% Warmed up ..." << endl;
                  clear_last = true;
                  _sim_state = running;
               }
            } else {
               if ( fabs( ( cur_accepted - prev_accepted ) / cur_accepted ) < warmup_threshold ) {
                  cout << "% Warmed up ..." << endl;
                  clear_last = true;
                  _sim_state = running;
               }
            }
         } else {
            if ( total_phases + 1 >= _warmup_periods ) {
               cout << "% Warmed up ..." << endl;
               clear_last = true;
               _sim_state = running;
            }
         }
      } else if ( _sim_state == running ) {
         if ( _sim_mode == latency ) {
            if ( ( fabs( ( cur_latency - prev_latency ) / cur_latency ) < stopping_threshold ) &&
                 ( fabs( ( cur_accepted - prev_accepted ) / cur_accepted ) < acc_stopping_threshold ) ) {
               ++converged;
            } else {
               converged = 0;
            }
         } else {
            if ( fabs( ( cur_accepted - prev_accepted ) / cur_accepted ) > acc_stopping_threshold ) {
               converged = 0;
            }
         } 
      }

      prev_latency  = cur_latency;
      prev_accepted = cur_accepted;

      ++total_phases;
   }

   if ( _sim_state == running ) {
      ++converged;

      if ( _sim_mode == latency ) {
         cout << "% Draining all recorded packets ..." << endl;
         _sim_state  = draining;
         _drain_time = _time;
         empty_steps = 0;
         while ( _PacketsOutstanding( ) ) {
            _Step( ); 
            ++empty_steps;

            if ( empty_steps % 1000 == 0 ) {
               _DisplayRemaining( ); 
            }
         }
      }
   } else {
      cout << "Too many sample periods needed to converge" << endl;
   }

   // Empty any remaining packets
   cout << "% Draining remaining packets ..." << endl;
   _empty_network = true;
   empty_steps = 0;
   while ( _total_in_flight > 0 ) {
      _Step( ); 
      ++empty_steps;

      if ( empty_steps % 1000 == 0 ) {
         _DisplayRemaining( ); 
      }
   }
   _empty_network = false;

   return( converged > 0 );
}

void TrafficManager::SetDrainState( )
{
   _sim_state  = draining;
   _drain_time = _time;

}

void TrafficManager::ShowOveralStat( )
{
   int c;

   for ( c = 0; c < _classes; ++c ) {
      cout << "=======Traffic["<<uid<<"]class" << c << " ======" << endl;

      cout << "Traffic["<<uid<<"]class" << c << "Overall average latency = " << _overall_latency[c]->Average( )
      << " (" << _overall_latency[c]->NumSamples( ) << " samples)" << endl;

      cout << "Traffic["<<uid<<"]class" << c << "Overall average accepted rate = " << _overall_accepted->Average( )
      << " (" << _overall_accepted->NumSamples( ) << " samples)" << endl;

      cout << "Traffic["<<uid<<"]class" << c << "Overall min accepted rate = " << _overall_accepted_min->Average( )
      << " (" << _overall_accepted_min->NumSamples( ) << " samples)" << endl;

      if ( DISPLAY_LAT_DIST ) {
         _latency_stats[c]->Display( );
      }
   }

   if ( _reorder ) {
      cout << "Traffic["<<uid<<"]class" << c << "Overall average reorder latency = " << _rob_latency->Average( ) << endl;
      cout << "Traffic["<<uid<<"]class" << c << "Overall average reorder size - " << _rob_size->Average( ) << endl;

      if ( DISPLAY_LAT_DIST ) {
         _rob_latency->Display( );
         _rob_size->Display( );
      }
   }

   if ( DISPLAY_HOP_DIST ) {
      cout << "Traffic["<<uid<<"]class" << c << "Average hops = " << _hop_stats->Average( )
      << " (" << _hop_stats->NumSamples( ) << " samples)" << endl;

      _hop_stats->Display( );
   }

   if ( DISPLAY_PAIR_LATENCY ) {
      for ( int i = 0; i < _dests; ++i ) {
         cout << "Traffic["<<uid<<"]class" << c << "  Average to " << i << " = " << _pair_latency[i]->Average( ) << "( " 
         << _pair_latency[i]->NumSamples( ) << " samples)" << endl;
         _pair_latency[i]->Display( );
      }
   }
}

void  TrafficManager::ShowStats() 
{
   double min, avg;

   static int  total_phases;

   double cur_latency;
   static double prev_latency;

   double cur_accepted;
   static double prev_accepted;

   //from step
   cout << "%=================================" << endl;

   cur_latency = _latency_stats[0]->Average( );
   int dmin = _ComputeAccepted( &avg, &min );
   cur_accepted = avg;

   cout << "% Average latency = " << cur_latency << endl;

   if ( _reorder ) {
      cout << "% Reorder latency = " << _rob_latency->Average( ) << endl;
      cout << "% Reorder size = " << _rob_size->Average( ) << endl;
   }

   cout << "% Accepted packets = " << min << " at node " << dmin << " (avg = " << avg << ")" << endl;

   if ( MATLAB_OUTPUT ) {
      cout << "lat(" << total_phases + 1 << ") = " << cur_latency << ";" << endl;
      cout << "thru(" << total_phases + 1 << ",:) = [ ";
      for ( int d = 0; d < _dests; ++d ) {
         cout << _accepted_packets[d]->Average( ) << " ";
      }
      cout << "];" << endl;
   }

   cout << "% latency change    = " << fabs( ( cur_latency - prev_latency ) / cur_latency ) << endl;
   cout << "% throughput change = " << fabs( ( cur_accepted - prev_accepted ) / cur_accepted ) << endl;

   prev_latency  = cur_latency;
   prev_accepted = cur_accepted;
   total_phases++;


   //from Run
   //save last Grid's stats
   for ( int c = 0; c < _classes; ++c ) {
      _overall_latency[c]->AddSample( _latency_stats[c]->Average( ) );
   }

   //_ComputeAccepted( &avg, &min );
   _overall_accepted->AddSample( avg );
   _overall_accepted_min->AddSample( min );
/* moved to interconnect_stats function in intreconnect_interface      
   cout << "%=================================" << endl;
   cout << "Link utilizations:" << endl;
   _net->Display();
*/
}
