#include "booksim.hpp"
#include <vector>
#include <sstream>

#include "fly.hpp"
#include "misc_utils.hpp"

//#define DEBUG_FLY

KNFly::KNFly( const Configuration &config ) :
Network( config )
{
   _ComputeSize( config );
   _Alloc( );
   _BuildNet( config );
}

void KNFly::_ComputeSize( const Configuration &config )
{
   _k = config.GetInt( "k" );
   _n = config.GetInt( "n" );

   gK = _k; gN = _n;

   _sources = powi( _k, _n );
   _dests   = powi( _k, _n );

   // n stages of k^(n-1) k x k switches
   _size     = _n*powi( _k, _n-1 );

   // n-1 sets of wiring between the stages
   _channels = (_n-1)*_sources;
}

void KNFly::_BuildNet( const Configuration &config )
{
   ostringstream router_name;

   int per_stage = powi( _k, _n-1 );

   int node = 0;
   int c;

   for ( int stage = 0; stage < _n; ++stage ) {
      for ( int addr = 0; addr < per_stage; ++addr ) {

         router_name << "router_" << stage << "_" << addr;
         _routers[node] = Router::NewRouter( config, this, router_name.str( ), 
                                             node, _k, _k );
         router_name.seekp( 0, ios::beg );

#ifdef DEBUG_FLY
         cout << "connecting node " << node << " to:" << endl;
#endif 

         for ( int port = 0; port < _k; ++port ) {
            // Input connections
            if ( stage == 0 ) {
               c = addr*_k + port;
               _routers[node]->AddInputChannel( &_inject[c], &_inject_cred[c] );
#ifdef DEBUG_FLY	  
               cout << "  injection channel " << c << endl;
#endif 
            } else {
               c = _InChannel( stage, addr, port );
               _routers[node]->AddInputChannel( &_chan[c], &_chan_cred[c] );
#ifdef DEBUG_FLY
               cout << "  input channel " << c << endl;
#endif 
            }

            // Output connections
            if ( stage == _n - 1 ) {
               c = addr*_k + port;
               _routers[node]->AddOutputChannel( &_eject[c], &_eject_cred[c] );
#ifdef DEBUG_FLY
               cout << "  ejection channel " << c << endl;
#endif 
            } else {
               c = _OutChannel( stage, addr, port );
               _routers[node]->AddOutputChannel( &_chan[c], &_chan_cred[c] );
#ifdef DEBUG_FLY
               cout << "  output channel " << c << endl;
#endif 
            }
         }

         ++node;
      }
   }
}

int KNFly::_OutChannel( int stage, int addr, int port ) const
{
   return stage*_sources + addr*_k + port;
}

int KNFly::_InChannel( int stage, int addr, int port ) const
{
   int in_addr;
   int in_port;

   // Channels are between {node,port}
   //   { d_{n-1} ... d_{n-stage} ... d_0 } and
   //   { d_{n-1} ... d_0         ... d_{n-stage} }

   int shift = powi( _k, _n-stage-1 );

   int last_digit = port;
   int zero_digit = ( addr / shift ) % _k;

   // swap zero and last digit to get first node's address
   in_addr = addr - zero_digit*shift + last_digit*shift;
   in_port = zero_digit;

   return(stage-1)*_sources + in_addr*_k + in_port;
}

int KNFly::GetN( ) const
{
   return _n;
}

int KNFly::GetK( ) const
{
   return _k;
}

double KNFly::Capacity( ) const
{
   return 1.0;
}

