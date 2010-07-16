#include "booksim.hpp"
#include "interconnect_interface.h"
#include <assert.h>

#include "network.hpp"

int gK = 0;
int gN = 0;
int gNodes = 0;

Network::Network( const Configuration &config ) :
Module( 0, "network" )
{
   _size     = -1; 
   _sources  = -1; 
   _dests    = -1;
   _channels = -1;
}

Network::~Network( )
{
   for ( int r = 0; r < _size; ++r ) {
      delete _routers[r];
   }

   delete [] _routers;

   delete [] _inject;
   delete [] _eject;
   delete [] _chan;

   delete [] _chan_use;

   delete [] _inject_cred;
   delete [] _eject_cred;
   delete [] _chan_cred;
}

void Network::_Alloc( )
{
   assert( ( _size != -1 ) && 
           ( _sources != -1 ) && 
           ( _dests != -1 ) && 
           ( _channels != -1 ) );

   _routers = new Router * [_size];
   gNodes = _sources;

   _inject = new Flit * [_sources];
   _eject  = new Flit * [_dests];
   _chan   = new Flit * [_channels];

   _chan_use = new int [_channels];

   for ( int i = 0; i < _channels; ++i ) {
      _chan_use[i] = 0;
   }

   _chan_use_cycles = 0;

   _inject_cred = new Credit * [_sources];
   _eject_cred  = new Credit * [_dests];
   _chan_cred   = new Credit * [_channels];
}

int Network::NumSources( ) const
{
   return _sources;
}

int Network::NumDests( ) const
{
   return _dests;
}

void Network::ReadInputs( )
{
   for ( int r = 0; r < _size; ++r ) {
      _routers[r]->ReadInputs( );
   }
}

void Network::InternalStep( )
{
   for ( int r = 0; r < _size; ++r ) {
      _routers[r]->InternalStep( );
   }
}

void Network::WriteOutputs( )
{
   for ( int r = 0; r < _size; ++r ) {
      _routers[r]->WriteOutputs( );
   }

   for ( int c = 0; c < _channels; ++c ) {
      if ( _chan[c] ) {
         _chan_use[c]++;
      }
   }
   _chan_use_cycles++;
}

void Network::WriteFlit( Flit *f, int source )
{
   assert( ( source >= 0 ) && ( source < _sources ) );
   _inject[source] = f;
}

Flit *Network::ReadFlit( int dest )
{
   assert( ( dest >= 0 ) && ( dest < _dests ) );
   return _eject[dest];
}

void Network::WriteCredit( Credit *c, int dest )
{
   assert( ( dest >= 0 ) && ( dest < _dests ) );
   _eject_cred[dest] = c;
}

Credit *Network::ReadCredit( int source )
{
   assert( ( source >= 0 ) && ( source < _sources ) );
   return _inject_cred[source];
}

void Network::InsertRandomFaults( const Configuration &config )
{
   Error( "InsertRandomFaults not implemented for this topology!" );
}

void Network::OutChannelFault( int r, int c, bool fault )
{
   assert( ( r >= 0 ) && ( r < _size ) );
   _routers[r]->OutChannelFault( c, fault );
}

double Network::Capacity( ) const
{
   return 1.0;
}

void Network::Display( ) const
{
   for ( int r = 0; r < _size; ++r ) {
      _routers[r]->Display( );
   }
//  if (icnt_config.GetInt("blah") ) {
   for ( int c = 0; c < _channels; ++c ) {
      cout << "channel " << c << " used " 
      << 100.0 * (double)_chan_use[c] / (double)_chan_use_cycles 
      << "% of the time" << endl;
   }
   // }
}
