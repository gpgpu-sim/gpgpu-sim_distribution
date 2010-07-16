#include "booksim.hpp"
#include <vector>

#include "singlenet.hpp"

SingleNet::SingleNet( const Configuration &config ) :
Network( config )
{
   _ComputeSize( config );
   _Alloc( );
   _BuildNet( config );
}

void SingleNet::_ComputeSize( const Configuration &config )
{
   _sources = config.GetInt( "in_ports" );
   _dests   = config.GetInt( "out_ports" );

   _size     = 1;
   _channels = 0;
}

void SingleNet::_BuildNet( const Configuration &config )
{
   int i;

   _routers[0] = Router::NewRouter( config, this, "router", 0, 
                                    _sources, _dests );

   for ( i = 0; i < _sources; ++i ) {
      _routers[0]->AddInputChannel( &_inject[i], &_inject_cred[i] );
   }

   for ( i = 0; i < _dests; ++i ) {
      _routers[0]->AddOutputChannel( &_eject[i], &_eject_cred[i] );
   }
}

void SingleNet::Display( ) const
{
   _routers[0]->Display( );
}

