// $Id: kncube.cpp 5516 2013-10-06 02:14:48Z dub $

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

/*kn.cpp
 *
 *Meshs, cube, torus
 *
 */

#include "booksim.hpp"
#include <vector>
#include <sstream>
#include "kncube.hpp"
#include "random_utils.hpp"
#include "misc_utils.hpp"
 //#include "iq_router.hpp"


KNCube::KNCube( const Configuration &config, const string & name, bool mesh ) :
Network( config, name )
{
  _mesh = mesh;

  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
}

void KNCube::_ComputeSize( const Configuration &config )
{
  _k = config.GetInt( "k" );
  _n = config.GetInt( "n" );

  gK = _k; gN = _n;
  _size     = powi( _k, _n );
  _channels = 2*_n*_size;

  _nodes = _size;
}

void KNCube::RegisterRoutingFunctions() {

}
void KNCube::_BuildNet( const Configuration &config )
{
  int left_node;
  int right_node;

  int right_input;
  int left_input;

  int right_output;
  int left_output;

  ostringstream router_name;

  //latency type, noc or conventional network
  bool use_noc_latency;
  use_noc_latency = (config.GetInt("use_noc_latency")==1);
  
  for ( int node = 0; node < _size; ++node ) {

    router_name << "router";
    
    if ( _k > 1 ) {
      for ( int dim_offset = _size / _k; dim_offset >= 1; dim_offset /= _k ) {
	router_name << "_" << ( node / dim_offset ) % _k;
      }
    }

    _routers[node] = Router::NewRouter( config, this, router_name.str( ), 
					node, 2*_n + 1, 2*_n + 1 );
    _timed_modules.push_back(_routers[node]);

    router_name.str("");

    for ( int dim = 0; dim < _n; ++dim ) {

      //find the neighbor 
      left_node  = _LeftNode( node, dim );
      right_node = _RightNode( node, dim );

      //
      // Current (N)ode
      // (L)eft node
      // (R)ight node
      //
      //   L--->N<---R
      //   L<---N--->R
      //

      // torus channel is longer due to wrap around
      int latency = _mesh ? 1 : 2 ;

      //get the input channel number
      right_input = _LeftChannel( right_node, dim );
      left_input  = _RightChannel( left_node, dim );

      //add the input channel
      _routers[node]->AddInputChannel( _chan[right_input], _chan_cred[right_input] );
      _routers[node]->AddInputChannel( _chan[left_input], _chan_cred[left_input] );

      //set input channel latency
      if(use_noc_latency){
	_chan[right_input]->SetLatency( latency );
	_chan[left_input]->SetLatency( latency );
	_chan_cred[right_input]->SetLatency( latency );
	_chan_cred[left_input]->SetLatency( latency );
      } else {
	_chan[left_input]->SetLatency( 1 );
	_chan_cred[right_input]->SetLatency( 1 );
	_chan_cred[left_input]->SetLatency( 1 );
	_chan[right_input]->SetLatency( 1 );
      }
      //get the output channel number
      right_output = _RightChannel( node, dim );
      left_output  = _LeftChannel( node, dim );
      
      //add the output channel
      _routers[node]->AddOutputChannel( _chan[right_output], _chan_cred[right_output] );
      _routers[node]->AddOutputChannel( _chan[left_output], _chan_cred[left_output] );

      //set output channel latency
      if(use_noc_latency){
	_chan[right_output]->SetLatency( latency );
	_chan[left_output]->SetLatency( latency );
	_chan_cred[right_output]->SetLatency( latency );
	_chan_cred[left_output]->SetLatency( latency );
      } else {
	_chan[right_output]->SetLatency( 1 );
	_chan[left_output]->SetLatency( 1 );
	_chan_cred[right_output]->SetLatency( 1 );
	_chan_cred[left_output]->SetLatency( 1 );

      }
    }
    //injection and ejection channel, always 1 latency
    _routers[node]->AddInputChannel( _inject[node], _inject_cred[node] );
    _routers[node]->AddOutputChannel( _eject[node], _eject_cred[node] );
    _inject[node]->SetLatency( 1 );
    _eject[node]->SetLatency( 1 );
  }
}

int KNCube::_LeftChannel( int node, int dim )
{
  // The base channel for a node is 2*_n*node
  int base = 2*_n*node;
  // The offset for a left channel is 2*dim + 1
  int off  = 2*dim + 1;

  return ( base + off );
}

int KNCube::_RightChannel( int node, int dim )
{
  // The base channel for a node is 2*_n*node
  int base = 2*_n*node;
  // The offset for a right channel is 2*dim 
  int off  = 2*dim;
  return ( base + off );
}

int KNCube::_LeftNode( int node, int dim )
{
  int k_to_dim = powi( _k, dim );
  int loc_in_dim = ( node / k_to_dim ) % _k;
  int left_node;
  // if at the left edge of the dimension, wraparound
  if ( loc_in_dim == 0 ) {
    left_node = node + (_k-1)*k_to_dim;
  } else {
    left_node = node - k_to_dim;
  }

  return left_node;
}

int KNCube::_RightNode( int node, int dim )
{
  int k_to_dim = powi( _k, dim );
  int loc_in_dim = ( node / k_to_dim ) % _k;
  int right_node;
  // if at the right edge of the dimension, wraparound
  if ( loc_in_dim == ( _k-1 ) ) {
    right_node = node - (_k-1)*k_to_dim;
  } else {
    right_node = node + k_to_dim;
  }

  return right_node;
}

int KNCube::GetN( ) const
{
  return _n;
}

int KNCube::GetK( ) const
{
  return _k;
}

/*legacy, not sure how this fits into the own scheme of things*/
void KNCube::InsertRandomFaults( const Configuration &config )
{
  int num_fails;
  unsigned long prev_seed;

  int node, chan;
  int i, j, t, n, c;
  bool available;

  bool edge;

  num_fails = config.GetInt( "link_failures" );
  
  if ( _size && num_fails ) {
    prev_seed = RandomIntLong( );
    RandomSeed( config.GetInt( "fail_seed" ) );

    vector<bool> fail_nodes(_size);

    for ( i = 0; i < _size; ++i ) {
      node = i;

      // edge test
      edge = false;
      for ( n = 0; n < _n; ++n ) {
	if ( ( ( node % _k ) == 0 ) ||
	     ( ( node % _k ) == _k - 1 ) ) {
	  edge = true;
	}
	node /= _k;
      }

      if ( edge ) {
	fail_nodes[i] = true;
      } else {
	fail_nodes[i] = false;
      }
    }

    for ( i = 0; i < num_fails; ++i ) {
      j = RandomInt( _size - 1 );
      available = false;

      for ( t = 0; ( t < _size ) && (!available); ++t ) {
	node = ( j + t ) % _size;
       
	if ( !fail_nodes[node] ) {
	  // check neighbors
	  c = RandomInt( 2*_n - 1 );

	  for ( n = 0; ( n < 2*_n ) && (!available); ++n ) {
	    chan = ( n + c ) % 2*_n;

	    if ( chan % 1 ) {
	      available = fail_nodes[_LeftNode( node, chan/2 )];
	    } else {
	      available = fail_nodes[_RightNode( node, chan/2 )];
	    }
	  }
	}
	
	if ( !available ) {
	  cout << "skipping " << node << endl;
	}
      }

      if ( t == _size ) {
	Error( "Could not find another possible fault channel" );
      }

      
      OutChannelFault( node, chan );
      fail_nodes[node] = true;

      for ( n = 0; ( n < _n ) && available ; ++n ) {
	fail_nodes[_LeftNode( node, n )]  = true;
	fail_nodes[_RightNode( node, n )] = true;
      }

      cout << "failure at node " << node << ", channel " 
	   << chan << endl;
    }

    RandomSeed( prev_seed );
  }
}

double KNCube::Capacity( ) const
{
  return (double)_k / ( _mesh ? 8.0 : 4.0 );
}
