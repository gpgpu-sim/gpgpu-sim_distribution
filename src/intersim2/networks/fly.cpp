// $Id: fly.cpp 5188 2012-08-30 00:31:31Z dub $

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
#include <vector>
#include <sstream>

#include "fly.hpp"
#include "misc_utils.hpp"

//#define DEBUG_FLY

KNFly::KNFly( const Configuration &config, const string & name ) :
Network( config, name )
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

  _nodes = powi( _k, _n );

  // n stages of k^(n-1) k x k switches
  _size     = _n*powi( _k, _n-1 );

  // n-1 sets of wiring between the stages
  _channels = (_n-1)*_nodes;
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
      _timed_modules.push_back(_routers[node]);
      router_name.str("");

#ifdef DEBUG_FLY
      cout << "connecting node " << node << " to:" << endl;
#endif 

      for ( int port = 0; port < _k; ++port ) {
	// Input connections
	if ( stage == 0 ) {
	  c = addr*_k + port;
	  _routers[node]->AddInputChannel( _inject[c], _inject_cred[c] );
#ifdef DEBUG_FLY	  
	  cout << "  injection channel " << c << endl;
#endif 
	} else {
	  c = _InChannel( stage, addr, port );
	  _routers[node]->AddInputChannel( _chan[c], _chan_cred[c] );
	  _chan[c]->SetLatency( 1 );

#ifdef DEBUG_FLY
	  cout << "  input channel " << c << endl;
#endif 
	}

	// Output connections
	if ( stage == _n - 1 ) {
	  c = addr*_k + port;
	  _routers[node]->AddOutputChannel( _eject[c], _eject_cred[c] );
#ifdef DEBUG_FLY
	  cout << "  ejection channel " << c << endl;
#endif 
	} else {
	  c = _OutChannel( stage, addr, port );
	  _routers[node]->AddOutputChannel( _chan[c], _chan_cred[c] );
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
  return stage*_nodes + addr*_k + port;
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

  return (stage-1)*_nodes + in_addr*_k + in_port;
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

