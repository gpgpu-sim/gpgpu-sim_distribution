// $Id: qtree.cpp 5188 2012-08-30 00:31:31Z dub $

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

////////////////////////////////////////////////////////////////////////
//
// QTree: A Quad-Tree Indirect Network.
//
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/05/17 17:14:07 $
//  $Id: qtree.cpp 5188 2012-08-30 00:31:31Z dub $
// 
////////////////////////////////////////////////////////////////////////

#include "booksim.hpp"
#include <vector>
#include <sstream>
#include "qtree.hpp"
#include "misc_utils.hpp"

QTree::QTree( const Configuration& config, const string & name )
: Network ( config, name )
{
  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
}


void QTree::_ComputeSize( const Configuration& config )
{

  _k = config.GetInt( "k" );
  _n = config.GetInt( "n" );

  assert( _k == 4 && _n == 3 );

  gK = _k; gN = _n;

  _nodes = powi( _k, _n );

  _size = 0;
  for (int i = 0; i < _n; i++)
    _size += powi( _k, i );

  _channels = 0;
  for (int j = 1; j < _n; j++)
    _channels += 2 * powi( _k, j );

}

void QTree::RegisterRoutingFunctions(){

}

void QTree::_BuildNet( const Configuration& config )
{

  ostringstream routerName;
  int h, r, pos, port;

  for (h = 0; h < _n; h++) {
    for (pos = 0 ; pos < powi( _k, h ) ; ++pos ) {
      
      int id = h * 256 + pos;  
      r = _RouterIndex( h, pos );

      routerName << "router_" << h << "_" << pos;

      int d = ( h == 0 ) ? _k : _k + 1;
      _routers[r] = Router::NewRouter( config, this,
				       routerName.str( ),
				       id, d, d);
      _timed_modules.push_back(_routers[r]);
      routerName.str("");
    }
  }
  
  // Injection & Ejection Channels
  for ( pos = 0 ; pos < powi( _k, _n-1 ) ; ++pos ) {
    r = _RouterIndex( _n-1, pos );
    for ( port = 0 ; port < _k ; port++ ) {

      _routers[r]->AddInputChannel( _inject[_k*pos+port],
				    _inject_cred[_k*pos+port]);

      _routers[r]->AddOutputChannel( _eject[_k*pos+port],
				     _eject_cred[_k*pos+port]);
    }
  }

  int c;
  for ( h = 0 ; h < _n ; ++h ) {
    for ( pos = 0 ; pos < powi( _k, h ) ; ++pos ) {
      for ( port = 0 ; port < _k ; port++ ) {

	r = _RouterIndex( h, pos );

	if ( h < _n-1 ) {
	  // Channels to Children Nodes
	  c = _InputIndex( h , pos, port );
	  _routers[r]->AddInputChannel( _chan[c], 
					_chan_cred[c] );

	  c = _OutputIndex( h, pos, port );
	  _routers[r]->AddOutputChannel( _chan[c], 
					 _chan_cred[c] );

	}
      }
      if ( h > 0 ) {
	// Channels to Parent Nodes
	c = _OutputIndex( h - 1, pos / _k, pos % _k );
	_routers[r]->AddInputChannel( _chan[c],
				      _chan_cred[c] );

	c = _InputIndex( h - 1, pos / _k, pos % _k );
	_routers[r]->AddOutputChannel( _chan[c],
				       _chan_cred[c]);
      }
    }
  }
}
 
int QTree::_RouterIndex( int height, int pos ) 
{
  int r = 0;
  for ( int h = 0; h < height; h++ ) 
    r += powi( _k, h );
  return (r + pos);
}

int QTree::_InputIndex( int height, int pos, int port )
{
  assert( height >= 0 && height < powi( _k,_n-1 ) );
  int c = 0;
  for ( int h = 0; h < height; h++) 
    c += powi( _k, h+1 );
  return ( c + _k * pos + port );
}

int QTree::_OutputIndex( int height, int pos, int port )
{
  assert( height >= 0 && height < powi( _k,_n-1 ) );
  int c = _channels / 2;
  for ( int h = 0; h < height; h++) 
    c += powi( _k, h+1 );
  return ( c + _k * pos + port );
}


int QTree::HeightFromID( int id ) 
{
  return id / 256;
}

int QTree::PosFromID( int id )
{
  return id % 256;
}
