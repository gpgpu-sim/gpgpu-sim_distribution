// $Id: tree4.cpp 5188 2012-08-30 00:31:31Z dub $

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
// Tree4: Network with 64 Terminal Nodes arranged in a tree topology
//        with 4 routers at the root of the tree
// 
//  Level 0 :  4  8 x 8 Routers   (8 Descending Links per Router)
//  Level 1 :  8  8 x 8 Routers   (4 Descending Links per Router)
//  Level 2 : 16  6 x 6 Routers   (4 Descending Links per Router)
//  Level 3 : 64  Terminal Nodes
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: tree4.cpp 5188 2012-08-30 00:31:31Z dub $
// 
////////////////////////////////////////////////////////////////////////

#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cmath>

#include "tree4.hpp"
#include "misc_utils.hpp"

Tree4::Tree4( const Configuration& config, const string & name )
: Network ( config, name )
{
  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
}

void Tree4::_ComputeSize( const Configuration& config )
{
  int h;

  _k = config.GetInt( "k" );
  assert(_k == 4);
  _n = config.GetInt( "n" );
  assert(_n == 3);
  
  gK = _k; gN = _n;
  
  _nodes = powi( _k, _n );
  
  _size = 0;
  for ( h = 0; h < _n; ++h ) 
    _size += (4 >> h) * powi( _k, h );

  _channels = 2                  // Two Channels per Connection
    * ( 2 * powi( _k, 1) )       // Number of Middle Routers
    * ( 2 * _k );                // Connectivity of Middle Routers
}

void Tree4::RegisterRoutingFunctions(){

}

void Tree4::_BuildNet( const Configuration& config )
{

  //
  // Allocate Routers
  //
  ostringstream name;
  int h, pos, nPos, degree, id;

  for ( h = 0; h < _n; h++ ) {
    nPos = (4 >> h) * powi( _k, h );
    for ( pos = 0; pos < nPos; ++pos) {
      if ( h < _n-1 ) 
	degree = 8;
      else
	degree = 6;
      
      name.str("");
      name << "router_" << h << "_" << pos;
      id = h * powi( _k, _n-1 ) + pos;
      Router * r = Router::NewRouter( config, this, name.str( ),
				      id, degree, degree );
      _Router( h, pos ) = r;
      _timed_modules.push_back(r);
    }
  }

  //
  // Connect Channels to Routers
  //
  int pp, pc;
  //
  // Connection Rule: Output Ports 0:3 Move DOWN Network
  //                  Output Ports 4:7 Move UP Network
  //
  
  // Injection & Ejection Channels
  nPos = powi( _k, _n - 1 );
  for ( pos = 0 ; pos < nPos ; ++pos ) {
    for ( int port = 0 ; port < _k ; ++port ) {
      
      _Router( _n-1, pos)->AddInputChannel( _inject[_k*pos+port],
					    _inject_cred[_k*pos+port]);
      

      _inject[_k*pos+port]->SetLatency( 1 );
      _inject_cred[_k*pos+port]->SetLatency( 1 );

      _Router( _n-1, pos)->AddOutputChannel( _eject[_k*pos+port],
					     _eject_cred[_k*pos+port]);

      _eject[_k*pos+port]->SetLatency( 1 );
      _eject_cred[_k*pos+port]->SetLatency( 1 );

    }
  }

  // Connections between h = 1 and h = 2 Levels
  int c = 0;
  nPos = 2 * powi( _k, 1 );
  for ( pos = 0; pos < nPos; ++pos ) {
    for ( int port = 0; port < _k; ++port ) {

      pp = pos;
      pc = _k * ( pos / 2 ) + port;
      
      // cout << "connecting (1,"<<pp<<") <-> (2,"<<pc<<")"<<endl;

      _Router( 1, pp)->AddOutputChannel( _chan[c], _chan_cred[c] );
      _Router( 2, pc)->AddInputChannel(  _chan[c], _chan_cred[c] );

      //_chan[c]->SetLatency( L );
      //_chan_cred[c]->SetLatency( L );

      _chan[c]->SetLatency( 1 );
      _chan_cred[c]->SetLatency( 1 );

      c++;

      _Router(1, pp)->AddInputChannel( _chan[c], _chan_cred[c] );
      _Router(2, pc)->AddOutputChannel( _chan[c], _chan_cred[c] );
      
      //_chan[c]->SetLatency( L );
      //_chan_cred[c]->SetLatency( L );
      _chan[c]->SetLatency( 1 );
      _chan_cred[c]->SetLatency( 1 );

      c++;
    }
  }

  // Connections between h = 0 and h = 1 Levels
  nPos = 4 * powi( _k, 0 );
  for ( pos  = 0; pos < nPos; ++pos ) {
    for ( int port = 0; port < 2 * _k; ++port ) {
      pp = pos;
      pc = port;

      // cout << "connecting (0,"<<pp<<") <-> (1,"<<pc<<")"<<endl;

      _Router(0, pp)->AddOutputChannel( _chan[c], _chan_cred[c] );
      _Router(1, pc)->AddInputChannel( _chan[c], _chan_cred[c] );

      //      _chan[c]->SetLatency( L );
      //_chan_cred[c]->SetLatency( L );
      _chan[c]->SetLatency( 1 );
      _chan_cred[c]->SetLatency( 1 );

      c++;

      _Router(0, pp)->AddInputChannel( _chan[c], _chan_cred[c] );
      _Router(1, pc)->AddOutputChannel( _chan[c], _chan_cred[c] );

      //  _chan[c]->SetLatency( L );
      // _chan_cred[c]->SetLatency( L );
      _chan[c]->SetLatency( 1 );
      _chan_cred[c]->SetLatency( 1 );
      c++;
    }
  }

  // cout << "Used " << c << " of " << _channels << " channels" << endl;

}
  
Router*& Tree4::_Router( int height, int pos )
{
  assert( height < _n );
  assert( pos < (4 >> height) * powi( _k, height) );

  int i = 0;
  for ( int h = 0; h < height; ++h )
    i += (4 >> h) * powi( _k, h );
  return _routers[i+pos];

}
  
int Tree4::_WireLatency( int height1, int pos1, int height2, int pos2 )
{
  int heightChild, heightParent, posChild, posParent;

  int L;

  if (height1 < height2) {
    heightChild  = height2;
    posChild     = pos2;
    heightParent = height1;
    posParent    = pos1;
  } else {
    heightChild  = height1;
    posChild     = pos1;
    heightParent = height2;
    posParent    = pos2;
  }

  int _length_d2_d1   = 2 ;
  int _length_d1_d0_0 = 2 ;
  int _length_d1_d0_1 = 2 ;
  int _length_d1_d0_2 = 6 ;
  int _length_d1_d0_3 = 6 ;

  assert( heightChild == heightParent+1 );

  // We must decrement the delays by one to account for how the 
  //  simulator interprets the specified delay (with 0 indicating one
  //  cycle of delay).

  if ( heightChild == 2 ) 
    L = _length_d2_d1;
  else {
       if ( posChild == 0 || posChild == 6 )
      switch ( posParent ) {
      case 0: L =_length_d1_d0_0; break;
      case 1: L =_length_d1_d0_1; break;
      case 2: L =_length_d1_d0_2; break;
      case 3: L =_length_d1_d0_3; break;
      }
    if ( posChild == 1 || posChild == 7 )
      switch ( posParent ) {
      case 0: L =_length_d1_d0_3; break;
      case 1: L =_length_d1_d0_2; break;
      case 2: L =_length_d1_d0_1; break;
      case 3: L =_length_d1_d0_0; break;
      }
    if ( posChild == 2 || posChild == 4 )
      switch ( posParent ) {
      case 0: L = _length_d1_d0_0; break;
      case 1: L = _length_d1_d0_1; break;
      case 2: L = _length_d1_d0_2; break;
      case 3: L = _length_d1_d0_3; break;
      }
    if ( posChild == 3|| posChild == 5 )
      switch ( posParent ) {
      case 0: L =_length_d1_d0_3; break;
      case 1: L =_length_d1_d0_2; break;
      case 2: L =_length_d1_d0_1; break;
      case 3: L =_length_d1_d0_0; break;
      }
  }
  return L;
}
