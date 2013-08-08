// $Id: fattree.cpp 5188 2012-08-30 00:31:31Z dub $

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
// FatTree
//
//       Each level of the hierarchical indirect Network has
//       k^(n-1) Routers. The Routers are organized such that 
//       each node has k descendents, and each parent is
//       replicated k  times.
//      most routers has 2K ports, excep the top level has only K
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:50:48 $
//  $Id: fattree.cpp 5188 2012-08-30 00:31:31Z dub $
// 
////////////////////////////////////////////////////////////////////////


#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cmath>

#include "fattree.hpp"
#include "misc_utils.hpp"


 //#define FATTREE_DEBUG

FatTree::FatTree( const Configuration& config,const string & name )
  : Network( config ,name)
{
  

  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );

}

void FatTree::_ComputeSize( const Configuration& config )
{

  _k = config.GetInt( "k" );
  _n = config.GetInt( "n" );
   
  gK = _k; gN = _n;
  
  _nodes = powi( _k, _n );

  //levels * routers_per_level
  _size = _n * powi( _k , _n - 1 );

  //(channels per level = k*routers_per_level* up/down) * (levels-1)
  _channels = (2*_k * powi( _k , _n-1 ))*(_n-1); 
  

}


void FatTree::RegisterRoutingFunctions() {

}

void FatTree::_BuildNet( const Configuration& config )
{
 cout << "Fat Tree" << endl;
  cout << " k = " << _k << " levels = " << _n << endl;
  cout << " each switch - total radix =  "<< 2*_k << endl;
  cout << " # of switches = "<<  _size << endl;
  cout << " # of channels = "<<  _channels << endl;
  cout << " # of nodes ( size of network ) = " << _nodes << endl;


  // Number of router positions at each depth of the network
  const int nPos = powi( _k, _n-1);

  //
  // Allocate Routers
  //
  ostringstream name;
  int level, pos, id, degree, port;
  for ( level = 0 ; level < _n ; ++level ) {
    for ( pos = 0 ; pos < nPos ; ++pos ) {
      
      if ( level == 0 ) //top routers is zero
	degree = _k;
      else
	degree = 2 * _k;

      id = level * nPos + pos;

      name.str("");
      name << "router_level" << level << "_" << pos;
      Router * r = Router::NewRouter( config, this, name.str( ), id,
				      degree, degree );
      _Router( level, pos ) = r;
      _timed_modules.push_back(r);
    }
  }

  //
  // Connect Channels to Routers
  //

  //
  // Router Connection Rule: Output Ports <gK Move DOWN Network
  //                         Output Ports >=gK Move UP Network
  //                         Input Ports <gK from DOWN Network
  //                         Input Ports >=gK  from up Network

  // Connecting  Injection & Ejection Channels  
  for ( pos = 0 ; pos < nPos ; ++pos ) {
    for(int index = 0; index<_k; index++){
      int link = pos*_k + index;
      _Router( _n-1, pos)->AddInputChannel( _inject[link],
					    _inject_cred[link]);
      _Router( _n-1, pos)->AddOutputChannel( _eject[link],
					     _eject_cred[link]);
      _inject[link]->SetLatency( 1 );
      _inject_cred[link]->SetLatency( 1 );
      _eject[link]->SetLatency( 1 );
      _eject_cred[link]->SetLatency( 1 );
    }
  }

#ifdef FATTREE_DEBUG
  cout<<"\nAssigning output\n";
#endif

  //channels are numbered sequentially from an output channel perspective
  int chan_per_direction = (_k * powi( _k , _n-1 )); //up or down
  int chan_per_level = 2*(_k * powi( _k , _n-1 )); //up+down

  //connect all down output channels
  //level n-1's down channel are injection channels
  for (level = 0; level<_n-1; level++){
    for ( pos = 0; pos < nPos; ++pos ) {
      for ( port = 0; port < _k; ++port ) {
	int link = (level*chan_per_level) + pos*_k + port;
	_Router(level, pos)->AddOutputChannel( _chan[link],
						_chan_cred[link] );
	_chan[link]->SetLatency( 1 );
	_chan_cred[link]->SetLatency( 1 ); 
#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"down output "<<port<<" "
	    <<"channel_id "<<link<<endl;
#endif

      }
    }
  }
  //connect all up output channels
  //level 0 has no up chnanels
  for (level = 1; level<_n; level++){
    for ( pos = 0; pos < nPos; ++pos ) {
      for ( port = 0; port < _k; ++port ) {
	int link = (level*chan_per_level - chan_per_direction) + pos*_k + port ;
	_Router(level, pos)->AddOutputChannel( _chan[link],
						_chan_cred[link] );
	_chan[link]->SetLatency( 1 );
	_chan_cred[link]->SetLatency( 1 ); 
#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"up output "<<port<<" "
	    <<"channel_id "<<link<<endl;
#endif
      }
    }
  }

#ifdef FATTREE_DEBUG
  cout<<"\nAssigning Input\n";
#endif

  //connect all down input channels
  for (level = 0; level<_n-1; level++){
    //input channel are numbered interleavely, the interleaev depends on level
    int routers_per_neighborhood = powi(_k,_n-1-(level)); 
    int routers_per_branch = powi(_k,_n-1-(level+1)); 
    int level_offset = routers_per_neighborhood*_k;
    for ( pos = 0; pos < nPos; ++pos ) {
      int neighborhood = pos/routers_per_neighborhood;
      int neighborhood_pos = pos%routers_per_neighborhood;
      for ( port = 0; port < _k; ++port ) {
	int link = 
	  ((level+1)*chan_per_level - chan_per_direction)  //which levellevel
	  +neighborhood*level_offset   //region in level
	  +port*routers_per_branch*gK  //sub region in region
	  +(neighborhood_pos)%routers_per_branch*gK  //router in subregion
	  +(neighborhood_pos)/routers_per_branch; //port on router

	_Router(level, pos)->AddInputChannel( _chan[link],
					      _chan_cred[link] );
#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"down input "<<port<<" "
	    <<"channel_id "<<link<<endl;
#endif
      }
    }
  }


 //connect all up input channels
  for (level = 1; level<_n; level++){
    //input channel are numbered interleavely, the interleaev depends on level
    int routers_per_neighborhood = powi(_k,_n-1-(level-1)); 
    int routers_per_branch = powi(_k,_n-1-(level)); 
    int level_offset = routers_per_neighborhood*_k;
    for ( pos = 0; pos < nPos; ++pos ) {
      int neighborhood = pos/routers_per_neighborhood;
      int neighborhood_pos = pos%routers_per_neighborhood;
      for ( port = 0; port < _k; ++port ) {
	int link = 
	  ((level-1)*chan_per_level) //which levellevel
	  +neighborhood*level_offset   //region in level
	  +port*routers_per_branch*gK  //sub region in region
	  +(neighborhood_pos)%routers_per_branch*gK //router in subregion
	  +(neighborhood_pos)/routers_per_branch; //port on router

	_Router(level, pos)->AddInputChannel( _chan[link],
					      _chan_cred[link] );
#ifdef FATTREE_DEBUG
	cout<<_Router(level, pos)->Name()<<" "
	    <<"up input "<<port<<" "
	    <<"channel_id "<<link<<endl;
#endif
      }
    }
  }
#ifdef FATTREE_DEBUG
  cout<<"\nChannel assigned\n";
#endif
}

Router*& FatTree::_Router( int depth, int pos ) 
{
  assert( depth < _n && pos < powi( _k, _n-1) );
  return _routers[depth * powi( _k, _n-1) + pos];
}
