// $Id: cmesh.cpp 5188 2012-08-30 00:31:31Z dub $

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

// ----------------------------------------------------------------------
//
// CMesh: Network with <Int> Terminal Nodes arranged in a concentrated
//        mesh topology
//
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
//  $Author: jbalfour $
//  $Date: 2007/06/28 17:24:35 $
//  $Id: cmesh.cpp 5188 2012-08-30 00:31:31Z dub $
//  Modified 11/6/2007 by Ted Jiang
//  Now handeling n = most power of 2: 16, 64, 256, 1024
// ----------------------------------------------------------------------
#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cassert>
#include "random_utils.hpp"
#include "misc_utils.hpp"
#include "cmesh.hpp"

int CMesh::_cX = 0 ;
int CMesh::_cY = 0 ;
int CMesh::_memo_NodeShiftX = 0 ;
int CMesh::_memo_NodeShiftY = 0 ;
int CMesh::_memo_PortShiftY = 0 ;

CMesh::CMesh( const Configuration& config, const string & name ) 
  : Network(config, name) 
{
  _ComputeSize( config );
  _Alloc();
  _BuildNet(config);
}

void CMesh::RegisterRoutingFunctions() {
  gRoutingFunctionMap["dor_cmesh"] = &dor_cmesh;
  gRoutingFunctionMap["dor_no_express_cmesh"] = &dor_no_express_cmesh;
  gRoutingFunctionMap["xy_yx_cmesh"] = &xy_yx_cmesh;
  gRoutingFunctionMap["xy_yx_no_express_cmesh"]  = &xy_yx_no_express_cmesh;
}

void CMesh::_ComputeSize( const Configuration &config ) {

  int k = config.GetInt( "k" );
  int n = config.GetInt( "n" );
  assert(n <= 2); // broken for n > 2
  int c = config.GetInt( "c" );
  assert(c == 4); // broken for c != 4

  ostringstream router_name;
  //how many routers in the x or y direction
  _xcount = config.GetInt("x");
  _ycount = config.GetInt("y");
  assert(_xcount == _ycount); // broken for asymmetric topologies
  //configuration of hohw many clients in X and Y per router
  _xrouter = config.GetInt("xr");
  _yrouter = config.GetInt("yr");
  assert(_xrouter == _yrouter); // broken for asymmetric concentration

  gK = _k = k ;
  gN = _n = n ;
  gC = _c = c ;

  assert(c == _xrouter*_yrouter);
  
  _nodes    = _c * powi( _k, _n); // Number of nodes in network
  _size     = powi( _k, _n);      // Number of routers in network
  _channels = 2 * _n * _size;     // Number of channels in network

  _cX = _c / _n ;   // Concentration in X Dimension 
  _cY = _c / _cX ;  // Concentration in Y Dimension

  //
  _memo_NodeShiftX = _cX >> 1 ;
  _memo_NodeShiftY = log_two(gK * _cX) + ( _cY >> 1 ) ;
  _memo_PortShiftY = log_two(gK * _cX)  ;

}

void CMesh::_BuildNet( const Configuration& config ) {

  int x_index ;
  int y_index ;

  //standard trace configuration 
  if(gTrace){
    cout<<"Setup Finished Router"<<endl;
  }

  //latency type, noc or conventional network
  bool use_noc_latency;
  use_noc_latency = (config.GetInt("use_noc_latency")==1);
  
  ostringstream name;
  // The following vector is used to check that every
  //  processor in the system is connected to the network
  vector<bool> channel_vector(_nodes, false) ;
  
  //
  // Routers and Channel
  //
  for (int node = 0; node < _size; ++node) {

    // Router index derived from mesh index
    y_index = node / _k ;
    x_index = node % _k ;

    const int degree_in  = 2 *_n + _c ;
    const int degree_out = 2 *_n + _c ;

    name << "router_" << y_index << '_' << x_index;
    _routers[node] = Router::NewRouter( config, 
					this, 
					name.str(), 
					node,
					degree_in,
					degree_out);
    _timed_modules.push_back(_routers[node]);
    name.str("");

    //
    // Port Numbering: as best as I can determine, the order in
    //  which the input and output channels are added to the
    //  router determines the associated port number that must be
    //  used by the router. Output port number increases with 
    //  each new channel
    //

    //
    // Processing node channels
    //
    for (int y = 0; y < _cY ; y++) {
      for (int x = 0; x < _cX ; x++) {
	int link = (_k * _cX) * (_cY * y_index + y) + (_cX * x_index + x) ;
	assert( link >= 0 ) ;
	assert( link < _nodes ) ;
	assert( channel_vector[ link ] == false ) ;
	channel_vector[link] = true ;
	// Ingress Ports
	_routers[node]->AddInputChannel(_inject[link], _inject_cred[link]);
	// Egress Ports
	_routers[node]->AddOutputChannel(_eject[link], _eject_cred[link]);
	//injeciton ejection latency is 1
	_inject[link]->SetLatency( 1 );
	_eject[link]->SetLatency( 1 );
      }
    }

    //
    // router to router channels
    //
    const int x = node % _k ;
    const int y = node / _k ;
    const int offset = powi( _k, _n ) ;

    //the channel number of the input output channels.
    int px_out = _k * y + x + 0 * offset ;
    int nx_out = _k * y + x + 1 * offset ;
    int py_out = _k * y + x + 2 * offset ;
    int ny_out = _k * y + x + 3 * offset ;
    int px_in  = _k * y + ((x+1)) + 1 * offset ;
    int nx_in  = _k * y + ((x-1)) + 0 * offset ;
    int py_in  = _k * ((y+1)) + x + 3 * offset ;
    int ny_in  = _k * ((y-1)) + x + 2 * offset ;

    // Express Channels
    if (x == 0){
      // Router on left edge of mesh. Connect to -x output of
      //  another router on the left edge of the mesh.
      if (y < _k / 2 )
	nx_in = _k * (y + _k/2) + x + offset ;
      else
	nx_in = _k * (y - _k/2) + x + offset ;
    }
    
    if (x == (_k-1)){
      // Router on right edge of mesh. Connect to +x output of 
      //  another router on the right edge of the mesh.
      if (y < _k / 2)
   	px_in = _k * (y + _k/2) + x ;
      else
   	px_in = _k * (y - _k/2) + x ;
    }
    
    if (y == 0) {
      // Router on bottom edge of mesh. Connect to -y output of
      //  another router on the bottom edge of the mesh.
      if (x < _k / 2) 
	ny_in = _k * y + (x + _k/2) + 3 * offset ;
      else
	ny_in = _k * y + (x - _k/2) + 3 * offset ;
    }
    
    if (y == (_k-1)) {
      // Router on top edge of mesh. Connect to +y output of
      //  another router on the top edge of the mesh
      if (x < _k / 2)
	py_in = _k * y + (x + _k/2) + 2 * offset ;
      else
	py_in = _k * y + (x - _k/2) + 2 * offset ;
    }

    /*set latency and add the channels*/

    // Port 0: +x channel
    if(use_noc_latency) {
      int const px_latency = (x == _k-1) ? (_cY*_k/2) : _cX;
      _chan[px_out]->SetLatency( px_latency );
      _chan_cred[px_out]->SetLatency( px_latency );
    } else {
      _chan[px_out]->SetLatency( 1 );
      _chan_cred[px_out]->SetLatency( 1 );
    }
    _routers[node]->AddOutputChannel( _chan[px_out], _chan_cred[px_out] );
    _routers[node]->AddInputChannel( _chan[px_in], _chan_cred[px_in] );
    
    if(gTrace) {
      cout<<"Link "<<" "<<px_out<<" "<<px_in<<" "<<node<<" "<<_chan[px_out]->GetLatency()<<endl;
    }

    // Port 1: -x channel
    if(use_noc_latency) {
      int const nx_latency = (x == 0) ? (_cY*_k/2) : _cX;
      _chan[nx_out]->SetLatency( nx_latency );
      _chan_cred[nx_out]->SetLatency( nx_latency );
    } else {
      _chan[nx_out]->SetLatency( 1 );
      _chan_cred[nx_out]->SetLatency( 1 );
    }
    _routers[node]->AddOutputChannel( _chan[nx_out], _chan_cred[nx_out] );
    _routers[node]->AddInputChannel( _chan[nx_in], _chan_cred[nx_in] );

    if(gTrace){
      cout<<"Link "<<" "<<nx_out<<" "<<nx_in<<" "<<node<<" "<<_chan[nx_out]->GetLatency()<<endl;
    }

    // Port 2: +y channel
    if(use_noc_latency) {
      int const py_latency = (y == _k-1) ? (_cX*_k/2) : _cY;
      _chan[py_out]->SetLatency( py_latency );
      _chan_cred[py_out]->SetLatency( py_latency );
    } else {
      _chan[py_out]->SetLatency( 1 );
      _chan_cred[py_out]->SetLatency( 1 );
    }
    _routers[node]->AddOutputChannel( _chan[py_out], _chan_cred[py_out] );
    _routers[node]->AddInputChannel( _chan[py_in], _chan_cred[py_in] );
    
    if(gTrace){
      cout<<"Link "<<" "<<py_out<<" "<<py_in<<" "<<node<<" "<<_chan[py_out]->GetLatency()<<endl;
    }

    // Port 3: -y channel
    if(use_noc_latency){
      int const ny_latency = (y == 0) ? (_cX*_k/2) : _cY;
      _chan[ny_out]->SetLatency( ny_latency );
      _chan_cred[ny_out]->SetLatency( ny_latency );
    } else {
      _chan[ny_out]->SetLatency( 1 );
      _chan_cred[ny_out]->SetLatency( 1 );
    }
    _routers[node]->AddOutputChannel( _chan[ny_out], _chan_cred[ny_out] );
    _routers[node]->AddInputChannel( _chan[ny_in], _chan_cred[ny_in] );    

    if(gTrace){
      cout<<"Link "<<" "<<ny_out<<" "<<ny_in<<" "<<node<<" "<<_chan[ny_out]->GetLatency()<<endl;
    }
    
  }    

  // Check that all processors were connected to the network
  for ( int i = 0 ; i < _nodes ; i++ ) 
    assert( channel_vector[i] == true ) ;
  
  if(gTrace){
    cout<<"Setup Finished Link"<<endl;
  }
}


// ----------------------------------------------------------------------
//
//  Routing Helper Functions
//
// ----------------------------------------------------------------------

int CMesh::NodeToRouter( int address ) {

  int y  = (address /  (_cX*gK))/_cY ;
  int x  = (address %  (_cX*gK))/_cY ;
  int router = y*gK + x ;
  
  return router ;
}

int CMesh::NodeToPort( int address ) {
  
  const int maskX  = _cX - 1 ;
  const int maskY  = _cY - 1 ;

  int x = address & maskX ;
  int y = (int)(address/(2*gK)) & maskY ;

  return (gC / 2) * y + x;
}

// ----------------------------------------------------------------------
//
//  Routing Functions
//
// ----------------------------------------------------------------------

// Concentrated Mesh: X-Y
int cmesh_xy( int cur, int dest ) {

  const int POSITIVE_X = 0 ;
  const int NEGATIVE_X = 1 ;
  const int POSITIVE_Y = 2 ;
  const int NEGATIVE_Y = 3 ;

  int cur_y  = cur / gK;
  int cur_x  = cur % gK;
  int dest_y = dest / gK;
  int dest_x = dest % gK;

  // Dimension-order Routing: x , y
  if (cur_x < dest_x) {
    // Express?
    if ((dest_x - cur_x) > 1){
      if (cur_y == 0)
    	return gC + NEGATIVE_Y ;
      if (cur_y == (gK-1))
    	return gC + POSITIVE_Y ;
    }
    return gC + POSITIVE_X ;
  }
  if (cur_x > dest_x) {
    // Express ? 
    if ((cur_x - dest_x) > 1){
      if (cur_y == 0)
    	return gC + NEGATIVE_Y ;
      if (cur_y == (gK-1))
    	return gC + POSITIVE_Y ;
    }
    return gC + NEGATIVE_X ;
  }
  if (cur_y < dest_y) {
    // Express?
    if ((dest_y - cur_y) > 1) {
      if (cur_x == 0)
    	return gC + NEGATIVE_X ;
      if (cur_x == (gK-1))
    	return gC + POSITIVE_X ;
    }
    return gC + POSITIVE_Y ;
  }
  if (cur_y > dest_y) {
    // Express ?
    if ((cur_y - dest_y) > 1 ){
      if (cur_x == 0)
    	return gC + NEGATIVE_X ;
      if (cur_x == (gK-1))
    	return gC + POSITIVE_X ;
    }
    return gC + NEGATIVE_Y ;
  }
  return 0;
}

// Concentrated Mesh: Y-X
int cmesh_yx( int cur, int dest ) {
  const int POSITIVE_X = 0 ;
  const int NEGATIVE_X = 1 ;
  const int POSITIVE_Y = 2 ;
  const int NEGATIVE_Y = 3 ;

  int cur_y  = cur / gK ;
  int cur_x  = cur % gK ;
  int dest_y = dest / gK ;
  int dest_x = dest % gK ;

  // Dimension-order Routing: y, x
  if (cur_y < dest_y) {
    // Express?
    if ((dest_y - cur_y) > 1) {
      if (cur_x == 0)
    	return gC + NEGATIVE_X ;
      if (cur_x == (gK-1))
    	return gC + POSITIVE_X ;
    }
    return gC + POSITIVE_Y ;
  }
  if (cur_y > dest_y) {
    // Express ?
    if ((cur_y - dest_y) > 1 ){
      if (cur_x == 0)
    	return gC + NEGATIVE_X ;
      if (cur_x == (gK-1))
    	return gC + POSITIVE_X ;
    }
    return gC + NEGATIVE_Y ;
  }
  if (cur_x < dest_x) {
    // Express?
    if ((dest_x - cur_x) > 1){
      if (cur_y == 0)
    	return gC + NEGATIVE_Y ;
      if (cur_y == (gK-1))
    	return gC + POSITIVE_Y ;
    }
    return gC + POSITIVE_X ;
  }
  if (cur_x > dest_x) {
    // Express ? 
    if ((cur_x - dest_x) > 1){
      if (cur_y == 0)
    	return gC + NEGATIVE_Y ;
      if (cur_y == (gK-1))
    	return gC + POSITIVE_Y ;
    }
    return gC + NEGATIVE_X ;
  }
  return 0;
}

void xy_yx_cmesh( const Router *r, const Flit *f, int in_channel, 
		  OutputSet *outputs, bool inject )
{

  // ( Traffic Class , Routing Order ) -> Virtual Channel Range
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    // Current Router
    int cur_router = r->GetID();

    // Destination Router
    int dest_router = CMesh::NodeToRouter( f->dest ) ;  

    if (dest_router == cur_router) {

      // Forward to processing element
      out_port = CMesh::NodeToPort( f->dest );      

    } else {

      // Forward to neighbouring router

      //each class must have at least 2 vcs assigned or else xy_yx will deadlock
      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      // randomly select dimension order at first hop
      bool x_then_y = ((in_channel < gC) ?
		       (RandomInt(1) > 0) :
		       (f->vc < (vcBegin + available_vcs)));

      if(x_then_y) {
	out_port = cmesh_xy( cur_router, dest_router );
	vcEnd -= available_vcs;
      } else {
	out_port = cmesh_yx( cur_router, dest_router );
	vcBegin += available_vcs;
      }
    }

  }

  outputs->Clear();

  outputs->AddRange( out_port , vcBegin, vcEnd );
}

// ----------------------------------------------------------------------
//
//  Concentrated Mesh: Random XY-YX w/o Express Links
//
//   <int> cur:  current router address 
///  <int> dest: destination router address 
//
// ----------------------------------------------------------------------

int cmesh_xy_no_express( int cur, int dest ) {
  
  const int POSITIVE_X = 0 ;
  const int NEGATIVE_X = 1 ;
  const int POSITIVE_Y = 2 ;
  const int NEGATIVE_Y = 3 ;

  const int cur_y  = cur  / gK ;
  const int cur_x  = cur  % gK ;
  const int dest_y = dest / gK ;
  const int dest_x = dest % gK ;


  //  Note: channel numbers bellow gC (degree of concentration) are
  //        injection and ejection links

  // Dimension-order Routing: X , Y
  if (cur_x < dest_x) {
    return gC + POSITIVE_X ;
  }
  if (cur_x > dest_x) {
    return gC + NEGATIVE_X ;
  }
  if (cur_y < dest_y) {
    return gC + POSITIVE_Y ;
  }
  if (cur_y > dest_y) {
    return gC + NEGATIVE_Y ;
  }
  return 0;
}

int cmesh_yx_no_express( int cur, int dest ) {

  const int POSITIVE_X = 0 ;
  const int NEGATIVE_X = 1 ;
  const int POSITIVE_Y = 2 ;
  const int NEGATIVE_Y = 3 ;
  
  const int cur_y  = cur / gK ;
  const int cur_x  = cur % gK ;
  const int dest_y = dest / gK ;
  const int dest_x = dest % gK ;

  //  Note: channel numbers bellow gC (degree of concentration) are
  //        injection and ejection links

  // Dimension-order Routing: X , Y
  if (cur_y < dest_y) {
    return gC + POSITIVE_Y ;
  }
  if (cur_y > dest_y) {
    return gC + NEGATIVE_Y ;
  }
  if (cur_x < dest_x) {
    return gC + POSITIVE_X ;
  }
  if (cur_x > dest_x) {
    return gC + NEGATIVE_X ;
  }
  return 0;
}

void xy_yx_no_express_cmesh( const Router *r, const Flit *f, int in_channel, 
			     OutputSet *outputs, bool inject )
{
  // ( Traffic Class , Routing Order ) -> Virtual Channel Range
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    // Current Router
    int cur_router = r->GetID();

    // Destination Router
    int dest_router = CMesh::NodeToRouter( f->dest );  

    if (dest_router == cur_router) {

      // Forward to processing element
      out_port = CMesh::NodeToPort( f->dest );

    } else {

      // Forward to neighbouring router
    
      //each class must have at least 2 vcs assigned or else xy_yx will deadlock
      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      // randomly select dimension order at first hop
      bool x_then_y = ((in_channel < gC) ?
		       (RandomInt(1) > 0) :
		       (f->vc < (vcBegin + available_vcs)));

      if(x_then_y) {
	out_port = cmesh_xy_no_express( cur_router, dest_router );
	vcEnd -= available_vcs;
      } else {
	out_port = cmesh_yx_no_express( cur_router, dest_router );
	vcBegin += available_vcs;
      }
    }
  }

  outputs->Clear();

  outputs->AddRange( out_port , vcBegin, vcEnd );
}
//============================================================
//
//=====
int cmesh_next( int cur, int dest ) {

  const int POSITIVE_X = 0 ;
  const int NEGATIVE_X = 1 ;
  const int POSITIVE_Y = 2 ;
  const int NEGATIVE_Y = 3 ;
  
  int cur_y  = cur / gK ;
  int cur_x  = cur % gK ;
  int dest_y = dest / gK ;
  int dest_x = dest % gK ;

  // Dimension-order Routing: x , y
  if (cur_x < dest_x) {
    // Express?
    if ((dest_x - cur_x) > gK/2-1){
      if (cur_y == 0)
	return gC + NEGATIVE_Y ;
      if (cur_y == (gK-1))
	return gC + POSITIVE_Y ;
    }
    return gC + POSITIVE_X ;
  }
  if (cur_x > dest_x) {
    // Express ? 
    if ((cur_x - dest_x) > gK/2-1){
      if (cur_y == 0)
	return gC + NEGATIVE_Y ;
      if (cur_y == (gK-1)) 
	return gC + POSITIVE_Y ;
    }
    return gC + NEGATIVE_X ;
  }
  if (cur_y < dest_y) {
    // Express?
    if ((dest_y - cur_y) > gK/2-1) {
      if (cur_x == 0)
	return gC + NEGATIVE_X ;
      if (cur_x == (gK-1))
	return gC + POSITIVE_X ;
    }
    return gC + POSITIVE_Y ;
  }
  if (cur_y > dest_y) {
    // Express ?
    if ((cur_y - dest_y) > gK/2-1){
      if (cur_x == 0)
	return gC + NEGATIVE_X ;
      if (cur_x == (gK-1))
	return gC + POSITIVE_X ;
    }
    return gC + NEGATIVE_Y ;
  }

  assert(false);
  return -1;
}

void dor_cmesh( const Router *r, const Flit *f, int in_channel, 
		OutputSet *outputs, bool inject )
{
  // ( Traffic Class , Routing Order ) -> Virtual Channel Range
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    // Current Router
    int cur_router = r->GetID();

    // Destination Router
    int dest_router = CMesh::NodeToRouter( f->dest ) ;  
  
    if (dest_router == cur_router) {

      // Forward to processing element
      out_port = CMesh::NodeToPort( f->dest ) ;

    } else {

      // Forward to neighbouring router
      out_port = cmesh_next( cur_router, dest_router );
    }
  }

  outputs->Clear();

  outputs->AddRange( out_port, vcBegin, vcEnd);
}

//============================================================
//
//=====
int cmesh_next_no_express( int cur, int dest ) {

  const int POSITIVE_X = 0 ;
  const int NEGATIVE_X = 1 ;
  const int POSITIVE_Y = 2 ;
  const int NEGATIVE_Y = 3 ;
  
  //magic constant 2, which is supose to be _cX and _cY
  int cur_y  = cur/gK ;
  int cur_x  = cur%gK ;
  int dest_y = dest/gK;
  int dest_x = dest%gK ;

  // Dimension-order Routing: x , y
  if (cur_x < dest_x) {
    return gC + POSITIVE_X ;
  }
  if (cur_x > dest_x) {
    return gC + NEGATIVE_X ;
  }
  if (cur_y < dest_y) {
    return gC + POSITIVE_Y ;
  }
  if (cur_y > dest_y) {
    return gC + NEGATIVE_Y ;
  }
  assert(false);
  return -1;
}

void dor_no_express_cmesh( const Router *r, const Flit *f, int in_channel, 
			   OutputSet *outputs, bool inject )
{
  // ( Traffic Class , Routing Order ) -> Virtual Channel Range
  int vcBegin = 0, vcEnd = gNumVCs-1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if(inject) {

    out_port = -1;

  } else {

    // Current Router
    int cur_router = r->GetID();

    // Destination Router
    int dest_router = CMesh::NodeToRouter( f->dest ) ;  
  
    if (dest_router == cur_router) {

      // Forward to processing element
      out_port = CMesh::NodeToPort( f->dest );

    } else {

      // Forward to neighbouring router
      out_port = cmesh_next_no_express( cur_router, dest_router );
    }
  }

  outputs->Clear();

  outputs->AddRange( out_port, vcBegin, vcEnd );
}
