// $Id: tree4.hpp 5188 2012-08-30 00:31:31Z dub $

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
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: tree4.hpp 5188 2012-08-30 00:31:31Z dub $
// 
////////////////////////////////////////////////////////////////////////

#ifndef _TREE4_HPP_
#define _TREE4_HPP_
#include <cassert>
#include "network.hpp"

class Tree4 : public Network {

  int _k;
  int _n;

  int _channelWidth;

  void _ComputeSize( const Configuration& config );
  void _BuildNet( const Configuration& config );


  Router*& _Router( int height, int pos );

  int _WireLatency( int height1, int pos1, int height2, int pos2 );

public:

  Tree4( const Configuration& config, const string & name );
  static void RegisterRoutingFunctions() ;
  
  static int HeightFromID( int id );
  static int PosFromID( int id );
  static int SpeedUp( int height );
};

#endif
