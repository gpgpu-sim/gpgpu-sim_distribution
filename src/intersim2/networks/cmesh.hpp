// $Id: cmesh.hpp 5188 2012-08-30 00:31:31Z dub $

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
// CMesh: Mesh topology with concentration and express links along the
//         edge of the network
//
////////////////////////////////////////////////////////////////////////
//
// RCS Information:
//  $Author: jbalfour $
//  $Date: 2007/06/26 22:49:23 $
//  $Id: cmesh.hpp 5188 2012-08-30 00:31:31Z dub $
// 
////////////////////////////////////////////////////////////////////////
#ifndef _CMESH_HPP_
#define _CMESH_HPP_

#include "network.hpp"
#include "routefunc.hpp"

class CMesh : public Network {
public:
  CMesh( const Configuration &config, const string & name );
  int GetN() const;
  int GetK() const;

  static int NodeToRouter( int address ) ;
  static int NodeToPort( int address ) ;

  static void RegisterRoutingFunctions() ;

private:

  static int _cX ;
  static int _cY ;

  static int _memo_NodeShiftX ;
  static int _memo_NodeShiftY ;
  static int _memo_PortShiftY ;

  void _ComputeSize( const Configuration &config );
  void _BuildNet( const Configuration& config );

  int _k ;
  int _n ;
  int _c ;
  int _xcount;
  int _ycount;
  int _xrouter;
  int _yrouter;
  bool _express_channels;
};

//
// Routing Functions
//
void xy_yx_cmesh( const Router *r, const Flit *f, int in_channel, 
		  OutputSet *outputs, bool inject ) ;

void xy_yx_no_express_cmesh( const Router *r, const Flit *f, int in_channel, 
			     OutputSet *outputs, bool inject ) ;

void dor_cmesh( const Router *r, const Flit *f, int in_channel, 
		OutputSet *outputs, bool inject ) ;

void dor_no_express_cmesh( const Router *r, const Flit *f, int in_channel, 
			   OutputSet *outputs, bool inject ) ;

#endif
