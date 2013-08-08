// $Id: flatfly_onchip.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _FlatFlyOnChip_HPP_
#define _FlatFlyOnChip_HPP_

#include "network.hpp"

#include "routefunc.hpp"
#include <cassert>


class FlatFlyOnChip : public Network {

  int _m;
  int _n;
  int _r;
  int _k;
  int _c;
  int _radix;
  int _net_size;
  int _stageout;
  int _numinput;
  int _stages;
  int _num_of_switch;

  void _ComputeSize( const Configuration &config );
  void _BuildNet( const Configuration &config );

  int _OutChannel( int stage, int addr, int port, int outputs ) const;
  int _InChannel( int stage, int addr, int port ) const;

public:
  FlatFlyOnChip( const Configuration &config, const string & name );

  int GetN( ) const;
  int GetK( ) const;

  static void RegisterRoutingFunctions() ;
  double Capacity( ) const;
  void InsertRandomFaults( const Configuration &config );
};
void adaptive_xyyx_flatfly( const Router *r, const Flit *f, int in_channel, 
		  OutputSet *outputs, bool inject );
void xyyx_flatfly( const Router *r, const Flit *f, int in_channel, 
		  OutputSet *outputs, bool inject );
void min_flatfly( const Router *r, const Flit *f, int in_channel, 
		  OutputSet *outputs, bool inject );
void ugal_xyyx_flatfly_onchip( const Router *r, const Flit *f, int in_channel,
			  OutputSet *outputs, bool inject );
void ugal_flatfly_onchip( const Router *r, const Flit *f, int in_channel,
			  OutputSet *outputs, bool inject );
void ugal_pni_flatfly_onchip( const Router *r, const Flit *f, int in_channel,
			      OutputSet *outputs, bool inject );
void valiant_flatfly( const Router *r, const Flit *f, int in_channel,
			  OutputSet *outputs, bool inject );

int find_distance (int src, int dest);
int find_ran_intm (int src, int dest);
int flatfly_outport(int dest, int rID);
int flatfly_transformation(int dest);
int flatfly_outport_yx(int dest, int rID);

#endif
