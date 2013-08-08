// Copyright (c) 2009-2013, Tor M. Aamodt, Dongdong Li, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _GPUTRAFFICMANAGER_HPP_
#define _GPUTRAFFICMANAGER_HPP_

#include <iostream>
#include <vector>
#include <list>

#include "config_utils.hpp"
#include "stats.hpp"
#include "trafficmanager.hpp"
#include "booksim.hpp"
#include "booksim_config.hpp"
#include "flit.hpp"

class GPUTrafficManager : public TrafficManager {
  
protected:
  virtual void _RetireFlit( Flit *f, int dest );
  virtual void _GeneratePacket(int source, int stype, int cl, int time, int subnet, int package_size, const Flit::FlitType& packet_type, void* const data, int dest);
  virtual int  _IssuePacket( int source, int cl );
  virtual void _Step();
  
  // record size of _partial_packets for each subnet
  vector<vector<vector<list<Flit *> > > > _input_queue;
  
public:
  
  GPUTrafficManager( const Configuration &config, const vector<Network *> & net );
  virtual ~GPUTrafficManager( );
  
  // correspond to TrafficManger::Run/SingleSim
  void Init();
  
  // TODO: if it is not good...
  friend class InterconnectInterface;
  
  
  
  //    virtual void WriteStats( ostream & os = cout ) const;
  //    virtual void DisplayStats( ostream & os = cout ) const;
  //    virtual void DisplayOverallStats( ostream & os = cout ) const;
  
};



#endif
