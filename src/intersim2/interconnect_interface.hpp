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

#ifndef _INTERCONNECT_INTERFACE_HPP_
#define _INTERCONNECT_INTERFACE_HPP_

#include <vector>
#include <queue>
#include <iostream>
#include <map>
using namespace std;


// Do not use #include since it will not compile in icnt_wrapper or change the makefile to make it
class Flit;
class GPUTrafficManager;
class IntersimConfig;
class Network;
class Stats;

//TODO: fixed_lat_icnt, add class support? support for signle network

class InterconnectInterface {
public:
  InterconnectInterface();
  virtual ~InterconnectInterface();
  static InterconnectInterface* New(const char* const config_file);
  virtual void CreateInterconnect(unsigned n_shader,  unsigned n_mem);
  
  //node side functions
  virtual void Init();
  virtual void Push(unsigned input_deviceID, unsigned output_deviceID, void* data, unsigned int size);
  virtual void* Pop(unsigned ouput_deviceID);
  virtual void Advance();
  virtual bool Busy() const;
  virtual bool HasBuffer(unsigned deviceID, unsigned int size) const;
  virtual void DisplayStats() const;
  virtual void DisplayOverallStats() const;
  unsigned GetFlitSize() const;
  
  virtual void DisplayState(FILE* fp) const;
  
  //booksim side functions
  void WriteOutBuffer( int subnet, int output, Flit* flit );
  void Transfer2BoundaryBuffer(int subnet, int output);
  
  int GetIcntTime() const;
  
  Stats* GetIcntStats(const string & name) const;
  
  Flit* GetEjectedFlit(int subnet, int node);
  
protected:
  
  class _BoundaryBufferItem {
  public:
    _BoundaryBufferItem():_packet_n(0) {}
    inline unsigned Size(void) const { return _buffer.size(); }
    inline bool HasPacket() const { return _packet_n; }
    void* PopPacket();
    void* TopPacket() const;
    void PushFlitData(void* data,bool is_tail);
    
  private:
    queue<void *> _buffer;
    queue<bool> _tail_flag;
    int _packet_n;
  };
  typedef queue<Flit*> _EjectionBufferItem;
  
  void _CreateBuffer( );
  void _CreateNodeMap(unsigned n_shader, unsigned n_mem, unsigned n_node, int use_map);
  void _DisplayMap(int dim,int count);
  
  // size: [subnets][nodes][vcs]
  vector<vector<vector<_BoundaryBufferItem> > > _boundary_buffer;
  unsigned int _boundary_buffer_capacity;
  // size: [subnets][nodes][vcs]
  vector<vector<vector<_EjectionBufferItem> > > _ejection_buffer;
  // size:[subnets][nodes]
  vector<vector<queue<Flit* > > > _ejected_flit_queue;
  
  unsigned int _ejection_buffer_capacity;
  unsigned int _input_buffer_capacity;
  
  vector<vector<int> > _round_robin_turn; //keep track of _boundary_buffer last used in icnt_pop
  
  GPUTrafficManager* _traffic_manager;
  unsigned _flit_size;
  IntersimConfig* _icnt_config;
  unsigned _n_shader, _n_mem;
  vector<Network *> _net;
  int _vcs;
  int _subnets;
  
  //deviceID to icntID map
  //deviceID : Starts from 0 for shaders and then continues until mem nodes
  //which starts at location n_shader and then continues to n_shader+n_mem (last device)
  map<unsigned, unsigned> _node_map;
  
  //icntID to deviceID map
  map<unsigned, unsigned> _reverse_node_map;

};

#endif


