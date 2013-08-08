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
  InterconnectInterface(const char* const config_file, unsigned int n_shader,  unsigned int n_mem);
  virtual ~InterconnectInterface();
  
  //node side functions
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data, unsigned int size);
  void* Pop(unsigned ouput_deviceID);
  void Advance();
  bool Busy() const;
  bool HasBuffer(unsigned deviceID, unsigned int size) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;
  
  void DisplayState(FILE* fp) const;
  
  //booksim side functions
  void WriteOutBuffer( int subnet, int output, Flit* flit );
  void Transfer2BoundaryBuffer(int subnet, int output);
  
  int GetIcntTime() const;
  
  Stats* GetIcntStats(const string & name) const;
  
  Flit* GetEjectedFlit(int subnet, int node);
  
  
private:
  
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
  void _CreateNodeMap(int n_shader, int n_mem, int n_node, int use_map);
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
  unsigned int _flit_size;
  IntersimConfig* _icnt_config;
  const int _n_shader, _n_mem;
  vector<Network *> _net;
  int _vcs;
  int _subnets;
  
  //deviceID to icntID map
  //deviceID : Starts from 0 for shaders and then continues until mem nodes
  //which starts at location n_shader and then continues to n_shader+n_mem (last device)
  map<int, int> _node_map;
  
  //icntID to deviceID map
  map<int, int> _reverse_node_map;

};

#endif


