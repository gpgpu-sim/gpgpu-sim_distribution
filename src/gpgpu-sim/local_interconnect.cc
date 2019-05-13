// Copyright (c) 2019, Mahmoud Khairy
// Purdue University
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

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <utility>
#include <algorithm>

#include "local_interconnect.h"
#include "mem_fetch.h"

xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type, unsigned n_shader, unsigned n_mem, unsigned m_in_buffer_limit, unsigned m_out_buffer_limit)
{
	m_id=router_id;
	router_type=m_type;
	_n_mem = n_mem;
	_n_shader = n_shader;
	total_nodes = n_shader+n_mem;
	in_buffers.resize(total_nodes);
	out_buffers.resize(total_nodes);
	next_node=0;
	in_buffer_limit = m_in_buffer_limit;
	out_buffer_limit = m_out_buffer_limit;
	if(m_type == REQ_NET) {
		active_in_buffers=n_shader;
		active_out_buffers=n_mem;
	}
	else if(m_type == REPLY_NET) {
		active_in_buffers=n_mem;
		active_out_buffers=n_shader;
	}

	cycles = 0;
	conflicts= 0;
	out_buffer_full=0;
	in_buffer_full=0;
	out_buffer_util=0;
	in_buffer_util=0;
	packets_num=0;
}


xbar_router::~xbar_router()
{

}

void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID, void* data, unsigned int size)
{
	assert(input_deviceID < total_nodes);
	in_buffers[input_deviceID].push(Packet(data, output_deviceID));
	packets_num++;
}

void* xbar_router::Pop(unsigned ouput_deviceID)
{
	assert(ouput_deviceID < total_nodes);
	void* data = NULL;

	if(!out_buffers[ouput_deviceID].empty()) {
		data = out_buffers[ouput_deviceID].front().data;
		out_buffers[ouput_deviceID].pop();
	}

	return data;
}


bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned size, bool update_counter){

	assert(input_deviceID < total_nodes);

	bool has_buffer = (in_buffers[input_deviceID].size() + size <= in_buffer_limit);
	if(update_counter && !has_buffer)
		in_buffer_full++;

	return has_buffer;

}

bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned size){
	return (out_buffers[output_deviceID].size() + size <= out_buffer_limit);
}

void xbar_router::Advance() {
	cycles++;

	vector<bool> issued(total_nodes, false);

	for(unsigned i=0; i<total_nodes; ++i){
		unsigned node_id = (i+next_node)%total_nodes;

		if(!in_buffers[node_id].empty()) {
			Packet _packet = in_buffers[node_id].front();
			//ensure that the outbuffer has space and not issued before in this cycle
			if(Has_Buffer_Out(_packet.output_deviceID, 1)){
				if(!issued[_packet.output_deviceID]) {
					out_buffers[_packet.output_deviceID].push(_packet);
					in_buffers[node_id].pop();
					issued[_packet.output_deviceID]=true;
				}
				else
					conflicts++;
			}
			else
				out_buffer_full++;
		}
	}

	next_node = (++next_node % total_nodes);

	//collect some stats about buffer util
	for(unsigned i=0; i<total_nodes; ++i){
		in_buffer_util+=in_buffers[i].size();
		out_buffer_util+=out_buffers[i].size();
	}
}

bool xbar_router::Busy() const {

	for(unsigned i=0; i<total_nodes; ++i){
		if(!in_buffers[i].empty())
			return true;

		if(!out_buffers[i].empty())
			return true;
	}
	return false;
}


////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

//assume all the packets are one flit
#define LOCAL_INCT_FLIT_SIZE 40

LocalInterconnect* LocalInterconnect::New(const struct inct_config& m_localinct_config)
{

	LocalInterconnect* icnt_interface = new LocalInterconnect(m_localinct_config);

	return icnt_interface;
}

LocalInterconnect::LocalInterconnect(const struct inct_config& m_localinct_config): m_inct_config(m_localinct_config){
	n_shader=0;
	n_mem=0;
	n_subnets = m_localinct_config.subnets;
}

LocalInterconnect::~LocalInterconnect(){
	for (int i=0; i<m_inct_config.subnets; ++i) {
		delete net[i];
	}
}

void LocalInterconnect::CreateInterconnect(unsigned m_n_shader, unsigned m_n_mem){
	n_shader = m_n_shader;
	n_mem = m_n_mem;

	net.resize(n_subnets);
	for (unsigned i = 0; i < n_subnets; ++i) {
		net[i] = new xbar_router( i, static_cast<Interconnect_type>(i), m_n_shader, m_n_mem, m_inct_config.in_buffer_limit, m_inct_config.out_buffer_limit );
	}

}


void LocalInterconnect::Init() {
	//empty
	//there is nothing to do

}

void LocalInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID, void* data, unsigned int size){

	unsigned subnet;
	if (n_subnets == 1) {
		subnet = 0;
	} else {
		if (input_deviceID < n_shader ) {
			subnet = 0;
		} else {
			subnet = 1;
		}
	}

	// it should have free buffer
	//assume all the packets have size of one
	//no flits are implemented
	assert(net[subnet]->Has_Buffer_In(input_deviceID, 1));

	net[subnet]->Push(input_deviceID, output_deviceID, data, size);

}

void* LocalInterconnect::Pop(unsigned ouput_deviceID){

	// 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
	int subnet = 0;
	if (ouput_deviceID < n_shader)
		subnet = 1;

	return net[subnet]->Pop(ouput_deviceID);

}

void LocalInterconnect::Advance(){

	for (unsigned i = 0; i < n_subnets; ++i) {
		net[i]->Advance();
	}

}

bool  LocalInterconnect::Busy() const{

	for (unsigned i = 0; i < n_subnets; ++i) {
		if(net[i]->Busy())
			return true;
	}
	return false;
}

bool  LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const{

	bool has_buffer = false;

	if ((n_subnets>1) && deviceID >= n_shader) // deviceID is memory node
		has_buffer = net[REPLY_NET]->Has_Buffer_In(deviceID, 1, true);
	else
		has_buffer = net[REQ_NET]->Has_Buffer_In(deviceID, 1, true);

	return has_buffer;

}

void  LocalInterconnect::DisplayStats() const{

	cout<<"Req_Network_injected_packets_num = "<<net[REQ_NET]->packets_num<<endl;
	cout<<"Req_Network_cycles = "<<net[REQ_NET]->cycles<<endl;
	cout<<"Req_Network_injected_packets_per_cycle = "<<(float)(net[REQ_NET]->packets_num) / (net[REQ_NET]->cycles)<<endl;
	cout<<"Req_Network_conflicts_per_cycle = "<<(float)(net[REQ_NET]->conflicts) / (net[REQ_NET]->cycles)<<endl;
	cout<<"Req_Network_in_buffer_full_per_cycle = "<<(float)(net[REQ_NET]->in_buffer_full) / (net[REQ_NET]->cycles)<<endl;
	cout<<"Req_Network_in_buffer_avg_util = "<<((float)(net[REQ_NET]->in_buffer_util) / (net[REQ_NET]->cycles) / net[REQ_NET]->active_in_buffers)<<endl;
	cout<<"Req_Network_out_buffer_full_per_cycle = "<<(float)(net[REQ_NET]->out_buffer_full) / (net[REQ_NET]->cycles)<<endl;
	cout<<"Req_Network_out_buffer_avg_util = "<<((float)(net[REQ_NET]->out_buffer_util) / (net[REQ_NET]->cycles) / net[REQ_NET]->active_out_buffers)<<endl;

	cout<<endl;
	cout<<"Reply_Network_injected_packets_num = "<<net[REPLY_NET]->packets_num<<endl;
	cout<<"Reply_Network_cycles = "<<net[REPLY_NET]->cycles<<endl;
	cout<<"Reply_Network_injected_packets_per_cycle = "<<(float)(net[REPLY_NET]->packets_num) / (net[REPLY_NET]->cycles)<<endl;
	cout<<"Reply_Network_conflicts_per_cycle = "<<(float)(net[REPLY_NET]->conflicts) / (net[REPLY_NET]->cycles)<<endl;
	cout<<"Reply_Network_in_buffer_full_per_cycle = "<<(float)(net[REPLY_NET]->in_buffer_full) / (net[REPLY_NET]->cycles)<<endl;
	cout<<"Reply_Network_in_buffer_avg_util = "<<((float)(net[REPLY_NET]->in_buffer_util) / (net[REPLY_NET]->cycles) / net[REPLY_NET]->active_in_buffers)<<endl;
	cout<<"Reply_Network_out_buffer_full_per_cycle = "<<(float)(net[REPLY_NET]->out_buffer_full) / (net[REPLY_NET]->cycles)<<endl;
	cout<<"Reply_Network_out_buffer_avg_util= "<<((float)(net[REPLY_NET]->out_buffer_util) / (net[REPLY_NET]->cycles) / net[REPLY_NET]->active_out_buffers)<<endl;

}

void  LocalInterconnect::DisplayOverallStats() const{

}

unsigned  LocalInterconnect::GetFlitSize() const{
	return LOCAL_INCT_FLIT_SIZE;
}

void  LocalInterconnect::DisplayState(FILE* fp) const{

	fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n");
}

