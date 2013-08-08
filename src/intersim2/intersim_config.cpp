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

#include "intersim_config.hpp"

IntersimConfig::IntersimConfig()
{
  // Add options for intersim
  
  _int_map["perfect_icnt"] = 0; // if set overrides fixed_lat_per_hop setting
  _int_map["fixed_lat_per_hop"] = 0; // if set icnt is NOT simulated instead packets are sent into destination based on a fixed_lat_per_hop
  _int_map["network_count"] = 2; // number of independent interconnection networks (if it is set to 2 then 2 identical networks are created: sh2mem and mem2shd )
  
  _int_map["output_extra_latency"] = 0;
  
  _int_map["use_map"] = 1;
  
  _int_map["flit_size"] = 32;
  //stats
  _int_map["enable_link_stats"]    = 0;     // show output link and VC utilization stats
  
  _int_map["MATLAB_OUTPUT"]        = 0;     // output data in MATLAB friendly format
  _int_map["DISPLAY_LAT_DIST"]     = 0; // distribution of packet latencies
  _int_map["DISPLAY_HOP_DIST"]     = 0;     // distribution of hop counts
  _int_map["DISPLAY_PAIR_LATENCY"] = 0;     // avg. latency for each s-d pair
  
  _int_map["input_buffer_size"] = 0;
  _int_map["ejection_buffer_size"] = 0; // if left zero the simulator will use the vc_buf_size instead
  _int_map["boundary_buffer_size"] = 16;
}