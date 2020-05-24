// Copyright (c) 2009-2011, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef STATS_INCLUDED
#define STATS_INCLUDED

enum mem_stage_access_type {
  C_MEM,
  T_MEM,
  S_MEM,
  G_MEM_LD,
  L_MEM_LD,
  G_MEM_ST,
  L_MEM_ST,
  N_MEM_STAGE_ACCESS_TYPE
};
enum tlb_request_status { TLB_HIT = 0, TLB_READY, TLB_PENDING };
enum mem_stage_stall_type {
  NO_RC_FAIL = 0,
  BK_CONF,
  MSHR_RC_FAIL,
  ICNT_RC_FAIL,
  COAL_STALL,
  TLB_STALL,
  DATA_PORT_STALL,
  WB_ICNT_RC_FAIL,
  WB_CACHE_RSRV_FAIL,
  N_MEM_STAGE_STALL_TYPE
};

#endif
