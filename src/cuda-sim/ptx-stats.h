// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt
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

#pragma once

#include "../option_parser.h"

#ifdef __cplusplus
// stat collection interface to cuda-sim
class ptx_instruction;
void ptx_file_line_stats_add_exec_count(const ptx_instruction* pInsn);
#endif

// stat collection interface to gpgpu-sim

void ptx_file_line_stats_create_exposed_latency_tracker(int n_shader_cores);
void ptx_file_line_stats_commit_exposed_latency(int sc_id, int exposed_latency);

class gpgpu_context;
class ptx_stats {
 public:
  ptx_stats(gpgpu_context* ctx) {
    ptx_line_stats_filename = NULL;
    gpgpu_ctx = ctx;
  }
  char* ptx_line_stats_filename;
  bool enable_ptx_file_line_stats;
  gpgpu_context* gpgpu_ctx;
  // set options
  void ptx_file_line_stats_options(option_parser_t opp);

  // output stats to a file
  void ptx_file_line_stats_write_file();
  // stat collection interface to gpgpu-sim
  void ptx_file_line_stats_add_latency(unsigned pc, unsigned latency);
  void ptx_file_line_stats_add_dram_traffic(unsigned pc, unsigned dram_traffic);
  void ptx_file_line_stats_add_smem_bank_conflict(unsigned pc,
                                                  unsigned n_way_bkconflict);
  void ptx_file_line_stats_add_uncoalesced_gmem(unsigned pc, unsigned n_access);
  void ptx_file_line_stats_add_inflight_memory_insn(int sc_id, unsigned pc);
  void ptx_file_line_stats_sub_inflight_memory_insn(int sc_id, unsigned pc);
  void ptx_file_line_stats_add_warp_divergence(unsigned pc,
                                               unsigned n_way_divergence);
};
