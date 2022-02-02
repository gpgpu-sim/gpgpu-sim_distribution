// Copyright (c) 2009-2021, Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef POWER_INTERFACE_H_
#define POWER_INTERFACE_H_

#include "gpu-sim.h"
#include "power_stat.h"
#include "shader.h"

#include "gpgpu_sim_wrapper.h"

void init_mcpat(const gpgpu_sim_config &config,
                class gpgpu_sim_wrapper *wrapper, unsigned stat_sample_freq,
                unsigned tot_inst, unsigned inst);
void mcpat_cycle(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst, bool dvfs_enabled);

void calculate_hw_mcpat(const gpgpu_sim_config &config,
                 const shader_core_config *shdr_config,
                 class gpgpu_sim_wrapper *wrapper,
                 class power_stat_t *power_stats, unsigned stat_sample_freq,
                 unsigned tot_cycle, unsigned cycle, unsigned tot_inst,
                 unsigned inst, int power_simulation_mode, bool dvfs_enabled, 
                 char* hwpowerfile, char* benchname, std::string executed_kernelname, 
                 const bool *accelwattch_hybrid_configuration, bool aggregate_power_stats);

bool parse_hw_file(char* hwpowerfile, bool find_target_kernel, vector<string> &hw_data, char* benchname, std::string executed_kernelname);

void mcpat_reset_perf_count(class gpgpu_sim_wrapper *wrapper);

#endif /* POWER_INTERFACE_H_ */
