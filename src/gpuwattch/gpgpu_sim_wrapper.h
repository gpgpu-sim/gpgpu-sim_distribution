// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington, Ahmed ElTantawy,
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

#ifndef GPGPU_SIM_WRAPPER_H_
#define GPGPU_SIM_WRAPPER_H_

#include "processor.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>
#include <zlib.h>
#include <string.h>


using namespace std;




class gpgpu_sim_wrapper {
public:
	gpgpu_sim_wrapper(bool power_simulation_enabled, char* xmlfile);
	~gpgpu_sim_wrapper();

	void init_mcpat(char* xmlfile, char* powerfile, char* power_trace_file,char* metric_trace_file,
			char * steady_state_file,bool power_sim_enabled,bool trace_enabled,bool steady_state_enabled,
			bool power_per_cycle_dump,double steady_power_deviation,double steady_min_period,int zlevel,
			double init_val,int stat_sample_freq);
	void detect_print_steady_state(int position, double init_val);
	void close_files();
	void open_files();
	void compute();
	void dump();
	void print_trace_files();
	void update_components_power();
	void update_coefficients();
	void print_power_kernel_stats(double gpu_sim_cycle, double gpu_tot_sim_cycle, double init_value);
	void power_metrics_calculations();
	void set_inst_power(bool clk_gated_lanes, double tot_cycles, double busy_cycles, double tot_inst, double int_inst, double fp_inst, double load_inst, double store_inst, double committed_inst);
	void set_regfile_power(double reads, double writes, double ops);
	void set_icache_power(double accesses, double misses);
	void set_ccache_power(double accesses, double misses);
	void set_tcache_power(double accesses, double misses);
	void set_shrd_mem_power(double accesses);
	void set_l1cache_power(double read_accesses, double read_misses, double write_accesses, double write_misses);
	void set_l2cache_power(double read_accesses, double read_misses, double write_accesses, double write_misses);
	void set_idle_core_power(double num_idle_core);
	void set_duty_cycle_power(double duty_cycle);
	void set_mem_ctrl_power(double reads, double writes, double dram_precharge);
	void set_exec_unit_power(double fpu_accesses, double ialu_accesses, double sfu_accesses);
	void set_active_lanes_power(double sp_avg_active_lane, double sfu_avg_active_lane);
	void set_NoC_power(double noc_tot_reads,double noc_tot_write);
	bool sanity_check(double a, double b);

private:
	Processor* proc;
	ParseXML * p;
    // power parameters
    double const_dynamic_power;
    double * pwr_cmp;
    double proc_power;
    double * pwr_cmp_avg;
    double * pwr_cmp_max;
    double * pwr_cmp_min;
    double *   perf_count_avg;
    double *   perf_count_min;
    double *   perf_count_max;
    double pavg;
    int count;
    unsigned num_per_counts;
    unsigned num_pwr_cmps;
    int gcount;
    double gpu_avg_power;
    double gpu_max_power;
    double gpu_min_power;
    double gpu_tot_avg_power;
    double gpu_tot_min_power;
    double gpu_tot_max_power;
    bool has_written_avg;

    double * perf_count;
    double * initpower_coeff;
    double * effpower_coeff;
    // For calculating steady-state average
    unsigned sample_start;
    double sample_val;
    double init_inst_val;
    std::vector<double> samples;
    std::vector< double > samples_counter;
    std::vector< double > pwr_counter;

    char *xml_filename;
    char *g_power_filename;
    char *g_power_trace_filename;
    char *g_metric_trace_filename;
    char * g_steady_state_tracking_filename;
    bool g_power_simulation_enabled;
    bool g_steady_power_levels_enabled;
    bool g_power_trace_enabled;
    bool g_power_per_cycle_dump;
	double   gpu_steady_power_deviation;
	double   gpu_steady_min_period;
	int g_power_trace_zlevel;
    double gpu_stat_sample_frequency;
    int gpu_stat_sample_freq;

    std::ofstream powerfile;
    gzFile power_trace_file;
    gzFile metric_trace_file;
    gzFile steady_state_tacking_file;
};

#endif /* GPGPU_SIM_WRAPPER_H_ */
