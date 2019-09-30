//developed by Mahmoud Khairy, Purdue Univ
//abdallm@purdue.edu

#include <time.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../cuda-sim/cuda-sim.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../../libcuda/gpgpu_context.h"
#include "trace_driven.h"
#include "trace_opcode.h"
#include "../gpgpusim_entrypoint.h"


int main ( int argc, const char **argv )
{

	gpgpu_context* m_gpgpu_context = new gpgpu_context();
	gpgpu_sim * m_gpgpu_sim = m_gpgpu_context->gpgpu_trace_sim_init_perf(argc,argv);
	m_gpgpu_sim->init();

	//for each kernel
	//load file
	//parse and create kernel info
	//launch
	//while loop till the end of the end kernel execution
	//prints stats

	trace_parser tracer(m_gpgpu_sim->get_config().get_traces_filename(), m_gpgpu_sim, m_gpgpu_context);

	std::vector<std::string> kernellist = tracer.parse_kernellist_file();

	for(unsigned i=0; i<kernellist.size(); ++i) {

		trace_kernel_info_t* kernel_info  = tracer.parse_kernel_info(kernellist[i]);
		m_gpgpu_sim->launch(kernel_info);

		bool active = false;
		bool sim_cycles = false;
		bool break_limit = false;

		do {
			if(!m_gpgpu_sim->active())
				break;

			//performance simulation
			if( m_gpgpu_sim->active() ) {
				m_gpgpu_sim->cycle();
				sim_cycles = true;
				m_gpgpu_sim->deadlock_check();
			}else {
				if(m_gpgpu_sim->cycle_insn_cta_max_hit()){
					m_gpgpu_context->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
					break_limit = true;
				}
			}

			active=m_gpgpu_sim->active() ;

		} while( active );

		tracer.kernel_finalizer(kernel_info);

		m_gpgpu_sim->print_stats();

		if(sim_cycles) {
			m_gpgpu_sim->update_stats();
			m_gpgpu_context->print_simulation_time();
		}

		if(break_limit) {
			printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **\n");
			fflush(stdout);
			exit(1);
		}
	}

	return 1;
}
