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

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be limited by disk speed)
 * 2- traces compression format
 * 	 a. cfg format and remove thread/block Id from the head
 * 	 b. using zlib library to save in binary format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in the buffer)
 * 4- Seeking capability - thread scheduler (save tb index and warp index info in the traces header)
 * 5- Get rid off traces intermediate files - change the tracer
 */

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

	std::vector<std::string> commandlist = tracer.parse_kernellist_file();

	for(unsigned i=0; i<commandlist.size(); ++i) {

		trace_kernel_info_t* kernel_info;
		if(commandlist[i].substr(0,6) == "Memcpy") {
			
			size_t addre, Bcount;
			tracer.parse_memcpy_info(commandlist[i], addre, Bcount);
			m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
			continue;
		}
		else if(commandlist[i].substr(0,6) == "kernel") {
			kernel_info  = tracer.parse_kernel_info(commandlist[i]);
			m_gpgpu_sim->launch(kernel_info);
		}

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

	//we print this message to inform the gpgpu-simulation stats_collect script that we are done
    printf("GPGPU-Sim: *** simulation thread exiting ***\n");
    printf("GPGPU-Sim: *** exit detected ***\n");

	return 1;
}
