//developed by Mahmoud Khairy, Purdue Univ
//abdallm@purdue.edu

#include "../abstract_hardware_model.h"
#include <time.h>
#include <stdio.h>

#include "../option_parser.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpu-sim/icnt_wrapper.h"
//#include "../stream_manager.h"



gpgpu_sim_config g_the_gpu_config;
gpgpu_sim *g_the_gpu;
time_t g_simulation_starttime;

#define MAX(a,b) (((a)>(b))?(a):(b))

static void print_simulation_time();


int main ( int argc, char **argv )
{
   srand(1);

   option_parser_t opp = option_parser_create();

   icnt_reg_options(opp);
   g_the_gpu_config.reg_options(opp); // register GPU microrachitecture options
   ptx_reg_options(opp);
   ptx_opcocde_latency_options(opp);

   //ptx_opcocde_latency_options(opp);  //do this for trace driven

   fprintf(stdout, "I am here:\n\n");
   option_parser_cmdline(opp, argc, (const char **)argv); // parse configuration options

   fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
   option_parser_print(opp, stdout);
   // Set the Numeric locale to a standard locale where a decimal point is a "dot" not a "comma"
   // so it does the parsing correctly independent of the system environment variables
   assert(setlocale(LC_NUMERIC,"C"));
   g_the_gpu_config.init();

   g_the_gpu = new gpgpu_sim(g_the_gpu_config);
   //g_stream_manager = new stream_manager(g_the_gpu,g_cuda_launch_blocking);

   //load file
   //create kernel info
   //launch
   //while loop till the end
   //prints stats
   //g_the_gpu->launch(grid);

   g_simulation_starttime = time((time_t *)NULL);

	  bool active = false;
	  bool sim_cycles = false;
	  bool break_limit = false;

	  g_the_gpu->init();
	  do {
		  // check if a kernel has completed
		  // launch operation on device if one is pending and can be run

		  // Need to break this loop when a kernel completes. This was a
		  // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
		  // If another stream operation is available, g_the_gpu remains active,
		  // causing this loop to not break. If the next operation happens to be
		  // another kernel, the gpu is not re-initialized and the inter-kernel
		  // behaviour may be incorrect. Check that a kernel has finished and
		  // no other kernel is currently running.
		  if(!g_the_gpu->active())
			  break;

		  //performance simulation
		  if( g_the_gpu->active() ) {
			  g_the_gpu->cycle();
			  sim_cycles = true;
			  g_the_gpu->deadlock_check();
		  }else {
			  if(g_the_gpu->cycle_insn_cta_max_hit()){
				  g_stream_manager->stop_all_running_kernels();
				  break_limit = true;
			  }
		  }

		  active=g_the_gpu->active() ;

	  } while( active );

		 printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
		 fflush(stdout);

	  g_the_gpu->print_stats();

	  if(sim_cycles) {
		  g_the_gpu->update_stats();
		  print_simulation_time();
	  }


		 printf("GPGPU-Sim: *** simulation thread exiting ***\n");
		 fflush(stdout);

	  if(break_limit) {
		printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **\n");
		exit(1);
	  }

      return 1;
}

void print_simulation_time()
{
   time_t current_time, difference, d, h, m, s;
   current_time = time((time_t *)NULL);
   difference = MAX(current_time - g_simulation_starttime, 1);

   d = difference/(3600*24);
   h = difference/3600 - 24*d;
   m = difference/60 - 60*(h + 24*d);
   s = difference - 60*(m + 60*(h + 24*d));

   fflush(stderr);
   printf("\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
          (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference );
   printf("gpgpu_simulation_rate = %u (inst/sec)\n", (unsigned)(g_the_gpu->gpu_tot_sim_insn / difference) );
   printf("gpgpu_simulation_rate = %u (cycle/sec)\n", (unsigned)(gpu_tot_sim_cycle / difference) );
   fflush(stdout);
}


