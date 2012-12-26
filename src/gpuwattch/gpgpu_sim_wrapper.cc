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

#include "gpgpu_sim_wrapper.h"
#define SP_BASE_POWER 0
#define SFU_BASE_POWER  0
#define NUM_COMPONENTS_MODELLED 18
#define NUM_PERFORMANCE_COUNTERS 30



static const char * pwr_cmp_label[] = {"IBP,", "ICP,", "DCP,", "TCP,", "CCP,", "SHRDP,", "RFP,", "SPP,",
								"SFUP,", "FPUP,", "SCHEDP,", "L2CP,", "MCP,", "NOCP,", "DRAMP,",
								"PIPEP,", "IDLE_COREP,", "CONST_DYNAMICP"};

enum pwr_cmp_t {
   IBP=0,
   ICP,
   DCP,
   TCP,
   CCP,
   SHRDP,
   RFP,
   SPP,
   SFUP,
   FPUP,
   SCHEDP,
   L2CP,
   MCP,
   NOCP,
   DRAMP,
   PIPEP,
   IDLE_COREP,
   CONST_DYNAMICP
};


gpgpu_sim_wrapper::gpgpu_sim_wrapper( bool power_simulation_enabled, char* xmlfile) {
	   count=0;
	   gcount=0;

	   gpu_avg_power=0;
	   gpu_tot_avg_power=0;

	   gpu_max_power=0;
	   gpu_tot_max_power=0;

	   num_pwr_cmps=NUM_COMPONENTS_MODELLED;
	   num_per_counts=NUM_PERFORMANCE_COUNTERS;
	   pavg=0;
	   perf_count=(double *)calloc(num_per_counts,sizeof(double));
	   initpower_coeff=(double *)calloc(num_per_counts,sizeof(double));
	   effpower_coeff=(double *)calloc(num_per_counts,sizeof(double));
	   perf_count_avg=(double *)calloc(num_per_counts,sizeof(double));
	   perf_count_min=(double *)calloc(num_per_counts,sizeof(double));
	   perf_count_max=(double *)calloc(num_per_counts,sizeof(double));
	   pwr_cmp=(double *)calloc(num_pwr_cmps,sizeof(double));
	   pwr_cmp_avg=(double *)calloc(num_pwr_cmps,sizeof(double));
	   pwr_cmp_min=(double *)calloc(num_pwr_cmps,sizeof(double));
	   pwr_cmp_max=(double *)calloc(num_pwr_cmps,sizeof(double));
	   const_dynamic_power=0;
	   gpu_min_power=0;
	   gpu_tot_min_power=0;
	   proc_power=0;

	   g_power_filename = NULL;
	   g_power_trace_filename = NULL;
	   g_metric_trace_filename = NULL;
	   g_steady_state_tracking_filename = NULL;
	   xml_filename= xmlfile;
	   g_power_simulation_enabled= power_simulation_enabled;
	   g_power_trace_enabled= false;
	   g_steady_power_levels_enabled= false;
	   g_power_trace_zlevel= 0;
	   g_power_per_cycle_dump= false;
	   gpu_steady_power_deviation= 0;
	   gpu_steady_min_period= 0;

	   gpu_stat_sample_freq=0;
	   p=new ParseXML();
	   if (g_power_simulation_enabled){
	       p->parse(xml_filename);
	   }
	   proc = new Processor(p);
	   power_trace_file = NULL;
	   metric_trace_file = NULL;
	   steady_state_tacking_file = NULL;
	   has_written_avg=false;
	   init_inst_val=false;

}

gpgpu_sim_wrapper::~gpgpu_sim_wrapper() { }

bool gpgpu_sim_wrapper::sanity_check(double a, double b)
{
	if (b == 0)
		return (abs(a-b)<0.00001);
	else
		return (abs(a-b)/abs(b)<0.00001);

	return false;
}
void gpgpu_sim_wrapper::init_mcpat(char* xmlfile, char* powerfilename, char* power_trace_filename,char* metric_trace_filename,
								   char * steady_state_filename, bool power_sim_enabled,bool trace_enabled,
								   bool steady_state_enabled,bool power_per_cycle_dump,double steady_power_deviation,
								   double steady_min_period, int zlevel, double init_val,int stat_sample_freq ){
	// Write File Headers for (-metrics trace, -power trace)

    count=0;
    gpu_avg_power=0;
    gpu_max_power=0;

    pavg=0;
    gpu_min_power=0;

   static bool mcpat_init=true;

   // initialize file name if it is not set
   time_t curr_time;
   time(&curr_time);
   char *date = ctime(&curr_time);
   char *s = date;
   while (*s) {
       if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
       if (*s == '\n' || *s == '\r' ) *s = 0;
       s++;
   }


   if(mcpat_init){
	   g_power_filename = powerfilename;
	   g_power_trace_filename =power_trace_filename;
	   g_metric_trace_filename = metric_trace_filename;
	   g_steady_state_tracking_filename = steady_state_filename;
	   xml_filename=xmlfile;
	   g_power_simulation_enabled=power_sim_enabled;
	   g_power_trace_enabled=trace_enabled;
	   g_steady_power_levels_enabled=steady_state_enabled;
	   g_power_trace_zlevel=zlevel;
	   g_power_per_cycle_dump=power_per_cycle_dump;
	   gpu_steady_power_deviation=steady_power_deviation;
	   gpu_steady_min_period=steady_min_period;

	   gpu_stat_sample_freq=stat_sample_freq;

	   //p->sys.total_cycles=gpu_stat_sample_freq*4;
	   p->sys.total_cycles=gpu_stat_sample_freq;
	   power_trace_file = NULL;
	   metric_trace_file = NULL;
	   steady_state_tacking_file = NULL;

	   if (g_power_trace_enabled ){
		   power_trace_file = gzopen(g_power_trace_filename, (mcpat_init)? "w" : "a");
		   metric_trace_file = gzopen(g_metric_trace_filename, (mcpat_init)? "w" : "a");
		   if ((power_trace_file == NULL) || (metric_trace_file == NULL)) {
			   printf("error - could not open trace files \n");
			   exit(1);
		   }
		   gzsetparams(power_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);

		   gzprintf(power_trace_file,"power,");
		   for(unsigned i=0; i<num_pwr_cmps; i++){
			   gzprintf(power_trace_file,pwr_cmp_label[i]);
		   }
		   gzprintf(power_trace_file,"\n");

		   gzsetparams(metric_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);
		   for(unsigned i=0; i<num_per_counts; i++){
			   gzprintf(metric_trace_file,perf_count_label[i]);
		   }
		   gzprintf(metric_trace_file,"\n");

		   gzclose(power_trace_file);
		   gzclose(metric_trace_file);
	   }
	   if(g_steady_power_levels_enabled){
		   steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, (mcpat_init)? "w" : "a");
		   if ((steady_state_tacking_file == NULL)) {
			   printf("error - could not open trace files \n");
			   exit(1);
		   }
		   gzsetparams(steady_state_tacking_file,g_power_trace_zlevel, Z_DEFAULT_STRATEGY);
		   gzprintf(steady_state_tacking_file,"start,end,power,IPC,");
		   for(unsigned i=0; i<num_per_counts; i++){
			   gzprintf(steady_state_tacking_file,perf_count_label[i]);
		   }
		   gzprintf(steady_state_tacking_file,"\n");

		   gzclose(steady_state_tacking_file);
	   }

	   mcpat_init = false;
	   has_written_avg=false;
	   powerfile.open(g_power_filename);

   }
   sample_val = 0;
   init_inst_val=init_val;//gpu_tot_sim_insn+gpu_sim_insn;

}

void gpgpu_sim_wrapper::set_inst_power(bool clk_gated_lanes, double tot_cycles, double busy_cycles, double tot_inst, double int_inst, double fp_inst, double load_inst, double store_inst, double committed_inst)
{
	p->sys.core[0].gpgpu_clock_gated_lanes = clk_gated_lanes;
	p->sys.core[0].total_cycles = tot_cycles;
	p->sys.core[0].busy_cycles = busy_cycles;
	p->sys.core[0].total_instructions  = tot_inst * p->sys.scaling_coefficients[TOT_INST];
	p->sys.core[0].int_instructions    = int_inst * p->sys.scaling_coefficients[FP_INT];
	p->sys.core[0].fp_instructions     = fp_inst  * p->sys.scaling_coefficients[FP_INT];
	p->sys.core[0].load_instructions  = load_inst;
	p->sys.core[0].store_instructions = store_inst;
	p->sys.core[0].committed_instructions = committed_inst;
	perf_count[FP_INT]=int_inst+fp_inst;
	perf_count[TOT_INST]=tot_inst;
}

void gpgpu_sim_wrapper::set_regfile_power(double reads, double writes,double ops)
{
	p->sys.core[0].int_regfile_reads = reads * p->sys.scaling_coefficients[REG_RD];
	p->sys.core[0].int_regfile_writes = writes * p->sys.scaling_coefficients[REG_WR];
	p->sys.core[0].non_rf_operands =  ops *p->sys.scaling_coefficients[NON_REG_OPs];
	perf_count[REG_RD]=reads;
	perf_count[REG_WR]=writes;
	perf_count[NON_REG_OPs]=ops;



}

void gpgpu_sim_wrapper::set_icache_power(double hits, double misses)
{
	p->sys.core[0].icache.read_accesses = hits * p->sys.scaling_coefficients[IC_H]+misses * p->sys.scaling_coefficients[IC_M];
	p->sys.core[0].icache.read_misses = misses * p->sys.scaling_coefficients[IC_M];
	perf_count[IC_H]=hits;
	perf_count[IC_M]=misses;


}

void gpgpu_sim_wrapper::set_ccache_power(double hits, double misses)
{
	p->sys.core[0].ccache.read_accesses = hits * p->sys.scaling_coefficients[CC_H]+misses * p->sys.scaling_coefficients[CC_M];
	p->sys.core[0].ccache.read_misses = misses * p->sys.scaling_coefficients[CC_M];
	perf_count[CC_H]=hits;
	perf_count[CC_M]=misses;
	// TODO: coalescing logic is counted as part of the caches power (this is not valid for no-caches architectures)

}

void gpgpu_sim_wrapper::set_tcache_power(double hits, double misses)
{
	p->sys.core[0].tcache.read_accesses = hits * p->sys.scaling_coefficients[TC_H]+misses * p->sys.scaling_coefficients[TC_M];
	p->sys.core[0].tcache.read_misses = misses;
	perf_count[TC_H]=hits;
	perf_count[TC_M]=misses;
	// TODO: coalescing logic is counted as part of the caches power (this is not valid for no-caches architectures)
}

void gpgpu_sim_wrapper::set_shrd_mem_power(double accesses)
{
	p->sys.core[0].sharedmemory.read_accesses = accesses * p->sys.scaling_coefficients[SHRD_ACC];
	perf_count[SHRD_ACC]=accesses;


}

void gpgpu_sim_wrapper::set_l1cache_power(double read_hits, double read_misses, double write_hits, double write_misses)
{
	p->sys.core[0].dcache.read_accesses = read_hits * p->sys.scaling_coefficients[DC_RH] +read_misses * p->sys.scaling_coefficients[DC_RM];
	p->sys.core[0].dcache.read_misses =  read_misses * p->sys.scaling_coefficients[DC_RM];
	p->sys.core[0].dcache.write_accesses = write_hits * p->sys.scaling_coefficients[DC_WH]+write_misses * p->sys.scaling_coefficients[DC_WM];
	p->sys.core[0].dcache.write_misses = write_misses * p->sys.scaling_coefficients[DC_WM];
	perf_count[DC_RH]=read_hits;
	perf_count[DC_RM]=read_misses;
	perf_count[DC_WH]=write_hits;
	perf_count[DC_WM]=write_misses;
	// TODO: coalescing logic is counted as part of the caches power (this is not valid for no-caches architectures)



}

void gpgpu_sim_wrapper::set_l2cache_power(double read_hits, double read_misses, double write_hits, double write_misses)
{
	p->sys.l2.total_accesses = read_hits* p->sys.scaling_coefficients[L2_RH]+read_misses * p->sys.scaling_coefficients[L2_RM]+ write_hits * p->sys.scaling_coefficients[L2_WH]+write_misses  * p->sys.scaling_coefficients[L2_WM];
	p->sys.l2.read_accesses = read_hits* p->sys.scaling_coefficients[L2_RH]+read_misses* p->sys.scaling_coefficients[L2_RM];
	p->sys.l2.write_accesses = write_hits * p->sys.scaling_coefficients[L2_WH]+write_misses * p->sys.scaling_coefficients[L2_WM];
	p->sys.l2.read_hits = read_hits * p->sys.scaling_coefficients[L2_RH];
	p->sys.l2.read_misses = read_misses  * p->sys.scaling_coefficients[L2_RM];
	p->sys.l2.write_hits =write_hits * p->sys.scaling_coefficients[L2_WH];
	p->sys.l2.write_misses = write_misses * p->sys.scaling_coefficients[L2_WM];
	perf_count[L2_RH]=read_hits;
	perf_count[L2_RM]=read_misses;
	perf_count[L2_WH]=write_hits;
	perf_count[L2_WM]=write_misses;
}

void gpgpu_sim_wrapper::set_idle_core_power(double num_idle_core)
{
	p->sys.num_idle_cores = num_idle_core;
	perf_count[IDLE_CORE_N]=num_idle_core;
}

void gpgpu_sim_wrapper::set_duty_cycle_power(double duty_cycle)
{
	p->sys.core[0].pipeline_duty_cycle = duty_cycle  * p->sys.scaling_coefficients[PIPE_A];
	perf_count[PIPE_A]=duty_cycle;

}

void gpgpu_sim_wrapper::set_mem_ctrl_power(double reads, double writes, double dram_precharge)
{
	p->sys.mc.memory_accesses = reads  * p->sys.scaling_coefficients[MEM_RD]+ writes * p->sys.scaling_coefficients[MEM_WR];
	p->sys.mc.memory_reads = reads *p->sys.scaling_coefficients[MEM_RD];
	p->sys.mc.memory_writes = writes*p->sys.scaling_coefficients[MEM_WR];
	p->sys.mc.dram_pre = dram_precharge*p->sys.scaling_coefficients[MEM_PRE];
	perf_count[MEM_RD]=reads;
	perf_count[MEM_WR]=writes;
	perf_count[MEM_PRE]=dram_precharge;

}

void gpgpu_sim_wrapper::set_exec_unit_power(double fpu_accesses, double ialu_accesses, double sfu_accesses)
{
	p->sys.core[0].fpu_accesses = fpu_accesses*p->sys.scaling_coefficients[FPU_ACC];
    //Integer ALU (not present in Tesla)
	p->sys.core[0].ialu_accesses = ialu_accesses*p->sys.scaling_coefficients[SP_ACC];
	//Sfu accesses
	p->sys.core[0].mul_accesses = sfu_accesses*p->sys.scaling_coefficients[SFU_ACC];

	perf_count[SP_ACC]=ialu_accesses;
	perf_count[SFU_ACC]=sfu_accesses;
	perf_count[FPU_ACC]=fpu_accesses;


}

void gpgpu_sim_wrapper::set_active_lanes_power(double sp_avg_active_lane, double sfu_avg_active_lane)
{
	p->sys.core[0].sp_average_active_lanes = sp_avg_active_lane;
	p->sys.core[0].sfu_average_active_lanes = sfu_avg_active_lane;
}

void gpgpu_sim_wrapper::set_NoC_power(double noc_tot_reads,double noc_tot_writes )
{
	p->sys.NoC[0].total_accesses = noc_tot_reads * p->sys.scaling_coefficients[NOC_A] + noc_tot_writes * p->sys.scaling_coefficients[NOC_A];
	perf_count[NOC_A]=noc_tot_reads+noc_tot_writes;
}


void gpgpu_sim_wrapper::power_metrics_calculations()
{
    gcount++;
    count++;
	gpu_avg_power=(gpu_avg_power+ proc->rt_power.readOp.dynamic);
	for(unsigned ind=0; ind<num_pwr_cmps; ++ind){
		pwr_cmp_avg[ind] += (double)pwr_cmp[ind];
	}

	for(unsigned ind=0; ind<num_per_counts; ++ind){
		perf_count_avg[ind] += (double)perf_count[ind];
	}



	if(proc->rt_power.readOp.dynamic>gpu_max_power){
		gpu_max_power=proc->rt_power.readOp.dynamic;
		for(unsigned ind=0; ind<num_pwr_cmps; ++ind){
		  pwr_cmp_max[ind] = (double)pwr_cmp[ind];
		}
		for(unsigned ind=0; ind<num_per_counts; ++ind){
			perf_count_max[ind] = perf_count[ind];
		}

	}
	if(proc->rt_power.readOp.dynamic<gpu_min_power ||(gpu_min_power==0) ){
	  gpu_min_power=proc->rt_power.readOp.dynamic;
	  for(unsigned ind=0; ind<num_pwr_cmps; ++ind){
		  pwr_cmp_min[ind] = (double)pwr_cmp[ind];
	  }
	  for(unsigned ind=0; ind<num_per_counts; ++ind){
		  perf_count_min[ind] = perf_count[ind];
	  }
	}

	gpu_tot_avg_power=(gpu_tot_avg_power+ proc->rt_power.readOp.dynamic);
	gpu_tot_max_power=(proc->rt_power.readOp.dynamic>gpu_tot_max_power)? proc->rt_power.readOp.dynamic:gpu_tot_max_power;
	gpu_tot_min_power=((proc->rt_power.readOp.dynamic<gpu_tot_min_power)||(gpu_tot_min_power==0))? proc->rt_power.readOp.dynamic:gpu_tot_min_power;

}


void gpgpu_sim_wrapper::print_trace_files()
{
	open_files();

	for(unsigned i=0; i<num_per_counts; ++i){
		gzprintf(metric_trace_file,"%f,",perf_count[i]);
	}
	gzprintf(metric_trace_file,"\n");

	gzprintf(power_trace_file,"%f,",proc_power);
	for(unsigned i=0; i<num_pwr_cmps; ++i){
		gzprintf(power_trace_file,"%f,",pwr_cmp[i]);
	}
	gzprintf(power_trace_file,"\n");

	close_files();

}

void gpgpu_sim_wrapper::update_coefficients()
{

	initpower_coeff[FP_INT]=proc->cores[0]->get_coefficient_fpint_insts();
	effpower_coeff[FP_INT]=initpower_coeff[FP_INT] * p->sys.scaling_coefficients[FP_INT];

	initpower_coeff[TOT_INST]=proc->cores[0]->get_coefficient_tot_insts();
	effpower_coeff[TOT_INST]=initpower_coeff[TOT_INST] * p->sys.scaling_coefficients[TOT_INST];

	initpower_coeff[REG_RD]=proc->cores[0]->get_coefficient_regreads_accesses()*(proc->cores[0]->exu->rf_fu_clockRate/proc->cores[0]->exu->clockRate);
	initpower_coeff[REG_WR]=proc->cores[0]->get_coefficient_regwrites_accesses()*(proc->cores[0]->exu->rf_fu_clockRate/proc->cores[0]->exu->clockRate);
	initpower_coeff[NON_REG_OPs]=proc->cores[0]->get_coefficient_noregfileops_accesses()*(proc->cores[0]->exu->rf_fu_clockRate/proc->cores[0]->exu->clockRate);
	effpower_coeff[REG_RD]=initpower_coeff[REG_RD]*p->sys.scaling_coefficients[REG_RD];
	effpower_coeff[REG_WR]=initpower_coeff[REG_WR]*p->sys.scaling_coefficients[REG_WR];
	effpower_coeff[NON_REG_OPs]=initpower_coeff[NON_REG_OPs]*p->sys.scaling_coefficients[NON_REG_OPs];

	initpower_coeff[IC_H]=proc->cores[0]->get_coefficient_icache_hits();
	initpower_coeff[IC_M]=proc->cores[0]->get_coefficient_icache_misses();
	effpower_coeff[IC_H]=initpower_coeff[IC_H]*p->sys.scaling_coefficients[IC_H];
	effpower_coeff[IC_M]=initpower_coeff[IC_M]*p->sys.scaling_coefficients[IC_M];

	initpower_coeff[CC_H]=(proc->cores[0]->get_coefficient_ccache_readhits()+proc->get_coefficient_readcoalescing());
	initpower_coeff[CC_M]=(proc->cores[0]->get_coefficient_ccache_readmisses()+proc->get_coefficient_readcoalescing());
	effpower_coeff[CC_H]=initpower_coeff[CC_H]*p->sys.scaling_coefficients[CC_H];
	effpower_coeff[CC_M]=initpower_coeff[CC_M]*p->sys.scaling_coefficients[CC_M];

	initpower_coeff[TC_H]=(proc->cores[0]->get_coefficient_tcache_readhits()+proc->get_coefficient_readcoalescing());
	initpower_coeff[TC_M]=(proc->cores[0]->get_coefficient_tcache_readmisses()+proc->get_coefficient_readcoalescing());
	effpower_coeff[TC_H]=initpower_coeff[TC_H]*p->sys.scaling_coefficients[TC_H];
	effpower_coeff[TC_M]=initpower_coeff[TC_M]*p->sys.scaling_coefficients[TC_M];

	initpower_coeff[SHRD_ACC]=proc->cores[0]->get_coefficient_sharedmemory_readhits();
	effpower_coeff[SHRD_ACC]=initpower_coeff[SHRD_ACC]*p->sys.scaling_coefficients[SHRD_ACC];

	initpower_coeff[DC_RH]=(proc->cores[0]->get_coefficient_dcache_readhits() + proc->get_coefficient_readcoalescing());
	initpower_coeff[DC_RM]=(proc->cores[0]->get_coefficient_dcache_readmisses() + proc->get_coefficient_readcoalescing());
	initpower_coeff[DC_WH]=(proc->cores[0]->get_coefficient_dcache_writehits() + proc->get_coefficient_writecoalescing());
	initpower_coeff[DC_WM]=(proc->cores[0]->get_coefficient_dcache_writemisses() + proc->get_coefficient_writecoalescing());
	effpower_coeff[DC_RH]=initpower_coeff[DC_RH]*p->sys.scaling_coefficients[DC_RH];
	effpower_coeff[DC_RM]=initpower_coeff[DC_RM]*p->sys.scaling_coefficients[DC_RM];
	effpower_coeff[DC_WH]=initpower_coeff[DC_WH]*p->sys.scaling_coefficients[DC_WH];
	effpower_coeff[DC_WM]=initpower_coeff[DC_WM]*p->sys.scaling_coefficients[DC_WM];

	initpower_coeff[L2_RH]=proc->get_coefficient_l2_read_hits();
	initpower_coeff[L2_RM]=proc->get_coefficient_l2_read_misses();
	initpower_coeff[L2_WH]=proc->get_coefficient_l2_write_hits();
	initpower_coeff[L2_WM]=proc->get_coefficient_l2_write_misses();
	effpower_coeff[L2_RH]=initpower_coeff[L2_RH]*p->sys.scaling_coefficients[L2_RH];
	effpower_coeff[L2_RM]=initpower_coeff[L2_RM]*p->sys.scaling_coefficients[L2_RM];
	effpower_coeff[L2_WH]=initpower_coeff[L2_WH]*p->sys.scaling_coefficients[L2_WH];
	effpower_coeff[L2_WM]=initpower_coeff[L2_WM]*p->sys.scaling_coefficients[L2_WM];

	initpower_coeff[IDLE_CORE_N]=p->sys.idle_core_power * proc->cores[0]->executionTime;
	effpower_coeff[IDLE_CORE_N]=initpower_coeff[IDLE_CORE_N]*p->sys.scaling_coefficients[IDLE_CORE_N];

	initpower_coeff[PIPE_A]=proc->cores[0]->get_coefficient_duty_cycle();
	effpower_coeff[PIPE_A]=initpower_coeff[PIPE_A]*p->sys.scaling_coefficients[PIPE_A];

	initpower_coeff[MEM_RD]=proc->get_coefficient_mem_reads();
	initpower_coeff[MEM_WR]=proc->get_coefficient_mem_writes();
	initpower_coeff[MEM_PRE]=proc->get_coefficient_mem_pre();
	effpower_coeff[MEM_RD]=initpower_coeff[MEM_RD]*p->sys.scaling_coefficients[MEM_RD];
	effpower_coeff[MEM_WR]=initpower_coeff[MEM_WR]*p->sys.scaling_coefficients[MEM_WR];
	effpower_coeff[MEM_PRE]=initpower_coeff[MEM_PRE]*p->sys.scaling_coefficients[MEM_PRE];

	initpower_coeff[SP_ACC]=proc->cores[0]->get_coefficient_ialu_accesses()*(proc->cores[0]->exu->rf_fu_clockRate/proc->cores[0]->exu->clockRate);;
	initpower_coeff[SFU_ACC]=proc->cores[0]->get_coefficient_sfu_accesses();
	initpower_coeff[FPU_ACC]=proc->cores[0]->get_coefficient_fpu_accesses();

	effpower_coeff[SP_ACC]=initpower_coeff[SP_ACC]*p->sys.scaling_coefficients[SP_ACC];
	effpower_coeff[SFU_ACC]=initpower_coeff[SFU_ACC]*p->sys.scaling_coefficients[SFU_ACC];
	effpower_coeff[FPU_ACC]=initpower_coeff[FPU_ACC]*p->sys.scaling_coefficients[FPU_ACC];

	initpower_coeff[NOC_A]=proc->get_coefficient_noc_accesses();
	effpower_coeff[NOC_A]=initpower_coeff[NOC_A]*p->sys.scaling_coefficients[NOC_A];

	const_dynamic_power=proc->get_const_dynamic_power()/(proc->cores[0]->executionTime);

	for(unsigned i=0; i<num_per_counts; i++){
		initpower_coeff[i]/=(proc->cores[0]->executionTime);
		effpower_coeff[i]/=(proc->cores[0]->executionTime);
	}
}

void gpgpu_sim_wrapper::update_components_power()
{

	update_coefficients();

	proc_power=proc->rt_power.readOp.dynamic;

	pwr_cmp[IBP]=(proc->cores[0]->ifu->IB->rt_power.readOp.dynamic
			    +proc->cores[0]->ifu->IB->rt_power.writeOp.dynamic
			    +proc->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic
			    +proc->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic
			    +proc->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic)/(proc->cores[0]->executionTime);

	pwr_cmp[ICP]=proc->cores[0]->ifu->icache.rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[DCP]=proc->cores[0]->lsu->dcache.rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[TCP]=proc->cores[0]->lsu->tcache.rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[CCP]=proc->cores[0]->lsu->ccache.rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[SHRDP]=proc->cores[0]->lsu->sharedmemory.rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[RFP]=(proc->cores[0]->exu->rfu->rt_power.readOp.dynamic/(proc->cores[0]->executionTime))
			   *(proc->cores[0]->exu->rf_fu_clockRate/proc->cores[0]->exu->clockRate);

	pwr_cmp[SPP]=(proc->cores[0]->exu->exeu->rt_power.readOp.dynamic/(proc->cores[0]->executionTime))
			   *(proc->cores[0]->exu->rf_fu_clockRate/proc->cores[0]->exu->clockRate);

	pwr_cmp[SFUP]=(proc->cores[0]->exu->mul->rt_power.readOp.dynamic/(proc->cores[0]->executionTime));

	pwr_cmp[FPUP]=(proc->cores[0]->exu->fp_u->rt_power.readOp.dynamic/(proc->cores[0]->executionTime));

	pwr_cmp[SCHEDP]=proc->cores[0]->exu->scheu->rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[L2CP]=(proc->XML->sys.number_of_L2s>0)? proc->l2array[0]->rt_power.readOp.dynamic/(proc->cores[0]->executionTime):0;

	pwr_cmp[MCP]=(proc->mc->rt_power.readOp.dynamic-proc->mc->dram->rt_power.readOp.dynamic)/(proc->cores[0]->executionTime);

	pwr_cmp[NOCP]=proc->nocs[0]->rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[DRAMP]=proc->mc->dram->rt_power.readOp.dynamic/(proc->cores[0]->executionTime);

	pwr_cmp[PIPEP]=proc->cores[0]->Pipeline_energy/(proc->cores[0]->executionTime);

	pwr_cmp[IDLE_COREP]=proc->cores[0]->IdleCoreEnergy/(proc->cores[0]->executionTime);

	//this constant dynamic part is estimated via regression model
	pwr_cmp[CONST_DYNAMICP]=0;
	pwr_cmp[CONST_DYNAMICP]=p->sys.scaling_coefficients[CONST_DYNAMICN]-proc->get_const_dynamic_power()/(proc->cores[0]->executionTime);
	pwr_cmp[CONST_DYNAMICP]= (pwr_cmp[CONST_DYNAMICP]>0)? pwr_cmp[CONST_DYNAMICP]:0;

	proc_power+=pwr_cmp[CONST_DYNAMICP];

	double sum_pwr_cmp=0;
	for(unsigned i=0; i<num_pwr_cmps; i++){
		sum_pwr_cmp+=pwr_cmp[i];
	}
	bool check=false;
	check=sanity_check(sum_pwr_cmp,proc_power);
	assert("Total Power does not equal the sum of the components\n" && (check));

}

void gpgpu_sim_wrapper::compute()
{
	proc->compute();
}
void gpgpu_sim_wrapper::print_power_kernel_stats(double gpu_sim_cycle, double gpu_tot_sim_cycle, double init_value)
{
	   detect_print_steady_state(1,init_value);
	   if(g_power_simulation_enabled){
		   powerfile<<"Kernel Average Power Data:"<<std::endl;
		   powerfile<<"gpu_avg_power = "<< gpu_avg_power/ count<<std::endl;

		   for(unsigned i=0; i<num_pwr_cmps; ++i){
				powerfile<<"gpu_avg_"<<pwr_cmp_label[i]<<" = "<<pwr_cmp_avg[i]/count<<std::endl;
		   }
		   for(unsigned i=0; i<num_per_counts; ++i){
				powerfile<<"gpu_avg_"<<perf_count_label[i]<<" = "<<perf_count_avg[i]/count<<std::endl;
		   }

		   powerfile<<std::endl<<"Kernel Maximum Power Data:"<<std::endl;
		   powerfile<<"gpu_max_power = "<< gpu_max_power<<std::endl;
		   for(unsigned i=0; i<num_pwr_cmps; ++i){
			   powerfile<<"gpu_max_"<<pwr_cmp_label[i]<<" = "<<pwr_cmp_max[i]<<std::endl;
		   }
		   for(unsigned i=0; i<num_per_counts; ++i){
				powerfile<<"gpu_max_"<<perf_count_label[i]<<" = "<<perf_count_max[i]<<std::endl;
		   }

		   powerfile<<std::endl<<"Kernel Minimum Power Data:"<<std::endl;
		   powerfile<<"gpu_min_power = "<< gpu_min_power<<std::endl;
		   for(unsigned i=0; i<num_pwr_cmps; ++i){
			   powerfile<<"gpu_min_"<<pwr_cmp_label[i]<<" = "<<pwr_cmp_min[i]<<std::endl;
		   }
		   for(unsigned i=0; i<num_per_counts; ++i){
				powerfile<<"gpu_min_"<<perf_count_label[i]<<" = "<<perf_count_min[i]<<std::endl;
		   }

		   powerfile<<std::endl<<"Accumulative Power Statistics Over Previous Kernels:"<<std::endl;
		   powerfile<<"gpu_tot_avg_power = "<< gpu_tot_avg_power/gcount<<std::endl;
		   powerfile<<"gpu_tot_max_power = "<<gpu_tot_max_power<<std::endl;
		   powerfile<<"gpu_tot_min_power = "<<gpu_tot_min_power<<std::endl;
		   powerfile<<std::endl<<std::endl;
		   powerfile.flush();
	   }

}
void gpgpu_sim_wrapper::dump()
{
	if(g_power_per_cycle_dump)
		proc->displayEnergy(2,5);
}

void gpgpu_sim_wrapper::detect_print_steady_state(int position, double init_val)
{
    if(g_steady_power_levels_enabled)
        steady_state_tacking_file = gzopen(g_steady_state_tracking_filename,"a");

	// Calculating Average
	if(position==0){
		if(g_steady_power_levels_enabled){

		    if(samples.size() == 0){
				// First sample
				sample_start = gcount;
				sample_val = proc->rt_power.readOp.dynamic;
				init_inst_val=init_val;//gpu_tot_sim_insn+gpu_sim_insn;
				samples.push_back(proc->rt_power.readOp.dynamic);
				assert(samples_counter.size() == 0);
				assert(pwr_counter.size() == 0);

				for(unsigned i=0; i<(num_per_counts); ++i){
					samples_counter.push_back(perf_count[i]);
				}

				for(unsigned i=0; i<(num_pwr_cmps); ++i){
					pwr_counter.push_back(pwr_cmp[i]);
				}
				assert(pwr_counter.size() == (double)num_pwr_cmps);
				assert(samples_counter.size() == (double)num_per_counts);
			}else{
				// Get current average
				double temp_avg = sample_val / (double)samples.size() ;
				double temp_ipc = (init_val-init_inst_val)/ (double) (samples.size()*gpu_stat_sample_freq);

				if( abs(proc->rt_power.readOp.dynamic-temp_avg) < gpu_steady_power_deviation){ // Value is within threshold
					sample_val += proc->rt_power.readOp.dynamic;
					samples.push_back(proc->rt_power.readOp.dynamic);
					for(unsigned i=0; i<(num_per_counts); ++i){
						samples_counter.at(i) += perf_count[i];
					}

					for(unsigned i=0; i<(num_pwr_cmps); ++i){
						pwr_counter.at(i) += pwr_cmp[i];
					}

				}else{	// Value exceeds threshold, not considered steady state

					if(samples.size() > gpu_steady_min_period){ // If steady state occurred for some time, print to file
						has_written_avg = true;
						gzprintf(steady_state_tacking_file,"%u,%d,%f,%f,",sample_start,gcount,temp_avg,temp_ipc);
						for(unsigned i=0; i<num_per_counts; ++i){
							gzprintf(steady_state_tacking_file,"%f,", samples_counter.at(i)/((double)samples.size()));
						}
						gzprintf(steady_state_tacking_file,"\n");
					}

					// Clear state
					sample_start = 0;
					sample_val = 0;
					init_inst_val=init_val;//gpu_tot_sim_insn+gpu_sim_insn;
					samples.clear();
					samples_counter.clear();
					pwr_counter.clear();
					assert(samples.size() == 0);
				}
			}
		}
	}else{
		   if(g_power_simulation_enabled && g_steady_power_levels_enabled){
			  double temp_avg = sample_val / (double)samples.size() ;
			  double temp_ipc = (init_val-init_inst_val)/ (double) (samples.size()*gpu_stat_sample_freq);
		  	  if((samples.size() > gpu_steady_min_period)){ // If steady state occurred for some time, print to file
		  		gzprintf(steady_state_tacking_file,"%u,%d,%f,%f,",sample_start,gcount,temp_avg,temp_ipc);
				for(unsigned i=0; i<num_per_counts; ++i){
					gzprintf(steady_state_tacking_file,"%f,", samples_counter.at(i)/((double)samples.size()));
				}
				gzprintf(steady_state_tacking_file,"\n");
		  	  }else{
		  	    if(!has_written_avg)
		  	        gzprintf(steady_state_tacking_file,"ERROR! Not enough steady state points to generate averag\n");
		  	  }
				sample_start = 0;
				sample_val = 0;
				init_inst_val=init_val;//gpu_tot_sim_insn+gpu_sim_insn;
				samples.clear();
				samples_counter.clear();
				pwr_counter.clear();
				assert(samples.size() == 0);
		   }

	}

	if(g_steady_power_levels_enabled)
	    gzclose(steady_state_tacking_file);

}

void gpgpu_sim_wrapper::open_files()
{
    if(g_power_simulation_enabled){
        if (g_power_trace_enabled ){
            power_trace_file = gzopen(g_power_trace_filename,  "a");
            metric_trace_file = gzopen(g_metric_trace_filename, "a");
        }
    }
}
void gpgpu_sim_wrapper::close_files()
{
    if(g_power_simulation_enabled){
  	  if(g_power_trace_enabled){
  		  gzclose(power_trace_file);
  		  gzclose(metric_trace_file);
  	  }
  	 }

}
