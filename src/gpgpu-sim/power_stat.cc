// Copyright (c) 2009-2011, Tor M. Aamodt, Ahmed El-Shafiey, Tayler Hetherington
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

#include "../abstract_hardware_model.h"
#include "power_stat.h"
#include "gpu-sim.h"
#include "gpu-misc.h"
#include "shader.h"
#include "mem_fetch.h"
#include "stat-tool.h"
#include "../cuda-sim/ptx-stats.h"
#include "visualizer.h"
#include "dram.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>



power_mem_stat_t::power_mem_stat_t(const struct memory_config *mem_config, const struct shader_core_config *shdr_config, memory_stats_t *mem_stats, shader_core_stats *shdr_stats){
	   assert( mem_config->m_valid );
	   m_mem_stats = mem_stats;
	   m_config = mem_config;
	   m_core_stats = shdr_stats;
	   m_core_config = shdr_config;

	   init();
}

void power_mem_stat_t::init(){
	inst_c_read_access[0] = m_core_stats->inst_c_read_access;
	inst_c_read_miss[0] = m_core_stats->inst_c_read_miss;
	const_c_read_access[0] = m_core_stats->const_c_read_access;
	const_c_read_miss[0] = m_core_stats->const_c_read_miss;
	text_c_read_access[0] = m_core_stats->text_c_read_access;
	text_c_read_miss[0] = m_core_stats->text_c_read_miss;
	l1d_read_access[0] = m_core_stats->l1d_read_access;
	l1d_read_miss[0] = m_core_stats->l1d_read_miss;
	l1d_write_access[0] = m_core_stats->l1d_write_access;
	l1d_write_miss[0] = m_core_stats->l1d_write_miss;

	shmem_read_access[0] = m_core_stats->gpgpu_n_shmem_bank_access; 	// Shared memory access

	inst_c_read_access[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	inst_c_read_miss[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	const_c_read_access[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	const_c_read_miss[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	text_c_read_access[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	text_c_read_miss[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	l1d_read_access[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	l1d_read_miss[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	l1d_write_access[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));
	l1d_write_miss[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));

	shmem_read_access[1] = (unsigned *)calloc(m_core_config->num_shader(),sizeof(unsigned));

	// Low-level DRAM/L2-cache stats
    for(unsigned i=0; i<2; ++i){
    	n_l2_read_access[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
        n_l2_read_miss[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
        n_l2_write_access[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
        n_l2_write_miss[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_cmd[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_activity[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_nop[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_act[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_pre[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_rd[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_wr[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));
		n_req[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned));

	    // Interconnect stats
	    n_mem_to_simt[i] = (unsigned *)calloc(m_config->m_n_mem,sizeof(unsigned)); // Counted at memory partition
	    n_simt_to_mem[i] = (unsigned *)calloc(m_core_config->n_simt_clusters,sizeof(unsigned)); // Counted at SM
    }
}

void power_mem_stat_t::save_stats(){
	for(unsigned i=0; i<m_core_config->num_shader(); ++i){
		inst_c_read_access[1][i] = inst_c_read_access[0][i] ;
		inst_c_read_miss[1][i] = inst_c_read_miss[0][i] ;
		const_c_read_access[1][i] = const_c_read_access[0][i] ;
		const_c_read_miss[1][i] = const_c_read_miss[0][i] ;
		text_c_read_access[1][i] = text_c_read_access[0][i] ;
		text_c_read_miss[1][i] = text_c_read_miss[0][i] ;
		l1d_read_access[1][i] = l1d_read_access[0][i] ;
		l1d_read_miss[1][i] = l1d_read_miss[0][i] ;
		l1d_write_access[1][i] = l1d_write_access[0][i] ;
		l1d_write_miss[1][i] = l1d_write_miss[0][i] ;
		shmem_read_access[1][i] = shmem_read_access[0][i] ; 	// Shared memory access

		n_simt_to_mem[1][i] = n_simt_to_mem[0][i]; // Interconnect
	}

	for(unsigned i=0; i<m_config->m_n_mem; ++i){
    	n_l2_read_access[1][i] = n_l2_read_access[0][i];
        n_l2_read_miss[1][i] = n_l2_read_miss[0][i];
        n_l2_write_access[1][i] = n_l2_write_access[0][i];
        n_l2_write_miss[1][i] = n_l2_write_miss[0][i];
		n_cmd[1][i] = n_cmd[0][i];
		n_activity[1][i] = n_activity[0][i];
		n_nop[1][i] = n_nop[0][i];
		n_act[1][i] = n_act[0][i];
		n_pre[1][i] = n_pre[0][i];
		n_rd[1][i] = n_rd[0][i];
		n_wr[1][i] = n_wr[0][i];
		n_req[1][i] = n_req[0][i];

		n_mem_to_simt[1][i] = n_mem_to_simt[0][i]; // Interconnect
	}
}

void power_mem_stat_t::visualizer_print( gzFile power_visualizer_file ){

}

void power_mem_stat_t::print (FILE *fout) const {
	fprintf(fout, "\n\n==========Power Metrics -- Memory==========\n");
	unsigned total_mem_reads=0;
	unsigned total_mem_writes=0;
	for(unsigned i=0; i<m_config->m_n_mem; ++i){
		total_mem_reads += n_rd[0][i];
		total_mem_writes += n_wr[0][i];
	}
	fprintf(fout, "Total memory controller accesses: %u\n", total_mem_reads+total_mem_writes);
	fprintf(fout, "Total memory controller reads: %u\n", total_mem_reads);
	fprintf(fout, "Total memory controller writes: %u\n", total_mem_writes);
	for(unsigned i=0; i<m_core_config->num_shader(); ++i){
		fprintf(fout, "Shader core %d\n", i);
		fprintf(fout, "\tTotal instruction cache access: %u\n", inst_c_read_access[0][i]);
		fprintf(fout, "\tTotal instruction cache miss: %u\n", inst_c_read_miss[0][i]);
		fprintf(fout, "\tTotal constant cache access: %u\n", const_c_read_access[0][i]);
		fprintf(fout, "\tTotal constant cache miss: %u\n", const_c_read_miss[0][i]);
		fprintf(fout, "\tTotal texture cache access: %u\n", text_c_read_access[0][i]);
		fprintf(fout, "\tTotal texture cache miss: %u\n", text_c_read_miss[0][i]);
		fprintf(fout, "\tTotal l1d read access: %u\n", l1d_read_access[0][i]);
		fprintf(fout, "\tTotal l1d read miss: %u\n", l1d_read_miss[0][i]);
		fprintf(fout, "\tTotal l1d write access: %u\n", l1d_write_access[0][i]);
		fprintf(fout, "\tTotal l1d write miss: %u\n", l1d_write_miss[0][i]);
		fprintf(fout, "\tTotal shared memory access: %u\n", shmem_read_access[0][i]);
	}
}


power_core_stat_t::power_core_stat_t( const struct shader_core_config *shader_config, shader_core_stats *core_stats )
{
     	assert( shader_config->m_valid );
        m_config = shader_config;
        shader_core_power_stats_pod *pod = this;
        memset(pod,0,sizeof(shader_core_power_stats_pod));
        m_core_stats=core_stats;

        init();

}

void power_core_stat_t::visualizer_print( gzFile visualizer_file )
{

}

void power_core_stat_t::print (FILE *fout)
{
	// per core statistics
	fprintf(fout,"Power Metrics: \n");
	for(unsigned i=0; i<m_config->num_shader();i++){
		fprintf(fout,"core %u:\n",i);
		fprintf(fout,"\tpipeline duty cycle =%f\n",m_pipeline_duty_cycle[0][i]);
		fprintf(fout,"\tTotal Deocded Instructions=%u\n",m_num_decoded_insn[0][i]);
		fprintf(fout,"\tTotal FP Deocded Instructions=%u\n",m_num_FPdecoded_insn[0][i]);
		fprintf(fout,"\tTotal INT Deocded Instructions=%u\n",m_num_INTdecoded_insn[0][i]);
		fprintf(fout,"\tTotal LOAD Queued Instructions=%u\n",m_num_loadqueued_insn[0][i]);
		fprintf(fout,"\tTotal STORE Queued Instructions=%u\n",m_num_storequeued_insn[0][i]);
		fprintf(fout,"\tTotal IALU Acesses=%u\n",m_num_ialu_acesses[0][i]);
		fprintf(fout,"\tTotal FP Acesses=%u\n",m_num_fp_acesses[0][i]);
		fprintf(fout,"\tTotal IMUL Acesses=%u\n",m_num_imul_acesses[0][i]);
		fprintf(fout,"\tTotal IMUL24 Acesses=%u\n",m_num_imul24_acesses[0][i]);
		fprintf(fout,"\tTotal IMUL32 Acesses=%u\n",m_num_imul32_acesses[0][i]);
		fprintf(fout,"\tTotal IDIV Acesses=%u\n",m_num_idiv_acesses[0][i]);
		fprintf(fout,"\tTotal FPMUL Acesses=%u\n",m_num_fpmul_acesses[0][i]);
		fprintf(fout,"\tTotal SFU Acesses=%u\n",m_num_trans_acesses[0][i]);
		fprintf(fout,"\tTotal FPDIV Acesses=%u\n",m_num_fpdiv_acesses[0][i]);
		fprintf(fout,"\tTotal SFU Acesses=%u\n",m_num_sfu_acesses[0][i]);
		fprintf(fout,"\tTotal SP Acesses=%u\n",m_num_sp_acesses[0][i]);
		fprintf(fout,"\tTotal MEM Acesses=%u\n",m_num_mem_acesses[0][i]);
		fprintf(fout,"\tTotal SFU Commissions=%u\n",m_num_sfu_committed[0][i]);
		fprintf(fout,"\tTotal SP Commissions=%u\n",m_num_sp_committed[0][i]);
		fprintf(fout,"\tTotal MEM Commissions=%u\n",m_num_mem_committed[0][i]);
		fprintf(fout,"\tTotal REG Reads=%u\n",m_read_regfile_acesses[0][i]);
		fprintf(fout,"\tTotal REG Writes=%u\n",m_write_regfile_acesses[0][i]);
		fprintf(fout,"\tTotal NON REG=%u\n",m_non_rf_operands[0][i]);
	}
}
void power_core_stat_t::init()
{
	m_pipeline_duty_cycle[0]=m_core_stats->m_pipeline_duty_cycle;
	m_num_decoded_insn[0]=m_core_stats->m_num_decoded_insn;
	m_num_FPdecoded_insn[0]=m_core_stats->m_num_FPdecoded_insn;
	m_num_INTdecoded_insn[0]=m_core_stats->m_num_INTdecoded_insn;
	m_num_storequeued_insn[0]=m_core_stats->m_num_storequeued_insn;
	m_num_loadqueued_insn[0]=m_core_stats->m_num_loadqueued_insn;
    m_num_ialu_acesses[0]=m_core_stats->m_num_ialu_acesses;
    m_num_fp_acesses[0]=m_core_stats->m_num_fp_acesses;
    m_num_imul_acesses[0]=m_core_stats->m_num_imul_acesses;
    m_num_imul24_acesses[0]=m_core_stats->m_num_imul24_acesses;
    m_num_imul32_acesses[0]=m_core_stats->m_num_imul32_acesses;
    m_num_fpmul_acesses[0]=m_core_stats->m_num_fpmul_acesses;
    m_num_idiv_acesses[0]=m_core_stats->m_num_idiv_acesses;
    m_num_fpdiv_acesses[0]=m_core_stats->m_num_fpdiv_acesses;
    m_num_sp_acesses[0]=m_core_stats->m_num_sp_acesses;
    m_num_sfu_acesses[0]=m_core_stats->m_num_sfu_acesses;
    m_num_trans_acesses[0]=m_core_stats->m_num_trans_acesses;
    m_num_mem_acesses[0]=m_core_stats->m_num_mem_acesses;
    m_num_sp_committed[0]=m_core_stats->m_num_sp_committed;
    m_num_sfu_committed[0]=m_core_stats->m_num_sfu_committed;
    m_num_mem_committed[0]=m_core_stats->m_num_mem_committed;
    m_read_regfile_acesses[0]=m_core_stats->m_read_regfile_acesses;
    m_write_regfile_acesses[0]=m_core_stats->m_write_regfile_acesses;
    m_non_rf_operands[0]=m_core_stats->m_non_rf_operands;
    m_active_sp_lanes[0]=m_core_stats->m_active_sp_lanes;
    m_active_sfu_lanes[0]=m_core_stats->m_active_sfu_lanes;
    m_num_tex_inst[0]=m_core_stats->m_num_tex_inst;


	m_pipeline_duty_cycle[1]=(float*)calloc(m_config->num_shader(),sizeof(float));
	m_num_decoded_insn[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
	m_num_FPdecoded_insn[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
	m_num_INTdecoded_insn[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
	m_num_storequeued_insn[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
	m_num_loadqueued_insn[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_ialu_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_fp_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_tex_inst[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_imul_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_imul24_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_imul32_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_fpmul_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_idiv_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_fpdiv_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_sp_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_sfu_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_trans_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_mem_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_sp_committed[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_sfu_committed[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_num_mem_committed[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_read_regfile_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_write_regfile_acesses[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_non_rf_operands[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_active_sp_lanes[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
    m_active_sfu_lanes[1]=(unsigned *)calloc(m_config->num_shader(),sizeof(unsigned));
}

void power_core_stat_t::save_stats(){
	for(unsigned i=0; i<m_config->num_shader(); ++i){
		m_pipeline_duty_cycle[1][i]=m_pipeline_duty_cycle[0][i];
		m_num_decoded_insn[1][i]=	m_num_decoded_insn[0][i];
		m_num_FPdecoded_insn[1][i]=m_num_FPdecoded_insn[0][i];
		m_num_INTdecoded_insn[1][i]=m_num_INTdecoded_insn[0][i];
		m_num_storequeued_insn[1][i]=m_num_storequeued_insn[0][i];
		m_num_loadqueued_insn[1][i]=m_num_loadqueued_insn[0][i];
		m_num_ialu_acesses[1][i]=m_num_ialu_acesses[0][i];
		m_num_fp_acesses[1][i]=m_num_fp_acesses[0][i];
		m_num_tex_inst[1][i]=m_num_tex_inst[0][i];
		m_num_imul_acesses[1][i]=m_num_imul_acesses[0][i];
		m_num_imul24_acesses[1][i]=m_num_imul24_acesses[0][i];
		m_num_imul32_acesses[1][i]=m_num_imul32_acesses[0][i];
		m_num_fpmul_acesses[1][i]=m_num_fpmul_acesses[0][i];
		m_num_idiv_acesses[1][i]=m_num_idiv_acesses[0][i];
		m_num_fpdiv_acesses[1][i]=m_num_fpdiv_acesses[0][i];
		m_num_sp_acesses[1][i]=m_num_sp_acesses[0][i];
		m_num_sfu_acesses[1][i]=m_num_sfu_acesses[0][i];
		m_num_trans_acesses[1][i]=m_num_trans_acesses[0][i];
		m_num_mem_acesses[1][i]=m_num_mem_acesses[0][i];
		m_num_sp_committed[1][i]=m_num_sp_committed[0][i];
		m_num_sfu_committed[1][i]=m_num_sfu_committed[0][i];
		m_num_mem_committed[1][i]=m_num_mem_committed[0][i];
		m_read_regfile_acesses[1][i]=m_read_regfile_acesses[0][i];
		m_write_regfile_acesses[1][i]=m_write_regfile_acesses[0][i];
		m_non_rf_operands[1][i]=m_non_rf_operands[0][i];
	    m_active_sp_lanes[1][i]=m_active_sp_lanes[0][i];
	    m_active_sfu_lanes[1][i]=m_active_sfu_lanes[0][i];
	}
}

power_stat_t::power_stat_t( const struct shader_core_config *shader_config,float * average_pipeline_duty_cycle,float *active_sms,shader_core_stats * shader_stats, const struct memory_config *mem_config,memory_stats_t * memory_stats)
{
	assert( shader_config->m_valid );
	assert( mem_config->m_valid );
	pwr_core_stat= new power_core_stat_t(shader_config,shader_stats);
	pwr_mem_stat= new power_mem_stat_t(mem_config,shader_config, memory_stats, shader_stats);
	m_average_pipeline_duty_cycle=average_pipeline_duty_cycle;
	m_active_sms=active_sms;
	m_config = shader_config;
	m_mem_config = mem_config;
}

void power_stat_t::visualizer_print( gzFile visualizer_file )
{
	pwr_core_stat->visualizer_print(visualizer_file);
	pwr_mem_stat->visualizer_print(visualizer_file);
}

void power_stat_t::print (FILE *fout) const
{
	fprintf(fout,"average_pipeline_duty_cycle=%f\n",*m_average_pipeline_duty_cycle);
	pwr_core_stat->print(fout);
	pwr_mem_stat->print(fout);
}

