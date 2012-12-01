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

#ifndef POWER_STAT_H
#define POWER_STAT_H

#include <stdio.h>
#include <zlib.h>
#include "mem_latency_stat.h"
#include "shader.h"
#include "gpu-sim.h"

struct shader_core_power_stats_pod {
	// [0] = Current stat, [1] = last reading
	float *m_pipeline_duty_cycle[2];
    unsigned *m_num_decoded_insn[2]; // number of instructions committed by this shader core
    unsigned *m_num_FPdecoded_insn[2]; // number of instructions committed by this shader core
    unsigned *m_num_INTdecoded_insn[2]; // number of instructions committed by this shader core
    unsigned *m_num_storequeued_insn[2];
    unsigned *m_num_loadqueued_insn[2];
    unsigned *m_num_ialu_acesses[2];
    unsigned *m_num_fp_acesses[2];
    unsigned *m_num_tex_inst[2];
    unsigned *m_num_imul_acesses[2];
    unsigned *m_num_imul32_acesses[2];
    unsigned *m_num_imul24_acesses[2];
    unsigned *m_num_fpmul_acesses[2];
    unsigned *m_num_idiv_acesses[2];
    unsigned *m_num_fpdiv_acesses[2];
    unsigned *m_num_sp_acesses[2];
    unsigned *m_num_sfu_acesses[2];
    unsigned *m_num_trans_acesses[2];
    unsigned *m_num_mem_acesses[2];
    unsigned *m_num_sp_committed[2];
    unsigned *m_num_sfu_committed[2];
    unsigned *m_num_mem_committed[2];
    unsigned *m_active_sp_lanes[2];
    unsigned *m_active_sfu_lanes[2];
    unsigned *m_read_regfile_acesses[2];
    unsigned *m_write_regfile_acesses[2];
    unsigned *m_non_rf_operands[2];
};

class power_core_stat_t : public shader_core_power_stats_pod {
public:
   power_core_stat_t(const struct shader_core_config *shader_config, shader_core_stats *core_stats);
   void visualizer_print( gzFile visualizer_file );
   void print (FILE *fout);
   void init();
   void save_stats();

private:
   shader_core_stats * m_core_stats;
   const shader_core_config *m_config;
   float average_duty_cycle;


};

struct mem_power_stats_pod{
	// [0] = Current stat, [1] = last reading
	unsigned *inst_c_read_access[2];	// Instruction cache read access
	unsigned *inst_c_read_miss[2];		// Instruction cache read miss
	unsigned *const_c_read_access[2];	// Constant cache read access
	unsigned *const_c_read_miss[2];		// Constant cache read miss
	unsigned *text_c_read_access[2];	// Texture cache read access
	unsigned *text_c_read_miss[2];		// Texture cache read miss
	unsigned *l1d_read_access[2];		// L1 Data cache read access
	unsigned *l1d_read_miss[2];			// L1 Data cache read miss
	unsigned *l1d_write_access[2];		// L1 Data cache write access
	unsigned *l1d_write_miss[2];		// L1 Data cache write miss
	unsigned *shmem_read_access[2]; 	// Shared memory access

	// Low level L2 stats
	unsigned *n_l2_read_access[2];
	unsigned *n_l2_read_miss[2];
	unsigned *n_l2_write_access[2];
	unsigned *n_l2_write_miss[2];

	// Low level DRAM stats
    unsigned *n_cmd[2];
    unsigned *n_activity[2];
    unsigned *n_nop[2];
    unsigned *n_act[2];
    unsigned *n_pre[2];
    unsigned *n_rd[2];
    unsigned *n_wr[2];
    unsigned *n_req[2];

    // Interconnect stats
    unsigned *n_simt_to_mem[2];
    unsigned *n_mem_to_simt[2];
};



class power_mem_stat_t : public mem_power_stats_pod{
public:
   power_mem_stat_t(const struct memory_config *mem_config, const struct shader_core_config *shdr_config, memory_stats_t *mem_stats, shader_core_stats *shdr_stats);
   void visualizer_print( gzFile visualizer_file );
   void print (FILE *fout) const;
   void init();
   void save_stats();
private:
   memory_stats_t *m_mem_stats;
   shader_core_stats * m_core_stats;
   const memory_config *m_config;
   const shader_core_config *m_core_config;
};


class power_stat_t {
public:
   power_stat_t( const struct shader_core_config *shader_config,float * average_pipeline_duty_cycle,float * active_sms,shader_core_stats * shader_stats, const struct memory_config *mem_config,memory_stats_t * memory_stats);
   void visualizer_print( gzFile visualizer_file );
   void print (FILE *fout) const;
   void save_stats(){
	   pwr_core_stat->save_stats();
	   pwr_mem_stat->save_stats();
	   *m_average_pipeline_duty_cycle=0;
	   *m_active_sms=0;
   }

   unsigned get_total_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_decoded_insn[0][i]) - (pwr_core_stat->m_num_decoded_insn[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_total_int_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_INTdecoded_insn[0][i]) - (pwr_core_stat->m_num_INTdecoded_insn[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_total_fp_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_FPdecoded_insn[0][i]) - (pwr_core_stat->m_num_FPdecoded_insn[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_total_load_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_loadqueued_insn[0][i]) - (pwr_core_stat->m_num_loadqueued_insn[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_total_store_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_storequeued_insn[0][i]) - (pwr_core_stat->m_num_storequeued_insn[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_sp_committed_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_sp_committed[0][i]) - (pwr_core_stat->m_num_sp_committed[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_sfu_committed_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_sfu_committed[0][i]) - (pwr_core_stat->m_num_sfu_committed[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_mem_committed_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_mem_committed[0][i]) - (pwr_core_stat->m_num_mem_committed[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_committed_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_mem_committed[0][i]) - (pwr_core_stat->m_num_mem_committed[1][i])
				       +(pwr_core_stat->m_num_sfu_committed[0][i]) - (pwr_core_stat->m_num_sfu_committed[1][i])
				       +(pwr_core_stat->m_num_sp_committed[0][i]) - (pwr_core_stat->m_num_sp_committed[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_regfile_reads(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_read_regfile_acesses[0][i]) - (pwr_core_stat->m_read_regfile_acesses[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_regfile_writes(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_write_regfile_acesses[0][i]) - (pwr_core_stat->m_write_regfile_acesses[1][i]);
	   }
	   return total_inst;
   }

   float get_pipeline_duty(){
	   float total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_pipeline_duty_cycle[0][i]) - (pwr_core_stat->m_pipeline_duty_cycle[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_non_regfile_operands(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_non_rf_operands[0][i]) - (pwr_core_stat->m_non_rf_operands[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_sp_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_sp_acesses[0][i]) - (pwr_core_stat->m_num_sp_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_sfu_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_sfu_acesses[0][i]) - (pwr_core_stat->m_num_sfu_acesses[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_trans_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_trans_acesses[0][i]) - (pwr_core_stat->m_num_trans_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_mem_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_mem_acesses[0][i]) - (pwr_core_stat->m_num_mem_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_intdiv_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_idiv_acesses[0][i]) - (pwr_core_stat->m_num_idiv_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_fpdiv_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_fpdiv_acesses[0][i]) - (pwr_core_stat->m_num_fpdiv_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_intmul32_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_imul32_acesses[0][i]) - (pwr_core_stat->m_num_imul32_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_intmul24_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_imul24_acesses[0][i]) - (pwr_core_stat->m_num_imul24_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_intmul_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_imul_acesses[0][i]) - (pwr_core_stat->m_num_imul_acesses[1][i])+
				   	   (pwr_core_stat->m_num_imul24_acesses[0][i]) - (pwr_core_stat->m_num_imul24_acesses[1][i])+
				   	   (pwr_core_stat->m_num_imul32_acesses[0][i]) - (pwr_core_stat->m_num_imul32_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_fpmul_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_fp_acesses[0][i]) - (pwr_core_stat->m_num_fp_acesses[1][i]);
	   }
	   return total_inst;
   }

   float get_sp_active_lanes(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_active_sp_lanes[0][i]) - (pwr_core_stat->m_active_sp_lanes[1][i]);
	   }
	   return (total_inst/m_config->num_shader())/m_config->gpgpu_num_sp_units;
   }

   float get_sfu_active_lanes(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_active_sfu_lanes[0][i]) - (pwr_core_stat->m_active_sfu_lanes[1][i]);	   }

	   return (total_inst/m_config->num_shader())/m_config->gpgpu_num_sfu_units;
   }

   unsigned get_tot_fpu_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_fp_acesses[0][i]) - (pwr_core_stat->m_num_fp_acesses[1][i])+
			   (pwr_core_stat->m_num_fpdiv_acesses[0][i]) - (pwr_core_stat->m_num_fpdiv_acesses[1][i])+
				       (pwr_core_stat->m_num_fpmul_acesses[0][i]) - (pwr_core_stat->m_num_fpmul_acesses[1][i])+
				       (pwr_core_stat->m_num_imul24_acesses[0][i]) - (pwr_core_stat->m_num_imul24_acesses[1][i])+
					(pwr_core_stat->m_num_imul_acesses[0][i]) - (pwr_core_stat->m_num_imul_acesses[1][i])       ;
		   //printf("imul_accesses0: %d imul_acccesses1: %d imul0 - imul1: %d\n",(pwr_core_stat->m_num_imul_acesses[0][i]),(pwr_core_stat->m_num_imul_acesses[1][i]),(pwr_core_stat->m_num_imul_acesses[0][i]-pwr_core_stat->m_num_imul_acesses[1][i]));
		   //printf("imul24_accesses0: %d imul24_acccesses1: %d imu24l0 - imul241: %d\n",(pwr_core_stat->m_num_imul24_acesses[0][i]),(pwr_core_stat->m_num_imul24_acesses[1][i]),(pwr_core_stat->m_num_imul24_acesses[0][i]-pwr_core_stat->m_num_imul24_acesses[1][i]));
		   //printf("total_insn:%d\n",total_inst);


	   }
	   total_inst += get_total_load_inst()+get_total_store_inst()+get_tex_inst();
	   return total_inst;
   }

   unsigned get_tot_sfu_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+= 
				        (pwr_core_stat->m_num_idiv_acesses[0][i]) - (pwr_core_stat->m_num_idiv_acesses[1][i])+
				        (pwr_core_stat->m_num_imul32_acesses[0][i]) - (pwr_core_stat->m_num_imul32_acesses[1][i])+
						(pwr_core_stat->m_num_trans_acesses[0][i]) - (pwr_core_stat->m_num_trans_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_ialu_accessess(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_ialu_acesses[0][i]) - (pwr_core_stat->m_num_ialu_acesses[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_tex_inst(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_core_stat->m_num_tex_inst[0][i]) - (pwr_core_stat->m_num_tex_inst[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_constant_c_accesses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->const_c_read_access[0][i]) - (pwr_mem_stat->const_c_read_access[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_constant_c_misses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->const_c_read_miss[0][i]) - (pwr_mem_stat->const_c_read_miss[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_constant_c_hits(){
	   return (get_constant_c_accesses()-get_constant_c_misses());
   }
   unsigned get_texture_c_accesses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->text_c_read_access[0][i]) - (pwr_mem_stat->text_c_read_access[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_texture_c_misses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->text_c_read_miss[0][i]) - (pwr_mem_stat->text_c_read_miss[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_texture_c_hits(){
	   return ( get_texture_c_accesses()- get_texture_c_misses());
   }
   unsigned get_inst_c_accesses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->inst_c_read_access[0][i]) - (pwr_mem_stat->inst_c_read_access[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_inst_c_misses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->inst_c_read_miss[0][i]) - (pwr_mem_stat->inst_c_read_miss[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_inst_c_hits(){
	   return (get_inst_c_accesses()-get_inst_c_misses());
   }
   unsigned get_l1d_read_accesses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->l1d_read_access[0][i]) - (pwr_mem_stat->l1d_read_access[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_l1d_read_misses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->l1d_read_miss[0][i]) - (pwr_mem_stat->l1d_read_miss[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_l1d_read_hits(){
	   return (get_l1d_read_accesses()-get_l1d_read_misses());
   }
   unsigned get_l1d_write_accesses(){
 	   unsigned total_inst=0;
 	   for(unsigned i=0; i<m_config->num_shader();i++){
 		   total_inst+=(pwr_mem_stat->l1d_write_access[0][i]) - (pwr_mem_stat->l1d_write_access[1][i]);
 	   }
 	   return total_inst;
    }
   unsigned get_l1d_write_misses(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->l1d_write_miss[0][i]) - (pwr_mem_stat->l1d_write_miss[1][i]);
	   }
	   return total_inst;
   }
   unsigned get_l1d_write_hits(){
	   return (get_l1d_write_accesses()-get_l1d_write_misses());
   }
	unsigned get_cache_misses(){
     return get_l1d_read_misses()+get_constant_c_misses()+get_l1d_write_misses()+
		      get_texture_c_misses();
	}
	
	unsigned get_cache_read_misses(){
     return get_l1d_read_misses()+get_constant_c_misses()+
		      get_texture_c_misses();
	}

	unsigned get_cache_write_misses(){
     return get_l1d_write_misses();
	}

   unsigned get_shmem_read_access(){
	   unsigned total_inst=0;
	   for(unsigned i=0; i<m_config->num_shader();i++){
		   total_inst+=(pwr_mem_stat->shmem_read_access[0][i]) - (pwr_mem_stat->shmem_read_access[1][i]);
	   }
	   return total_inst;
   }

   unsigned get_l2_read_accesses(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_l2_read_access[0][i] - pwr_mem_stat->n_l2_read_access[1][i]);
	   }
	   return total;
   }

   unsigned get_l2_read_misses(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_l2_read_miss[0][i] - pwr_mem_stat->n_l2_read_miss[1][i]);
	   }
	   return total;
   }

   unsigned get_l2_read_hits(){
	   return (get_l2_read_accesses()-get_l2_read_misses());
   }

   unsigned get_l2_write_accesses(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_l2_write_access[0][i] - pwr_mem_stat->n_l2_write_access[1][i]);
	   }
	   return total;
   }

   unsigned get_l2_write_misses(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_l2_write_miss[0][i] - pwr_mem_stat->n_l2_write_miss[1][i]);
	   }
	   return total;
   }
   unsigned get_l2_write_hits(){
	   return (get_l2_write_accesses()-get_l2_write_misses());
   }
   unsigned get_dram_cmd(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_cmd[0][i] - pwr_mem_stat->n_cmd[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_activity(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_activity[0][i] - pwr_mem_stat->n_activity[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_nop(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_nop[0][i] - pwr_mem_stat->n_nop[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_act(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_act[0][i] - pwr_mem_stat->n_act[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_pre(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_pre[0][i] - pwr_mem_stat->n_pre[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_rd(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_rd[0][i] - pwr_mem_stat->n_rd[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_wr(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_wr[0][i] - pwr_mem_stat->n_wr[1][i]);
	   }
	   return total;
   }
   unsigned get_dram_req(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_req[0][i] - pwr_mem_stat->n_req[1][i]);
	   }
	   return total;
   }

   unsigned get_icnt_simt_to_mem(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_config->num_shader(); ++i){
		   total += (pwr_mem_stat->n_simt_to_mem[0][i] - pwr_mem_stat->n_simt_to_mem[1][i]);
	   }
	   return total;
   }

   unsigned get_icnt_mem_to_simt(){
	   unsigned total=0;
	   for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
		   total += (pwr_mem_stat->n_mem_to_simt[0][i] - pwr_mem_stat->n_mem_to_simt[1][i]);
	   }
	   return total;
   }

   power_core_stat_t * pwr_core_stat;
   power_mem_stat_t * pwr_mem_stat;
   float * m_average_pipeline_duty_cycle;
   float * m_active_sms;
   const shader_core_config *m_config;
   const struct memory_config *m_mem_config;
};


#endif /*POWER_LATENCY_STAT_H*/
