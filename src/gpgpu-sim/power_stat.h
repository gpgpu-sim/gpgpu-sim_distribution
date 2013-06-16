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
#include "gpu-sim.h"

typedef enum _stat_idx{
    CURRENT_STAT_IDX = 0,    // Current activity count
    PREV_STAT_IDX,           // Previous sample activity count
    NUM_STAT_IDX     // Total number of samples
}stat_idx;


struct shader_core_power_stats_pod {
    // [CURRENT_STAT_IDX] = CURRENT_STAT_IDX stat, [PREV_STAT_IDX] = last reading
    float *m_pipeline_duty_cycle[NUM_STAT_IDX];
    unsigned *m_num_decoded_insn[NUM_STAT_IDX]; // number of instructions committed by this shader core
    unsigned *m_num_FPdecoded_insn[NUM_STAT_IDX]; // number of instructions committed by this shader core
    unsigned *m_num_INTdecoded_insn[NUM_STAT_IDX]; // number of instructions committed by this shader core
    unsigned *m_num_storequeued_insn[NUM_STAT_IDX];
    unsigned *m_num_loadqueued_insn[NUM_STAT_IDX];
    unsigned *m_num_ialu_acesses[NUM_STAT_IDX];
    unsigned *m_num_fp_acesses[NUM_STAT_IDX];
    unsigned *m_num_tex_inst[NUM_STAT_IDX];
    unsigned *m_num_imul_acesses[NUM_STAT_IDX];
    unsigned *m_num_imul32_acesses[NUM_STAT_IDX];
    unsigned *m_num_imul24_acesses[NUM_STAT_IDX];
    unsigned *m_num_fpmul_acesses[NUM_STAT_IDX];
    unsigned *m_num_idiv_acesses[NUM_STAT_IDX];
    unsigned *m_num_fpdiv_acesses[NUM_STAT_IDX];
    unsigned *m_num_sp_acesses[NUM_STAT_IDX];
    unsigned *m_num_sfu_acesses[NUM_STAT_IDX];
    unsigned *m_num_trans_acesses[NUM_STAT_IDX];
    unsigned *m_num_mem_acesses[NUM_STAT_IDX];
    unsigned *m_num_sp_committed[NUM_STAT_IDX];
    unsigned *m_num_sfu_committed[NUM_STAT_IDX];
    unsigned *m_num_mem_committed[NUM_STAT_IDX];
    unsigned *m_active_sp_lanes[NUM_STAT_IDX];
    unsigned *m_active_sfu_lanes[NUM_STAT_IDX];
    unsigned *m_read_regfile_acesses[NUM_STAT_IDX];
    unsigned *m_write_regfile_acesses[NUM_STAT_IDX];
    unsigned *m_non_rf_operands[NUM_STAT_IDX];
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
    // [CURRENT_STAT_IDX] = CURRENT_STAT_IDX stat, [PREV_STAT_IDX] = last reading
    class cache_stats core_cache_stats[NUM_STAT_IDX]; // Total core stats
    class cache_stats l2_cache_stats[NUM_STAT_IDX]; // Total L2 partition stats

    unsigned *shmem_read_access[NUM_STAT_IDX];   // Shared memory access

    // Low level DRAM stats
    unsigned *n_cmd[NUM_STAT_IDX];
    unsigned *n_activity[NUM_STAT_IDX];
    unsigned *n_nop[NUM_STAT_IDX];
    unsigned *n_act[NUM_STAT_IDX];
    unsigned *n_pre[NUM_STAT_IDX];
    unsigned *n_rd[NUM_STAT_IDX];
    unsigned *n_wr[NUM_STAT_IDX];
    unsigned *n_req[NUM_STAT_IDX];

    // Interconnect stats
    long *n_simt_to_mem[NUM_STAT_IDX];
    long *n_mem_to_simt[NUM_STAT_IDX];
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
            total_inst+=(pwr_core_stat->m_num_decoded_insn[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_decoded_insn[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_total_int_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_INTdecoded_insn[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_total_fp_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_FPdecoded_insn[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_total_load_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_loadqueued_insn[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_loadqueued_insn[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_total_store_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_storequeued_insn[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_storequeued_insn[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_sp_committed_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sp_committed[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_sfu_committed_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sfu_committed[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_mem_committed_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_mem_committed[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_committed_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_mem_committed[PREV_STAT_IDX][i])
                    +(pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sfu_committed[PREV_STAT_IDX][i])
                    +(pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sp_committed[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_regfile_reads(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_read_regfile_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_read_regfile_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_regfile_writes(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_write_regfile_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_write_regfile_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    float get_pipeline_duty(){
        float total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_pipeline_duty_cycle[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_pipeline_duty_cycle[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_non_regfile_operands(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_non_rf_operands[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_non_rf_operands[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_sp_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_sp_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sp_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_sfu_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_sfu_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_sfu_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }
    unsigned get_trans_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_trans_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_trans_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_mem_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_mem_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_mem_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_intdiv_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_idiv_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_fpdiv_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_fpdiv_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_intmul32_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_intmul24_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_intmul_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul_acesses[PREV_STAT_IDX][i])+
                    (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i])+
                    (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_fpmul_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_fp_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    float get_sp_active_lanes(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_active_sp_lanes[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_active_sp_lanes[PREV_STAT_IDX][i]);
        }
        return (total_inst/m_config->num_shader())/m_config->gpgpu_num_sp_units;
    }

    float get_sfu_active_lanes(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_active_sfu_lanes[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_active_sfu_lanes[PREV_STAT_IDX][i]);
        }

        return (total_inst/m_config->num_shader())/m_config->gpgpu_num_sfu_units;
    }

    unsigned get_tot_fpu_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_fp_acesses[PREV_STAT_IDX][i])+
                    (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_fpdiv_acesses[PREV_STAT_IDX][i])+
                    (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_fpmul_acesses[PREV_STAT_IDX][i])+
                    (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i])+
                    (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul_acesses[PREV_STAT_IDX][i]);
        }
        total_inst += get_total_load_inst()+get_total_store_inst()+get_tex_inst();
        return total_inst;
    }

    unsigned get_tot_sfu_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
                total_inst+= (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_idiv_acesses[PREV_STAT_IDX][i])+
                            (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i])+
                            (pwr_core_stat->m_num_trans_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_trans_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_ialu_accessess(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_ialu_acesses[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_ialu_acesses[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_tex_inst(){
        unsigned total_inst=0;
        for(unsigned i=0; i<m_config->num_shader();i++){
            total_inst+=(pwr_core_stat->m_num_tex_inst[CURRENT_STAT_IDX][i]) - (pwr_core_stat->m_num_tex_inst[PREV_STAT_IDX][i]);
        }
        return total_inst;
    }

    unsigned get_constant_c_accesses(){
        enum mem_access_type access_type[] = {CONST_ACC_R};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_constant_c_misses(){
        enum mem_access_type access_type[] = {CONST_ACC_R};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_constant_c_hits(){
        return (get_constant_c_accesses()-get_constant_c_misses());
    }
    unsigned get_texture_c_accesses(){
        enum mem_access_type access_type[] = {TEXTURE_ACC_R};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_texture_c_misses(){
        enum mem_access_type access_type[] = {TEXTURE_ACC_R};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_texture_c_hits(){
        return ( get_texture_c_accesses()- get_texture_c_misses());
    }
    unsigned get_inst_c_accesses(){
        enum mem_access_type access_type[] = {INST_ACC_R};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_inst_c_misses(){
        enum mem_access_type access_type[] = {INST_ACC_R};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_inst_c_hits(){
        return (get_inst_c_accesses()-get_inst_c_misses());
    }

    unsigned get_l1d_read_accesses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_l1d_read_misses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_l1d_read_hits(){
        return (get_l1d_read_accesses()-get_l1d_read_misses());
    }
    unsigned get_l1d_write_accesses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_l1d_write_misses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_l1d_write_hits(){
        return (get_l1d_write_accesses()-get_l1d_write_misses());
    }
    unsigned get_cache_misses(){
        return get_l1d_read_misses()+get_constant_c_misses()+get_l1d_write_misses()+get_texture_c_misses();
    }
	
    unsigned get_cache_read_misses(){
        return get_l1d_read_misses()+get_constant_c_misses()+get_texture_c_misses();
    }

    unsigned get_cache_write_misses(){
        return get_l1d_write_misses();
    }

    unsigned get_shmem_read_access(){
       unsigned total_inst=0;
       for(unsigned i=0; i<m_config->num_shader();i++){
           total_inst+=(pwr_mem_stat->shmem_read_access[CURRENT_STAT_IDX][i]) - (pwr_mem_stat->shmem_read_access[PREV_STAT_IDX][i]);
       }
       return total_inst;
    }

    unsigned get_l2_read_accesses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, INST_ACC_R};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }

    unsigned get_l2_read_misses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, INST_ACC_R};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }

    unsigned get_l2_read_hits(){
        return (get_l2_read_accesses()-get_l2_read_misses());
    }

    unsigned get_l2_write_accesses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W, L1_WRBK_ACC};
        enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }

    unsigned get_l2_write_misses(){
        enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W, L1_WRBK_ACC};
        enum cache_request_status request_status[] = {MISS};
        unsigned num_access_type = sizeof(access_type)/sizeof(enum mem_access_type);
        unsigned num_request_status = sizeof(request_status)/sizeof(enum cache_request_status);

        return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status)) -
                (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(access_type, num_access_type, request_status, num_request_status));
    }
    unsigned get_l2_write_hits(){
        return (get_l2_write_accesses()-get_l2_write_misses());
    }
    unsigned get_dram_cmd(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_cmd[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_activity(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_activity[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_nop(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_nop[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_act(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_act[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_act[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_pre(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_pre[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_rd(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_rd[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_wr(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_wr[PREV_STAT_IDX][i]);
        }
        return total;
    }
    unsigned get_dram_req(){
        unsigned total=0;
        for(unsigned i=0; i<m_mem_config->m_n_mem; ++i){
            total += (pwr_mem_stat->n_req[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_req[PREV_STAT_IDX][i]);
        }
        return total;
    }

    long get_icnt_simt_to_mem(){
        long total=0;
        for(unsigned i=0; i<m_config->n_simt_clusters; ++i){
            total += (pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_simt_to_mem[PREV_STAT_IDX][i]);
        }
        return total;
    }

    long get_icnt_mem_to_simt(){
        long total=0;
        for(unsigned i=0; i<m_config->n_simt_clusters; ++i){
            total += (pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i] - pwr_mem_stat->n_mem_to_simt[PREV_STAT_IDX][i]);
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
