/* 
 * cflogger.h 
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, and the 
 * University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#ifndef CFLOGGER_H
#define CFLOGGER_H

void try_snap_shot (unsigned long long  current_cycle);
void set_spill_interval (unsigned long long  interval);
void spill_log_to_file (FILE *fout, int final, unsigned long long  current_cycle);

void create_thread_CFlogger( int n_loggers, int n_threads, int n_insn, address_type start_pc, unsigned long long  logging_interval);
void destroy_thread_CFlogger( );
void cflog_update_thread_pc( int logger_id, int thread_id, address_type pc );
void cflog_snapshot( int logger_id, unsigned long long  cycle );
void cflog_print(FILE *fout);
void cflog_print_path_expression(FILE *fout);
void cflog_visualizer_print(FILE *fout);


void insn_warp_occ_create( int n_loggers, int simd_width, int n_insn );
void insn_warp_occ_log( int logger_id, address_type pc, int warp_occ );
void insn_warp_occ_print( FILE *fout );


void shader_warp_occ_create( int n_loggers, int simd_width, unsigned long long  logging_interval );
void shader_warp_occ_log( int logger_id, int warp_occ );
void shader_warp_occ_snapshot( int logger_id, unsigned long long  current_cycle );
void shader_warp_occ_print( FILE *fout );


void shader_mem_acc_create( int n_loggers, int n_dram, int n_bank, unsigned long long  logging_interval );
void shader_mem_acc_log( int logger_id, int dram_id, int bank, char rw );
void shader_mem_acc_snapshot( int logger_id, unsigned long long  current_cycle );
void shader_mem_acc_print( FILE *fout );


void shader_mem_lat_create( int n_loggers, unsigned long long  logging_interval );
void shader_mem_lat_log( int logger_id, int latency );
void shader_mem_lat_snapshot( int logger_id, unsigned long long  current_cycle );
void shader_mem_lat_print( FILE *fout );


int get_shader_normal_cache_id();
int get_shader_texture_cache_id();
int get_shader_constant_cache_id();
void shader_cache_access_create( int n_loggers, int n_types, unsigned long long  logging_interval );
void shader_cache_access_log( int logger_id, int type, int miss);
void shader_cache_access_unlog( int logger_id, int type, int miss);
void shader_cache_access_print( FILE *fout );


void shader_CTA_count_create( int n_shaders, unsigned long long  logging_interval);
void shader_CTA_count_log( int shader_id, int nCTAadded );
void shader_CTA_count_unlog( int shader_id, int nCTAdone );
void shader_CTA_count_resetnow( );
void shader_CTA_count_print( FILE *fout );
void shader_CTA_count_visualizer_print( FILE *fout );

#endif /* CFLOGGER_H */
