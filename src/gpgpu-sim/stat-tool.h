// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
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

#ifndef STAT_TOOL_H
#define STAT_TOOL_H

#include "../abstract_hardware_model.h"
#include "histogram.h"
#include "../tr1_hash_map.h"

#include <stdio.h>
#include <zlib.h>

/////////////////////////////////////////////////////////////////////////////////////
// logger snapshot trigger: 
// - automate the snap_shot part of loggers to avoid modifying simulation loop everytime 
//   a new time-dependent stat is added
/////////////////////////////////////////////////////////////////////////////////////

class snap_shot_trigger {
public:
   snap_shot_trigger(unsigned long long  interval) : m_snap_shot_interval(interval) {}
   virtual ~snap_shot_trigger() {}
   
   void try_snap_shot(unsigned long long  current_cycle) {
      if ((current_cycle % m_snap_shot_interval == 0) && current_cycle != 0) {
         snap_shot(current_cycle);
      }
   }
   
   virtual void snap_shot(unsigned long long  current_cycle) = 0;

   const unsigned long long & get_interval() const { return m_snap_shot_interval;}

protected:
   unsigned long long  m_snap_shot_interval;
};


/////////////////////////////////////////////////////////////////////////////////////
// spill log interface: 
// - unified interface to spill log to file to avoid infinite memory usage for logging
/////////////////////////////////////////////////////////////////////////////////////

class spill_log_interface {
public:
   spill_log_interface() {}
   virtual ~spill_log_interface() {}
   
   virtual void spill(FILE *fout, bool final) = 0;
};

/////////////////////////////////////////////////////////////////////////////////////
// thread control-flow locality logger
/////////////////////////////////////////////////////////////////////////////////////

class thread_insn_span {
public:
   thread_insn_span(unsigned long long  cycle);
   thread_insn_span(const thread_insn_span& other);
   ~thread_insn_span();
   
   thread_insn_span& operator=(const thread_insn_span& other);
   thread_insn_span& operator+=(const thread_insn_span& other);
   void set_span( address_type pc );
   void reset(unsigned long long  cycle);
   
   void print_span(FILE *fout) const;
   void print_histo(FILE *fout) const;
   void print_sparse_histo(FILE *fout) const;
   void print_sparse_histo(gzFile fout) const;

private: 
   typedef tr1_hash_map<address_type, int> span_count_map;
   unsigned long long  m_cycle;
   span_count_map m_insn_span_count;
};

class thread_CFlocality : public snap_shot_trigger, public spill_log_interface {
public:
   thread_CFlocality(std::string name, unsigned long long  snap_shot_interval, 
                     int nthreads, address_type start_pc, unsigned long long  start_cycle = 0);
   ~thread_CFlocality();
   
   void update_thread_pc( int thread_id, address_type pc );
   void snap_shot(unsigned long long  current_cycle);
   void spill(FILE *fout, bool final);
   
   void print_visualizer(FILE *fout);
   void print_visualizer(gzFile fout);
   void print_span(FILE *fout) const;
   void print_histo(FILE *fout) const;
private:
   std::string m_name;

   int m_nthreads;
   std::vector<address_type> m_thread_pc;
   
   unsigned long long  m_cycle;
   thread_insn_span m_thd_span;
   std::list<thread_insn_span> m_thd_span_archive;
};

/////////////////////////////////////////////////////////////////////////////////////
// per-insn active thread distribution (warp occ) logger
/////////////////////////////////////////////////////////////////////////////////////

class insn_warp_occ_logger {
public:
   insn_warp_occ_logger(int simd_width)
      : m_simd_width(simd_width), 
        m_insn_warp_occ(1,linear_histogram(1, "", m_simd_width)),
        m_id(s_ids++) {}
   
   insn_warp_occ_logger(const insn_warp_occ_logger& other)
      : m_simd_width(other.m_simd_width), 
        m_insn_warp_occ(other.m_insn_warp_occ.size(), linear_histogram(1, "", m_simd_width)),
        m_id(s_ids++) {}
   
   ~insn_warp_occ_logger() {}

   insn_warp_occ_logger& operator=(const insn_warp_occ_logger& p) {
      printf("insn_warp_occ_logger Operator= called: %02d \n", m_id);
      assert(0);
      return *this;
   }   
   
   void set_id(int id) { m_id = id; }
   
   void log(address_type pc, int warp_occ) {
       if( pc >= m_insn_warp_occ.size() ) 
           m_insn_warp_occ.resize(2*pc, linear_histogram(1, "", m_simd_width));
       m_insn_warp_occ[pc].add2bin(warp_occ - 1);
   }
   
   void print(FILE *fout) const 
   {
      for (unsigned i = 0; i < m_insn_warp_occ.size(); i++) {
         fprintf(fout, "InsnWarpOcc%02d-%d", m_id, i);
         m_insn_warp_occ[i].fprint(fout);
         fprintf(fout, "\n");
      }
   }

private:

   int m_simd_width;
   std::vector<linear_histogram> m_insn_warp_occ;
   int m_id;
   static int s_ids;
};


/////////////////////////////////////////////////////////////////////////////////////
// generic linear histogram logger
/////////////////////////////////////////////////////////////////////////////////////

class linear_histogram_snapshot {
public:
   linear_histogram_snapshot(int n_bins, unsigned long long  cycle) 
      : m_cycle(cycle), 
        m_linear_histogram(n_bins,0) 
   { }
   
   linear_histogram_snapshot(const linear_histogram_snapshot& other) 
      : m_cycle(other.m_cycle), 
        m_linear_histogram(other.m_linear_histogram)
   { }
   
   ~linear_histogram_snapshot() { }
   
   void addsample(int pos) {
      assert((size_t)pos < m_linear_histogram.size());
      m_linear_histogram[pos] += 1;
   }
   
   void subsample(int pos) {
      assert((size_t)pos < m_linear_histogram.size());
      m_linear_histogram[pos] -= 1;
   }
   
   void reset(unsigned long long  cycle) {
      m_cycle = cycle;
      m_linear_histogram.assign(m_linear_histogram.size(), 0);
   }
   
   void set_cycle(unsigned long long  cycle) { m_cycle = cycle; }
   
   void print(FILE *fout) const {
      fprintf(fout, "%d = ", (int)m_cycle);
      for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
         fprintf(fout, "%d ", m_linear_histogram[i]);
      }
   }

   void print_visualizer(FILE *fout) const {
      for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
         fprintf(fout, "%d ", m_linear_histogram[i]);
      }
   }

   void print_visualizer(gzFile fout) const {
      for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
         gzprintf(fout, "%d ", m_linear_histogram[i]);
      }
   }

private:
   unsigned long long  m_cycle;
   std::vector<int> m_linear_histogram;
};

class linear_histogram_logger : public snap_shot_trigger, public spill_log_interface {
public:
   linear_histogram_logger(int n_bins, 
                           unsigned long long  snap_shot_interval, 
                           const char *name, 
                           bool reset_at_snap_shot = true, 
                           unsigned long long  start_cycle = 0);
   linear_histogram_logger(const linear_histogram_logger& other);
   
   ~linear_histogram_logger();
   
   void set_id(int id) { m_id = id; }
   void log(int pos) { m_curr_lin_hist.addsample(pos); }
   void unlog(int pos) { m_curr_lin_hist.subsample(pos); }
   void snap_shot(unsigned long long  current_cycle);
   void spill(FILE *fout, bool final); 

   void print(FILE *fout) const;
   void print_visualizer(FILE *fout);
   void print_visualizer(gzFile fout);

private:
   int m_n_bins;
   linear_histogram_snapshot m_curr_lin_hist;
   std::list<linear_histogram_snapshot> m_lin_hist_archive;
   unsigned long long  m_cycle;
   bool m_reset_at_snap_shot;
   std::string m_name;
   int m_id;
   static int s_ids;
};

void try_snap_shot (unsigned long long  current_cycle);
void set_spill_interval (unsigned long long  interval);
void spill_log_to_file (FILE *fout, int final, unsigned long long  current_cycle);

void create_thread_CFlogger( int n_loggers, int n_threads, address_type start_pc, unsigned long long  logging_interval);
void destroy_thread_CFlogger( );
void cflog_update_thread_pc( int logger_id, int thread_id, address_type pc );
void cflog_snapshot( int logger_id, unsigned long long  cycle );
void cflog_print(FILE *fout);
void cflog_print_path_expression(FILE *fout);
void cflog_visualizer_print(FILE *fout);
void cflog_visualizer_gzprint(gzFile fout);

void insn_warp_occ_create( int n_loggers, int simd_width );
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
int get_shader_instruction_cache_id();
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
void shader_CTA_count_visualizer_gzprint(gzFile fout);

#endif /* CFLOGGER_H */
