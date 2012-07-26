// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
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

#include "stat-tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <zlib.h>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

////////////////////////////////////////////////////////////////////////////////

static unsigned long long  min_snap_shot_interval = 0;
static unsigned long long  next_snap_shot_cycle = 0;
static std::list<snap_shot_trigger*> list_ss_trigger;

void add_snap_shot_trigger (snap_shot_trigger* ss_trigger)
{
   // quick optimization assuming that all snap shot intervals are perfect multiples of each other
   if (min_snap_shot_interval == 0 || min_snap_shot_interval > ss_trigger->get_interval()) {
      min_snap_shot_interval = ss_trigger->get_interval();
      next_snap_shot_cycle = min_snap_shot_interval; // assume that snap shots haven't started yet
   }
   list_ss_trigger.push_back(ss_trigger);
}

void remove_snap_shot_trigger (snap_shot_trigger* ss_trigger)
{
   list_ss_trigger.remove(ss_trigger);
}

void try_snap_shot (unsigned long long  current_cycle)
{
   if (min_snap_shot_interval == 0) return;
   if (current_cycle != next_snap_shot_cycle) return;
   
   std::list<snap_shot_trigger*>::iterator ss_trigger_iter = list_ss_trigger.begin();
   for(; ss_trigger_iter != list_ss_trigger.end(); ++ss_trigger_iter) {
      (*ss_trigger_iter)->snap_shot(current_cycle); // WF: should be try_snap_shot
   }
   next_snap_shot_cycle = current_cycle + min_snap_shot_interval; // WF: stateful testing, maybe bad
}

////////////////////////////////////////////////////////////////////////////////
 
static unsigned long long  spill_interval = 0;
static unsigned long long  next_spill_cycle = 0;
static std::list<spill_log_interface*> list_spill_log;

void add_spill_log (spill_log_interface* spill_log)
{
   list_spill_log.push_back(spill_log);
}

void remove_spill_log (spill_log_interface* spill_log)
{
   list_spill_log.remove(spill_log);
}

void set_spill_interval (unsigned long long  interval)
{
   spill_interval = interval;
   next_spill_cycle = spill_interval;
}

void spill_log_to_file (FILE *fout, int final, unsigned long long  current_cycle)
{
   if (!final && spill_interval == 0) return;
   if (!final && current_cycle <= next_spill_cycle) return;

   fprintf(fout, "\n"); // ensure that the spill occurs at a new line
   std::list<spill_log_interface*>::iterator i_spill_log = list_spill_log.begin();
   for(; i_spill_log != list_spill_log.end(); ++i_spill_log) {
      (*i_spill_log)->spill(fout, final); 
   }
   fflush(fout);

   next_spill_cycle = current_cycle + spill_interval; // WF: stateful testing, maybe bad
}

////////////////////////////////////////////////////////////////////////////////

unsigned translate_pc_to_ptxlineno(unsigned pc);

static int n_thread_CFloggers = 0;
static thread_CFlocality** thread_CFlogger = NULL;

void create_thread_CFlogger( int n_loggers, int n_threads, address_type start_pc, unsigned long long  logging_interval) 
{
   destroy_thread_CFlogger();
   
   n_thread_CFloggers = n_loggers;
   thread_CFlogger = new thread_CFlocality*[n_loggers];

   std::string name_tpl("CFLog");
   char buffer[32];
   for (int i = 0; i < n_thread_CFloggers; i++) {
      snprintf(buffer, 32, "%02d", i);
      thread_CFlogger[i] = new thread_CFlocality( name_tpl + buffer, logging_interval, n_threads, start_pc);
      if (logging_interval != 0) {
         add_snap_shot_trigger(thread_CFlogger[i]);
         add_spill_log(thread_CFlogger[i]);
      }
   }
}

void destroy_thread_CFlogger( ) 
{
   if (thread_CFlogger != NULL) {
      for (int i = 0; i < n_thread_CFloggers; i++) {
         remove_snap_shot_trigger(thread_CFlogger[i]);
         remove_spill_log(thread_CFlogger[i]);
         delete thread_CFlogger[i];
      }
      delete [] thread_CFlogger;
      thread_CFlogger = NULL;
   }
}

void cflog_update_thread_pc( int logger_id, int thread_id, address_type pc ) 
{
   if (thread_CFlogger == NULL) return;  // this means no visualizer output 
   if (thread_id < 0) return;
   thread_CFlogger[logger_id]->update_thread_pc(thread_id, pc);
}

// deprecated 
void cflog_snapshot( int logger_id, unsigned long long  cycle ) 
{
   thread_CFlogger[logger_id]->snap_shot(cycle);
}

void cflog_print(FILE *fout) 
{
   if (thread_CFlogger == NULL) return;  // this means no visualizer output 
   for (int i = 0; i < n_thread_CFloggers; i++) {
      thread_CFlogger[i]->print_histo(fout);
   }
}

void cflog_visualizer_print(FILE *fout) 
{
   if (thread_CFlogger == NULL) return;  // this means no visualizer output 
   for (int i = 0; i < n_thread_CFloggers; i++) {
      thread_CFlogger[i]->print_visualizer(fout);
   }
}

void cflog_visualizer_gzprint(gzFile fout) 
{
   if (thread_CFlogger == NULL) return;  // this means no visualizer output 
   for (int i = 0; i < n_thread_CFloggers; i++) {
      thread_CFlogger[i]->print_visualizer(fout);
   }
}

////////////////////////////////////////////////////////////////////////////////

int insn_warp_occ_logger::s_ids = 0;

static std::vector<insn_warp_occ_logger> iwo_logger;

void insn_warp_occ_create( int n_loggers, int simd_width )
{
   iwo_logger.clear();
   iwo_logger.assign(n_loggers, insn_warp_occ_logger(simd_width));
   for (unsigned i = 0; i < iwo_logger.size(); i++) {
      iwo_logger[i].set_id(i);
   }
}

void insn_warp_occ_log( int logger_id, address_type pc, int warp_occ)
{
   if (warp_occ <= 0) return;
   iwo_logger[logger_id].log(pc, warp_occ);
}

void insn_warp_occ_print( FILE *fout )
{
   for (unsigned i = 0; i < iwo_logger.size(); i++) {
      iwo_logger[i].print(fout);
   }
}

////////////////////////////////////////////////////////////////////////////////

int linear_histogram_logger::s_ids = 0;

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore active thread distribution (warp occ) logger
/////////////////////////////////////////////////////////////////////////////////////

static std::vector<linear_histogram_logger> s_warp_occ_logger;

void shader_warp_occ_create( int n_loggers, int simd_width, unsigned long long  logging_interval)
{
   // simd_width + 1 to include the case with full warp
   s_warp_occ_logger.assign(n_loggers, 
                            linear_histogram_logger(simd_width + 1, logging_interval, "ShdrWarpOcc"));
   for (unsigned i = 0; i < s_warp_occ_logger.size(); i++) {
      s_warp_occ_logger[i].set_id(i);
      add_snap_shot_trigger(&(s_warp_occ_logger[i]));
      add_spill_log(&(s_warp_occ_logger[i]));
   }
}

void shader_warp_occ_log( int logger_id, int warp_occ)
{
   s_warp_occ_logger[logger_id].log(warp_occ);
}

void shader_warp_occ_snapshot( int logger_id, unsigned long long  current_cycle)
{
   s_warp_occ_logger[logger_id].snap_shot(current_cycle);
}

void shader_warp_occ_print( FILE *fout )
{
   for (unsigned i = 0; i < s_warp_occ_logger.size(); i++) {
      s_warp_occ_logger[i].print(fout);
   }
}


/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore memory-access logger
/////////////////////////////////////////////////////////////////////////////////////

static int s_mem_acc_logger_n_dram = 0;
static int s_mem_acc_logger_n_bank = 0;
static std::vector<linear_histogram_logger> s_mem_acc_logger;

void shader_mem_acc_create( int n_loggers, int n_dram, int n_bank, unsigned long long  logging_interval)
{
   // (n_bank + 1) to space data out; 2x to separate read and write
   s_mem_acc_logger.assign(n_loggers, 
                           linear_histogram_logger(2 * n_dram * (n_bank + 1), logging_interval, "ShdrMemAcc"));

   s_mem_acc_logger_n_dram = n_dram;
   s_mem_acc_logger_n_bank = n_bank;
   for (unsigned i = 0; i < s_mem_acc_logger.size(); i++) {
      s_mem_acc_logger[i].set_id(i);
      add_snap_shot_trigger(&(s_mem_acc_logger[i]));
      add_spill_log(&(s_mem_acc_logger[i]));
   }
}

void shader_mem_acc_log( int logger_id, int dram_id, int bank, char rw)
{
   if (s_mem_acc_logger_n_dram == 0) return;
   int write_offset = 0;
   switch(rw) {
   case 'r': write_offset = 0; break;
   case 'w': write_offset = (s_mem_acc_logger_n_bank + 1) * s_mem_acc_logger_n_dram; break;
   default: assert(0); break;
   }
   s_mem_acc_logger[logger_id].log(dram_id * s_mem_acc_logger_n_bank + bank + write_offset);
}

void shader_mem_acc_snapshot( int logger_id, unsigned long long  current_cycle)
{
   s_mem_acc_logger[logger_id].snap_shot(current_cycle);
}

void shader_mem_acc_print( FILE *fout )
{
   for (unsigned i = 0; i < s_mem_acc_logger.size(); i++) {
      s_mem_acc_logger[i].print(fout);
   }
}


/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore memory-latency logger
/////////////////////////////////////////////////////////////////////////////////////

static bool s_mem_lat_logger_used = false;
static int s_mem_lat_logger_nbins = 48;     // up to 2^24 = 16M
static std::vector<linear_histogram_logger> s_mem_lat_logger;

void shader_mem_lat_create( int n_loggers, unsigned long long  logging_interval)
{
   s_mem_lat_logger.assign(n_loggers, 
                           linear_histogram_logger(s_mem_lat_logger_nbins, logging_interval, "ShdrMemLat"));

   for (unsigned i = 0; i < s_mem_lat_logger.size(); i++) {
      s_mem_lat_logger[i].set_id(i);
      add_snap_shot_trigger(&(s_mem_lat_logger[i]));
      add_spill_log(&(s_mem_lat_logger[i]));
   }
   
   s_mem_lat_logger_used = true;
}

void shader_mem_lat_log( int logger_id, int latency)
{
   if (s_mem_lat_logger_used == false) return;
   if (latency > (1<<(s_mem_lat_logger_nbins/2))) assert(0); // guard for out of bound bin
   assert(latency > 0);
   
   int latency_bin;
   
   int bin; // LOG_2(latency)
   int v = latency;
   register unsigned int shift;

   bin =   (v > 0xFFFF) << 4; v >>= bin;
   shift = (v > 0xFF  ) << 3; v >>= shift; bin |= shift;
   shift = (v > 0xF   ) << 2; v >>= shift; bin |= shift;
   shift = (v > 0x3   ) << 1; v >>= shift; bin |= shift;
                                           bin |= (v >> 1);
   latency_bin = 2 * bin;
   if (bin > 0) {
      latency_bin += ((latency & (1 << (bin - 1))) != 0)? 1 : 0; // approx. for LOG_sqrt2(latency)
   }

   s_mem_lat_logger[logger_id].log(latency_bin);
}

void shader_mem_lat_snapshot( int logger_id, unsigned long long  current_cycle)
{
   s_mem_lat_logger[logger_id].snap_shot(current_cycle);
}

void shader_mem_lat_print( FILE *fout )
{
   for (unsigned i = 0; i < s_mem_lat_logger.size(); i++) {
      s_mem_lat_logger[i].print(fout);
   }
}


/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore cache-miss logger
/////////////////////////////////////////////////////////////////////////////////////

static int s_cache_access_logger_n_types = 0;
static std::vector<linear_histogram_logger> s_cache_access_logger;

enum cache_access_logger_types {
   NORMAL, TEXTURE, CONSTANT, INSTRUCTION
};

int get_shader_normal_cache_id() { return NORMAL; }
int get_shader_texture_cache_id() { return TEXTURE; }
int get_shader_constant_cache_id() { return CONSTANT; }
int get_shader_instruction_cache_id() { return INSTRUCTION; }

void shader_cache_access_create( int n_loggers, int n_types, unsigned long long  logging_interval)
{
   // There are different type of cache (x2 for recording accesses and misses)
   s_cache_access_logger.assign(n_loggers, 
                                linear_histogram_logger(n_types * 2, logging_interval, "ShdrCacheMiss"));

   s_cache_access_logger_n_types = n_types;
   for (unsigned i = 0; i < s_cache_access_logger.size(); i++) {
      s_cache_access_logger[i].set_id(i);
      add_snap_shot_trigger(&(s_cache_access_logger[i]));
      add_spill_log(&(s_cache_access_logger[i]));
   }
}

void shader_cache_access_log( int logger_id, int type, int miss)
{
   if (s_cache_access_logger_n_types == 0) return;
   if (logger_id < 0) return;
   assert(type == NORMAL || type == TEXTURE || type == CONSTANT || type == INSTRUCTION);
   assert(miss == 0 || miss == 1);
   
   s_cache_access_logger[logger_id].log(2 * type + miss);
}

void shader_cache_access_unlog( int logger_id, int type, int miss)
{
   if (s_cache_access_logger_n_types == 0) return;
   if (logger_id < 0) return;
   assert(type == NORMAL || type == TEXTURE || type == CONSTANT || type == INSTRUCTION);
   assert(miss == 0 || miss == 1);
   
   s_cache_access_logger[logger_id].unlog(2 * type + miss);
}

void shader_cache_access_print( FILE *fout )
{
   for (unsigned i = 0; i < s_cache_access_logger.size(); i++) {
      s_cache_access_logger[i].print(fout);
   }
}


/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore CTA count logger (only make sense with gpgpu_spread_blocks_across_cores)
/////////////////////////////////////////////////////////////////////////////////////

static linear_histogram_logger *s_CTA_count_logger = NULL;

void shader_CTA_count_create( int n_shaders, unsigned long long  logging_interval)
{
   // only need one logger to track all the shaders
   if (s_CTA_count_logger != NULL) delete s_CTA_count_logger;
   s_CTA_count_logger = new linear_histogram_logger(n_shaders, logging_interval, "ShdrCTACount", false);

   s_CTA_count_logger->set_id(-1);
   if (logging_interval != 0) {
      add_snap_shot_trigger(s_CTA_count_logger);
      add_spill_log(s_CTA_count_logger);
   }
}

void shader_CTA_count_log( int shader_id, int nCTAadded )
{
   if (s_CTA_count_logger == NULL) return;
   
   for (int i = 0; i < nCTAadded; i++) {
      s_CTA_count_logger->log(shader_id);
   }
}

void shader_CTA_count_unlog( int shader_id, int nCTAdone )
{
   if (s_CTA_count_logger == NULL) return;
   
   for (int i = 0; i < nCTAdone; i++) {
      s_CTA_count_logger->unlog(shader_id);
   }
}

void shader_CTA_count_print( FILE *fout )
{
   if (s_CTA_count_logger == NULL) return;
   s_CTA_count_logger->print(fout);
}

void shader_CTA_count_visualizer_print( FILE *fout )
{
   if (s_CTA_count_logger == NULL) return;
   s_CTA_count_logger->print_visualizer(fout);
}

void shader_CTA_count_visualizer_gzprint( gzFile fout )
{
   if (s_CTA_count_logger == NULL) return;
   s_CTA_count_logger->print_visualizer(fout);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

thread_insn_span::thread_insn_span(unsigned long long  cycle)
  : m_cycle(cycle),
#if (tr1_hash_map_ismap == 1)
     m_insn_span_count() 
#else 
     m_insn_span_count(32*1024) 
#endif
{ 
}

thread_insn_span::~thread_insn_span() { }
   
thread_insn_span::thread_insn_span(const thread_insn_span& other)
      : m_cycle(other.m_cycle),
        m_insn_span_count(other.m_insn_span_count) 
{ 
}
      
thread_insn_span& thread_insn_span::operator=(const thread_insn_span& other)
{
   printf("thread_insn_span& operator=\n");
   if (this != &other) {
      m_insn_span_count = other.m_insn_span_count;
      m_cycle = other.m_cycle;
   }
   return *this;
}
   
thread_insn_span& thread_insn_span::operator+=(const thread_insn_span& other)
{
   span_count_map::const_iterator i_sc = other.m_insn_span_count.begin();
   for (; i_sc != other.m_insn_span_count.end(); ++i_sc) {
      m_insn_span_count[i_sc->first] += i_sc->second;
   }
   return *this;
}
   
void thread_insn_span::set_span( address_type pc ) 
{
   if( ((int)pc) >= 0 )
      m_insn_span_count[pc] += 1;
}
   
void thread_insn_span::reset(unsigned long long  cycle) 
{
   m_cycle = cycle;
   m_insn_span_count.clear(); 
}
   
void thread_insn_span::print_span(FILE *fout) const
{
   fprintf(fout, "%d: ", (int)m_cycle);
   span_count_map::const_iterator i_sc = m_insn_span_count.begin();
   for (; i_sc != m_insn_span_count.end(); ++i_sc) {
      fprintf(fout, "%d ", i_sc->first);
   }
   fprintf(fout, "\n");
}

void thread_insn_span::print_histo(FILE *fout) const
{
   fprintf(fout, "%d:", (int)m_cycle);
   span_count_map::const_iterator i_sc = m_insn_span_count.begin();
   for (; i_sc != m_insn_span_count.end(); ++i_sc) {
      fprintf(fout, "%d ", i_sc->second);
   }
   fprintf(fout, "\n");
}

void thread_insn_span::print_sparse_histo(FILE *fout) const
{
   int n_printed_entries = 0;
   span_count_map::const_iterator i_sc = m_insn_span_count.begin();
   for (; i_sc != m_insn_span_count.end(); ++i_sc) {
      unsigned ptx_lineno = translate_pc_to_ptxlineno(i_sc->first);
      fprintf(fout, "%u %d ", ptx_lineno, i_sc->second);
      n_printed_entries++;
   }
   if (n_printed_entries == 0) {
      fprintf(fout, "0 0 ");
   }
   fprintf(fout, "\n");
}

void thread_insn_span::print_sparse_histo(gzFile fout) const
{
   int n_printed_entries = 0;
   span_count_map::const_iterator i_sc = m_insn_span_count.begin();
   for (; i_sc != m_insn_span_count.end(); ++i_sc) {
      unsigned ptx_lineno = translate_pc_to_ptxlineno(i_sc->first);
      gzprintf(fout, "%u %d ", ptx_lineno, i_sc->second);
      n_printed_entries++;
   }
   if (n_printed_entries == 0) {
      gzprintf(fout, "0 0 ");
   }
   gzprintf(fout, "\n");
}

////////////////////////////////////////////////////////////////////////////////

thread_CFlocality::thread_CFlocality(std::string name, 
                                     unsigned long long  snap_shot_interval, 
                                     int nthreads, 
                                     address_type start_pc, 
                                     unsigned long long  start_cycle)
      : snap_shot_trigger(snap_shot_interval), m_name(name),
        m_nthreads(nthreads), m_thread_pc(nthreads, start_pc), m_cycle(start_cycle),
        m_thd_span(start_cycle)
{
   std::fill(m_thread_pc.begin(), m_thread_pc.end(), -1); // so that hw thread with no work assigned will not clobber results
}
   
thread_CFlocality::~thread_CFlocality() 
{
} 
   
void thread_CFlocality::update_thread_pc( int thread_id, address_type pc ) 
{
   m_thread_pc[thread_id] = pc;
   m_thd_span.set_span(pc);
}
   
void thread_CFlocality::snap_shot(unsigned long long  current_cycle) 
{
   m_thd_span_archive.push_back(m_thd_span);
   m_thd_span.reset(current_cycle);
   for (int i = 0; i < (int)m_thread_pc.size(); i++) {
      m_thd_span.set_span(m_thread_pc[i]);
   }
}
   
void thread_CFlocality::spill(FILE *fout, bool final) 
{
   std::list<thread_insn_span>::iterator lit = m_thd_span_archive.begin();
   for (; lit != m_thd_span_archive.end(); lit = m_thd_span_archive.erase(lit) ) {
      fprintf(fout, "%s-", m_name.c_str());
      lit->print_histo(fout);
   }
   assert( m_thd_span_archive.empty() );
   if (final) {
      fprintf(fout, "%s-", m_name.c_str());
      m_thd_span.print_histo(fout);
   }
}
   
      
void thread_CFlocality::print_visualizer(FILE *fout)  
{
   fprintf(fout, "%s: ", m_name.c_str());
   if (m_thd_span_archive.empty()) {
   
      // visualizer do no require snap_shots
      m_thd_span.print_sparse_histo(fout);
      
      // clean the thread span
      m_thd_span.reset(0);
      for (int i = 0; i < (int)m_thread_pc.size(); i++) 
         m_thd_span.set_span(m_thread_pc[i]);
   } else { 
      assert(0); // TODO: implement fall back so that visualizer can work with snap shots
   }
}
   
void thread_CFlocality::print_visualizer(gzFile fout)
{
   gzprintf(fout, "%s: ", m_name.c_str());
   if (m_thd_span_archive.empty()) {
   
      // visualizer do no require snap_shots
      m_thd_span.print_sparse_histo(fout);
      
      // clean the thread span
      m_thd_span.reset(0);
      for (int i = 0; i < (int)m_thread_pc.size(); i++) {
         m_thd_span.set_span(m_thread_pc[i]);
      }
   } else { 
      assert(0); // TODO: implement fall back so that visualizer can work with snap shots
   }
}
   
void thread_CFlocality::print_span(FILE *fout) const
{
   std::list<thread_insn_span>::const_iterator lit = m_thd_span_archive.begin();
   for (; lit != m_thd_span_archive.end(); ++lit) {
      fprintf(fout, "%s-", m_name.c_str());
      lit->print_span(fout);
   }
   fprintf(fout, "%s-", m_name.c_str());
   m_thd_span.print_span(fout);
}

void thread_CFlocality::print_histo(FILE *fout) const
{
   std::list<thread_insn_span>::const_iterator lit = m_thd_span_archive.begin();
   for (; lit != m_thd_span_archive.end(); ++lit) {
      fprintf(fout, "%s-", m_name.c_str());
      lit->print_histo(fout);
   }
   fprintf(fout, "%s-", m_name.c_str());
   m_thd_span.print_histo(fout);
}

////////////////////////////////////////////////////////////////////////////////

linear_histogram_logger::linear_histogram_logger(int n_bins, 
                           unsigned long long  snap_shot_interval, 
                           const char *name, 
                           bool reset_at_snap_shot, 
                           unsigned long long  start_cycle )
      : snap_shot_trigger(snap_shot_interval), 
        m_n_bins(n_bins), 
        m_curr_lin_hist(m_n_bins, start_cycle),
        m_lin_hist_archive(),
        m_cycle(start_cycle),
        m_reset_at_snap_shot(reset_at_snap_shot), 
        m_name(name),
        m_id(s_ids++) 
{
}

linear_histogram_logger::linear_histogram_logger(const linear_histogram_logger& other) 
      : snap_shot_trigger(other.get_interval()), 
        m_n_bins(other.m_n_bins), 
        m_curr_lin_hist(m_n_bins, other.m_cycle),
        m_lin_hist_archive(),
        m_cycle(other.m_cycle),
        m_reset_at_snap_shot(other.m_reset_at_snap_shot), 
        m_name(other.m_name),
        m_id(s_ids++) 
{
}

linear_histogram_logger::~linear_histogram_logger() 
{
      remove_snap_shot_trigger(this);
      remove_spill_log(this);
}
   
void linear_histogram_logger::snap_shot(unsigned long long  current_cycle) {
   m_lin_hist_archive.push_back(m_curr_lin_hist);
   if (m_reset_at_snap_shot) {
      m_curr_lin_hist.reset(current_cycle);
   } else {
      m_curr_lin_hist.set_cycle(current_cycle);
   }
}
   
void linear_histogram_logger::spill(FILE *fout, bool final) 
{
   std::list<linear_histogram_snapshot>::iterator iter = m_lin_hist_archive.begin();
   for (; iter != m_lin_hist_archive.end(); iter = m_lin_hist_archive.erase(iter) ) {
      fprintf(fout, "%s%02d-", m_name.c_str(), (m_id >= 0)? m_id : 0);
      iter->print(fout);
      fprintf(fout, "\n");
   }
   assert( m_lin_hist_archive.empty() );
   if (final) {
      fprintf(fout, "%s%02d-", m_name.c_str(), (m_id >= 0)? m_id : 0);
      m_curr_lin_hist.print(fout);
      fprintf(fout, "\n");
   }
}
   
void linear_histogram_logger::print(FILE *fout) const
{
   std::list<linear_histogram_snapshot>::const_iterator iter = m_lin_hist_archive.begin();
   for (; iter != m_lin_hist_archive.end(); ++iter) {
      fprintf(fout, "%s%02d-", m_name.c_str(), m_id);
      iter->print(fout);
      fprintf(fout, "\n");
   }
   fprintf(fout, "%s%02d-", m_name.c_str(), m_id);
   m_curr_lin_hist.print(fout);
   fprintf(fout, "\n");
}

void linear_histogram_logger::print_visualizer(FILE *fout)
{
   assert(m_lin_hist_archive.empty()); // don't support snapshot for now
   fprintf(fout, "%s", m_name.c_str());
   if (m_id >= 0) {
      fprintf(fout, "%02d: ", m_id);
   } else {
      fprintf(fout, ": ");
   }
   m_curr_lin_hist.print_visualizer(fout);
   fprintf(fout, "\n");
   if (m_reset_at_snap_shot) {
      m_curr_lin_hist.reset(0);
   } 
}

void linear_histogram_logger::print_visualizer(gzFile fout)
{
   assert(m_lin_hist_archive.empty()); // don't support snapshot for now
   gzprintf(fout, "%s", m_name.c_str());
   if (m_id >= 0) {
      gzprintf(fout, "%02d: ", m_id);
   } else {
      gzprintf(fout, ": ");
   }
   m_curr_lin_hist.print_visualizer(fout);
   gzprintf(fout, "\n");
   if (m_reset_at_snap_shot) {
      m_curr_lin_hist.reset(0);
   } 
}

