/* 
 * stat-tool.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the University of British Columbia, Vancouver, 
 * BC V6T 1Z4, All Rights Reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <zlib.h>
#include <string>

// detect gcc 4.3 and use unordered map (part of c++0x)
// unordered map doesn't play nice with _GLIBCXX_DEBUG, just use a map if its enabled.
#if  defined( __GNUC__ ) and not defined( _GLIBCXX_DEBUG )
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
   #include <unordered_map>
   #define my_hash_map std::unordered_map
#else
   #include <ext/hash_map>
   namespace std {
      using namespace __gnu_cxx;
   }
   #define my_hash_map std::hash_map
#endif
#else
   #include <map>
   #define my_hash_map std::map
   #define USE_MAP
#endif

#include "histogram.h"

binned_histogram::binned_histogram (std::string name, int nbins, int* bins) 
   : m_name(name), m_nbins(nbins), m_bins(NULL), m_bin_cnts(new int[m_nbins]), m_maximum(0)
{
   if (bins) {
      m_bins = new int[m_nbins];
      for (int i = 0; i < nbins; i++) {
         m_bins[i] = bins[i];
      }
   }

   reset_bins();
}

binned_histogram::binned_histogram (const binned_histogram& other)
   : m_name(other.m_name), m_nbins(other.m_nbins), m_bins(NULL), 
     m_bin_cnts(new int[m_nbins]), m_maximum(0)
{
   for (int i = 0; i < m_nbins; i++) {
      m_bin_cnts[i] = other.m_bin_cnts[i];
   }
}

void binned_histogram::reset_bins () {
   for (int i = 0; i < m_nbins; i++) {
      m_bin_cnts[i] = 0;
   }
}

void binned_histogram::add2bin (int sample) {
   assert(0);
   m_maximum = (sample > m_maximum)? sample : m_maximum;
}

void binned_histogram::fprint (FILE *fout) {
   if (m_name.c_str() != NULL) fprintf(fout, "%s = ", m_name.c_str());
   for (int i = 0; i < m_nbins; i++) {
      fprintf(fout, "%d ", m_bin_cnts[i]);
   }
   fprintf(fout, "max=%d ", m_maximum);
}

binned_histogram::~binned_histogram () {
   if (m_bins) delete[] m_bins;
   delete[] m_bin_cnts;
}

pow2_histogram::pow2_histogram (std::string name, int nbins, int* bins) 
   : binned_histogram (name, nbins, bins) {}

void pow2_histogram::add2bin (int sample) {
   assert(sample >= 0);
   
   int bin;
   int v = sample;
   register unsigned int shift;

   bin =   (v > 0xFFFF) << 4; v >>= bin;
   shift = (v > 0xFF  ) << 3; v >>= shift; bin |= shift;
   shift = (v > 0xF   ) << 2; v >>= shift; bin |= shift;
   shift = (v > 0x3   ) << 1; v >>= shift; bin |= shift;
                                           bin |= (v >> 1);
   bin += (sample > 0)? 1:0;
   
   m_bin_cnts[bin] += 1;
   
   m_maximum = (sample > m_maximum)? sample : m_maximum;
}

linear_histogram::linear_histogram (int stride, const char *name, int nbins, int* bins) 
   : binned_histogram (name, nbins, bins), m_stride(stride)
{
}

void linear_histogram::add2bin (int sample) {
   assert(sample >= 0);

   int bin = sample / m_stride;      
   
   m_bin_cnts[bin] += 1;
   
   m_maximum = (sample > m_maximum)? sample : m_maximum;
}


#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include "../util.h"

#include "cflogger.h"

/////////////////////////////////////////////////////////////////////////////////////
// logger snapshot trigger: 
// - automate the snap_shot part of loggers to avoid modifying simulation loop everytime 
//   a new time-dependent stat is added
/////////////////////////////////////////////////////////////////////////////////////

class snap_shot_trigger {
protected:
   unsigned long long  m_snap_shot_interval;

public:
   snap_shot_trigger(unsigned long long  interval) : m_snap_shot_interval(interval) {}
   virtual ~snap_shot_trigger() {}
   
   const unsigned long long & get_interval() const { return m_snap_shot_interval;}
   
   void try_snap_shot(unsigned long long  current_cycle) {
      if ((current_cycle % m_snap_shot_interval == 0) && current_cycle != 0) {
         snap_shot(current_cycle);
      }
   }
   
   virtual void snap_shot(unsigned long long  current_cycle) = 0;
};

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

/////////////////////////////////////////////////////////////////////////////////////
// thread control-flow locality logger
/////////////////////////////////////////////////////////////////////////////////////
unsigned translate_pc_to_ptxlineno(unsigned pc);
class thread_insn_span {
private: 
   
   typedef my_hash_map<address_type, int> span_count_map;
   unsigned long long  m_cycle;
   int m_n_insn;
   span_count_map m_insn_span_count;
   
public:
   
   thread_insn_span(unsigned long long  cycle, int n_insn)
      : m_cycle(cycle), m_n_insn(n_insn), 
#ifdef USE_MAP
        m_insn_span_count() 
#else 
        m_insn_span_count(n_insn * 2) 
#endif
   { }

   ~thread_insn_span() { }
   
   thread_insn_span(const thread_insn_span& other)
      : m_cycle(other.m_cycle), m_n_insn(other.m_n_insn), 
        m_insn_span_count(other.m_insn_span_count) 
   { }
      
   thread_insn_span& operator=(const thread_insn_span& other)
   {
      printf("thread_insn_span& operator=\n");
      if (this != &other && m_n_insn != other.m_n_insn) {
         m_n_insn = other.m_n_insn;
         m_insn_span_count = other.m_insn_span_count;
         m_cycle = other.m_cycle;
      }
      return *this;
   }
   
   thread_insn_span& operator+=(const thread_insn_span& other)
   {
      assert(m_n_insn == other.m_n_insn); // no way to aggregate if they are different programs
      span_count_map::const_iterator i_sc = other.m_insn_span_count.begin();
      for (; i_sc != other.m_insn_span_count.end(); ++i_sc) {
         m_insn_span_count[i_sc->first] += i_sc->second;
      }
      return *this;
   }
   
   void set_span( address_type pc ) {
      if( ((int)pc) >= 0 )
         m_insn_span_count[pc] += 1;
   }
   
   void reset(unsigned long long  cycle) {
      m_cycle = cycle;
      m_insn_span_count.clear(); 
   }
   
   void print_span(FILE *fout) {
      fprintf(fout, "%d: ", (int)m_cycle);
      span_count_map::const_iterator i_sc = m_insn_span_count.begin();
      for (; i_sc != m_insn_span_count.end(); ++i_sc) {
         fprintf(fout, "%d ", i_sc->first);
      }
      fprintf(fout, "\n");
   }

   void print_histo(FILE *fout) {
      fprintf(fout, "%d:", (int)m_cycle);
      span_count_map::const_iterator i_sc = m_insn_span_count.begin();
      for (; i_sc != m_insn_span_count.end(); ++i_sc) {
         fprintf(fout, "%d ", i_sc->second);
      }
      fprintf(fout, "\n");
   }

   void print_sparse_histo(FILE *fout) {
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

   void print_sparse_histo(gzFile fout) {
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
};

class thread_CFlocality : public snap_shot_trigger, public spill_log_interface {
private:
   
   std::string m_name;

   int m_nthreads;
   std::vector<address_type> m_thread_pc;
   
   unsigned long long  m_cycle;
   thread_insn_span m_thd_span;
   std::list<thread_insn_span> m_thd_span_archive;
   
public:
   
   thread_CFlocality(std::string name, unsigned long long  snap_shot_interval, 
                     int nthreads, int n_insn, address_type start_pc, unsigned long long  start_cycle = 0)
      : snap_shot_trigger(snap_shot_interval), m_name(name),
        m_nthreads(nthreads), m_thread_pc(nthreads, start_pc), m_cycle(start_cycle),
        m_thd_span(start_cycle, n_insn)
   {
      std::fill(m_thread_pc.begin(), m_thread_pc.end(), -1); // so that hw thread with no work assigned will not clobber results
   }
   
   ~thread_CFlocality() {} 
   
   void update_thread_pc( int thread_id, address_type pc ) {
      m_thread_pc[thread_id] = pc;
      m_thd_span.set_span(pc);
   }
   
   void snap_shot(unsigned long long  current_cycle) {
      m_thd_span_archive.push_back(m_thd_span);
      m_thd_span.reset(current_cycle);
      for (int i = 0; i < (int)m_thread_pc.size(); i++) {
         m_thd_span.set_span(m_thread_pc[i]);
      }
   }
   
   void spill(FILE *fout, bool final) {
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
   
   void print_visualizer(FILE *fout) {
      fprintf(fout, "%s: ", m_name.c_str());
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
   
   void print_visualizer(gzFile fout) {
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
   
   void print_span(FILE *fout) {
      std::list<thread_insn_span>::iterator lit = m_thd_span_archive.begin();
      for (; lit != m_thd_span_archive.end(); ++lit) {
         fprintf(fout, "%s-", m_name.c_str());
         lit->print_span(fout);
      }
      fprintf(fout, "%s-", m_name.c_str());
      m_thd_span.print_span(fout);
   }

   void print_histo(FILE *fout) {
      std::list<thread_insn_span>::iterator lit = m_thd_span_archive.begin();
      for (; lit != m_thd_span_archive.end(); ++lit) {
         fprintf(fout, "%s-", m_name.c_str());
         lit->print_histo(fout);
      }
      fprintf(fout, "%s-", m_name.c_str());
      m_thd_span.print_histo(fout);
   }
};

static int n_thread_CFloggers = 0;
static thread_CFlocality** thread_CFlogger = NULL;

void create_thread_CFlogger( int n_loggers, int n_threads, int n_insn, address_type start_pc, unsigned long long  logging_interval) 
{
   destroy_thread_CFlogger();
   
   n_thread_CFloggers = n_loggers;
   thread_CFlogger = new thread_CFlocality*[n_loggers];

   std::string name_tpl("CFLog");
   char buffer[32];
   for (int i = 0; i < n_thread_CFloggers; i++) {
      snprintf(buffer, 32, "%02d", i);
      thread_CFlogger[i] = new thread_CFlocality( name_tpl + buffer, logging_interval, n_threads, n_insn, start_pc);
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
      delete thread_CFlogger;
      thread_CFlogger = NULL;
   }
}

void cflog_update_thread_pc( int logger_id, int thread_id, address_type pc ) 
{
   if (thread_id < 0) return;
   thread_CFlogger[logger_id]->update_thread_pc(thread_id, pc);
}

void cflog_snapshot( int logger_id, unsigned long long  cycle ) 
{
   thread_CFlogger[logger_id]->snap_shot(cycle);
}

void cflog_print(FILE *fout) 
{
   for (int i = 0; i < n_thread_CFloggers; i++) {
      thread_CFlogger[i]->print_histo(fout);
   }
}

void cflog_visualizer_print(FILE *fout) 
{
   for (int i = 0; i < n_thread_CFloggers; i++) {
      thread_CFlogger[i]->print_visualizer(fout);
   }
}

void cflog_visualizer_gzprint(gzFile fout) 
{
   for (int i = 0; i < n_thread_CFloggers; i++) {
      thread_CFlogger[i]->print_visualizer(fout);
   }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-insn active thread distribution (warp occ) logger
/////////////////////////////////////////////////////////////////////////////////////

class insn_warp_occ_logger{
private:
   int m_simd_width;
   std::vector<linear_histogram> m_insn_warp_occ;
   int m_id;
   static int s_ids;

public:
   insn_warp_occ_logger(int simd_width, int n_insn)
      : m_simd_width(simd_width), 
        m_insn_warp_occ(n_insn, linear_histogram(1, "", m_simd_width)),
        m_id(s_ids++) {}
   
   insn_warp_occ_logger(const insn_warp_occ_logger& other)
      : m_simd_width(other.m_simd_width), 
        m_insn_warp_occ(other.m_insn_warp_occ.size(), linear_histogram(1, "", m_simd_width)),
        m_id(s_ids++) {}
   
   insn_warp_occ_logger& operator=(const insn_warp_occ_logger& p) {
      printf("insn_warp_occ_logger Operator= called: %02d \n", m_id);
      assert(0);
      return *this;
   }   

   ~insn_warp_occ_logger() {}
   
   void set_id(int id) {
      m_id = id;
   }
   
   void log(address_type pc, int warp_occ) {
      m_insn_warp_occ[pc].add2bin(warp_occ - 1);
   }
   
   void print(FILE *fout) {
      for (unsigned i = 0; i < m_insn_warp_occ.size(); i++) {
         fprintf(fout, "InsnWarpOcc%02d-%d", m_id, i);
         m_insn_warp_occ[i].fprint(fout);
         fprintf(fout, "\n");
      }
   }
};
int insn_warp_occ_logger::s_ids = 0;

static std::vector<insn_warp_occ_logger> iwo_logger;

void insn_warp_occ_create( int n_loggers, int simd_width, int n_insn)
{
   iwo_logger.clear();
   iwo_logger.assign(n_loggers, insn_warp_occ_logger(simd_width, n_insn));
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

/////////////////////////////////////////////////////////////////////////////////////
// generic linear histogram logger
/////////////////////////////////////////////////////////////////////////////////////

class linear_histogram_snapshot {
private:
   unsigned long long  m_cycle;
   std::vector<int> m_linear_histogram;
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
   
   void set_cycle(unsigned long long  cycle) {
      m_cycle = cycle;
   }
   
   void print(FILE *fout) {
      fprintf(fout, "%d = ", (int)m_cycle);
      for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
         fprintf(fout, "%d ", m_linear_histogram[i]);
      }
   }

   void print_visualizer(FILE *fout) {
      for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
         fprintf(fout, "%d ", m_linear_histogram[i]);
      }
   }

   void print_visualizer(gzFile fout) {
      for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
         gzprintf(fout, "%d ", m_linear_histogram[i]);
      }
   }
};

class linear_histogram_logger : public snap_shot_trigger, public spill_log_interface {
private:
   int m_n_bins;
   linear_histogram_snapshot m_curr_lin_hist;
   std::list<linear_histogram_snapshot> m_lin_hist_archive;
   unsigned long long  m_cycle;
   bool m_reset_at_snap_shot;
   std::string m_name;
   int m_id;
   static int s_ids;

public:
   linear_histogram_logger(int n_bins, 
                           unsigned long long  snap_shot_interval, 
                           const char *name, 
                           bool reset_at_snap_shot = true, 
                           unsigned long long  start_cycle = 0)
      : snap_shot_trigger(snap_shot_interval), 
        m_n_bins(n_bins), 
        m_curr_lin_hist(m_n_bins, start_cycle),
        m_lin_hist_archive(),
        m_cycle(start_cycle),
        m_reset_at_snap_shot(reset_at_snap_shot), 
        m_name(name),
        m_id(s_ids++) {}
   
   linear_histogram_logger(const linear_histogram_logger& other) // WF: Buggy - Not really copying data over
      : snap_shot_trigger(other.get_interval()), 
        m_n_bins(other.m_n_bins), 
        m_curr_lin_hist(m_n_bins, other.m_cycle),
        m_lin_hist_archive(),
        m_cycle(other.m_cycle),
        m_reset_at_snap_shot(other.m_reset_at_snap_shot), 
        m_name(other.m_name),
        m_id(s_ids++) {}
   
   // using default assignment operator!
   
   ~linear_histogram_logger() {
      // printf("Destroyer called: %s%02d \n", m_name.c_str(), m_id);
      remove_snap_shot_trigger(this);
      remove_spill_log(this);
   }
   
   void set_id(int id) {
      m_id = id;
   }
   
   void log(int pos) {
      m_curr_lin_hist.addsample(pos);
   }
   
   void unlog(int pos) {
      m_curr_lin_hist.subsample(pos);
   }
   
   void snap_shot(unsigned long long  current_cycle) {
      m_lin_hist_archive.push_back(m_curr_lin_hist);
      if (m_reset_at_snap_shot) {
         m_curr_lin_hist.reset(current_cycle);
      } else {
         m_curr_lin_hist.set_cycle(current_cycle);
      }
   }
   
   void spill(FILE *fout, bool final) {
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
   
   void print(FILE *fout) {
      std::list<linear_histogram_snapshot>::iterator iter = m_lin_hist_archive.begin();
      for (; iter != m_lin_hist_archive.end(); ++iter) {
         fprintf(fout, "%s%02d-", m_name.c_str(), m_id);
         iter->print(fout);
         fprintf(fout, "\n");
      }
      fprintf(fout, "%s%02d-", m_name.c_str(), m_id);
      m_curr_lin_hist.print(fout);
      fprintf(fout, "\n");
   }

   void print_visualizer(FILE *fout) {
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

   void print_visualizer(gzFile fout) {
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
};
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
   NORMAL, TEXTURE, CONSTANT
};

int get_shader_normal_cache_id() { return NORMAL; }
int get_shader_texture_cache_id() { return TEXTURE; }
int get_shader_constant_cache_id() { return CONSTANT; }

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
   assert(type == NORMAL || type == TEXTURE || type == CONSTANT);
   assert(miss == 0 || miss == 1);
   
   s_cache_access_logger[logger_id].log(2 * type + miss);
}

void shader_cache_access_unlog( int logger_id, int type, int miss)
{
   if (s_cache_access_logger_n_types == 0) return;
   if (logger_id < 0) return;
   assert(type == NORMAL || type == TEXTURE || type == CONSTANT);
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

