// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "visualizer.h"

#include "../option_parser.h"
#include "gpu-sim.h"
#include "l2cache.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "shader.h"
// #include "../../../mcpat/processor.h"
#include "gpu-cache.h"
#include "stat-tool.h"

#include <string.h>
#include <time.h>
#include <zlib.h>

static void time_vector_print_interval2gzfile(gzFile outfile);

void gpgpu_sim::visualizer_printstat() {
  gzFile visualizer_file = NULL;  // gzFile is basically a pointer to a struct,
                                  // so it is fine to initialize it as NULL
  if (!m_config.g_visualizer_enabled) return;

  // clean the content of the visualizer log if it is the first time, otherwise
  // attach at the end
  static bool visualizer_first_printstat = true;

  visualizer_file = gzopen(m_config.g_visualizer_filename,
                           (visualizer_first_printstat) ? "w" : "a");
  if (visualizer_file == NULL) {
    printf("error - could not open visualizer trace file.\n");
    exit(1);
  }
  gzsetparams(visualizer_file, m_config.g_visualizer_zlevel,
              Z_DEFAULT_STRATEGY);
  visualizer_first_printstat = false;

  cflog_visualizer_gzprint(visualizer_file);
  shader_CTA_count_visualizer_gzprint(visualizer_file);

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->visualizer_print(visualizer_file);
  m_shader_stats->visualizer_print(visualizer_file);
  m_memory_stats->visualizer_print(visualizer_file);
  m_power_stats->visualizer_print(visualizer_file);
  // proc->visualizer_print(visualizer_file);
  // other parameters for graphing
  gzprintf(visualizer_file, "globalcyclecount: %lld\n", gpu_sim_cycle);
  gzprintf(visualizer_file, "globalinsncount: %lld\n", gpu_sim_insn);
  gzprintf(visualizer_file, "globaltotinsncount: %lld\n", gpu_tot_sim_insn);

  time_vector_print_interval2gzfile(visualizer_file);

  gzclose(visualizer_file);
  /*
     gzprintf(visualizer_file, "CacheMissRate_GlobalLocalL1_All: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1_windowed_cache_miss_rate(0)); gzprintf(visualizer_file, "\n");
     gzprintf(visualizer_file, "CacheMissRate_TextureL1_All: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1tex_windowed_cache_miss_rate(0)); gzprintf(visualizer_file,
     "\n"); gzprintf(visualizer_file, "CacheMissRate_ConstL1_All: "); for
     (unsigned i=0;i<m_n_shader;i++) gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1const_windowed_cache_miss_rate(0)); gzprintf(visualizer_file,
     "\n"); gzprintf(visualizer_file, "CacheMissRate_GlobalLocalL1_noMgHt: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1_windowed_cache_miss_rate(1)); gzprintf(visualizer_file, "\n");
     gzprintf(visualizer_file, "CacheMissRate_TextureL1_noMgHt: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1tex_windowed_cache_miss_rate(1)); gzprintf(visualizer_file,
     "\n"); gzprintf(visualizer_file, "CacheMissRate_ConstL1_noMgHt: "); for
     (unsigned i=0;i<m_n_shader;i++) gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1const_windowed_cache_miss_rate(1)); gzprintf(visualizer_file,
     "\n");
     // reset for next interval
     for (unsigned i=0;i<m_n_shader;i++)
        m_sc[i]->new_cache_window();
  */
}

#include <iostream>
#include <list>
#include <map>
#include <vector>
#include "../gpgpu-sim/shader.h"
class my_time_vector {
 private:
  std::map<unsigned int, std::vector<long int> > ld_time_map;
  std::map<unsigned int, std::vector<long int> > st_time_map;
  unsigned ld_vector_size;
  unsigned st_vector_size;
  std::vector<double> ld_time_dist;
  std::vector<double> st_time_dist;

  std::vector<double> overal_ld_time_dist;
  std::vector<double> overal_st_time_dist;
  int overal_ld_count;
  int overal_st_count;

 public:
  my_time_vector(int ld_size, int st_size) {
    ld_vector_size = ld_size;
    st_vector_size = st_size;
    ld_time_dist.resize(ld_size);
    st_time_dist.resize(st_size);
    overal_ld_time_dist.resize(ld_size);
    overal_st_time_dist.resize(st_size);
    overal_ld_count = 0;
    overal_st_count = 0;
  }
  void update_ld(unsigned int uid, unsigned int slot, long int time) {
    if (ld_time_map.find(uid) != ld_time_map.end()) {
      ld_time_map[uid][slot] = time;
    } else if (slot < NUM_MEM_REQ_STAT) {
      std::vector<long int> time_vec;
      time_vec.resize(ld_vector_size);
      time_vec[slot] = time;
      ld_time_map[uid] = time_vec;
    } else {
      // It's a merged mshr! forget it
    }
  }
  void update_st(unsigned int uid, unsigned int slot, long int time) {
    if (st_time_map.find(uid) != st_time_map.end()) {
      st_time_map[uid][slot] = time;
    } else {
      std::vector<long int> time_vec;
      time_vec.resize(st_vector_size);
      time_vec[slot] = time;
      st_time_map[uid] = time_vec;
    }
  }
  void check_ld_update(unsigned int uid, unsigned int slot, long int latency) {
    if (ld_time_map.find(uid) != ld_time_map.end()) {
      int our_latency =
          ld_time_map[uid][slot] - ld_time_map[uid][IN_ICNT_TO_MEM];
      assert(our_latency == latency);
    } else if (slot < NUM_MEM_REQ_STAT) {
      abort();
    }
  }
  void check_st_update(unsigned int uid, unsigned int slot, long int latency) {
    if (st_time_map.find(uid) != st_time_map.end()) {
      int our_latency =
          st_time_map[uid][slot] - st_time_map[uid][IN_ICNT_TO_MEM];
      assert(our_latency == latency);
    } else {
      abort();
    }
  }

 private:
  void calculate_ld_dist(void) {
    unsigned i, first;
    long int last_update, diff;
    int finished_count = 0;
    ld_time_dist.clear();
    ld_time_dist.resize(ld_vector_size);
    std::map<unsigned int, std::vector<long int> >::iterator iter, iter_temp;
    iter = ld_time_map.begin();
    while (iter != ld_time_map.end()) {
      last_update = 0;
      first = -1;
      if (!iter->second[IN_SHADER_FETCHED]) {
        // this request is not done yet skip it!
        ++iter;
        continue;
      }
      while (!last_update) {
        first++;
        assert(first < iter->second.size());
        last_update = iter->second[first];
      }

      for (i = first; i < ld_vector_size; i++) {
        diff = iter->second[i] - last_update;
        if (diff > 0) {
          ld_time_dist[i] += diff;
          last_update = iter->second[i];
        }
      }
      iter_temp = iter;
      iter++;
      ld_time_map.erase(iter_temp);
      finished_count++;
    }
    if (finished_count) {
      for (i = 0; i < ld_vector_size; i++) {
        overal_ld_time_dist[i] =
            (overal_ld_time_dist[i] * overal_ld_count + ld_time_dist[i]) /
            (overal_ld_count + finished_count);
      }
      overal_ld_count += finished_count;
      for (i = 0; i < ld_vector_size; i++) {
        ld_time_dist[i] /= finished_count;
      }
    }
  }

  void calculate_st_dist(void) {
    unsigned i, first;
    long int last_update, diff;
    int finished_count = 0;
    st_time_dist.clear();
    st_time_dist.resize(st_vector_size);
    std::map<unsigned int, std::vector<long int> >::iterator iter, iter_temp;
    iter = st_time_map.begin();
    while (iter != st_time_map.end()) {
      last_update = 0;
      first = -1;
      if (!iter->second[IN_SHADER_FETCHED]) {
        // this request is not done yet skip it!
        ++iter;
        continue;
      }
      while (!last_update) {
        first++;
        assert(first < iter->second.size());
        last_update = iter->second[first];
      }

      for (i = first; i < st_vector_size; i++) {
        diff = iter->second[i] - last_update;
        if (diff > 0) {
          st_time_dist[i] += diff;
          last_update = iter->second[i];
        }
      }
      iter_temp = iter;
      iter++;
      st_time_map.erase(iter_temp);
      finished_count++;
    }
    if (finished_count) {
      for (i = 0; i < st_vector_size; i++) {
        overal_st_time_dist[i] =
            (overal_st_time_dist[i] * overal_st_count + st_time_dist[i]) /
            (overal_st_count + finished_count);
      }
      overal_st_count += finished_count;
      for (i = 0; i < st_vector_size; i++) {
        st_time_dist[i] /= finished_count;
      }
    }
  }

 public:
  void clear_time_map_vectors(void) {
    ld_time_map.clear();
    st_time_map.clear();
  }
  void print_all_ld(void) {
    unsigned i;
    std::map<unsigned int, std::vector<long int> >::iterator iter;
    for (iter = ld_time_map.begin(); iter != ld_time_map.end(); ++iter) {
      std::cout << "ld_uid" << iter->first;
      for (i = 0; i < ld_vector_size; i++) {
        std::cout << " " << iter->second[i];
      }
      std::cout << std::endl;
    }
  }

  void print_all_st(void) {
    unsigned i;
    std::map<unsigned int, std::vector<long int> >::iterator iter;

    for (iter = st_time_map.begin(); iter != st_time_map.end(); ++iter) {
      std::cout << "st_uid" << iter->first;
      for (i = 0; i < st_vector_size; i++) {
        std::cout << " " << iter->second[i];
      }
      std::cout << std::endl;
    }
  }

  void calculate_dist() {
    calculate_ld_dist();
    calculate_st_dist();
  }
  void print_dist(void) {
    unsigned i;
    calculate_dist();
    std::cout << "LD_mem_lat_dist ";
    for (i = 0; i < ld_vector_size; i++) {
      std::cout << " " << (int)overal_ld_time_dist[i];
    }
    std::cout << std::endl;
    std::cout << "ST_mem_lat_dist ";
    for (i = 0; i < st_vector_size; i++) {
      std::cout << " " << (int)overal_st_time_dist[i];
    }
    std::cout << std::endl;
  }
  void print_to_file(FILE* outfile) {
    unsigned i;
    calculate_dist();
    fprintf(outfile, "LDmemlatdist:");
    for (i = 0; i < ld_vector_size; i++) {
      fprintf(outfile, " %d", (int)ld_time_dist[i]);
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "STmemlatdist:");
    for (i = 0; i < st_vector_size; i++) {
      fprintf(outfile, " %d", (int)st_time_dist[i]);
    }
    fprintf(outfile, "\n");
  }
  void print_to_gzfile(gzFile outfile) {
    unsigned i;
    calculate_dist();
    gzprintf(outfile, "LDmemlatdist:");
    for (i = 0; i < ld_vector_size; i++) {
      gzprintf(outfile, " %d", (int)ld_time_dist[i]);
    }
    gzprintf(outfile, "\n");
    gzprintf(outfile, "STmemlatdist:");
    for (i = 0; i < st_vector_size; i++) {
      gzprintf(outfile, " %d", (int)st_time_dist[i]);
    }
    gzprintf(outfile, "\n");
  }
};

my_time_vector* g_my_time_vector;

void time_vector_create(int size) {
  g_my_time_vector = new my_time_vector(size, size);
}

void time_vector_print(void) { g_my_time_vector->print_dist(); }

void time_vector_print_interval2gzfile(gzFile outfile) {
  g_my_time_vector->print_to_gzfile(outfile);
}

#include "../gpgpu-sim/mem_fetch.h"

void time_vector_update(unsigned int uid, int slot, long int cycle, int type) {
  if ((type == READ_REQUEST) || (type == READ_REPLY)) {
    g_my_time_vector->update_ld(uid, slot, cycle);
  } else if ((type == WRITE_REQUEST) || (type == WRITE_ACK)) {
    g_my_time_vector->update_st(uid, slot, cycle);
  } else {
    abort();
  }
}

void check_time_vector_update(unsigned int uid, int slot, long int latency,
                              int type) {
  if ((type == READ_REQUEST) || (type == READ_REPLY)) {
    g_my_time_vector->check_ld_update(uid, slot, latency);
  } else if ((type == WRITE_REQUEST) || (type == WRITE_ACK)) {
    g_my_time_vector->check_st_update(uid, slot, latency);
  } else {
    abort();
  }
}
