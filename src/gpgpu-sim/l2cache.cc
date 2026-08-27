// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "basic_components.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  return mf;
}
memory_partition_unit::memory_partition_unit(
    unsigned partition_id, const memory_config *config,
    class memory_stats_t *stats, class gpgpu_sim *gpu,
    std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
    std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0)
    : m_id(partition_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config),
      m_gpu(gpu) {
  unsigned partitions_per_chiplet =
      config->m_n_mem / config->m_address_mapping.get_n_chiplet_partition();
  m_chiplet_id = partition_id / partitions_per_chiplet;

  m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] = new memory_sub_partition(
        sub_partition_id, m_config, stats, gpu, request_0_to_1, request_1_to_0,
        reply_0_to_1, reply_1_to_0);
    assert(m_sub_partition[p]->get_chiplet_id() == m_chiplet_id);
  }
}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = false;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition

  unsigned dram_cycles = m_config->simple_dram_clock_multiplier;
  for (unsigned c = 0; c < dram_cycles; c++) {
    if (m_dram_latency_queue.empty()) break;
    if (((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
         m_dram_latency_queue.front().ready_cycle)) {
      mem_fetch *mf_return = m_dram_latency_queue.front().req;
      if (mf_return->get_access_type() != L1_WRBK_ACC &&
          mf_return->get_access_type() != L2_WRBK_ACC) {
        mf_return->set_reply();
        unsigned dest_global_spid = mf_return->get_sub_partition_id();
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
          // Update stats for simple dram when request is actually dequeued
          m_stats->memlatstat_dram_access(mf_return);
          if (mf_return->get_access_type() == L1_WRBK_ACC) {
            m_sub_partition[dest_spid]->set_done(mf_return);
            delete mf_return;
          } else {
            m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
            mf_return->set_status(
                IN_PARTITION_DRAM_TO_L2_QUEUE,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            m_arbitration_metadata.return_credit(dest_spid);
            MEMPART_DPRINTF(
                "mem_fetch request %p return from dram to sub partition %d\n",
                mf_return, dest_spid);
          }
          m_dram_latency_queue.pop_front();
        } else {
          break;
        }
      } else {
        // Update stats for simple dram when request is actually dequeued
        m_stats->memlatstat_dram_access(mf_return);
        this->set_done(mf_return);
        delete mf_return;
        m_dram_latency_queue.pop_front();
      }
    } else {
      // The front of the DRAM latency queue is not ready yet, so we cannot
      // process any more requests in this cycle
      break;
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  unsigned processed = 0;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) {
        continue;
      }

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      processed++;
      if (processed >= dram_cycles) {
        break;  // the DRAM should only accept these request per cycle
      }
    }
  }
  //}
}

void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      m_dram->return_queue_pop();
    }
  } else {
    m_dram->return_queue_pop();
  }

  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  m_dram->print(fp);
}

memory_sub_partition::memory_sub_partition(
    unsigned sub_partition_id, const memory_config *config,
    class memory_stats_t *stats, class gpgpu_sim *gpu,
    std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
    std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0) {
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;
  m_chiplet_id = sub_partition_id / config->m_n_sub_partition_per_chiplet;
  uint32_t local_sub_partition_id =
      sub_partition_id % config->m_n_sub_partition_per_chiplet;
  m_chiplet_icnt.init(m_chiplet_id, local_sub_partition_id, request_0_to_1,
                      request_1_to_0, reply_0_to_1, reply_1_to_0);

  if (gpu->getShaderCoreConfig()->n_chiplet == 1) {
    m_chiplet_disabled = true;
  } else {
    m_chiplet_disabled = false;
  }

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_chiplet_interface = new Chipletinterface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_chiplet_interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE,
                             m_id, m_chiplet_id, stats);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  wb_addr = -1;

  // Initialize the LRC if enabled
  if (m_config->lrc_enabled) {
    m_lrc = new L2RequestCoalescer(m_config->lrc_max_entries,
                                   m_config->lrc_max_merged);
  } else {
    m_lrc = nullptr;
  }
}

memory_sub_partition::~memory_sub_partition() {
  delete m_lrc;
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  if (!m_config->m_L2_config.disabled()) {
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      mem_fetch *mf = m_L2cache->next_access();
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      } else {
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
        }
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  if (!m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        m_dram_L2_queue->pop();
      }
    } else if (!m_L2_icnt_queue->full()) {
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  mem_fetch *mf = nullptr;

  // Alternate priority each call to balance latency
  bool prioritize_chiplet = (cycle % 2 == 0);
  bool from_icnt = false;

  if (prioritize_chiplet) {
    mf = get_chiplet_req();
    if (!mf && !m_icnt_L2_queue->empty()) {
      mf = m_icnt_L2_queue->top();
      from_icnt = true;
    }
  } else {
    if (!m_icnt_L2_queue->empty()) {
      mf = m_icnt_L2_queue->top();
      from_icnt = true;
    }
    if (!mf) mf = get_chiplet_req();
  }

  if (mf) {
    if (mf->get_type() == WRITE_FORWARD) {
      // pseudo-coherence update from other chiplet, directly update L2 tag.
      assert(mf->is_write());
      assert(from_icnt == false);  // can only come from chiplet
      new_addr_type addr = mf->get_addr();
      mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
      force_l2_tag_update(addr, cycle, sector_mask);
      m_chiplet_icnt.from_peer_request()->pop();
      delete mf;
    } else if (!m_config->m_L2_config.disabled() &&
               ((m_config->m_L2_texure_only && mf->istexture()) ||
                (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_queue->full();
      bool port_free = m_L2cache->data_port_free();
      // can only accept write if request queue has space for write forward
      bool interchip_free =
          !mf->is_write() || !m_chiplet_icnt.to_peer_request()->full();
      if (!interchip_free) {
        ++m_stats->chiplet_write_fail[m_id];
      }
      if (!output_full && port_free && interchip_free) {
        bool accepted = false;
        std::list<cache_event> events;
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);

        if (status == HIT) {
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            accepted = true;
          } else {
            assert(write_sent);
            accepted = true;
          }
        } else if (status != RESERVATION_FAIL) {
          if (!m_chiplet_disabled && mf->is_write()) {
            forward_write_to_peer_chiplet(mf);
          }
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
          }
          // L2 cache accepted request
          accepted = true;
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
        if (accepted) {
          if (from_icnt) {
            m_icnt_L2_queue->pop();
          } else {
            m_chiplet_icnt.from_peer_request()->pop();
          }
        }
      }
    } else if (!m_L2_dram_queue->full()) {
      // L2 is disabled or non-texture access to texture-only L2
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      if (from_icnt) {
        m_icnt_L2_queue->pop();
      } else {
        m_chiplet_icnt.from_peer_request()->pop();
      }
    }
  }

  // ROP delay queue
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

bool memory_sub_partition::lrc_full() const {
  return m_lrc != nullptr && m_lrc->full();
}

bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::lrc_full(unsigned size) const {
  return m_lrc != nullptr && m_lrc->full(size);
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (auto r = m_request_tracker.begin(); r != m_request_tracker.end();
         ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      // Here we insert the read request into the LRC queue
      // And we only send request down if a new entry is allocated
      // For write request, it is not coalesced with LRC
      bool allocated = true;
      if (m_lrc && !(req->is_write())) {
        // insert into LRC queue
        // if new entry is allocated, we need to send request down
        allocated = m_lrc->insert(req->get_addr(), req);
        // Increment LRC stats for ICNT to LRC
        m_stats->add_icnt_to_lrc_sectors(get_id(), req);
      }

      // Either a new entry is allocated in LRC or we disable it
      // either way, this mem_fetch needs to be sent down to L2
      if (allocated) {
        m_request_tracker.insert(req);
        if (req->istexture()) {
          m_icnt_L2_queue->push(req);
          req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        } else {
          rop_delay_t r;
          r.req = req;
          r.ready_cycle = cycle + m_config->rop_latency;
          m_rop.push(r);
          req->set_status(IN_PARTITION_ROP_DELAY,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        }

        // Increment LRC stats for LRC to L2 if LRC is enabled
        if (m_lrc && !(req->is_write())) {
          // Increment LRC stats for LRC to L2
          m_stats->add_lrc_to_l2_sectors(get_id(), req);
          // Update the size of the LRC queue
          m_stats->update_lrc_queue_size(get_id(), m_lrc->size());
          // Update the maximum coalescing size for this sub-partition
          // m_stats->update_current_max_coalesced_count(
          //     get_id(), m_lrc->max_coalescing_count());
          // Update the average coalescing size for this sub-partition
          m_stats->update_current_avg_coalesced_count(
              get_id(), m_lrc->avg_coalescing_count());
        }
      }
    }
  }
}

void memory_sub_partition::push_direct(mem_fetch *m_req,
                                       unsigned long long cycle) {
  assert(m_req);
  assert(!m_icnt_L2_queue->full());
  if (m_config->m_L2_config.m_cache_type == SECTOR) {
    // for debug. remove this after verification
    assert(m_req->get_data_size() ==
           SECTOR_SIZE);  // should be broken down to sector access already
  }
  m_icnt_L2_queue->push(m_req);
  m_req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE, cycle);
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}

bool L2RequestCoalescer::insert(new_addr_type sector_addr, mem_fetch *mf) {
  // This assumes there are space left in the queue
  // First we search the queue for mergeable entries
  auto entries = m_lrc_queue.equal_range(sector_addr);
  // Then we iterate through existing entries to find mergeable entry
  for (auto it = entries.first; it != entries.second; ++it) {
    if (it->first == sector_addr && it->second.size() < m_max_merged) {
      // Found the sector address in the queue and the entry still have space
      // left to merge
      it->second.push_back(std::make_pair(mf, false));
      m_total_coalesced_count++;
      return false;
    }
  }

  // Now we allocate a new entry
  LRCEntry new_entry;
  new_entry.push_back(std::make_pair(mf, false));
  m_lrc_queue.insert(std::make_pair(sector_addr, new_entry));
  m_total_coalesced_count++;
  assert(m_lrc_queue.size() <= m_max_entries &&
         "LRC queue is full in insert()");
  return true;
}

LRCEntry &L2RequestCoalescer::get_entry(new_addr_type sector_addr,
                                        unsigned uid) {
  auto entries = m_lrc_queue.equal_range(sector_addr);
  for (auto it = entries.first; it != entries.second; ++it) {
    auto &entry = it->second;
    if (entry.front().first->get_request_uid() == uid) {
      return entry;
    }
  }
  assert(false && "No matched entry found");
}

void L2RequestCoalescer::remove_entry(new_addr_type sector_addr, unsigned uid) {
  auto entries = m_lrc_queue.equal_range(sector_addr);
  for (auto it = entries.first; it != entries.second; ++it) {
    if (it->second.front().first->get_request_uid() == uid) {
      m_total_coalesced_count -= it->second.size();
      m_lrc_queue.erase(it);
      return;
    }
  }
  assert(false && "No matched entry found");
}
void L2interface::push(mem_fetch *mf) {
  mf->set_status(
      IN_PARTITION_L2_TO_DRAM_QUEUE,
      m_unit->m_gpu->gpu_sim_cycle + m_unit->m_gpu->gpu_tot_sim_cycle);
  m_unit->m_L2_dram_queue->push(mf);
}

void Chipletinterface::push(mem_fetch *mf) {
  // Reclassify for separate statistics
  if (mf->get_access_type() == GLOBAL_ACC_R) {
    mf->set_access_type(CHIPLET_ACC_R);
  }
  mf->set_status(
      IN_PARTITION_L2_TO_CHIPLET_REQUEST_QUEUE,
      m_unit->m_gpu->gpu_sim_cycle + m_unit->m_gpu->gpu_tot_sim_cycle);
  m_unit->m_chiplet_icnt.to_peer_request()->push(mf);
  ++m_unit->m_stats->interchip_read_requests;
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

void l2_cache::cycle() {
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    uint32_t chiplet_idx = mf->get_dest_chiplet();
    if (chiplet_idx == m_chiplet_id || mf->get_type() == WRITE_REQUEST) {
      // This request belongs to the same chiplet, send it to DRAM
      // for writes, a seperate WRITE_FORWARD is generated
      // always write to local
      if (!m_memport->full(mf->size(), mf->get_is_write())) {
        m_miss_queue.pop_front();
        m_memport->push(mf);
      } else {
        ++m_mem_stats->L2_dram_queue_full[m_sub_partition_id];
      }
    } else {
      // This request belongs to a different chiplet, send it to the chiplet
      // interface
      if (!m_chiplet_port->full(mf->size(), mf->get_is_write())) {
        m_miss_queue.pop_front();
        unsigned sub_partition_per_chiplet =
            m_gpu->getMemoryConfig()->m_n_sub_partition_per_chiplet;
        unsigned peer_l2 = m_sub_partition_id % sub_partition_per_chiplet +
                           (chiplet_idx * sub_partition_per_chiplet);
        mf->set_partition(peer_l2);
        mf->set_chip(
            peer_l2 /
            m_gpu->getMemoryConfig()->m_n_sub_partition_per_memory_channel);
        m_chiplet_port->push(mf);
      } else {
        ++m_mem_stats->chiplet_queue_full[m_sub_partition_id];
      }
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}
mem_fetch *memory_sub_partition::get_chiplet_req() const {
  if (!m_chiplet_icnt.from_peer_request()->front_ready()) {
    return nullptr;
  }

  mem_fetch *mf = m_chiplet_icnt.from_peer_request()->front().data;
  assert(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_FORWARD);
  assert(mf->get_dest_chiplet() == m_chiplet_id);
  return mf;
}
bool memory_sub_partition::push_chiplet_reply(mem_fetch *mf) {
  // note that here is still request queue
  // request is for sending, reply is for receiving
  if (m_chiplet_icnt.to_peer_reply()->full()) {
    return false;
  }
  m_chiplet_icnt.to_peer_reply()->push(mf);
  return true;
}
bool memory_sub_partition::handle_chiplet_reply() {
  if (!m_chiplet_icnt.from_peer_reply()->front_ready()) {
    return false;
  }
  if (m_dram_L2_queue->full()) {
    return false;
  }

  mem_fetch *mf = m_chiplet_icnt.from_peer_reply()->front().data;
  assert(mf->get_type() == READ_REPLY);
  assert(mf->get_src_chiplet() == m_chiplet_id);
  m_request_tracker.erase(mf);
  m_chiplet_icnt.from_peer_reply()->pop();
  m_dram_L2_queue->push(mf);
  return true;
}

void memory_sub_partition::forward_write_to_peer_chiplet(mem_fetch *mf) {
  uint64_t cycle = m_gpu->global_cycle();
  mem_fetch *new_mf = new mem_fetch(
      mf->get_mem_access(), NULL, mf->get_streamID(), mf->get_data_size(),
      mf->get_wid(), mf->get_sid(), mf->get_tpc(), m_config, cycle, mf);
  // Reclassify for separate statistics
  if (new_mf->get_access_type() == GLOBAL_ACC_W) {
    new_mf->set_access_type(CHIPLET_ACC_W);
  }

  // Assuming 2 chiplets, flip it
  assert(m_gpu->getShaderCoreConfig()->n_chiplet <= 2);
  uint32_t peer_chiplet_id = m_chiplet_id ^ 1;
  new_mf->set_write_interchip(peer_chiplet_id);
  new_mf->set_status(IN_PARTITION_L2_TO_CHIPLET_REQUEST_QUEUE, cycle);
  if (m_chiplet_icnt.to_peer_request()->full()) {
    delete new_mf;
    return;
  }
  m_chiplet_icnt.to_peer_request()->push(new_mf);
  ++m_stats->interchip_write_requests;
}
