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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include <algorithm>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <unordered_set>
#include "../abstract_hardware_model.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_latency_stat.h"

// std::pair<mem_fetch *, bool>: mf, reply_sent
typedef std::vector<std::pair<mem_fetch *, bool>> LRCEntry;
class mem_fetch;
class L2RequestCoalescer;

// FIFO queue with latency modeling - elements have a ready cycle.
// sim_cycle and tot_sim_cycle are non-owning references to gpgpu_sim's
// cycle counters, which outlive this queue.
template <typename T>
class LatencyQueue {
 public:
  struct entry_t {
    uint64_t ready_cycle;
    T data;
  };

  LatencyQueue(const std::string &name, unsigned max_size, unsigned latency,
               const unsigned long long &sim_cycle,
               const unsigned long long &tot_sim_cycle)
      : m_name(name),
        m_max_size(max_size),
        m_latency(latency),
        m_sim_cycle(sim_cycle),
        m_tot_sim_cycle(tot_sim_cycle) {}
  ~LatencyQueue() = default;

  // Check if queue is full
  bool full() const { return m_queue.size() >= m_max_size; }

  // Check if queue is empty
  bool empty() const { return m_queue.empty(); }

  // Get current queue size
  unsigned size() const { return m_queue.size(); }

  // Get max queue size
  unsigned max_size() const { return m_max_size; }

  // Get latency
  unsigned latency() const { return m_latency; }

  // Get name
  const std::string &name() const { return m_name; }

  // Push data - ready_cycle is automatically calculated from cycle reference
  void push(T data) {
    entry_t entry;
    entry.ready_cycle = m_sim_cycle + m_tot_sim_cycle + m_latency;
    entry.data = data;
    m_queue.push_back(entry);
  }

  // Get the front element (for inspection)
  const entry_t &front() const { return m_queue.front(); }

  // Get the front element (mutable)
  entry_t &front() { return m_queue.front(); }

  // Pop the front element
  void pop() { m_queue.pop_front(); }

  // Check if the front element is ready to be popped
  bool front_ready() const {
    if (m_queue.empty()) return false;
    return m_queue.front().ready_cycle <= m_sim_cycle + m_tot_sim_cycle;
  }

  // Pop and return the front element if ready, otherwise return default T
  // T pop_ready() {
  //   if (!front_ready()) return T();
  //   T data = m_queue.front().data;
  //   m_queue.pop_front();
  //   return data;
  // }

 private:
  std::string m_name;
  unsigned m_max_size;
  unsigned m_latency;
  const unsigned long long &m_sim_cycle;
  const unsigned long long &m_tot_sim_cycle;
  std::deque<entry_t> m_queue;
};

// Encapsulates the four queues for chiplet-to-chiplet communication
class chiplet_icnt {
 public:
  chiplet_icnt() = default;

  // Initialize the queues based on chiplet ID
  // Chiplet 0 sends to 1 via request_0_to_1/reply_0_to_1
  // Chiplet 1 sends to 0 via request_1_to_0/reply_1_to_0
  void init(uint32_t chiplet_id, uint32_t local_sub_partition_id,
            std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
            std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
            std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
            std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0) {
    if (chiplet_id == 0) {
      m_to_peer_request = &request_0_to_1[local_sub_partition_id];
      m_to_peer_reply = &reply_0_to_1[local_sub_partition_id];
      m_from_peer_request = &request_1_to_0[local_sub_partition_id];
      m_from_peer_reply = &reply_1_to_0[local_sub_partition_id];
    } else {
      m_to_peer_request = &request_1_to_0[local_sub_partition_id];
      m_to_peer_reply = &reply_1_to_0[local_sub_partition_id];
      m_from_peer_request = &request_0_to_1[local_sub_partition_id];
      m_from_peer_reply = &reply_0_to_1[local_sub_partition_id];
    }
  }

  // Accessors for the queues
  LatencyQueue<mem_fetch *> *to_peer_request() const {
    return m_to_peer_request;
  }
  LatencyQueue<mem_fetch *> *to_peer_reply() const { return m_to_peer_reply; }
  LatencyQueue<mem_fetch *> *from_peer_request() const {
    return m_from_peer_request;
  }
  LatencyQueue<mem_fetch *> *from_peer_reply() const {
    return m_from_peer_reply;
  }

 private:
  LatencyQueue<mem_fetch *> *m_to_peer_request = nullptr;
  LatencyQueue<mem_fetch *> *m_to_peer_reply = nullptr;
  LatencyQueue<mem_fetch *> *m_from_peer_request = nullptr;
  LatencyQueue<mem_fetch *> *m_from_peer_reply = nullptr;
};

/// Models second level shared cache with global write-back
/// and write-allocate policies
class l2_cache : public data_cache {
 public:
  l2_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_interface *chiplet_port,
           mem_fetch_allocator *mfcreator, enum mem_fetch_status status,
           class gpgpu_sim *gpu, enum cache_gpu_level level,
           uint32_t sub_partition_id, uint32_t chiplet_id,
           class memory_stats_t *stats)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L2_WR_ALLOC_R, L2_WRBK_ACC, gpu, level) {
    m_chiplet_port = chiplet_port;
    m_sub_partition_id = sub_partition_id;
    m_chiplet_id = chiplet_id;
    m_mem_stats = stats;
  }

  virtual ~l2_cache() {}

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);
  void cycle();

 private:
  mem_fetch_interface *m_chiplet_port;
  uint32_t m_sub_partition_id;
  class memory_stats_t *m_mem_stats;
};

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu,
                        std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
                        std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
                        std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
                        std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }
  unsigned get_chiplet_id() const { return m_chiplet_id; }

  class gpgpu_sim *get_mgpu() const { return m_gpu; }

 private:
  unsigned m_id;
  unsigned m_chiplet_id;
  const memory_config *m_config;
  class memory_stats_t *m_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  class gpgpu_sim *m_gpu;
};

class memory_sub_partition {
 public:
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu,
                       std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
                       std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
                       std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
                       std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }
  unsigned get_chiplet_id() const { return m_chiplet_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  bool lrc_full() const;
  bool lrc_full(unsigned size) const;
  L2RequestCoalescer *get_lrc() { return m_lrc; }
  bool lrc_enabled() const { return m_lrc != nullptr; }
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  void push_direct(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

  mem_fetch *get_chiplet_req() const;

  bool push_chiplet_reply(mem_fetch *mf);

  bool handle_chiplet_reply();

  void forward_write_to_peer_chiplet(mem_fetch *mf);

 private:
  // data
  unsigned m_id;  //< the global sub partition ID
  unsigned m_chiplet_id;
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class Chipletinterface *m_chiplet_interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue
  chiplet_icnt m_chiplet_icnt;

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::unordered_set<mem_fetch *> m_request_tracker;

  friend class L2interface;
  friend class Chipletinterface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;

  // Load request coalescer
  L2RequestCoalescer *m_lrc;

  bool m_chiplet_disabled;
};

class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  virtual void push(mem_fetch *mf);

 private:
  memory_sub_partition *m_unit;
};

class Chipletinterface : public mem_fetch_interface {
 public:
  Chipletinterface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~Chipletinterface() {}
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_chiplet_icnt.to_peer_request()->full();
  }
  virtual void push(mem_fetch *mf);

 private:
  memory_sub_partition *m_unit;
};

class L2RequestCoalescer {
 public:
  L2RequestCoalescer(unsigned max_entries, unsigned max_merged)
      : m_max_entries(max_entries),
        m_max_merged(max_merged),
        m_total_coalesced_count(0) {}
  ~L2RequestCoalescer() = default;

  // Insert a mem_fetch assuming there is space left in the queue
  // Either in a new entry or merge with existing entry
  // Return true if a new entry is allocated, false if merged with existing
  // entry
  bool insert(new_addr_type sector_addr, mem_fetch *mf);

  // Check if there is still space left in the queue
  bool full() { return m_lrc_queue.size() >= m_max_entries; }
  // Check if there is still space left in the queue to allocate several entries
  bool full(unsigned size) {
    return m_lrc_queue.size() + size >= m_max_entries;
  }

  // Get the size of the LRC queue
  unsigned size() const { return m_lrc_queue.size(); }

  // Get the maximum coalescing size across all entries in the LRC queue
  // unsigned max_coalescing_count() const {
  //   unsigned max_count = 0;
  //   for (const auto &entry : m_lrc_queue) {
  //     max_count = std::max<unsigned>(max_count, entry.second.size());
  //   }
  //   return max_count;
  // }

  // Get the average coalescing size across all entries in the LRC queue
  float avg_coalescing_count() const {
    if (m_lrc_queue.empty()) {
      return 0.0;
    }
    return static_cast<float>(m_total_coalesced_count) /
           static_cast<float>(m_lrc_queue.size());
  }

  /**
   * @brief Get the matched entry object
   *
   * @param sector_addr
   * @param uid
   * @return LRCEntry&
   */
  LRCEntry &get_entry(new_addr_type sector_addr, unsigned uid);

  /**
   * @brief Remove the matched entry object
   *
   * @param sector_addr
   * @param uid
   */
  void remove_entry(new_addr_type sector_addr, unsigned uid);

 protected:
  // Size of the LRC queue
  const unsigned m_max_entries;
  // Max number of requests that can be merged in a single entry
  const unsigned m_max_merged;

  // Queue for coalescing requests
  // Each entry is indexed by sector address
  // Each entry is a vector of mem_fetch pointers
  // Use multimap as LRC can support multiple entries with
  // same sector address as long as each entry belong to
  // different GPC, but we are not modeling GPC here,
  // so we assume LRC can coalesce with all requests from
  // all GPCs
  std::multimap<new_addr_type, LRCEntry> m_lrc_queue;

  // Running total of all entry sizes for O(1) average calculation
  unsigned m_total_coalesced_count;
};

#endif
