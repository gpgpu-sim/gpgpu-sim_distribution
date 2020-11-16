// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#ifndef memory_h_INCLUDED
#define memory_h_INCLUDED

#include "../abstract_hardware_model.h"

#include "../tr1_hash_map.h"
#define mem_map tr1_hash_map
#if tr1_hash_map_ismap == 1
#define MEM_MAP_RESIZE(hash_size)
#else
#define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
#endif

#include <assert.h>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>

typedef address_type mem_addr_t;

typedef struct _eviction_t {
  mem_addr_t addr;
  size_t size;
  unsigned long long cycle;
  uint32_t access_counter;
  uint8_t RW;
} eviction_t;

#define MEM_BLOCK_SIZE (4 * 1024)

template <unsigned BSIZE>
class mem_storage {
 public:
  mem_storage(const mem_storage &another) {
    m_data = (unsigned char *)calloc(1, BSIZE);
    memcpy(m_data, another.m_data, BSIZE);

    // initialize page as unmanaged
    managed = false;

    // initialize page flags to default value
    valid = false;
    dirty = false;
    access = false;

    counter = 0;
  }
  mem_storage() { 
    m_data = (unsigned char *)calloc(1, BSIZE); 

    // initialize page as unmanaged
    managed = false;

    // initialize page flags to default value
    valid = false;
    dirty = false;
    access = false;

    counter = 0;
  }
  ~mem_storage() { free(m_data); }

  void write(unsigned offset, size_t length, const unsigned char *data) {
    assert(offset + length <= BSIZE);
    memcpy(m_data + offset, data, length);
  }

  void read(unsigned offset, size_t length, unsigned char *data) const {
    assert(offset + length <= BSIZE);
    memcpy(data, m_data + offset, length);
  }

  void print(const char *format, FILE *fout) const {
    unsigned int *i_data = (unsigned int *)m_data;
    for (int d = 0; d < (BSIZE / sizeof(unsigned int)); d++) {
      if (d % 1 == 0) {
        fprintf(fout, "\n");
      }
      fprintf(fout, format, i_data[d]);
      fprintf(fout, " ");
    }
    fprintf(fout, "\n");
    fflush(fout);
  }

  // set the flag of managed into true in order to distinguish it from the
  // unmanaged allocation
  void set_managed() { managed = true; }
  bool is_managed() { return managed; }

  // methods to query and modify page table flags
  bool is_valid() { return valid; }
  void validate_page() { valid = true; }
  void invalidate_page() { valid = false; }

  void set_dirty() { dirty = true; }
  void clear_dirty() { dirty = false; }
  bool is_dirty() { return dirty; }

  void set_access() { access = true; }
  void clear_access() { access = false; }
  bool is_access() { return access; }

 private:
  unsigned m_nbytes;
  unsigned char *m_data;

  // flag to differentiate whether a page is a managed allocation or traditional
  // unmanged by deafult it is false to denote cudaMalloc and cudaMallocArray on
  // managed allocation set this to true check this at the generation of
  // mem_fetch to determine which path to take managed may take the longer
  // latency path
  bool managed;

  // flags for page table
  bool valid;

  bool dirty;

  bool access;

  unsigned counter;
};

class ptx_thread_info;
class ptx_instruction;

class memory_space {
 public:
  virtual ~memory_space() {}
  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI) = 0;
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data) = 0;
  virtual void read(mem_addr_t addr, size_t length, void *data) const = 0;
  virtual void print(const char *format, FILE *fout) const = 0;
  virtual void set_watch(addr_t addr, unsigned watchpoint) = 0;

  // method to find out whether or not to follow the managed time simulation
  virtual bool is_page_managed(mem_addr_t addr, size_t length) = 0;
  // method to set the pages as managed allocation
  virtual void set_pages_managed(mem_addr_t addr, size_t length) = 0;

  // method to allocate page(s) from free pages and change the count of free
  // pages
  virtual bool alloc_page_by_byte(size_t size) = 0;
  virtual void alloc_pages(size_t num) = 0;
  virtual void free_pages(size_t num) = 0;
  virtual size_t get_free_pages() = 0;

  virtual void set_page_dirty(mem_addr_t pg_index) = 0;
  virtual bool is_page_dirty(mem_addr_t pg_index) = 0;
  virtual void clear_page_dirty(mem_addr_t pg_index) = 0;

  virtual void set_page_access(mem_addr_t pg_index) = 0;
  virtual bool is_page_access(mem_addr_t pg_index) = 0;
  virtual void clear_page_access(mem_addr_t pg_index) = 0;

  // methods to query page table
  virtual void validate_page(mem_addr_t pg_index) = 0;
  virtual void invalidate_page(mem_addr_t pg_index) = 0;
  virtual std::list<mem_addr_t> get_faulty_pages(mem_addr_t addr,
                                                 size_t length) = 0;
  virtual mem_addr_t get_page_num(mem_addr_t addr) = 0;

  virtual size_t get_data_size(mem_addr_t addr) = 0;
  virtual size_t get_page_size() = 0;
  virtual mem_addr_t get_mem_addr(mem_addr_t pg_index) = 0;
  virtual bool is_valid(mem_addr_t pg_index) = 0;
  virtual bool should_evict_page(size_t read_stage_queue_size,
                                 size_t write_stage_queue_size,
                                 float eviction_buffer_percentage) = 0;
  virtual float get_projected_occupancy(size_t read_stage_queue_size,
                                        size_t write_stage_queue_size,
                                        float eviction_buffer_percentage) = 0;

  virtual void reset() = 0;
};

template <unsigned BSIZE>
class memory_space_impl : public memory_space {
 public:
  memory_space_impl(std::string name, unsigned hash_size,
                    unsigned long long gddr_size = 0);

  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI);
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data);
  virtual void read(mem_addr_t addr, size_t length, void *data) const;
  virtual void print(const char *format, FILE *fout) const;

  virtual void set_watch(addr_t addr, unsigned watchpoint);

  // method to find out whether or not to follow the managed time simulation
  virtual bool is_page_managed(mem_addr_t addr, size_t length);
  // method to set the pages as managed allocation
  virtual void set_pages_managed(mem_addr_t addr, size_t length);

  // methods to query page table
  virtual void validate_page(mem_addr_t pg_index);
  virtual void invalidate_page(mem_addr_t pg_index);
  virtual std::list<mem_addr_t> get_faulty_pages(mem_addr_t addr,
                                                 size_t length);
  virtual mem_addr_t get_page_num(mem_addr_t addr);

  // methods to implement gddr size constraint
  virtual bool alloc_page_by_byte(size_t size);
  virtual void alloc_pages(size_t num);
  virtual void free_pages(size_t num);
  virtual size_t get_free_pages();

  virtual void set_page_dirty(mem_addr_t pg_index);
  virtual bool is_page_dirty(mem_addr_t pg_index);
  virtual void clear_page_dirty(mem_addr_t pg_index);

  virtual void set_page_access(mem_addr_t pg_index);
  virtual bool is_page_access(mem_addr_t pg_index);
  virtual void clear_page_access(mem_addr_t pg_index);

  virtual size_t get_data_size(mem_addr_t addr);
  virtual size_t get_page_size();
  virtual mem_addr_t get_mem_addr(mem_addr_t pg_index);

  virtual bool is_valid(mem_addr_t pg_index);
  virtual bool should_evict_page(size_t read_stage_queue_size,
                                 size_t write_stage_queue_size,
                                 float eviction_buffer_percentage);
  virtual float get_projected_occupancy(size_t read_stage_queue_size,
                                        size_t write_stage_queue_size,
                                        float eviction_buffer_percentage);

  virtual void reset();

 private:
  void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                         void *data) const;
  std::string m_name;
  unsigned m_log2_block_size;

  // map_t m_data closely resembles to a page table
  // the dictionary is keyed by the virtual address
  // mem_storage acts as the physical page
  typedef mem_map<mem_addr_t, mem_storage<BSIZE> > map_t;
  map_t m_data;

  // variable to store total number of 8KB pages in global memory
  // calculated based on the GDDR5 size specified in config
  // it is used to enforce size restriction on both managed and unmanaged malloc
  // it should be decremented on every allocation either managed or unmanaged
  // i.e., gpu_malloc, gpu_mallocmanaged, gpu_mallocarray
  size_t num_free_pages;

  // the size of gddr in number of pages
  const size_t num_gddr_pages;
  
  std::map<unsigned, mem_addr_t> m_watchpoints;
};

#endif
