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

#include "memory.h"
#include <stdlib.h>
#include "../../libcuda/gpgpu_context.h"
#include "../debug.h"

template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size,
                                            unsigned long long gddr_size)
: num_gddr_pages(gddr_size / BSIZE) {
  m_name = name;
  MEM_MAP_RESIZE(hash_size);

  m_log2_block_size = -1;
  for (unsigned n = 0, mask = 1; mask != 0; mask <<= 1, n++) {
    if (BSIZE & mask) {
      assert(m_log2_block_size == (unsigned)-1);
      m_log2_block_size = n;
    }
  }
  assert(m_log2_block_size != (unsigned)-1);

  // initialize the number of free pages based on size of GDDR5 and page size
  num_free_pages = num_gddr_pages;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  m_data[index].write(offset, length, (const unsigned char *)data);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  mem_addr_t index = addr >> m_log2_block_size;

  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    m_data[index].write(offset, nbytes, (const unsigned char *)data);
  } else {
    // slow route for inter-block access
    unsigned nbytes_remain = length;
    unsigned src_offset = 0;
    mem_addr_t current_addr = addr;

    while (nbytes_remain > 0) {
      unsigned offset = current_addr & (BSIZE - 1);
      mem_addr_t page = current_addr >> m_log2_block_size;
      mem_addr_t access_limit = offset + nbytes_remain;
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }

      size_t tx_bytes = access_limit - offset;
      m_data[page].write(offset, tx_bytes,
                         &((const unsigned char *)data)[src_offset]);

      // advance pointers
      src_offset += tx_bytes;
      current_addr += tx_bytes;
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
  if (!m_watchpoints.empty()) {
    std::map<unsigned, mem_addr_t>::iterator i;
    for (i = m_watchpoints.begin(); i != m_watchpoints.end(); i++) {
      mem_addr_t wa = i->second;
      if (((addr <= wa) && ((addr + length) > wa)) ||
          ((addr > wa) && (addr < (wa + 4))))
        thd->get_gpu()->gpgpu_ctx->the_gpgpusim->g_the_gpu->hit_watchpoint(
            i->first, thd, pI);
    }
  }
}

template <unsigned BSIZE> void memory_space_impl<BSIZE>::reset() {
  num_free_pages = num_gddr_pages;
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_managed(mem_addr_t addr, size_t length) {
  mem_addr_t page_index = get_page_num(addr + length - 1);
  return m_data[page_index].is_managed();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_pages_managed(mem_addr_t addr,
                                                 size_t length) {
  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);
  while (start_page <= end_page) {
    m_data[start_page].set_managed();
    start_page++;
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {
  if ((addr + length) > (blk_idx + 1) * BSIZE) {
    printf(
        "GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : "
        "addr=0x%x, length=%zu\n",
        m_name.c_str(), addr, length);
    printf(
        "GPGPU-Sim PTX: (addr+length)=0x%lx > 0x%x=(index+1)*BSIZE, "
        "index=0x%x, BSIZE=0x%x\n",
        (addr + length), (blk_idx + 1) * BSIZE, blk_idx, BSIZE);
    throw 1;
  }
  typename map_t::const_iterator i = m_data.find(blk_idx);
  if (i == m_data.end()) {
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0;
    // printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized
    // memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
  } else {
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    i->second.read(offset, nbytes, (unsigned char *)data);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  mem_addr_t index = addr >> m_log2_block_size;
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    read_single_block(index, addr, length, data);
  } else {
    // slow route for inter-block access
    unsigned nbytes_remain = length;
    unsigned dst_offset = 0;
    mem_addr_t current_addr = addr;

    while (nbytes_remain > 0) {
      unsigned offset = current_addr & (BSIZE - 1);
      mem_addr_t page = current_addr >> m_log2_block_size;
      mem_addr_t access_limit = offset + nbytes_remain;
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }

      size_t tx_bytes = access_limit - offset;
      read_single_block(page, current_addr, tx_bytes,
                        &((unsigned char *)data)[dst_offset]);

      // advance pointers
      dst_offset += tx_bytes;
      current_addr += tx_bytes;
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
  printf("Yechen :: memory.cc :: Reading %zu bytes from address 0x%x in space %s: ", length, addr, m_name.c_str());
  for (int i=0; i < length; i++) {
      printf("%hhu ", ((unsigned char *)data)[i]);
  }
  printf("\n");
  fflush(stdout);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::print(const char *format, FILE *fout) const {
  typename map_t::const_iterator i_page;

  for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) {
    fprintf(fout, "%s %08x:", m_name.c_str(), i_page->first);
    i_page->second.print(format, fout);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_watch(addr_t addr, unsigned watchpoint) {
  m_watchpoints[watchpoint] = addr;
}

// get page number from a virtual address
template <unsigned BSIZE>
mem_addr_t memory_space_impl<BSIZE>::get_page_num(mem_addr_t addr) {
  return addr >> m_log2_block_size;
}

// check whether the valid flag of corresponding physical page is set or not
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_valid(mem_addr_t pg_index) {
  // asserts whether the physical page is allocated.
  // should never happen as they are allocated while memcpy.
  assert(m_data.find(pg_index) != m_data.end());
  return m_data[pg_index].is_valid();
}

// set the valid flag of corresponding physical page
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::validate_page(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].validate_page();
}

// clear the valid flag of corresponding physical page
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::invalidate_page(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].invalidate_page();
}

// a variable accessed by a memory address and the datatype size may exceed a
// page boundary method returns list of page numbers if at all they are faulty
// or invalid
template <unsigned BSIZE>
std::list<mem_addr_t>
memory_space_impl<BSIZE>::get_faulty_pages(mem_addr_t addr, size_t length) {
  std::list<mem_addr_t> page_list;

  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);

  while (start_page <= end_page) {
    if (!is_valid(start_page)) {
      page_list.push_back(start_page);
    }
    start_page++;
  }

  return page_list;
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::alloc_page_by_byte(size_t size) {
  size_t page_num = (size - 1) / BSIZE + 1;
  if (num_free_pages >= page_num) {
    num_free_pages -= page_num;
    return true;
  } else {
    return false;
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::alloc_pages(size_t num) {
  assert(num_free_pages >= num);
  num_free_pages -= num;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::free_pages(size_t num) {
  num_free_pages += num;
}

template <unsigned BSIZE> size_t memory_space_impl<BSIZE>::get_free_pages() {
  return num_free_pages;
}

// if the already allocated pages and about to allocate pages(in read stage
// queue) reaches the buffer size in gddr, then it should start eviction
// procedure
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::should_evict_page(
    size_t read_stage_queue_size, size_t write_stage_queue_size,
    float eviction_buffer_percentage) {
  return ((float)(write_stage_queue_size + num_gddr_pages)) <
         ((((float)num_gddr_pages) * eviction_buffer_percentage / 100) +
          ((float)(num_gddr_pages - num_free_pages + read_stage_queue_size)));
}

template <unsigned BSIZE>
float memory_space_impl<BSIZE>::get_projected_occupancy(
    size_t read_stage_queue_size, size_t write_stage_queue_size,
    float eviction_buffer_percentage) {
  return ((((float)num_gddr_pages) * eviction_buffer_percentage / 100) +
          ((float)(num_gddr_pages - num_free_pages + read_stage_queue_size))) /
         ((float)(write_stage_queue_size + num_gddr_pages));
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_page_dirty(mem_addr_t pg_index) {
  m_data[pg_index].set_dirty();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::clear_page_dirty(mem_addr_t pg_index) {
  m_data[pg_index].clear_dirty();
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_dirty(mem_addr_t pg_index) {
  return m_data[pg_index].is_dirty();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_page_access(mem_addr_t pg_index) {
  return m_data[pg_index].set_access();
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_access(mem_addr_t pg_index) {
  return m_data[pg_index].is_access();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::clear_page_access(mem_addr_t pg_index) {
  return m_data[pg_index].clear_access();
}

// get size in bytes starting from the addr to the end of the page
// first get starting address of the page containing the address
// then get how many bytes are there starting from the page to the given address
// then subtract it from the total page size
template <unsigned BSIZE>
size_t memory_space_impl<BSIZE>::get_data_size(mem_addr_t addr) {
  return BSIZE - (addr - (mem_addr_t)((addr >> m_log2_block_size)
                                      << m_log2_block_size));
}

template <unsigned BSIZE> size_t memory_space_impl<BSIZE>::get_page_size() {
  return BSIZE;
}

template <unsigned BSIZE>
mem_addr_t memory_space_impl<BSIZE>::get_mem_addr(mem_addr_t pg_index) {
  return pg_index << m_log2_block_size;
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
//template class memory_space_impl<8192>;
template class memory_space_impl<4096>;
template class memory_space_impl<1024 * 1024 * 2>;
template class memory_space_impl<16 * 1024>;

void g_print_memory_space(memory_space *mem, const char *format = "%08x",
                          FILE *fout = stdout) {
  mem->print(format, fout);
}

#ifdef UNIT_TEST

int main(int argc, char *argv[]) {
  int errors_found = 0;
  memory_space *mem = new memory_space_impl<32>("test", 4);
  // write address to [address]
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4)
    mem->write(addr, 4, &addr, NULL, NULL);

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4) {
    unsigned tmp = 0;
    mem->read(addr, 4, &tmp);
    if (tmp != addr) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr);
    }
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned char val = (addr + 128) % 256;
    mem->write(addr, 1, &val, NULL, NULL);
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned tmp = 0;
    mem->read(addr, 1, &tmp);
    unsigned char val = (addr + 128) % 256;
    if (tmp != val) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp,
             (unsigned)val);
    }
  }

  if (errors_found) {
    printf("SUMMARY:  ERRORS FOUND\n");
  } else {
    printf("SUMMARY: UNIT TEST PASSED\n");
  }
}

#endif
