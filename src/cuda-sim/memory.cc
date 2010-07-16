/*
 * memory.cc
 *  
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung and the University of 
 * British Columbia, Vancouver, BC V6T 1Z4, All Rights Reserved.
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

#include "memory.h"
#include <stdlib.h>

template<unsigned BSIZE> memory_space_impl<BSIZE>::memory_space_impl( std::string name, unsigned hash_size )
{
   m_name = name;
   MEM_MAP_RESIZE(hash_size);

   m_log2_block_size = -1;
   for( unsigned n=0, mask=1; mask != 0; mask <<= 1, n++ ) {
      if( BSIZE & mask ) {
         assert( m_log2_block_size == (unsigned)-1 );
         m_log2_block_size = n; 
      }
   }
   assert( m_log2_block_size != (unsigned)-1 );
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::write( mem_addr_t addr, size_t length, const void *data )
{
   mem_addr_t index = addr >> m_log2_block_size;
   unsigned offset = addr & (BSIZE-1);
   unsigned nbytes = length;
   assert( (addr+length) <= (index+1)*BSIZE );
   m_data[index].write(offset,nbytes,(const unsigned char*)data);
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::read( mem_addr_t addr, size_t length, void *data ) const
{
   mem_addr_t index = addr >> m_log2_block_size;
   unsigned offset = addr & (BSIZE-1);
   unsigned nbytes = length;
   typename map_t::const_iterator i = m_data.find(index);
   assert( (addr+length) <= (index+1)*BSIZE );
   if( i == m_data.end() ) {
      for( size_t n=0; n < length; n++ ) 
         ((unsigned char*)data)[n] = (unsigned char) 0;
      //printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
   } else {
      i->second.read(offset,nbytes,(unsigned char*)data);
   }
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
template class memory_space_impl<8192>;
template class memory_space_impl<16*1024>;


#ifdef UNIT_TEST

int main(int argc, char *argv[] )
{
   int errors_found=0;
   memory_space *mem = new memory_space_impl<32>("test",4);
   // write address to [address]
   for( mem_addr_t addr=0; addr < 16*1024; addr+=4) 
      mem->write(addr,4,&addr);

   for( mem_addr_t addr=0; addr < 16*1024; addr+=4) {
      unsigned tmp=0;
      mem->read(addr,4,&tmp);
      if( tmp != addr ) {
         errors_found=1;
         printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr );
      }
   }

   for( mem_addr_t addr=0; addr < 16*1024; addr+=1) {
      unsigned char val = (addr + 128) % 256;
      mem->write(addr,1,&val);
   }

   for( mem_addr_t addr=0; addr < 16*1024; addr+=1) {
      unsigned tmp=0;
      mem->read(addr,1,&tmp);
      unsigned char val = (addr + 128) % 256;
      if( tmp != val ) {
         errors_found=1;
         printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, (unsigned)val );
      }
   }

   if( errors_found ) {
      printf("SUMMARY:  ERRORS FOUND\n");
   } else {
      printf("SUMMARY: UNIT TEST PASSED\n");
   }
}

#endif
