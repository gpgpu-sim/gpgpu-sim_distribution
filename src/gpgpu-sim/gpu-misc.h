/* 
 * gpu-misc.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the 
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

#ifndef GPU_MISC_H
#define GPU_MISC_H

#define CONSTC  100
#define DCACHE   200
#define TEXTC      300
#define SHD_CACHE_TAG(x,shdr) ((x) & (~((unsigned long long int)shdr->L1cache->line_sz - 1)))
#define SHD_TEXCACHE_TAG(x,shdr) ((x) & (~((unsigned long long int)shdr->L1texcache->line_sz - 1)))
#define SHD_CONSTCACHE_TAG(x,shdr) ((x) & (~((unsigned long long int)shdr->L1constcache->line_sz - 1)))
#define CACHE_TAG_OF(x,cache) ((x) & (~((unsigned long long int)cache->line_sz - 1)))
#define CACHE_TAG_OF_64(x) ((x) & (~((unsigned long long int)64 - 1)))

#define ispowerof2(x)   ((((x) - 1) & (x)) == 0)
#define powerof2(x)  (1 << (x))


enum mem_space {  //used for cudasim
   SHARED_SPACE, 
   CONST_SPACE, 
   GLOBAL_SPACE,
   LOCAL_SPACE,
   TEX_SPACE
};
//enables a verbose printout of all L1 cache misses and all MSHR status changes 
//good for a single shader configuration
#define DEBUGL1MISS 0

unsigned int LOGB2( unsigned int v );

unsigned int MAX2NUM( unsigned int a, unsigned int b );

unsigned int MIN2NUM( unsigned int a, unsigned int b );


#define gs_max2(a,b) (((a)>(b))?(a):(b))
#define gs_min2(a,b) (((a)<(b))?(a):(b))
#define min3(x,y,z) (((x)<(y) && (x)<(z))?(x):(gs_min2((y),(z))))

#endif

