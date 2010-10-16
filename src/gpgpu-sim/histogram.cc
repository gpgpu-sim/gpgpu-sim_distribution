/* 
 * histogram.cc
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung and the 
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

#include "histogram.h"

#include <assert.h>

binned_histogram::binned_histogram (std::string name, int nbins, int* bins) 
   : m_name(name), m_nbins(nbins), m_bins(NULL), m_bin_cnts(new int[m_nbins]), m_maximum(0), m_sum(0) 
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
     m_bin_cnts(new int[m_nbins]), m_maximum(0), m_sum(0)
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

void binned_histogram::fprint (FILE *fout) const
{
   if (m_name.c_str() != NULL) fprintf(fout, "%s = ", m_name.c_str());
   int total_sample = 0;
   for (int i = 0; i < m_nbins; i++) {
      fprintf(fout, "%d ", m_bin_cnts[i]);
      total_sample += m_bin_cnts[i];
   }
   fprintf(fout, "max=%d ", m_maximum);
   float avg = 0.0f;
   if (total_sample > 0) {
      avg = (float)m_sum / total_sample;
   }
   fprintf(fout, "avg=%0.2f ", avg);
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
   m_sum += sample;
}

linear_histogram::linear_histogram (int stride, const char *name, int nbins, int* bins) 
   : binned_histogram (name, nbins, bins), m_stride(stride)
{
}

void linear_histogram::add2bin (int sample) {
   assert(sample >= 0);

   int bin = sample / m_stride;      
   if (bin >= m_nbins) bin = m_nbins - 1;
   
   m_bin_cnts[bin] += 1;
   
   m_maximum = (sample > m_maximum)? sample : m_maximum;
   m_sum += sample;
}
