// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung
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

#include "histogram.h"

#include <assert.h>

binned_histogram::binned_histogram(std::string name, int nbins, int* bins)
    : m_name(name),
      m_nbins(nbins),
      m_bins(NULL),
      m_bin_cnts(new int[m_nbins]),
      m_maximum(0),
      m_sum(0) {
  if (bins) {
    m_bins = new int[m_nbins];
    for (int i = 0; i < nbins; i++) {
      m_bins[i] = bins[i];
    }
  }

  reset_bins();
}

binned_histogram::binned_histogram(const binned_histogram& other)
    : m_name(other.m_name),
      m_nbins(other.m_nbins),
      m_bins(NULL),
      m_bin_cnts(new int[m_nbins]),
      m_maximum(0),
      m_sum(0) {
  for (int i = 0; i < m_nbins; i++) {
    m_bin_cnts[i] = other.m_bin_cnts[i];
  }
}

void binned_histogram::reset_bins() {
  for (int i = 0; i < m_nbins; i++) {
    m_bin_cnts[i] = 0;
  }
}

void binned_histogram::add2bin(int sample) {
  assert(0);
  m_maximum = (sample > m_maximum) ? sample : m_maximum;
}

void binned_histogram::fprint(FILE* fout) const {
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

binned_histogram::~binned_histogram() {
  if (m_bins) delete[] m_bins;
  delete[] m_bin_cnts;
}

pow2_histogram::pow2_histogram(std::string name, int nbins, int* bins)
    : binned_histogram(name, nbins, bins) {}

void pow2_histogram::add2bin(int sample) {
  assert(sample >= 0);

  int bin;
  int v = sample;
  register unsigned int shift;

  bin = (v > 0xFFFF) << 4;
  v >>= bin;
  shift = (v > 0xFF) << 3;
  v >>= shift;
  bin |= shift;
  shift = (v > 0xF) << 2;
  v >>= shift;
  bin |= shift;
  shift = (v > 0x3) << 1;
  v >>= shift;
  bin |= shift;
  bin |= (v >> 1);
  bin += (sample > 0) ? 1 : 0;

  m_bin_cnts[bin] += 1;

  m_maximum = (sample > m_maximum) ? sample : m_maximum;
  m_sum += sample;
}

linear_histogram::linear_histogram(int stride, const char* name, int nbins,
                                   int* bins)
    : binned_histogram(name, nbins, bins), m_stride(stride) {}

void linear_histogram::add2bin(int sample) {
  assert(sample >= 0);

  int bin = sample / m_stride;
  if (bin >= m_nbins) bin = m_nbins - 1;

  m_bin_cnts[bin] += 1;

  m_maximum = (sample > m_maximum) ? sample : m_maximum;
  m_sum += sample;
}
