// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
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

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#ifdef __cplusplus

#include <stdio.h>
#include <string>

class binned_histogram {
 public:
  // creators
  binned_histogram(std::string name = "", int nbins = 32, int* bins = NULL);
  binned_histogram(const binned_histogram& other);
  virtual ~binned_histogram();

  // modifiers:
  void reset_bins();
  void add2bin(int sample);

  // accessors:
  void fprint(FILE* fout) const;

 protected:
  std::string m_name;
  int m_nbins;
  int* m_bins;                 // bin boundaries
  int* m_bin_cnts;             // counters
  int m_maximum;               // the maximum sample
  signed long long int m_sum;  // for calculating the average
};

class pow2_histogram : public binned_histogram {
 public:
  pow2_histogram(std::string name = "", int nbins = 32, int* bins = NULL);
  ~pow2_histogram() {}

  void add2bin(int sample);
};

class linear_histogram : public binned_histogram {
 public:
  linear_histogram(int stride = 1, const char* name = NULL, int nbins = 32,
                   int* bins = NULL);
  ~linear_histogram() {}

  void add2bin(int sample);

 private:
  int m_stride;
};

#endif

#endif /* HISTOGRAM_H */
