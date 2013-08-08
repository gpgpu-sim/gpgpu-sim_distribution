// $Id: traffic.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _TRAFFIC_HPP_
#define _TRAFFIC_HPP_

#include <vector>
#include <set>
#include "config_utils.hpp"

using namespace std;

class TrafficPattern {
protected:
  int _nodes;
  TrafficPattern(int nodes);
public:
  virtual ~TrafficPattern() {}
  virtual void reset();
  virtual int dest(int source) = 0;
  static TrafficPattern * New(string const & pattern, int nodes, 
			      Configuration const * const config = NULL);
};

class PermutationTrafficPattern : public TrafficPattern {
protected:
  PermutationTrafficPattern(int nodes);
};

class BitPermutationTrafficPattern : public PermutationTrafficPattern {
protected:
  BitPermutationTrafficPattern(int nodes);
};

class BitCompTrafficPattern : public BitPermutationTrafficPattern {
public:
  BitCompTrafficPattern(int nodes);
  virtual int dest(int source);
};

class TransposeTrafficPattern : public BitPermutationTrafficPattern {
protected:
  int _shift;
public:
  TransposeTrafficPattern(int nodes);
  virtual int dest(int source);
};

class BitRevTrafficPattern : public BitPermutationTrafficPattern {
public:
  BitRevTrafficPattern(int nodes);
  virtual int dest(int source);
};

class ShuffleTrafficPattern : public BitPermutationTrafficPattern {
public:
  ShuffleTrafficPattern(int nodes);
  virtual int dest(int source);
};

class DigitPermutationTrafficPattern : public PermutationTrafficPattern {
protected:
  int _k;
  int _n;
  int _xr;
  DigitPermutationTrafficPattern(int nodes, int k, int n, int xr = 1);
};

class TornadoTrafficPattern : public DigitPermutationTrafficPattern {
public:
  TornadoTrafficPattern(int nodes, int k, int n, int xr = 1);
  virtual int dest(int source);
};

class NeighborTrafficPattern : public DigitPermutationTrafficPattern {
public:
  NeighborTrafficPattern(int nodes, int k, int n, int xr = 1);
  virtual int dest(int source);
};

class RandomPermutationTrafficPattern : public TrafficPattern {
private:
  vector<int> _dest;
  inline void randomize(int seed);
public:
  RandomPermutationTrafficPattern(int nodes, int seed);
  virtual int dest(int source);
};

class RandomTrafficPattern : public TrafficPattern {
protected:
  RandomTrafficPattern(int nodes);
};

class UniformRandomTrafficPattern : public RandomTrafficPattern {
public:
  UniformRandomTrafficPattern(int nodes);
  virtual int dest(int source);
};

class UniformBackgroundTrafficPattern : public RandomTrafficPattern {
private:
  set<int> _excluded;
public:
  UniformBackgroundTrafficPattern(int nodes, vector<int> excluded_nodes);
  virtual int dest(int source);
};

class DiagonalTrafficPattern : public RandomTrafficPattern {
public:
  DiagonalTrafficPattern(int nodes);
  virtual int dest(int source);
};

class AsymmetricTrafficPattern : public RandomTrafficPattern {
public:
  AsymmetricTrafficPattern(int nodes);
  virtual int dest(int source);
};

class Taper64TrafficPattern : public RandomTrafficPattern {
public:
  Taper64TrafficPattern(int nodes);
  virtual int dest(int source);
};

class BadPermDFlyTrafficPattern : public DigitPermutationTrafficPattern {
public:
  BadPermDFlyTrafficPattern(int nodes, int k, int n);
  virtual int dest(int source);
};

class BadPermYarcTrafficPattern : public DigitPermutationTrafficPattern {
public:
  BadPermYarcTrafficPattern(int nodes, int k, int n, int xr = 1);
  virtual int dest(int source);
};

class HotSpotTrafficPattern : public TrafficPattern {
private:
  vector<int> _hotspots;
  vector<int> _rates;
  int _max_val;
public:
  HotSpotTrafficPattern(int nodes, vector<int> hotspots, 
			vector<int> rates = vector<int>());
  virtual int dest(int source);
};

#endif
