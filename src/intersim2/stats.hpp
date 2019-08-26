// $Id: stats.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _STATS_HPP_
#define _STATS_HPP_

#include "module.hpp"

class Stats : public Module {
  int    _num_samples;
  double _sample_sum;
  double _sample_squared_sum;

  //bool _reset;
  double _min;
  double _max;

  int    _num_bins;
  double _bin_size;

  vector<int> _hist;

public:
  Stats( Module *parent, const string &name,
	 double bin_size = 1.0, int num_bins = 10 );

  void Clear( );

  double Average( ) const;
  double Variance( ) const;
  double Max( ) const;
  double Min( ) const;
  double Sum( ) const;
  double SquaredSum( ) const;
  int    NumSamples( ) const;

  void AddSample( double val );
  inline void AddSample( int val ) {
    AddSample( (double)val );
  }
  inline void AddSample( unsigned long long val ) {
    AddSample( (double)val );
  }

  int GetBin(int b){ return _hist[b];}

  void Display( ostream & os = cout ) const;

  friend ostream & operator<<(ostream & os, const Stats & s);

};

ostream & operator<<(ostream & os, const Stats & s);

#endif
