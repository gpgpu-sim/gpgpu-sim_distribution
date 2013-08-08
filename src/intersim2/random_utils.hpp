// $Id: random_utils.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _RANDOM_UTILS_HPP_
#define _RANDOM_UTILS_HPP_

 // interface to Knuth's RANARRAY RNG
void   ran_start(long seed);
long   ran_next( );
void   ranf_start(long seed);
double ranf_next( );

inline void RandomSeed( long seed ) {
  ran_start( seed );
  ranf_start( seed );
}

inline unsigned long RandomIntLong( ) {
  return ran_next( );
}

// Returns a random integer in the range [0,max]
inline int RandomInt( int max ) {
  return ( ran_next( ) % (max+1) );
}

// Returns a random floating-point value in the rage [0,1]
inline double RandomFloat(  ) {
  return ranf_next( );
}

// Returns a random floating-point value in the rage [0,max]
inline double RandomFloat( double max ) {
  return ( ranf_next( ) * max );
}

#endif
