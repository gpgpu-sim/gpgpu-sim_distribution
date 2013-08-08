// $Id: buffer_monitor.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _BUFFER_MONITOR_HPP_
#define _BUFFER_MONITOR_HPP_

#include <vector>
#include <iostream>

using namespace std;

class Flit;

class BufferMonitor {
  int  _cycles ;
  int  _inputs ;
  int  _classes ;
  vector<int> _reads ;
  vector<int> _writes ;
  int index( int input, int cl ) const ;
public:
  BufferMonitor( int inputs, int classes ) ;
  void cycle() ;
  void write( int input, Flit const * f ) ;
  void read( int input, Flit const * f ) ;
  inline const vector<int> & GetReads() const {
    return _reads;
  }
  inline const vector<int> & GetWrites() const {
    return _writes;
  }
  inline int NumInputs() const {
    return _inputs;
  }
  inline int NumClasses() const {
    return _classes;
  }
  void display(ostream & os) const;

} ;

ostream & operator<<( ostream & os, BufferMonitor const & obj ) ;

#endif
