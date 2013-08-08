// $Id: buffer_monitor.cpp 5188 2012-08-30 00:31:31Z dub $

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

#include "buffer_monitor.hpp"

#include "flit.hpp"

BufferMonitor::BufferMonitor( int inputs, int classes ) 
: _cycles(0), _inputs(inputs), _classes(classes) {
  _reads.resize(inputs * classes, 0) ;
  _writes.resize(inputs * classes, 0) ;
}

int BufferMonitor::index( int input, int cl ) const {
  assert((input >= 0) && (input < _inputs)); 
  assert((cl >= 0) && (cl < _classes));
  return cl + _classes * input ;
}

void BufferMonitor::cycle() {
  _cycles++ ;
}

void BufferMonitor::write( int input, Flit const * f ) {
  _writes[ index(input, f->cl) ]++ ;
}

void BufferMonitor::read( int input, Flit const * f ) {
  _reads[ index(input, f->cl) ]++ ;
}

void BufferMonitor::display(ostream & os) const {
  for ( int i = 0 ; i < _inputs ; i++ ) {
    os << "[ " << i << " ] " ;
    for ( int c = 0 ; c < _classes ; c++ ) {
      os << "Type=" << c
	 << ":(R#" << _reads[ index( i, c) ]  << ","
	 << "W#" << _writes[ index( i, c) ] << ")" << " " ;
    }
    os << endl ;
  }
}

ostream & operator<<( ostream & os, BufferMonitor const & obj ) {
  obj.display(os);
  return os ;
}
