// $Id: switch_monitor.cpp 5188 2012-08-30 00:31:31Z dub $

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

#include "switch_monitor.hpp"

#include "flit.hpp"

SwitchMonitor::SwitchMonitor( int inputs, int outputs, int classes )
: _cycles(0), _inputs(inputs), _outputs(outputs), _classes(classes) {
  _event.resize(inputs * outputs * classes, 0) ;
}

int SwitchMonitor::index( int input, int output, int cl ) const {
  assert((input >= 0) && (input < _inputs));
  assert((output >= 0) && (output < _outputs));
  assert((cl >= 0) && (cl < _classes));
  return cl + _classes * ( output + _outputs * input ) ;
}

void SwitchMonitor::cycle() {
  _cycles++ ;
}

void SwitchMonitor::traversal( int input, int output, Flit const * f ) {
  _event[ index( input, output, f->cl) ]++ ;
}

void SwitchMonitor::display(ostream & os) const {
  for ( int i = 0 ; i < _inputs ; i++ ) {
    for ( int o = 0 ; o < _outputs ; o++) {
      os << "[" << i << " -> " << o << "] " ;
      for ( int c = 0 ; c < _classes ; c++ ) {
	os << c << ":" << _event[index(i,o,c)] << " " ;
      }
      os << endl ;
    }
  }
}

ostream & operator<<( ostream & os, SwitchMonitor const & obj ) {
  obj.display(os);
  return os ;
}
