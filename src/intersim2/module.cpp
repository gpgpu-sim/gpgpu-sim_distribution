// $Id: module.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*module.cpp
 *
 *The basic class that is extended by all other components of the network
 *Provides the basic hierarchy structure and basic fuctions
 *
 */

#include <iostream>
#include <cstdlib>

#include "booksim.hpp"
#include "module.hpp"

Module::Module( Module *parent, const string& name )
{
  _name = name;

  if ( parent ) { 
    parent->_AddChild( this );
    _fullname = parent->_fullname + "/" + name;
  } else {
    _fullname = name;
  }
}

void Module::_AddChild( Module *child )
{
  _children.push_back( child );
}

void Module::DisplayHierarchy( int level, ostream & os ) const
{
  vector<Module *>::const_iterator mod_iter;

  for ( int l = 0; l < level; l++ ) {
    os << "  ";  
  }

  os << _name << endl;

  for ( mod_iter = _children.begin( );
	mod_iter != _children.end( ); mod_iter++ ) {
    (*mod_iter)->DisplayHierarchy( level + 1 );
  }
}

void Module::Error( const string& msg ) const
{
  cout << "Error in " << _fullname << " : " << msg << endl;
  exit( -1 );
}

void Module::Debug( const string& msg ) const
{
  cout << "Debug (" << _fullname << ") : " << msg << endl;
}

void Module::Display( ostream & os ) const 
{
  os << "Display method not implemented for " << _fullname << endl;
}
