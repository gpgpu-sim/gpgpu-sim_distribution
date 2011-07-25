// Copyright (c) 2009-2011, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef PTX_DEBUG_INCLUDED
#define PTX_DEBUG_INCLUDED

#include "abstract_hardware_model.h"

#include <string>

class brk_pt {
public:
   brk_pt() { m_valid=false; }
   brk_pt( const char *fileline, unsigned uid )
   {
      m_valid = true;
      m_watch = false;
      m_fileline = std::string(fileline);
      m_thread_uid=uid;
   }
   brk_pt( unsigned addr, unsigned value )
   {
      m_valid = true;
      m_watch = true;
      m_addr = addr;
      m_value = value;
   }

   unsigned get_value() const { return m_value; }
   addr_t get_addr() const { return m_addr; }
   bool is_valid() const { return m_valid; }
   bool is_watchpoint() const { return m_watch; }
   bool is_equal( const std::string &fileline, unsigned uid ) const
   {
      if( m_watch ) 
         return false; 
      if( (m_thread_uid != (unsigned)-1) && (uid != m_thread_uid) ) 
         return false;
      return m_fileline == fileline;
   }
   std::string location() const
   {
      char buffer[1024];
      sprintf(buffer,"%s thread uid = %u", m_fileline.c_str(), m_thread_uid);
      return buffer;
   }

   unsigned set_value( unsigned val ) { return m_value=val; }
private:
   bool         m_valid;
   bool         m_watch;

   // break point
   std::string  m_fileline;
   unsigned     m_thread_uid;

   // watch point
   unsigned     m_addr;
   unsigned     m_value;
};

extern int gpgpu_ptx_instruction_classification ;

class ptx_thread_info;
class ptx_instruction;
bool thread_at_brkpt( ptx_thread_info *thd_info, const struct brk_pt &b );
void hit_watchpoint( unsigned watchpoint_num, ptx_thread_info *thd, const ptx_instruction *pI );

#endif
