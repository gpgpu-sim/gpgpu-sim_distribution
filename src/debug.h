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

class ptx_thread_info;
class ptx_instruction;
bool thread_at_brkpt( void *ptx_thd_info, const struct brk_pt &b );
unsigned read_location( addr_t addr );
void hit_watchpoint( unsigned watchpoint_num, ptx_thread_info *thd, const ptx_instruction *pI );
void gpgpu_debug();

#endif
