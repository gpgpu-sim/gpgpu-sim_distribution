#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

#ifdef __cplusplus

class core_t {
public:
   virtual ~core_t() {}
	virtual void set_at_barrier( unsigned cta_id, unsigned warp_id ) = 0;
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) = 0;
};

#define MAX_REG_OPERANDS 8
extern unsigned int warp_size; 
#endif

typedef unsigned address_type;
typedef unsigned addr_t;

// these are operations the timing model can see
#define NO_OP -1
#define ALU_OP 1000
#define LOAD_OP 2000
#define STORE_OP 3000
#define BRANCH_OP 4000
#define BARRIER_OP 5000

enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space
};

#ifdef __cplusplus

class memory_space_t {
public:
   memory_space_t() { m_type = undefined_space; m_bank=0; }
   memory_space_t( const enum _memory_space_t &from ) { m_type = from; m_bank = 0; }
   bool operator==( const memory_space_t &x ) const { return (m_bank == x.m_bank) && (m_type == x.m_type); }
   bool operator!=( const memory_space_t &x ) const { return !(*this == x); }
   bool operator<( const memory_space_t &x ) const 
   { 
      if(m_type < x.m_type)
         return true;
      else if(m_type > x.m_type)
         return false;
      else if( m_bank < x.m_bank )
         return true; 
      return false;
   }
   enum _memory_space_t get_type() const { return m_type; }
   unsigned get_bank() const { return m_bank; }
   void set_bank( unsigned b ) { m_bank = b; }
private:
   enum _memory_space_t m_type;
   unsigned m_bank; // n in ".const[n]"; note .const == .const[0] (see PTX 2.1 manual, sec. 5.1.3)
};

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

bool thread_at_brkpt( void *ptx_thd_info, const struct brk_pt &b );
unsigned read_location( addr_t addr );
 
#endif

#endif
