#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

#include <string.h>
#include <stdio.h>

typedef unsigned long long new_addr_type;
typedef unsigned address_type;
typedef unsigned addr_t;

// the following are operations the timing model can see 

enum uarch_op_t {
   NO_OP=-1,
   ALU_OP=1,
   SFU_OP,
   ALU_SFU_OP,
   LOAD_OP,
   STORE_OP,
   BRANCH_OP,
   BARRIER_OP,
   MEMORY_BARRIER_OP
};
typedef enum uarch_op_t op_type;

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
   generic_space,
   instruction_space
};

enum _memory_op_t {
	no_memory_op = 0,
	memory_load,
	memory_store
};

#ifdef __cplusplus

#include <list>

#if !defined(__VECTOR_TYPES_H__)
struct dim3 {
   unsigned int x, y, z;
};
#endif

class core_t {
public:
   virtual ~core_t() {}
   virtual void set_at_barrier( unsigned cta_id, unsigned warp_id ) = 0;
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) const = 0;
   virtual bool warp_waiting_for_atomics( unsigned warp_id ) const = 0;
   virtual class gpgpu_sim *get_gpu() = 0;
};

struct gpgpu_ptx_sim_kernel_info 
{
   // Holds properties of the kernel (Kernel's resource use). 
   // These will be set to zero if a ptxinfo file is not present.
   int lmem;
   int smem;
   int cmem;
   int regs;
   unsigned ptx_version;
   unsigned sm_target;
};

struct gpgpu_ptx_sim_arg {
   gpgpu_ptx_sim_arg() { m_start=NULL; }
   gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset)
   {
      m_start=arg;
      m_nbytes=size;
      m_offset=offset;
   }
   const void *m_start;
   size_t m_nbytes;
   size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

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


#define MAX_REG_OPERANDS 8

struct dram_callback_t {
   void (*function)(void* pI, void* gOldGThread);
   void* instruction;
   void* thread;
};

class inst_t {
public:
    inst_t()
    {
        m_decoded=false;
        pc = (address_type)-1;
        op=NO_OP; 
        memset(out, 0, sizeof(unsigned)); 
        memset(in, 0, sizeof(unsigned)); 
        is_vectorin=0; 
        is_vectorout=0;
        memreqaddr=0; 
        hw_thread_id=-1; 
        wlane=-1;
        uid = (unsigned)-1;
        warp_active_mask = 0;
        issue_cycle = 0;
        cache_miss = false;
        space = memory_space_t();
        cycles = 0;
        for( unsigned i=0; i < MAX_REG_OPERANDS; i++ )
           arch_reg[i]=-1;
        callback.function = NULL;
        callback.instruction = NULL;
        callback.thread = NULL;
        isize=0;
    }
    bool valid() const { return m_decoded; }
    virtual void print_insn( FILE *fp ) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
    }

    unsigned uid;           // unique id (for debugging)
    address_type pc;        // program counter address of instruction
    unsigned isize;         // size of instruction in bytes 
    op_type op;             // opcode (uarch visible)
    _memory_op_t memory_op; // ptxplus 
    short hw_thread_id;     // scalar hardware thread id
    short wlane;            // SIMT lane
    
    unsigned warp_active_mask;
    unsigned long long  issue_cycle;

    unsigned out[4];
    unsigned in[4];
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred;
    int ar1, ar2;
    int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation
    unsigned cycles; // 1/throughput for instruction

    unsigned long long int memreqaddr; // effective address
    unsigned data_size; // what is the size of the word being operated on?
    memory_space_t space;
    dram_callback_t callback;
    bool cache_miss;

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};


const struct gpgpu_ptx_sim_kernel_info * get_kernel_info(const char *kernel_key);
size_t get_kernel_code_size( class function_info *entry );

#endif // #ifdef __cplusplus
#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
