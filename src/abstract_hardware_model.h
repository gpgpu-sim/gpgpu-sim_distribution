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

#include <bitset>
#include <list>
#include <vector>
#include <assert.h>
#include <stdlib.h>

#if !defined(__VECTOR_TYPES_H__)
struct dim3 {
   unsigned int x, y, z;
};
#endif

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound);

class kernel_info_t {
public:
   kernel_info_t()
   {
      m_valid=false;
      m_kernel_entry=NULL;
   }
   kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry )
   {
      m_valid=true;
      m_kernel_entry=entry;
      m_grid_dim=gridDim;
      m_block_dim=blockDim;
      m_next_cta.x=0;
      m_next_cta.y=0;
      m_next_cta.z=0;
      m_next_tid=m_next_cta;
   }

   class function_info *entry() { return m_kernel_entry; }

   size_t num_blocks() const
   {
      return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
   }

   size_t threads_per_cta() const
   {
      return m_block_dim.x * m_block_dim.y * m_block_dim.z;
   } 

   dim3 get_grid_dim() const { return m_grid_dim; }
   dim3 get_cta_dim() const { return m_block_dim; }

   void increment_cta_id() 
   { 
      increment_x_then_y_then_z(m_next_cta,m_grid_dim); 
      m_next_tid.x=0;
      m_next_tid.y=0;
      m_next_tid.z=0;
   }
   dim3 get_next_cta_id() const { return m_next_cta; }
   bool no_more_ctas_to_run() const 
   {
      return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y || m_next_cta.z >= m_grid_dim.z );
   }

   void increment_thread_id() { increment_x_then_y_then_z(m_next_tid,m_block_dim); }
   dim3 get_next_thread_id_3d() const  { return m_next_tid; }
   unsigned get_next_thread_id() const 
   { 
      return m_next_tid.x + m_block_dim.x*m_next_tid.y + m_block_dim.x*m_block_dim.y*m_next_tid.z;
   }
   bool more_threads_in_cta() const 
   {
      return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y && m_next_tid.z < m_block_dim.x;
   }

private:
   bool m_valid;
   class function_info *m_kernel_entry;

   dim3 m_grid_dim;
   dim3 m_block_dim;
   dim3 m_next_cta;
   dim3 m_next_tid;
};

class core_t {
public:
   virtual ~core_t() {}
   virtual void set_at_barrier( unsigned cta_id, unsigned warp_id ) = 0;
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) const = 0;
   virtual bool warp_waiting_for_atomics( unsigned warp_id ) const = 0;
   virtual class gpgpu_sim *get_gpu() = 0;
};

#define GLOBAL_HEAP_START 0x80000000
   // start allocating from this address (lower values used for allocating globals in .ptx file)
#define SHARED_MEM_SIZE_MAX (64*1024)
#define LOCAL_MEM_SIZE_MAX (16*1024)
#define MAX_STREAMING_MULTIPROCESSORS 64
#define MAX_THREAD_PER_SM 1024
#define TOTAL_LOCAL_MEM_PER_SM (MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define TOTAL_SHARED_MEM (MAX_STREAMING_MULTIPROCESSORS*SHARED_MEM_SIZE_MAX)
#define TOTAL_LOCAL_MEM (MAX_STREAMING_MULTIPROCESSORS*MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX)
#define SHARED_GENERIC_START (GLOBAL_HEAP_START-TOTAL_SHARED_MEM)
#define LOCAL_GENERIC_START (SHARED_GENERIC_START-TOTAL_LOCAL_MEM)
#define STATIC_ALLOC_LIMIT (GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM+TOTAL_SHARED_MEM))

class gpgpu_t {
public:
   gpgpu_t();
   void* gpgpu_ptx_sim_malloc( size_t size );
   void* gpgpu_ptx_sim_mallocarray( size_t count );
   void  gpgpu_ptx_sim_memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
   void  gpgpu_ptx_sim_memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
   void  gpgpu_ptx_sim_memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
   void  gpgpu_ptx_sim_memset( size_t dst_start_addr, int c, size_t count );

   class memory_space *get_global_memory() { return g_global_mem; }
   class memory_space *get_tex_memory() { return g_tex_mem; }
   class memory_space *get_surf_memory() { return g_surf_mem; }
   class memory_space *get_param_memory() { return g_param_mem; }

protected:
   // functional simulation state 
   class memory_space *g_global_mem;
   class memory_space *g_tex_mem;
   class memory_space *g_surf_mem;
   class memory_space *g_param_mem;

   unsigned long long g_dev_malloc;
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
   dram_callback_t() { function=NULL; instruction=NULL; thread=NULL; }
   void (*function)(const class inst_t*, class ptx_thread_info*);
   const class inst_t* instruction;
   class ptx_thread_info *thread;
};

class inst_t {
public:
    inst_t()
    {
        m_decoded=false;
        pc=(address_type)-1;
        op=NO_OP; 
        memset(out, 0, sizeof(unsigned)); 
        memset(in, 0, sizeof(unsigned)); 
        is_vectorin=0; 
        is_vectorout=0;
        space = memory_space_t();
        cycles = 0;
        for( unsigned i=0; i < MAX_REG_OPERANDS; i++ )
           arch_reg[i]=-1;
        isize=0;
    }
    bool valid() const { return m_decoded; }
    virtual void print_insn( FILE *fp ) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
    }

    address_type pc;        // program counter address of instruction
    unsigned isize;         // size of instruction in bytes 
    op_type op;             // opcode (uarch visible)
    _memory_op_t memory_op; // memory_op used by ptxplus 
    
    unsigned out[4];
    unsigned in[4];
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred; // predicate register number
    int ar1, ar2;
    int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation
    unsigned cycles; // 1/throughput for instruction

    unsigned data_size; // what is the size of the word being operated on?
    memory_space_t space;

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

#define MAX_WARP_SIZE 32

class warp_inst_t: public inst_t {
public:
    // constructors
    warp_inst_t( unsigned warp_size ) 
    { 
        assert(warp_size<=MAX_WARP_SIZE); 
        m_warp_size=warp_size;
        m_empty=true; 
        m_isatomic=false;
        m_per_scalar_thread_valid=false;
    }

    // modifiers
    void do_atomic()
    {
        assert( m_isatomic && !m_empty );
        std::vector<per_thread_info>::iterator t;
        for( t=m_per_scalar_thread.begin(); t != m_per_scalar_thread.end(); ++t ) {
            dram_callback_t &cb = t->callback;
            if( cb.thread ) 
                cb.function(cb.instruction, cb.thread);
        }
    }
    void clear() 
    { 
        m_empty=true; 
    }
    void issue( unsigned mask, unsigned warp_id, unsigned long long cycle ) 
    {
        for (int i=(int)m_warp_size-1; i>=0; i--) {
            if( mask & (1<<i) )
                warp_active_mask.set(i);
        }
        m_warp_id = warp_id;
        issue_cycle = cycle;
        m_empty=false;
    }
    void set_addr( unsigned n, new_addr_type addr ) 
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_warp_size);
            m_per_scalar_thread_valid=true;
        }
        m_per_scalar_thread[n].memreqaddr = addr;
    }
    void add_callback( unsigned lane_id, 
                       void (*function)(const class inst_t*, class ptx_thread_info*),
                       const inst_t *inst, 
                       class ptx_thread_info *thread )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_warp_size);
            m_per_scalar_thread_valid=true;
            m_isatomic=true;
        }
        m_per_scalar_thread[lane_id].callback.function = function;
        m_per_scalar_thread[lane_id].callback.instruction = inst;
        m_per_scalar_thread[lane_id].callback.thread = thread;
    }
    void set_active( std::vector<unsigned> &active ) 
    {
       warp_active_mask.reset();
       for( std::vector<unsigned>::iterator i=active.begin(); i!=active.end(); ++i ) {
          unsigned t = *i;
          assert( t < m_warp_size );
          warp_active_mask.set(t);
       }
       if( m_isatomic ) {
          for( unsigned i=0; i < m_warp_size; i++ ) {
             if( !warp_active_mask.test(i) ) {
                 m_per_scalar_thread[i].callback.function = NULL;
                 m_per_scalar_thread[i].callback.instruction = NULL;
                 m_per_scalar_thread[i].callback.thread = NULL;
             }
          }
       }
    }

    // accessors
    virtual void print_insn(FILE *fp) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
        for (int i=(int)m_warp_size-1; i>=0; i--)
            fprintf(fp, "%c", ((warp_active_mask[i])?'1':'0') );
    }
    bool active( unsigned thread ) const { return warp_active_mask.test(thread); }
    unsigned active_count() const { return warp_active_mask.count(); }
    bool empty() const { return m_empty; }
    unsigned warp_id() const 
    { 
        assert( !m_empty );
        return m_warp_id; 
    }
    bool has_callback( unsigned n ) const
    {
        return warp_active_mask[n] && m_per_scalar_thread_valid && 
            (m_per_scalar_thread[n].callback.function!=NULL);
    }
    new_addr_type get_addr( unsigned n ) const
    {
        assert( m_per_scalar_thread_valid );
        return m_per_scalar_thread[n].memreqaddr;
    }

    bool isatomic() const { return m_isatomic; }

protected:
    bool m_empty;
    unsigned long long issue_cycle;
    bool m_isatomic;
    unsigned m_warp_id;
    unsigned m_warp_size;
    std::bitset<MAX_WARP_SIZE> warp_active_mask;

    struct per_thread_info {
        per_thread_info() {
            cache_miss=false;
            memreqaddr=0;
        }
        dram_callback_t callback;
        new_addr_type memreqaddr; // effective address
        bool cache_miss;
    };
    bool m_per_scalar_thread_valid;
    std::vector<per_thread_info> m_per_scalar_thread;
};

void move_warp( warp_inst_t *&dst, warp_inst_t *&src );

size_t get_kernel_code_size( class function_info *entry );

#endif // #ifdef __cplusplus
#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
