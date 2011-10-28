// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh,
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

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

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

#ifdef __cplusplus

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

enum _memory_op_t {
	no_memory_op = 0,
	memory_load,
	memory_store
};

#include <bitset>
#include <list>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <map>

#if !defined(__VECTOR_TYPES_H__)
struct dim3 {
   unsigned int x, y, z;
};
#endif

#if 0

// detect gcc 4.3 and use unordered map (part of c++0x)
// unordered map doesn't play nice with _GLIBCXX_DEBUG, just use a map if its enabled.
#if  defined( __GNUC__ ) and not defined( _GLIBCXX_DEBUG )
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
   #include <unordered_map>
   #define my_hash_map std::unordered_map
#else
   #include <ext/hash_map>
   namespace std {
      using namespace __gnu_cxx;
   }
   #define my_hash_map std::hash_map
#endif
#else
   #include <map>
   #define my_hash_map std::map
   #define USE_MAP
#endif

#endif

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound);

class kernel_info_t {
public:
//   kernel_info_t()
//   {
//      m_valid=false;
//      m_kernel_entry=NULL;
//      m_uid=0;
//      m_num_cores_running=0;
//      m_param_mem=NULL;
//   }
   kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry );
   ~kernel_info_t();

   void inc_running() { m_num_cores_running++; }
   void dec_running()
   {
       assert( m_num_cores_running > 0 );
       m_num_cores_running--; 
   }
   bool running() const { return m_num_cores_running>0; }
   bool done() const 
   {
       return no_more_ctas_to_run() && !running();
   }
   class function_info *entry() { return m_kernel_entry; }
   const class function_info *entry() const { return m_kernel_entry; }

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
   unsigned get_uid() const { return m_uid; }
   std::string name() const;

   std::list<class ptx_thread_info *> &active_threads() { return m_active_threads; }
   class memory_space *get_param_memory() { return m_param_mem; }

private:
   kernel_info_t( const kernel_info_t & ); // disable copy constructor
   void operator=( const kernel_info_t & ); // disable copy operator

   class function_info *m_kernel_entry;

   unsigned m_uid;
   static unsigned m_next_uid;

   dim3 m_grid_dim;
   dim3 m_block_dim;
   dim3 m_next_cta;
   dim3 m_next_tid;

   unsigned m_num_cores_running;

   std::list<class ptx_thread_info *> m_active_threads;
   class memory_space *m_param_mem;
};

struct core_config {
    core_config() 
    { 
        m_valid = false; 
        num_shmem_bank=16; 
    }
    virtual void init() = 0;

    bool m_valid;
    unsigned warp_size;

    // off-chip memory request architecture parameters
    int gpgpu_coalesce_arch;

    // shared memory bank conflict checking parameters
    static const address_type WORD_SIZE=4;
    unsigned num_shmem_bank;
    unsigned shmem_bank_func(address_type addr) const
    {
        return ((addr/WORD_SIZE) % num_shmem_bank);
    }
    unsigned mem_warp_parts;  
    unsigned gpgpu_shmem_size;

    // texture and constant cache line sizes (used to determine number of memory accesses)
    unsigned gpgpu_cache_texl1_linesize;
    unsigned gpgpu_cache_constl1_linesize;

	unsigned gpgpu_max_insn_issue_per_warp;
};

class core_t {
public:
   virtual ~core_t() {}
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) const = 0;
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

#if !defined(__CUDA_RUNTIME_API_H__)

enum cudaChannelFormatKind {
   cudaChannelFormatKindSigned,
   cudaChannelFormatKindUnsigned,
   cudaChannelFormatKindFloat
};

struct cudaChannelFormatDesc {
   int                        x;
   int                        y;
   int                        z;
   int                        w;
   enum cudaChannelFormatKind f;
};

struct cudaArray {
   void *devPtr;
   int devPtr32;
   struct cudaChannelFormatDesc desc;
   int width;
   int height;
   int size; //in bytes
   unsigned dimensions;
};

enum cudaTextureAddressMode {
   cudaAddressModeWrap,
   cudaAddressModeClamp
};

enum cudaTextureFilterMode {
   cudaFilterModePoint,
   cudaFilterModeLinear
};

enum cudaTextureReadMode {
   cudaReadModeElementType,
   cudaReadModeNormalizedFloat
};

struct textureReference {
   int                           normalized;
   enum cudaTextureFilterMode    filterMode;
   enum cudaTextureAddressMode   addressMode[2];
   struct cudaChannelFormatDesc  channelDesc;
};

#endif

class gpgpu_functional_sim_config 
{
public:
    void reg_options(class OptionParser * opp);

    void ptx_set_tex_cache_linesize(unsigned linesize);

    unsigned get_forced_max_capability() const { return m_ptx_force_max_capability; }
    bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
    bool saved_converted_ptxplus() const { return m_ptx_save_converted_ptxplus; }

    int         get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
    const char* get_ptx_inst_debug_file() const  { return g_ptx_inst_debug_file; }
    int         get_ptx_inst_debug_thread_uid() const { return g_ptx_inst_debug_thread_uid; }
    unsigned    get_texcache_linesize() const { return m_texcache_linesize; }

private:
    // PTX options
    int m_ptx_convert_to_ptxplus;
    int m_ptx_save_converted_ptxplus;
    unsigned m_ptx_force_max_capability;

    int   g_ptx_inst_debug_to_file;
    char* g_ptx_inst_debug_file;
    int   g_ptx_inst_debug_thread_uid;

    unsigned m_texcache_linesize;
};

class gpgpu_t {
public:
    gpgpu_t( const gpgpu_functional_sim_config &config );
    void* gpu_malloc( size_t size );
    void* gpu_mallocarray( size_t count );
    void  gpu_memset( size_t dst_start_addr, int c, size_t count );
    void  memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
    void  memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
    void  memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
    
    class memory_space *get_global_memory() { return m_global_mem; }
    class memory_space *get_tex_memory() { return m_tex_mem; }
    class memory_space *get_surf_memory() { return m_surf_mem; }

    void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
    void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
    const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);

    const struct textureReference* get_texref(const std::string &texname) const
    {
        std::map<std::string, const struct textureReference*>::const_iterator t=m_NameToTextureRef.find(texname);
        assert( t != m_NameToTextureRef.end() );
        return t->second;
    }
    const struct cudaArray* get_texarray( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*,const struct cudaArray*>::const_iterator t=m_TextureRefToCudaArray.find(texref);
        assert(t != m_TextureRefToCudaArray.end());
        return t->second;
    }
    const struct textureInfo* get_texinfo( const struct textureReference *texref ) const
    {
        std::map<const struct textureReference*, const struct textureInfo*>::const_iterator t=m_TextureRefToTexureInfo.find(texref);
        assert(t != m_TextureRefToTexureInfo.end());
        return t->second;
    }

    const gpgpu_functional_sim_config &get_config() const { return m_function_model_config; }
    FILE* get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

protected:
    const gpgpu_functional_sim_config &m_function_model_config;
    FILE* ptx_inst_debug_file;

    class memory_space *m_global_mem;
    class memory_space *m_tex_mem;
    class memory_space *m_surf_mem;
    
    unsigned long long m_dev_malloc;
    
    std::map<std::string, const struct textureReference*> m_NameToTextureRef;
    std::map<const struct textureReference*,const struct cudaArray*> m_TextureRefToCudaArray;
    std::map<const struct textureReference*, const struct textureInfo*> m_TextureRefToTexureInfo;
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
   bool is_const() const { return (m_type == const_space) || (m_type == param_space_kernel); }
   bool is_local() const { return (m_type == local_space) || (m_type == param_space_local); }

private:
   enum _memory_space_t m_type;
   unsigned m_bank; // n in ".const[n]"; note .const == .const[0] (see PTX 2.1 manual, sec. 5.1.3)
};

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;

enum mem_access_type {
   GLOBAL_ACC_R, 
   LOCAL_ACC_R, 
   CONST_ACC_R, 
   TEXTURE_ACC_R, 
   GLOBAL_ACC_W, 
   LOCAL_ACC_W,
   L1_WRBK_ACC,
   L2_WRBK_ACC, 
   INST_ACC_R,
   NUM_MEM_ACCESS_TYPE
};

enum cache_operator_type {
    CACHE_UNDEFINED, 

    // loads
    CACHE_ALL,          // .ca
    CACHE_LAST_USE,     // .lu
    CACHE_VOLATILE,     // .cv
                       
    // loads and stores 
    CACHE_STREAMING,    // .cs
    CACHE_GLOBAL,       // .cg

    // stores
    CACHE_WRITE_BACK,   // .wb
    CACHE_WRITE_THROUGH // .wt
};

class mem_access_t {
public:
   mem_access_t() { init(); }
   mem_access_t( mem_access_type type, 
                 address_type address, 
                 unsigned size,
                 bool wr )
   {
       init();
       m_type = type;
       m_addr = address;
       m_req_size = size;
       m_write = wr;
   }
   mem_access_t( mem_access_type type, 
                 address_type address, 
                 unsigned size, 
                 bool wr, 
                 const active_mask_t &active_mask,
                 const mem_access_byte_mask_t &byte_mask )
    : m_warp_mask(active_mask), m_byte_mask(byte_mask)
   {
      init();
      m_type = type;
      m_addr = address;
      m_req_size = size;
      m_write = wr;
   }

   new_addr_type get_addr() const { return m_addr; }
   unsigned get_size() const { return m_req_size; }
   const active_mask_t &get_warp_mask() const { return m_warp_mask; }
   bool is_write() const { return m_write; }
   enum mem_access_type get_type() const { return m_type; }
   mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }

   void print(FILE *fp) const
   {
       fprintf(fp,"addr=0x%llx, %s, size=%u, ", m_addr, m_write?"store":"load ", m_req_size );
       switch(m_type) {
       case GLOBAL_ACC_R:  fprintf(fp,"GLOBAL_R"); break;
       case LOCAL_ACC_R:   fprintf(fp,"LOCAL_R "); break;
       case CONST_ACC_R:   fprintf(fp,"CONST   "); break;
       case TEXTURE_ACC_R: fprintf(fp,"TEXTURE "); break;
       case GLOBAL_ACC_W:  fprintf(fp,"GLOBAL_W"); break;
       case LOCAL_ACC_W:   fprintf(fp,"LOCAL_W "); break;
       case L2_WRBK_ACC:   fprintf(fp,"L2_WRBK "); break;
       case INST_ACC_R:    fprintf(fp,"INST    "); break;
       case L1_WRBK_ACC:   fprintf(fp,"L1_WRBK "); break;
       default:            fprintf(fp,"unknown "); break;
       }
   }

private:
   void init() 
   {
      m_uid=++sm_next_access_uid;
      m_addr=0;
      m_req_size=0;
   }

   unsigned      m_uid;
   new_addr_type m_addr;     // request address
   bool          m_write;
   unsigned      m_req_size; // bytes
   mem_access_type m_type;
   active_mask_t m_warp_mask;
   mem_access_byte_mask_t m_byte_mask;

   static unsigned sm_next_access_uid;
};

class mem_fetch;

class mem_fetch_interface {
public:
    virtual bool full( unsigned size, bool write ) const = 0;
    virtual void push( mem_fetch *mf ) = 0;
};

class mem_fetch_allocator {
public:
    virtual mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr ) const = 0;
    virtual mem_fetch *alloc( const class warp_inst_t &inst, const mem_access_t &access ) const = 0;
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
        reconvergence_pc=(address_type)-1;
        op=NO_OP; 
        memset(out, 0, sizeof(unsigned)); 
        memset(in, 0, sizeof(unsigned)); 
        is_vectorin=0; 
        is_vectorout=0;
        space = memory_space_t();
        cache_op = CACHE_UNDEFINED;
        latency = 1;
        initiation_interval = 1;
        for( unsigned i=0; i < MAX_REG_OPERANDS; i++ )
           arch_reg[i]=-1;
        isize=0;
    }
    bool valid() const { return m_decoded; }
    virtual void print_insn( FILE *fp ) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
    }
    bool is_load() const { return (op == LOAD_OP || memory_op == memory_load); }
    bool is_store() const { return (op == STORE_OP || memory_op == memory_store); }

    address_type pc;        // program counter address of instruction
    unsigned isize;         // size of instruction in bytes 
    op_type op;             // opcode (uarch visible)
    _memory_op_t memory_op; // memory_op used by ptxplus 

    address_type reconvergence_pc; // -1 => not a branch, -2 => use function return address
    
    unsigned out[4];
    unsigned in[4];
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred; // predicate register number
    int ar1, ar2;
    int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation
    unsigned latency; // operation latency 
    unsigned initiation_interval;

    unsigned data_size; // what is the size of the word being operated on?
    memory_space_t space;
    cache_operator_type cache_op;

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

enum divergence_support_t {
   POST_DOMINATOR = 1,
   NUM_SIMD_MODEL
};

class warp_inst_t: public inst_t {
public:
    // constructors
    warp_inst_t() 
    {
        m_uid=0;
        m_empty=true; 
        m_config=NULL; 
    }
    warp_inst_t( const core_config *config ) 
    { 
        m_uid=0;
        assert(config->warp_size<=MAX_WARP_SIZE); 
        m_config=config;
        m_empty=true; 
        m_isatomic=false;
        m_per_scalar_thread_valid=false;
        m_mem_accesses_created=false;
        m_cache_hit=false;
    }

    // modifiers
    void do_atomic();
    void do_atomic( const active_mask_t& access_mask );
    void clear() 
    { 
        m_empty=true; 
    }
    void issue( const active_mask_t &mask, unsigned warp_id, unsigned long long cycle ) 
    {
        m_warp_active_mask=mask;
        m_uid = ++sm_next_uid;
        m_warp_id = warp_id;
        issue_cycle = cycle;
        cycles = initiation_interval;
        m_cache_hit=false;
        m_empty=false;
    }
    void set_addr( unsigned n, new_addr_type addr ) 
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        m_per_scalar_thread[n].memreqaddr[0] = addr;
    }
    void set_addr( unsigned n, new_addr_type* addr, unsigned num_addrs )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
        }
        for(unsigned i=0; i<num_addrs; i++)
            m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }
    void generate_mem_accesses();
    void add_callback( unsigned lane_id, 
                       void (*function)(const class inst_t*, class ptx_thread_info*),
                       const inst_t *inst, 
                       class ptx_thread_info *thread )
    {
        if( !m_per_scalar_thread_valid ) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid=true;
            m_isatomic=true;
        }
        m_per_scalar_thread[lane_id].callback.function = function;
        m_per_scalar_thread[lane_id].callback.instruction = inst;
        m_per_scalar_thread[lane_id].callback.thread = thread;
    }
    void set_active( const active_mask_t &active );

    void clear_active( const active_mask_t &inactive );
    void set_not_active( unsigned lane_id );

    // accessors
    virtual void print_insn(FILE *fp) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
        for (int i=(int)m_config->warp_size-1; i>=0; i--)
            fprintf(fp, "%c", ((m_warp_active_mask[i])?'1':'0') );
    }
    bool active( unsigned thread ) const { return m_warp_active_mask.test(thread); }
    unsigned active_count() const { return m_warp_active_mask.count(); }
    bool empty() const { return m_empty; }
    unsigned warp_id() const 
    { 
        assert( !m_empty );
        return m_warp_id; 
    }
    bool has_callback( unsigned n ) const
    {
        return m_warp_active_mask[n] && m_per_scalar_thread_valid && 
            (m_per_scalar_thread[n].callback.function!=NULL);
    }
    new_addr_type get_addr( unsigned n ) const
    {
        assert( m_per_scalar_thread_valid );
        return m_per_scalar_thread[n].memreqaddr[0];
    }

    bool isatomic() const { return m_isatomic; }

    unsigned warp_size() const { return m_config->warp_size; }

    bool accessq_empty() const { return m_accessq.empty(); }
    const mem_access_t &accessq_back() { return m_accessq.back(); }
    void accessq_pop_back() { m_accessq.pop_back(); }

    bool dispatch_delay()
    { 
        if( cycles > 0 ) 
            cycles--;
        return cycles > 0;
    }

    void print( FILE *fout ) const;

protected:

    unsigned m_uid;
    bool m_empty;
    bool m_cache_hit;
    unsigned long long issue_cycle;
    unsigned cycles; // used for implementing initiation interval delay
    bool m_isatomic;
    unsigned m_warp_id;
    const core_config *m_config; 
    active_mask_t m_warp_active_mask;

    struct per_thread_info {
        per_thread_info() {
            for(unsigned i=0; i<8; i++)
                memreqaddr[i] = 0;
        }
        dram_callback_t callback;
        new_addr_type memreqaddr[8]; // effective address, upto 8 different requests (to support 32B access in 8 chunks of 4B each)
    };
    bool m_per_scalar_thread_valid;
    std::vector<per_thread_info> m_per_scalar_thread;
    bool m_mem_accesses_created;
    std::list<mem_access_t> m_accessq;

    static unsigned sm_next_uid;
};

void move_warp( warp_inst_t *&dst, warp_inst_t *&src );

size_t get_kernel_code_size( class function_info *entry );

#endif // #ifdef __cplusplus

#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
