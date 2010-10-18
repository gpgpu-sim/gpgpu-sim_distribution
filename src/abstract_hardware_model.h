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

struct core_config {
    core_config() { m_valid = false; }
    virtual void init() = 0;

    bool m_valid;
    unsigned warp_size;

    // memory request architecture parameters
    int gpgpu_coalesce_arch;
    int gpgpu_shmem_pipe_speedup;  
    unsigned gpgpu_cache_texl1_linesize;
    unsigned gpgpu_cache_constl1_linesize;

    static const address_type WORD_SIZE=4;
    unsigned null_bank_func(address_type, unsigned) const { return 1; }
    int gpgpu_n_shmem_bank;
    unsigned shmem_bank_func(address_type addr, unsigned) const;
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

class gpgpu_t {
public:
    gpgpu_t();
    void* gpu_malloc( size_t size );
    void* gpu_mallocarray( size_t count );
    void  gpu_memset( size_t dst_start_addr, int c, size_t count );
    void  memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
    void  memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
    void  memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
    
    class memory_space *get_global_memory() { return m_global_mem; }
    class memory_space *get_tex_memory() { return m_tex_mem; }
    class memory_space *get_surf_memory() { return m_surf_mem; }
    class memory_space *get_param_memory() { return m_param_mem; }

    void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
    void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
    const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);
    unsigned ptx_set_tex_cache_linesize(unsigned linesize);

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

protected:
   class memory_space *m_global_mem;
   class memory_space *m_tex_mem;
   class memory_space *m_surf_mem;
   class memory_space *m_param_mem;

   unsigned long long m_dev_malloc;

   std::map<std::string, const struct textureReference*> m_NameToTextureRef;
   std::map<const struct textureReference*,const struct cudaArray*> m_TextureRefToCudaArray;
   std::map<const struct textureReference*, const struct textureInfo*> m_TextureRefToTexureInfo;
   unsigned int m_texcache_linesize;
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

class mem_access_t {
public:
   mem_access_t()
   { 
      init();
   }
   mem_access_t(address_type address, unsigned size, unsigned quarter, unsigned idx )
   {
      init();
      addr = address;
      req_size = size;
      quarter_count[quarter]++;
      warp_indices.push_back(idx);
   }

   bool operator<(const mem_access_t &other) const {return (order > other.order);}//this is reverse

private:
   void init() 
   {
      uid=++next_access_uid;
      addr=0;
      req_size=0;
      order=0;
      _quarter_count_all=0;
      cache_hit = false;
      cache_checked = false;
      recheck_cache = false;
      need_wb = false;
      wb_addr = 0;
   }

public:

   unsigned uid;
   address_type addr; //address of the segment to load.
   unsigned req_size; //bytes
   unsigned order; // order of accesses, based on banks.
   union{
     unsigned _quarter_count_all;
     char quarter_count[4]; //access counts to each quarter of segment, for compaction;
   };
   std::vector<unsigned> warp_indices; // warp indicies for this request.
   bool cache_hit;
   bool cache_checked;
   bool recheck_cache;
   bool need_wb;
   address_type wb_addr; // writeback address (if necessary).

private:
   static unsigned next_access_uid;
};

class mem_fetch;

class mem_fetch_interface {
public:
    virtual bool full( unsigned size, bool write ) const = 0;
    virtual void push( mem_fetch *mf ) = 0;
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

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

#define MAX_WARP_SIZE 32

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
        for (int i=(int)m_config->warp_size-1; i>=0; i--) {
            if( mask & (1<<i) )
                warp_active_mask.set(i);
        }
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
        m_per_scalar_thread[n].memreqaddr = addr;
    }
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
    void set_active( std::vector<unsigned> &active ) 
    {
       warp_active_mask.reset();
       for( std::vector<unsigned>::iterator i=active.begin(); i!=active.end(); ++i ) {
          unsigned t = *i;
          assert( t < m_config->warp_size );
          warp_active_mask.set(t);
       }
       if( m_isatomic ) {
          for( unsigned i=0; i < m_config->warp_size; i++ ) {
             if( !warp_active_mask.test(i) ) {
                 m_per_scalar_thread[i].callback.function = NULL;
                 m_per_scalar_thread[i].callback.instruction = NULL;
                 m_per_scalar_thread[i].callback.thread = NULL;
             }
          }
       }
    }
    void clear_active( std::vector<unsigned> &inactive )
    {
        std::vector<unsigned>::iterator i;
        for(i=inactive.begin(); i!=inactive.end();i++) {
            unsigned t=*i;
            warp_active_mask.reset(t);
        }
    }
    void set_not_active( unsigned lane_id )
    {
        warp_active_mask.reset(lane_id);
    }
    void get_memory_access_list();

    // accessors
    virtual void print_insn(FILE *fp) const 
    {
        fprintf(fp," [inst @ pc=0x%04x] ", pc );
        for (int i=(int)m_config->warp_size-1; i>=0; i--)
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

    unsigned warp_size() const { return m_config->warp_size; }

    bool mem_accesses_created() const { return m_mem_accesses_created; }
    void set_mem_accesses_created() { m_mem_accesses_created=true; }
    bool accessq_empty() const { return m_accessq.empty(); }
    unsigned get_accessq_size() const { return m_accessq.size(); }
    mem_access_t &accessq( unsigned n ) { return m_accessq[n]; }
    mem_access_t &accessq_back() { return m_accessq.back(); }
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
    std::bitset<MAX_WARP_SIZE> warp_active_mask;

    struct per_thread_info {
        per_thread_info() {
            memreqaddr=0;
        }
        dram_callback_t callback;
        new_addr_type memreqaddr; // effective address
    };
    bool m_per_scalar_thread_valid;
    std::vector<per_thread_info> m_per_scalar_thread;
    bool m_mem_accesses_created;
    std::vector<mem_access_t> m_accessq;

    static unsigned sm_next_uid;
};

void move_warp( warp_inst_t *&dst, warp_inst_t *&src );

size_t get_kernel_code_size( class function_info *entry );

#endif // #ifdef __cplusplus

#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
