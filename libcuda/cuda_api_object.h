#ifndef __cuda_api_object_h__
#define __cuda_api_object_h__

#include <list>
#include <map>
#include <set>
#include <string>

#include "builtin_types.h"

#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "cuobjdump.h"

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

struct glbmap_entry {
  GLuint m_bufferObj;
  void *m_devPtr;
  size_t m_size;
  struct glbmap_entry *m_next;
};

typedef struct glbmap_entry glbmap_entry_t;

struct _cuda_device_id {
  _cuda_device_id(gpgpu_sim *gpu) {
    m_id = 0;
    m_next = NULL;
    m_gpgpu = gpu;
  }
  struct _cuda_device_id *next() {
    return m_next;
  }
  unsigned num_shader() const { return m_gpgpu->get_config().num_shader(); }
  int num_devices() const {
    if (m_next == NULL)
      return 1;
    else
      return 1 + m_next->num_devices();
  }
  struct _cuda_device_id *get_device(unsigned n) {
    assert(n < (unsigned)num_devices());
    struct _cuda_device_id *p = this;
    for (unsigned i = 0; i < n; i++) p = p->m_next;
    return p;
  }
  const struct cudaDeviceProp *get_prop() const { return m_gpgpu->get_prop(); }
  unsigned get_id() const { return m_id; }

  gpgpu_sim *get_gpgpu() { return m_gpgpu; }

 private:
  unsigned m_id;
  class gpgpu_sim *m_gpgpu;
  struct _cuda_device_id *m_next;
};

struct CUctx_st {
  CUctx_st(_cuda_device_id *gpu) {
    m_gpu = gpu;
    m_binary_info.cmem = 0;
    m_binary_info.gmem = 0;
    no_of_ptx = 0;
  }

  _cuda_device_id *get_device() { return m_gpu; }

  void add_binary(symbol_table *symtab, unsigned fat_cubin_handle) {
    m_code[fat_cubin_handle] = symtab;
    m_last_fat_cubin_handle = fat_cubin_handle;
  }

  void add_ptxinfo(const char *deviceFun,
                   const struct gpgpu_ptx_sim_info &info) {
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
    assert(s != NULL);
    function_info *f = s->get_pc();
    assert(f != NULL);
    f->set_kernel_info(info);
  }

  void add_ptxinfo(const struct gpgpu_ptx_sim_info &info) {
    m_binary_info = info;
  }

  void register_function(unsigned fat_cubin_handle, const char *hostFun,
                         const char *deviceFun) {
    if (m_code.find(fat_cubin_handle) != m_code.end()) {
      symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
      if (s != NULL) {
        function_info *f = s->get_pc();
        assert(f != NULL);
        m_kernel_lookup[hostFun] = f;
      } else {
        printf("Warning: cannot find deviceFun %s\n", deviceFun);
        m_kernel_lookup[hostFun] = NULL;
      }
      //		assert( s != NULL );
      //		function_info *f = s->get_pc();
      //		assert( f != NULL );
      //		m_kernel_lookup[hostFun] = f;
    } else {
      m_kernel_lookup[hostFun] = NULL;
    }
  }

  void register_hostFun_function(const char *hostFun, function_info *f) {
    m_kernel_lookup[hostFun] = f;
  }

  function_info *get_kernel(const char *hostFun) {
    std::map<const void *, function_info *>::iterator i =
        m_kernel_lookup.find(hostFun);
    assert(i != m_kernel_lookup.end());
    return i->second;
  }

  int no_of_ptx;

 private:
  _cuda_device_id *m_gpu;  // selected gpu
  std::map<unsigned, symbol_table *>
      m_code;  // fat binary handle => global symbol table
  unsigned m_last_fat_cubin_handle;
  std::map<const void *, function_info *>
      m_kernel_lookup;  // unique id (CUDA app function address) => kernel entry
                        // point
  struct gpgpu_ptx_sim_info m_binary_info;
};

class kernel_config {
 public:
  kernel_config(dim3 GridDim, dim3 BlockDim, size_t sharedMem,
                struct CUstream_st *stream) {
    m_GridDim = GridDim;
    m_BlockDim = BlockDim;
    m_sharedMem = sharedMem;
    m_stream = stream;
  }
  kernel_config() {
    m_GridDim = dim3(-1, -1, -1);
    m_BlockDim = dim3(-1, -1, -1);
    m_sharedMem = 0;
    m_stream = NULL;
  }
  void set_arg(const void *arg, size_t size, size_t offset) {
    m_args.push_front(gpgpu_ptx_sim_arg(arg, size, offset));
  }
  dim3 grid_dim() const { return m_GridDim; }
  dim3 block_dim() const { return m_BlockDim; }
  void set_grid_dim(dim3 *d) { m_GridDim = *d; }
  void set_block_dim(dim3 *d) { m_BlockDim = *d; }
  gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
  struct CUstream_st *get_stream() {
    return m_stream;
  }

 private:
  dim3 m_GridDim;
  dim3 m_BlockDim;
  size_t m_sharedMem;
  struct CUstream_st *m_stream;
  gpgpu_ptx_sim_arg_list_t m_args;
};

class cuda_runtime_api {
 public:
  cuda_runtime_api(gpgpu_context *ctx) {
    g_glbmap = NULL;
    g_active_device = 0;  // active gpu that runs the code
    gpgpu_ctx = ctx;
  }
  // global list
  std::list<cuobjdumpSection *> cuobjdumpSectionList;
  std::list<cuobjdumpSection *> libSectionList;
  std::list<kernel_config> g_cuda_launch_stack;
  std::map<int, bool> fatbin_registered;
  std::map<int, std::string> fatbinmap;
  std::map<std::string, symbol_table *> name_symtab;
  std::map<unsigned long long, size_t> g_mallocPtr_Size;
  // maps sm version number to set of filenames
  std::map<unsigned, std::set<std::string> > version_filename;
  std::map<void *, void **> pinned_memory;  // support for pinned memories added
  std::map<void *, size_t> pinned_memory_size;
  glbmap_entry_t *g_glbmap;
  int g_active_device;  // active gpu that runs the code
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // member function list
  void cuobjdumpInit();
  void extract_code_using_cuobjdump();
  void extract_ptx_files_using_cuobjdump(CUctx_st *context);
  std::list<cuobjdumpSection *> pruneSectionList(CUctx_st *context);
  std::list<cuobjdumpSection *> mergeMatchingSections(std::string identifier);
  std::list<cuobjdumpSection *> mergeSections();
  cuobjdumpELFSection *findELFSection(const std::string identifier);
  cuobjdumpPTXSection *findPTXSection(const std::string identifier);
  cuobjdumpPTXSection *findPTXSectionInList(
      std::list<cuobjdumpSection *> &sectionlist, const std::string identifier);
  void cuobjdumpRegisterFatBinary(unsigned int handle, const char *filename,
                                  CUctx_st *context);
  kernel_info_t *gpgpu_cuda_ptx_sim_init_grid(const char *kernel_key,
                                              gpgpu_ptx_sim_arg_list_t args,
                                              struct dim3 gridDim,
                                              struct dim3 blockDim,
                                              struct CUctx_st *context);
  int load_static_globals(symbol_table *symtab, unsigned min_gaddr,
                          unsigned max_gaddr, gpgpu_t *gpu);
  int load_constants(symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu);
};
#endif /* __cuda_api_object_h__ */
