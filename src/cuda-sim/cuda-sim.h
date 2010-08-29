#ifndef CUDASIM_H_INCLUDED
#define CUDASIM_H_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram_callback.h"
#include <stdlib.h>
#include <map>
#include <string>

class memory_space;
class function_info;
class symbol_table;

extern const char *g_gpgpusim_version_string;
extern int g_ptx_sim_mode;
extern memory_space *g_global_mem;
extern int g_debug_execution;
extern int g_debug_thread_uid;
extern std::map<std::string,function_info*> *g_kernel_name_to_function_lookup;
extern void ** g_inst_classification_stat;
extern void ** g_inst_op_classification_stat;
extern int g_ptx_kernel_count; // used for classification stat collection purposes 


extern void   gpgpu_cuda_ptx_sim_init_grid( const char *kernel_key,
                                            struct gpgpu_ptx_sim_arg *args, 
                                            struct dim3 gridDim, 
                                            struct dim3 blockDim );
extern void   gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                            struct gpgpu_ptx_sim_arg *args, 
                                            struct dim3 gridDim, 
                                            struct dim3 blockDim );
extern void   gpgpu_cuda_ptx_sim_main_func( const char *kernel_key, dim3 gridDim, dim3 blockDim, struct gpgpu_ptx_sim_arg *);
extern void   print_splash();
extern void*  gpgpu_ptx_sim_malloc( size_t count );
extern void*  gpgpu_ptx_sim_mallocarray( size_t count );
extern void   gpgpu_ptx_sim_memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
extern void   gpgpu_ptx_sim_memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
extern void   gpgpu_ptx_sim_memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
extern void   gpgpu_ptx_sim_memset( size_t dst_start_addr, int c, size_t count );
extern void   gpgpu_ptx_sim_init_memory();
extern void   gpgpu_ptx_sim_load_gpu_kernels();
extern void   gpgpu_ptx_sim_register_kernel(const char *hostFun, const char *deviceFun);
extern void   gpgpu_ptx_sim_register_const_variable(void*, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to );

extern void   gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
extern void   gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
extern int    gpgpu_ptx_sim_sizeofTexture(const char* name);
extern const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);
extern const struct textureReference* gpgpu_ptx_sim_accessTextureofName(const char* name); 
extern void read_sim_environment_variables();
extern void register_function_implementation( const char *name, function_info *impl );
extern void ptxinfo_cuda_addinfo();
extern void ptxinfo_opencl_addinfo( std::map<std::string,function_info*> &kernels );
extern void ptx_sim_free_sm( class ptx_thread_info** thread_info );
unsigned ptx_sim_init_thread( class ptx_thread_info** thread_info,
                              int sid,
                              unsigned tid,
                              unsigned threads_left,
                              unsigned num_threads, 
                              class core_t *core, 
                              unsigned hw_cta_id, 
                              unsigned hw_warp_id );
unsigned ptx_sim_cta_size();
const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info();
void set_option_gpgpu_spread_blocks_across_cores(int option);
int ptx_thread_done( void *thd );
unsigned ptx_thread_donecycle( void *thr );
int ptx_thread_get_next_pc( void *thd );
void* ptx_thread_get_next_finfo( void *thd );
int ptx_thread_at_barrier( void *thd );
int ptx_thread_all_at_barrier( void *thd );
unsigned long long ptx_thread_get_cta_uid( void *thd );
void ptx_thread_reset_barrier( void *thd );
void ptx_thread_release_barrier( void *thd );
void ptx_print_insn( address_type pc, FILE *fp );
unsigned int ptx_set_tex_cache_linesize( unsigned linesize);
void ptx_decode_inst( void *thd, 
                      unsigned *op, 
                      int *i1, 
                      int *i2, 
                      int *i3, 
                      int *i4, 
                      int *o1, 
                      int *o2, 
                      int *o3, 
                      int *o4, 
                      int *vectorin, 
                      int *vectorout, 
                      int *arch_reg, 
                      int *pred,
                      int *ar1, 
                      int *ar2 );
void ptx_exec_inst( void *thd, 
                    address_type *addr, 
                    memory_space_t *space, 
                    unsigned *data_size,
                    unsigned *cycles, 
                    dram_callback_t* callback, 
                    unsigned warp_active_mask );

#endif
