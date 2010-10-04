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
extern void ** g_inst_classification_stat;
extern void ** g_inst_op_classification_stat;
extern int g_ptx_kernel_count; // used for classification stat collection purposes 
extern FILE* ptx_inst_debug_file;


extern class kernel_info_t gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                            gpgpu_ptx_sim_arg_list_t args, 
                                            struct dim3 gridDim, 
                                            struct dim3 blockDim );
extern void gpgpu_cuda_ptx_sim_main_func( kernel_info_t kernel, dim3 gridDim, dim3 blockDim, gpgpu_ptx_sim_arg_list_t args);
extern void   print_splash();
extern void*  gpgpu_ptx_sim_malloc( size_t count );
extern void*  gpgpu_ptx_sim_mallocarray( size_t count );
extern void   gpgpu_ptx_sim_memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count );
extern void   gpgpu_ptx_sim_memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count );
extern void   gpgpu_ptx_sim_memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count );
extern void   gpgpu_ptx_sim_memset( size_t dst_start_addr, int c, size_t count );
extern void   gpgpu_ptx_sim_init_memory();
extern void   gpgpu_ptx_sim_register_kernel(void **fatCubinHandle,const char *hostFun, const char *deviceFun);
extern void   gpgpu_ptx_sim_register_const_variable(void*, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to );

extern void   gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array);
extern void   gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref);
extern int    gpgpu_ptx_sim_sizeofTexture(const char* name);
extern const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref);
extern const struct textureReference* gpgpu_ptx_sim_accessTextureofName(const char* name); 
extern void read_sim_environment_variables();
extern void ptxinfo_opencl_addinfo( std::map<std::string,function_info*> &kernels );
unsigned ptx_sim_init_thread( kernel_info_t &kernel,
                              class ptx_thread_info** thread_info,
                              int sid,
                              unsigned tid,
                              unsigned threads_left,
                              unsigned num_threads, 
                              class core_t *core, 
                              unsigned hw_cta_id, 
                              unsigned hw_warp_id );
const inst_t *ptx_fetch_inst( address_type pc );
const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info(class function_info *kernel);
unsigned ptx_thread_donecycle( void *thr );
void* ptx_thread_get_next_finfo( void *thd );
int ptx_thread_at_barrier( void *thd );
int ptx_thread_all_at_barrier( void *thd );
unsigned long long ptx_thread_get_cta_uid( void *thd );
void ptx_thread_reset_barrier( void *thd );
void ptx_thread_release_barrier( void *thd );
void ptx_print_insn( address_type pc, FILE *fp );
unsigned int ptx_set_tex_cache_linesize( unsigned linesize);

void dwf_process_reconv_pts(function_info *entry);
void set_param_gpgpu_num_shaders(int num_shaders);
unsigned int get_converge_point(unsigned int pc, void *thd);
const char *get_ptxinfo_kname();
void print_ptxinfo();
void clear_ptxinfo();
struct gpgpu_ptx_sim_kernel_info get_ptxinfo_kinfo();

#endif
