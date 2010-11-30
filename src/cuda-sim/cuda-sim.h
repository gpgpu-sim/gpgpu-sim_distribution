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
extern int g_debug_execution;
extern int g_debug_thread_uid;
extern void ** g_inst_classification_stat;
extern void ** g_inst_op_classification_stat;
extern int g_ptx_kernel_count; // used for classification stat collection purposes 


extern class kernel_info_t gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                            gpgpu_ptx_sim_arg_list_t args, 
                                            struct dim3 gridDim, 
                                            struct dim3 blockDim, 
                                                          class gpgpu_t *gpu );
extern void gpgpu_cuda_ptx_sim_main_func( kernel_info_t kernel, dim3 gridDim, dim3 blockDim, gpgpu_ptx_sim_arg_list_t args);
extern void   print_splash();
extern void   gpgpu_ptx_sim_register_const_variable(void*, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size );
extern void   gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to, gpgpu_t *gpu );

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
                              unsigned hw_warp_id,
                              gpgpu_t *gpu );
const warp_inst_t *ptx_fetch_inst( address_type pc );
const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info(const class function_info *kernel);
void ptx_print_insn( address_type pc, FILE *fp );
void set_param_gpgpu_num_shaders(int num_shaders);

#define RECONVERGE_RETURN_PC ((address_type)-2)
#define NO_BRANCH_DIVERGENCE ((address_type)-1)
address_type get_return_pc( void *thd );
const char *get_ptxinfo_kname();
void print_ptxinfo();
void clear_ptxinfo();
struct gpgpu_ptx_sim_kernel_info get_ptxinfo_kinfo();

#endif
