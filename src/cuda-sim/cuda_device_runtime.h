#ifndef __cuda_device_runtime_h__
#define __cuda_device_runtime_h__
// Jin: cuda_device_runtime.h
// Defines CUDA device runtime APIs for CDP support
class device_launch_config_t {
 public:
  device_launch_config_t() {}

  device_launch_config_t(dim3 _grid_dim, dim3 _block_dim,
                         unsigned int _shared_mem, function_info* _entry)
      : grid_dim(_grid_dim),
        block_dim(_block_dim),
        shared_mem(_shared_mem),
        entry(_entry) {}

  dim3 grid_dim;
  dim3 block_dim;
  unsigned int shared_mem;
  function_info* entry;
};

class device_launch_operation_t {
 public:
  device_launch_operation_t() {}
  device_launch_operation_t(kernel_info_t* _grid, CUstream_st* _stream)
      : grid(_grid), stream(_stream) {}

  kernel_info_t* grid;  // a new child grid

  CUstream_st* stream;
};

class gpgpu_context;

class cuda_device_runtime {
 public:
  cuda_device_runtime(gpgpu_context* ctx) {
    g_total_param_size = 0;
    g_max_total_param_size = 0;
    gpgpu_ctx = ctx;
  }
  unsigned long long g_total_param_size;
  std::map<void*, device_launch_config_t> g_cuda_device_launch_param_map;
  std::list<device_launch_operation_t> g_cuda_device_launch_op;
  unsigned g_kernel_launch_latency;
  unsigned g_TB_launch_latency;
  unsigned long long g_max_total_param_size;
  bool g_cdp_enabled;

  // backward pointer
  class gpgpu_context* gpgpu_ctx;
#if (CUDART_VERSION >= 5000)
#pragma once
  void gpgpusim_cuda_launchDeviceV2(const ptx_instruction* pI,
                                    ptx_thread_info* thread,
                                    const function_info* target_func);
  void gpgpusim_cuda_streamCreateWithFlags(const ptx_instruction* pI,
                                           ptx_thread_info* thread,
                                           const function_info* target_func);
  void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction* pI,
                                          ptx_thread_info* thread,
                                          const function_info* target_func);
  void launch_all_device_kernels();
  void launch_one_device_kernel();
#endif
};

#endif /* __cuda_device_runtime_h__  */
