// Jin: cuda_device_runtime.cc
// Defines CUDA device runtime APIs for CDP support

#include <iostream>
#include <map>

#if (CUDART_VERSION >= 5000)
#define __CUDA_RUNTIME_API_H__

#include <builtin_types.h>
#include <driver_types.h>
#include "../../libcuda/gpgpu_context.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpusim_entrypoint.h"
#include "../stream_manager.h"
#include "cuda-sim.h"
#include "cuda_device_runtime.h"
#include "ptx_ir.h"

#define DEV_RUNTIME_REPORT(a)                                       \
  if (g_debug_execution) {                                          \
    std::cout << __FILE__ << ", " << __LINE__ << ": " << a << "\n"; \
    std::cout.flush();                                              \
  }

// Handling device runtime api:
// void * cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3
// blockDimension, unsigned int sharedMemSize)
void cuda_device_runtime::gpgpusim_cuda_getParameterBufferV2(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const function_info *target_func) {
  DEV_RUNTIME_REPORT("Calling cudaGetParameterBufferV2");

  unsigned n_return = target_func->has_return();
  assert(n_return);
  unsigned n_args = target_func->num_args();
  assert(n_args == 4);

  function_info *child_kernel_entry;
  struct dim3 grid_dim, block_dim;
  unsigned int shared_mem;

  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);  // param#
    const symbol *formal_param =
        target_func->get_arg(arg);  // cudaGetParameterBufferV2_param_#
    unsigned size = formal_param->get_size_in_bytes();
    assert(formal_param->is_param_local());
    assert(actual_param_op.is_param_local());
    addr_t from_addr = actual_param_op.get_symbol()->get_address();

    if (arg == 0) {  // function_info* for the child kernel
      unsigned long long buf;
      assert(size == sizeof(function_info *));
      thread->m_local_mem->read(from_addr, size, &buf);
      child_kernel_entry = (function_info *)buf;
      assert(child_kernel_entry);
      DEV_RUNTIME_REPORT("child kernel name "
                         << child_kernel_entry->get_name());
    } else if (arg == 1) {  // dim3 grid_dim for the child kernel
      assert(size == sizeof(struct dim3));
      thread->m_local_mem->read(from_addr, size, &grid_dim);
      DEV_RUNTIME_REPORT("grid (" << grid_dim.x << ", " << grid_dim.y << ", "
                                  << grid_dim.z << ")");
    } else if (arg == 2) {  // dim3 block_dim for the child kernel
      assert(size == sizeof(struct dim3));
      thread->m_local_mem->read(from_addr, size, &block_dim);
      DEV_RUNTIME_REPORT("block (" << block_dim.x << ", " << block_dim.y << ", "
                                   << block_dim.z << ")");
    } else if (arg == 3) {  // unsigned int shared_mem
      assert(size == sizeof(unsigned int));
      thread->m_local_mem->read(from_addr, size, &shared_mem);
      DEV_RUNTIME_REPORT("shared memory " << shared_mem);
    }
  }

  // get total child kernel argument size and malloc buffer in global memory
  unsigned child_kernel_arg_size = child_kernel_entry->get_args_aligned_size();
  void *param_buffer = thread->get_gpu()->gpu_malloc(child_kernel_arg_size);
  g_total_param_size += ((child_kernel_arg_size + 255) / 256 * 256);
  DEV_RUNTIME_REPORT("child kernel arg size total "
                     << child_kernel_arg_size
                     << ", parameter buffer allocated at " << param_buffer);
  if (g_total_param_size > g_max_total_param_size)
    g_max_total_param_size = g_total_param_size;

  // store param buffer address and launch config
  device_launch_config_t device_launch_config(grid_dim, block_dim, shared_mem,
                                              child_kernel_entry);
  assert(g_cuda_device_launch_param_map.find(param_buffer) ==
         g_cuda_device_launch_param_map.end());
  g_cuda_device_launch_param_map[param_buffer] = device_launch_config;

  // copy the buffer address to retval0
  const operand_info &actual_return_op = pI->operand_lookup(0);  // retval0
  const symbol *formal_return = target_func->get_return_var();   // void *
  unsigned int return_size = formal_return->get_size_in_bytes();
  DEV_RUNTIME_REPORT("cudaGetParameterBufferV2 return value has size of "
                     << return_size);
  assert(actual_return_op.is_param_local());
  assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size &&
         return_size == sizeof(void *));
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
  thread->m_local_mem->write(ret_param_addr, return_size, &param_buffer, NULL,
                             NULL);
}

// Handling device runtime api:
// cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
void cuda_device_runtime::gpgpusim_cuda_launchDeviceV2(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const function_info *target_func) {
  DEV_RUNTIME_REPORT("Calling cudaLaunchDeviceV2");

  unsigned n_return = target_func->has_return();
  assert(n_return);
  unsigned n_args = target_func->num_args();
  assert(n_args == 2);

  kernel_info_t *device_grid = NULL;
  function_info *device_kernel_entry = NULL;
  void *parameter_buffer;
  struct CUstream_st *child_stream;
  device_launch_config_t config;
  device_launch_operation_t device_launch_op;

  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);  // param#
    const symbol *formal_param =
        target_func->get_arg(arg);  // cudaLaunchDeviceV2_param_#
    unsigned size = formal_param->get_size_in_bytes();
    assert(formal_param->is_param_local());
    assert(actual_param_op.is_param_local());
    addr_t from_addr = actual_param_op.get_symbol()->get_address();

    if (arg == 0) {  // paramter buffer for child kernel (in global memory)
      // get parameter_buffer from the cudaLaunchDeviceV2_param0
      assert(size == sizeof(void *));
      thread->m_local_mem->read(from_addr, size, &parameter_buffer);
      assert((size_t)parameter_buffer >= GLOBAL_HEAP_START);
      DEV_RUNTIME_REPORT("Parameter buffer locating at global memory "
                         << parameter_buffer);

      // get child grid info through parameter_buffer address
      assert(g_cuda_device_launch_param_map.find(parameter_buffer) !=
             g_cuda_device_launch_param_map.end());
      config = g_cuda_device_launch_param_map[parameter_buffer];
      // device_grid = op.grid;
      device_kernel_entry = config.entry;
      DEV_RUNTIME_REPORT("find device kernel "
                         << device_kernel_entry->get_name());

      // PDOM analysis is done for Parent kernel but not for child kernel.
      if (device_kernel_entry->is_pdom_set()) {
        printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
               device_kernel_entry->get_name().c_str());
      } else {
        printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
               device_kernel_entry->get_name().c_str());
        /*
         * Some of the instructions like printf() gives the gpgpusim the wrong
         * impression that it is a function call. As printf() doesnt have a body
         * like functions do, doing pdom analysis for printf() causes a crash.
         */
        if (device_kernel_entry->get_function_size() > 0)
          device_kernel_entry->do_pdom();
        device_kernel_entry->set_pdom();
      }

      // copy data in parameter_buffer to device kernel param memory
      unsigned device_kernel_arg_size =
          device_kernel_entry->get_args_aligned_size();
      DEV_RUNTIME_REPORT("device_kernel_arg_size " << device_kernel_arg_size);
      memory_space *device_kernel_param_mem;

      // create child kernel_info_t and index it with parameter_buffer address
      gpgpu_t *gpu = thread->get_gpu();
      device_grid = new kernel_info_t(
          config.grid_dim, config.block_dim, device_kernel_entry,
          gpu->getNameArrayMapping(), gpu->getNameInfoMapping());
      device_grid->launch_cycle = gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
      kernel_info_t &parent_grid = thread->get_kernel();
      DEV_RUNTIME_REPORT(
          "child kernel launched by "
          << parent_grid.name() << ", cta (" << thread->get_ctaid().x << ", "
          << thread->get_ctaid().y << ", " << thread->get_ctaid().z
          << "), thread (" << thread->get_tid().x << ", " << thread->get_tid().y
          << ", " << thread->get_tid().z << ")");
      device_grid->set_parent(&parent_grid, thread->get_ctaid(),
                              thread->get_tid());
      device_launch_op = device_launch_operation_t(device_grid, NULL);
      device_kernel_param_mem = device_grid->get_param_memory();  // kernel
                                                                  // param
      size_t param_start_address = 0;
      // copy in word
      for (unsigned n = 0; n < device_kernel_arg_size; n += 4) {
        unsigned int oneword;
        thread->get_gpu()->get_global_memory()->read(
            (size_t)parameter_buffer + n, 4, &oneword);
        device_kernel_param_mem->write(param_start_address + n, 4, &oneword,
                                       NULL, NULL);
      }
    } else if (arg == 1) {  // cudaStream for the child kernel

      assert(size == sizeof(cudaStream_t));
      thread->m_local_mem->read(from_addr, size, &child_stream);

      kernel_info_t &parent_kernel = thread->get_kernel();
      if (child_stream == 0) {  // default stream on device for current CTA
        child_stream =
            parent_kernel.get_default_stream_cta(thread->get_ctaid());
        DEV_RUNTIME_REPORT("launching child kernel "
                           << device_grid->get_uid()
                           << " to default stream of the cta "
                           << child_stream->get_uid() << ": " << child_stream);
      } else {
        assert(parent_kernel.cta_has_stream(thread->get_ctaid(), child_stream));
        DEV_RUNTIME_REPORT("launching child kernel "
                           << device_grid->get_uid() << " to stream "
                           << child_stream->get_uid() << ": " << child_stream);
      }

      device_launch_op.stream = child_stream;
    }
  }

  // launch child kernel
  g_cuda_device_launch_op.push_back(device_launch_op);
  g_cuda_device_launch_param_map.erase(parameter_buffer);

  // set retval0
  const operand_info &actual_return_op = pI->operand_lookup(0);  // retval0
  const symbol *formal_return = target_func->get_return_var();   // cudaError_t
  unsigned int return_size = formal_return->get_size_in_bytes();
  DEV_RUNTIME_REPORT("cudaLaunchDeviceV2 return value has size of "
                     << return_size);
  assert(actual_return_op.is_param_local());
  assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size &&
         return_size == sizeof(cudaError_t));
  cudaError_t error = cudaSuccess;
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
  thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);
}

// Handling device runtime api:
// cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int
// flags) flags can only be cudaStreamNonBlocking
void cuda_device_runtime::gpgpusim_cuda_streamCreateWithFlags(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const function_info *target_func) {
  DEV_RUNTIME_REPORT("Calling cudaStreamCreateWithFlags");

  unsigned n_return = target_func->has_return();
  assert(n_return);
  unsigned n_args = target_func->num_args();
  assert(n_args == 2);

  size_t generic_pStream_addr;
  addr_t pStream_addr;
  unsigned int flags;
  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);  // param#
    const symbol *formal_param =
        target_func->get_arg(arg);  // cudaStreamCreateWithFlags_param_#
    unsigned size = formal_param->get_size_in_bytes();
    assert(formal_param->is_param_local());
    assert(actual_param_op.is_param_local());
    addr_t from_addr = actual_param_op.get_symbol()->get_address();

    if (arg == 0) {  // cudaStream_t * pStream, address of cudaStream_t
      assert(size == sizeof(cudaStream_t *));
      thread->m_local_mem->read(from_addr, size, &generic_pStream_addr);

      // pStream should be non-zero address in local memory
      pStream_addr = generic_to_local(
          thread->get_hw_sid(), thread->get_hw_tid(), generic_pStream_addr);

      DEV_RUNTIME_REPORT("pStream locating at local memory " << pStream_addr);
    } else if (arg ==
               1) {  // unsigned int flags, should be cudaStreamNonBlocking
      assert(size == sizeof(unsigned int));
      thread->m_local_mem->read(from_addr, size, &flags);
      assert(flags == cudaStreamNonBlocking);
    }
  }

  // create stream and write back to param0
  CUstream_st *stream =
      thread->get_kernel().create_stream_cta(thread->get_ctaid());
  DEV_RUNTIME_REPORT("Create stream " << stream->get_uid() << ": " << stream);
  thread->m_local_mem->write(pStream_addr, sizeof(cudaStream_t), &stream, NULL,
                             NULL);

  // set retval0
  const operand_info &actual_return_op = pI->operand_lookup(0);  // retval0
  const symbol *formal_return = target_func->get_return_var();   // cudaError_t
  unsigned int return_size = formal_return->get_size_in_bytes();
  DEV_RUNTIME_REPORT("cudaStreamCreateWithFlags return value has size of "
                     << return_size);
  assert(actual_return_op.is_param_local());
  assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size &&
         return_size == sizeof(cudaError_t));
  cudaError_t error = cudaSuccess;
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
  thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);
}

void cuda_device_runtime::launch_one_device_kernel() {
  if (!g_cuda_device_launch_op.empty()) {
    device_launch_operation_t &op = g_cuda_device_launch_op.front();

    stream_operation stream_op = stream_operation(
        op.grid, gpgpu_ctx->func_sim->g_ptx_sim_mode, op.stream);
    gpgpu_ctx->the_gpgpusim->g_stream_manager->push(stream_op);
    g_cuda_device_launch_op.pop_front();
  }
}

void cuda_device_runtime::launch_all_device_kernels() {
  while (!g_cuda_device_launch_op.empty()) {
    launch_one_device_kernel();
  }
}
#endif
