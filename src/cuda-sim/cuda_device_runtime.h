//Jin: cuda_device_runtime.h
//Defines CUDA device runtime APIs for CDP support
#if (CUDART_VERSION >= 5000)
#pragma once

void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func);
void gpgpusim_cuda_launchDeviceV2(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func);
void gpgpusim_cuda_streamCreateWithFlags(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func);
void launch_all_device_kernels();
void launch_one_device_kernel();
#endif
