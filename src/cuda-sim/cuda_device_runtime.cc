//Jin: cuda_device_runtime.cc
//Defines CUDA device runtime APIs for CDP support

#include <iostream>
#include <map>

#define __CUDA_RUNTIME_API_H__

#include <builtin_types.h>
#include <driver_types.h>
#include "../gpgpu-sim/gpu-sim.h"
#include "cuda-sim.h"
#include "ptx_ir.h"
#include "../stream_manager.h"
#include "cuda_device_runtime.h"

#define DEV_RUNTIME_REPORT(a) \
   if( g_debug_execution ) { \
      std::cout << __FILE__ << ", " << __LINE__ << ": " << a << "\n"; \
      std::cout.flush(); \
   }

std::map<void *, kernel_info_t *> g_cuda_device_launch_map;
struct CUstream_st * g_device_default_stream = NULL;
extern stream_manager *g_stream_manager;

//Handling device runtime api:
//void * cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize)
void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func)
{
    DEV_RUNTIME_REPORT("Calling cudaGetParameterBufferV2");
      
    unsigned n_return = target_func->has_return();
    assert(n_return);
    unsigned n_args = target_func->num_args();
    assert( n_args == 4 );

    function_info * child_kernel_entry;
    struct dim3 gridDim, blockDim;
    unsigned int sharedMem;

    for( unsigned arg=0; arg < n_args; arg ++ ) {
        const operand_info &actual_param_op = pI->operand_lookup(n_return+1+arg); //param#
        const symbol *formal_param = target_func->get_arg(arg); //cudaGetParameterBufferV2_param_#
        unsigned size=formal_param->get_size_in_bytes();
        assert( formal_param->is_param_local() );
        assert( actual_param_op.is_param_local() );
        addr_t from_addr = actual_param_op.get_symbol()->get_address();

        if(arg == 0) {//function_info* for the child kernel
            unsigned long long buf;
			assert(size == sizeof(function_info *));
            thread->m_local_mem->read(from_addr, size, &buf);
            child_kernel_entry = (function_info *)buf;
            assert(child_kernel_entry);
            DEV_RUNTIME_REPORT("child kernel name " << child_kernel_entry->get_name());
        }
        else if(arg == 1) { //dim3 gridDim for the child kernel
			assert(size == sizeof(struct dim3));
            thread->m_local_mem->read(from_addr, size, & gridDim);
            DEV_RUNTIME_REPORT("grid (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")");
        }
        else if(arg == 2) { //dim3 blockDim for the child kernel
			assert(size == sizeof(struct dim3));
            thread->m_local_mem->read(from_addr, size, & blockDim);
            DEV_RUNTIME_REPORT("block (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")");
        }
        else if(arg == 3) { //unsigned int sharedMem
			assert(size == sizeof(unsigned int));
            thread->m_local_mem->read(from_addr, size, & sharedMem);
            DEV_RUNTIME_REPORT("shared memory " << sharedMem);
        }
    }

	//get total child kernel argument size and malloc buffer in global memory
	unsigned child_kernel_arg_size = child_kernel_entry->get_args_aligned_size();
	void * param_buffer = thread->get_gpu()->gpu_malloc(child_kernel_arg_size);
	DEV_RUNTIME_REPORT("child kernel arg size total " << child_kernel_arg_size << ", parameter buffer allocated at " << param_buffer);
	
	//create child kernel_info_t and index it with parameter_buffer address
	kernel_info_t * child_grid = new kernel_info_t(gridDim, blockDim, child_kernel_entry); 
    assert(g_cuda_device_launch_map.find(param_buffer) == g_cuda_device_launch_map.end());
    g_cuda_device_launch_map[param_buffer] = child_grid;

	//copy the buffer address to retval0
    const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
    const symbol *formal_return = target_func->get_return_var(); //void *
	unsigned int return_size = formal_return->get_size_in_bytes();
	DEV_RUNTIME_REPORT("cudaGetParameterBufferV2 return value has size of " << return_size);
	assert(actual_return_op.is_param_local());
	assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size && return_size == sizeof(void *));
    addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
	thread->m_local_mem->write(ret_param_addr, return_size, &param_buffer, NULL, NULL);

}

//Handling device runtime api:
//cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
void gpgpusim_cuda_launchDeviceV2(const ptx_instruction * pI, ptx_thread_info * thread, const function_info * target_func) {
    DEV_RUNTIME_REPORT("Calling cudaLaunchDeviceV2");

    unsigned n_return = target_func->has_return();
    assert(n_return);
    unsigned n_args = target_func->num_args();
    assert( n_args == 2 );

    kernel_info_t * child_grid = NULL;
    function_info * child_kernel_entry = NULL;
    void * parameter_buffer;
    struct CUstream_st * child_stream;
    for( unsigned arg=0; arg < n_args; arg ++ ) {
        const operand_info &actual_param_op = pI->operand_lookup(n_return+1+arg); //param#
        const symbol *formal_param = target_func->get_arg(arg); //cudaLaunchDeviceV2_param_#
        unsigned size=formal_param->get_size_in_bytes();
        assert( formal_param->is_param_local() );
        assert( actual_param_op.is_param_local() );
        addr_t from_addr = actual_param_op.get_symbol()->get_address();

        if(arg == 0) {//paramter buffer for child kernel (in global memory)
            //get parameter_buffer from the cudaDeviceLaunchV2_param0
			assert(size == sizeof(void *));
            thread->m_local_mem->read(from_addr, size, &parameter_buffer);
            assert((size_t)parameter_buffer >= GLOBAL_HEAP_START);
            DEV_RUNTIME_REPORT("Parameter buffer locating at global memory " << parameter_buffer);

            //get child grid info through parameter_buffer address
            assert(g_cuda_device_launch_map.find(parameter_buffer) != g_cuda_device_launch_map.end());
            child_grid = g_cuda_device_launch_map[parameter_buffer];
            child_kernel_entry = child_grid->entry();
            DEV_RUNTIME_REPORT("find child kernel " << child_kernel_entry->get_name());

            //copy data in parameter_buffer to child kernel param memory
	        unsigned child_kernel_arg_size = child_kernel_entry->get_args_aligned_size();
            DEV_RUNTIME_REPORT("child_kernel_arg_size " << child_kernel_arg_size);
            memory_space *child_kernel_param_mem = child_grid->get_param_memory();
            size_t param_start_address = 0;
            for(unsigned n = 0; n < child_kernel_arg_size; n++) {
                unsigned char one_byte;
                thread->get_gpu()->get_global_memory()->read((size_t)parameter_buffer + n, 1, &one_byte);
                child_kernel_param_mem->write(param_start_address + n, 1, &one_byte, NULL, NULL); 
            }
        }
        else if(arg == 1) { //cudaStream for the child kernel
			assert(size == sizeof(cudaStream_t));
            thread->m_local_mem->read(from_addr, size, &child_stream);
            if(child_stream == 0) { //default stream on device
                if(!g_device_default_stream) {
                    //g_device_default_stream = new struct CUstream_st();
                    //g_stream_manager->add_stream(g_device_default_stream);
                }
                child_stream = g_device_default_stream;
            }
//            DEV_RUNTIME_REPORT("launching child kernel to stream " << child_stream->get_uid());
        }
        
    }

    //launch child kernel
    stream_operation op(child_grid, g_ptx_sim_mode, child_stream);
//    g_stream_manager->push(op);
//    g_cuda_device_launch_map.erase(parameter_buffer);

    //set retval0
    const operand_info &actual_return_op = pI->operand_lookup(0); //retval0
    const symbol *formal_return = target_func->get_return_var(); //cudaError_t
	unsigned int return_size = formal_return->get_size_in_bytes();
	DEV_RUNTIME_REPORT("cudaLaunchDeviceV2 return value has size of " << return_size);
	assert(actual_return_op.is_param_local());
	assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size 
        && return_size == sizeof(cudaError_t));
    cudaError_t error = cudaSuccess;
    addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
	thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);

}
