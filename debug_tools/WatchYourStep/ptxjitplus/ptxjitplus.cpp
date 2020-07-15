/** Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * This sample uses the Driver API to just-in-time compile (JIT) a Kernel from PTX code.
 * Additionally, this sample demonstrates the seamless interoperability capability of CUDA runtime
 * Runtime and CUDA Driver API calls.
 * This sample requires Compute Capability 2.0 and higher.
 *
 */

/**
 * Modified by: Jonathan Lew
 * PTX JIT PLUS
 * 
 * **********
 * User Guide
 * **********
 *
 * Welcome to WatchYourStep, a debugging tool that allows you launch individual
 * kernels using parameters captured from cudaLaunch and outputs the values in 
 * the arrays from the kernel. It allows you to watch each step you program takes, 
 * kernel by kernel. 
 *
 * 1. Set environment variables to create params.config* and ptx.config* files. 
 *      a)export PTX_SIM_DEBUG=4
 *      b)export PTX_JIT_PATH=[path to this file]
 *      c)export WYS_EXEC_PATH=[path to executable (program to debug)]
 *      d)export WYS_EXEC_NAME=[name of executable (program to debug)]
 *      e)Make sure all GPGPU-Sim path variables are set (see GPGPU-Sim documentation)
 * 2. Run executable (program to debug) using GPGPU-Sim 
 * 3. export PTX_SIM_DEBUG=[less than 4 to not dump config files again]
 * 4-1. Run one kernel at a time: export WYS_LAUNCH_NUM=[kernel to launch] and compile ptxjitplus and run ptxjitplus
 * 4-2. Run all kernels: compile and run ". launchkernels 0 [max number of kernels]" in terminal
 * 5. Find output in ../data/wys.out* where * is the launch number
 */

// System includes
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <map>

// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// sample include
#include "ptxjitplus.h"

const char *sSDKname = "PTX Just In Time (JIT) Compilation Plus";
char *wys_exec_path;
char *wys_exec_name;
char *wys_launch_num;
dim3 gridDim, blockDim;
std::string kernelName;

void ptxJIT(int argc, char **argv, CUmodule *phModule, CUfunction *phKernel, CUlinkState *lState)
{
    CUjit_option options[6];
    void *optionVals[6];
    float walltime;
    char error_log[8192],
         info_log[8192];
    unsigned int logSize = 8192;
    void *cuOut;
    size_t outSize;
    int myErr = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void *) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void *) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void *) (long)logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void *) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void *) (long) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void *) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6,options, optionVals, lState));

    if (sizeof(void *)==4)
    {
        // Load the PTX from the string myPtx32
        printf("Loading myPtx32[] program...\n");
        printf("WARNING: 32-bit execution is untested");
    }
    else
    {
        // Load the PTX from the string myPtx (64-bit)
        printf("Loading myPtx[] program\n");
    }
    std::string ptx_file (std::string("../data/ptx.config") + wys_launch_num);
    myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, ptx_file.c_str(),0,0,0);

    if (myErr != CUDA_SUCCESS)
    {
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
    }

    // Complete the linker step
    checkCudaErrors(cuLinkComplete(*lState, &cuOut, &outSize));

    // Linker walltime and info_log were requested in options above.
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n",walltime,info_log);

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(phModule, cuOut));

    // Locate the kernel entry poin
    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, kernelName.c_str()));

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(*lState));
}

void initializeData(std::vector<param>& v_params)
{
    wys_exec_path = getenv("WYS_EXEC_PATH");
    assert(wys_exec_path!=NULL);
    wys_exec_name = getenv("WYS_EXEC_NAME");
    assert(wys_exec_name!=NULL);
    std::string path_to_search = std::string(wys_exec_path) + "/" + wys_exec_name + ".*.ptx";
    wys_launch_num = getenv("WYS_LAUNCH_NUM");
    assert(wys_launch_num!=NULL);
    std::string filename = std::string("../data/params.config") + wys_launch_num;

    FILE *fin = fopen(filename.c_str(), "r");
    assert(fin);
    char buff[1024];
    fscanf(fin, "%s\n", buff);
    printf("Processing :%s ...\n", buff);
    fflush(stdout);
    kernelName = std::string(buff);
    fscanf(fin, "%u,%u,%u %u,%u,%u\n", &gridDim.x, &gridDim.y, &gridDim.z, &blockDim.x, &blockDim.y, &blockDim.z);
    //fill data structure to pass in params later
    while (!feof(fin)){
        param p;
        int err;
        size_t len;
        unsigned val;
        int start = fgetc(fin);
        if (start == '*'){
            p.isPointer = true;
        }else{
            p.isPointer = false;
            int c = ungetc(start,fin);
            assert(c==start&&"Couldn't ungetc\n");
        }
        err = fscanf(fin, "%lu : ", &len);
        assert( err==1 );
        p.size = len;
        unsigned char* params = (unsigned char*) malloc(len*sizeof(unsigned char));
        for (size_t i=0; i<len; i++)
        {
            err = fscanf(fin, "%u ", &val);
            assert( err==1 );
            params[i] = (unsigned char) val;
        }
        p.data = params;
        unsigned offset;
        err = fscanf(fin, " : %u", &offset);
        assert( err==1 );
        p.offset = offset;
        v_params.push_back(p);
        err = fscanf(fin, "\n");
        assert(err==0);
    }
    fclose(fin);
}

int main(int argc, char **argv)
{
    const unsigned int nThreads = 2;
    const unsigned int nBlocks  = 2;

    CUmodule     hModule  = 0;
    CUfunction   hKernel  = 0;
    CUlinkState  lState;

    int cuda_device = 0;
    cudaDeviceProp deviceProp;

    printf("[%s] - Starting...\n", sSDKname);
    //parameter data
    std::vector<param> v_params;
    initializeData(v_params);

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        cuda_device = getCmdLineArgumentInt(argc, (const char **)argv, "device=");

        if (cuda_device < 0)
        {
            printf("Invalid command line parameters\n");
            exit(EXIT_FAILURE);
        }
        else
        {
            printf("cuda_device = %d\n", cuda_device);
            cuda_device = gpuDeviceInit(cuda_device);

            if (cuda_device < 0)
            {
                printf("No CUDA Capable devices found, exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with the highest Gflops/s
        cuda_device = gpuGetMaxGflopsDeviceId();
    }

    checkCudaErrors(cudaSetDevice(cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
    printf("> Using CUDA device [%d]: %s\n", cuda_device, deviceProp.name);

    if (deviceProp.major < 2)
    {
        fprintf(stderr, "Compute Capability 2.0 or greater required for this sample.\n");
        fprintf(stderr, "Maximum Compute Capability of device[%d] is %d.%d.\n", cuda_device,deviceProp.major,deviceProp.minor);
        exit(EXIT_WAIVED);
    }

    // Allocate memory on host and device (Runtime API)
    // NOTE: The runtime API will create the GPU Context implicitly here
    int         *d_tmp   = 0;
    checkCudaErrors(cudaMalloc(&d_tmp, 1));

    // JIT Compile the Kernel from PTX and get the Handles (Driver API)
    ptxJIT(argc, argv, &hModule, &hKernel, &lState);
    checkCudaErrors(cudaFree(d_tmp));

    //maps param number to pointer to device data
    std::map< size_t, void* > m_device_data;
    std::map< size_t, void* > m_cleanup;
    void * paramKernels[v_params.size()];
    //Initialize param data for kernel
    int paramOffset = 0;

    int index = 0;
    for(std::vector<param>::iterator p = v_params.begin(); p!=v_params.end(); p++){
        if(p->isPointer){
            unsigned char **d_data = (unsigned char **) malloc(sizeof(unsigned char **));
            checkCudaErrors(cudaMalloc((void**)d_data, p->size));
            checkCudaErrors(cudaMemcpy((void*)*d_data,(void*)p->data,p->size,cudaMemcpyHostToDevice));
            paramKernels[index] = (void*)d_data;
            m_device_data[index]=*d_data;
            m_cleanup[index]=d_data;
            paramOffset = p->offset + 8;
        }else{
            paramKernels[index] = (void*)p->data;
            paramOffset = p->offset + p->size;
        }
        index ++;
    }

    // Launch the kernel (Driver API_)
    CUDAAPI cuLaunchKernel(hKernel, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 
        0, NULL, paramKernels, NULL);
    std::cout << "CUDA kernel launched" << std::endl;

    //maps param number to pointer to output data
    std::map< size_t, unsigned char* > m_output_data;
    for(std::map< size_t, void* >::iterator i = m_device_data.begin(); i!=m_device_data.end(); i++){
        unsigned char *h_data   = 0;
        if ((h_data = (unsigned char *)malloc(v_params[i->first].size)) == NULL)
        {
            std::cerr << "Could not allocate host memory" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Copy the result back to the host
        checkCudaErrors(cudaMemcpy(h_data, i->second, v_params[i->first].size, cudaMemcpyDeviceToHost));
        m_output_data[i->first] = h_data;
    }

    std::string filename = std::string("../data/wys.out") + wys_launch_num;
    FILE *fout = fopen(filename.c_str(), "w");
    assert(fout);
    for(std::map< size_t, unsigned char* >::iterator i = m_output_data.begin(); i!=m_output_data.end(); i++){
        fprintf(fout, "param %zu: size = %zu, data = ", i->first, v_params[i->first].size);
        for (size_t j = 0; j<v_params[i->first].size; j++){
            if (!(j%24)){ 
                fprintf(fout, "\n");
            }
            fprintf(fout, " %u", i->second[j]);
        }
        fprintf(fout, "\n");
    }
    fflush(fout);
    fclose(fout);

    //Cleanup
    for(std::map< size_t, void* >::iterator i = m_device_data.begin(); i!=m_device_data.end(); i++){
        if (i->second){
            checkCudaErrors(cudaFree(i->second));
            i->second = 0;
        }
    }

    for(std::map< size_t, void* >::iterator i = m_cleanup.begin(); i!=m_cleanup.end(); i++){
        if (i->second){
            free(i->second);
            i->second = 0;
        }
    }

    for(std::map< size_t, unsigned char* >::iterator i = m_output_data.begin(); i!=m_output_data.begin(); i++){
        if (i->second)
        {
            free(i->second);
            i->second = 0;
        }
    }

    if (hModule)
    {
        checkCudaErrors(cuModuleUnload(hModule));
        hModule = 0;
    }

    return 0;
}
