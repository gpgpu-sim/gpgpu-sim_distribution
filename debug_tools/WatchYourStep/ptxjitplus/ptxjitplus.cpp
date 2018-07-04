/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample uses the Driver API to just-in-time compile (JIT) a Kernel from PTX code.
 * Additionally, this sample demonstrates the seamless interoperability capability of CUDA runtime
 * Runtime and CUDA Driver API calls.
 * This sample requires Compute Capability 2.0 and higher.
 *
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
#include "ptxinst.h"

const char *sSDKname = "PTX Just In Time (JIT) Compilation (no-qatest)";
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
        printf("Loading myPtx32[] program\n");
        // PTX May also be loaded from file, as per below.
        myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)myPtx32, strlen(myPtx32)+1, 0, 0, 0, 0);
    }
    else
    {
        // Load the PTX from the string myPtx (64-bit)
        printf("Loading myPtx[] program\n");
        //myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)myPtx64, strlen(myPtx64)+1, 0, 0, 0, 0);
        // PTX May also be loaded from file, as per below.
        myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, "myPtx64.ptx",0,0,0);
    }

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

void initializeData(std::vector<unsigned char*>& param_data, std::vector< std::pair<size_t, bool> >& param_info)
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
        std::pair<size_t, bool> info;
        int err;
        size_t len;
        unsigned val;
        int start = fgetc(fin);
        if (start == '*'){
            info.second=true;
        }else{
            info.second=false;
            int c = ungetc(start,fin);
            assert(c==start&&"Couldn't ungetc\n");
        }
        err = fscanf(fin, "%lu : ", &len);
        info.first = len;
        assert( err==1 );
        //printf("%lu : ", len);
        unsigned char* params = (unsigned char*) malloc(len*sizeof(unsigned char));
        for (size_t i=0; i<len; i++)
        {
            err = fscanf(fin, "%u ", &val);
            assert( err==1 );
            //printf("%u ", val);
            params[i] = (unsigned char) val;
        }
        param_info.push_back(info);
        param_data.push_back(params);
        err = fscanf(fin, "\n");
        assert(err==0);
        //printf("\n");
    }
    fclose(fin);
    //filename = std::string("../data/wys.out") + wys_launch_num + "_param";
    //fout = fopen(filename.c_str(), "w");
    //assert(fout);
    //fprintf(fout, "param %zu: size = %zu, data = ", 0,param_info[0].first);
    //for (size_t j = 0; j<param_info[0].first; j++){
    //    fprintf(fout, " %u", i->second[j]);
    //}
    //fprintf(fout, "\n");
    //fflush(fout);
    //fclose(fout);
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
    std::vector<unsigned char*> param_data;
    //parameter data size and isPointer
    std::vector< std::pair<size_t, bool> > param_info;
    initializeData(param_data,param_info);



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
    std::map< size_t, unsigned char* > m_device_data;
    //Initialize param_data for kernel
    int paramOffset = 0;
    for( size_t i = 0; i<param_data.size(); i++){
        if(param_info[i].second){
            unsigned char *d_data = 0;
            checkCudaErrors(cudaMalloc((void**)&d_data, param_info[i].first));
            checkCudaErrors(cudaMemcpy((void*)d_data,(void*)param_data[i],param_info[i].first,cudaMemcpyHostToDevice));
            checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_data, sizeof(d_data)));
            m_device_data[i]=d_data;
            paramOffset += 8;
        }else{
            checkCudaErrors(cuParamSetv(hKernel, paramOffset, param_data[i], param_info[i].first));
            paramOffset += param_info[i].first;
        }
    }
    checkCudaErrors(cuParamSetSize(hKernel, paramOffset));

    // Launch the kernel (Driver API_)
    // TODO: automatically load these values in
    CUDAAPI cuLaunchKernel(hKernel, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 
        0, NULL, NULL, NULL);
    std::cout << "CUDA kernel launched" << std::endl;

    //maps param number to pointer to output data
    std::map< size_t, unsigned char* > m_output_data;
    for(std::map< size_t, unsigned char* >::iterator i = m_device_data.begin(); i!=m_device_data.end(); i++){
        unsigned char *h_data   = 0;
        if ((h_data = (unsigned char *)malloc(param_info[i->first].first)) == NULL)
        {
            std::cerr << "Could not allocate host memory" << std::endl;
            exit(EXIT_FAILURE);
        }
        // Copy the result back to the host
        checkCudaErrors(cudaMemcpy(h_data, i->second, param_info[i->first].first, cudaMemcpyDeviceToHost));
        m_output_data[i->first] = h_data;
    }

    std::string filename = std::string("../data/wys.out") + wys_launch_num;
    FILE *fout = fopen(filename.c_str(), "w");
    assert(fout);
    for(std::map< size_t, unsigned char* >::iterator i = m_output_data.begin(); i!=m_output_data.end(); i++){
        fprintf(fout, "param %zu: size = %zu, data = ", i->first,param_info[i->first].first);
        for (size_t j = 0; j<param_info[i->first].first; j++){
            fprintf(fout, " %u", i->second[j]);
            if (j&&(!(j%20))){
                fprintf(fout, "\n");
            }
        }
        fprintf(fout, "\n");
    }
    fflush(fout);
    fclose(fout);

//    int* h_data = (int*) m_output_data[0];
//    // Check the result
//    bool dataGood = true;
//
//    for (unsigned int i = 0 ; dataGood && i < nBlocks * nThreads ; i++)
//    {
//        if (h_data[i] != (int)i)
//        {
//            std::cerr << "Error at " << i << std::endl;
//            dataGood = false;
//        }
//    }
//    if(dataGood){
//        std::cout<<"OK!"<<std::endl;
//    }

    //Cleanup
    for(std::map< size_t, unsigned char* >::iterator i = m_device_data.begin(); i!=m_device_data.end(); i++){
        if (i->second){
            checkCudaErrors(cudaFree(i->second));
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

    //return dataGood ? EXIT_SUCCESS : EXIT_FAILURE;
}
