/* 
 * Copyright © 2009 by Tor M. Aamodt and the University of British Columbia, 
 * Vancouver, BC V6T 1Z4, All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define PREAMBLE "GPGPU-Sim nvopencl_wrapper"

void vmyexit(int code, const char *str,va_list ap)
{
   char buffer[1024];
   snprintf(buffer,1024,"%s: ERROR ** %s\n", PREAMBLE, str);
   vprintf(buffer,ap);
   fflush(stdout);
   if( code ) 
      exit(code);
}
void myexit(int code, const char *str, ... )
{
   va_list ap;
   va_start(ap,str);
   vmyexit(code,str,ap);
   va_end(ap);
}
int main(int argc, const char **argv)
{
   cl_context context;
   cl_program pgm;
   cl_int errcode;
   cl_uint num_devices;

   bool debug=false;

   printf("%s: command line = \'",PREAMBLE);
   for( int i=0; i < argc; i++ ) {
      printf("%s", argv[i]);
      if( (i+1) < argc ) printf(" ");
   }
   printf("'\n");

   if( !strncmp(argv[1],"-d",2) ) {
      printf("nvopencl_wrapper started\n");
      fflush(stdout);
      debug=true;
      argv = argv+1;
      argc--; 
   }

   FILE *fp = fopen(argv[1],"r");
   if ( fp == NULL ) myexit(1,"Could not open file \'%s\'",argv[1]);
   if ( debug ) { printf("opened \'%s\'\n", argv[1]); fflush(stdout); }
   fseek(fp,0,SEEK_END);
   size_t source_length = ftell(fp);
   if ( source_length == 0 ) myexit(2,"OpenCL file is empty");
   if ( debug ) { printf("file \'%s\' has length %zu bytes\n", argv[1], source_length); fflush(stdout); }
   char *source = (char*)calloc(source_length+1,1);
   if ( source == 0 ) myexit(2,"Memory allocation failed"); 
   fseek(fp,0,SEEK_SET);
   fread(source,1,source_length,fp);
   if ( debug ) { printf( "read in file \'%s\'\n", argv[1] ); fflush(stdout); }

   char buffer[1024];
   cl_uint num_platforms;
   cl_platform_id* platforms;

   errcode = clGetPlatformIDs(0, NULL, &num_platforms);
   if ( errcode != CL_SUCCESS ) myexit(1,"clGetPlatformaIDs returned %d",errcode);
   if ( num_platforms == 0 ) myexit(2,"No OpenCL platforms found");
   platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
   errcode = clGetPlatformIDs(num_platforms, platforms, NULL);
   if ( errcode != CL_SUCCESS ) myexit(3,"clGetPlatformIDs returned %d",errcode);
   unsigned use_platform = 0; 
   if (num_platforms > 1) {
      char platformName[1024]; 
      printf("%s: Multiple OpenCL platforms found.  Searching for compatible platform...\n",PREAMBLE);
      for (unsigned p = 0; p < num_platforms; p++) {
         errcode = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, &platformName, NULL);
         if ( errcode != CL_SUCCESS ) myexit(3,"clGetPlatformInfo returned %d",errcode);
         printf("%s:     OpenCL platform \'%s\'\n",PREAMBLE, platformName);
         if (strstr(platformName, "NVIDIA") != NULL) {
            use_platform = p; 
         }
      }
   }
   errcode = clGetPlatformInfo(platforms[use_platform], CL_PLATFORM_NAME, 1024, &buffer, NULL);
   if ( errcode != CL_SUCCESS ) myexit(3,"clGetPlatformInfo returned %d",errcode);
   printf("%s: Generating PTX using OpenCL platform \'%s\'\n",PREAMBLE,buffer);

   errcode = clGetDeviceIDs(platforms[use_platform], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
   if ( errcode != CL_SUCCESS ) myexit(4,"clGetDeviceIDs returned %d",errcode);
   printf("%s: found %u native OpenCL devices\n",PREAMBLE,num_devices);

   cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id) );
   errcode = clGetDeviceIDs(platforms[use_platform], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
   if ( errcode != CL_SUCCESS ) myexit(5,"clGetDeviceIDs returned %d",errcode);
   context = clCreateContext(0, num_devices, devices, NULL, NULL, &errcode);
   if ( errcode != CL_SUCCESS ) myexit(6,"clCreateContext returned %d",errcode);
   pgm = clCreateProgramWithSource(context, 1, (const char **)&source, &source_length, &errcode);
   if( errcode != CL_SUCCESS ) myexit(7,"clCreateProgramWithSource returned %d",errcode);

   char options[4096];
   unsigned n=0;
   options[0]=0;
   for ( int i=3; i < argc; i++ ) {
      snprintf(options+n,4096-n," %s ", argv[i] );
      n+= strlen(argv[i]);
      n+= 2;
   }
   errcode = clBuildProgram(pgm, 0, NULL, options, NULL, NULL);
   if ( errcode != CL_SUCCESS ) {
      printf("%s: clBuildProgram returned %d (error) -- build log:\n\n",PREAMBLE,errcode);
      size_t build_log_length=0;
      errcode = clGetProgramBuildInfo(pgm,devices[0],CL_PROGRAM_BUILD_LOG,0,NULL,&build_log_length);
      if( errcode != CL_SUCCESS ) myexit(8,"clGetProgramBuildInfo returned %d",errcode);
      char *build_log = (char*)calloc(1,build_log_length);
      errcode = clGetProgramBuildInfo(pgm,devices[0],CL_PROGRAM_BUILD_LOG,build_log_length,
                                      build_log,&build_log_length);
      printf("%s",build_log);
      printf("\n\n%s: end of build log\n", PREAMBLE);
      printf("%s: exiting early because the OpenCL code had errors (see above).\n", PREAMBLE);
      exit(8);
   }

   size_t nbytes1=0;
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_NUM_DEVICES,sizeof(cl_uint),&num_devices,&nbytes1);
   if ( errcode != CL_SUCCESS ) myexit(9,"clGetProgramInfo returned %d",errcode);

   size_t nbytes2=0;
   size_t *binary_sizes = (size_t*)calloc(num_devices,sizeof(size_t));
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_BINARY_SIZES,sizeof(size_t)*num_devices,binary_sizes,&nbytes2);
   if ( errcode != CL_SUCCESS ) myexit(10,"clGetProgramInfo returned %d",errcode);

   unsigned char **binaries = (unsigned char**)calloc(num_devices,sizeof(unsigned char*));
   size_t bytes_to_read = 0;

   for (unsigned int i=0; i < num_devices; i++ ) {
      binaries[i] = (unsigned char*) calloc(binary_sizes[i],1);
      bytes_to_read += binary_sizes[i];
   }

   size_t nbytes3=0;
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_BINARIES,bytes_to_read,binaries,&nbytes3);
   if ( errcode != CL_SUCCESS ) myexit(11,"clGetProgramInfo returned %d",errcode);

   fp = fopen(argv[2],"w");
   fprintf(fp,"%s",binaries[0]);
   fclose(fp);
   return 0;
}
