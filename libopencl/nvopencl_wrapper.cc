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

int main(int argc, const char **argv)
{
   cl_context context;
   cl_program pgm;
   cl_int errcode;

   FILE *fp = fopen(argv[1],"r");
   if ( fp == NULL ) exit(1);
   fseek(fp,0,SEEK_END);
   size_t source_length = ftell(fp);
   if ( source_length == 0 ) exit(2);
   char *source = (char*)calloc(source_length+1,1);
   fseek(fp,0,SEEK_SET);
   fread(source,1,source_length,fp);

   context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errcode);
   if ( errcode != CL_SUCCESS ) exit(3);
   pgm = clCreateProgramWithSource(context, 1, (const char **)&source, &source_length, &errcode);
   if ( errcode != CL_SUCCESS ) exit(4);

   char options[4096];
   unsigned n=0;
   options[0]=0;
   for ( int i=3; i < argc; i++ ) {
      snprintf(options+n,4096-n," %s ", argv[i] );
      n+= strlen(argv[i]);
      n+= 2;
   }
   errcode = clBuildProgram(pgm, 0, NULL, options, NULL, NULL);
   if ( errcode != CL_SUCCESS ) exit(5);

   size_t nbytes1=0;
   cl_uint num_devices;
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_NUM_DEVICES,sizeof(cl_uint),&num_devices,&nbytes1);
   if ( errcode != CL_SUCCESS ) exit(6);

   size_t nbytes2=0;
   size_t *binary_sizes = (size_t*)calloc(num_devices,sizeof(size_t));
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_BINARY_SIZES,sizeof(size_t)*num_devices,binary_sizes,&nbytes2);
   if ( errcode != CL_SUCCESS ) exit(7);

   unsigned char **binaries = (unsigned char**)calloc(num_devices,sizeof(unsigned char*));
   size_t bytes_to_read = 0;

   for (unsigned int i=0; i < num_devices; i++ ) {
      binaries[i] = (unsigned char*) calloc(binary_sizes[i],1);
      bytes_to_read += binary_sizes[i];
   }

   size_t nbytes3=0;
   errcode = clGetProgramInfo(pgm,CL_PROGRAM_BINARIES,bytes_to_read,binaries,&nbytes3);
   if ( errcode != CL_SUCCESS ) exit(8);

   fp = fopen(argv[2],"w");
   fprintf(fp,"%s",binaries[0]);
   fclose(fp);
   return 0;
}
