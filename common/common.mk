# Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
# George L. Yuan, Jimmy Kwa and the 
# University of British Columbia
# Vancouver, BC  V6T 1Z4
# All Rights Reserved.
# 
# THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
# TERMS AND CONDITIONS.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 
# NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
# are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
# (property of NVIDIA).  The files benchmarks/BlackScholes/* and 
# benchmarks/template/* are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files 
# src/gpgpusim_entrypoint.c and src/simplesim-3.0/* are derived from the 
# SimpleScalar Toolset available from http://www.simplescalar.com/ 
# (property of SimpleScalar LLC) and the files src/intersim/* are derived 
# from Booksim (Simulator provided with the textbook "Principles and 
# Practices of Interconnection Networks" available from 
# http://cva.stanford.edu/books/ppin/).  As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# version 1.1, which is distributed seperately by NVIDIA under separate terms 
# and conditions.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the University of British Columbia nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
#  
# 5. No nonprofit user may place any restrictions on the use of this software,
#  including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4

CPP = g++ $(SNOW)
CC = gcc $(SNOW)

NVCC_VERSION:=$(shell nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
INCFLAGEXTRA ?=
CFLAGS     ?= 
CCFLAGS    ?=
CUFLAGS    ?=
NEWLIBDIR  ?=
LIB        ?=
SRCDIR     ?=
ROOTDIR    ?= 
ROOTBINDIR ?= bin
BINDIR     ?= $(ROOTBINDIR)
ROOTOBJDIR ?= obj
ifneq ($(NVCC_VERSION),2.3)
LIBDIR     := "$(NVIDIA_CUDA_SDK_LOCATION)/lib"
COMMONDIR  := "$(NVIDIA_CUDA_SDK_LOCATION)/common"
SDKINCDIR  := "$(NVIDIA_CUDA_SDK_LOCATION)/common/inc/"
else
LIBDIR     := "$(NVIDIA_CUDA_SDK_LOCATION)/C/lib"
COMMONDIR  := "$(NVIDIA_CUDA_SDK_LOCATION)/C/common"
SDKINCDIR  := "$(NVIDIA_CUDA_SDK_LOCATION)/C/common/inc/"
endif
LIB += -lm -lz
ifeq ($(shell uname),Linux)
	CUTIL:=cutil
	LIB += -lGL
else
	CUTIL:=cutil_i386
endif
GPGPUSIM_ROOT ?= ../..
INTERMED_FILES := *.cpp*.i *.cpp*.ii *.cu.c *.cudafe*.* *.fatbin.c *.cu.cpp *.linkinfo *.cpp_o core

SIM_OBJDIR := 
SIM_OBJS +=  $(patsubst %.cpp,$(SIM_OBJDIR)%.cpp_o,$(CCFILES))
SIM_OBJS +=  $(patsubst %.c,$(SIM_OBJDIR)%.c_o,$(CFILES))
SIM_OBJS +=  $(patsubst %.cu,$(SIM_OBJDIR)%.cu_o,$(CUFILES))

.SUFFIXES:

gpgpu_ptx_sim__$(EXECUTABLE): $(SIM_OBJS) $(GPGPUSIM_ROOT)/src/cuda-sim/libgpgpu_ptx_sim.a $(GPGPUSIM_ROOT)/src/libgpgpusim.a $(GPGPUSIM_ROOT)/src/intersim/libintersim.a 
	$(CPP) $(CFLAGS) -g $(notdir $(SIM_OBJS)) -L$(GPGPUSIM_ROOT)/libcuda/ -lcuda \
		-L$(LIBDIR) -l$(CUTIL) \
		-L$(GPGPUSIM_ROOT)/src/ -lgpgpusim \
		-L$(GPGPUSIM_ROOT)/src/intersim -lintersim \
		-L$(GPGPUSIM_ROOT)/src/cuda-sim/ -lgpgpu_ptx_sim \
		$(NEWLIBDIR) $(LIB) -o gpgpu_ptx_sim__$(EXECUTABLE)
	rm -rf $(INTERMED_FILES) *.cubin cubin.bin *_o *.hash $(EXECUTABLE)

%.cpp_o: %.cpp
	$(CPP) $(CCFLAGS) $(INCFLAGEXTRA) -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@)

%.c_o: %.c
	$(CC) $(CFLAGS) $(INCFLAGEXTRA)  -I$(CUDAHOME)/include -I$(SDKINCDIR) -L$(LIBDIR) -g -c $< -o $(notdir $@)

%.cu_o: %.cu
	nvcc $(CUFLAGS) -c -arch sm_11 --keep --compiler-options -fno-strict-aliasing \
		-I. -I$(CUDAHOME)/include/ -I$(SDKINCDIR) \
		 $(INCFLAGEXTRA) -L$(LIBDIR) -lcutil -DUNIX $< -o $(EXECUTABLE) 
ifneq ($(NVCC_VERSION),1.1)
	$(CC) -g -c $(notdir $<.cpp) -o $(notdir $@)
else
	$(CC) -g -c $(notdir $<.c) -o $(notdir $@)
endif
	$(GPGPUSIM_ROOT)/scripts/gen_ptxinfo

%.cu: %.cu.c

clean:
	rm -f $(INTERMED_FILES) *.cubin *.o *_o *.hash *.ptx *.ptxinfo cubin.bin $(EXECUTABLE) gpgpu_ptx_sim__$(EXECUTABLE)

