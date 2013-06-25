# Copyright (c) 2009 by Tor M. Aamodt and the 
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
# (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
# benchmarks/template/ are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
# src/intersim/ are derived from Booksim (a simulator provided with the 
# textbook "Principles and Practices of Interconnection Networks" available 
# from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# which is distributed seperately by NVIDIA under separate terms and 
# conditions.  To use this version of GPGPU-Sim with OpenCL requires a
# recent version of NVIDIA's drivers which support OpenCL.
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
# including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4


include ../version_detection.mk

ifeq ($(OPENGL_SUPPORT),1) 
	GL = -DOPENGL_SUPPORT
endif

CPP = g++
CC = gcc
CREATELIBRARY    =  1
DEBUG ?= 0
CCFLAGS = -O3 -g -Wall -fPIC $(GL)
ifeq ($(DEBUG),1)
	CCFLAGS =  -Wall -g -fPIC $(GL)
endif

ifeq ($(GNUC_CPP0X), 1)
    CCFLAGS += -std=c++0x
endif


PROG     =opencl

CXX_SRCS =  opencl_runtime_api.cc
CCFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

.PHONY: clean

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/libopencl

OBJS = $(CXX_SRCS:%.cc=$(OUTPUT_DIR)/%.o)

#--- Make rules ---

all: lib$(PROG).a $(OUTPUT_DIR)/nvopencl_wrapper

lib$(PROG).a: $(OBJS)
	ar rcs $(OUTPUT_DIR)/lib$(PROG).a $(OBJS)

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(CXX_SRCS) 2> /dev/null

$(OUTPUT_DIR)/nvopencl_wrapper: nvopencl_wrapper.cc
	if [ ! -d bin ]; then mkdir bin; fi
	g++ $(CCFLAGS) nvopencl_wrapper.cc -I./ -I$(NVOPENCL_INCDIR)/ -L $(NVOPENCL_LIBDIR) -lOpenCL  -o $(OUTPUT_DIR)/bin/nvopencl_wrapper
	

$(OUTPUT_DIR)/%.o: %.cc
	$(CPP) $(CCFLAGS) -I./ -I$(CUDA_INSTALL_PATH)/include  -c $< -o $@

clean:
	rm -f $(PROG)
	rm -f *.o
	rm -f lib$(PROG).a
	rm -f bin/nvopencl_wrapper
	rm -f $(OUTPUT_DIR)/Makefile.makedepend $(OUTPUT_DIR)/Makefile.makedepend.bak

include $(OUTPUT_DIR)/Makefile.makedepend

