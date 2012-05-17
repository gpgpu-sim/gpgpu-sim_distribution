# Copyright (c) 2009 by Tor M. Aamodt, Ali Bakhoda and the 
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


CUDART_VERSION:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//' | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

ifeq ($(OPENGL_SUPPORT),1) 
	GL = -DOPENGL_SUPPORT
endif

GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')
ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS = -std=c++0x
endif


CPP = g++ $(SNOW)
CC = gcc $(SNOW)
CREATELIBRARY    =  1
DEBUG ?= 0
ifeq ($(DEBUG),1)
	CXXFLAGS +=  -Wall -Wno-unused-function -Wno-sign-compare -g -fPIC $(GL)
	CCFLAGS += -Wall -Wno-unused-function -Wno-sign-compare -ggdb -fPIC
else
	CXXFLAGS += -O3 -g -Wall -Wno-unused-function -Wno-sign-compare -fPIC $(GL)
	CCFLAGS += -Wall -Wno-unused-function -Wno-sign-compare -fPIC
endif

PROG     =cuda

CXX_SRCS =  cuda_runtime_api.cc
LEX_SRCS = cuobjdump.l
YACC_SRCS = cuobjdump.y
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)
LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d -v --report=all

.PHONY: clean

OUTPUT_DIR=../$(SIM_OBJ_FILES_DIR)/libcuda

OBJS = $(CXX_SRCS:%.cc=$(OUTPUT_DIR)/%.o)
OBJS += $(LEX_SRCS:%.l=$(OUTPUT_DIR)/%_lexer.o)
OBJS += $(YACC_SRCS:%.y=$(OUTPUT_DIR)/%_parser.o)

#--- Make rules ---
lib$(PROG).a: $(OBJS)
	echo $(OBJS)
	ar rcs $(OUTPUT_DIR)/lib$(PROG).a $(OBJS)

$(OUTPUT_DIR)/%.o: %.cc
	$(CPP) $(CXXFLAGS) -I./ -I$(CUDA_INSTALL_PATH)/include  -c $< -o $@

$(OUTPUT_DIR)/%.o: %.c
	$(CC) $(CCFLAGS) -I./ -I$(CUDA_INSTALL_PATH)/include  -c $< -o $@

cuobjdump_parser.c: cuobjdump.y
	$(YACC) $(YFLAGS) -p cuobjdump_ -o$@ $<

cuobjdump_lexer.c: cuobjdump.l cuobjdump_parser.h
	$(LEX) $(LEXFLAGS) -P cuobjdump_ -o$@ $<
	
# The next rule means just get parser.c and you will get parser.h with it
# in other words, get parser.c and do nothing to get parser.h
%_parser.h: %_parser.c
	:

depend:
	makedepend -p$(OUTPUT_DIR)/ $(CXX_SRCS) 2> /dev/null

clean:
	rm -f $(PROG)
	rm -f *.o
	rm -f lib$(PROG).a
	rm -f *_parser.*
	rm -f *_lexer.*

# DO NOT DELETE

../build/release/libcuda/cuda_runtime_api.o: /usr/include/stdlib.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/features.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/sys/cdefs.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/wordsize.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/gnu/stubs.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/gnu/stubs-64.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/sys/types.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/types.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/typesizes.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/time.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/endian.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/endian.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/byteswap.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/sys/select.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/select.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/sigset.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/time.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/sys/sysmacros.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/pthreadtypes.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/alloca.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/stdio.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/libio.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/_G_config.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/wchar.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/stdio_lim.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/sys_errlist.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/string.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/xlocale.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/assert.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/gpu-sim.h
../build/release/libcuda/cuda_runtime_api.o: ../src/abstract_hardware_model.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/addrdec.h
../build/release/libcuda/cuda_runtime_api.o: ../src/option_parser.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/shader.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/math.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/huge_val.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/huge_valf.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/huge_vall.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/inf.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/nan.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/mathdef.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/mathcalls.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/ptx.tab.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/delayqueue.h
../build/release/libcuda/cuda_runtime_api.o: ../src/intersim/statwraper.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/gpu-misc.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/stack.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/dram.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/zlib.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/zconf.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/unistd.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/posix_opt.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/confname.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/getopt.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/scoreboard.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/mem_fetch.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/mem_fetch_status.tup
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/stats.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/gpu-cache.h
../build/release/libcuda/cuda_runtime_api.o: ../src/tr1_hash_map.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/ptx_loader.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/cuda-sim.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpu-sim/shader.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/ptx_sim.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/opcodes.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/opcodes.def
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/memory.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/ptx_ir.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/ptx.tab.h
../build/release/libcuda/cuda_runtime_api.o: ../src/cuda-sim/ptx_parser.h
../build/release/libcuda/cuda_runtime_api.o: ../src/gpgpusim_entrypoint.h
../build/release/libcuda/cuda_runtime_api.o: ../src/abstract_hardware_model.h
../build/release/libcuda/cuda_runtime_api.o: ../src/stream_manager.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/pthread.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/sched.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/sched.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/signal.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/setjmp.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/semaphore.h
../build/release/libcuda/cuda_runtime_api.o: /usr/include/bits/semaphore.h
