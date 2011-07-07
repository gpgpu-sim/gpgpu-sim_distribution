# Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
# George L. Yuan, Ivan Sham and the 
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
# including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4

# GPGPU-Sim Makefile

DEBUG?=0

CXXFLAGS = -Wall -DDEBUG

GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')
ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS += 
endif

OPTFLAGS += -g3 -fPIC

CPP = g++ $(SNOW)
OEXT = o

OUTPUT_DIR=../$(SIM_OBJ_FILES_DIR)
SRCS = $(shell ls *.cc)
OBJS = $(SRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))

$(OUTPUT_DIR)/libgpgpusim.a:	$(OBJS) gpu_uarch_simlib
	ar rcs  $(OUTPUT_DIR)/libgpgpusim.a $(OBJS) $(OUTPUT_DIR)/gpgpu-sim/*.o

gpu_uarch_simlib:
	make   -C ./gpgpu-sim
	

depend:
	makedepend -p$(OUTPUT_DIR)/ $(SRCS) 2> /dev/null

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $(OUTPUT_DIR)/$*.$(OEXT) -c $*.cc

clean:
	rm -f *.o core *~ *.a 

option_parser.$(OEXT): option_parser.h

# DO NOT DELETE

../build/debug/abstract_hardware_model.o: abstract_hardware_model.h
../build/debug/abstract_hardware_model.o: cuda-sim/memory.h
../build/debug/abstract_hardware_model.o: abstract_hardware_model.h
../build/debug/abstract_hardware_model.o: /usr/include/assert.h
../build/debug/abstract_hardware_model.o: /usr/include/features.h
../build/debug/abstract_hardware_model.o: /usr/include/sys/cdefs.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/wordsize.h
../build/debug/abstract_hardware_model.o: /usr/include/gnu/stubs.h
../build/debug/abstract_hardware_model.o: /usr/include/gnu/stubs-64.h
../build/debug/abstract_hardware_model.o: /usr/include/string.h
../build/debug/abstract_hardware_model.o: /usr/include/stdio.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/types.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/typesizes.h
../build/debug/abstract_hardware_model.o: /usr/include/libio.h
../build/debug/abstract_hardware_model.o: /usr/include/_G_config.h
../build/debug/abstract_hardware_model.o: /usr/include/wchar.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/stdio_lim.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/sys_errlist.h
../build/debug/abstract_hardware_model.o: /usr/include/stdlib.h
../build/debug/abstract_hardware_model.o: /usr/include/sys/types.h
../build/debug/abstract_hardware_model.o: /usr/include/time.h
../build/debug/abstract_hardware_model.o: /usr/include/endian.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/endian.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/byteswap.h
../build/debug/abstract_hardware_model.o: /usr/include/sys/select.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/select.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/sigset.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/time.h
../build/debug/abstract_hardware_model.o: /usr/include/sys/sysmacros.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/pthreadtypes.h
../build/debug/abstract_hardware_model.o: /usr/include/alloca.h
../build/debug/abstract_hardware_model.o: option_parser.h cuda-sim/ptx_ir.h
../build/debug/abstract_hardware_model.o: cuda-sim/ptx.tab.h
../build/debug/abstract_hardware_model.o: cuda-sim/ptx_sim.h
../build/debug/abstract_hardware_model.o: cuda-sim/dram_callback.h
../build/debug/abstract_hardware_model.o: tr1_hash_map.h cuda-sim/opcodes.h
../build/debug/abstract_hardware_model.o: cuda-sim/opcodes.def
../build/debug/abstract_hardware_model.o: cuda-sim/memory.h option_parser.h
../build/debug/abstract_hardware_model.o: cuda-sim/ptx-stats.h
../build/debug/abstract_hardware_model.o: /usr/include/math.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/huge_val.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/mathdef.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/mathcalls.h
../build/debug/debug.o: debug.h abstract_hardware_model.h gpgpu-sim/shader.h
../build/debug/debug.o: /usr/include/stdio.h /usr/include/features.h
../build/debug/debug.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
../build/debug/debug.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
../build/debug/debug.o: /usr/include/bits/types.h
../build/debug/debug.o: /usr/include/bits/typesizes.h /usr/include/libio.h
../build/debug/debug.o: /usr/include/_G_config.h /usr/include/wchar.h
../build/debug/debug.o: /usr/include/bits/stdio_lim.h
../build/debug/debug.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
../build/debug/debug.o: /usr/include/sys/types.h /usr/include/time.h
../build/debug/debug.o: /usr/include/endian.h /usr/include/bits/endian.h
../build/debug/debug.o: /usr/include/bits/byteswap.h
../build/debug/debug.o: /usr/include/sys/select.h /usr/include/bits/select.h
../build/debug/debug.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
../build/debug/debug.o: /usr/include/sys/sysmacros.h
../build/debug/debug.o: /usr/include/bits/pthreadtypes.h
../build/debug/debug.o: /usr/include/alloca.h /usr/include/math.h
../build/debug/debug.o: /usr/include/bits/huge_val.h
../build/debug/debug.o: /usr/include/bits/mathdef.h
../build/debug/debug.o: /usr/include/bits/mathcalls.h /usr/include/assert.h
../build/debug/debug.o: cuda-sim/ptx.tab.h cuda-sim/dram_callback.h
../build/debug/debug.o: gpgpu-sim/delayqueue.h intersim/statwraper.h
../build/debug/debug.o: gpgpu-sim/gpu-misc.h gpgpu-sim/stack.h
../build/debug/debug.o: abstract_hardware_model.h gpgpu-sim/dram.h
../build/debug/debug.o: /usr/include/zlib.h /usr/include/zconf.h
../build/debug/debug.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
../build/debug/debug.o: /usr/include/bits/confname.h /usr/include/getopt.h
../build/debug/debug.o: gpgpu-sim/scoreboard.h gpgpu-sim/mem_fetch.h
../build/debug/debug.o: gpgpu-sim/addrdec.h option_parser.h gpgpu-sim/stats.h
../build/debug/debug.o: gpgpu-sim/gpu-cache.h tr1_hash_map.h
../build/debug/debug.o: gpgpu-sim/gpu-sim.h gpgpu-sim/shader.h
../build/debug/debug.o: cuda-sim/ptx_sim.h cuda-sim/dram_callback.h
../build/debug/debug.o: cuda-sim/opcodes.h cuda-sim/opcodes.def
../build/debug/debug.o: cuda-sim/memory.h cuda-sim/cuda-sim.h
../build/debug/debug.o: cuda-sim/ptx_ir.h cuda-sim/ptx.tab.h
../build/debug/debug.o: cuda-sim/ptx_sim.h /usr/include/string.h
../build/debug/gpgpusim_entrypoint.o: gpgpusim_entrypoint.h
../build/debug/gpgpusim_entrypoint.o: abstract_hardware_model.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/time.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/types.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/features.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/sys/cdefs.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/wordsize.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/gnu/stubs.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/gnu/stubs-64.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/typesizes.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/stdio.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/libio.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/_G_config.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/wchar.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/stdio_lim.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/sys_errlist.h
../build/debug/gpgpusim_entrypoint.o: option_parser.h /usr/include/stdlib.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/sys/types.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/endian.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/endian.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/byteswap.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/sys/select.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/select.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/sigset.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/time.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/sys/sysmacros.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/pthreadtypes.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/alloca.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/cuda-sim.h
../build/debug/gpgpusim_entrypoint.o: abstract_hardware_model.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/dram_callback.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/ptx_ir.h /usr/include/assert.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/ptx.tab.h cuda-sim/ptx_sim.h
../build/debug/gpgpusim_entrypoint.o: tr1_hash_map.h cuda-sim/opcodes.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/opcodes.def cuda-sim/memory.h
../build/debug/gpgpusim_entrypoint.o: option_parser.h cuda-sim/ptx_parser.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/gpu-sim.h gpgpu-sim/addrdec.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/shader.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/icnt_wrapper.h
../build/debug/gpgpusim_entrypoint.o: stream_manager.h /usr/include/pthread.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/sched.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/sched.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/signal.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/setjmp.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/semaphore.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/semaphore.h
../build/debug/option_parser.o: /usr/include/stdio.h /usr/include/features.h
../build/debug/option_parser.o: /usr/include/sys/cdefs.h
../build/debug/option_parser.o: /usr/include/bits/wordsize.h
../build/debug/option_parser.o: /usr/include/gnu/stubs.h
../build/debug/option_parser.o: /usr/include/gnu/stubs-64.h
../build/debug/option_parser.o: /usr/include/bits/types.h
../build/debug/option_parser.o: /usr/include/bits/typesizes.h
../build/debug/option_parser.o: /usr/include/libio.h /usr/include/_G_config.h
../build/debug/option_parser.o: /usr/include/wchar.h
../build/debug/option_parser.o: /usr/include/bits/stdio_lim.h
../build/debug/option_parser.o: /usr/include/bits/sys_errlist.h
../build/debug/option_parser.o: /usr/include/stdlib.h
../build/debug/option_parser.o: /usr/include/sys/types.h /usr/include/time.h
../build/debug/option_parser.o: /usr/include/endian.h
../build/debug/option_parser.o: /usr/include/bits/endian.h
../build/debug/option_parser.o: /usr/include/bits/byteswap.h
../build/debug/option_parser.o: /usr/include/sys/select.h
../build/debug/option_parser.o: /usr/include/bits/select.h
../build/debug/option_parser.o: /usr/include/bits/sigset.h
../build/debug/option_parser.o: /usr/include/bits/time.h
../build/debug/option_parser.o: /usr/include/sys/sysmacros.h
../build/debug/option_parser.o: /usr/include/bits/pthreadtypes.h
../build/debug/option_parser.o: /usr/include/alloca.h /usr/include/assert.h
../build/debug/option_parser.o: /usr/include/string.h option_parser.h
../build/debug/stream_manager.o: stream_manager.h abstract_hardware_model.h
../build/debug/stream_manager.o: /usr/include/pthread.h
../build/debug/stream_manager.o: /usr/include/features.h
../build/debug/stream_manager.o: /usr/include/sys/cdefs.h
../build/debug/stream_manager.o: /usr/include/bits/wordsize.h
../build/debug/stream_manager.o: /usr/include/gnu/stubs.h
../build/debug/stream_manager.o: /usr/include/gnu/stubs-64.h
../build/debug/stream_manager.o: /usr/include/endian.h
../build/debug/stream_manager.o: /usr/include/bits/endian.h
../build/debug/stream_manager.o: /usr/include/bits/byteswap.h
../build/debug/stream_manager.o: /usr/include/sched.h
../build/debug/stream_manager.o: /usr/include/bits/types.h
../build/debug/stream_manager.o: /usr/include/bits/typesizes.h
../build/debug/stream_manager.o: /usr/include/time.h
../build/debug/stream_manager.o: /usr/include/bits/sched.h
../build/debug/stream_manager.o: /usr/include/signal.h
../build/debug/stream_manager.o: /usr/include/bits/sigset.h
../build/debug/stream_manager.o: /usr/include/bits/pthreadtypes.h
../build/debug/stream_manager.o: /usr/include/bits/setjmp.h
../build/debug/stream_manager.o: gpgpusim_entrypoint.h cuda-sim/cuda-sim.h
../build/debug/stream_manager.o: abstract_hardware_model.h
../build/debug/stream_manager.o: cuda-sim/dram_callback.h
../build/debug/stream_manager.o: /usr/include/stdlib.h
../build/debug/stream_manager.o: /usr/include/sys/types.h
../build/debug/stream_manager.o: /usr/include/sys/select.h
../build/debug/stream_manager.o: /usr/include/bits/select.h
../build/debug/stream_manager.o: /usr/include/bits/time.h
../build/debug/stream_manager.o: /usr/include/sys/sysmacros.h
../build/debug/stream_manager.o: /usr/include/alloca.h gpgpu-sim/gpu-sim.h
../build/debug/stream_manager.o: gpgpu-sim/addrdec.h /usr/include/stdio.h
../build/debug/stream_manager.o: /usr/include/libio.h
../build/debug/stream_manager.o: /usr/include/_G_config.h
../build/debug/stream_manager.o: /usr/include/wchar.h
../build/debug/stream_manager.o: /usr/include/bits/stdio_lim.h
../build/debug/stream_manager.o: /usr/include/bits/sys_errlist.h
../build/debug/stream_manager.o: /usr/include/assert.h option_parser.h
../build/debug/stream_manager.o: gpgpu-sim/shader.h
