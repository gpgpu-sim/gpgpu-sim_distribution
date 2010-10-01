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

DEBUG=0

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

SRCS = $(shell ls *.cc)
OBJS = $(SRCS:.cc=.$(OEXT))

libgpgpusim.a:	$(OBJS) gpu_uarch_simlib
	ar rcs  libgpgpusim.a $(OBJS) gpgpu-sim/*.o

gpu_uarch_simlib:
	make   -C ./gpgpu-sim

depend:
	makedepend $(SRCS) 2> /dev/null

.cc.$(OEXT):
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $*.$(OEXT) -c $*.cc

clean:
	rm -f *.o core *~ *.a 

option_parser.$(OEXT): option_parser.h

# DO NOT DELETE

debug.o: debug.h abstract_hardware_model.h gpgpu-sim/shader.h
debug.o: /usr/include/stdio.h /usr/include/features.h
debug.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
debug.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
debug.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
debug.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
debug.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
debug.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
debug.o: /usr/include/endian.h /usr/include/bits/endian.h
debug.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
debug.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
debug.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
debug.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
debug.o: /usr/include/math.h /usr/include/bits/huge_val.h
debug.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
debug.o: /usr/include/assert.h cuda-sim/ptx.tab.h cuda-sim/dram_callback.h
debug.o: gpgpu-sim/gpu-cache.h abstract_hardware_model.h
debug.o: gpgpu-sim/delayqueue.h gpgpu-sim/stack.h gpgpu-sim/dram.h
debug.o: gpgpu-sim/scoreboard.h gpgpu-sim/gpu-sim.h cuda-sim/ptx_sim.h
debug.o: cuda-sim/dram_callback.h tr1_hash_map.h cuda-sim/opcodes.h
debug.o: cuda-sim/opcodes.def cuda-sim/memory.h /usr/include/string.h
debug.o: cuda-sim/cuda-sim.h cuda-sim/ptx_ir.h cuda-sim/ptx.tab.h
debug.o: cuda-sim/ptx_sim.h option_parser.h
gpgpusim_entrypoint.o: gpgpusim_entrypoint.h abstract_hardware_model.h
gpgpusim_entrypoint.o: /usr/include/time.h /usr/include/bits/types.h
gpgpusim_entrypoint.o: /usr/include/features.h /usr/include/sys/cdefs.h
gpgpusim_entrypoint.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
gpgpusim_entrypoint.o: /usr/include/gnu/stubs-64.h
gpgpusim_entrypoint.o: /usr/include/bits/typesizes.h /usr/include/stdio.h
gpgpusim_entrypoint.o: /usr/include/libio.h /usr/include/_G_config.h
gpgpusim_entrypoint.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
gpgpusim_entrypoint.o: /usr/include/bits/sys_errlist.h option_parser.h
gpgpusim_entrypoint.o: /usr/include/stdlib.h /usr/include/sys/types.h
gpgpusim_entrypoint.o: /usr/include/endian.h /usr/include/bits/endian.h
gpgpusim_entrypoint.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
gpgpusim_entrypoint.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
gpgpusim_entrypoint.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
gpgpusim_entrypoint.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
gpgpusim_entrypoint.o: cuda-sim/cuda-sim.h abstract_hardware_model.h
gpgpusim_entrypoint.o: cuda-sim/dram_callback.h cuda-sim/ptx_ir.h
gpgpusim_entrypoint.o: /usr/include/assert.h cuda-sim/ptx.tab.h
gpgpusim_entrypoint.o: cuda-sim/ptx_sim.h cuda-sim/memory.h
gpgpusim_entrypoint.o: /usr/include/string.h option_parser.h
gpgpusim_entrypoint.o: cuda-sim/ptx_parser.h gpgpu-sim/gpu-sim.h
gpgpusim_entrypoint.o: gpgpu-sim/icnt_wrapper.h
option_parser.o: /usr/include/stdio.h /usr/include/features.h
option_parser.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
option_parser.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
option_parser.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
option_parser.o: /usr/include/libio.h /usr/include/_G_config.h
option_parser.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
option_parser.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
option_parser.o: /usr/include/sys/types.h /usr/include/time.h
option_parser.o: /usr/include/endian.h /usr/include/bits/endian.h
option_parser.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
option_parser.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
option_parser.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
option_parser.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
option_parser.o: /usr/include/assert.h /usr/include/string.h option_parser.h
