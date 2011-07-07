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

ifeq ($(DEBUG),1)
	CXXFLAGS = -Wall -DDEBUG
	CXXFLAGS_L2CACHE = -Wall -DDEBUG
else
	CXXFLAGS = -Wall
	CXXFLAGS_L2CACHE = -Wall
endif

GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')
ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
    CXXFLAGS_L2CACHE += -std=c++0x
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS_L2CACHE += 
	CXXFLAGS += 
endif

OPTFLAGS += -g3 -fPIC

CPP = g++ $(SNOW)
OEXT = o

OUTPUT_DIR=../../$(SIM_OBJ_FILES_DIR)/gpgpu-sim

SRCS = $(shell ls *.cc)
OBJS = $(SRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))
 
libgpu_uarch_sim.a:$(OBJS)
	ar rcs  $(OUTPUT_DIR)/libgpu_uarch_sim.a $(OBJS)

depend:
	makedepend -p$(OUTPUT_DIR)/ $(SRCS) 2> /dev/null

$(OUTPUT_DIR)/l2cache.$(OEXT): l2cache.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS_L2CACHE) -o $*.$(OEXT) -c l2cache.cc

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $(OUTPUT_DIR)/$*.$(OEXT) -c $*.cc

clean:
	rm -f *.o core *~ *.a 

$(OUTPUT_DIR)/option_parser.$(OEXT): option_parser.h

$(OUTPUT_DIR)/dram_sched.$(OEXT): ../cuda-sim/ptx.tab.h

../cuda-sim/ptx.tab.h:
	make -C ../cuda-sim/ ptx.tab.c

# DO NOT DELETE

../../build/debug/gpgpu-sim/addrdec.o: /usr/include/string.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/features.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/gnu/stubs-64.h addrdec.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/time.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/addrdec.o: ../option_parser.h
../../build/debug/gpgpu-sim/addrdec.o: ../abstract_hardware_model.h gpu-sim.h
../../build/debug/gpgpu-sim/addrdec.o: shader.h /usr/include/math.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/addrdec.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/addrdec.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/addrdec.o: delayqueue.h ../intersim/statwraper.h
../../build/debug/gpgpu-sim/addrdec.o: gpu-misc.h stack.h dram.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/addrdec.o: /usr/include/getopt.h scoreboard.h
../../build/debug/gpgpu-sim/addrdec.o: mem_fetch.h stats.h gpu-cache.h
../../build/debug/gpgpu-sim/addrdec.o: ../tr1_hash_map.h
../../build/debug/gpgpu-sim/dram.o: gpu-sim.h ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/dram.o: addrdec.h /usr/include/stdio.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/features.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/time.h /usr/include/endian.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/assert.h ../option_parser.h
../../build/debug/gpgpu-sim/dram.o: shader.h /usr/include/math.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/dram.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/dram.o: ../cuda-sim/dram_callback.h delayqueue.h
../../build/debug/gpgpu-sim/dram.o: ../intersim/statwraper.h gpu-misc.h
../../build/debug/gpgpu-sim/dram.o: stack.h dram.h /usr/include/zlib.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/dram.o: /usr/include/getopt.h scoreboard.h
../../build/debug/gpgpu-sim/dram.o: mem_fetch.h stats.h gpu-cache.h
../../build/debug/gpgpu-sim/dram.o: ../tr1_hash_map.h mem_latency_stat.h
../../build/debug/gpgpu-sim/dram.o: dram_sched.h l2cache.h
../../build/debug/gpgpu-sim/dram_sched.o: dram_sched.h dram.h delayqueue.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/features.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/time.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/dram_sched.o: ../intersim/statwraper.h gpu-misc.h
../../build/debug/gpgpu-sim/dram_sched.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/getopt.h shader.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/math.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/dram_sched.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/dram_sched.o: ../cuda-sim/ptx.tab.h stack.h
../../build/debug/gpgpu-sim/dram_sched.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/dram_sched.o: scoreboard.h mem_fetch.h addrdec.h
../../build/debug/gpgpu-sim/dram_sched.o: ../option_parser.h stats.h
../../build/debug/gpgpu-sim/dram_sched.o: gpu-cache.h ../tr1_hash_map.h
../../build/debug/gpgpu-sim/dram_sched.o: gpu-sim.h mem_latency_stat.h
../../build/debug/gpgpu-sim/gpu-cache.o: gpu-cache.h /usr/include/stdio.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/features.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/time.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/alloca.h gpu-misc.h
../../build/debug/gpgpu-sim/gpu-cache.o: mem_fetch.h addrdec.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/gpu-cache.o: ../option_parser.h
../../build/debug/gpgpu-sim/gpu-cache.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/gpu-cache.o: ../tr1_hash_map.h stat-tool.h
../../build/debug/gpgpu-sim/gpu-cache.o: histogram.h /usr/include/zlib.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/getopt.h shader.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/math.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/gpu-cache.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/gpu-cache.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/gpu-cache.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/gpu-cache.o: delayqueue.h
../../build/debug/gpgpu-sim/gpu-cache.o: ../intersim/statwraper.h stack.h
../../build/debug/gpgpu-sim/gpu-cache.o: dram.h scoreboard.h stats.h
../../build/debug/gpgpu-sim/gpu-misc.o: gpu-misc.h
../../build/debug/gpgpu-sim/gpu-sim.o: gpu-sim.h ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/gpu-sim.o: addrdec.h /usr/include/stdio.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/features.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/time.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../option_parser.h shader.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/math.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/gpu-sim.o: delayqueue.h ../intersim/statwraper.h
../../build/debug/gpgpu-sim/gpu-sim.o: gpu-misc.h stack.h dram.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/getopt.h scoreboard.h
../../build/debug/gpgpu-sim/gpu-sim.o: mem_fetch.h stats.h gpu-cache.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../tr1_hash_map.h icnt_wrapper.h
../../build/debug/gpgpu-sim/gpu-sim.o: stat-tool.h histogram.h l2cache.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../cuda-sim/ptx-stats.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/interconnect_interface.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/flit.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/booksim.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/trafficmanager.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/module.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/config_utils.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/network.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/credit.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/router.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/buffer_state.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/stats.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/traffic.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/routefunc.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/outputset.hpp
../../build/debug/gpgpu-sim/gpu-sim.o: ../intersim/injection.hpp ../debug.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../gpgpusim_entrypoint.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../cuda-sim/cuda-sim.h
../../build/debug/gpgpu-sim/gpu-sim.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/gpu-sim.o: mem_latency_stat.h visualizer.h
../../build/debug/gpgpu-sim/gpu-sim.o: /usr/include/string.h
../../build/debug/gpgpu-sim/histogram.o: histogram.h /usr/include/assert.h
../../build/debug/gpgpu-sim/histogram.o: /usr/include/features.h
../../build/debug/gpgpu-sim/histogram.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/histogram.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/histogram.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/histogram.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: icnt_wrapper.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/features.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/interconnect_interface.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/flit.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/booksim.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/trafficmanager.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/module.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/config_utils.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/network.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/credit.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/router.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/buffer_state.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/stats.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/traffic.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/routefunc.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/outputset.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../intersim/injection.hpp
../../build/debug/gpgpu-sim/icnt_wrapper.o: ../option_parser.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/time.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/icnt_wrapper.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/features.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/time.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/string.h
../../build/debug/gpgpu-sim/l2cache.o: ../option_parser.h mem_fetch.h
../../build/debug/gpgpu-sim/l2cache.o: addrdec.h /usr/include/assert.h
../../build/debug/gpgpu-sim/l2cache.o: ../abstract_hardware_model.h dram.h
../../build/debug/gpgpu-sim/l2cache.o: delayqueue.h ../intersim/statwraper.h
../../build/debug/gpgpu-sim/l2cache.o: gpu-misc.h ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/getopt.h gpu-cache.h
../../build/debug/gpgpu-sim/l2cache.o: ../tr1_hash_map.h histogram.h
../../build/debug/gpgpu-sim/l2cache.o: l2cache.h gpu-sim.h shader.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/math.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/l2cache.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/l2cache.o: ../cuda-sim/ptx.tab.h stack.h
../../build/debug/gpgpu-sim/l2cache.o: scoreboard.h stats.h
../../build/debug/gpgpu-sim/l2cache.o: mem_latency_stat.h
../../build/debug/gpgpu-sim/mem_fetch.o: mem_fetch.h addrdec.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/features.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/time.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/mem_fetch.o: ../option_parser.h
../../build/debug/gpgpu-sim/mem_fetch.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/mem_fetch.o: mem_latency_stat.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/getopt.h shader.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/math.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/mem_fetch.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/mem_fetch.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/mem_fetch.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/mem_fetch.o: delayqueue.h
../../build/debug/gpgpu-sim/mem_fetch.o: ../intersim/statwraper.h gpu-misc.h
../../build/debug/gpgpu-sim/mem_fetch.o: stack.h dram.h scoreboard.h stats.h
../../build/debug/gpgpu-sim/mem_fetch.o: gpu-cache.h ../tr1_hash_map.h
../../build/debug/gpgpu-sim/mem_fetch.o: visualizer.h gpu-sim.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: mem_latency_stat.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/features.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/time.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/getopt.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: gpu-sim.h addrdec.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../option_parser.h shader.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/math.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: delayqueue.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../intersim/statwraper.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: gpu-misc.h stack.h dram.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: scoreboard.h mem_fetch.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: stats.h gpu-cache.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../tr1_hash_map.h stat-tool.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: histogram.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: ../cuda-sim/ptx-stats.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: visualizer.h
../../build/debug/gpgpu-sim/mem_latency_stat.o: /usr/include/string.h
../../build/debug/gpgpu-sim/scoreboard.o: scoreboard.h /usr/include/stdio.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/features.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/time.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/scoreboard.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/scoreboard.o: shader.h /usr/include/math.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/scoreboard.o: delayqueue.h
../../build/debug/gpgpu-sim/scoreboard.o: ../intersim/statwraper.h gpu-misc.h
../../build/debug/gpgpu-sim/scoreboard.o: stack.h dram.h /usr/include/zlib.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/getopt.h mem_fetch.h
../../build/debug/gpgpu-sim/scoreboard.o: addrdec.h ../option_parser.h
../../build/debug/gpgpu-sim/scoreboard.o: stats.h gpu-cache.h
../../build/debug/gpgpu-sim/scoreboard.o: ../tr1_hash_map.h
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/ptx_sim.h
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/opcodes.h
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/opcodes.def
../../build/debug/gpgpu-sim/scoreboard.o: ../cuda-sim/memory.h
../../build/debug/gpgpu-sim/scoreboard.o: /usr/include/string.h
../../build/debug/gpgpu-sim/shader.o: shader.h /usr/include/stdio.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/features.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/time.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/math.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/shader.o: delayqueue.h ../intersim/statwraper.h
../../build/debug/gpgpu-sim/shader.o: gpu-misc.h stack.h
../../build/debug/gpgpu-sim/shader.o: ../abstract_hardware_model.h dram.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/getopt.h scoreboard.h
../../build/debug/gpgpu-sim/shader.o: mem_fetch.h addrdec.h
../../build/debug/gpgpu-sim/shader.o: ../option_parser.h stats.h gpu-cache.h
../../build/debug/gpgpu-sim/shader.o: ../tr1_hash_map.h gpu-sim.h stat-tool.h
../../build/debug/gpgpu-sim/shader.o: histogram.h ../cuda-sim/ptx_sim.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/opcodes.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/opcodes.def
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/memory.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/string.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/ptx-stats.h
../../build/debug/gpgpu-sim/shader.o: ../cuda-sim/cuda-sim.h
../../build/debug/gpgpu-sim/shader.o: mem_latency_stat.h visualizer.h
../../build/debug/gpgpu-sim/shader.o: ../intersim/interconnect_interface.h
../../build/debug/gpgpu-sim/shader.o: ../intersim/flit.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/booksim.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/trafficmanager.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/module.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/config_utils.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/network.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/credit.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/router.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/buffer_state.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/stats.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/traffic.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/routefunc.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/outputset.hpp
../../build/debug/gpgpu-sim/shader.o: ../intersim/injection.hpp
../../build/debug/gpgpu-sim/shader.o: icnt_wrapper.h /usr/include/limits.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/posix1_lim.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/local_lim.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/linux/limits.h
../../build/debug/gpgpu-sim/shader.o: /usr/include/bits/posix2_lim.h
../../build/debug/gpgpu-sim/stack.o: stack.h ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/features.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/time.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/stack.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/stat-tool.o: stat-tool.h
../../build/debug/gpgpu-sim/stat-tool.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/stat-tool.o: histogram.h ../tr1_hash_map.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/stdio.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/features.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/time.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/getopt.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/stdlib.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/stat-tool.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/visualizer.o: visualizer.h /usr/include/stdio.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/features.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/sys/cdefs.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/wordsize.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/gnu/stubs.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/gnu/stubs-64.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/types.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/typesizes.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/libio.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/_G_config.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/wchar.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/stdio_lim.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/sys_errlist.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/zlib.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/zconf.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/sys/types.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/time.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/endian.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/endian.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/byteswap.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/sys/select.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/select.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/sigset.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/time.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/sys/sysmacros.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/pthreadtypes.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/unistd.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/posix_opt.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/confname.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/getopt.h gpu-sim.h
../../build/debug/gpgpu-sim/visualizer.o: ../abstract_hardware_model.h
../../build/debug/gpgpu-sim/visualizer.o: addrdec.h /usr/include/stdlib.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/alloca.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/assert.h
../../build/debug/gpgpu-sim/visualizer.o: ../option_parser.h shader.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/math.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/huge_val.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/mathdef.h
../../build/debug/gpgpu-sim/visualizer.o: /usr/include/bits/mathcalls.h
../../build/debug/gpgpu-sim/visualizer.o: ../cuda-sim/ptx.tab.h
../../build/debug/gpgpu-sim/visualizer.o: ../cuda-sim/dram_callback.h
../../build/debug/gpgpu-sim/visualizer.o: delayqueue.h
../../build/debug/gpgpu-sim/visualizer.o: ../intersim/statwraper.h gpu-misc.h
../../build/debug/gpgpu-sim/visualizer.o: stack.h dram.h scoreboard.h
../../build/debug/gpgpu-sim/visualizer.o: mem_fetch.h stats.h gpu-cache.h
../../build/debug/gpgpu-sim/visualizer.o: ../tr1_hash_map.h l2cache.h
../../build/debug/gpgpu-sim/visualizer.o: mem_latency_stat.h stat-tool.h
../../build/debug/gpgpu-sim/visualizer.o: histogram.h /usr/include/string.h
../../build/debug/gpgpu-sim/visualizer.o: ../gpgpu-sim/shader.h
../../build/debug/gpgpu-sim/visualizer.o: ../gpgpu-sim/mem_fetch.h
