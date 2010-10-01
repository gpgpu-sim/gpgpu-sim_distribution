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
CXXFLAGS_L2CACHE = -Wall -DDEBUG

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

SRCS = $(shell ls *.cc)
OBJS = $(SRCS:.cc=.$(OEXT))
 
libgpu_uarch_sim.a:$(OBJS)
	ar rcs  libgpu_uarch_sim.a $(OBJS)

depend:
	makedepend $(SRCS) 2> /dev/null

l2cache.$(OEXT): l2cache.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS_L2CACHE) -o $*.$(OEXT) -c $*.cc

.cc.$(OEXT):
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $*.$(OEXT) -c $*.cc

clean:
	rm -f *.o core *~ *.a 

option_parser.$(OEXT): option_parser.h

dram_sched.$(OEXT): ../cuda-sim/ptx.tab.h

../cuda-sim/ptx.tab.h:
	make -C ../cuda-sim/ ptx.tab.c

# DO NOT DELETE

addrdec.o: /usr/include/string.h /usr/include/features.h
addrdec.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
addrdec.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h addrdec.h
addrdec.o: /usr/include/stdio.h /usr/include/bits/types.h
addrdec.o: /usr/include/bits/typesizes.h /usr/include/libio.h
addrdec.o: /usr/include/_G_config.h /usr/include/wchar.h
addrdec.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
addrdec.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
addrdec.o: /usr/include/endian.h /usr/include/bits/endian.h
addrdec.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
addrdec.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
addrdec.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
addrdec.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
addrdec.o: /usr/include/assert.h ../option_parser.h gpu-sim.h
delayqueue.o: delayqueue.h /usr/include/stdio.h /usr/include/features.h
delayqueue.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
delayqueue.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
delayqueue.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
delayqueue.o: /usr/include/libio.h /usr/include/_G_config.h
delayqueue.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
delayqueue.o: /usr/include/bits/sys_errlist.h /usr/include/assert.h
delayqueue.o: /usr/include/stdlib.h /usr/include/sys/types.h
delayqueue.o: /usr/include/time.h /usr/include/endian.h
delayqueue.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
delayqueue.o: /usr/include/sys/select.h /usr/include/bits/select.h
delayqueue.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
delayqueue.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
delayqueue.o: /usr/include/alloca.h gpu-misc.h gpu-sim.h
delayqueue.o: ../intersim/statwraper.h
dram.o: gpu-sim.h gpu-misc.h dram.h /usr/include/stdio.h
dram.o: /usr/include/features.h /usr/include/sys/cdefs.h
dram.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
dram.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
dram.o: /usr/include/bits/typesizes.h /usr/include/libio.h
dram.o: /usr/include/_G_config.h /usr/include/wchar.h
dram.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
dram.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
dram.o: /usr/include/endian.h /usr/include/bits/endian.h
dram.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
dram.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
dram.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
dram.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h delayqueue.h
dram.o: /usr/include/assert.h ../cuda-sim/dram_callback.h mem_latency_stat.h
dram_sched.o: dram_sched.h dram.h /usr/include/stdio.h
dram_sched.o: /usr/include/features.h /usr/include/sys/cdefs.h
dram_sched.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
dram_sched.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
dram_sched.o: /usr/include/bits/typesizes.h /usr/include/libio.h
dram_sched.o: /usr/include/_G_config.h /usr/include/wchar.h
dram_sched.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
dram_sched.o: /usr/include/stdlib.h /usr/include/sys/types.h
dram_sched.o: /usr/include/time.h /usr/include/endian.h
dram_sched.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
dram_sched.o: /usr/include/sys/select.h /usr/include/bits/select.h
dram_sched.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
dram_sched.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
dram_sched.o: /usr/include/alloca.h delayqueue.h /usr/include/assert.h
dram_sched.o: ../cuda-sim/dram_callback.h shader.h /usr/include/math.h
dram_sched.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
dram_sched.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h gpu-cache.h
dram_sched.o: ../abstract_hardware_model.h stack.h scoreboard.h gpu-sim.h
dram_sched.o: gpu-misc.h mem_latency_stat.h
dwf.o: dwf.h ../abstract_hardware_model.h histogram.h /usr/include/string.h
dwf.o: /usr/include/features.h /usr/include/sys/cdefs.h
dwf.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
dwf.o: /usr/include/gnu/stubs-64.h
gpu-cache.o: gpu-cache.h /usr/include/stdio.h /usr/include/features.h
gpu-cache.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
gpu-cache.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
gpu-cache.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
gpu-cache.o: /usr/include/libio.h /usr/include/_G_config.h
gpu-cache.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
gpu-cache.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
gpu-cache.o: /usr/include/sys/types.h /usr/include/time.h
gpu-cache.o: /usr/include/endian.h /usr/include/bits/endian.h
gpu-cache.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
gpu-cache.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
gpu-cache.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
gpu-cache.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
gpu-cache.o: ../abstract_hardware_model.h gpu-misc.h addrdec.h
gpu-cache.o: /usr/include/assert.h ../option_parser.h stat-tool.h
gpu-cache.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
gpu-cache.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
gpu-cache.o: /usr/include/getopt.h gpu-sim.h /usr/include/string.h
gpu-misc.o: gpu-misc.h
gpu-sim.o: gpu-sim.h /usr/include/stdio.h /usr/include/features.h
gpu-sim.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
gpu-sim.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
gpu-sim.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
gpu-sim.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
gpu-sim.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
gpu-sim.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
gpu-sim.o: /usr/include/endian.h /usr/include/bits/endian.h
gpu-sim.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
gpu-sim.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
gpu-sim.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
gpu-sim.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
gpu-sim.o: /usr/include/math.h /usr/include/bits/huge_val.h
gpu-sim.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
gpu-sim.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
gpu-sim.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
gpu-sim.o: /usr/include/getopt.h ../option_parser.h shader.h
gpu-sim.o: /usr/include/assert.h ../cuda-sim/ptx.tab.h
gpu-sim.o: ../cuda-sim/dram_callback.h gpu-cache.h
gpu-sim.o: ../abstract_hardware_model.h delayqueue.h stack.h dram.h
gpu-sim.o: scoreboard.h mem_fetch.h addrdec.h gpu-misc.h icnt_wrapper.h dwf.h
gpu-sim.o: warp_tracker.h stat-tool.h l2cache.h ../cuda-sim/ptx-stats.h
gpu-sim.o: ../intersim/statwraper.h ../debug.h ../abstract_hardware_model.h
gpu-sim.o: ../gpgpusim_entrypoint.h ../cuda-sim/cuda-sim.h
gpu-sim.o: ../cuda-sim/dram_callback.h mem_latency_stat.h
gpu-sim.o: /usr/include/string.h
icnt_wrapper.o: icnt_wrapper.h /usr/include/assert.h /usr/include/features.h
icnt_wrapper.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
icnt_wrapper.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
icnt_wrapper.o: ../intersim/interconnect_interface.h ../intersim/flit.hpp
icnt_wrapper.o: ../intersim/booksim.hpp ../intersim/trafficmanager.hpp
icnt_wrapper.o: ../intersim/module.hpp ../intersim/config_utils.hpp
icnt_wrapper.o: /usr/include/stdio.h /usr/include/bits/types.h
icnt_wrapper.o: /usr/include/bits/typesizes.h /usr/include/libio.h
icnt_wrapper.o: /usr/include/_G_config.h /usr/include/wchar.h
icnt_wrapper.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
icnt_wrapper.o: ../intersim/network.hpp ../intersim/credit.hpp
icnt_wrapper.o: ../intersim/router.hpp ../intersim/buffer_state.hpp
icnt_wrapper.o: ../intersim/stats.hpp ../intersim/traffic.hpp
icnt_wrapper.o: ../intersim/routefunc.hpp ../intersim/outputset.hpp
icnt_wrapper.o: ../intersim/injection.hpp ../option_parser.h
icnt_wrapper.o: /usr/include/stdlib.h /usr/include/sys/types.h
icnt_wrapper.o: /usr/include/time.h /usr/include/endian.h
icnt_wrapper.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
icnt_wrapper.o: /usr/include/sys/select.h /usr/include/bits/select.h
icnt_wrapper.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
icnt_wrapper.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
icnt_wrapper.o: /usr/include/alloca.h
l2cache.o: /usr/include/stdlib.h /usr/include/features.h
l2cache.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
l2cache.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
l2cache.o: /usr/include/sys/types.h /usr/include/bits/types.h
l2cache.o: /usr/include/bits/typesizes.h /usr/include/time.h
l2cache.o: /usr/include/endian.h /usr/include/bits/endian.h
l2cache.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
l2cache.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
l2cache.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
l2cache.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
l2cache.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
l2cache.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
l2cache.o: /usr/include/bits/sys_errlist.h /usr/include/string.h
l2cache.o: ../tr1_hash_map.h ../option_parser.h mem_fetch.h shader.h
l2cache.o: /usr/include/math.h /usr/include/bits/huge_val.h
l2cache.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
l2cache.o: /usr/include/assert.h ../cuda-sim/ptx.tab.h
l2cache.o: ../cuda-sim/dram_callback.h gpu-cache.h
l2cache.o: ../abstract_hardware_model.h delayqueue.h stack.h dram.h
l2cache.o: scoreboard.h addrdec.h histogram.h l2cache.h
l2cache.o: ../intersim/statwraper.h gpu-sim.h
mem_latency_stat.o: ../abstract_hardware_model.h mem_latency_stat.h gpu-sim.h
mem_latency_stat.o: gpu-misc.h shader.h /usr/include/stdio.h
mem_latency_stat.o: /usr/include/features.h /usr/include/sys/cdefs.h
mem_latency_stat.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
mem_latency_stat.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
mem_latency_stat.o: /usr/include/bits/typesizes.h /usr/include/libio.h
mem_latency_stat.o: /usr/include/_G_config.h /usr/include/wchar.h
mem_latency_stat.o: /usr/include/bits/stdio_lim.h
mem_latency_stat.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
mem_latency_stat.o: /usr/include/sys/types.h /usr/include/time.h
mem_latency_stat.o: /usr/include/endian.h /usr/include/bits/endian.h
mem_latency_stat.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
mem_latency_stat.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
mem_latency_stat.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
mem_latency_stat.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
mem_latency_stat.o: /usr/include/math.h /usr/include/bits/huge_val.h
mem_latency_stat.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
mem_latency_stat.o: /usr/include/assert.h ../cuda-sim/ptx.tab.h
mem_latency_stat.o: ../cuda-sim/dram_callback.h gpu-cache.h delayqueue.h
mem_latency_stat.o: stack.h dram.h scoreboard.h mem_fetch.h addrdec.h
mem_latency_stat.o: ../option_parser.h stat-tool.h /usr/include/zlib.h
mem_latency_stat.o: /usr/include/zconf.h /usr/include/unistd.h
mem_latency_stat.o: /usr/include/bits/posix_opt.h
mem_latency_stat.o: /usr/include/bits/confname.h /usr/include/getopt.h
mem_latency_stat.o: ../cuda-sim/ptx-stats.h visualizer.h
mem_latency_stat.o: /usr/include/string.h
scoreboard.o: scoreboard.h /usr/include/stdio.h /usr/include/features.h
scoreboard.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
scoreboard.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
scoreboard.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
scoreboard.o: /usr/include/libio.h /usr/include/_G_config.h
scoreboard.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
scoreboard.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
scoreboard.o: /usr/include/sys/types.h /usr/include/time.h
scoreboard.o: /usr/include/endian.h /usr/include/bits/endian.h
scoreboard.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
scoreboard.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
scoreboard.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
scoreboard.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
scoreboard.o: /usr/include/assert.h shader.h /usr/include/math.h
scoreboard.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
scoreboard.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h
scoreboard.o: ../cuda-sim/dram_callback.h gpu-cache.h
scoreboard.o: ../abstract_hardware_model.h delayqueue.h stack.h dram.h
scoreboard.o: ../cuda-sim/ptx_sim.h ../cuda-sim/dram_callback.h
scoreboard.o: ../tr1_hash_map.h ../cuda-sim/opcodes.h ../cuda-sim/opcodes.def
scoreboard.o: ../cuda-sim/memory.h /usr/include/string.h
shader.o: shader.h /usr/include/stdio.h /usr/include/features.h
shader.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
shader.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
shader.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
shader.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
shader.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
shader.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
shader.o: /usr/include/endian.h /usr/include/bits/endian.h
shader.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
shader.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
shader.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
shader.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
shader.o: /usr/include/math.h /usr/include/bits/huge_val.h
shader.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
shader.o: /usr/include/assert.h ../cuda-sim/ptx.tab.h
shader.o: ../cuda-sim/dram_callback.h gpu-cache.h
shader.o: ../abstract_hardware_model.h delayqueue.h stack.h dram.h
shader.o: scoreboard.h gpu-sim.h addrdec.h ../option_parser.h dwf.h
shader.o: warp_tracker.h stat-tool.h /usr/include/zlib.h /usr/include/zconf.h
shader.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
shader.o: /usr/include/bits/confname.h /usr/include/getopt.h gpu-misc.h
shader.o: ../cuda-sim/ptx_sim.h ../cuda-sim/dram_callback.h ../tr1_hash_map.h
shader.o: ../cuda-sim/opcodes.h ../cuda-sim/opcodes.def ../cuda-sim/memory.h
shader.o: /usr/include/string.h ../cuda-sim/ptx-stats.h
shader.o: ../cuda-sim/cuda-sim.h mem_fetch.h mem_latency_stat.h visualizer.h
shader.o: /usr/include/limits.h /usr/include/bits/posix1_lim.h
shader.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
shader.o: /usr/include/bits/posix2_lim.h
stack.o: stack.h ../abstract_hardware_model.h /usr/include/stdlib.h
stack.o: /usr/include/features.h /usr/include/sys/cdefs.h
stack.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
stack.o: /usr/include/gnu/stubs-64.h /usr/include/sys/types.h
stack.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
stack.o: /usr/include/time.h /usr/include/endian.h /usr/include/bits/endian.h
stack.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
stack.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
stack.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
stack.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
stack.o: /usr/include/assert.h
stat-tool.o: stat-tool.h ../abstract_hardware_model.h /usr/include/stdio.h
stat-tool.o: /usr/include/features.h /usr/include/sys/cdefs.h
stat-tool.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
stat-tool.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
stat-tool.o: /usr/include/bits/typesizes.h /usr/include/libio.h
stat-tool.o: /usr/include/_G_config.h /usr/include/wchar.h
stat-tool.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
stat-tool.o: /usr/include/zlib.h /usr/include/zconf.h
stat-tool.o: /usr/include/sys/types.h /usr/include/time.h
stat-tool.o: /usr/include/endian.h /usr/include/bits/endian.h
stat-tool.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
stat-tool.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
stat-tool.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
stat-tool.o: /usr/include/bits/pthreadtypes.h /usr/include/unistd.h
stat-tool.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
stat-tool.o: /usr/include/getopt.h /usr/include/stdlib.h
stat-tool.o: /usr/include/alloca.h /usr/include/assert.h histogram.h
visualizer.o: visualizer.h /usr/include/stdio.h /usr/include/features.h
visualizer.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
visualizer.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
visualizer.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
visualizer.o: /usr/include/libio.h /usr/include/_G_config.h
visualizer.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
visualizer.o: /usr/include/bits/sys_errlist.h /usr/include/zlib.h
visualizer.o: /usr/include/zconf.h /usr/include/sys/types.h
visualizer.o: /usr/include/time.h /usr/include/endian.h
visualizer.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
visualizer.o: /usr/include/sys/select.h /usr/include/bits/select.h
visualizer.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
visualizer.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
visualizer.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
visualizer.o: /usr/include/bits/confname.h /usr/include/getopt.h gpu-sim.h
visualizer.o: l2cache.h dram.h /usr/include/stdlib.h /usr/include/alloca.h
visualizer.o: delayqueue.h /usr/include/assert.h ../cuda-sim/dram_callback.h
visualizer.o: shader.h /usr/include/math.h /usr/include/bits/huge_val.h
visualizer.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
visualizer.o: ../cuda-sim/ptx.tab.h gpu-cache.h ../abstract_hardware_model.h
visualizer.o: stack.h scoreboard.h ../option_parser.h mem_latency_stat.h
visualizer.o: stat-tool.h /usr/include/string.h ../gpgpu-sim/shader.h
visualizer.o: ../gpgpu-sim/mem_fetch.h addrdec.h
warp_tracker.o: warp_tracker.h ../abstract_hardware_model.h shader.h
warp_tracker.o: /usr/include/stdio.h /usr/include/features.h
warp_tracker.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
warp_tracker.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
warp_tracker.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
warp_tracker.o: /usr/include/libio.h /usr/include/_G_config.h
warp_tracker.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
warp_tracker.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
warp_tracker.o: /usr/include/sys/types.h /usr/include/time.h
warp_tracker.o: /usr/include/endian.h /usr/include/bits/endian.h
warp_tracker.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
warp_tracker.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
warp_tracker.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
warp_tracker.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
warp_tracker.o: /usr/include/math.h /usr/include/bits/huge_val.h
warp_tracker.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
warp_tracker.o: /usr/include/assert.h ../cuda-sim/ptx.tab.h
warp_tracker.o: ../cuda-sim/dram_callback.h gpu-cache.h delayqueue.h stack.h
warp_tracker.o: dram.h scoreboard.h gpu-sim.h
