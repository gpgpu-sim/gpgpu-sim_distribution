# Copyright (c) 2009-2011, The University of British Columbia
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
addrdec.o: /usr/include/assert.h ../option_parser.h
addrdec.o: ../abstract_hardware_model.h gpu-sim.h shader.h
addrdec.o: /usr/include/math.h /usr/include/bits/huge_val.h
addrdec.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
addrdec.o: ../cuda-sim/ptx.tab.h ../cuda-sim/dram_callback.h delayqueue.h
addrdec.o: ../intersim/statwraper.h gpu-misc.h stack.h dram.h
addrdec.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
addrdec.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
addrdec.o: /usr/include/getopt.h scoreboard.h mem_fetch.h stats.h gpu-cache.h
addrdec.o: ../tr1_hash_map.h
dram.o: gpu-sim.h ../abstract_hardware_model.h addrdec.h /usr/include/stdio.h
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
dram.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
dram.o: /usr/include/assert.h ../option_parser.h shader.h /usr/include/math.h
dram.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
dram.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h
dram.o: ../cuda-sim/dram_callback.h delayqueue.h ../intersim/statwraper.h
dram.o: gpu-misc.h stack.h dram.h /usr/include/zlib.h /usr/include/zconf.h
dram.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
dram.o: /usr/include/bits/confname.h /usr/include/getopt.h scoreboard.h
dram.o: mem_fetch.h stats.h gpu-cache.h ../tr1_hash_map.h mem_latency_stat.h
dram.o: dram_sched.h l2cache.h
dram_sched.o: dram_sched.h dram.h delayqueue.h /usr/include/stdio.h
dram_sched.o: /usr/include/features.h /usr/include/sys/cdefs.h
dram_sched.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
dram_sched.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
dram_sched.o: /usr/include/bits/typesizes.h /usr/include/libio.h
dram_sched.o: /usr/include/_G_config.h /usr/include/wchar.h
dram_sched.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
dram_sched.o: /usr/include/assert.h /usr/include/stdlib.h
dram_sched.o: /usr/include/sys/types.h /usr/include/time.h
dram_sched.o: /usr/include/endian.h /usr/include/bits/endian.h
dram_sched.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
dram_sched.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
dram_sched.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
dram_sched.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
dram_sched.o: ../intersim/statwraper.h gpu-misc.h ../cuda-sim/dram_callback.h
dram_sched.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
dram_sched.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
dram_sched.o: /usr/include/getopt.h shader.h /usr/include/math.h
dram_sched.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
dram_sched.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h stack.h
dram_sched.o: ../abstract_hardware_model.h scoreboard.h mem_fetch.h addrdec.h
dram_sched.o: ../option_parser.h stats.h gpu-cache.h ../tr1_hash_map.h
dram_sched.o: gpu-sim.h mem_latency_stat.h
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
gpu-cache.o: gpu-misc.h mem_fetch.h addrdec.h /usr/include/assert.h
gpu-cache.o: ../option_parser.h ../abstract_hardware_model.h
gpu-cache.o: ../tr1_hash_map.h stat-tool.h histogram.h /usr/include/zlib.h
gpu-cache.o: /usr/include/zconf.h /usr/include/unistd.h
gpu-cache.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
gpu-cache.o: /usr/include/getopt.h
gpu-misc.o: gpu-misc.h
gpu-sim.o: gpu-sim.h ../abstract_hardware_model.h addrdec.h
gpu-sim.o: /usr/include/stdio.h /usr/include/features.h
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
gpu-sim.o: /usr/include/assert.h ../option_parser.h shader.h
gpu-sim.o: /usr/include/math.h /usr/include/bits/huge_val.h
gpu-sim.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
gpu-sim.o: ../cuda-sim/ptx.tab.h ../cuda-sim/dram_callback.h delayqueue.h
gpu-sim.o: ../intersim/statwraper.h gpu-misc.h stack.h dram.h
gpu-sim.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
gpu-sim.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
gpu-sim.o: /usr/include/getopt.h scoreboard.h mem_fetch.h stats.h gpu-cache.h
gpu-sim.o: ../tr1_hash_map.h icnt_wrapper.h stat-tool.h histogram.h l2cache.h
gpu-sim.o: ../cuda-sim/ptx-stats.h ../intersim/interconnect_interface.h
gpu-sim.o: ../intersim/flit.hpp ../intersim/booksim.hpp
gpu-sim.o: ../intersim/trafficmanager.hpp ../intersim/module.hpp
gpu-sim.o: ../intersim/config_utils.hpp ../intersim/network.hpp
gpu-sim.o: ../intersim/credit.hpp ../intersim/router.hpp
gpu-sim.o: ../intersim/buffer_state.hpp ../intersim/stats.hpp
gpu-sim.o: ../intersim/traffic.hpp ../intersim/routefunc.hpp
gpu-sim.o: ../intersim/outputset.hpp ../intersim/injection.hpp ../debug.h
gpu-sim.o: ../abstract_hardware_model.h ../gpgpusim_entrypoint.h
gpu-sim.o: ../cuda-sim/cuda-sim.h ../cuda-sim/dram_callback.h
gpu-sim.o: mem_latency_stat.h visualizer.h /usr/include/string.h
histogram.o: histogram.h /usr/include/assert.h /usr/include/features.h
histogram.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
histogram.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
icnt_wrapper.o: icnt_wrapper.h /usr/include/stdio.h /usr/include/features.h
icnt_wrapper.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
icnt_wrapper.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
icnt_wrapper.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
icnt_wrapper.o: /usr/include/libio.h /usr/include/_G_config.h
icnt_wrapper.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
icnt_wrapper.o: /usr/include/bits/sys_errlist.h /usr/include/assert.h
icnt_wrapper.o: ../intersim/interconnect_interface.h ../intersim/flit.hpp
icnt_wrapper.o: ../intersim/booksim.hpp ../intersim/trafficmanager.hpp
icnt_wrapper.o: ../intersim/module.hpp ../intersim/config_utils.hpp
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
l2cache.o: ../option_parser.h mem_fetch.h addrdec.h /usr/include/assert.h
l2cache.o: ../abstract_hardware_model.h dram.h delayqueue.h
l2cache.o: ../intersim/statwraper.h gpu-misc.h ../cuda-sim/dram_callback.h
l2cache.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
l2cache.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
l2cache.o: /usr/include/getopt.h gpu-cache.h ../tr1_hash_map.h histogram.h
l2cache.o: l2cache.h gpu-sim.h shader.h /usr/include/math.h
l2cache.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
l2cache.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h stack.h
l2cache.o: scoreboard.h stats.h mem_latency_stat.h
mem_fetch.o: mem_fetch.h addrdec.h /usr/include/stdio.h
mem_fetch.o: /usr/include/features.h /usr/include/sys/cdefs.h
mem_fetch.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
mem_fetch.o: /usr/include/gnu/stubs-64.h /usr/include/bits/types.h
mem_fetch.o: /usr/include/bits/typesizes.h /usr/include/libio.h
mem_fetch.o: /usr/include/_G_config.h /usr/include/wchar.h
mem_fetch.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
mem_fetch.o: /usr/include/stdlib.h /usr/include/sys/types.h
mem_fetch.o: /usr/include/time.h /usr/include/endian.h
mem_fetch.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
mem_fetch.o: /usr/include/sys/select.h /usr/include/bits/select.h
mem_fetch.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
mem_fetch.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
mem_fetch.o: /usr/include/alloca.h /usr/include/assert.h ../option_parser.h
mem_fetch.o: ../abstract_hardware_model.h mem_latency_stat.h
mem_fetch.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
mem_fetch.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
mem_fetch.o: /usr/include/getopt.h shader.h /usr/include/math.h
mem_fetch.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
mem_fetch.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h
mem_fetch.o: ../cuda-sim/dram_callback.h delayqueue.h
mem_fetch.o: ../intersim/statwraper.h gpu-misc.h stack.h dram.h scoreboard.h
mem_fetch.o: stats.h gpu-cache.h ../tr1_hash_map.h visualizer.h gpu-sim.h
mem_latency_stat.o: ../abstract_hardware_model.h mem_latency_stat.h
mem_latency_stat.o: /usr/include/stdio.h /usr/include/features.h
mem_latency_stat.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
mem_latency_stat.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
mem_latency_stat.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
mem_latency_stat.o: /usr/include/libio.h /usr/include/_G_config.h
mem_latency_stat.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
mem_latency_stat.o: /usr/include/bits/sys_errlist.h /usr/include/zlib.h
mem_latency_stat.o: /usr/include/zconf.h /usr/include/sys/types.h
mem_latency_stat.o: /usr/include/time.h /usr/include/endian.h
mem_latency_stat.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
mem_latency_stat.o: /usr/include/sys/select.h /usr/include/bits/select.h
mem_latency_stat.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
mem_latency_stat.o: /usr/include/sys/sysmacros.h
mem_latency_stat.o: /usr/include/bits/pthreadtypes.h /usr/include/unistd.h
mem_latency_stat.o: /usr/include/bits/posix_opt.h
mem_latency_stat.o: /usr/include/bits/confname.h /usr/include/getopt.h
mem_latency_stat.o: gpu-sim.h addrdec.h /usr/include/stdlib.h
mem_latency_stat.o: /usr/include/alloca.h /usr/include/assert.h
mem_latency_stat.o: ../option_parser.h shader.h /usr/include/math.h
mem_latency_stat.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
mem_latency_stat.o: /usr/include/bits/mathcalls.h ../cuda-sim/ptx.tab.h
mem_latency_stat.o: ../cuda-sim/dram_callback.h delayqueue.h
mem_latency_stat.o: ../intersim/statwraper.h gpu-misc.h stack.h dram.h
mem_latency_stat.o: scoreboard.h mem_fetch.h stats.h gpu-cache.h
mem_latency_stat.o: ../tr1_hash_map.h stat-tool.h histogram.h
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
scoreboard.o: /usr/include/assert.h ../abstract_hardware_model.h shader.h
scoreboard.o: /usr/include/math.h /usr/include/bits/huge_val.h
scoreboard.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
scoreboard.o: ../cuda-sim/ptx.tab.h ../cuda-sim/dram_callback.h delayqueue.h
scoreboard.o: ../intersim/statwraper.h gpu-misc.h stack.h dram.h
scoreboard.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
scoreboard.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
scoreboard.o: /usr/include/getopt.h mem_fetch.h addrdec.h ../option_parser.h
scoreboard.o: stats.h gpu-cache.h ../tr1_hash_map.h ../cuda-sim/ptx_sim.h
scoreboard.o: ../cuda-sim/dram_callback.h ../cuda-sim/opcodes.h
scoreboard.o: ../cuda-sim/opcodes.def ../cuda-sim/memory.h
scoreboard.o: /usr/include/string.h
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
shader.o: ../cuda-sim/dram_callback.h delayqueue.h ../intersim/statwraper.h
shader.o: gpu-misc.h stack.h ../abstract_hardware_model.h dram.h
shader.o: /usr/include/zlib.h /usr/include/zconf.h /usr/include/unistd.h
shader.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
shader.o: /usr/include/getopt.h scoreboard.h mem_fetch.h addrdec.h
shader.o: ../option_parser.h stats.h gpu-cache.h ../tr1_hash_map.h gpu-sim.h
shader.o: stat-tool.h histogram.h ../cuda-sim/ptx_sim.h
shader.o: ../cuda-sim/dram_callback.h ../cuda-sim/opcodes.h
shader.o: ../cuda-sim/opcodes.def ../cuda-sim/memory.h /usr/include/string.h
shader.o: ../cuda-sim/ptx-stats.h ../cuda-sim/cuda-sim.h mem_latency_stat.h
shader.o: visualizer.h ../intersim/interconnect_interface.h
shader.o: ../intersim/flit.hpp ../intersim/booksim.hpp
shader.o: ../intersim/trafficmanager.hpp ../intersim/module.hpp
shader.o: ../intersim/config_utils.hpp ../intersim/network.hpp
shader.o: ../intersim/credit.hpp ../intersim/router.hpp
shader.o: ../intersim/buffer_state.hpp ../intersim/stats.hpp
shader.o: ../intersim/traffic.hpp ../intersim/routefunc.hpp
shader.o: ../intersim/outputset.hpp ../intersim/injection.hpp icnt_wrapper.h
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
stat-tool.o: stat-tool.h ../abstract_hardware_model.h histogram.h
stat-tool.o: ../tr1_hash_map.h /usr/include/stdio.h /usr/include/features.h
stat-tool.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
stat-tool.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
stat-tool.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
stat-tool.o: /usr/include/libio.h /usr/include/_G_config.h
stat-tool.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
stat-tool.o: /usr/include/bits/sys_errlist.h /usr/include/zlib.h
stat-tool.o: /usr/include/zconf.h /usr/include/sys/types.h
stat-tool.o: /usr/include/time.h /usr/include/endian.h
stat-tool.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
stat-tool.o: /usr/include/sys/select.h /usr/include/bits/select.h
stat-tool.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
stat-tool.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
stat-tool.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
stat-tool.o: /usr/include/bits/confname.h /usr/include/getopt.h
stat-tool.o: /usr/include/stdlib.h /usr/include/alloca.h
stat-tool.o: /usr/include/assert.h
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
visualizer.o: ../abstract_hardware_model.h addrdec.h /usr/include/stdlib.h
visualizer.o: /usr/include/alloca.h /usr/include/assert.h ../option_parser.h
visualizer.o: shader.h /usr/include/math.h /usr/include/bits/huge_val.h
visualizer.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
visualizer.o: ../cuda-sim/ptx.tab.h ../cuda-sim/dram_callback.h delayqueue.h
visualizer.o: ../intersim/statwraper.h gpu-misc.h stack.h dram.h scoreboard.h
visualizer.o: mem_fetch.h stats.h gpu-cache.h ../tr1_hash_map.h l2cache.h
visualizer.o: mem_latency_stat.h stat-tool.h histogram.h
visualizer.o: /usr/include/string.h ../gpgpu-sim/shader.h
visualizer.o: ../gpgpu-sim/mem_fetch.h
