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

../../build/release/gpgpu-sim/addrdec.o: /usr/include/string.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/features.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/addrdec.o: addrdec.h /usr/include/stdio.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/libio.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/time.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/endian.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/assert.h
../../build/release/gpgpu-sim/addrdec.o: ../option_parser.h
../../build/release/gpgpu-sim/addrdec.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/addrdec.o: gpu-sim.h shader.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/math.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/addrdec.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/addrdec.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/addrdec.o: delayqueue.h
../../build/release/gpgpu-sim/addrdec.o: ../intersim/statwraper.h gpu-misc.h
../../build/release/gpgpu-sim/addrdec.o: stack.h dram.h /usr/include/zlib.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/addrdec.o: /usr/include/getopt.h scoreboard.h
../../build/release/gpgpu-sim/addrdec.o: mem_fetch.h stats.h gpu-cache.h
../../build/release/gpgpu-sim/addrdec.o: ../tr1_hash_map.h
../../build/release/gpgpu-sim/dram.o: gpu-sim.h ../abstract_hardware_model.h
../../build/release/gpgpu-sim/dram.o: addrdec.h /usr/include/stdio.h
../../build/release/gpgpu-sim/dram.o: /usr/include/features.h
../../build/release/gpgpu-sim/dram.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/dram.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/dram.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/dram.o: /usr/include/libio.h
../../build/release/gpgpu-sim/dram.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/dram.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/dram.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/dram.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/dram.o: /usr/include/time.h
../../build/release/gpgpu-sim/dram.o: /usr/include/endian.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/dram.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/dram.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/dram.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/dram.o: /usr/include/assert.h
../../build/release/gpgpu-sim/dram.o: ../option_parser.h shader.h
../../build/release/gpgpu-sim/dram.o: /usr/include/math.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/dram.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/dram.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/dram.o: delayqueue.h ../intersim/statwraper.h
../../build/release/gpgpu-sim/dram.o: gpu-misc.h stack.h dram.h
../../build/release/gpgpu-sim/dram.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/dram.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/dram.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/dram.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/dram.o: /usr/include/getopt.h scoreboard.h
../../build/release/gpgpu-sim/dram.o: mem_fetch.h stats.h gpu-cache.h
../../build/release/gpgpu-sim/dram.o: ../tr1_hash_map.h mem_latency_stat.h
../../build/release/gpgpu-sim/dram.o: dram_sched.h l2cache.h
../../build/release/gpgpu-sim/dram_sched.o: dram_sched.h dram.h delayqueue.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/stdio.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/features.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/libio.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/assert.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/time.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/endian.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/dram_sched.o: ../intersim/statwraper.h
../../build/release/gpgpu-sim/dram_sched.o: gpu-misc.h
../../build/release/gpgpu-sim/dram_sched.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/getopt.h shader.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/math.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/dram_sched.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/dram_sched.o: ../cuda-sim/ptx.tab.h stack.h
../../build/release/gpgpu-sim/dram_sched.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/dram_sched.o: scoreboard.h mem_fetch.h
../../build/release/gpgpu-sim/dram_sched.o: addrdec.h ../option_parser.h
../../build/release/gpgpu-sim/dram_sched.o: stats.h gpu-cache.h
../../build/release/gpgpu-sim/dram_sched.o: ../tr1_hash_map.h gpu-sim.h
../../build/release/gpgpu-sim/dram_sched.o: mem_latency_stat.h
../../build/release/gpgpu-sim/gpu-cache.o: gpu-cache.h /usr/include/stdio.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/features.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/libio.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/time.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/endian.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/alloca.h gpu-misc.h
../../build/release/gpgpu-sim/gpu-cache.o: mem_fetch.h addrdec.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/assert.h
../../build/release/gpgpu-sim/gpu-cache.o: ../option_parser.h
../../build/release/gpgpu-sim/gpu-cache.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/gpu-cache.o: ../tr1_hash_map.h stat-tool.h
../../build/release/gpgpu-sim/gpu-cache.o: histogram.h /usr/include/zlib.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/gpu-cache.o: /usr/include/getopt.h
../../build/release/gpgpu-sim/gpu-misc.o: gpu-misc.h
../../build/release/gpgpu-sim/gpu-sim.o: gpu-sim.h
../../build/release/gpgpu-sim/gpu-sim.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/gpu-sim.o: addrdec.h /usr/include/stdio.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/features.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/libio.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/time.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/endian.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/assert.h
../../build/release/gpgpu-sim/gpu-sim.o: ../option_parser.h shader.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/math.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/gpu-sim.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/gpu-sim.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/gpu-sim.o: delayqueue.h
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/statwraper.h gpu-misc.h
../../build/release/gpgpu-sim/gpu-sim.o: stack.h dram.h /usr/include/zlib.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/getopt.h scoreboard.h
../../build/release/gpgpu-sim/gpu-sim.o: mem_fetch.h stats.h gpu-cache.h
../../build/release/gpgpu-sim/gpu-sim.o: ../tr1_hash_map.h icnt_wrapper.h
../../build/release/gpgpu-sim/gpu-sim.o: stat-tool.h histogram.h l2cache.h
../../build/release/gpgpu-sim/gpu-sim.o: ../cuda-sim/ptx-stats.h
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/interconnect_interface.h
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/flit.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/booksim.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/trafficmanager.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/module.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/config_utils.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/network.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/credit.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/router.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/buffer_state.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/stats.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/traffic.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/routefunc.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/outputset.hpp
../../build/release/gpgpu-sim/gpu-sim.o: ../intersim/injection.hpp ../debug.h
../../build/release/gpgpu-sim/gpu-sim.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/gpu-sim.o: ../gpgpusim_entrypoint.h
../../build/release/gpgpu-sim/gpu-sim.o: ../cuda-sim/cuda-sim.h
../../build/release/gpgpu-sim/gpu-sim.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/gpu-sim.o: mem_latency_stat.h visualizer.h
../../build/release/gpgpu-sim/gpu-sim.o: /usr/include/string.h
../../build/release/gpgpu-sim/histogram.o: histogram.h /usr/include/assert.h
../../build/release/gpgpu-sim/histogram.o: /usr/include/features.h
../../build/release/gpgpu-sim/histogram.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/histogram.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/histogram.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/histogram.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/icnt_wrapper.o: icnt_wrapper.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/stdio.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/features.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/libio.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/assert.h
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/interconnect_interface.h
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/flit.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/booksim.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/trafficmanager.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/module.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/config_utils.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/network.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/credit.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/router.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/buffer_state.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/stats.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/traffic.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/routefunc.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/outputset.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../intersim/injection.hpp
../../build/release/gpgpu-sim/icnt_wrapper.o: ../option_parser.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/time.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/endian.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/icnt_wrapper.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/features.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/time.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/endian.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/stdio.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/libio.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/string.h
../../build/release/gpgpu-sim/l2cache.o: ../option_parser.h mem_fetch.h
../../build/release/gpgpu-sim/l2cache.o: addrdec.h /usr/include/assert.h
../../build/release/gpgpu-sim/l2cache.o: ../abstract_hardware_model.h dram.h
../../build/release/gpgpu-sim/l2cache.o: delayqueue.h
../../build/release/gpgpu-sim/l2cache.o: ../intersim/statwraper.h gpu-misc.h
../../build/release/gpgpu-sim/l2cache.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/getopt.h gpu-cache.h
../../build/release/gpgpu-sim/l2cache.o: ../tr1_hash_map.h histogram.h
../../build/release/gpgpu-sim/l2cache.o: l2cache.h gpu-sim.h shader.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/math.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/l2cache.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/l2cache.o: ../cuda-sim/ptx.tab.h stack.h
../../build/release/gpgpu-sim/l2cache.o: scoreboard.h stats.h
../../build/release/gpgpu-sim/l2cache.o: mem_latency_stat.h
../../build/release/gpgpu-sim/mem_fetch.o: mem_fetch.h addrdec.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/stdio.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/features.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/libio.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/time.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/endian.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/assert.h
../../build/release/gpgpu-sim/mem_fetch.o: ../option_parser.h
../../build/release/gpgpu-sim/mem_fetch.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/mem_fetch.o: mem_latency_stat.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/getopt.h shader.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/math.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/mem_fetch.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/mem_fetch.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/mem_fetch.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/mem_fetch.o: delayqueue.h
../../build/release/gpgpu-sim/mem_fetch.o: ../intersim/statwraper.h
../../build/release/gpgpu-sim/mem_fetch.o: gpu-misc.h stack.h dram.h
../../build/release/gpgpu-sim/mem_fetch.o: scoreboard.h stats.h gpu-cache.h
../../build/release/gpgpu-sim/mem_fetch.o: ../tr1_hash_map.h visualizer.h
../../build/release/gpgpu-sim/mem_fetch.o: gpu-sim.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/mem_latency_stat.o: mem_latency_stat.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/stdio.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/features.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/libio.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/time.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/endian.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/getopt.h
../../build/release/gpgpu-sim/mem_latency_stat.o: gpu-sim.h addrdec.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/assert.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../option_parser.h shader.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/math.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/mem_latency_stat.o: delayqueue.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../intersim/statwraper.h
../../build/release/gpgpu-sim/mem_latency_stat.o: gpu-misc.h stack.h dram.h
../../build/release/gpgpu-sim/mem_latency_stat.o: scoreboard.h mem_fetch.h
../../build/release/gpgpu-sim/mem_latency_stat.o: stats.h gpu-cache.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../tr1_hash_map.h
../../build/release/gpgpu-sim/mem_latency_stat.o: stat-tool.h histogram.h
../../build/release/gpgpu-sim/mem_latency_stat.o: ../cuda-sim/ptx-stats.h
../../build/release/gpgpu-sim/mem_latency_stat.o: visualizer.h
../../build/release/gpgpu-sim/mem_latency_stat.o: /usr/include/string.h
../../build/release/gpgpu-sim/scoreboard.o: scoreboard.h /usr/include/stdio.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/features.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/libio.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/time.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/endian.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/assert.h
../../build/release/gpgpu-sim/scoreboard.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/scoreboard.o: shader.h /usr/include/math.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/scoreboard.o: delayqueue.h
../../build/release/gpgpu-sim/scoreboard.o: ../intersim/statwraper.h
../../build/release/gpgpu-sim/scoreboard.o: gpu-misc.h stack.h dram.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/getopt.h mem_fetch.h
../../build/release/gpgpu-sim/scoreboard.o: addrdec.h ../option_parser.h
../../build/release/gpgpu-sim/scoreboard.o: stats.h gpu-cache.h
../../build/release/gpgpu-sim/scoreboard.o: ../tr1_hash_map.h
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/ptx_sim.h
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/opcodes.h
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/opcodes.def
../../build/release/gpgpu-sim/scoreboard.o: ../cuda-sim/memory.h
../../build/release/gpgpu-sim/scoreboard.o: /usr/include/string.h
../../build/release/gpgpu-sim/shader.o: shader.h /usr/include/stdio.h
../../build/release/gpgpu-sim/shader.o: /usr/include/features.h
../../build/release/gpgpu-sim/shader.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/shader.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/shader.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/shader.o: /usr/include/libio.h
../../build/release/gpgpu-sim/shader.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/shader.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/shader.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/shader.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/shader.o: /usr/include/time.h
../../build/release/gpgpu-sim/shader.o: /usr/include/endian.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/shader.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/shader.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/shader.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/shader.o: /usr/include/math.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/shader.o: /usr/include/assert.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/shader.o: delayqueue.h ../intersim/statwraper.h
../../build/release/gpgpu-sim/shader.o: gpu-misc.h stack.h
../../build/release/gpgpu-sim/shader.o: ../abstract_hardware_model.h dram.h
../../build/release/gpgpu-sim/shader.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/shader.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/shader.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/shader.o: /usr/include/getopt.h scoreboard.h
../../build/release/gpgpu-sim/shader.o: mem_fetch.h addrdec.h
../../build/release/gpgpu-sim/shader.o: ../option_parser.h stats.h
../../build/release/gpgpu-sim/shader.o: gpu-cache.h ../tr1_hash_map.h
../../build/release/gpgpu-sim/shader.o: gpu-sim.h stat-tool.h histogram.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/ptx_sim.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/opcodes.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/opcodes.def
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/memory.h
../../build/release/gpgpu-sim/shader.o: /usr/include/string.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/ptx-stats.h
../../build/release/gpgpu-sim/shader.o: ../cuda-sim/cuda-sim.h
../../build/release/gpgpu-sim/shader.o: mem_latency_stat.h visualizer.h
../../build/release/gpgpu-sim/shader.o: ../intersim/interconnect_interface.h
../../build/release/gpgpu-sim/shader.o: ../intersim/flit.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/booksim.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/trafficmanager.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/module.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/config_utils.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/network.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/credit.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/router.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/buffer_state.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/stats.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/traffic.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/routefunc.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/outputset.hpp
../../build/release/gpgpu-sim/shader.o: ../intersim/injection.hpp
../../build/release/gpgpu-sim/shader.o: icnt_wrapper.h /usr/include/limits.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/posix1_lim.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/local_lim.h
../../build/release/gpgpu-sim/shader.o: /usr/include/linux/limits.h
../../build/release/gpgpu-sim/shader.o: /usr/include/bits/posix2_lim.h
../../build/release/gpgpu-sim/stack.o: stack.h ../abstract_hardware_model.h
../../build/release/gpgpu-sim/stack.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/stack.o: /usr/include/features.h
../../build/release/gpgpu-sim/stack.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/stack.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/stack.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/stack.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/stack.o: /usr/include/time.h
../../build/release/gpgpu-sim/stack.o: /usr/include/endian.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/stack.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/stack.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/stack.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/stack.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/stack.o: /usr/include/assert.h
../../build/release/gpgpu-sim/stat-tool.o: stat-tool.h
../../build/release/gpgpu-sim/stat-tool.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/stat-tool.o: histogram.h ../tr1_hash_map.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/stdio.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/features.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/libio.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/time.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/endian.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/getopt.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/stdlib.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/stat-tool.o: /usr/include/assert.h
../../build/release/gpgpu-sim/visualizer.o: visualizer.h /usr/include/stdio.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/features.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/sys/cdefs.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/wordsize.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/gnu/stubs.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/gnu/stubs-64.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/types.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/typesizes.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/libio.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/_G_config.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/wchar.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/stdio_lim.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/sys_errlist.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/zlib.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/zconf.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/sys/types.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/time.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/endian.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/endian.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/byteswap.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/sys/select.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/select.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/sigset.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/time.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/sys/sysmacros.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/pthreadtypes.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/unistd.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/posix_opt.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/confname.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/getopt.h gpu-sim.h
../../build/release/gpgpu-sim/visualizer.o: ../abstract_hardware_model.h
../../build/release/gpgpu-sim/visualizer.o: addrdec.h /usr/include/stdlib.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/alloca.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/assert.h
../../build/release/gpgpu-sim/visualizer.o: ../option_parser.h shader.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/math.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/huge_val.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/mathdef.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/bits/mathcalls.h
../../build/release/gpgpu-sim/visualizer.o: ../cuda-sim/ptx.tab.h
../../build/release/gpgpu-sim/visualizer.o: ../cuda-sim/dram_callback.h
../../build/release/gpgpu-sim/visualizer.o: delayqueue.h
../../build/release/gpgpu-sim/visualizer.o: ../intersim/statwraper.h
../../build/release/gpgpu-sim/visualizer.o: gpu-misc.h stack.h dram.h
../../build/release/gpgpu-sim/visualizer.o: scoreboard.h mem_fetch.h stats.h
../../build/release/gpgpu-sim/visualizer.o: gpu-cache.h ../tr1_hash_map.h
../../build/release/gpgpu-sim/visualizer.o: l2cache.h mem_latency_stat.h
../../build/release/gpgpu-sim/visualizer.o: stat-tool.h histogram.h
../../build/release/gpgpu-sim/visualizer.o: /usr/include/string.h
../../build/release/gpgpu-sim/visualizer.o: ../gpgpu-sim/shader.h
../../build/release/gpgpu-sim/visualizer.o: ../gpgpu-sim/mem_fetch.h
