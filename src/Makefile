# Copyright (c) 2009-2011, Tor M. Aamodt, Timothy G. Rogers, Wilson W.L. Fung
# Ali Bakhoda, Ivan Sham 
# The University of British Columbia
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

../build/release/abstract_hardware_model.o: abstract_hardware_model.h
../build/release/abstract_hardware_model.o: cuda-sim/memory.h
../build/release/abstract_hardware_model.o: abstract_hardware_model.h
../build/release/abstract_hardware_model.o: /usr/include/assert.h
../build/release/abstract_hardware_model.o: /usr/include/features.h
../build/release/abstract_hardware_model.o: /usr/include/sys/cdefs.h
../build/release/abstract_hardware_model.o: /usr/include/bits/wordsize.h
../build/release/abstract_hardware_model.o: /usr/include/gnu/stubs.h
../build/release/abstract_hardware_model.o: /usr/include/gnu/stubs-64.h
../build/release/abstract_hardware_model.o: /usr/include/string.h
../build/release/abstract_hardware_model.o: /usr/include/xlocale.h
../build/release/abstract_hardware_model.o: /usr/include/stdio.h
../build/release/abstract_hardware_model.o: /usr/include/bits/types.h
../build/release/abstract_hardware_model.o: /usr/include/bits/typesizes.h
../build/release/abstract_hardware_model.o: /usr/include/libio.h
../build/release/abstract_hardware_model.o: /usr/include/_G_config.h
../build/release/abstract_hardware_model.o: /usr/include/wchar.h
../build/release/abstract_hardware_model.o: /usr/include/bits/stdio_lim.h
../build/release/abstract_hardware_model.o: /usr/include/bits/sys_errlist.h
../build/release/abstract_hardware_model.o: /usr/include/stdlib.h
../build/release/abstract_hardware_model.o: /usr/include/sys/types.h
../build/release/abstract_hardware_model.o: /usr/include/time.h
../build/release/abstract_hardware_model.o: /usr/include/endian.h
../build/release/abstract_hardware_model.o: /usr/include/bits/endian.h
../build/release/abstract_hardware_model.o: /usr/include/bits/byteswap.h
../build/release/abstract_hardware_model.o: /usr/include/sys/select.h
../build/release/abstract_hardware_model.o: /usr/include/bits/select.h
../build/release/abstract_hardware_model.o: /usr/include/bits/sigset.h
../build/release/abstract_hardware_model.o: /usr/include/bits/time.h
../build/release/abstract_hardware_model.o: /usr/include/sys/sysmacros.h
../build/release/abstract_hardware_model.o: /usr/include/bits/pthreadtypes.h
../build/release/abstract_hardware_model.o: /usr/include/alloca.h
../build/release/abstract_hardware_model.o: option_parser.h cuda-sim/ptx_ir.h
../build/release/abstract_hardware_model.o: cuda-sim/ptx.tab.h
../build/release/abstract_hardware_model.o: cuda-sim/ptx_sim.h tr1_hash_map.h
../build/release/abstract_hardware_model.o: cuda-sim/opcodes.h
../build/release/abstract_hardware_model.o: cuda-sim/opcodes.def
../build/release/abstract_hardware_model.o: cuda-sim/memory.h option_parser.h
../build/release/abstract_hardware_model.o: cuda-sim/ptx-stats.h
../build/release/debug.o: debug.h abstract_hardware_model.h
../build/release/debug.o: gpgpu-sim/shader.h /usr/include/stdio.h
../build/release/debug.o: /usr/include/features.h /usr/include/sys/cdefs.h
../build/release/debug.o: /usr/include/bits/wordsize.h
../build/release/debug.o: /usr/include/gnu/stubs.h
../build/release/debug.o: /usr/include/gnu/stubs-64.h
../build/release/debug.o: /usr/include/bits/types.h
../build/release/debug.o: /usr/include/bits/typesizes.h /usr/include/libio.h
../build/release/debug.o: /usr/include/_G_config.h /usr/include/wchar.h
../build/release/debug.o: /usr/include/bits/stdio_lim.h
../build/release/debug.o: /usr/include/bits/sys_errlist.h
../build/release/debug.o: /usr/include/stdlib.h /usr/include/sys/types.h
../build/release/debug.o: /usr/include/time.h /usr/include/endian.h
../build/release/debug.o: /usr/include/bits/endian.h
../build/release/debug.o: /usr/include/bits/byteswap.h
../build/release/debug.o: /usr/include/sys/select.h
../build/release/debug.o: /usr/include/bits/select.h
../build/release/debug.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
../build/release/debug.o: /usr/include/sys/sysmacros.h
../build/release/debug.o: /usr/include/bits/pthreadtypes.h
../build/release/debug.o: /usr/include/alloca.h /usr/include/math.h
../build/release/debug.o: /usr/include/bits/huge_val.h
../build/release/debug.o: /usr/include/bits/huge_valf.h
../build/release/debug.o: /usr/include/bits/huge_vall.h
../build/release/debug.o: /usr/include/bits/inf.h /usr/include/bits/nan.h
../build/release/debug.o: /usr/include/bits/mathdef.h
../build/release/debug.o: /usr/include/bits/mathcalls.h /usr/include/assert.h
../build/release/debug.o: cuda-sim/ptx.tab.h gpgpu-sim/delayqueue.h
../build/release/debug.o: intersim/statwraper.h gpgpu-sim/gpu-misc.h
../build/release/debug.o: gpgpu-sim/stack.h abstract_hardware_model.h
../build/release/debug.o: gpgpu-sim/dram.h /usr/include/zlib.h
../build/release/debug.o: /usr/include/zconf.h /usr/include/unistd.h
../build/release/debug.o: /usr/include/bits/posix_opt.h
../build/release/debug.o: /usr/include/bits/confname.h /usr/include/getopt.h
../build/release/debug.o: gpgpu-sim/scoreboard.h gpgpu-sim/mem_fetch.h
../build/release/debug.o: gpgpu-sim/addrdec.h option_parser.h
../build/release/debug.o: gpgpu-sim/stats.h gpgpu-sim/gpu-cache.h
../build/release/debug.o: tr1_hash_map.h gpgpu-sim/gpu-sim.h
../build/release/debug.o: gpgpu-sim/shader.h cuda-sim/ptx_sim.h
../build/release/debug.o: cuda-sim/opcodes.h cuda-sim/opcodes.def
../build/release/debug.o: cuda-sim/memory.h cuda-sim/cuda-sim.h
../build/release/debug.o: cuda-sim/ptx_ir.h cuda-sim/ptx.tab.h
../build/release/debug.o: cuda-sim/ptx_sim.h /usr/include/string.h
../build/release/debug.o: /usr/include/xlocale.h
../build/release/gpgpusim_entrypoint.o: gpgpusim_entrypoint.h
../build/release/gpgpusim_entrypoint.o: abstract_hardware_model.h
../build/release/gpgpusim_entrypoint.o: /usr/include/time.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/types.h
../build/release/gpgpusim_entrypoint.o: /usr/include/features.h
../build/release/gpgpusim_entrypoint.o: /usr/include/sys/cdefs.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/wordsize.h
../build/release/gpgpusim_entrypoint.o: /usr/include/gnu/stubs.h
../build/release/gpgpusim_entrypoint.o: /usr/include/gnu/stubs-64.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/typesizes.h
../build/release/gpgpusim_entrypoint.o: /usr/include/stdio.h
../build/release/gpgpusim_entrypoint.o: /usr/include/libio.h
../build/release/gpgpusim_entrypoint.o: /usr/include/_G_config.h
../build/release/gpgpusim_entrypoint.o: /usr/include/wchar.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/stdio_lim.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/sys_errlist.h
../build/release/gpgpusim_entrypoint.o: option_parser.h /usr/include/stdlib.h
../build/release/gpgpusim_entrypoint.o: /usr/include/sys/types.h
../build/release/gpgpusim_entrypoint.o: /usr/include/endian.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/endian.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/byteswap.h
../build/release/gpgpusim_entrypoint.o: /usr/include/sys/select.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/select.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/sigset.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/time.h
../build/release/gpgpusim_entrypoint.o: /usr/include/sys/sysmacros.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/pthreadtypes.h
../build/release/gpgpusim_entrypoint.o: /usr/include/alloca.h
../build/release/gpgpusim_entrypoint.o: cuda-sim/cuda-sim.h
../build/release/gpgpusim_entrypoint.o: abstract_hardware_model.h
../build/release/gpgpusim_entrypoint.o: cuda-sim/ptx_ir.h
../build/release/gpgpusim_entrypoint.o: /usr/include/assert.h
../build/release/gpgpusim_entrypoint.o: cuda-sim/ptx.tab.h cuda-sim/ptx_sim.h
../build/release/gpgpusim_entrypoint.o: tr1_hash_map.h cuda-sim/opcodes.h
../build/release/gpgpusim_entrypoint.o: cuda-sim/opcodes.def
../build/release/gpgpusim_entrypoint.o: cuda-sim/memory.h option_parser.h
../build/release/gpgpusim_entrypoint.o: cuda-sim/ptx_parser.h
../build/release/gpgpusim_entrypoint.o: gpgpu-sim/gpu-sim.h
../build/release/gpgpusim_entrypoint.o: gpgpu-sim/addrdec.h
../build/release/gpgpusim_entrypoint.o: gpgpu-sim/shader.h
../build/release/gpgpusim_entrypoint.o: gpgpu-sim/icnt_wrapper.h
../build/release/gpgpusim_entrypoint.o: stream_manager.h
../build/release/gpgpusim_entrypoint.o: /usr/include/pthread.h
../build/release/gpgpusim_entrypoint.o: /usr/include/sched.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/sched.h
../build/release/gpgpusim_entrypoint.o: /usr/include/signal.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/setjmp.h
../build/release/gpgpusim_entrypoint.o: /usr/include/semaphore.h
../build/release/gpgpusim_entrypoint.o: /usr/include/bits/semaphore.h
../build/release/option_parser.o: /usr/include/stdio.h
../build/release/option_parser.o: /usr/include/features.h
../build/release/option_parser.o: /usr/include/sys/cdefs.h
../build/release/option_parser.o: /usr/include/bits/wordsize.h
../build/release/option_parser.o: /usr/include/gnu/stubs.h
../build/release/option_parser.o: /usr/include/gnu/stubs-64.h
../build/release/option_parser.o: /usr/include/bits/types.h
../build/release/option_parser.o: /usr/include/bits/typesizes.h
../build/release/option_parser.o: /usr/include/libio.h
../build/release/option_parser.o: /usr/include/_G_config.h
../build/release/option_parser.o: /usr/include/wchar.h
../build/release/option_parser.o: /usr/include/bits/stdio_lim.h
../build/release/option_parser.o: /usr/include/bits/sys_errlist.h
../build/release/option_parser.o: /usr/include/stdlib.h
../build/release/option_parser.o: /usr/include/sys/types.h
../build/release/option_parser.o: /usr/include/time.h /usr/include/endian.h
../build/release/option_parser.o: /usr/include/bits/endian.h
../build/release/option_parser.o: /usr/include/bits/byteswap.h
../build/release/option_parser.o: /usr/include/sys/select.h
../build/release/option_parser.o: /usr/include/bits/select.h
../build/release/option_parser.o: /usr/include/bits/sigset.h
../build/release/option_parser.o: /usr/include/bits/time.h
../build/release/option_parser.o: /usr/include/sys/sysmacros.h
../build/release/option_parser.o: /usr/include/bits/pthreadtypes.h
../build/release/option_parser.o: /usr/include/alloca.h /usr/include/assert.h
../build/release/option_parser.o: /usr/include/string.h
../build/release/option_parser.o: /usr/include/xlocale.h option_parser.h
../build/release/stream_manager.o: stream_manager.h abstract_hardware_model.h
../build/release/stream_manager.o: /usr/include/pthread.h
../build/release/stream_manager.o: /usr/include/features.h
../build/release/stream_manager.o: /usr/include/sys/cdefs.h
../build/release/stream_manager.o: /usr/include/bits/wordsize.h
../build/release/stream_manager.o: /usr/include/gnu/stubs.h
../build/release/stream_manager.o: /usr/include/gnu/stubs-64.h
../build/release/stream_manager.o: /usr/include/endian.h
../build/release/stream_manager.o: /usr/include/bits/endian.h
../build/release/stream_manager.o: /usr/include/bits/byteswap.h
../build/release/stream_manager.o: /usr/include/sched.h
../build/release/stream_manager.o: /usr/include/bits/types.h
../build/release/stream_manager.o: /usr/include/bits/typesizes.h
../build/release/stream_manager.o: /usr/include/time.h
../build/release/stream_manager.o: /usr/include/bits/sched.h
../build/release/stream_manager.o: /usr/include/signal.h
../build/release/stream_manager.o: /usr/include/bits/sigset.h
../build/release/stream_manager.o: /usr/include/bits/pthreadtypes.h
../build/release/stream_manager.o: /usr/include/bits/setjmp.h
../build/release/stream_manager.o: gpgpusim_entrypoint.h cuda-sim/cuda-sim.h
../build/release/stream_manager.o: abstract_hardware_model.h
../build/release/stream_manager.o: /usr/include/stdlib.h
../build/release/stream_manager.o: /usr/include/sys/types.h
../build/release/stream_manager.o: /usr/include/sys/select.h
../build/release/stream_manager.o: /usr/include/bits/select.h
../build/release/stream_manager.o: /usr/include/bits/time.h
../build/release/stream_manager.o: /usr/include/sys/sysmacros.h
../build/release/stream_manager.o: /usr/include/alloca.h gpgpu-sim/gpu-sim.h
../build/release/stream_manager.o: gpgpu-sim/addrdec.h /usr/include/stdio.h
../build/release/stream_manager.o: /usr/include/libio.h
../build/release/stream_manager.o: /usr/include/_G_config.h
../build/release/stream_manager.o: /usr/include/wchar.h
../build/release/stream_manager.o: /usr/include/bits/stdio_lim.h
../build/release/stream_manager.o: /usr/include/bits/sys_errlist.h
../build/release/stream_manager.o: /usr/include/assert.h option_parser.h
../build/release/stream_manager.o: gpgpu-sim/shader.h
