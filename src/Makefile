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

CUDART_VERSION:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//' | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

CXXFLAGS = -Wall -DDEBUG
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

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
../build/debug/abstract_hardware_model.o: /usr/include/xlocale.h
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
../build/debug/abstract_hardware_model.o: cuda-sim/ptx_sim.h tr1_hash_map.h
../build/debug/abstract_hardware_model.o: cuda-sim/opcodes.h
../build/debug/abstract_hardware_model.o: cuda-sim/opcodes.def
../build/debug/abstract_hardware_model.o: cuda-sim/memory.h option_parser.h
../build/debug/abstract_hardware_model.o: cuda-sim/ptx-stats.h
../build/debug/abstract_hardware_model.o: cuda-sim/cuda-sim.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/shader.h
../build/debug/abstract_hardware_model.o: /usr/include/math.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/huge_val.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/huge_valf.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/huge_vall.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/inf.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/nan.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/mathdef.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/mathcalls.h
../build/debug/abstract_hardware_model.o: cuda-sim/ptx.tab.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/delayqueue.h
../build/debug/abstract_hardware_model.o: intersim/statwraper.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/gpu-misc.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/stack.h gpgpu-sim/dram.h
../build/debug/abstract_hardware_model.o: /usr/include/zlib.h
../build/debug/abstract_hardware_model.o: /usr/include/zconf.h
../build/debug/abstract_hardware_model.o: /usr/include/unistd.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/posix_opt.h
../build/debug/abstract_hardware_model.o: /usr/include/bits/confname.h
../build/debug/abstract_hardware_model.o: /usr/include/getopt.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/scoreboard.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/mem_fetch.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/addrdec.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/mem_fetch_status.tup
../build/debug/abstract_hardware_model.o: gpgpu-sim/stats.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/gpu-cache.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/gpu-sim.h
../build/debug/abstract_hardware_model.o: gpgpu-sim/shader.h
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
../build/debug/debug.o: /usr/include/bits/huge_valf.h
../build/debug/debug.o: /usr/include/bits/huge_vall.h /usr/include/bits/inf.h
../build/debug/debug.o: /usr/include/bits/nan.h /usr/include/bits/mathdef.h
../build/debug/debug.o: /usr/include/bits/mathcalls.h /usr/include/assert.h
../build/debug/debug.o: cuda-sim/ptx.tab.h gpgpu-sim/delayqueue.h
../build/debug/debug.o: intersim/statwraper.h gpgpu-sim/gpu-misc.h
../build/debug/debug.o: gpgpu-sim/stack.h abstract_hardware_model.h
../build/debug/debug.o: gpgpu-sim/dram.h /usr/include/zlib.h
../build/debug/debug.o: /usr/include/zconf.h /usr/include/unistd.h
../build/debug/debug.o: /usr/include/bits/posix_opt.h
../build/debug/debug.o: /usr/include/bits/confname.h /usr/include/getopt.h
../build/debug/debug.o: gpgpu-sim/scoreboard.h gpgpu-sim/mem_fetch.h
../build/debug/debug.o: gpgpu-sim/addrdec.h option_parser.h
../build/debug/debug.o: gpgpu-sim/mem_fetch_status.tup gpgpu-sim/stats.h
../build/debug/debug.o: gpgpu-sim/gpu-cache.h tr1_hash_map.h
../build/debug/debug.o: gpgpu-sim/gpu-sim.h gpgpu-sim/shader.h
../build/debug/debug.o: cuda-sim/ptx_sim.h cuda-sim/opcodes.h
../build/debug/debug.o: cuda-sim/opcodes.def cuda-sim/memory.h
../build/debug/debug.o: cuda-sim/cuda-sim.h gpgpu-sim/shader.h
../build/debug/debug.o: cuda-sim/ptx_sim.h cuda-sim/ptx_ir.h
../build/debug/debug.o: cuda-sim/ptx.tab.h /usr/include/string.h
../build/debug/debug.o: /usr/include/xlocale.h
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
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/shader.h /usr/include/math.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/huge_val.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/huge_valf.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/huge_vall.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/inf.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/nan.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/mathdef.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/mathcalls.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/assert.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/ptx.tab.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/delayqueue.h
../build/debug/gpgpusim_entrypoint.o: intersim/statwraper.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/gpu-misc.h gpgpu-sim/stack.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/dram.h /usr/include/zlib.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/zconf.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/unistd.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/posix_opt.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/bits/confname.h
../build/debug/gpgpusim_entrypoint.o: /usr/include/getopt.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/scoreboard.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/mem_fetch.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/addrdec.h option_parser.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/mem_fetch_status.tup
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/stats.h gpgpu-sim/gpu-cache.h
../build/debug/gpgpusim_entrypoint.o: tr1_hash_map.h cuda-sim/ptx_sim.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/opcodes.h cuda-sim/opcodes.def
../build/debug/gpgpusim_entrypoint.o: cuda-sim/memory.h cuda-sim/ptx_ir.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/ptx.tab.h
../build/debug/gpgpusim_entrypoint.o: cuda-sim/ptx_parser.h
../build/debug/gpgpusim_entrypoint.o: gpgpu-sim/gpu-sim.h gpgpu-sim/shader.h
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
../build/debug/option_parser.o: /usr/include/string.h /usr/include/xlocale.h
../build/debug/option_parser.o: option_parser.h
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
../build/debug/stream_manager.o: abstract_hardware_model.h gpgpu-sim/shader.h
../build/debug/stream_manager.o: /usr/include/stdio.h /usr/include/libio.h
../build/debug/stream_manager.o: /usr/include/_G_config.h
../build/debug/stream_manager.o: /usr/include/wchar.h
../build/debug/stream_manager.o: /usr/include/bits/stdio_lim.h
../build/debug/stream_manager.o: /usr/include/bits/sys_errlist.h
../build/debug/stream_manager.o: /usr/include/stdlib.h
../build/debug/stream_manager.o: /usr/include/sys/types.h
../build/debug/stream_manager.o: /usr/include/sys/select.h
../build/debug/stream_manager.o: /usr/include/bits/select.h
../build/debug/stream_manager.o: /usr/include/bits/time.h
../build/debug/stream_manager.o: /usr/include/sys/sysmacros.h
../build/debug/stream_manager.o: /usr/include/alloca.h /usr/include/math.h
../build/debug/stream_manager.o: /usr/include/bits/huge_val.h
../build/debug/stream_manager.o: /usr/include/bits/huge_valf.h
../build/debug/stream_manager.o: /usr/include/bits/huge_vall.h
../build/debug/stream_manager.o: /usr/include/bits/inf.h
../build/debug/stream_manager.o: /usr/include/bits/nan.h
../build/debug/stream_manager.o: /usr/include/bits/mathdef.h
../build/debug/stream_manager.o: /usr/include/bits/mathcalls.h
../build/debug/stream_manager.o: /usr/include/assert.h cuda-sim/ptx.tab.h
../build/debug/stream_manager.o: gpgpu-sim/delayqueue.h intersim/statwraper.h
../build/debug/stream_manager.o: gpgpu-sim/gpu-misc.h gpgpu-sim/stack.h
../build/debug/stream_manager.o: gpgpu-sim/dram.h /usr/include/zlib.h
../build/debug/stream_manager.o: /usr/include/zconf.h /usr/include/unistd.h
../build/debug/stream_manager.o: /usr/include/bits/posix_opt.h
../build/debug/stream_manager.o: /usr/include/bits/confname.h
../build/debug/stream_manager.o: /usr/include/getopt.h gpgpu-sim/scoreboard.h
../build/debug/stream_manager.o: gpgpu-sim/mem_fetch.h gpgpu-sim/addrdec.h
../build/debug/stream_manager.o: option_parser.h
../build/debug/stream_manager.o: gpgpu-sim/mem_fetch_status.tup
../build/debug/stream_manager.o: gpgpu-sim/stats.h gpgpu-sim/gpu-cache.h
../build/debug/stream_manager.o: tr1_hash_map.h cuda-sim/ptx_sim.h
../build/debug/stream_manager.o: cuda-sim/opcodes.h cuda-sim/opcodes.def
../build/debug/stream_manager.o: cuda-sim/memory.h gpgpu-sim/gpu-sim.h
../build/debug/stream_manager.o: gpgpu-sim/shader.h
