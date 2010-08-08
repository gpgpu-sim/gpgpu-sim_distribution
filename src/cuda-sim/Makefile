# Copyright (c) 2009 by Tor M. Aamodt and the University of British Columbia
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
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files 
# src/gpgpusim_entrypoint.c and src/simplesim-3.0/ are derived from the 
# SimpleScalar Toolset available from http://www.simplescalar.com/ 
# (property of SimpleScalar LLC) and the files src/intersim/ are derived 
# from Booksim (Simulator provided with the textbook "Principles and 
# Practices of Interconnection Networks" available from 
# http://cva.stanford.edu/books/ppin/).  As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA
# which is distributed seperately by NVIDIA under separate terms and
# conditions.
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
#  including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4

default: libgpgpu_ptx_sim.a 
 
INTEL=0
DEBUG?=0

CPP = g++ $(SNOW)
CC = gcc $(SNOW)
ifeq ($(INTEL),1)
	CPP = icpc
	CC = icc
endif

CC_VERSION := $(shell gcc --version | perl -ne '/gcc\s+\(.*\)\s+([0-9.]+)/ and print $$1;')
GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')

OPT	:=  -O3 -g3 -Wall -Wno-unused-function -Wno-sign-compare
ifeq ($(DEBUG),1)
	OPT := -g3 -Wall  -Wno-unused-function -Wno-sign-compare
endif
OPT += -I$(CUDAHOME)/include 
OPT += -fPIC 

CXX_OPT = $(OPT)
ifeq ($(INTEL),1)
    CXX_OPT += -std=c++0x
else 
ifeq ($(GNUC_CPP0X),1)
    CXX_OPT += -std=c++0x
endif
endif

OBJS	:= ptx_parser.o ptx_loader.o cuda_device_printf.o instructions.o cuda-sim.o ptx_ir.o ptx_sim.o memory.o ptx-stats.o ptx.tab.o lex.ptx_.o ptxinfo.tab.o lex.ptxinfo_.o

CUDART_VERSION:=$(shell nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//' | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

OPT += -DCUDART_VERSION=$(CUDART_VERSION)

libgpgpu_ptx_sim.a: $(OBJS) 
	ar rcs libgpgpu_ptx_sim.a ptx.tab.o lex.ptx_.o ptxinfo.tab.o lex.ptxinfo_.o $(OBJS)

ptx.tab.o: ptx.tab.c
	$(CC) -c $(OPT) -DYYDEBUG ptx.tab.c

lex.ptx_.o: lex.ptx_.c
	$(CC) -c $(OPT) lex.ptx_.c

ptxinfo.tab.o: ptxinfo.tab.c
	$(CC) -c $(OPT) -DYYDEBUG ptxinfo.tab.c

lex.ptxinfo_.o: lex.ptxinfo_.c
	$(CC) -c $(OPT) lex.ptxinfo_.c

ptx.tab.c: ptx.y
	bison --name-prefix=ptx_ -v -d ptx.y

ptxinfo.tab.c: ptxinfo.y
	bison --name-prefix=ptxinfo_ -v -d ptxinfo.y

lex.ptx_.c: ptx.l
	flex ptx.l

lex.ptxinfo_.c: ptxinfo.l
	flex ptxinfo.l

clean:
	rm -f *~ *.o *.gcda *.gcno *.gcov libgpgpu_ptx_sim.a \
		ptx.tab.h ptx.tab.c ptx.output lex.ptx_.c \
		ptxinfo.tab.h ptxinfo.tab.c ptxinfo.output lex.ptxinfo_.c \
		instructions.h ptx_parser_decode.def directed_tests.log

.c.o:
	$(CC) -c $(OPT) $< -o $*.o
.cc.o:
	$(CPP) -c $(CXX_OPT) $< -o $*.o

instructions.h: instructions.cc
	@touch $*.h
	@chmod +w $*.h
	@echo "// DO NOT EDIT THIS FILE! IT IS AUTOMATICALLY GENERATED BY THE MAKEFILE (see target for instructions.h)" > $*.h
	@echo "#include \"ptx_ir.h\"" >> $*.h
	@echo "#ifndef instructions_h_included" >> $*.h
	@echo "#define instructions_h_included" >> $*.h
	@echo "#ifdef __cplusplus" >> $*.h
	@echo "extern \"C\" {" >> $*.h
	@echo "#endif" >> $*.h
	@cat $< | grep "_impl(" | sed 's/{.*//' | sed 's/$$/;/' >> $*.h
	@echo "#ifdef __cplusplus" >> $*.h
	@echo "}" >> $*.h
	@echo "#endif" >> $*.h
	@echo "#endif" >> $*.h
	@chmod -w $*.h
	@echo "created instructions.h"

ptx_parser_decode.def: ptx.tab.c
ifeq ($(shell uname),Linux)
	cat ptx.tab.h | grep "=" | sed 's/^[ ]\+//' | sed 's/[=,]//g' | sed 's/\([_A-Z1-9]\+\)[ ]\+\([0-9]\+\)/\1 \1/' | sed 's/^/DEF(/' | sed 's/ /,"/' | sed 's/$$/")/' > ptx_parser_decode.def
else
	cat ptx.tab.h | grep "=" | sed -E 's/^ +//' | sed 's/[=,]//g' | sed -E 's/([_A-Z1-9]+).*/\1 \1/' | sed 's/^/DEF(/' | sed 's/ /,"/' | sed 's/$$/")/' > ptx_parser_decode.def
endif

depend:
	makedepend *.cc *.c 2> /dev/null

instructions.o: instructions.h ptx.tab.c
cuda_device_printf.o: ptx.tab.c
ptx_ir.o: ptx.tab.c ptx_parser_decode.def
ptx_loader.o: ptx.tab.c ptx_parser_decode.def
ptx_parser.o: ptx.tab.c ptx_parser_decode.def

# DO NOT DELETE

cuda_device_printf.o: ptx_ir.h ../abstract_hardware_model.h
cuda_device_printf.o: /usr/include/assert.h /usr/include/features.h
cuda_device_printf.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
cuda_device_printf.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
cuda_device_printf.o: ptx_sim.h /usr/include/stdlib.h
cuda_device_printf.o: /usr/include/sys/types.h /usr/include/bits/types.h
cuda_device_printf.o: /usr/include/bits/typesizes.h /usr/include/time.h
cuda_device_printf.o: /usr/include/endian.h /usr/include/bits/endian.h
cuda_device_printf.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
cuda_device_printf.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
cuda_device_printf.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
cuda_device_printf.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
cuda_device_printf.o: dram_callback.h opcodes.h opcodes.def memory.h
cuda_device_printf.o: /usr/include/string.h /usr/include/stdio.h
cuda_device_printf.o: /usr/include/libio.h /usr/include/_G_config.h
cuda_device_printf.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
cuda_device_printf.o: /usr/include/bits/sys_errlist.h ../option_parser.h
cuda-sim.o: cuda-sim.h ../abstract_hardware_model.h /usr/include/stdlib.h
cuda-sim.o: /usr/include/features.h /usr/include/sys/cdefs.h
cuda-sim.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
cuda-sim.o: /usr/include/gnu/stubs-64.h /usr/include/sys/types.h
cuda-sim.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
cuda-sim.o: /usr/include/time.h /usr/include/endian.h
cuda-sim.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
cuda-sim.o: /usr/include/sys/select.h /usr/include/bits/select.h
cuda-sim.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
cuda-sim.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
cuda-sim.o: /usr/include/alloca.h ptx_ir.h /usr/include/assert.h ptx_sim.h
cuda-sim.o: dram_callback.h opcodes.h opcodes.def memory.h
cuda-sim.o: /usr/include/string.h /usr/include/stdio.h /usr/include/libio.h
cuda-sim.o: /usr/include/_G_config.h /usr/include/wchar.h
cuda-sim.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
cuda-sim.o: ../option_parser.h ../intersim/statwraper.h ptx-stats.h
cuda-sim.o: ptx_loader.h ptx_parser.h
instructions.o: ptx_ir.h ../abstract_hardware_model.h /usr/include/assert.h
instructions.o: /usr/include/features.h /usr/include/sys/cdefs.h
instructions.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
instructions.o: /usr/include/gnu/stubs-64.h ptx_sim.h /usr/include/stdlib.h
instructions.o: /usr/include/sys/types.h /usr/include/bits/types.h
instructions.o: /usr/include/bits/typesizes.h /usr/include/time.h
instructions.o: /usr/include/endian.h /usr/include/bits/endian.h
instructions.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
instructions.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
instructions.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
instructions.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
instructions.o: dram_callback.h opcodes.h opcodes.def memory.h
instructions.o: /usr/include/string.h /usr/include/stdio.h
instructions.o: /usr/include/libio.h /usr/include/_G_config.h
instructions.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
instructions.o: /usr/include/bits/sys_errlist.h ../option_parser.h
instructions.o: /usr/include/math.h /usr/include/bits/huge_val.h
instructions.o: /usr/include/bits/mathdef.h /usr/include/bits/mathcalls.h
instructions.o: /usr/include/fenv.h /usr/include/bits/fenv.h cuda-math.h
instructions.o: ptx_loader.h
memory.o: memory.h ../abstract_hardware_model.h /usr/include/assert.h
memory.o: /usr/include/features.h /usr/include/sys/cdefs.h
memory.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
memory.o: /usr/include/gnu/stubs-64.h /usr/include/string.h
memory.o: /usr/include/stdio.h /usr/include/bits/types.h
memory.o: /usr/include/bits/typesizes.h /usr/include/libio.h
memory.o: /usr/include/_G_config.h /usr/include/wchar.h
memory.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
memory.o: /usr/include/stdlib.h /usr/include/sys/types.h /usr/include/time.h
memory.o: /usr/include/endian.h /usr/include/bits/endian.h
memory.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
memory.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
memory.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
memory.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h ../debug.h
memory.o: ../abstract_hardware_model.h
ptx_ir.o: ptx_parser.h ../abstract_hardware_model.h ptx_ir.h
ptx_ir.o: /usr/include/assert.h /usr/include/features.h
ptx_ir.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
ptx_ir.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h ptx_sim.h
ptx_ir.o: /usr/include/stdlib.h /usr/include/sys/types.h
ptx_ir.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
ptx_ir.o: /usr/include/time.h /usr/include/endian.h
ptx_ir.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
ptx_ir.o: /usr/include/sys/select.h /usr/include/bits/select.h
ptx_ir.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
ptx_ir.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
ptx_ir.o: /usr/include/alloca.h dram_callback.h opcodes.h opcodes.def
ptx_ir.o: memory.h /usr/include/string.h /usr/include/stdio.h
ptx_ir.o: /usr/include/libio.h /usr/include/_G_config.h /usr/include/wchar.h
ptx_ir.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
ptx_ir.o: ../option_parser.h cuda-sim.h
ptx_loader.o: ptx_loader.h ptx_ir.h ../abstract_hardware_model.h
ptx_loader.o: /usr/include/assert.h /usr/include/features.h
ptx_loader.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
ptx_loader.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h ptx_sim.h
ptx_loader.o: /usr/include/stdlib.h /usr/include/sys/types.h
ptx_loader.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
ptx_loader.o: /usr/include/time.h /usr/include/endian.h
ptx_loader.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
ptx_loader.o: /usr/include/sys/select.h /usr/include/bits/select.h
ptx_loader.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
ptx_loader.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
ptx_loader.o: /usr/include/alloca.h dram_callback.h opcodes.h opcodes.def
ptx_loader.o: memory.h /usr/include/string.h /usr/include/stdio.h
ptx_loader.o: /usr/include/libio.h /usr/include/_G_config.h
ptx_loader.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
ptx_loader.o: /usr/include/bits/sys_errlist.h ../option_parser.h cuda-sim.h
ptx_loader.o: ptx_parser.h /usr/include/dirent.h /usr/include/bits/dirent.h
ptx_loader.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
ptx_loader.o: /usr/include/linux/limits.h
ptx_parser.o: ptx_parser.h ../abstract_hardware_model.h ptx_ir.h
ptx_parser.o: /usr/include/assert.h /usr/include/features.h
ptx_parser.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
ptx_parser.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h ptx_sim.h
ptx_parser.o: /usr/include/stdlib.h /usr/include/sys/types.h
ptx_parser.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
ptx_parser.o: /usr/include/time.h /usr/include/endian.h
ptx_parser.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
ptx_parser.o: /usr/include/sys/select.h /usr/include/bits/select.h
ptx_parser.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
ptx_parser.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
ptx_parser.o: /usr/include/alloca.h dram_callback.h opcodes.h opcodes.def
ptx_parser.o: memory.h /usr/include/string.h /usr/include/stdio.h
ptx_parser.o: /usr/include/libio.h /usr/include/_G_config.h
ptx_parser.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
ptx_parser.o: /usr/include/bits/sys_errlist.h ../option_parser.h
ptx_sim.o: ptx_sim.h /usr/include/stdlib.h /usr/include/features.h
ptx_sim.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
ptx_sim.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
ptx_sim.o: /usr/include/sys/types.h /usr/include/bits/types.h
ptx_sim.o: /usr/include/bits/typesizes.h /usr/include/time.h
ptx_sim.o: /usr/include/endian.h /usr/include/bits/endian.h
ptx_sim.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
ptx_sim.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
ptx_sim.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
ptx_sim.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
ptx_sim.o: dram_callback.h ../abstract_hardware_model.h /usr/include/assert.h
ptx_sim.o: opcodes.h opcodes.def ptx_ir.h memory.h /usr/include/string.h
ptx_sim.o: /usr/include/stdio.h /usr/include/libio.h /usr/include/_G_config.h
ptx_sim.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
ptx_sim.o: /usr/include/bits/sys_errlist.h ../option_parser.h
ptx-stats.o: ptx_ir.h ../abstract_hardware_model.h /usr/include/assert.h
ptx-stats.o: /usr/include/features.h /usr/include/sys/cdefs.h
ptx-stats.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
ptx-stats.o: /usr/include/gnu/stubs-64.h ptx_sim.h /usr/include/stdlib.h
ptx-stats.o: /usr/include/sys/types.h /usr/include/bits/types.h
ptx-stats.o: /usr/include/bits/typesizes.h /usr/include/time.h
ptx-stats.o: /usr/include/endian.h /usr/include/bits/endian.h
ptx-stats.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
ptx-stats.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
ptx-stats.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
ptx-stats.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
ptx-stats.o: dram_callback.h opcodes.h opcodes.def memory.h
ptx-stats.o: /usr/include/string.h /usr/include/stdio.h /usr/include/libio.h
ptx-stats.o: /usr/include/_G_config.h /usr/include/wchar.h
ptx-stats.o: /usr/include/bits/stdio_lim.h /usr/include/bits/sys_errlist.h
ptx-stats.o: ../option_parser.h ptx-stats.h
