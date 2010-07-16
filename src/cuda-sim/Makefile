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

OBJS	:= instructions.o cuda-sim.o ptx_ir.o ptx_sim.o memory.o ptx-stats.o ptx.tab.o lex.ptx_.o ptxinfo.tab.o lex.ptxinfo_.o

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

ptx.tab.c:
	bison --name-prefix=ptx_ -v -d ptx.y

ptxinfo.tab.c:
	bison --name-prefix=ptxinfo_ -v -d ptxinfo.y

lex.ptx_.c:
	flex ptx.l

lex.ptxinfo_.c:
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
ptx_ir.o: ptx.tab.c ptx_parser_decode.def

# DO NOT DELETE

