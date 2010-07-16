# Copyright (c) 2009 by Tor M. Aamodt, Ali Bakhoda and the 
# University of British Columbia
# Vancouver, BC  V6T 1Z4
# All Rights Reserved.
#
# Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
# George L. Yuan and the University of British Columbia, Vancouver, 
# BC V6T 1Z4, All Rights Reserved.
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
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
# src/intersim/ are derived from Booksim (a simulator provided with the 
# textbook "Principles and Practices of Interconnection Networks" available 
# from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# which is distributed seperately by NVIDIA under separate terms and 
# conditions.  To use this version of GPGPU-Sim with OpenCL requires a
# recent version of NVIDIA's drivers which support OpenCL.
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



# comment out next line to disable OpenGL support
# export OPENGL_SUPPORT=1

export DEBUG?=0 
export SNOW?=

LIBS = cuda-sim gpgpu-sim_uarch intersim gpgpusimlib

TARGETS =
ifeq ($(shell uname),Linux)
	TARGETS += lib/libcudart.so
else
	TARGETS += lib/libcudart.dylib
endif
ifneq  ($(NVOPENCL_LIBDIR),)
	TARGETS += lib/libOpenCL.so
endif

gpgpusim: $(TARGETS)

lib/libcudart.so: $(LIBS) cudalib
	if [ ! -d lib ]; then mkdir lib; fi;
	g++ $(SNOW) -shared -Wl,-soname,libcudart.so \
			./libcuda/*.o \
			./src/cuda-sim/*.o \
			./src/gpgpu-sim/*.o \
			./src/intersim/*.o \
			./src/*.o -lm -lz -lGL \
			-o lib/libcudart.so
	if [ ! -f lib/libcudart.so.2 ]; then ln -s libcudart.so lib/libcudart.so.2; fi

lib/libcudart.dylib: $(LIBS) cudalib
	if [ ! -d lib ]; then mkdir lib; fi;
	g++ $(SNOW) -dynamiclib -Wl,-headerpad_max_install_names,-undefined,dynamic_lookup,-compatibility_version,1.1,-current_version,1.1\
			./libcuda/*.o \
			./src/cuda-sim/*.o \
			./src/gpgpu-sim/*.o \
			./src/intersim/*.o \
			./src/*.o -lm -lz \
			-o lib/libcudart.dylib

lib/libOpenCL.so: $(LIBS) opencllib
	g++ $(SNOW) -shared -Wl,-soname,libOpenCL.so \
			./libopencl/*.o \
			./src/cuda-sim/*.o \
			./src/gpgpu-sim/*.o \
			./src/intersim/*.o \
			./src/*.o -lm -lz -lGL \
			-o lib/libOpenCL.so 
	if [ ! -f lib/libOpenCL.so.1 ]; then ln -s libOpenCL.so lib/libOpenCL.so.1; fi
	if [ ! -f lib/libOpenCL.so.1.1 ]; then ln -s libOpenCL.so lib/libOpenCL.so.1.1; fi

cudalib:
	make -e -C ./libcuda/

cuda-sim:
	make -C ./src/cuda-sim/ depend
	make -C ./src/cuda-sim/

gpgpu-sim_uarch:
	make -C ./src/gpgpu-sim/ depend
	make -C ./src/gpgpu-sim/

intersim:
	make "CREATELIBRARY=1" "DEBUG=$(DEBUG)" -C ./src/intersim	

gpgpusimlib:
	make -C ./src/ depend
	make -C ./src/

opencllib:
	make -e -C ./libopencl/

bench:
	make -C ./benchmarks/CUDA/BlackScholes
	make -C ./benchmarks/CUDA/template
	if [ -f ./benchmarks/Makefile ]; then make -C ./benchmarks/; fi
	
all:
	make gpgpusim
	make bench

clean: 
	make cleangpgpusim
	make cleanbench 

cleangpgpusim:
	make clean -C ./libcuda/
ifneq  ($(NVOPENCL_LIBDIR),)
	make clean -C ./libopencl/
endif
	make clean -C ./src/intersim/
	make clean -C ./src/cuda-sim/
	make clean -C ./src/gpgpu-sim/
	make clean -C ./src/
	rm -rf ./lib/*.so*

cleanbench:
	make clean -C ./benchmarks/CUDA/BlackScholes
	make clean -C ./benchmarks/CUDA/template
	if [ -f ./benchmarks/Makefile ]; then make clean -C ./benchmarks/; fi
