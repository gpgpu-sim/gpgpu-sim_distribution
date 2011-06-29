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
	TARGETS += decuda_to_ptxplus/decuda_to_ptxplus
	TARGETS += decuda

gpgpusim: $(TARGETS)

lib/libcudart.so: $(LIBS) cudalib
	if [ ! -d lib ]; then mkdir lib; fi;
	g++ $(SNOW) -shared -Wl,-soname,libcudart.so \
			./libcuda/*.o \
			./src/cuda-sim/*.o \
			./src/cuda-sim/decuda_pred_table/*.o \
			./src/gpgpu-sim/*.o \
			./src/intersim/*.o \
			./src/*.o -lm -lz -lGL -pthread \
			-o lib/libcudart.so
	if [ ! -f lib/libcudart.so.2 ]; then ln -s libcudart.so lib/libcudart.so.2; fi
	if [ ! -f lib/libcudart.so.3 ]; then ln -s libcudart.so lib/libcudart.so.3; fi

lib/libcudart.dylib: $(LIBS) cudalib
	if [ ! -d lib ]; then mkdir lib; fi;
	g++ $(SNOW) -dynamiclib -Wl,-headerpad_max_install_names,-undefined,dynamic_lookup,-compatibility_version,1.1,-current_version,1.1\
			./libcuda/*.o \
			./src/cuda-sim/*.o \
			./src/cuda-sim/decuda_pred_table/*.o \
			./src/gpgpu-sim/*.o \
			./src/intersim/*.o \
			./src/*.o -lm -lz -pthread \
			-o lib/libcudart.dylib

lib/libOpenCL.so: $(LIBS) opencllib
	g++ $(SNOW) -shared -Wl,-soname,libOpenCL.so \
			./libopencl/*.o \
			./src/cuda-sim/*.o \
			./src/cuda-sim/decuda_pred_table/*.o \
			./src/gpgpu-sim/*.o \
			./src/intersim/*.o \
			./src/*.o -lm -lz -lGL -pthread \
			-o lib/libOpenCL.so 
	if [ ! -f lib/libOpenCL.so.1 ]; then ln -s libOpenCL.so lib/libOpenCL.so.1; fi
	if [ ! -f lib/libOpenCL.so.1.1 ]; then ln -s libOpenCL.so lib/libOpenCL.so.1.1; fi

cudalib: cuda-sim
	$(MAKE) -C ./libcuda/

cuda-sim:
	$(MAKE) -C ./src/cuda-sim/ depend
	$(MAKE) -C ./src/cuda-sim/

gpgpu-sim_uarch: cuda-sim
	$(MAKE) -C ./src/gpgpu-sim/ depend
	$(MAKE) -C ./src/gpgpu-sim/

intersim: cuda-sim gpgpu-sim_uarch
	$(MAKE) "CREATELIBRARY=1" "DEBUG=$(DEBUG)" -C ./src/intersim	

gpgpusimlib: cuda-sim gpgpu-sim_uarch intersim
	$(MAKE) -C ./src/ depend
	$(MAKE) -C ./src/

opencllib: cuda-sim
	$(MAKE) -C ./libopencl/

decuda_to_ptxplus/decuda_to_ptxplus: 
	$(MAKE) -C ./decuda_to_ptxplus/

decuda:
	./getDecuda/getDecuda.sh

all:
	$(MAKE) gpgpusim

clean: 
	$(MAKE) cleangpgpusim

cleangpgpusim:
	$(MAKE) clean -C ./libcuda/
ifneq  ($(NVOPENCL_LIBDIR),)
	$(MAKE) clean -C ./libopencl/
endif
	$(MAKE) clean -C ./src/intersim/
	$(MAKE) clean -C ./src/cuda-sim/
	$(MAKE) clean -C ./src/gpgpu-sim/
	$(MAKE) clean -C ./src/
	$(MAKE) clean -C ./decuda_to_ptxplus/
	rm -rf ./lib/*.so*
