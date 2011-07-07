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

ifeq ($(DEBUG), 1)
	export SIM_LIB_DIR=lib/debug
	export SIM_OBJ_FILES_DIR=build/debug
else
	export SIM_LIB_DIR=lib/release
	export SIM_OBJ_FILES_DIR=build/release
endif

LIBS = cuda-sim gpgpu-sim_uarch intersim gpgpusimlib

TARGETS =
ifeq ($(shell uname),Linux)
	TARGETS += $(SIM_LIB_DIR)/libcudart.so
else
	TARGETS += $(SIM_LIB_DIR)/libcudart.dylib
endif
ifneq  ($(NVOPENCL_LIBDIR),)
	TARGETS += $(SIM_LIB_DIR)/libOpenCL.so
endif
	TARGETS += decuda_to_ptxplus/decuda_to_ptxplus
	TARGETS += decuda

gpgpusim: makedirs $(TARGETS)

$(SIM_LIB_DIR)/libcudart.so: $(LIBS) cudalib
	g++ $(SNOW) -shared -Wl,-soname,libcudart.so \
			$(SIM_OBJ_FILES_DIR)/libcuda/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/intersim/*.o \
            $(SIM_OBJ_FILES_DIR)/*.o -lm -lz -lGL -pthread \
			-o $(SIM_LIB_DIR)/libcudart.so
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.2 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.2; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.3 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.3; fi

$(SIM_LIB_DIR)/libcudart.dylib: $(LIBS) cudalib
	g++ $(SNOW) -dynamiclib -Wl,-headerpad_max_install_names,-undefined,dynamic_lookup,-compatibility_version,1.1,-current_version,1.1\
			$(SIM_OBJ_FILES_DIR)/libcuda/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/intersim/*.o \
			$(SIM_OBJ_FILES_DIR)/*.o -lm -lz -pthread \
			-o $(SIM_LIB_DIR)/libcudart.dylib

$(SIM_LIB_DIR)/libOpenCL.so: $(LIBS) opencllib
	g++ $(SNOW) -shared -Wl,-soname,libOpenCL.so \
			$(SIM_OBJ_FILES_DIR)/libopencl/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/intersim/*.o \
			$(SIM_OBJ_FILES_DIR)/*.o -lm -lz -lGL -pthread \
			-o $(SIM_LIB_DIR)/libOpenCL.so 
	if [ ! -f $(SIM_LIB_DIR)/libOpenCL.so.1 ]; then ln -s libOpenCL.so $(SIM_LIB_DIR)/libOpenCL.so.1; fi
	if [ ! -f $(SIM_LIB_DIR)/libOpenCL.so.1.1 ]; then ln -s libOpenCL.so $(SIM_LIB_DIR)/libOpenCL.so.1.1; fi

cudalib: cuda-sim
	$(MAKE) -C ./libcuda/ depend
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
	$(MAKE) -C ./libopencl/ depend
	$(MAKE) -C ./libopencl/

decuda_to_ptxplus/decuda_to_ptxplus:
	$(MAKE) -C ./decuda_to_ptxplus/ depend 
	$(MAKE) -C ./decuda_to_ptxplus/

decuda:
	./getDecuda/getDecuda.sh

makedirs:
	if [ ! -d $(SIM_LIB_DIR) ]; then mkdir -p $(SIM_LIB_DIR); fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/libcuda ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/libcuda; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/cuda-sim ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/cuda-sim; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/gpgpu-sim ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/gpgpu-sim; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/libopencl ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/libopencl; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/intersim ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/intersim; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/decuda_to_ptxplus ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/decuda_to_ptxplus; fi;

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
	rm -rf $(SIM_LIB_DIR)
	rm -rf $(SIM_OBJ_FILES_DIR)
