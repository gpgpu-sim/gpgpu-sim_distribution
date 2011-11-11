# Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Timothy Rogers, 
# Jimmy Kwa, and The University of British Columbia
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
export SNOW?=

ifeq ($(GPGPUSIM_CONFIG), debug)
	export DEBUG=1
else
	export DEBUG=0
endif

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

ifeq  ($(NVOPENCL_LIBDIR),)
	TARGETS += no_opencl_support
else ifeq ($(NVOPENCL_INCDIR),)
	TARGETS += no_opencl_support
else
	TARGETS += $(SIM_LIB_DIR)/libOpenCL.so
endif
	TARGETS += decuda_to_ptxplus/decuda_to_ptxplus
	TARGETS += decuda

gpgpusim: makedirs $(TARGETS)

no_opencl_support:
	@echo "Warning: gpgpu-sim is building without opencl support. Make sure NVOPENCL_LIBDIR and NVOPENCL_INCDIR are set"

$(SIM_LIB_DIR)/libcudart.so: $(LIBS) cudalib
	g++ $(SNOW) -shared -Wl,-soname,libcudart.so \
			$(SIM_OBJ_FILES_DIR)/libcuda/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/intersim/*.o \
            $(SIM_OBJ_FILES_DIR)/*.o -lm -lz -lGL -pthread \
			-o $(SIM_LIB_DIR)/libcudart.so
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.2 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.2; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.3 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.3; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.4 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.4; fi

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
			$(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table/*.o \
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

docs:
	$(MAKE) -C doc/doxygen/

cleandocs:
	$(MAKE) clean -C doc/doxygen/

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
