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

# Specify INTERSIM folder, the folder will be located at $GPGPUSIM_ROOT/src/$INTERSIM
INTERSIM ?= intersim2

include version_detection.mk

ifeq ($(GPGPUSIM_CONFIG), gcc-$(CC_VERSION)/cuda-$(CUDART_VERSION)/debug)
	export DEBUG=1
else
	export DEBUG=0
endif

BUILD_ROOT?=$(shell pwd)
export TRACE?=1

NVCC_PATH=$(shell which nvcc)
ifneq ($(shell which nvcc), "")
	ifeq ($(DEBUG), 1)
		export SIM_LIB_DIR=lib/gcc-$(CC_VERSION)/cuda-$(CUDART_VERSION)/debug
		export SIM_OBJ_FILES_DIR=$(BUILD_ROOT)/build/gcc-$(CC_VERSION)/cuda-$(CUDART_VERSION)/debug
	else
		export SIM_LIB_DIR=lib/gcc-$(CC_VERSION)/cuda-$(CUDART_VERSION)/release
		export SIM_OBJ_FILES_DIR=$(BUILD_ROOT)/build/gcc-$(CC_VERSION)/cuda-$(CUDART_VERSION)/release
	endif
endif


$(shell mkdir -p $(SIM_OBJ_FILES_DIR)/libcuda && echo "const char *g_gpgpusim_build_string=\"$(GPGPUSIM_BUILD)\";" > $(SIM_OBJ_FILES_DIR)/detailed_version)

LIBS = cuda-sim gpgpu-sim_uarch $(INTERSIM) gpgpusimlib


TARGETS =
ifeq ($(shell uname),Linux)
	TARGETS += $(SIM_LIB_DIR)/libcudart.so 
else # MAC
	TARGETS += $(SIM_LIB_DIR)/libcudart.dylib
endif

ifeq  ($(NVOPENCL_LIBDIR),)
	TARGETS += no_opencl_support
else ifeq ($(NVOPENCL_INCDIR),)
	TARGETS += no_opencl_support
else
	TARGETS += $(SIM_LIB_DIR)/libOpenCL.so
endif
	TARGETS += cuobjdump_to_ptxplus/cuobjdump_to_ptxplus

MCPAT=
MCPAT_OBJ_DIR=
MCPAT_DBG_FLAG=
ifneq ($(GPGPUSIM_POWER_MODEL),)
	LIBS += mcpat


	ifeq ($(DEBUG), 1)
		MCPAT_DBG_FLAG = dbg
	endif

	MCPAT_OBJ_DIR = $(SIM_OBJ_FILES_DIR)/gpuwattch

	MCPAT = $(MCPAT_OBJ_DIR)/*.o
endif


.PHONY: check_setup_environment check_power
gpgpusim: check_setup_environment check_power makedirs $(TARGETS)


check_setup_environment:
	 @if [ ! -n "$(GPGPUSIM_ROOT)" -o ! -n "$(CUDA_INSTALL_PATH)" -o ! -n "$(GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN)" ]; then \
		echo "ERROR *** run 'source setup_environment' before 'make'; please see README."; \
		exit 101; \
	 else \
		NVCC_PATH=`which nvcc`; \
		if [ $$? = 1 ]; then \
			echo ""; \
			echo "ERROR ** nvcc (from CUDA Toolkit) was not found in PATH but required to build GPGPU-Sim."; \
			echo "         Try adding $(CUDA_INSTALL_PATH)/bin/ to your PATH environment variable."; \
			echo "         Please also be sure to read the README file if you have not done so."; \
			echo ""; \
			exit 102; \
		else \
			echo; echo "	Building GPGPU-Sim version $(GPGPUSIM_VERSION) (build $(GPGPUSIM_BUILD)) with CUDA version $(CUDA_VERSION_STRING)"; echo; \
	 		true; \
		fi \
	 fi 

check_power:
	@if [ -d "$(GPGPUSIM_ROOT)/src/gpuwattch/" -a ! -n "$(GPGPUSIM_POWER_MODEL)" ]; then \
		echo ""; \
		echo "	Power model detected in default directory ($(GPGPUSIM_ROOT)/src/gpuwattch) but GPGPUSIM_POWER_MODEL not set."; \
		echo "	Please re-run setup_environment or manually set GPGPUSIM_POWER_MODEL to the gpuwattch directory if you would like to include the GPGPU-Sim Power Model."; \
		echo ""; \
		true; \
	elif [ ! -d "$(GPGPUSIM_POWER_MODEL)" ]; then \
		echo ""; \
		echo "ERROR ** Power model directory invalid."; \
		echo "($(GPGPUSIM_POWER_MODEL)) is not a valid directory."; \
		echo "Please set GPGPUSIM_POWER_MODEL to the GPGPU-Sim gpuwattch directory."; \
		echo ""; \
		exit 101; \
	elif [ -n "$(GPGPUSIM_POWER_MODEL)" -a ! -f "$(GPGPUSIM_POWER_MODEL)/gpgpu_sim.verify" ]; then \
		echo ""; \
		echo "ERROR ** Power model directory invalid."; \
		echo "gpgpu_sim.verify not found in $(GPGPUSIM_POWER_MODEL)."; \
		echo "Please ensure that GPGPUSIM_POWER_MODEL points to a valid gpuwattch directory and that you have the correct GPGPU-Sim mcpat distribution."; \
		echo ""; \
		exit 102; \
	fi

no_opencl_support:
	@echo "Warning: gpgpu-sim is building without opencl support. Make sure NVOPENCL_LIBDIR and NVOPENCL_INCDIR are set"

$(SIM_LIB_DIR)/libcudart.so: makedirs $(LIBS) cudalib
	g++ -shared -Wl,-soname,libcudart.so -Wl,--version-script=linux-so-version.txt\
			$(SIM_OBJ_FILES_DIR)/libcuda/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/$(INTERSIM)/*.o \
			$(SIM_OBJ_FILES_DIR)/*.o -lm -lz -lGL -pthread \
			$(MCPAT) \
			-o $(SIM_LIB_DIR)/libcudart.so
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.2 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.2; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.3 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.3; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.4 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.4; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.5.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.5.0; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.5.5 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.5.5; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.6.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.6.0; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.6.5 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.6.5; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.7.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.7.0; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.7.5 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.7.5; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.8.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.8.0; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.9.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.9.0; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.9.1 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.9.1; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.9.2 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.9.2; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.10.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.10.0; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.10.1 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.10.1; fi
	if [ ! -f $(SIM_LIB_DIR)/libcudart.so.11.0 ]; then ln -s libcudart.so $(SIM_LIB_DIR)/libcudart.so.11.0; fi

$(SIM_LIB_DIR)/libcudart.dylib: makedirs $(LIBS) cudalib
	g++ -dynamiclib -Wl,-headerpad_max_install_names,-undefined,dynamic_lookup,-compatibility_version,1.1,-current_version,1.1\
			$(SIM_OBJ_FILES_DIR)/libcuda/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/$(INTERSIM)/*.o  \
			$(SIM_OBJ_FILES_DIR)/*.o -lm -lz -pthread \
			$(MCPAT) \
			-o $(SIM_LIB_DIR)/libcudart.dylib

$(SIM_LIB_DIR)/libOpenCL.so: makedirs $(LIBS) opencllib
	g++ -shared -Wl,-soname,libOpenCL_$(GPGPUSIM_BUILD).so \
			$(SIM_OBJ_FILES_DIR)/libopencl/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table/*.o \
			$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o \
			$(SIM_OBJ_FILES_DIR)/$(INTERSIM)/*.o \
			$(SIM_OBJ_FILES_DIR)/*.o -lm -lz -lGL -pthread \
			$(MCPAT) \
			-o $(SIM_LIB_DIR)/libOpenCL.so 
	if [ ! -f $(SIM_LIB_DIR)/libOpenCL.so.1 ]; then ln -s libOpenCL.so $(SIM_LIB_DIR)/libOpenCL.so.1; fi
	if [ ! -f $(SIM_LIB_DIR)/libOpenCL.so.1.1 ]; then ln -s libOpenCL.so $(SIM_LIB_DIR)/libOpenCL.so.1.1; fi

cudalib: makedirs cuda-sim
	$(MAKE) -C ./libcuda/ depend
	$(MAKE) -C ./libcuda/

ifneq ($(GPGPUSIM_POWER_MODEL),)
mcpat: makedirs
	$(MAKE) -C $(GPGPUSIM_POWER_MODEL) depend
	$(MAKE) -C $(GPGPUSIM_POWER_MODEL) $(MCPAT_DBG_FLAG)
endif

cuda-sim: makedirs
	$(MAKE) -C ./src/cuda-sim/ depend
	$(MAKE) -C ./src/cuda-sim/

TFLAGS = -std=c++0x -I$(CUDA_INSTALL_PATH)/include
ifneq ($(DEBUG),1)
	TFLAGS += -O3
endif
TFLAGS += -g3 -fPIC

gpgpu-sim_uarch: makedirs cuda-sim
	$(MAKE) -C ./src/gpgpu-sim/ depend
	$(MAKE) -C ./src/gpgpu-sim/

$(INTERSIM): makedirs cuda-sim gpgpu-sim_uarch
	$(MAKE) "CREATE_LIBRARY=1" "DEBUG=$(DEBUG)" -C ./src/$(INTERSIM)

gpgpusimlib: makedirs cuda-sim gpgpu-sim_uarch $(INTERSIM)
	$(MAKE) -C ./src/ depend
	$(MAKE) -C ./src/

opencllib: makedirs cuda-sim
	$(MAKE) -C ./libopencl/ depend
	$(MAKE) -C ./libopencl/

.PHONY: cuobjdump_to_ptxplus/cuobjdump_to_ptxplus
cuobjdump_to_ptxplus/cuobjdump_to_ptxplus: cuda-sim makedirs
	$(MAKE) -C ./cuobjdump_to_ptxplus/ depend
	$(MAKE) -C ./cuobjdump_to_ptxplus/

makedirs:
	if [ ! -d $(SIM_LIB_DIR) ]; then mkdir -p $(SIM_LIB_DIR); fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/libcuda ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/libcuda; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/cuda-sim ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/cuda-sim; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/cuda-sim/decuda_pred_table; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/gpgpu-sim ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/gpgpu-sim; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/libopencl ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/libopencl; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/libopencl/bin ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/libopencl/bin; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/$(INTERSIM) ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/$(INTERSIM); fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/cuobjdump_to_ptxplus ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/cuobjdump_to_ptxplus; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/gpuwattch ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/gpuwattch; fi;
	if [ ! -d $(SIM_OBJ_FILES_DIR)/gpuwattch/cacti ]; then mkdir -p $(SIM_OBJ_FILES_DIR)/gpuwattch/cacti; fi;

all:
	$(MAKE) gpgpusim

docs:
	$(MAKE) -C doc/doxygen/

cleandocs:
	$(MAKE) clean -C doc/doxygen/

clean: makedirs
	$(MAKE) cleangpgpusim

cleangpgpusim: cleandocs
	rm -rf $(SIM_LIB_DIR)
	rm -rf $(SIM_OBJ_FILES_DIR)
	rm -f *~ *.o ./src/cuda-sim/*.gcda ./src/cuda-sim/*.gcno ./src/cuda-sim/*.gcov ./src/cuda-sim/libgpgpu_ptx_sim.a \
                ./src/cuda-sim/ptx.tab.h ./src/cuda-sim/ptx.tab.c ./src/cuda-sim/ptx.output ./src/cuda-sim/lex.ptx_.c \
                ./src/cuda-sim/ptxinfo.tab.h ./src/cuda-sim/ptxinfo.tab.c ./src/cuda-sim/ptxinfo.output ./src/cuda-sim/lex.ptxinfo_.c \
                ./src/cuda-sim/instructions.h ./src/cuda-sim/ptx_parser_decode.def ./src/cuda-sim/directed_tests.log \
		./cuobjdump_to_ptxplus/elf_lexer.cc ./cuobjdump_to_ptxplus/elf_parser.cc ./cuobjdump_to_ptxplus/elf_parser.hh \
		./cuobjdump_to_ptxplus/header_lexer.cc ./cuobjdump_to_ptxplus/header_parser.cc ./cuobjdump_to_ptxplus/header_parser.hh \
		./cuobjdump_to_ptxplus/lex.ptx_.c ./cuobjdump_to_ptxplus/ptx.output ./cuobjdump_to_ptxplus/ptx.tab.h cuobjdump_to_ptxplus/ptx.tab.h \
		./cuobjdump_to_ptxplus/sass_lexer.cc ./cuobjdump_to_ptxplus/sass_parser.cc ./cuobjdump_to_ptxplus/sass_parser.hh \
		./cuobjdump_to_ptxplus/ptx.tab.cc ./cuobjdump_to_ptxplus/*.output

