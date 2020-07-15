# Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
# Timothy G. Rogers
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
TRACE?=0

ifeq ($(DEBUG),1)
	CXXFLAGS = -Wall -DDEBUG
else
	CXXFLAGS = -Wall
endif

ifeq ($(TRACE),1)
	CXXFLAGS += -DTRACING_ON=1
endif

include ../../version_detection.mk

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS += 
endif

CXXFLAGS += -I$(CUDA_INSTALL_PATH)/include

POWER_FLAGS=
ifneq ($(GPGPUSIM_POWER_MODEL),)
	POWER_FLAGS = -I$(GPGPUSIM_POWER_MODEL) -DGPGPUSIM_POWER_MODEL
endif

OPTFLAGS += -g3 -fPIC
OPTFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

CPP = g++ $(SNOW)
OEXT = o

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/gpgpu-sim

SRCS = $(shell ls *.cc)

EXCLUDES = 

ifeq ($(GPGPUSIM_POWER_MODEL), )
EXCLUDES += power_interface.cc
endif

CSRCS = $(filter-out $(EXCLUDES), $(SRCS))

OBJS = $(CSRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))
 
libgpu_uarch_sim.a:$(OBJS)
	ar rcs  $(OUTPUT_DIR)/libgpu_uarch_sim.a $(OBJS)

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(CSRCS) 2> /dev/null

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $(OUTPUT_DIR)/$*.$(OEXT) -c $*.cc

clean:
	rm -f *.o core *~ *.a 
	rm -f Makefile.makedepend Makefile.makedepend.bak

$(OUTPUT_DIR)/option_parser.$(OEXT): option_parser.h

$(OUTPUT_DIR)/dram_sched.$(OEXT): $(OUTPUT_DIR)/../cuda-sim/ptx.tab.h

$(OUTPUT_DIR)/../cuda-sim/ptx.tab.h:
	make -C ../cuda-sim/ $(OUTPUT_DIR)/../cuda-sim/ptx.tab.c

include $(OUTPUT_DIR)/Makefile.makedepend

