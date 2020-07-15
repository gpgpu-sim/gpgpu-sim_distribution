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
TRACE?=1

include ../version_detection.mk

CXXFLAGS = -Wall -DDEBUG
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
endif

ifeq ($(TRACE),1)
	CXXFLAGS += -DTRACING_ON=1
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS += 
endif

CXXFLAGS += -I$(CUDA_INSTALL_PATH)/include

OPTFLAGS += -g3 -fPIC

CPP = g++ $(SNOW)
OEXT = o

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)
SRCS = $(shell ls *.cc)
OBJS = $(SRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))

$(OUTPUT_DIR)/libgpgpusim.a:	$(OBJS) gpu_uarch_simlib
	ar rcs  $(OUTPUT_DIR)/libgpgpusim.a $(OBJS) $(OUTPUT_DIR)/gpgpu-sim/*.o

gpu_uarch_simlib:
	make   -C ./gpgpu-sim
	
$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(SRCS) 2> /dev/null

clean:
	rm -f *.o core *~ *.a Makefile.makedepend Makefile.makedepend.bak

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $(OUTPUT_DIR)/$*.$(OEXT) -c $*.cc

option_parser.$(OEXT): option_parser.h

include $(OUTPUT_DIR)/Makefile.makedepend

