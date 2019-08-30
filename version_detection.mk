# Copyright (c) 2009-2011, Tor M. Aamodt
# Wilson W.L. Fung, Ali Bakhoda
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

# Detect GPGPU-Sim Version
ifeq ($(GPGPUSIM_ROOT),)
else
GPGPUSIM_VERSION=$(shell cat $(GPGPUSIM_ROOT)/version | awk '/Version/ {print $$8}' )

#Detect Git branch and commit #
GIT_COMMIT := $(shell git log -n 1 | head -1 | sed -re 's/commit (.*)/\1/')
GIT_FILES_CHANGED_A:=$(shell git diff --numstat | wc | sed -re 's/^\s+([0-9]+).*/\1./')
GIT_FILES_CHANGED:= $(GIT_FILES_CHANGED_A)$(shell git diff --numstat --cached | wc | sed -re 's/^\s+([0-9]+).*/\1/')
GPGPUSIM_BUILD := gpgpu-sim_git-commit-$(GIT_COMMIT)_modified_$(GIT_FILES_CHANGED)
endif

# Detect CUDA Runtime Version 
CUDA_VERSION_STRING:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
CUDART_VERSION:=$(shell echo $(CUDA_VERSION_STRING) | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

# Detect GCC Version 
CC_VERSION := $(shell gcc --version | head -1 | awk '{for(i=1;i<=NF;i++){ if(match($$i,/^[0-9]\.[0-9]\.[0-9]$$/))  {print $$i; exit 0 }}}')

# Detect Support for C++11 (C++0x) from GCC Version 
GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')
