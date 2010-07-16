################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= laplace
# CUDA source files (compiled with cudacc)
CUFILES		:= laplace3d.cu
# CUDA dependency files
CU_DEPS		:= \
	laplace3d_kernel.cu \

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	laplace3d_gold.cpp \

GPGPUSIM_ROOT  := ../../..

################################################################################
# Rules and targets

include ../../../common/common.mk

