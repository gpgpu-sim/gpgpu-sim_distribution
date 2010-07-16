# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# Programs
CUDACC=nvcc
# Paths
# Flags
CUDACFLAGS=$(INCLUDEFLAGS) -g -Xcompiler "-m32" $(EXTRA_CUDACFLAGS)
CUDALDFLAGS=$(LDFLAGS) -Xcompiler "-m32"	\
	-L$(GPGPUSIM_ROOT)/lib/ \
	-L$(PARBOIL_ROOT)/common/src $(EXTRA_CUDALDFLAGS)
CUDALIBS=-lcudart $(LIBS) -lm -lz -lGL
