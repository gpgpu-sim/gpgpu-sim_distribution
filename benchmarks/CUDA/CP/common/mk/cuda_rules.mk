# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related rules common to all benchmarks

########################################
# Derived variables
########################################

CUDAOBJS = $(call INBUILDDIR,$(SRCDIR_CUDAOBJS))

########################################
# Rules
########################################

ifeq ("$(LINK_MODE)", "CUDA")
$(BIN) : $(OBJS) $(CUDAOBJS)
	$(CUDACC) $(CUDALDFLAGS) $^ -o $@ $(CUDALIBS)
endif

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(CUDACC) $(CUDACFLAGS) -c $< -o $@

$(BUILDDIR)/%.ptx : $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(CUDACC) $(CUDACFLAGS) -ptx $< -o $@

$(BUILDDIR)/%.cubin : $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	$(CUDACC) $(CUDACFLAGS) -cubin $< -o $@
