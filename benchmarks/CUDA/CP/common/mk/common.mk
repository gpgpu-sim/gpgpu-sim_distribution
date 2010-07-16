# (c) 2007 The Board of Trustees of the University of Illinois.

# Definitions common to all makefiles

########################################
# Environment variable check
#
# The environment variables BUILDDIR, SRCDIR, and PARBOIL_ROOT
# should be set on the command line to `make'.
########################################

ifeq ("$(SRCDIR)", "")
$(error $$SRCDIR is not set)
endif

ifeq ("$(PARBOIL_ROOT)", "")
$(error $$PARBOIL_ROOT is not set)
endif

########################################
# Variables
########################################

# Programs
CC=gcc
CXX=g++
AR=ar
RANLIB=ranlib

# Command line options
INCLUDEFLAGS=-I$(CUDAHOME)/include -I$(PARBOIL_ROOT)/common/include
CFLAGS=$(GCCSTD) $(INCLUDEFLAGS) -g $(EXTRA_CFLAGS)
CXXFLAGS=$(INCLUDEFLAGS) -g $(EXTRA_CXXFLAGS)
LDFLAGS=-L$(PARBOIL_ROOT)/common/lib $(EXTRA_LDFLAGS)
LIBS=-lparboil $(EXTRA_LIBS)

# Pass an extra source language option to GCC
ifeq ("$(CC)","gcc")
GCCSTD=-std=gnu99
endif


########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))
