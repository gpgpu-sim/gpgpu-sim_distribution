# (c) 2007 The Board of Trustees of the University of Illinois.

SRCDIR=$(shell pwd)
BUILDDIR=$(shell pwd)/../lib

include $(PARBOIL_ROOT)/common/mk/common.mk
include $(PARBOIL_ROOT)/common/mk/cuda.mk

OBJS=$(BUILDDIR)/parboil.o
TARGET=$(BUILDDIR)/libparboil.a

all : $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) $^ -o $@

$(TARGET) : $(OBJS)
	$(AR) rc $@ $?
	$(RANLIB) $@

