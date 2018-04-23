
OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/gpuwattch
TARGET = mcpat
SHELL = /bin/sh
.PHONY: all depend clean
.SUFFIXES: .cc .o

ifndef NTHREADS
  NTHREADS = 4
endif


INCS = -I/usr/lib/ -I/usr/lib64/
LIBS = -lm -lz

ifeq ($(TAG),dbg)
  DBG = -Wall 
  OPT = -ggdb -fPIC -g -O0 -DNTHREADS=1 -Icacti
else
  DBG = 
  OPT = -O3 -fPIC -DNTHREADS=$(NTHREADS) -Icacti
  #OPT = -O0 -DNTHREADS=$(NTHREADS)
endif

#CXXFLAGS = -Wall -Wno-unknown-pragmas -Winline $(DBG) $(OPT) 
CXXFLAGS += -Wno-unknown-pragmas $(DBG) $(OPT) 

VPATH = cacti

SRCS  = \
  Ucache.cc \
  XML_Parse.cc \
  arbiter.cc \
  area.cc \
  array.cc \
  bank.cc \
  basic_circuit.cc \
  basic_components.cc \
  cacti_interface.cc \
  component.cc \
  core.cc \
  crossbar.cc \
  decoder.cc \
  htree2.cc \
  interconnect.cc \
  io.cc \
  iocontrollers.cc \
  logic.cc \
  main.cc \
  mat.cc \
  memoryctrl.cc \
  noc.cc \
  nuca.cc \
  parameter.cc \
  processor.cc \
  router.cc \
  sharedcache.cc \
  subarray.cc \
  technology.cc \
  uca.cc \
  wire.cc \
  xmlParser.cc \
  gpgpu_sim_wrapper.cc \



OBJS = $(patsubst %.cc,$(OUTPUT_DIR)/%.o,$(SRCS))

all: $(OUTPUT_DIR)/$(TARGET)

$(OUTPUT_DIR)/$(TARGET) : $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS) -pthread

#obj_$(TAG)/%.o : %.cc
#	$(CXX) -c $(CXXFLAGS) $(INCS) -o $@ $<

$(OUTPUT_DIR)/%.o : %.cc
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(SRCS) 2> /dev/null
	$(MAKE) -C ./cacti/ depend

clean:
	-rm -f *.o $(TARGET)
	rm -f Makefile.makedepend Makefile.makedepend.bak

include $(OUTPUT_DIR)/Makefile.makedepend
