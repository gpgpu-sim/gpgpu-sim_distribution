
OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/gpuwattch/cacti
TARGET = cacti
SHELL = /bin/sh
.PHONY: all depend clean
.SUFFIXES: .cc .o

ifndef NTHREADS
  NTHREADS = 8
endif

LIBS = 
INCS = -lm

ifeq ($(TAG),dbg)
  DBG = -Wall 
  OPT = -ggdb -g -O0 -DNTHREADS=1  -gstabs+
else
  DBG = 
  OPT = -O3  -DNTHREADS=$(NTHREADS)
endif
LDFLAGS += -lc++
#CXXFLAGS = -Wall -Wno-unknown-pragmas -Winline $(DBG) $(OPT) 
CXXFLAGS += -Wno-unknown-pragmas $(DBG) $(OPT) 

SRCS  = area.cc bank.cc mat.cc main.cc Ucache.cc io.cc technology.cc basic_circuit.cc parameter.cc \
		decoder.cc component.cc uca.cc subarray.cc wire.cc htree2.cc \
		cacti_interface.cc router.cc nuca.cc crossbar.cc arbiter.cc 

OBJS = $(patsubst %.cc,$(OUTPUT_DIR)/%.o,$(SRCS))
PYTHONLIB_SRCS = $(patsubst main.cc, ,$(SRCS)) $(OUTPUT_DIR)/cacti_wrap.cc
PYTHONLIB_OBJS = $(patsubst %.cc,%.o,$(PYTHONLIB_SRCS)) 
INCLUDES       = -I /usr/include/python2.4 -I /usr/lib/python2.4/config

all: $(OUTPUT_DIR)/$(TARGET)

$(OUTPUT_DIR)/$(TARGET) : $(OBJS)
	$(CXX) $(OBJS) -o $@ $(INCS) $(LDFLAGS) $(LIBS) -pthread

#obj_$(TAG)/%.o : %.cc
#	$(CXX) -c $(CXXFLAGS) $(INCS) -o $@ $<

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(SRCS) 2> /dev/null

$(OUTPUT_DIR)/%.o : %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f *.o _cacti.so cacti.py $(TARGET)

include $(OUTPUT_DIR)/Makefile.makedepend
