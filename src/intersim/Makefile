CREATESHAREDLIB  ?=0
CPP = g++ $(SNOW)
CC = gcc $(SNOW)

ifeq ($(INTEL),1)
	CPP = icpc
	CC = icc
endif


YACC   = bison -d
LEX    = flex
PURIFY = /usr/bin/purify
QUANT  = /usr/bin/quantify

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/intersim

INTERFACE = interconnect_interface.cpp
DEBUG = 0 
CPPFLAGS = -g -Wall
ifneq ($(DEBUG),1)
CPPFLAGS = -O3 -g 
else
CPPFLAGS += 
endif

CPPFLAGS += -I$(OUTPUT_DIR) -I.

TEST = -DUNIT_TEST 
ifneq ($(UNIT_TEST),1)
TEST = 
endif

        CPPFLAGS += -fPIC

PROG     = intersim


CPP_SRCS = $(INTERFACE) \
	config_utils.cpp \
	booksim_config.cpp \
	module.cpp \
	router.cpp \
	iq_router.cpp \
	event_router.cpp \
	vc.cpp \
	routefunc.cpp \
	traffic.cpp \
	allocator.cpp \
	maxsize.cpp \
	network.cpp \
	singlenet.cpp \
	kncube.cpp \
	fly.cpp \
	trafficmanager.cpp \
	random_utils.cpp \
	buffer_state.cpp \
	stats.cpp \
	pim.cpp \
	islip.cpp \
	loa.cpp \
	wavefront.cpp \
	misc_utils.cpp \
	credit.cpp \
	outputset.cpp \
	flit.cpp \
	selalloc.cpp \
	arbiter.cpp \
	injection.cpp \
	rng_wrapper.cpp \
	rng_double_wrapper.cpp \
	statwraper.cpp

LEX_OBJS  = $(OUTPUT_DIR)/configlex.o
YACC_OBJS = $(OUTPUT_DIR)/config_tab.o

#--- Make rules ---

OBJS = $(CPP_SRCS:%.cpp=$(OUTPUT_DIR)/%.o) $(LEX_OBJS) $(YACC_OBJS)

.PHONY: clean
.PRECIOUS: $(OUTPUT_DIR)/%_tab.cpp $(OUTPUT_DIR)/%_tab.hpp $(OUTPUT_DIR)/%lex.cpp

lib$(PROG).a: $(OBJS)
	ar rcs $(OUTPUT_DIR)/lib$(PROG).a $(OBJS)

purify: $(OBJS)
	$(PURIFY) -always-use-cache-dir $(CPP) $(OBJS) -o $(PROG) -L/usr/lib

quantify: $(OBJS)
	$(QUANT) -always-use-cache-dir $(CPP) $(OBJS) -o $(PROG) -L/usr/lib

$(OUTPUT_DIR)/%lex.o: $(OUTPUT_DIR)/%lex.cpp $(OUTPUT_DIR)/%_tab.hpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

$(OUTPUT_DIR)/%.o: %.cpp
	$(CPP) $(EXTRA) $(CPPFLAGS) $(TEST) -c $< -o $@

$(OUTPUT_DIR)/%.o: $(OUTPUT_DIR)/%.cpp
	$(CPP) $(EXTRA) $(CPPFLAGS) $(TEST) -c $< -o $@
 
$(OUTPUT_DIR)/%.o: %.c
	$(CPP) $(CPPFLAGS) $(TEST) $(VCSFLAGS) -c $< -o $@

$(OUTPUT_DIR)/%_tab.cpp: %.y
	$(YACC) -b$* -p$* $< --file-prefix=$(OUTPUT_DIR)/$*
	cp -f $(OUTPUT_DIR)/$*.tab.c $(OUTPUT_DIR)/$*_tab.cpp

$(OUTPUT_DIR)/%_tab.hpp: $(OUTPUT_DIR)/%_tab.cpp
	cp -f $(OUTPUT_DIR)/$*.tab.h $(OUTPUT_DIR)/$*_tab.hpp

$(OUTPUT_DIR)/%lex.cpp: %.l
	$(LEX) --outfile=$@ -P$* $<
	cp $(OUTPUT_DIR)/configlex.cpp $(OUTPUT_DIR)/configlex.cpp.orig
	awk '/configlineno = 1/ {if (line == 0) {line = 1; print} next;} //{print}' $(OUTPUT_DIR)/configlex.cpp.orig > $(OUTPUT_DIR)/configlex.cpp

clean:
	rm -f $(OBJS) *_tab.cpp *_tab.hpp *.tab.c *.tab.h *lex.cpp
	rm -f $(PROG)
	rm -f lib$(PROG).a 
	rm -f lib$(PROG).so
	rm -f $(OBJS) *.o 

$(OUTPUT_DIR)/interconnect_interface.o: $(OUTPUT_DIR)/../cuda-sim/ptx.tab.h

$(OUTPUT_DIR)/../cuda-sim/ptx.tab.h:
	        make -C ../cuda-sim/ $(OUTPUT_DIR)/../cuda-sim/ptx.tab.c
