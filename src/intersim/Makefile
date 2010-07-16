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


INTERFACE = interconnect_interface.cpp
DEBUG = 0 
CPPFLAGS = -g -Wall 
ifneq ($(DEBUG),1)
CPPFLAGS = -O3 -g 
else
CPPFLAGS += -D_GLIBCXX_DEBUG -DGLIBCXX_DEBUG_PEDANTIC
endif

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

LEX_OBJS  = configlex.o
YACC_OBJS = config_tab.o

#--- Make rules ---

OBJS = $(CPP_SRCS:.cpp=.o) $(LEX_OBJS) $(YACC_OBJS)

.PHONY: clean
.PRECIOUS: %_tab.cpp %_tab.hpp %lex.cpp

lib$(PROG).a: $(OBJS)
	ar rcs lib$(PROG).a $(OBJS)

purify: $(OBJS)
	$(PURIFY) -always-use-cache-dir $(CPP) $(OBJS) -o $(PROG) -L/usr/lib

quantify: $(OBJS)
	$(QUANT) -always-use-cache-dir $(CPP) $(OBJS) -o $(PROG) -L/usr/lib

%lex.o: %lex.cpp %_tab.hpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

%.o: %.cpp
	$(CPP) $(EXTRA) $(CPPFLAGS) $(TEST) -c $< -o $@
 
%.o: %.c
	$(CPP) $(CPPFLAGS) $(TEST) $(VCSFLAGS) -c $< -o $@

%_tab.cpp: %.y
	$(YACC) -b$* -p$* $<
	cp -f $*.tab.c $*_tab.cpp

%_tab.hpp: %_tab.cpp
	cp -f $*.tab.h $*_tab.hpp

%lex.cpp: %.l
	$(LEX) -P$* -o$@ $<
	cp configlex.cpp configlex.cpp.orig
	awk '/configlineno = 1/ {if (line == 0) {line = 1; print} next;} //{print}' configlex.cpp.orig > configlex.cpp

clean:
	rm -f $(OBJS) *_tab.cpp *_tab.hpp *.tab.c *.tab.h *lex.cpp
	rm -f $(PROG)
	rm -f lib$(PROG).a 
	rm -f lib$(PROG).so 

interconnect_interface.o: ../cuda-sim/ptx.tab.h

../cuda-sim/ptx.tab.h:
	        make -C ../cuda-sim/ ptx.tab.c
