LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d --report=all --verbose
CC			= gcc
CCFLAGS		= -ggdb -fPIC -Wall -Wno-unused-function -Wno-sign-compare
CXX			= g++
CXXFLAGS	= ${CCFLAGS}
LD			= g++
LDFLAGS		= ${CCFLAGS}


SASS_PARSER_OBJECTS = sass_lexer.o sass_parser.o 
ELF_PARSER_OBJECTS = elf_lexer.o elf_parser.o
HEADER_PARSER_OBJECTS = header_parser.o header_lexer.o
PTX_PARSER_OBJECTS = ptx.tab.o lex.ptx_.o

all: cuobjdump_to_ptxplus

MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.SECONDARY:

cuobjdump_to_ptxplus: cuobjdumpInst.o cuobjdumpInstList.o cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)
	${LD} ${LDFLAGS} -o $@ cuobjdumpInst.o cuobjdumpInstList.o cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)

lex.ptx_.c : ../src/cuda-sim/ptx.l
	${LEX} ${LEXFLAGS} ../src/cuda-sim/ptx.l

ptx.tab.c : ../src/cuda-sim/ptx.y
	${YACC} ${YFLAGS} --name-prefix=ptx_ -v ../src/cuda-sim/ptx.y

ptx.tab.h : ptx.tab.c

lex.ptx_.o : lex.ptx_.c ptx.tab.h
	rm -f opcodes.def opcodes.h
	ln -s ../src/cuda-sim/opcodes.def opcodes.def
	ln -s ../src/cuda-sim/opcodes.h opcodes.h
	${CXX} ${CXXFLAGS} -c lex.ptx_.c

ptx.tab.o : ptx.tab.c ptx_parser.h
	${CXX} ${CXXFLAGS} -c ptx.tab.c


%_lexer.cc: %.l %_parser.hh
	$(LEX) $(LEXFLAGS) -P $*_ -o$@ $<

%_parser.cc: %.y
	$(YACC) $(YFLAGS) -p $*_ -o$@ $<

# The ':' is needed otherwise make gets confused and doesn't understand that it knows how to make %Parser.hpp from %Parser.cpp.
# : in bash is equivalent to NOP (no operation)
%_parser.hh: %_parser.cc
	:

%.o: %.cc
	$(CXX) ${CXXFLAGS} -c -o $@ $<

%.o: %.cpp %.h
	$(CXX) ${CXXFLAGS} -c -o $@ $<

%.o: %.c %.h
	${CC} ${CCFLAGS} -c -o $@ $<

SRCS = $(shell ls *.cc)

Makefile.makedepend: depend

depend:
	touch Makefile.makedepend
	makedepend -fMakefile.makedepend $(SRCS) 2> /dev/null

clean:
	rm -f opcodes.def opcodes.h
	rm -f lex.ptx_.c ptx.tab.c ptx.tab.h lex.ptx_.o ptx.tab.o ptx.output
	rm -f elf_lexer.cc  elf_parser.cc  elf_parser.hh elf_parser.output
	rm -f sass_lexer.cc  sass_parser.cc  sass_parser.hh  sass_parser.output
	rm -f header_lexer.cc  header_parser.cc  header_parser.hh  header_parser.output
	rm -f *.o
	rm -f cuobjdump_to_ptxplus
	rm -f Makefile.makedepend Makefile.makedepend.bak

include Makefile.makedepend

