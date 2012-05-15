LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d -v --report=all
CC			= gcc
CCFLAGS		= -ggdb -fPIC
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

cuobjdump_to_ptxplus: stringList.o cuobjdumpInst.o cuobjdumpInstList.o cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)
	${LD} ${LDFLAGS} -o $@ stringList.o cuobjdumpInst.o cuobjdumpInstList.o cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)

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

depend:
	makedepend $(SRCS) 2> /dev/null

clean:
	rm -f opcodes.def opcodes.h
	rm -f lex.ptx_.c ptx.tab.c ptx.tab.h lex.ptx_.o ptx.tab.o ptx.output
	rm -f elf_lexer.cc  elf_parser.cc  elf_parser.hh elf_parser.output
	rm -f sass_lexer.cc  sass_parser.cc  sass_parser.hh  sass_parser.output
	rm -f header_lexer.cc  header_parser.cc  header_parser.hh  header_parser.output
	rm -f *.o
	rm -f cuobjdump_to_ptxplus
# DO NOT DELETE

cuobjdumpInst.o: cuobjdumpInst.h /usr/include/string.h
cuobjdumpInst.o: /usr/include/features.h /usr/include/sys/cdefs.h
cuobjdumpInst.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
cuobjdumpInst.o: /usr/include/gnu/stubs-64.h /usr/include/xlocale.h
cuobjdumpInst.o: /usr/include/stdio.h /usr/include/bits/types.h
cuobjdumpInst.o: /usr/include/bits/typesizes.h /usr/include/libio.h
cuobjdumpInst.o: /usr/include/_G_config.h /usr/include/wchar.h
cuobjdumpInst.o: /usr/include/bits/stdio_lim.h
cuobjdumpInst.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
cuobjdumpInst.o: /usr/include/bits/waitflags.h /usr/include/bits/waitstatus.h
cuobjdumpInst.o: /usr/include/endian.h /usr/include/bits/endian.h
cuobjdumpInst.o: /usr/include/bits/byteswap.h /usr/include/sys/types.h
cuobjdumpInst.o: /usr/include/time.h /usr/include/sys/select.h
cuobjdumpInst.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
cuobjdumpInst.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
cuobjdumpInst.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
cuobjdumpInst.o: stringList.h
cuobjdumpInstList.o: cuobjdumpInstList.h cuobjdumpInst.h
cuobjdumpInstList.o: /usr/include/string.h /usr/include/features.h
cuobjdumpInstList.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
cuobjdumpInstList.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
cuobjdumpInstList.o: /usr/include/xlocale.h /usr/include/stdio.h
cuobjdumpInstList.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
cuobjdumpInstList.o: /usr/include/libio.h /usr/include/_G_config.h
cuobjdumpInstList.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
cuobjdumpInstList.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
cuobjdumpInstList.o: /usr/include/bits/waitflags.h
cuobjdumpInstList.o: /usr/include/bits/waitstatus.h /usr/include/endian.h
cuobjdumpInstList.o: /usr/include/bits/endian.h /usr/include/bits/byteswap.h
cuobjdumpInstList.o: /usr/include/sys/types.h /usr/include/time.h
cuobjdumpInstList.o: /usr/include/sys/select.h /usr/include/bits/select.h
cuobjdumpInstList.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
cuobjdumpInstList.o: /usr/include/sys/sysmacros.h
cuobjdumpInstList.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
cuobjdumpInstList.o: stringList.h
cuobjdump_to_ptxplus.o: /usr/include/stdio.h /usr/include/features.h
cuobjdump_to_ptxplus.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
cuobjdump_to_ptxplus.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
cuobjdump_to_ptxplus.o: /usr/include/bits/types.h
cuobjdump_to_ptxplus.o: /usr/include/bits/typesizes.h /usr/include/libio.h
cuobjdump_to_ptxplus.o: /usr/include/_G_config.h /usr/include/wchar.h
cuobjdump_to_ptxplus.o: /usr/include/bits/stdio_lim.h
cuobjdump_to_ptxplus.o: /usr/include/bits/sys_errlist.h cuobjdumpInstList.h
cuobjdump_to_ptxplus.o: cuobjdumpInst.h /usr/include/string.h
cuobjdump_to_ptxplus.o: /usr/include/xlocale.h /usr/include/stdlib.h
cuobjdump_to_ptxplus.o: /usr/include/bits/waitflags.h
cuobjdump_to_ptxplus.o: /usr/include/bits/waitstatus.h /usr/include/endian.h
cuobjdump_to_ptxplus.o: /usr/include/bits/endian.h
cuobjdump_to_ptxplus.o: /usr/include/bits/byteswap.h /usr/include/sys/types.h
cuobjdump_to_ptxplus.o: /usr/include/time.h /usr/include/sys/select.h
cuobjdump_to_ptxplus.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
cuobjdump_to_ptxplus.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
cuobjdump_to_ptxplus.o: /usr/include/bits/pthreadtypes.h
cuobjdump_to_ptxplus.o: /usr/include/alloca.h stringList.h
stringList.o: stringList.h /usr/include/stdio.h /usr/include/features.h
stringList.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
stringList.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
stringList.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
stringList.o: /usr/include/libio.h /usr/include/_G_config.h
stringList.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
stringList.o: /usr/include/bits/sys_errlist.h
