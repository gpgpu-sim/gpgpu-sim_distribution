BOOST_ROOT ?= /usr/include/boost

LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d -v --report=all
CC			= gcc
CCFLAGS		= -ggdb -I${BOOST_ROOT} -lboost_regex -lboost_filesystem -fPIC
CXX			= g++
CXXFLAGS	= ${CCFLAGS}
LD			= g++
LDFLAGS		= ${CCFLAGS}


SASS_PARSER_OBJECTS = sass_lexer.o sass_parser.o 
ELF_PARSER_OBJECTS = elf_lexer.o elf_parser.o
HEADER_PARSER_OBJECTS = headerParser.o headerLexer.o
PTX_PARSER_OBJECTS = ptx.tab.o lex.ptx_.o

all: cuobjdump_to_ptxplus

cuobjdump_to_ptxplus: stringList.o decudaInst.o decudaInstList.o cuobjdumpInst.o cuobjdumpInstList.o cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)
	${LD} ${LDFLAGS} -ocuobjdump_to_ptxplus stringList.o decudaInst.o decudaInstList.o headerParser.o headerLexer.o cuobjdumpInst.o cuobjdumpInstList.o cuobjdump_to_ptxplus.o lex.ptx_.o ptx.tab.o $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) 

stringList.o: ../decuda_to_ptxplus/stringList.cc ../decuda_to_ptxplus/stringList.h
	${CXX} ${CXXFLAGS} -c -ostringList.o ../decuda_to_ptxplus/stringList.cc

decudaInst.o: ../decuda_to_ptxplus/decudaInst.cc ../decuda_to_ptxplus/decudaInst.h ../decuda_to_ptxplus/stringList.h
	${CXX} ${CXXFLAGS} -c -odecudaInst.o ../decuda_to_ptxplus/decudaInst.cc

decudaInstList.o: ../decuda_to_ptxplus/decudaInstList.cc ../decuda_to_ptxplus/decudaInstList.h ../decuda_to_ptxplus/decudaInst.h
	${CXX} ${CXXFLAGS} -c -odecudaInstList.o ../decuda_to_ptxplus/decudaInstList.cc

cuobjdumpInst.o: cuobjdumpInst.cc cuobjdumpInst.h ../decuda_to_ptxplus/decudaInst.h ./decudaInstList.h
	${CXX} ${CXXFLAGS} -c -ocuobjdumpInst.o cuobjdumpInst.cc

cuobjdumpInstList.o: cuobjdumpInstList.cc cuobjdumpInstList.h ../decuda_to_ptxplus/decudaInstList.h
	${CXX} ${CXXFLAGS} -c -ocuobjdumpInstList.o cuobjdumpInstList.cc

#sassParser.cpp: sassParser.y ../distribution/decuda_to_ptxplus/decudaInstList.h cuobjdumpInstList.h
#	${YACC} ${YFLAGS} -osassParser.cpp sassParser.y

#sassParser.hpp: sassParser.cpp

#sassLexer.cpp: sassLexer.l sassParser.hpp
#	${LEX} ${LEXFLAGS} -osassLexer.cpp sassLexer.l

headerParser.cpp : ../decuda_to_ptxplus/header.y ../decuda_to_ptxplus/decudaInstList.h
	${YACC} ${YFLAGS} --name-prefix=header_ -oheaderParser.cpp ../decuda_to_ptxplus/header.y

headerParser.hpp : headerParser.cpp

headerLexer.cpp : ../decuda_to_ptxplus/header.l headerParser.hpp
	${LEX} ${LEXFLAGS} -oheaderLexer.cpp ../decuda_to_ptxplus/header.l

#sassLexer.o: sassLexer.cpp
#	${CXX} ${CXXFLAGS} -c -Igenerated -osassLexer.o sassLexer.cpp
#
#sassParser.o: sassParser.cpp ../distribution/decuda_to_ptxplus/stringList.h ../distribution/decuda_to_ptxplus/decudaInst.h ../distribution/decuda_to_ptxplus/decudaInstList.h cuobjdumpInstList.h
#	${CXX} ${CXXFLAGS} -c -Igenerated -osassParser.o sassParser.cpp

headerLexer.o : headerLexer.cpp
	${CXX} ${CXXFLAGS} -c -Igenerated -oheaderLexer.o headerLexer.cpp

headerParser.o : headerParser.cpp ../decuda_to_ptxplus/stringList.h ../decuda_to_ptxplus/decudaInst.h ./decudaInstList.h
	${CXX} ${CXXFLAGS} -c -Igenerated -oheaderParser.o headerParser.cpp

decudaInstList.h : ../decuda_to_ptxplus/decudaInstList.h ./decudaInst.h
	rm -f decudaInstList.h
	ln -s ../decuda_to_ptxplus/decudaInstList.h decudaInstList.h

decudaInst.h : ../decuda_to_ptxplus/decudaInst.h ./stringList.h
	rm -f decudaInst.h
	ln -s ../decuda_to_ptxplus/decudaInst.h decudaInst.h

stringList.h : ../decuda_to_ptxplus/stringList.h
	rm -f stringList.h
	ln -s ../decuda_to_ptxplus/stringList.h stringList.h

cuobjdump_to_ptxplus.o: cuobjdump_to_ptxplus.cc ../decuda_to_ptxplus/decudaInstList.h cuobjdumpInstList.h
	${CXX} ${CXXFLAGS} -c -ocuobjdump_to_ptxplus.o cuobjdump_to_ptxplus.cc

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

.PRECIOUS: %.c %.h %_lexer.c %_parser.c %_parser.h %_parser.cpp %_parser.hpp %_lexer.cpp

%_lexer.c: %.l %_parser.h
	$(LEX) $(LEXFLAGS) -P $*_ -o$@ $<

%_lexer.cpp: %.l %_parser.hpp
	$(LEX) $(LEXFLAGS) -P $*_ -o$@ $<

%_parser.c: %.y
	$(YACC) $(YFLAGS) -p $*_ -o$@ $<

%_parser.cpp: %.y
	$(YACC) $(YFLAGS) -p $*_ -o$@ $<

# The ':' is needed otherwise make gets confused and doesn't understand that it knows how to make %Parser.hpp from %Parser.cpp.
# : in bash is equivalent to NOP (no operation)
%_parser.h: %_parser.c
	:

%_parser.hpp: %_parser.cpp
	:

cuobjdump_parser.o: cuobjdump_parser.c
	${CC} ${CCFLAGS} -c -o $@ $<

cuobjdump_lexer.o: cuobjdump_lexer.c
	${CC} ${CCFLAGS} -c -o $@ $<

%.o: %.cpp
	$(CXX) ${CXXFLAGS} -c -o $@ $<

SRCS = $(shell ls *.cc)

depend:
	makedepend $(SRCS) 2> /dev/null

clean:
	rm -f ./opcodes.def ./opcodes.h
	rm -f ./lex.ptx_.c ./ptx.tab.c ./ptx.tab.h ./lex.ptx_.o ./ptx.tab.o ptx.output
	rm -f ./decudaInstList.h ./decudaInst.h ./stringList.h
	rm -f ./headerParser.o ./headerLexer.o ./headerParser.cpp ./headerParser.hpp ./headerLexer.cpp ./headerParser.output
	rm -f ./stringList.o ./decudaInst.o ./decudaInstList.o
	rm -f ./cuobjdumpInstList.o ./cuobjdumpInst.o
	rm -f ./sass_lexer.o ./sass_parser.o ./cuobjdump_to_ptxplus.o
	rm -f ./sass_parser.hpp ./sass_parser.output
	rm -f ./sass_parser.cpp ./sass_lexer.cpp
	rm -f ./cuobjdump_to_ptxplus
	rm -f ./cuobjdump_parser.o ./cuobjdump_parser.c ./cuobjdump_parser.output ./cuobjdump_parser.h
	rm -f ./cuobjdump_lexer.o ./cuobjdump_lexer.c
	rm -f ./elf_lexer.o ./elf_parser.o ./elf_lexer.cpp ./elf_parser.cpp ./elf_parser.hpp ./elf_parser.output
# DO NOT DELETE

cuobjdumpInst.o: cuobjdumpInst.h
cuobjdumpInstList.o: cuobjdumpInstList.h cuobjdumpInst.h
cuobjdump_to_ptxplus.o: cuobjdumpInstList.h cuobjdumpInst.h
cuobjdump_to_ptxplus.o: /usr/include/stdio.h /usr/include/features.h
cuobjdump_to_ptxplus.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
cuobjdump_to_ptxplus.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
cuobjdump_to_ptxplus.o: /usr/include/bits/types.h
cuobjdump_to_ptxplus.o: /usr/include/bits/typesizes.h /usr/include/libio.h
cuobjdump_to_ptxplus.o: /usr/include/_G_config.h /usr/include/wchar.h
cuobjdump_to_ptxplus.o: /usr/include/bits/stdio_lim.h
cuobjdump_to_ptxplus.o: /usr/include/bits/sys_errlist.h
