LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d --report=all --verbose
CCFLAGS		= -ggdb -fPIC -Wall -Wno-unused-function -Wno-sign-compare 
CXX			= g++
CXXFLAGS	= ${CCFLAGS}
LD			= g++
LDFLAGS		= ${CCFLAGS}
OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/cuobjdump_to_ptxplus

CXXFLAGS += -I $(OUTPUT_DIR) -I . -I ../src/cuda-sim/

SASS_PARSER_OBJECTS = $(OUTPUT_DIR)/sass_lexer.o $(OUTPUT_DIR)/sass_parser.o 
ELF_PARSER_OBJECTS = $(OUTPUT_DIR)/elf_lexer.o $(OUTPUT_DIR)/elf_parser.o
HEADER_PARSER_OBJECTS = $(OUTPUT_DIR)/header_parser.o $(OUTPUT_DIR)/header_lexer.o
PTX_PARSER_OBJECTS = $(OUTPUT_DIR)/ptx.tab.o $(OUTPUT_DIR)/lex.ptx_.o

all: $(OUTPUT_DIR)/cuobjdump_to_ptxplus

MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.SECONDARY:

$(OUTPUT_DIR)/cuobjdump_to_ptxplus: $(OUTPUT_DIR)/cuobjdumpInst.o $(OUTPUT_DIR)/cuobjdumpInstList.o $(OUTPUT_DIR)/cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)
	${LD} ${LDFLAGS} -o $@ $(OUTPUT_DIR)/cuobjdumpInst.o $(OUTPUT_DIR)/cuobjdumpInstList.o $(OUTPUT_DIR)/cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)


lex.ptx_.c : ../src/cuda-sim/ptx.l
	${LEX} ${LEXFLAGS} ../src/cuda-sim/ptx.l 

ptx.tab.c : ../src/cuda-sim/ptx.y
	${YACC} ${YFLAGS} --name-prefix=ptx_ -v ../src/cuda-sim/ptx.y 

ptx.tab.h : ptx.tab.c

$(OUTPUT_DIR)/lex.ptx_.o : lex.ptx_.c ptx.tab.h
	${CXX} ${CXXFLAGS} -c lex.ptx_.c -o $@

$(OUTPUT_DIR)/ptx.tab.o : ptx.tab.c ptx_parser.h
	${CXX} ${CXXFLAGS} -c ptx.tab.c -o $@


%_lexer.cc: %.l %_parser.hh
	$(LEX) $(LEXFLAGS) -P $*_ -o$@ $<

%_parser.cc: %.y
	$(YACC) $(YFLAGS) -p $*_ -o$@ $<

# The ':' is needed otherwise make gets confused and doesn't understand that it knows how to make %Parser.hpp from %Parser.cpp.
# : in bash is equivalent to NOP (no operation)
%_parser.hh: %_parser.cc
	:

$(OUTPUT_DIR)/%.o: %.cc
	$(CXX) ${CXXFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: %.cpp %.h
	$(CXX) ${CXXFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: %.c %.h
	${CXX} ${CCFLAGS} -c -o $@ $<

SRCS = $(shell ls *.cc)

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend $(SRCS) 2> /dev/null

clean:
	rm -f lex.ptx_.c ptx.tab.c ptx.tab.h ptx.output
	rm -f elf_lexer.cc  elf_parser.cc  elf_parser.hh elf_parser.output
	rm -f sass_lexer.cc  sass_parser.cc  sass_parser.hh  sass_parser.output
	rm -f header_lexer.cc  header_parser.cc  header_parser.hh  header_parser.output
	rm -rf $(OUTPUT_DIR)
	rm -f $(OUTPUT_DIR)/Makefile.makedepend $(OUTPUT_DIR)/Makefile.makedepend.bak

include $(OUTPUT_DIR)/Makefile.makedepend

