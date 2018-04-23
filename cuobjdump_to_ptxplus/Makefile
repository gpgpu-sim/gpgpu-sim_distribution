LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d --report=all --verbose
CCFLAGS		= -ggdb -fPIC 
CXXFLAGS	+= ${CCFLAGS}
OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/cuobjdump_to_ptxplus

CXXFLAGS += -I $(OUTPUT_DIR) -I $(OUTPUT_DIR)/../cuda-sim/ -I . -I ../src/cuda-sim/

SASS_PARSER_OBJECTS = $(OUTPUT_DIR)/sass_lexer.o $(OUTPUT_DIR)/sass_parser.o 
ELF_PARSER_OBJECTS = $(OUTPUT_DIR)/elf_lexer.o $(OUTPUT_DIR)/elf_parser.o
HEADER_PARSER_OBJECTS = $(OUTPUT_DIR)/header_parser.o $(OUTPUT_DIR)/header_lexer.o
PTX_PARSER_OBJECTS = $(OUTPUT_DIR)/ptx.tab.o $(OUTPUT_DIR)/lex.ptx_.o

all: $(OUTPUT_DIR)/cuobjdump_to_ptxplus

MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.SECONDARY:

$(OUTPUT_DIR)/cuobjdump_to_ptxplus: $(OUTPUT_DIR)/cuobjdumpInst.o $(OUTPUT_DIR)/cuobjdumpInstList.o $(OUTPUT_DIR)/cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)
	$(CXX) ${LDFLAGS} -o $@ $(OUTPUT_DIR)/cuobjdumpInst.o $(OUTPUT_DIR)/cuobjdumpInstList.o $(OUTPUT_DIR)/cuobjdump_to_ptxplus.o $(PTX_PARSER_OBJECTS) $(SASS_PARSER_OBJECTS) $(ELF_PARSER_OBJECTS) $(HEADER_PARSER_OBJECTS)


$(OUTPUT_DIR)/lex.ptx_.cc : ../src/cuda-sim/ptx.l
	${LEX} ${LEXFLAGS} -o$(OUTPUT_DIR)/lex.ptx_.cc ../src/cuda-sim/ptx.l

$(OUTPUT_DIR)/ptx.tab.cpp : ../src/cuda-sim/ptx.ypp
	${YACC} ${YFLAGS} --name-prefix=ptx_ -v ../src/cuda-sim/ptx.ypp  --file-prefix=$(OUTPUT_DIR)/ptx

$(OUTPUT_DIR)/ptx.tab.hpp : $(OUTPUT_DIR)/ptx.tab.cpp

$(OUTPUT_DIR)/lex.ptx_.o : $(OUTPUT_DIR)/lex.ptx_.cc $(OUTPUT_DIR)/ptx.tab.hpp
	${CXX} ${CXXFLAGS} -c $(OUTPUT_DIR)/lex.ptx_.cc -o $@

$(OUTPUT_DIR)/ptx.tab.o : $(OUTPUT_DIR)/ptx.tab.cpp ptx_parser.h
	${CXX} ${CXXFLAGS} -c $(OUTPUT_DIR)/ptx.tab.cpp -o $@

$(OUTPUT_DIR)/%_lexer.cc: %.l $(OUTPUT_DIR)/%_parser.hh
	$(LEX) $(LEXFLAGS) -P $*_ -o$@ $<

$(OUTPUT_DIR)/%_parser.cc: %.y
	$(YACC) $(YFLAGS) -p $*_ -o$@ $< --file-prefix=$(OUTPUT_DIR)/$@

# The ':' is needed otherwise make gets confused and doesn't understand that it knows how to make %Parser.hpp from %Parser.cpp.
# : in bash is equivalent to NOP (no operation)
$(OUTPUT_DIR)/%_parser.hh: $(OUTPUT_DIR)/%_parser.cc
	:

$(OUTPUT_DIR)/%_parser.hh: %_parser.cc
	:

$(OUTPUT_DIR)/%.o: %.cc
	$(CXX) ${CXXFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: $(OUTPUT_DIR)/%.cc
	$(CXX) ${CXXFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: %.cpp %.h
	$(CXX) ${CXXFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: $(OUTPUT_DIR)/%.cpp $(OUTPUT_DIR)/%.h
	$(CXX) ${CXXFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: %.c %.h
	${CXX} ${CCFLAGS} -c -o $@ $<

$(OUTPUT_DIR)/%.o: $(OUTPUT_DIR)/%.c $(OUTPUT_DIR)/%.h
	${CXX} ${CCFLAGS} -c -o $@ $<

SRCS = $(shell ls *.cc)

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend $(SRCS) 2> /dev/null

clean:
	rm -f lex.ptx_.cc ptx.tab.cc ptx.tab.hpp ptx.output
	rm -f elf_lexer.cc  elf_parser.cc  elf_parser.hh elf_parser.output
	rm -f sass_lexer.cc  sass_parser.cc  sass_parser.hh  sass_parser.output
	rm -f header_lexer.cc  header_parser.cc  header_parser.hh  header_parser.output
	rm -rf $(OUTPUT_DIR)
	rm -f $(OUTPUT_DIR)/Makefile.makedepend $(OUTPUT_DIR)/Makefile.makedepend.bak

include $(OUTPUT_DIR)/Makefile.makedepend
