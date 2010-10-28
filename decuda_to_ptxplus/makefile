FLAGS = -ggdb -I${BOOST_ROOT} -lboost_regex -lboost_filesystem

all: decuda_to_ptxplus

decuda_to_ptxplus: stringList.o decudaInst.o decudaInstList.o decudaLexer.o decudaParser.o headerParser.o headerLexer.o decuda_to_ptxplus.o lex.ptx_.o ptx.tab.o
	g++ ${FLAGS} -odecuda_to_ptxplus stringList.o decudaInst.o decudaInstList.o decudaLexer.o decudaParser.o headerParser.o headerLexer.o decuda_to_ptxplus.o lex.ptx_.o ptx.tab.o

stringList.o : stringList.cc stringList.h
	g++ ${FLAGS} -c -ostringList.o stringList.cc

decudaInst.o :  decudaInst.cc decudaInst.h stringList.h
	g++ ${FLAGS} -c -odecudaInst.o decudaInst.cc

decudaInstList.o : decudaInstList.cc decudaInstList.h decudaInst.h
	g++ ${FLAGS} -c -odecudaInstList.o decudaInstList.cc

decudaParser.cpp : decuda.y decudaInstList.h
	bison -d -odecudaParser.cpp decuda.y

decudaParser.hpp : decudaParser.cpp

decudaLexer.cpp : decuda.l decudaParser.hpp
	flex -B -odecudaLexer.cpp decuda.l

headerParser.cpp : header.y decudaInstList.h
	bison --name-prefix=header_ -d -oheaderParser.cpp header.y

headerParser.hpp : headerParser.cpp

headerLexer.cpp : header.l headerParser.hpp
	flex -B -oheaderLexer.cpp header.l

decudaLexer.o : decudaLexer.cpp
	g++ ${FLAGS} -c -Igenerated -odecudaLexer.o decudaLexer.cpp

decudaParser.o : decudaParser.cpp stringList.h decudaInst.h decudaInstList.h 
	g++ ${FLAGS} -c -Igenerated -odecudaParser.o decudaParser.cpp

headerLexer.o : headerLexer.cpp
	g++ ${FLAGS} -c -Igenerated -oheaderLexer.o headerLexer.cpp

headerParser.o : headerParser.cpp stringList.h decudaInst.h decudaInstList.h 
	g++ ${FLAGS} -c -Igenerated -oheaderParser.o headerParser.cpp

decuda_to_ptxplus.o : decuda_to_ptxplus.cc decudaInstList.h
	g++ ${FLAGS} -c -odecuda_to_ptxplus.o decuda_to_ptxplus.cc

lex.ptx_.c : ../distribution/src/cuda-sim/ptx.l
	flex ../distribution/src/cuda-sim/ptx.l

ptx.tab.c : ../distribution/src/cuda-sim/ptx.y
	bison --name-prefix=ptx_ -v -d ../distribution/src/cuda-sim/ptx.y

ptx.tab.h : ptx.tab.c

lex.ptx_.o : lex.ptx_.c ptx.tab.h
	g++ ${FLAGS} -c lex.ptx_.c

ptx.tab.o : ptx.tab.c
	g++ ${FLAGS} -c ptx.tab.c

clean:
	rm -f ./stringList.o ./decudaInst.o ./decudaInstList.o ./decudaParser.o ./decudaLexer.o ./headerParser.o ./headerLexer.o ./decuda_to_ptxplus.o
	rm -f ./decudaParser.hpp ./headerParser.hpp
	rm -f ./decudaParser.cpp ./decudaLexer.cpp ./headerParser.cpp ./headerLexer.cpp
	rm -f ./decuda_to_ptxplus
	rm -f ./ptx.output ./ptx.tab.c ./ptx.tab.h ./ptx.tab.o
	rm -f ./lex.ptx_.c ./lex.ptx_.o
