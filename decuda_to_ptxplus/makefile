# Copyright (c) 2009-2011, Jimmy Kwa,
# The University of British Columbia
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

lex.ptx_.c : ../src/cuda-sim/ptx.l
	flex ../src/cuda-sim/ptx.l

ptx.tab.c : ../src/cuda-sim/ptx.y
	bison --name-prefix=ptx_ -v -d ../src/cuda-sim/ptx.y

ptx.tab.h : ptx.tab.c

lex.ptx_.o : lex.ptx_.c ptx.tab.h
	g++ ${FLAGS} -c lex.ptx_.c

ptx.tab.o : ptx.tab.c
	g++ ${FLAGS} -c ptx.tab.c

SRCS = $(shell ls *.cc)

depend:
	makedepend $(SRCS) 2> /dev/null
	
clean:
	rm -f ./stringList.o ./decudaInst.o ./decudaInstList.o ./decudaParser.o ./decudaLexer.o ./headerParser.o ./headerLexer.o ./decuda_to_ptxplus.o
	rm -f ./decudaParser.hpp ./headerParser.hpp
	rm -f ./decudaParser.cpp ./decudaLexer.cpp ./headerParser.cpp ./headerLexer.cpp
	rm -f ./decuda_to_ptxplus
	rm -f ./ptx.output ./ptx.tab.c ./ptx.tab.h ./ptx.tab.o
	rm -f ./lex.ptx_.c ./lex.ptx_.o
# DO NOT DELETE

decudaInst.o: decudaInst.h stringList.h /usr/include/assert.h
decudaInst.o: /usr/include/features.h /usr/include/sys/cdefs.h
decudaInst.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
decudaInst.o: /usr/include/gnu/stubs-64.h /usr/include/stdlib.h
decudaInst.o: /usr/include/sys/types.h /usr/include/bits/types.h
decudaInst.o: /usr/include/bits/typesizes.h /usr/include/time.h
decudaInst.o: /usr/include/endian.h /usr/include/bits/endian.h
decudaInst.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
decudaInst.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
decudaInst.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
decudaInst.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
decudaInstList.o: decudaInstList.h decudaInst.h stringList.h
decudaInstList.o: /usr/include/assert.h /usr/include/features.h
decudaInstList.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
decudaInstList.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
decudaInstList.o: /usr/include/stdio.h /usr/include/bits/types.h
decudaInstList.o: /usr/include/bits/typesizes.h /usr/include/libio.h
decudaInstList.o: /usr/include/_G_config.h /usr/include/wchar.h
decudaInstList.o: /usr/include/bits/stdio_lim.h
decudaInstList.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
decudaInstList.o: /usr/include/sys/types.h /usr/include/time.h
decudaInstList.o: /usr/include/endian.h /usr/include/bits/endian.h
decudaInstList.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
decudaInstList.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
decudaInstList.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
decudaInstList.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
decuda_to_ptxplus.o: decudaInstList.h decudaInst.h stringList.h
decuda_to_ptxplus.o: /usr/include/assert.h /usr/include/features.h
decuda_to_ptxplus.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
decuda_to_ptxplus.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
decuda_to_ptxplus.o: /usr/include/stdio.h /usr/include/bits/types.h
decuda_to_ptxplus.o: /usr/include/bits/typesizes.h /usr/include/libio.h
decuda_to_ptxplus.o: /usr/include/_G_config.h /usr/include/wchar.h
decuda_to_ptxplus.o: /usr/include/bits/stdio_lim.h
decuda_to_ptxplus.o: /usr/include/bits/sys_errlist.h /usr/include/stdlib.h
decuda_to_ptxplus.o: /usr/include/sys/types.h /usr/include/time.h
decuda_to_ptxplus.o: /usr/include/endian.h /usr/include/bits/endian.h
decuda_to_ptxplus.o: /usr/include/bits/byteswap.h /usr/include/sys/select.h
decuda_to_ptxplus.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
decuda_to_ptxplus.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
decuda_to_ptxplus.o: /usr/include/bits/pthreadtypes.h /usr/include/alloca.h
stringList.o: stringList.h /usr/include/stdio.h /usr/include/features.h
stringList.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
stringList.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
stringList.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
stringList.o: /usr/include/libio.h /usr/include/_G_config.h
stringList.o: /usr/include/wchar.h /usr/include/bits/stdio_lim.h
stringList.o: /usr/include/bits/sys_errlist.h
