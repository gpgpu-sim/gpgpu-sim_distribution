include ../Makefile.in

BINDIR = ../Graphs

INCLUDES = -I./ -I../ParMETISLib $(INCDIR) 
CFLAGS = $(COPTIONS) $(OPTFLAGS) $(INCLUDES)


LIBSDIR = -L.. $(LIBDIR) 
LIBS = -lparmetis -lmetis $(XTRALIBS) -lm
PARMETISLIB = ../libparmetis.a ../libmetis.a


PTESTOBJS = ptest.o io.o adaptgraph.o 
MESHTESTOBJS = mtest.o io.o
PARMETISOBJS = parmetis.o io.o adaptgraph.o

          
.c.o:
	$(CC) $(CFLAGS) -c $*.c


default: $(BINDIR)/ptest$(VERNUM) $(BINDIR)/mtest$(VERNUM) 

$(BINDIR)/mtest$(VERNUM): $(MESHTESTOBJS) $(PARMETISLIB)
	$(LD)  -o $@ $(MESHTESTOBJS) $(LIBSDIR) $(LIBS)
	chmod 744 $@

$(BINDIR)/ptest$(VERNUM): $(PTESTOBJS) $(PARMETISLIB)
	$(LD)  -o $@ $(PTESTOBJS) $(LIBSDIR) $(LIBS)
	chmod 744 $@

clean:
	rm -f *.o ;\
	rm -f $(BINDIR)/ptest$(VERNUM) 
	rm -f $(BINDIR)/mtest$(VERNUM) 

realclean:
	rm -f *.o ;\
	rm -f $(BINDIR)/ptest$(VERNUM) 
	rm -f $(BINDIR)/mtest$(VERNUM) 


checkin:
	@for file in *.[c,h]; \
	do \
	ci -u -m'Maintance' $$file;\
	done 

checkin2:
	@for file in *.[c,h]; \
	do \
	ci $$file;\
	rcs -U $$file;\
	co $$file;\
	done
