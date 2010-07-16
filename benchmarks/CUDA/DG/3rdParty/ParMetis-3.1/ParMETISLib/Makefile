include ../Makefile.in


CFLAGS = $(COPTIONS) $(OPTFLAGS) -I. $(INCDIR)


OBJS = comm.o util.o debug.o setup.o grsetup.o timer.o \
       node_refine.o initmsection.o order.o \
       xyzpart.o pspases.o frename.o \
       iintsort.o iidxsort.o ikeysort.o ikeyvalsort.o \
       kmetis.o gkmetis.o ometis.o \
       initpart.o match.o coarsen.o \
       kwayfm.o kwayrefine.o kwaybalance.o \
       remap.o stat.o fpqueue.o \
       ametis.o rmetis.o lmatch.o initbalance.o \
       mdiffusion.o diffutil.o wave.o \
       csrmatch.o redomylink.o balancemylink.o \
       selectq.o akwayfm.o serial.o move.o \
       mmetis.o mesh.o memory.o weird.o backcompat.o

.c.o:
	$(CC) $(CFLAGS) -c $*.c


../libparmetis.a: $(OBJS)
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

clean:
	rm -f *.o 

realclean:
	rm -f *.o ; rm -f ../libparmetis.a


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
