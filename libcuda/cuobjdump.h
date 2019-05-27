#ifndef __cuobjdump_h__
#define __cuobjdump_h__
class cuobjdump_parser {

    public:
	yyscan_t scanner;
	int elfserial;
	int ptxserial;
	FILE *ptxfile;
	FILE *elffile;
	FILE *sassfile;
	char filename [1024];
	cuobjdump_parser() {
	    int elfserial = 1;
	    int ptxserial = 1;
	}
};
#endif /* __cuobjdump_h__ */
