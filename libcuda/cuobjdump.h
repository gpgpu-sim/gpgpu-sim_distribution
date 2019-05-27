#ifndef __cuobjdump_h__
#define __cuobjdump_h__
struct cuobjdump_parser {
    yyscan_t scanner;
    int elfserial;
    int ptxserial;
    FILE *ptxfile;
    FILE *elffile;
    FILE *sassfile;
    char filename [1024];
};
#endif /* __cuobjdump_h__ */
