#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mpi.h"

#define max(a,b)  ( (a>b)?a:b )
#define min(a,b)  ( (a<b)?a:b )

void ParallelPairs(void *objs, int Nmyobjs, int sizeobj,
		   int  (*numget)(const void *),
		   void (*numset)(const void *, int ),
		   int  (*procget)(const void *),
		   void (*marry)(const void *, const void *),
		   int (*compare_objs)(const void *, const void *)){

  char *myobjs = (char*) objs;

  int n, p, sk, cnt, num;
  int maxind = 0;

  int procid, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* local sort */
  qsort(myobjs, Nmyobjs, sizeobj, compare_objs);

  /* TW: homework replace from here --------> */
  maxind = 0;
  for(n=0;n<Nmyobjs;++n){
    num = numget(myobjs+sizeobj*n);
    maxind = max(maxind, num);
  }

  int globalmaxind;
  MPI_Allreduce(&maxind, &globalmaxind, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
  int binsize = ceil( (double)(globalmaxind)/(double)nprocs ) + 10;

  int *outN = (int*) calloc(nprocs, sizeof(int));
  int *inN  = (int*) calloc(nprocs, sizeof(int));

  int *cumoutN = (int*) calloc(nprocs, sizeof(int));
  int *cuminN  = (int*) calloc(nprocs, sizeof(int));

  sk = 0;
  /* count the number of objs in each bin */
  int binup = binsize;
  for(p=0;p<nprocs;++p){
    while( numget(myobjs+(sk*sizeobj) ) <= binup ){
      ++(outN[p]);
      ++sk;
      if(sk==Nmyobjs){
	break;
      }
    }
    binup += binsize;
    if(sk==Nmyobjs){
      break;
    }
  }
  /* TW: <---------- replace to here */

  /* communicate numbers to be sent to each bin */
  MPI_Alltoall(outN, 1, MPI_INT, 
	       inN,  1, MPI_INT, 
	       MPI_COMM_WORLD);

  /* build incoming buffer */
  int Notherobjs = 0;
  for(p=0;p<nprocs;++p)
    Notherobjs += inN[p];

  for(p=0;p<nprocs;++p){
    outN[p] *=  sizeobj/sizeof(int);
    inN[p]  *=  sizeobj/sizeof(int);
  }

  for(p=1;p<nprocs;++p){
    cumoutN[p] = cumoutN[p-1]+outN[p-1];
    cuminN[p]  = cuminN[p-1] + inN[p-1];
  }

  /* fill up bins of objects from cloud */
  char *otherobjs = (char*) calloc(Notherobjs*sizeobj, sizeof(char));

  MPI_Alltoallv(myobjs,    outN, cumoutN, MPI_INT,
		otherobjs, inN,  cuminN,  MPI_INT,
		MPI_COMM_WORLD);

  /* sort the bin */
  qsort(otherobjs, Notherobjs, sizeobj, compare_objs);

  /* number unique objs consecutively in each bin */
  for(n=1;n<Notherobjs;++n){
    /* match ? */
    if(!compare_objs(otherobjs+    n*sizeobj, 
		     otherobjs+(n-1)*sizeobj)){
      
      marry(otherobjs+n*sizeobj, otherobjs+(n-1)*sizeobj);
    }
  }

  char *outobjs = (char*) calloc(Notherobjs*sizeobj, sizeof(char));
  sk = 0;
  for(p=0;p<nprocs;++p)
    for(n=0;n<Notherobjs;++n)
      if(procget(otherobjs+n*sizeobj)==p){
	memcpy(outobjs+sk*sizeobj, otherobjs+n*sizeobj, sizeobj);
	++sk;
      }

  /* send results out */
  MPI_Alltoallv(outobjs,  inN,  cuminN, MPI_INT,
		 myobjs, outN, cumoutN, MPI_INT,
		MPI_COMM_WORLD);


  free(otherobjs);
  free(outN); free(inN); free(cumoutN); free(cuminN);
}

