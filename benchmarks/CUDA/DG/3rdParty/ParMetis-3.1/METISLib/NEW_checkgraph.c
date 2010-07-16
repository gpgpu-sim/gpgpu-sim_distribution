/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * checkgraph.c
 *
 * This file contains routines related to I/O
 *
 * Started 8/28/94
 * George
 *
 * $Id: NEW_checkgraph.c,v 1.1 2003/07/16 15:55:13 karypis Exp $
 *
 */

#include <metis.h>



/*************************************************************************
* This function checks if a graph is valid
**************************************************************************/
int CheckGraph(GraphType *graph)
{
  int i, j, k, l;
  int nvtxs, ncon, err=0;
  int minedge, maxedge, minewgt, maxewgt;
  float minvwgt[MAXNCON], maxvwgt[MAXNCON];
  idxtype *xadj, *adjncy, *adjwgt, *htable;
  float *nvwgt, ntvwgts[MAXNCON];

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  htable = idxsmalloc(nvtxs, 0, "htable");

  if (ncon > 1) {
    for (j=0; j<ncon; j++) { 
      minvwgt[j] = maxvwgt[j] = nvwgt[j];
      ntvwgts[j] = 0.0;
    }
  }

  minedge = maxedge = adjncy[0];
  minewgt = maxewgt = adjwgt[0];

  for (i=0; i<nvtxs; i++) {
    if (ncon > 1) {
      for (j=0; j<ncon; j++) {
        ntvwgts[j] += nvwgt[i*ncon+j];
        minvwgt[j] = (nvwgt[i*ncon+j] < minvwgt[j]) ? nvwgt[i*ncon+j] : minvwgt[j];
        maxvwgt[j] = (nvwgt[i*ncon+j] > maxvwgt[j]) ? nvwgt[i*ncon+j] : maxvwgt[j];
      }
    }

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];

      minedge = (k < minedge) ? k : minedge;
      maxedge = (k > maxedge) ? k : maxedge;
      minewgt = (adjwgt[j] < minewgt) ? adjwgt[j] : minewgt;
      maxewgt = (adjwgt[j] > maxewgt) ? adjwgt[j] : maxewgt;

      if (i == k) {
        printf("Vertex %d contains a self-loop (i.e., diagonal entry in the matrix)!\n", i);
        err++;
      }
      else {
        for (l=xadj[k]; l<xadj[k+1]; l++) {
          if (adjncy[l] == i) {
            if (adjwgt != NULL && adjwgt[l] != adjwgt[j]) {
              printf("Edges (%d %d) and (%d %d) do not have the same weight! %d %d\n", i,k,k,i, adjwgt[l], adjwgt[j]);
              err++;
            }
            break;
          }
        }
        if (l == xadj[k+1]) {
          printf("Missing edge: (%d %d)!\n", k, i);
          err++;
        }
      }

      if (htable[k] == 0) {
        htable[k]++;
      }
      else {
        printf("Edge %d from vertex %d is repeated %d times\n", k, i, htable[k]++);
        err++;
      }
    }

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      htable[adjncy[j]] = 0;
    }
  }

  if (ncon > 1) {
    for (j=0; j<ncon; j++) {
      if (fabs(ntvwgts[j] - 1.0) > 0.0001) {
        printf("Normalized vwgts don't sum to one.  Weight %d = %.8f.\n", j, ntvwgts[j]);
        err++;
      }
    }
  }

/*
  printf("errs: %d, adjncy: [%d %d], adjwgt: [%d %d]\n",
  err, minedge, maxedge, minewgt, maxewgt);
  if (ncon > 1) {
    for (j=0; j<ncon; j++)
      printf("[%.5f %.5f] ", minvwgt[j], maxvwgt[j]);
    printf("\n");
  }
*/
 
  if (err > 0) { 
    printf("A total of %d errors exist in the input file. Correct them, and run again!\n", err);
  }

  GKfree(&htable, LTERM);
  return (err == 0 ? 1 : 0);
}

