/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * stat.c
 *
 * This file computes various statistics
 *
 * Started 7/25/97
 * George
 *
 * $Id: NEW_stats.c,v 1.1 2003/07/16 15:55:15 karypis Exp $
 *
 */

#include <metis.h>


/*************************************************************************
* This function computes the balance of the partitioning
**************************************************************************/
void Moc_ComputePartitionBalance(GraphType *graph, int nparts, idxtype *where, float *ubvec)
{
  int i, j, nvtxs, ncon;
  float *kpwgts, *nvwgt;
  float balance;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  nvwgt = graph->nvwgt;

  kpwgts = fmalloc(nparts, "ComputePartitionInfo: kpwgts");

  for (j=0; j<ncon; j++) {
    sset(nparts, 0.0, kpwgts);
    for (i=0; i<graph->nvtxs; i++)
      kpwgts[where[i]] += nvwgt[i*ncon+j];

    ubvec[j] = (float)nparts*kpwgts[samax(nparts, kpwgts)]/ssum(nparts, kpwgts);
  }

  free(kpwgts);

}

