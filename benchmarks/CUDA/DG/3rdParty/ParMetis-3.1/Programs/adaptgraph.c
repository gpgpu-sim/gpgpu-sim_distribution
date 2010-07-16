/*
 * Copyright 1998, Regents of the University of Minnesota
 *
 * tstadpt.c
 * 
 * This file contains code for testing teh adaptive partitioning routines
 *
 * Started 5/19/97
 * George
 *
 * $Id: adaptgraph.c,v 1.2 2003/07/21 17:50:22 karypis Exp $
 *
 */

#include <parmetisbin.h>


/*************************************************************************
* This function implements a simple graph adaption strategy.
**************************************************************************/
void AdaptGraph(GraphType *graph, int afactor, MPI_Comm comm)
{
  int i, nvtxs, nadapt, firstvtx, lastvtx;
  int npes, mype, mypwgt, max, min, sum;
  idxtype *vwgt, *xadj, *adjncy, *adjwgt, *perm;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  srand(mype*afactor);
  srand48(mype*afactor);

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  if (graph->adjwgt == NULL)
    adjwgt = graph->adjwgt = idxsmalloc(graph->nedges, 1, "AdaptGraph: adjwgt");
  else
    adjwgt = graph->adjwgt;
  vwgt = graph->vwgt;

  firstvtx = graph->vtxdist[mype];
  lastvtx = graph->vtxdist[mype+1];

  perm = idxmalloc(nvtxs, "AdaptGraph: perm");
  FastRandomPermute(nvtxs, perm, 1);

  nadapt = RandomInRange(nvtxs);
  nadapt = RandomInRange(nvtxs);
  nadapt = RandomInRange(nvtxs);

  for (i=0; i<nadapt; i++)
    vwgt[perm[i]] = afactor*vwgt[perm[i]];

/*
  for (i=0; i<nvtxs; i++) {
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (k >= firstvtx && k < lastvtx) {
	adjwgt[j] = (int)pow(1.0*(amin(vwgt[i],vwgt[k-firstvtx])), .6667);
        if (adjwgt[j] == 0)
          adjwgt[j] = 1;
      }
    }
  }
*/

  mypwgt = idxsum(nvtxs, vwgt);

  MPI_Allreduce((void *)&mypwgt, (void *)&max, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce((void *)&mypwgt, (void *)&min, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce((void *)&mypwgt, (void *)&sum, 1, MPI_INT, MPI_SUM, comm);

  if (mype == 0)
    printf("Initial Load Imbalance: %5.4f, [%5d %5d %5d] for afactor: %d\n", (1.0*max*npes)/(1.0*sum), min, max, sum, afactor);

  free(perm);
}


/*************************************************************************
* This function implements a simple graph adaption strategy.
**************************************************************************/
void AdaptGraph2(GraphType *graph, int afactor, MPI_Comm comm)
{
  int i, j, k, nvtxs, firstvtx, lastvtx;
  int npes, mype, mypwgt, max, min, sum;
  idxtype *vwgt, *xadj, *adjncy, *adjwgt;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  srand(mype*afactor);
  srand48(mype*afactor);

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  if (graph->adjwgt == NULL)
    adjwgt = graph->adjwgt = idxsmalloc(graph->nedges, 1, "AdaptGraph: adjwgt");
  else
    adjwgt = graph->adjwgt;
  vwgt = graph->vwgt;

  firstvtx = graph->vtxdist[mype];
  lastvtx = graph->vtxdist[mype+1];


/*  if (RandomInRange(npes+1) < .05*npes) { */ 
  if (RandomInRange(npes+1) < 2) { 
    printf("[%d] is adapting\n", mype);
    for (i=0; i<nvtxs; i++)
      vwgt[i] = afactor*vwgt[i];
  }

  for (i=0; i<nvtxs; i++) {
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (k >= firstvtx && k < lastvtx) {
	adjwgt[j] = (int)pow(1.0*(amin(vwgt[i],vwgt[k-firstvtx])), .6667);
        if (adjwgt[j] == 0)
          adjwgt[j] = 1;
      }
    }
  }
      
  mypwgt = idxsum(nvtxs, vwgt);

  MPI_Allreduce((void *)&mypwgt, (void *)&max, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce((void *)&mypwgt, (void *)&min, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce((void *)&mypwgt, (void *)&sum, 1, MPI_INT, MPI_SUM, comm);

  if (mype == 0)
    printf("Initial Load Imbalance: %5.4f, [%5d %5d %5d]\n", (1.0*max*npes)/(1.0*sum), min, max, sum);

}


/*************************************************************************
* This function implements a simple graph adaption strategy.
**************************************************************************/
void Mc_AdaptGraph(GraphType *graph, idxtype *part, int ncon, int nparts, MPI_Comm comm)
{
  int h, i;
  int nvtxs;
  int npes, mype;
  idxtype *vwgt, *pwgts;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  nvtxs = graph->nvtxs;
  vwgt = graph->vwgt;
  pwgts = idxsmalloc(nparts*ncon, 1, "pwgts");

  if (mype == 0) {
    for (i=0; i<nparts; i++)
      for (h=0; h<ncon; h++)
        pwgts[i*ncon+h] = RandomInRange(20)+1;
  }

  MPI_Bcast((void *)pwgts, nparts*ncon, IDX_DATATYPE, 0, comm);

  for (i=0; i<nvtxs; i++)
    for (h=0; h<ncon; h++)
      vwgt[i*ncon+h] = pwgts[part[i]*ncon+h];

  free(pwgts);
  return;
}


