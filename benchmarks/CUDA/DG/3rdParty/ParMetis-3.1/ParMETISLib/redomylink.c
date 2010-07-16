/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * redomylink.c
 *
 * This file contains code that implements the edge-based FM refinement
 *
 * Started 7/23/97
 * George
 *
 * $Id: redomylink.c,v 1.2 2003/07/21 17:18:53 karypis Exp $
 */

#include <parmetislib.h>
#define	PE	0

/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void RedoMyLink(CtrlType *ctrl, GraphType *graph, idxtype *home, int me,
  int you, float *flows, float *sr_cost, float *sr_lbavg)
{
  int h, i, r;
  int nvtxs, nedges, ncon;
  int  pass, lastseed, totalv;
  idxtype *xadj, *adjncy, *adjwgt, *where, *vsize;
  idxtype *costwhere, *lbwhere, *selectwhere;
  idxtype *rdata, *ed, *id, *bndptr, *bndind, *perm;
  float *nvwgt, mycost;
  float lbavg, lbvec[MAXNCON];
  float best_lbavg, other_lbavg = -1.0, bestcost, othercost = -1.0;
  float npwgts[2*MAXNCON], pwgts[MAXNCON*2], tpwgts[MAXNCON*2];
  float ipc_factor, redist_factor, ftmp;
int mype;
MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  nvtxs = graph->nvtxs;
  nedges = graph->nedges;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  vsize = graph->vsize;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  ipc_factor = ctrl->ipc_factor;
  redist_factor = ctrl->redist_factor;

  /**************************/
  /* set up data structures */
  /**************************/
  rdata = idxmalloc(7*nvtxs, "rdata");
  id = graph->sendind = rdata;
  ed = graph->recvind = rdata + nvtxs;
  bndptr = graph->sendptr = rdata + 2*nvtxs;
  bndind = graph->recvptr = rdata + 3*nvtxs;
  costwhere = rdata + 4*nvtxs;
  lbwhere = rdata + 5*nvtxs;
  perm = rdata + 6*nvtxs;
  graph->gnpwgts = npwgts;

  RandomPermute(nvtxs, perm, 1);
  idxcopy(nvtxs, where, costwhere);
  idxcopy(nvtxs, where, lbwhere);

  /*****************************/
  /* compute target pwgts      */
  /*****************************/
  sset(ncon*2, 0.0, pwgts);
  for (h=0; h<ncon; h++) {
    tpwgts[h] = -1.0 * flows[h];
    tpwgts[ncon+h] = flows[h];
  }

  for (i=0; i<nvtxs; i++) {
    if (where[i] == me) {
      for (h=0; h<ncon; h++) {
        tpwgts[h] += nvwgt[i*ncon+h];
        pwgts[h] += nvwgt[i*ncon+h];
      }
    }
    else {
      ASSERTS(where[i] == you);
      for (h=0; h<ncon; h++) {
        tpwgts[ncon+h] += nvwgt[i*ncon+h];
        pwgts[ncon+h] += nvwgt[i*ncon+h];
      }
    }
  }

  /* we don't want any weights to be less than zero */
  for (h=0; h<ncon; h++) {
    if (tpwgts[h] < 0.0) {
      tpwgts[ncon+h] += tpwgts[h];
      tpwgts[h] = 0.0;
    }

    if (tpwgts[ncon+h] < 0.0) {
      tpwgts[h] += tpwgts[ncon+h];
      tpwgts[ncon+h] = 0.0;
    }
  } 

  /*****************************/
  /* now compute new bisection */
  /*****************************/
  bestcost = (float)idxsum(nedges, adjwgt)*ipc_factor + (float)idxsum(nvtxs, vsize)*redist_factor;
  best_lbavg = 10.0;

  lastseed = 0;
  for (pass = N_MOC_REDO_PASSES; pass>0; pass--) {
    idxset(nvtxs, 1, where);

    /***************************/
    /* find seed vertices      */
    /***************************/
    r = perm[lastseed] % nvtxs;
    lastseed = (lastseed+1) % nvtxs;
    where[r] = 0;

    Moc_Serial_Compute2WayPartitionParams(graph);
    Moc_Serial_Init2WayBalance(graph, tpwgts);
    Moc_Serial_FM_2WayRefine(graph, tpwgts, 4);
    Moc_Serial_Balance2Way(graph, tpwgts, 1.02);
    Moc_Serial_FM_2WayRefine(graph, tpwgts, 4);

    for (i=0; i<nvtxs; i++)
      where[i] = (where[i] == 0) ? me : you;

    for (i=0; i<ncon; i++) {
      ftmp = (pwgts[i]+pwgts[ncon+i])/2.0;
      if (ftmp != 0.0)
        lbvec[i] = fabs(npwgts[i]-tpwgts[i])/ftmp;
      else
        lbvec[i] = 0.0;
    }
    lbavg = savg(ncon, lbvec);

    totalv = 0;
    for (i=0; i<nvtxs; i++)
      if (where[i] != home[i])
        totalv += vsize[i];

    mycost = (float)(graph->mincut)*ipc_factor + (float)totalv*redist_factor;

    if (bestcost >= mycost) {
      bestcost = mycost;
      other_lbavg = lbavg;
      idxcopy(nvtxs, where, costwhere);
    }

    if (best_lbavg >= lbavg) {
      best_lbavg = lbavg;
      othercost = mycost;
      idxcopy(nvtxs, where, lbwhere);
    }
  }

  if (other_lbavg <= .05) {
    selectwhere = costwhere;
    *sr_cost = bestcost;
    *sr_lbavg = other_lbavg;
  }
  else {
    selectwhere = lbwhere;
    *sr_cost = othercost;
    *sr_lbavg = best_lbavg;
  }

  idxcopy(nvtxs, selectwhere, where);

  GKfree((void **)&rdata, LTERM);
  return;
}

