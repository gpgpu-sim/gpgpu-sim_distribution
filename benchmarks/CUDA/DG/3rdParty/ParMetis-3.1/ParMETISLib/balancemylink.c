/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * balancemylink.c
 *
 * This file contains code that implements the edge-based FM refinement
 *
 * Started 7/23/97
 * George
 *
 * $Id: balancemylink.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 */

#include <parmetislib.h>
#define	PE	0

/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
int BalanceMyLink(CtrlType *ctrl, GraphType *graph, idxtype *home, int me,
  int you, float *flows, float maxdiff, float *diff_cost, float *diff_lbavg,
  float avgvwgt)
{
  int h, i, ii, j, k;
  int nvtxs, ncon;
  int nqueues, minval, maxval, higain, vtx, edge, totalv;
  int from, to, qnum, index, nchanges, cut, tmp;
  int pass, nswaps, nmoves, multiplier;
  idxtype *xadj, *vsize, *adjncy, *adjwgt, *where, *ed, *id;
  idxtype *hval, *nvpq, *inq, *map, *rmap, *ptr, *myqueue, *changes;
  float *nvwgt, lbvec[MAXNCON], pwgts[MAXNCON*2], tpwgts[MAXNCON*2], my_wgt[MAXNCON];
  float newgain, oldgain = 0.0;
  float lbavg, bestflow, mycost;
  float ipc_factor, redist_factor, ftmp;
  FPQueueType *queues;
int mype;
MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  vsize = graph->vsize;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  ipc_factor = ctrl->ipc_factor;
  redist_factor = ctrl->redist_factor;

  hval = idxmalloc(nvtxs*7, "hval");
  id = hval + nvtxs;
  ed = hval + nvtxs*2;
  map = hval + nvtxs*3;
  rmap = hval + nvtxs*4;
  myqueue = hval + nvtxs*5;
  changes = hval + nvtxs*6;

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

  /* we don't want any tpwgts to be less than zero */
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

  /*******************************/
  /* insert vertices into queues */
  /*******************************/
  minval = maxval = 0;
  multiplier = 1;
  for (i=0; i<ncon; i++) {
    multiplier *= (i+1);
    maxval += i*multiplier;
    minval += (ncon-1-i)*multiplier;
  }

  nqueues = maxval-minval+1;
  nvpq = idxsmalloc(nqueues, 0, "nvpq");
  ptr = idxmalloc(nqueues+1, "ptr");
  inq = idxmalloc(nqueues*2, "inq");
  queues = (FPQueueType *)(GKmalloc(sizeof(FPQueueType)*nqueues*2, "queues"));

  for (i=0; i<nvtxs; i++)
    hval[i] = Moc_HashVwgts(ncon, nvwgt+i*ncon) - minval;

  for (i=0; i<nvtxs; i++)
    nvpq[hval[i]]++;

  ptr[0] = 0;
  for (i=0; i<nqueues; i++)
    ptr[i+1] = ptr[i] + nvpq[i];

  for (i=0; i<nvtxs; i++) {
    map[i] = ptr[hval[i]];
    rmap[ptr[hval[i]]++] = i;
  }

  for (i=nqueues-1; i>0; i--)
    ptr[i] = ptr[i-1];
  ptr[0] = 0;

  /* initialize queues */
  for (i=0; i<nqueues; i++)
    if (nvpq[i] > 0) {
      FPQueueInit(queues+i, nvpq[i]);
      FPQueueInit(queues+i+nqueues, nvpq[i]);
    }

  /* compute internal/external degrees */
  idxset(nvtxs, 0, id);
  idxset(nvtxs, 0, ed);
  for (j=0; j<nvtxs; j++)
    for (k=xadj[j]; k<xadj[j+1]; k++)
      if (where[adjncy[k]] == where[j])
        id[j] += adjwgt[k];
      else 
        ed[j] += adjwgt[k];

  nswaps = 0;
  for (pass=0; pass<N_MOC_BAL_PASSES; pass++) {
    idxset(nvtxs, -1, myqueue); 
    idxset(nqueues*2, 0, inq);

    /* insert vertices into correct queues */
    for (j=0; j<nvtxs; j++) {
      index = (where[j] == me) ? 0 : nqueues;

      newgain = ipc_factor*(float)(ed[j]-id[j]);
      if (home[j] == me || home[j] == you) {
        if (where[j] == home[j])
          newgain -= redist_factor*(float)vsize[j];
        else
          newgain += redist_factor*(float)vsize[j];
      }

      FPQueueInsert(queues+hval[j]+index, map[j]-ptr[hval[j]], newgain);
      myqueue[j] = (where[j] == me) ? 0 : 1;
      inq[hval[j]+index]++;
    }

/*    bestflow = sfavg(ncon, flows); */
    for (j=0, h=0; h<ncon; h++)
      if (fabs(flows[h]) > fabs(flows[j])) j = h;
        bestflow = fabs(flows[j]);

    nchanges = nmoves = 0;
    for (ii=0; ii<nvtxs/2; ii++) {
      from = -1;
      Moc_DynamicSelectQueue(nqueues, ncon, me, you, inq, flows, &from,
      &qnum, minval, avgvwgt, maxdiff);

      /* can't find a vertex in one subdomain, try the other */
      if (from != -1 && qnum == -1) {
        from = (from == me) ? you : me;

        if (from == me) {
          for (j=0; j<ncon; j++)
            if (flows[j] > avgvwgt)
              break;
        }
        else {
          for (j=0; j<ncon; j++)
            if (flows[j] < -1.0*avgvwgt)
              break;
        }

        if (j != ncon)
          Moc_DynamicSelectQueue(nqueues, ncon, me, you, inq, flows, &from,
          &qnum, minval, avgvwgt, maxdiff);
      }

      if (qnum == -1)
        break;

      to = (from == me) ? you : me;
      index = (from == me) ? 0 : nqueues;
      higain = FPQueueGetMax(queues+qnum+index);
      inq[qnum+index]--;
      ASSERTS(higain != -1);

      /*****************/
      /* make the swap */
      /*****************/
      vtx = rmap[higain+ptr[qnum]];
      myqueue[vtx] = -1;
      where[vtx] = to;
      nswaps++;
      nmoves++;

      /* update the flows */
      for (j=0; j<ncon; j++)
        flows[j] += (to == me) ? nvwgt[vtx*ncon+j] : -1.0*nvwgt[vtx*ncon+j];
 
/*      ftmp = sfavg(ncon, flows); */
      for (j=0, h=0; h<ncon; h++)
        if (fabs(flows[h]) > fabs(flows[j])) j = h;
          ftmp = fabs(flows[j]);

      if (ftmp < bestflow) {
        bestflow = ftmp;
        nchanges = 0;
      }
      else {
        changes[nchanges++] = vtx;
      }

      SWAP(id[vtx], ed[vtx], tmp);

      for (j=xadj[vtx]; j<xadj[vtx+1]; j++) {
        edge = adjncy[j];

        /* must compute oldgain before changing id/ed */
        if (myqueue[edge] != -1) {
          oldgain = ipc_factor*(float)(ed[edge]-id[edge]);
          if (home[edge] == me || home[edge] == you) {
            if (where[edge] == home[edge])
              oldgain -= redist_factor*(float)vsize[edge];
            else
              oldgain += redist_factor*(float)vsize[edge];
          }
        }

        tmp = (to == where[edge] ? adjwgt[j] : -adjwgt[j]);
        INC_DEC(id[edge], ed[edge], tmp);

        if (myqueue[edge] != -1) {
          newgain = ipc_factor*(float)(ed[edge]-id[edge]);
          if (home[edge] == me || home[edge] == you) {
            if (where[edge] == home[edge])
              newgain -= redist_factor*(float)vsize[edge];
            else
              newgain += redist_factor*(float)vsize[edge];
          }

          FPQueueUpdate(queues+hval[edge]+(nqueues*myqueue[edge]),
          map[edge]-ptr[hval[edge]], oldgain, newgain);
        }
      }
    }

    /****************************/
    /* now go back to best flow */
    /****************************/
    nswaps -= nchanges;
    nmoves -= nchanges;
    for (i=0; i<nchanges; i++) {
      vtx = changes[i];
      from = where[vtx];
      where[vtx] = to = (from == me) ? you : me;

      SWAP(id[vtx], ed[vtx], tmp);
      for (j=xadj[vtx]; j<xadj[vtx+1]; j++) {
        edge = adjncy[j];
        tmp = (to == where[edge] ? adjwgt[j] : -adjwgt[j]);
        INC_DEC(id[edge], ed[edge], tmp);
      }
    }

    for (i=0; i<nqueues; i++) {
      if (nvpq[i] > 0) {
        FPQueueReset(queues+i);
        FPQueueReset(queues+i+nqueues);
      }
    }

    if (nmoves == 0)
      break;
  }

  /***************************/
  /* compute 2-way imbalance */
  /***************************/
  sset(ncon, 0.0, my_wgt);
  for (i=0; i<nvtxs; i++)
    if (where[i] == me)
      for (h=0; h<ncon; h++)
        my_wgt[h] += nvwgt[i*ncon+h];

  for (i=0; i<ncon; i++) {
    ftmp =  (pwgts[i]+pwgts[ncon+i])/2.0;
    if (ftmp != 0.0)
      lbvec[i] = fabs(my_wgt[i]-tpwgts[i]) / ftmp;
    else
      lbvec[i] = 0.0;
  }
  lbavg = savg(ncon, lbvec);
  *diff_lbavg = lbavg;

  /****************/
  /* compute cost */
  /****************/
  cut = totalv = 0;
  for (i=0; i<nvtxs; i++) {
    if (where[i] != home[i])
      totalv += vsize[i];

      for (j=xadj[i]; j<xadj[i+1]; j++) 
        if (where[adjncy[j]] != where[i])
          cut += adjwgt[j];
  }
  cut /= 2;
  mycost = cut*ipc_factor + totalv*redist_factor;
  *diff_cost = mycost;

  /* free memory */
  for (i=0; i<nqueues; i++)
    if (nvpq[i] > 0) {
      FPQueueFree(queues+i);
      FPQueueFree(queues+i+nqueues);
    }

  GKfree((void **)&hval, (void **)&nvpq, (void **)&ptr, (void **)&inq, (void **)&queues, LTERM);
  return nswaps;
}

