/*
 * serial.c
 *
 * This file contains code that implements k-way refinement
 *
 * Started 7/28/97
 * George
 *
 * $Id: serial.c,v 1.2 2003/07/21 17:18:53 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Moc_SerialKWayAdaptRefine(GraphType *graph, int nparts, idxtype *home,
     float *orgubvec, int npasses)
{
  int i, ii, iii, j, k;
  int nvtxs, ncon, pass, nmoves, myndegrees;
  int from, me, myhome, to, oldcut, gain, tmp;
  idxtype *xadj, *adjncy, *adjwgt;
  idxtype *where;
  EdgeType *mydegrees;
  RInfoType *rinfo, *myrinfo;
  float *npwgts, *nvwgt, *minwgt, *maxwgt, ubvec[MAXNCON];
  int gain_is_greater, gain_is_same, fit_in_to, fit_in_from, going_home;
  int zero_gain, better_balance_ft, better_balance_tt;
  KeyValueType *cand;
int mype;
MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  rinfo = graph->rinfo;
  npwgts = graph->gnpwgts;
  
  /* Setup the weight intervals of the various subdomains */
  cand = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "cand");
  minwgt =  fmalloc(nparts*ncon, "minwgt");
  maxwgt = fmalloc(nparts*ncon, "maxwgt");

  ComputeHKWayLoadImbalance(ncon, nparts, npwgts, ubvec);
  for (i=0; i<ncon; i++)
    ubvec[i] = amax(ubvec[i], orgubvec[i]);

  for (i=0; i<nparts; i++) {
    for (j=0; j<ncon; j++) {
      maxwgt[i*ncon+j] = ubvec[j]/(float)nparts;
      minwgt[i*ncon+j] = ubvec[j]*(float)nparts;
    }
  }

  for (pass=0; pass<npasses; pass++) {
    oldcut = graph->mincut;

    for (i=0; i<nvtxs; i++) {
      cand[i].key = rinfo[i].id-rinfo[i].ed;
      cand[i].val = i;
    }
    ikeysort(nvtxs, cand);

    nmoves = 0;
    for (iii=0; iii<nvtxs; iii++) {
      i = cand[iii].val;

      myrinfo = rinfo+i;

      if (myrinfo->ed >= myrinfo->id) {
        from = where[i];
        myhome = home[i];
        nvwgt = graph->nvwgt+i*ncon;

        if (myrinfo->id > 0 &&
        AreAllHVwgtsBelow(ncon, 1.0, npwgts+from*ncon, -1.0, nvwgt, minwgt+from*ncon)) 
          continue;

        mydegrees = myrinfo->degrees;
        myndegrees = myrinfo->ndegrees;

        for (k=0; k<myndegrees; k++) {
          to = mydegrees[k].edge;
          gain = mydegrees[k].ewgt - myrinfo->id; 
          if (gain >= 0 && 
             (AreAllHVwgtsBelow(ncon, 1.0, npwgts+to*ncon, 1.0, nvwgt, maxwgt+to*ncon) ||
             IsHBalanceBetterFT(ncon,npwgts+from*ncon,npwgts+to*ncon,nvwgt,ubvec))) {
            break;
          }
        }

        /* break out if you did not find a candidate */
        if (k == myndegrees)
          continue;

        for (j=k+1; j<myndegrees; j++) {
          to = mydegrees[j].edge;
          going_home = (myhome == to);
          gain_is_same = (mydegrees[j].ewgt == mydegrees[k].ewgt);
          gain_is_greater = (mydegrees[j].ewgt > mydegrees[k].ewgt);
          fit_in_to = AreAllHVwgtsBelow(ncon,1.0,npwgts+to*ncon,1.0,nvwgt,maxwgt+to*ncon);
          better_balance_ft = IsHBalanceBetterFT(ncon,npwgts+from*ncon,
                              npwgts+to*ncon,nvwgt,ubvec);
          better_balance_tt = IsHBalanceBetterTT(ncon,npwgts+mydegrees[k].edge*ncon,
                              npwgts+to*ncon,nvwgt,ubvec);

          if (
               (gain_is_greater &&
                 (fit_in_to ||
                  better_balance_ft)
               )
            ||
               (gain_is_same &&
                 (
                   (fit_in_to &&
                    going_home)
                ||
                    better_balance_tt
                 )
               )
             ) {
            k = j;
          }
        }

        to = mydegrees[k].edge;
        going_home = (myhome == to);
        zero_gain = (mydegrees[k].ewgt == myrinfo->id);

        fit_in_from = AreAllHVwgtsBelow(ncon,1.0,npwgts+from*ncon,0.0,npwgts+from*ncon,
                      maxwgt+from*ncon);
        better_balance_ft = IsHBalanceBetterFT(ncon,npwgts+from*ncon,
                            npwgts+to*ncon,nvwgt,ubvec);

        if (zero_gain &&
            !going_home &&
            !better_balance_ft &&
            fit_in_from)
          continue;

        /*=====================================================================
        * If we got here, we can now move the vertex from 'from' to 'to' 
        *======================================================================*/
        graph->mincut -= mydegrees[k].ewgt-myrinfo->id;

        /* Update where, weight, and ID/ED information of the vertex you moved */
        saxpy2(ncon, 1.0, nvwgt, 1, npwgts+to*ncon, 1);
        saxpy2(ncon, -1.0, nvwgt, 1, npwgts+from*ncon, 1);
        where[i] = to;
        myrinfo->ed += myrinfo->id-mydegrees[k].ewgt;
        SWAP(myrinfo->id, mydegrees[k].ewgt, tmp);

        if (mydegrees[k].ewgt == 0) {
          myrinfo->ndegrees--;
          mydegrees[k].edge = mydegrees[myrinfo->ndegrees].edge;
          mydegrees[k].ewgt = mydegrees[myrinfo->ndegrees].ewgt;
        }
        else
          mydegrees[k].edge = from;

        /* Update the degrees of adjacent vertices */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          ii = adjncy[j];
          me = where[ii];

          myrinfo = rinfo+ii;
          mydegrees = myrinfo->degrees;

          if (me == from) {
            INC_DEC(myrinfo->ed, myrinfo->id, adjwgt[j]);
          }
          else {
            if (me == to) {
              INC_DEC(myrinfo->id, myrinfo->ed, adjwgt[j]);
            }
          }

          /* Remove contribution of the ed from 'from' */
          if (me != from) {
            for (k=0; k<myrinfo->ndegrees; k++) {
              if (mydegrees[k].edge == from) {
                if (mydegrees[k].ewgt == adjwgt[j]) {
                  myrinfo->ndegrees--;
                  mydegrees[k].edge = mydegrees[myrinfo->ndegrees].edge;
                  mydegrees[k].ewgt = mydegrees[myrinfo->ndegrees].ewgt;
                }
                else
                  mydegrees[k].ewgt -= adjwgt[j];
                break;
              }
            }
          }

          /* Add contribution of the ed to 'to' */
          if (me != to) {
            for (k=0; k<myrinfo->ndegrees; k++) {
              if (mydegrees[k].edge == to) {
                mydegrees[k].ewgt += adjwgt[j];
                break;
              }
            }
            if (k == myrinfo->ndegrees) {
              mydegrees[myrinfo->ndegrees].edge = to;
              mydegrees[myrinfo->ndegrees++].ewgt = adjwgt[j];
            }
          }

        }
        nmoves++;
      }
    }

    if (graph->mincut == oldcut)
      break;
  }

  GKfree((void **)&minwgt, (void **)&maxwgt, (void **)&cand, LTERM);

  return;
}


/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void Moc_ComputeSerialPartitionParams(GraphType *graph, int nparts,
     EdgeType *degrees)
{
  int i, j, k;
  int nvtxs, nedges, ncon, mincut, me, other;
  idxtype *xadj, *adjncy, *adjwgt, *where;
  RInfoType *rinfo, *myrinfo;
  EdgeType *mydegrees;
  float *nvwgt, *npwgts;
int mype;
MPI_Comm_rank(MPI_COMM_WORLD, &mype);


  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  rinfo = graph->rinfo;

  npwgts = sset(ncon*nparts, 0.0, graph->gnpwgts);

  /*------------------------------------------------------------
  / Compute now the id/ed degrees
  /------------------------------------------------------------*/
  nedges = mincut = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    saxpy2(ncon, 1.0, nvwgt+i*ncon, 1, npwgts+me*ncon, 1);

    myrinfo = rinfo+i;
    myrinfo->id = myrinfo->ed = myrinfo->ndegrees = 0;
    myrinfo->degrees = degrees + nedges;
    nedges += xadj[i+1]-xadj[i];

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (me == where[adjncy[j]]) {
        myrinfo->id += adjwgt[j];
      }
      else {
        myrinfo->ed += adjwgt[j];
      }
    }

    mincut += myrinfo->ed;

    /* Time to compute the particular external degrees */
    if (myrinfo->ed > 0) {
      mydegrees = myrinfo->degrees;

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        other = where[adjncy[j]];
        if (me != other) {
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (mydegrees[k].edge == other) {
              mydegrees[k].ewgt += adjwgt[j];
              break;
            }
          }
          if (k == myrinfo->ndegrees) {
            mydegrees[myrinfo->ndegrees].edge = other;
            mydegrees[myrinfo->ndegrees++].ewgt = adjwgt[j];
          }
        }
      }
    }
  }

  graph->mincut = mincut/2;

  return;
}


/*************************************************************************
* This function checks if the vertex weights of two vertices are below 
* a given set of values
**************************************************************************/
int AreAllHVwgtsBelow(int ncon, float alpha, float *vwgt1, float beta, float *vwgt2, float *limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] > limit[i])
      return 0;

  return 1;
}


/*************************************************************************
* This function computes the load imbalance over all the constrains
* For now assume that we just want balanced partitionings
**************************************************************************/ 
void ComputeHKWayLoadImbalance(int ncon, int nparts, float *npwgts, float *lbvec)
{
  int i, j;
  float max;

  for (i=0; i<ncon; i++) {
    max = 0.0;
    for (j=0; j<nparts; j++) {
      if (npwgts[j*ncon+i] > max)
        max = npwgts[j*ncon+i];
    }

    lbvec[i] = max*nparts;
  }
}


/**************************************************************
*  This subroutine remaps a partitioning on a single processor
**************************************************************/
void SerialRemap(GraphType *graph, int nparts, idxtype *base, idxtype *scratch,
     idxtype *remap, float *tpwgts)
{
  int i, ii, j, k;
  int nvtxs, nmapped, max_mult;
  int from, to, current_from, smallcount, bigcount;
  KeyValueType *flowto, *bestflow;
  KeyKeyValueType *sortvtx;
  idxtype *vsize, *htable, *map, *rowmap;

  nvtxs = graph->nvtxs;
  vsize = graph->vsize;
  max_mult = amin(MAX_NPARTS_MULTIPLIER, nparts);

  sortvtx = (KeyKeyValueType *)GKmalloc(nvtxs*sizeof(KeyKeyValueType), "sortvtx");
  flowto = (KeyValueType *)GKmalloc((nparts*max_mult+nparts)*sizeof(KeyValueType), "flowto");
  bestflow = flowto+nparts;
  map = htable = idxsmalloc(nparts*2, -1, "htable");
  rowmap = map+nparts;

  for (i=0; i<nvtxs; i++) {
    sortvtx[i].key1 = base[i];
    sortvtx[i].key2 = vsize[i];
    sortvtx[i].val = i;
  }

  qsort((void *)sortvtx, (size_t)nvtxs, (size_t)sizeof(KeyKeyValueType), SSMIncKeyCmp);

  for (j=0; j<nparts; j++) {
    flowto[j].key = 0;
    flowto[j].val = j;
  }

  /* this step has nparts*nparts*log(nparts) computational complexity */
  bigcount = smallcount = current_from = 0;
  for (ii=0; ii<nvtxs; ii++) {
    i = sortvtx[ii].val;
    from = base[i];
    to = scratch[i];

    if (from > current_from) {
      /* reset the hash table */
      for (j=0; j<smallcount; j++)
        htable[flowto[j].val] = -1;
      ASSERTS(idxsum(nparts, htable) == -nparts);

      ikeysort(smallcount, flowto);

      for (j=0; j<amin(smallcount, max_mult); j++, bigcount++) {
        bestflow[bigcount].key = flowto[j].key;
        bestflow[bigcount].val = current_from*nparts+flowto[j].val;
      }

      smallcount = 0;
      current_from = from;
    }

    if (htable[to] == -1) {
      htable[to] = smallcount;
      flowto[smallcount].key = -vsize[i];
      flowto[smallcount].val = to;
      smallcount++;
    }
    else {
      flowto[htable[to]].key += -vsize[i];
    }
  }

  /* reset the hash table */
  for (j=0; j<smallcount; j++)
    htable[flowto[j].val] = -1;
  ASSERTS(idxsum(nparts, htable) == -nparts);

  ikeysort(smallcount, flowto);

  for (j=0; j<amin(smallcount, max_mult); j++, bigcount++) {
    bestflow[bigcount].key = flowto[j].key;
    bestflow[bigcount].val = current_from*nparts+flowto[j].val;
  }
  ikeysort(bigcount, bestflow);

  ASSERTS(idxsum(nparts, map) == -nparts);
  ASSERTS(idxsum(nparts, rowmap) == -nparts);
  nmapped = 0;

  /* now make as many assignments as possible */
  for (ii=0; ii<bigcount; ii++) {
    i = bestflow[ii].val;
    j = i % nparts;  /* to */
    k = i / nparts;  /* from */

    if (map[j] == -1 && rowmap[k] == -1 && SimilarTpwgts(tpwgts, graph->ncon, j, k)) {
      map[j] = k;
      rowmap[k] = j;
      nmapped++;
    }

    if (nmapped == nparts)
      break;
  }


  /* remap the rest */
  /* it may help try remapping to the same label first */
  if (nmapped < nparts) {
    for (j=0; j<nparts && nmapped<nparts; j++) {
      if (map[j] == -1) {
        for (ii=0; ii<nparts; ii++) {
          i = (j+ii) % nparts;
          if (rowmap[i] == -1 && SimilarTpwgts(tpwgts, graph->ncon, i, j)) {
            map[j] = i;
            rowmap[i] = j;
            nmapped++;
            break;
          }
        }
      }
    }
  }

  /* check to see if remapping fails (due to dis-similar tpwgts) */
  /* if remapping fails, revert to original mapping */
  if (nmapped < nparts)
    for (i=0; i<nparts; i++)
      map[i] = i;

  for (i=0; i<nvtxs; i++)
    remap[i] = map[remap[i]];

  GKfree((void **)&sortvtx, (void **)&flowto, (void **)&htable, LTERM);
}


/*************************************************************************
*  This is a comparison function for Serial Remap
**************************************************************************/
int SSMIncKeyCmp(const void *fptr, const void *sptr)
{
  KeyKeyValueType *first, *second;

  first = (KeyKeyValueType *)(fptr);
  second = (KeyKeyValueType *)(sptr);

  if (first->key1 > second->key1)
    return 1;

  if (first->key1 < second->key1)
     return -1;

  if (first->key2 < second->key2)
    return 1;

  if (first->key2 > second->key2)
     return -1;

   return 0;
}


/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void Moc_Serial_FM_2WayRefine(GraphType *graph, float *tpwgts, int npasses)
{
  int i, ii, j, k;
  int kwgt, nvtxs, ncon, nbnd, nswaps, from, to, pass, limit, tmp, cnum;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *moved, *swaps, *qnum;
  float *nvwgt, *npwgts, mindiff[MAXNCON], origbal, minbal, newbal;
  FPQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut, initcut, newcut, mincutorder;
  float rtpwgts[MAXNCON*2];
  KeyValueType *cand;
int mype;
MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->sendind;
  ed = graph->recvind;
  npwgts = graph->gnpwgts;
  bndptr = graph->sendptr;
  bndind = graph->recvptr;

  moved = idxmalloc(nvtxs, "moved");
  swaps = idxmalloc(nvtxs, "swaps");
  qnum = idxmalloc(nvtxs, "qnum");
  cand = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "cand");

  limit = amin(amax(0.01*nvtxs, 25), 150);

  /* Initialize the queues */
  for (i=0; i<ncon; i++) {
    FPQueueInit(&parts[i][0], nvtxs);
    FPQueueInit(&parts[i][1], nvtxs);
  }
  for (i=0; i<nvtxs; i++)
    qnum[i] = samax(ncon, nvwgt+i*ncon);

  origbal = Serial_Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

  for (i=0; i<ncon; i++) {
    rtpwgts[i] = origbal*tpwgts[i];
    rtpwgts[ncon+i] = origbal*tpwgts[ncon+i];
  }

  idxset(nvtxs, -1, moved);
  for (pass=0; pass<npasses; pass++) { /* Do a number of passes */
    for (i=0; i<ncon; i++) {
      FPQueueReset(&parts[i][0]);
      FPQueueReset(&parts[i][1]);
    }

    mincutorder = -1;
    newcut = mincut = initcut = graph->mincut;
    for (i=0; i<ncon; i++)
      mindiff[i] = fabs(tpwgts[i]-npwgts[i]);
    minbal = Serial_Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

    /* Insert boundary nodes in the priority queues */
    nbnd = graph->gnvtxs;

    for (i=0; i<nbnd; i++) {
      cand[i].key = id[i]-ed[i];
      cand[i].val = i;
    }
    ikeysort(nbnd, cand);

    for (ii=0; ii<nbnd; ii++) {
      i = bndind[cand[ii].val];
      FPQueueInsert(&parts[qnum[i]][where[i]], i, (float)(ed[i]-id[i]));
    }

    for (nswaps=0; nswaps<nvtxs; nswaps++) {
      Serial_SelectQueue(ncon, npwgts, rtpwgts, &from, &cnum, parts);
      to = (from+1)%2;

      if (from == -1 || (higain = FPQueueGetMax(&parts[cnum][from])) == -1)
        break;

      saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
      saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);

      newcut -= (ed[higain]-id[higain]);
      newbal = Serial_Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

      if ((newcut < mincut && newbal-origbal <= .00001) ||
          (newcut == mincut && (newbal < minbal ||
                                (newbal == minbal && Serial_BetterBalance(ncon, npwgts, tpwgts, mindiff))))) {
        mincut = newcut;
        minbal = newbal;
        mincutorder = nswaps;
        for (i=0; i<ncon; i++)
          mindiff[i] = fabs(tpwgts[i]-npwgts[i]);
      }
      else if (nswaps-mincutorder > limit) { /* We hit the limit, undo last move */
        newcut += (ed[higain]-id[higain]);
        saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);
        saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
        break;
      }

      where[higain] = to;
      moved[higain] = nswaps;
      swaps[nswaps] = higain;

      /**************************************************************
      * Update the id[i]/ed[i] values of the affected nodes
      ***************************************************************/
      SWAP(id[higain], ed[higain], tmp);
      if (ed[higain] == 0 && xadj[higain] < xadj[higain+1])
        BNDDelete(nbnd, bndind,  bndptr, higain);

      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];
        oldgain = ed[k]-id[k];

        kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
        INC_DEC(id[k], ed[k], kwgt);

        /* Update its boundary information and queue position */
        if (bndptr[k] != -1) { /* If k was a boundary vertex */
          if (ed[k] == 0) { /* Not a boundary vertex any more */
            BNDDelete(nbnd, bndind, bndptr, k);
            if (moved[k] == -1)  /* Remove it if in the queues */
              FPQueueDelete(&parts[qnum[k]][where[k]], k);
          }
          else { /* If it has not been moved, update its position in the queue */
            if (moved[k] == -1)
              FPQueueUpdate(&parts[qnum[k]][where[k]], k, (float)oldgain, (float)(ed[k]-id[k]));
          }
        }
        else {
          if (ed[k] > 0) {  /* It will now become a boundary vertex */
            BNDInsert(nbnd, bndind, bndptr, k);
            if (moved[k] == -1)
              FPQueueInsert(&parts[qnum[k]][where[k]], k, (float)(ed[k]-id[k]));
          }
        }
      }
    }

    /****************************************************************
    * Roll back computations
    *****************************************************************/
    for (i=0; i<nswaps; i++)
      moved[swaps[i]] = -1;  /* reset moved array */
    for (nswaps--; nswaps>mincutorder; nswaps--) {
      higain = swaps[nswaps];

      to = where[higain] = (where[higain]+1)%2;
      SWAP(id[higain], ed[higain], tmp);
      if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain+1])
        BNDDelete(nbnd, bndind,  bndptr, higain);
      else if (ed[higain] > 0 && bndptr[higain] == -1)
        BNDInsert(nbnd, bndind,  bndptr, higain);

      saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
      saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+((to+1)%2)*ncon, 1);
      for (j=xadj[higain]; j<xadj[higain+1]; j++) {
        k = adjncy[j];

        kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
        INC_DEC(id[k], ed[k], kwgt);

        if (bndptr[k] != -1 && ed[k] == 0)
          BNDDelete(nbnd, bndind, bndptr, k);
        if (bndptr[k] == -1 && ed[k] > 0)
          BNDInsert(nbnd, bndind, bndptr, k);
      }
    }

    graph->mincut = mincut;
    graph->gnvtxs = nbnd;

    if (mincutorder == -1 || mincut == initcut)
      break;
  }

  for (i=0; i<ncon; i++) {
    FPQueueFree(&parts[i][0]);
    FPQueueFree(&parts[i][1]);
  }

  GKfree((void **)&cand, (void **)&qnum, (void **)&moved, (void **)&swaps, LTERM);
  return;
}

/*************************************************************************
* This function selects the partition number and the queue from which
* we will move vertices out
**************************************************************************/
void Serial_SelectQueue(int ncon, float *npwgts, float *tpwgts, int *from, int *cnum,
     FPQueueType queues[MAXNCON][2])
{
  int i, part;
  float maxgain=0.0;
  float max = -1.0, maxdiff=0.0;
int mype;
MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  *from = -1;
  *cnum = -1;

  /* First determine the side and the queue, irrespective of the presence of nodes */
  for (part=0; part<2; part++) {
    for (i=0; i<ncon; i++) {
      if (npwgts[part*ncon+i]-tpwgts[part*ncon+i] >= maxdiff) {
        maxdiff = npwgts[part*ncon+i]-tpwgts[part*ncon+i];
        *from = part;
        *cnum = i;
      }
    }
  }

  if (*from != -1 && FPQueueGetQSize(&queues[*cnum][*from]) == 0) {
    /* The desired queue is empty, select a node from that side anyway */
    for (i=0; i<ncon; i++) {
      if (FPQueueGetQSize(&queues[i][*from]) > 0) {
        max = npwgts[(*from)*ncon + i];
        *cnum = i;
        break;
      }
    }

    for (i++; i<ncon; i++) {
      if (npwgts[(*from)*ncon + i] > max && FPQueueGetQSize(&queues[i][*from]) > 0) {
        max = npwgts[(*from)*ncon + i];
        *cnum = i;
      }
    }
  }


  /* Check to see if you can focus on the cut */
  if (maxdiff <= 0.0 || *from == -1) {
    maxgain = -100000.0;

    for (part=0; part<2; part++) {
      for (i=0; i<ncon; i++) {
        if (FPQueueGetQSize(&queues[i][part]) > 0 &&
            FPQueueSeeMaxGain(&queues[i][part]) > maxgain) {
          maxgain = FPQueueSeeMaxGain(&queues[i][part]);
          *from = part;
          *cnum = i;
        }
      }
    }
  }

  return;
}

/*************************************************************************
* This function checks if the balance achieved is better than the diff
* For now, it uses a 2-norm measure
**************************************************************************/
int Serial_BetterBalance(int ncon, float *npwgts, float *tpwgts, float *diff)
{
  int i;
  float ndiff[MAXNCON];

  for (i=0; i<ncon; i++)
    ndiff[i] = fabs(tpwgts[i]-npwgts[i]);

  return snorm2(ncon, ndiff) < snorm2(ncon, diff);
}



/*************************************************************************
* This function computes the load imbalance over all the constrains
**************************************************************************/
float Serial_Compute2WayHLoadImbalance(int ncon, float *npwgts, float *tpwgts)
{
  int i;
  float max=0.0, temp;

  for (i=0; i<ncon; i++) {
    if (tpwgts[i] == 0.0)
      temp = 0.0;
    else
      temp = fabs(tpwgts[i]-npwgts[i])/tpwgts[i];
    max = (max < temp ? temp : max);
  }
  return 1.0+max;
}



/*************************************************************************
* This function performs an edge-based FM refinement
**************************************************************************/
void Moc_Serial_Balance2Way(GraphType *graph, float *tpwgts, float lbfactor)
{
  int i, ii, j, k, kwgt, nvtxs, ncon, nbnd, nswaps, from, to, limit, tmp, cnum;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *moved, *swaps, *qnum;
  float *nvwgt, *npwgts, mindiff[MAXNCON], origbal, minbal, newbal;
  FPQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut, newcut, mincutorder;
  int qsizes[MAXNCON][2];
  KeyValueType *cand;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->sendind;
  ed = graph->recvind;
  npwgts = graph->gnpwgts;
  bndptr = graph->sendptr;
  bndind = graph->recvptr;

  moved = idxmalloc(nvtxs, "moved");
  swaps = idxmalloc(nvtxs, "swaps");
  qnum = idxmalloc(nvtxs, "qnum");
  cand = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "cand");


  limit = amin(amax(0.01*nvtxs, 15), 100);

  /* Initialize the queues */
  for (i=0; i<ncon; i++) {
    FPQueueInit(&parts[i][0], nvtxs);
    FPQueueInit(&parts[i][1], nvtxs);
    qsizes[i][0] = qsizes[i][1] = 0;
  }

  for (i=0; i<nvtxs; i++) {
    qnum[i] = samax(ncon, nvwgt+i*ncon);
    qsizes[qnum[i]][where[i]]++;
  }

  for (from=0; from<2; from++) {
    for (j=0; j<ncon; j++) {
      if (qsizes[j][from] == 0) {
        for (i=0; i<nvtxs; i++) {
          if (where[i] != from)
            continue;

          k = samax2(ncon, nvwgt+i*ncon);
          if (k == j &&
               qsizes[qnum[i]][from] > qsizes[j][from] &&
               nvwgt[i*ncon+qnum[i]] < 1.3*nvwgt[i*ncon+j]) {
            qsizes[qnum[i]][from]--;
            qsizes[j][from]++;
            qnum[i] = j;
          }
        }
      }
    }
  }


  for (i=0; i<ncon; i++)
    mindiff[i] = fabs(tpwgts[i]-npwgts[i]);
  minbal = origbal = Serial_Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);
  newcut = mincut = graph->mincut;
  mincutorder = -1;

  idxset(nvtxs, -1, moved);

  /* Insert all nodes in the priority queues */
  nbnd = graph->gnvtxs;
  for (i=0; i<nvtxs; i++) {
    cand[i].key = id[i]-ed[i];
    cand[i].val = i;
  }
  ikeysort(nvtxs, cand);

  for (ii=0; ii<nvtxs; ii++) {
    i = cand[ii].val;
    FPQueueInsert(&parts[qnum[i]][where[i]], i, (float)(ed[i]-id[i]));
  }

  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    if (minbal < lbfactor)
      break;

    Serial_SelectQueue(ncon, npwgts, tpwgts, &from, &cnum, parts);
    to = (from+1)%2;

    if (from == -1 || (higain = FPQueueGetMax(&parts[cnum][from])) == -1)
      break;

    saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
    saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);
    newcut -= (ed[higain]-id[higain]);
    newbal = Serial_Compute2WayHLoadImbalance(ncon, npwgts, tpwgts);

    if (newbal < minbal || (newbal == minbal &&
        (newcut < mincut || (newcut == mincut &&
          Serial_BetterBalance(ncon, npwgts, tpwgts, mindiff))))) {
      mincut = newcut;
      minbal = newbal;
      mincutorder = nswaps;
      for (i=0; i<ncon; i++)
        mindiff[i] = fabs(tpwgts[i]-npwgts[i]);
    }
    else if (nswaps-mincutorder > limit) { /* We hit the limit, undo last move */
      newcut += (ed[higain]-id[higain]);
      saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);
      saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
      break;
    }

    where[higain] = to;
    moved[higain] = nswaps;
    swaps[nswaps] = higain;

    /**************************************************************
    * Update the id[i]/ed[i] values of the affected nodes
    ***************************************************************/
    SWAP(id[higain], ed[higain], tmp);
    if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain+1])
      BNDDelete(nbnd, bndind,  bndptr, higain);
    if (ed[higain] > 0 && bndptr[higain] == -1)
      BNDInsert(nbnd, bndind,  bndptr, higain);

    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];
      oldgain = ed[k]-id[k];

      kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
      INC_DEC(id[k], ed[k], kwgt);

      /* Update the queue position */
      if (moved[k] == -1)
        FPQueueUpdate(&parts[qnum[k]][where[k]], k, (float)(oldgain), (float)(ed[k]-id[k]));

      /* Update its boundary information */
      if (ed[k] == 0 && bndptr[k] != -1)
        BNDDelete(nbnd, bndind, bndptr, k);
      else if (ed[k] > 0 && bndptr[k] == -1)
        BNDInsert(nbnd, bndind, bndptr, k);
    }
  }


  /****************************************************************
  * Roll back computations
  *****************************************************************/
  for (nswaps--; nswaps>mincutorder; nswaps--) {
    higain = swaps[nswaps];

    to = where[higain] = (where[higain]+1)%2;
    SWAP(id[higain], ed[higain], tmp);
    if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain+1])
      BNDDelete(nbnd, bndind,  bndptr, higain);
    else if (ed[higain] > 0 && bndptr[higain] == -1)
      BNDInsert(nbnd, bndind,  bndptr, higain);

    saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
    saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+((to+1)%2)*ncon, 1);
    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];

      kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
      INC_DEC(id[k], ed[k], kwgt);

      if (bndptr[k] != -1 && ed[k] == 0)
        BNDDelete(nbnd, bndind, bndptr, k);
      if (bndptr[k] == -1 && ed[k] > 0)
        BNDInsert(nbnd, bndind, bndptr, k);
    }
  }

  graph->mincut = mincut;
  graph->gnvtxs = nbnd;


  for (i=0; i<ncon; i++) {
    FPQueueFree(&parts[i][0]);
    FPQueueFree(&parts[i][1]);
  }

  GKfree((void **)&cand, (void **)&qnum, (void **)&moved, (void **)&swaps, LTERM);
  return;
}

/*************************************************************************
* This function balances two partitions by moving the highest gain
* (including negative gain) vertices to the other domain.
* It is used only when tha unbalance is due to non contigous
* subdomains. That is, the are no boundary vertices.
* It moves vertices from the domain that is overweight to the one that
* is underweight.
**************************************************************************/
void Moc_Serial_Init2WayBalance(GraphType *graph, float *tpwgts)
{
  int i, ii, j, k;
  int kwgt, nvtxs, nbnd, ncon, nswaps, from, to, cnum, tmp;
  idxtype *xadj, *adjncy, *adjwgt, *where, *id, *ed, *bndptr, *bndind;
  idxtype *qnum;
  float *nvwgt, *npwgts;
  FPQueueType parts[MAXNCON][2];
  int higain, oldgain, mincut;
  KeyValueType *cand;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  nvwgt = graph->nvwgt;
  adjwgt = graph->adjwgt;
  where = graph->where;
  id = graph->sendind;
  ed = graph->recvind;
  npwgts = graph->gnpwgts;
  bndptr = graph->sendptr;
  bndind = graph->recvptr;

  qnum = idxmalloc(nvtxs, "qnum");
  cand = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "cand");

  /* This is called for initial partitioning so we know from where to pick nodes */
  from = 1;
  to = (from+1)%2;

  for (i=0; i<ncon; i++) {
    FPQueueInit(&parts[i][0], nvtxs);
    FPQueueInit(&parts[i][1], nvtxs);
  }

  /* Compute the queues in which each vertex will be assigned to */
  for (i=0; i<nvtxs; i++)
    qnum[i] = samax(ncon, nvwgt+i*ncon);

  for (i=0; i<nvtxs; i++) {
    cand[i].key = id[i]-ed[i];
    cand[i].val = i;
  }
  ikeysort(nvtxs, cand);

  /* Insert the nodes of the proper partition in the appropriate priority queue */
  for (ii=0; ii<nvtxs; ii++) {
    i = cand[ii].val;
    if (where[i] == from) {
      if (ed[i] > 0)
        FPQueueInsert(&parts[qnum[i]][0], i, (float)(ed[i]-id[i]));
      else
        FPQueueInsert(&parts[qnum[i]][1], i, (float)(ed[i]-id[i]));
    }
  }

  mincut = graph->mincut;
  nbnd = graph->gnvtxs;
  for (nswaps=0; nswaps<nvtxs; nswaps++) {
    if (Serial_AreAnyVwgtsBelow(ncon, 1.0, npwgts+from*ncon, 0.0, nvwgt, tpwgts+from*ncon))
      break;

    if ((cnum = Serial_SelectQueueOneWay(ncon, npwgts, tpwgts, from, parts)) == -1)
      break;


    if ((higain = FPQueueGetMax(&parts[cnum][0])) == -1)
      higain = FPQueueGetMax(&parts[cnum][1]);

    mincut -= (ed[higain]-id[higain]);
    saxpy2(ncon, 1.0, nvwgt+higain*ncon, 1, npwgts+to*ncon, 1);
    saxpy2(ncon, -1.0, nvwgt+higain*ncon, 1, npwgts+from*ncon, 1);

    where[higain] = to;

    /**************************************************************
    * Update the id[i]/ed[i] values of the affected nodes
    ***************************************************************/
    SWAP(id[higain], ed[higain], tmp);
    if (ed[higain] == 0 && bndptr[higain] != -1 && xadj[higain] < xadj[higain+1])
      BNDDelete(nbnd, bndind,  bndptr, higain);
    if (ed[higain] > 0 && bndptr[higain] == -1)
      BNDInsert(nbnd, bndind,  bndptr, higain);

    for (j=xadj[higain]; j<xadj[higain+1]; j++) {
      k = adjncy[j];
      oldgain = ed[k]-id[k];

      kwgt = (to == where[k] ? adjwgt[j] : -adjwgt[j]);
      INC_DEC(id[k], ed[k], kwgt);

      /* Update the queue position */
      if (where[k] == from) {
        if (ed[k] > 0 && bndptr[k] == -1) {  /* It moves in boundary */
          FPQueueDelete(&parts[qnum[k]][1], k);
          FPQueueInsert(&parts[qnum[k]][0], k, (float)(ed[k]-id[k]));
        }
        else { /* It must be in the boundary already */
          FPQueueUpdate(&parts[qnum[k]][0], k, (float)(oldgain), (float)(ed[k]-id[k]));
        }
      }

      /* Update its boundary information */
      if (ed[k] == 0 && bndptr[k] != -1)
        BNDDelete(nbnd, bndind, bndptr, k);
      else if (ed[k] > 0 && bndptr[k] == -1)
        BNDInsert(nbnd, bndind, bndptr, k);
    }
  }

  graph->mincut = mincut;
  graph->gnvtxs = nbnd;

  for (i=0; i<ncon; i++) {
    FPQueueFree(&parts[i][0]);
    FPQueueFree(&parts[i][1]);
  }

  GKfree((void **)&cand, (void **)&qnum, LTERM);
}


/*************************************************************************
* This function selects the partition number and the queue from which
* we will move vertices out
**************************************************************************/
int Serial_SelectQueueOneWay(int ncon, float *npwgts, float *tpwgts, int from,
    FPQueueType queues[MAXNCON][2])
{
  int i, cnum=-1;
  float max=0.0;

  for (i=0; i<ncon; i++) {
    if (npwgts[from*ncon+i]-tpwgts[from*ncon+i] >= max &&
        FPQueueGetQSize(&queues[i][0]) + FPQueueGetQSize(&queues[i][1]) > 0) {
      max = npwgts[from*ncon+i]-tpwgts[i];
      cnum = i;
    }
  }

  return cnum;
}


/*************************************************************************
* This function computes the initial id/ed
**************************************************************************/
void Moc_Serial_Compute2WayPartitionParams(GraphType *graph)
{
  int i, j, me, nvtxs, ncon, nbnd, mincut;
  idxtype *xadj, *adjncy, *adjwgt;
  float *nvwgt, *npwgts;
  idxtype *id, *ed, *where;
  idxtype *bndptr, *bndind;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;

  npwgts = sset(2*ncon, 0.0, graph->gnpwgts);
  id = idxset(nvtxs, 0, graph->sendind);
  ed = idxset(nvtxs, 0, graph->recvind);
  bndptr = idxset(nvtxs, -1, graph->sendptr);
  bndind = graph->recvptr;

  /*------------------------------------------------------------
  / Compute now the id/ed degrees
  /------------------------------------------------------------*/
  nbnd = mincut = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    saxpy2(ncon, 1.0, nvwgt+i*ncon, 1, npwgts+me*ncon, 1);

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (me == where[adjncy[j]])
        id[i] += adjwgt[j];
      else
        ed[i] += adjwgt[j];
    }

    if (ed[i] > 0 || xadj[i] == xadj[i+1]) {
      mincut += ed[i];
      bndptr[i] = nbnd;
      bndind[nbnd++] = i;
    }
  }

  graph->mincut = mincut/2;
  graph->gnvtxs = nbnd;

}

/*************************************************************************
* This function checks if the vertex weights of two vertices are below
* a given set of values
**************************************************************************/
int Serial_AreAnyVwgtsBelow(int ncon, float alpha, float *vwgt1, float beta, float *vwgt2, float *limit)
{
  int i;

  for (i=0; i<ncon; i++)
    if (alpha*vwgt1[i] + beta*vwgt2[i] < limit[i])
      return 1;

  return 0;
}


/*************************************************************************
*  This function computes the edge-cut of a serial graph.
**************************************************************************/
int ComputeSerialEdgeCut(GraphType *graph)
{
  int i, j;
  int cut = 0;

  for (i=0; i<graph->nvtxs; i++) {
    for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++)
      if (graph->where[i] != graph->where[graph->adjncy[j]])
        cut += graph->adjwgt[j];
  }
  graph->mincut = cut/2;

  return graph->mincut;
}

/*************************************************************************
*  This function computes the TotalV of a serial graph.
**************************************************************************/
int ComputeSerialTotalV(GraphType *graph, idxtype *home)
{
  int i;
  int totalv = 0;

  for (i=0; i<graph->nvtxs; i++)
    if (graph->where[i] != home[i])
      totalv += (graph->vsize == NULL) ? graph->vwgt[i] : graph->vsize[i];

  return totalv;
}


