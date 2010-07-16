/*
 * premap.c
 *
 * This file contains code that computes the assignment of processors to
 * partition numbers so that it will minimize the redistribution cost
 *
 * Started 4/16/98
 * George
 *
 * $Id: remap.c,v 1.2 2003/07/21 17:18:53 karypis Exp $
 *
 */

#include <parmetislib.h>

/*************************************************************************
* This function remaps that graph so that it will minimize the 
* redistribution cost
**************************************************************************/
void ParallelReMapGraph(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, nvtxs, nparts;
  idxtype *where, *vsize, *map, *lpwgts;

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->RemapTmr));

  if (ctrl->npes != ctrl->nparts) {
    IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RemapTmr));
    return;
  }

  nvtxs = graph->nvtxs;
  where = graph->where;
  vsize = graph->vsize;
  nparts = ctrl->nparts;

  map = wspace->pv1;
  lpwgts = idxset(nparts, 0, wspace->pv2);

  for (i=0; i<nvtxs; i++)
    lpwgts[where[i]] += (vsize == NULL) ? 1 : vsize[i];

  ParallelTotalVReMap(ctrl, lpwgts, map, wspace, NREMAP_PASSES, graph->ncon);

  for (i=0; i<nvtxs; i++)
    where[i] = map[where[i]];

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->RemapTmr));
}


/*************************************************************************
* This function computes the assignment using the the objective the 
* minimization of the total volume of data that needs to move
**************************************************************************/
void ParallelTotalVReMap(CtrlType *ctrl, idxtype *lpwgts, idxtype *map,
     WorkSpaceType *wspace, int npasses, int ncon)
{
  int i, ii, j, k, nparts, mype;
  int pass, maxipwgt, nmapped, oldwgt, newwgt, done;
  idxtype *rowmap, *mylpwgts;
  KeyValueType *recv, send;
  int nsaved, gnsaved;

  mype = ctrl->mype;
  nparts = ctrl->nparts;
  recv = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*nparts, "remap: recv");
  mylpwgts = idxmalloc(nparts, "mylpwgts");

  done = nmapped = 0;
  idxset(nparts, -1, map);
  rowmap = idxset(nparts, -1, wspace->pv3);
  idxcopy(nparts, lpwgts, mylpwgts);
  for (pass=0; pass<npasses; pass++) {
    maxipwgt = idxamax(nparts, mylpwgts);

    if (mylpwgts[maxipwgt] > 0 && !done) {
      send.key = -mylpwgts[maxipwgt];
      send.val = mype*nparts+maxipwgt;
    }
    else {
      send.key = 0;
      send.val = -1;
    }

    /* each processor sends its selection */
    MPI_Allgather((void *)&send, 2, IDX_DATATYPE, (void *)recv, 2, IDX_DATATYPE, ctrl->comm); 

    ikeysort(nparts, recv);
    if (recv[0].key == 0)
      break;

    /* now make as many assignments as possible */
    for (ii=0; ii<nparts; ii++) {
      i = recv[ii].val;

      if (i == -1)
        continue;

      j = i % nparts;
      k = i / nparts;
      if (map[j] == -1 && rowmap[k] == -1 && SimilarTpwgts(ctrl->tpwgts, ncon, j, k)) {
        map[j] = k;
        rowmap[k] = j;
        nmapped++;
        mylpwgts[j] = 0;
        if (mype == k)
          done = 1;
      }

      if (nmapped == nparts)
        break;
    }

    if (nmapped == nparts)
      break;
  }

  /* Map unmapped partitions */
  if (nmapped < nparts) {
    for (i=j=0; j<nparts && nmapped<nparts; j++) {
      if (map[j] == -1) {
        for (; i<nparts; i++) {
          if (rowmap[i] == -1 && SimilarTpwgts(ctrl->tpwgts, ncon, i, j)) {
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
  if (nmapped < nparts) {
    for (i=0; i<nparts; i++)
      map[i] = i; 
    IFSET(ctrl->dbglvl, DBG_REMAP, rprintf(ctrl, "Savings from parallel remapping: %0\n")); 
  }
  else {
    /* check for a savings */
    oldwgt = lpwgts[mype];
    newwgt = lpwgts[rowmap[mype]];
    nsaved = newwgt - oldwgt;
    gnsaved = GlobalSESum(ctrl, nsaved);

    /* undo everything if we don't see a savings */
    if (gnsaved <= 0) {
      for (i=0; i<nparts; i++)
        map[i] = i;
    }
    IFSET(ctrl->dbglvl, DBG_REMAP, rprintf(ctrl, "Savings from parallel remapping: %d\n", amax(0,gnsaved))); 
  }

  GKfree((void **)&recv, (void **)&mylpwgts, LTERM);

}


/*************************************************************************
* This function computes the assignment using the the objective the
* minimization of the total volume of data that needs to move
**************************************************************************/
int SimilarTpwgts(float *tpwgts, int ncon, int s1, int s2)
{
  int i;

  for (i=0; i<ncon; i++)
    if (fabs(tpwgts[s1*ncon+i]-tpwgts[s2*ncon+i]) > SMALLFLOAT)
      break;

  if (i == ncon)
    return 1;

  return 0;
}

