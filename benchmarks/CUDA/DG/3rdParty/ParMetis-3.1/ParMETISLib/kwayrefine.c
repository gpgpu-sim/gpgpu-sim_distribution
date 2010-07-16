/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * medge_refine.c
 *
 * This file contains code that performs the k-way refinement
 *
 * Started 3/1/96
 * George
 *
 * $Id: kwayrefine.c,v 1.2 2003/07/21 17:18:49 karypis Exp $
 */

#include <parmetislib.h>

#define ProperSide(c, from, other) \
              (((c) == 0 && (from)-(other) < 0) || ((c) == 1 && (from)-(other) > 0))

/*************************************************************************
* This function projects a partition.
**************************************************************************/
void Moc_ProjectPartition(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, nvtxs, nnbrs = -1, firstvtx, cfirstvtx;
  idxtype *match, *cmap, *where, *cwhere;
  idxtype *peind, *slens = NULL, *rlens = NULL;
  KeyValueType *rcand, *scand = NULL;
  GraphType *cgraph;


  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ProjectTmr));

  cgraph = graph->coarser;
  cwhere = cgraph->where;
  cfirstvtx = cgraph->vtxdist[ctrl->mype];

  nvtxs = graph->nvtxs;
  match = graph->match;
  cmap = graph->cmap;
  where = graph->where = idxmalloc(nvtxs+graph->nrecv, "ProjectPartition: graph->where");
  firstvtx = graph->vtxdist[ctrl->mype];


  if (graph->match_type == MATCH_GLOBAL) {  /* Only if global matching is on */
    /*------------------------------------------------------------
    / Start the transmission of the remote where information 
    /------------------------------------------------------------*/
    scand = wspace->pairs;
    nnbrs = graph->nnbrs;
    peind = graph->peind;
    slens = graph->slens;
    rlens = graph->rlens;
    rcand = graph->rcand;

    /* Issue the receives first */
    for (i=0; i<nnbrs; i++) {
      if (slens[i+1]-slens[i] > 0) /* Issue a receive only if you are getting something */
        MPI_Irecv((void *)(scand+slens[i]), 2*(slens[i+1]-slens[i]), IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->rreq+i);
    }

#ifdef DEBUG_PROJECT
    PrintPairs(ctrl, rlens[nnbrs], rcand, "rcand"); 
#endif

    /* Put the where[rcand[].key] into the val field */
    for (i=0; i<rlens[nnbrs]; i++) {
      ASSERT(ctrl, rcand[i].val >= 0 && rcand[i].val < cgraph->nvtxs);
      rcand[i].val = cwhere[rcand[i].val];
    }

#ifdef DEBUG_PROJECT
    PrintPairs(ctrl, rlens[nnbrs], rcand, "rcand");
    PrintVector(ctrl, nvtxs, firstvtx, cmap, "cmap");
#endif

    /* Issue the sends next */
    for (i=0; i<nnbrs; i++) {
      if (rlens[i+1]-rlens[i] > 0) /* Issue a send only if you are sending something */
        MPI_Isend((void *)(rcand+rlens[i]), 2*(rlens[i+1]-rlens[i]), IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
    }
  }

  /*------------------------------------------------------------
  / Project local vertices first
  /------------------------------------------------------------*/
  for (i=0; i<nvtxs; i++) {
    if (match[i] >= KEEP_BIT) {
      ASSERT(ctrl, cmap[i]-cfirstvtx>=0 && cmap[i]-cfirstvtx<cgraph->nvtxs);
      where[i] = cwhere[cmap[i]-cfirstvtx];
    }
  }

  if (graph->match_type == MATCH_GLOBAL) {  /* Only if global matching is on */
    /*------------------------------------------------------------
    / Wait for the nonblocking operations to finish
    /------------------------------------------------------------*/
    for (i=0; i<nnbrs; i++) {
      if (rlens[i+1]-rlens[i] > 0)  
        MPI_Wait(ctrl->sreq+i, &ctrl->status);
    }
    for (i=0; i<nnbrs; i++) {
      if (slens[i+1]-slens[i] > 0)  
        MPI_Wait(ctrl->rreq+i, &ctrl->status);
    }

#ifdef DEBUG_PROJECT
    PrintPairs(ctrl, slens[nnbrs], scand, "scand"); 
#endif

    /*------------------------------------------------------------
    / Project received vertices now
    /------------------------------------------------------------*/
    for (i=0; i<slens[nnbrs]; i++) {
      ASSERTP(ctrl, scand[i].key-firstvtx>=0 && scand[i].key-firstvtx<graph->nvtxs, (ctrl, "%d %d %d\n", scand[i].key, firstvtx, graph->nvtxs));
      where[scand[i].key-firstvtx] = scand[i].val;
    }
  }


  FreeGraph(graph->coarser);
  graph->coarser = NULL;

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ProjectTmr));
}



/*************************************************************************
* This function computes the initial id/ed 
**************************************************************************/
void Moc_ComputePartitionParams(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int h, i, j, k;
  int nvtxs, ncon;
  int firstvtx, lastvtx;
  idxtype *xadj, *ladjncy, *adjwgt, *vtxdist;
  float *lnpwgts, *gnpwgts;
  idxtype *where, *swhere, *rwhere;
  RInfoType *rinfo, *myrinfo;
  EdgeType *edegrees;
  int me, other;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->KWayInitTmr));


  nvtxs = graph->nvtxs;
  ncon = graph->ncon;

  vtxdist = graph->vtxdist;
  xadj = graph->xadj;
  ladjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  where = graph->where;
  rinfo = graph->rinfo = (RInfoType *)GKmalloc(sizeof(RInfoType)*nvtxs, "CPP: rinfo");
  lnpwgts = graph->lnpwgts = fmalloc(ctrl->nparts*ncon, "CPP: lnpwgts");
  gnpwgts = graph->gnpwgts = fmalloc(ctrl->nparts*ncon, "CPP: gnpwgts");

  sset(ctrl->nparts*ncon, 0, lnpwgts);

  firstvtx = vtxdist[ctrl->mype];
  lastvtx = vtxdist[ctrl->mype+1];

  /*------------------------------------------------------------
  / Send/Receive the where information of interface vertices
  /------------------------------------------------------------*/
  swhere = wspace->indices;
  rwhere = where + nvtxs;

  CommInterfaceData(ctrl, graph, where, swhere, rwhere); 

#ifdef DEBUG_COMPUTEPPARAM
  PrintVector(ctrl, nvtxs, firstvtx, where, "where");
#endif

  ASSERT(ctrl, wspace->nlarge >= xadj[nvtxs]);

  /*------------------------------------------------------------
  / Compute now the id/ed degrees
  /------------------------------------------------------------*/
  graph->lmincut = 0;
  for (i=0; i<nvtxs; i++) {
    me = where[i];
    myrinfo = rinfo+i;

    for (h=0; h<ncon; h++)
      lnpwgts[me*ncon+h] += graph->nvwgt[i*ncon+h];

    myrinfo->degrees = wspace->degrees + xadj[i];
    myrinfo->ndegrees = myrinfo->id = myrinfo->ed = 0;

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      if (me == where[ladjncy[j]])
        myrinfo->id += adjwgt[j];
      else
        myrinfo->ed += adjwgt[j];
    }


    if (myrinfo->ed > 0) {  /* Time to do some serious work */
      graph->lmincut += myrinfo->ed;
      edegrees = myrinfo->degrees;

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        other = where[ladjncy[j]];
        if (me != other) {
          for (k=0; k<myrinfo->ndegrees; k++) {
            if (edegrees[k].edge == other) {
              edegrees[k].ewgt += adjwgt[j];
              break;
            }
          }
          if (k == myrinfo->ndegrees) {
            edegrees[k].edge = other;
            edegrees[k].ewgt = adjwgt[j];
            myrinfo->ndegrees++;
          }
          ASSERT(ctrl, myrinfo->ndegrees <= xadj[i+1]-xadj[i]);
        }
      }
    }
  }

#ifdef DEBUG_COMPUTEPPARAM
  PrintVector(ctrl, ctrl->nparts*ncon, 0, lnpwgts, "lnpwgts");
#endif

  /* Finally, sum-up the partition weights */
  MPI_Allreduce((void *)lnpwgts, (void *)gnpwgts, ctrl->nparts*ncon, MPI_FLOAT, MPI_SUM, ctrl->comm);

  graph->mincut = GlobalSESum(ctrl, graph->lmincut)/2;

#ifdef DEBUG_COMPUTEPPARAM
  PrintVector(ctrl, ctrl->nparts*ncon, 0, gnpwgts, "gnpwgts");
#endif

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->KWayInitTmr));
}

