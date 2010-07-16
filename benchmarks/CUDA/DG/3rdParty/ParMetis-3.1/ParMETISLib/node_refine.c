/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * node_refine.c
 *
 * This file contains code that performs the k-way refinement
 *
 * Started 3/1/96
 * George
 *
 * $Id: node_refine.c,v 1.2 2003/07/21 17:18:50 karypis Exp $
 */

#include <parmetislib.h>

#define PackWeightWhereInfo(a, b) (((a)<<10) + (b))
#define SelectWhere(a) ((a)%1024)
#define SelectWeight(a) (((a)>>10))



/*************************************************************************
* This function computes the initial id/ed 
**************************************************************************/
void ComputeNodePartitionParams(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, j, nparts, nvtxs, nsep, firstvtx, lastvtx;
  idxtype *xadj, *ladjncy, *adjwgt, *vtxdist, *vwgt, *lpwgts, *gpwgts, *sepind;
  idxtype *where, *swhere, *rwhere;
  NRInfoType *rinfo, *myrinfo;
  int me, other, otherwgt;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->KWayInitTmr));

  nvtxs = graph->nvtxs;
  nparts = ctrl->nparts;

  vtxdist = graph->vtxdist;
  xadj = graph->xadj;
  ladjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  vwgt = graph->vwgt;

  where = graph->where;
  rinfo = graph->nrinfo = (NRInfoType *)GKmalloc(sizeof(NRInfoType)*nvtxs, "ComputeNodePartitionParams: rinfo");
  lpwgts = graph->lpwgts = idxsmalloc(2*nparts, 0, "ComputePartitionParams: lpwgts");
  gpwgts = graph->gpwgts = idxmalloc(2*nparts, "ComputePartitionParams: gpwgts");
  sepind = graph->sepind = idxmalloc(nvtxs, "ComputePartitionParams: sepind");

  firstvtx = vtxdist[ctrl->mype];
  lastvtx = vtxdist[ctrl->mype+1];

  /*------------------------------------------------------------
  / Send/Receive the where information of interface vertices.
  / Also use this to also encode the vwgt information of this
  / vertex. This is a hack, but it should work for now!
  /------------------------------------------------------------*/
  swhere = wspace->indices;
  rwhere = where + nvtxs;

  for (i=0; i<nvtxs; i++) {
    ASSERTP(ctrl, where[i] >= 0 && where[i] < 2*nparts, (ctrl, "%d\n", where[i]) );
    where[i] = PackWeightWhereInfo(vwgt[i], where[i]);
  }

  CommInterfaceData(ctrl, graph, where, swhere, rwhere); 

  /*------------------------------------------------------------
  / Compute now the degrees
  /------------------------------------------------------------*/
  for (nsep=i=0; i<nvtxs; i++) {
    me = SelectWhere(where[i]);
    ASSERT(ctrl, me >= 0 && me < 2*nparts);
    lpwgts[me] += vwgt[i];

    if (me >= nparts) {  /* If it is a separator vertex */
      sepind[nsep++] = i;
      lpwgts[2*nparts-1] += vwgt[i];

      myrinfo = rinfo+i;
      myrinfo->edegrees[0] = myrinfo->edegrees[1] = 0;

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        other = SelectWhere(where[ladjncy[j]]);
        otherwgt = SelectWeight(where[ladjncy[j]]);
        if (me != other)
          myrinfo->edegrees[other%2] += otherwgt;
      }
    }
  }
  graph->nsep = nsep;

  /* Finally, sum-up the partition weights */
  MPI_Allreduce((void *)lpwgts, (void *)gpwgts, 2*nparts, IDX_DATATYPE, MPI_SUM, ctrl->comm);
  graph->mincut = gpwgts[2*nparts-1];

#ifdef XX
  /* Print Weight information */
  if (ctrl->mype == 0) {
    for (i=0; i<nparts; i+=2) 
      printf("[%5d %5d %5d] ", gpwgts[i], gpwgts[i+1], gpwgts[nparts+i]); 
    printf("\n");
  }
#endif

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->KWayInitTmr));
}



/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void KWayNodeRefine(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace, int npasses, float ubfraction)
{
  int i, ii, j, k, pass, nvtxs, firstvtx, lastvtx, otherlastvtx, c, nmoves, 
      nlupd, nsupd, nnbrs, nchanged, nsep;
  int npes = ctrl->npes, mype = ctrl->mype, nparts = ctrl->nparts;
  idxtype *xadj, *ladjncy, *adjwgt, *vtxdist, *vwgt;
  idxtype *where, *lpwgts, *gpwgts, *sepind;
  idxtype *peind, *recvptr, *sendptr;
  idxtype *update, *supdate, *rupdate, *pe_updates, *htable, *changed;
  idxtype *badminpwgt, *badmaxpwgt;
  KeyValueType *swchanges, *rwchanges;
  int *nupds_pe;
  NRInfoType *rinfo, *myrinfo;
  int from, me, other, otherwgt, oldcut;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->KWayTmr));

  nvtxs = graph->nvtxs;

  vtxdist = graph->vtxdist;
  xadj = graph->xadj;
  ladjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  vwgt = graph->vwgt;

  firstvtx = vtxdist[mype];
  lastvtx = vtxdist[mype+1];

  where = graph->where;
  rinfo = graph->nrinfo;
  lpwgts = graph->lpwgts;
  gpwgts = graph->gpwgts;

  nsep = graph->nsep;
  sepind = graph->sepind;

  nnbrs = graph->nnbrs;
  peind = graph->peind;
  recvptr = graph->recvptr;
  sendptr = graph->sendptr;

  changed = idxmalloc(nvtxs, "KWayRefine: changed");
  rwchanges = wspace->pairs;
  swchanges = rwchanges + recvptr[nnbrs];

  update = idxmalloc(nvtxs, "KWayRefine: update");
  supdate = wspace->indices;
  rupdate = supdate + recvptr[nnbrs];
  nupds_pe = imalloc(npes, "KWayRefine: nupds_pe");

  htable = idxsmalloc(nvtxs+graph->nrecv, 0, "KWayRefine: lhtable");

  badminpwgt = wspace->pv1;
  badmaxpwgt = wspace->pv2;

  for (i=0; i<nparts; i+=2) {
    badminpwgt[i] = badminpwgt[i+1] = (1.0/ubfraction)*(gpwgts[i]+gpwgts[i+1])/2;
    badmaxpwgt[i] = badmaxpwgt[i+1] = ubfraction*(gpwgts[i]+gpwgts[i+1])/2;
  }

  IFSET(ctrl->dbglvl, DBG_REFINEINFO, PrintNodeBalanceInfo(ctrl, nparts, gpwgts, badminpwgt, badmaxpwgt, 1));

  for (pass=0; pass<npasses; pass++) {
    oldcut = graph->mincut;

    for (c=0; c<2; c++) {
      for (i=0; i<nparts; i+=2) {
        badminpwgt[i] = badminpwgt[i+1] = (1.0/ubfraction)*(gpwgts[i]+gpwgts[i+1])/2;
        badmaxpwgt[i] = badmaxpwgt[i+1] = ubfraction*(gpwgts[i]+gpwgts[i+1])/2;
      }

      nlupd = nsupd = nmoves = nchanged = 0;
      for (ii=0; ii<nsep; ii++) {
        i = sepind[ii];
        from = SelectWhere(where[i]);

        ASSERT(ctrl, from >= nparts);

        /* Go through the loop if gain is possible for the separator vertex */
        if (rinfo[i].edegrees[(c+1)%2] <= vwgt[i]) {
          other = from%nparts+c;  /* It is one-sided move so we know where it goes */

          if (gpwgts[other]+vwgt[i] > badmaxpwgt[other]) {
            /* printf("Skip because of weight! %d\n", vwgt[i]-rinfo[i].edegrees[(c+1)%2]); */
            continue;   /* We cannot move it there because it gets too heavy */
          }

          /* Update where, weight, and ID/ED information of the vertex you moved */
          where[i] = PackWeightWhereInfo(vwgt[i], other);

          /* Remove this vertex from the sepind. Note the trick for looking at the sepind[ii] again */
          sepind[ii--] = sepind[--nsep]; 

          /* myprintf(ctrl, "Vertex %d [%d %d] is moving to %d from %d [%d]\n", i+firstvtx, vwgt[i], rinfo[i].edegrees[(c+1)%2], other, from, SelectWhere(where[i])); */

          lpwgts[from] -= vwgt[i];
          lpwgts[2*nparts-1] -= vwgt[i];
          lpwgts[other] += vwgt[i];
          gpwgts[other] += vwgt[i];

          /* 
           * Put the vertices adjacent to i that belong to either the separator or
           * the (c+1)%2 partition into the update array 
           */
          for (j=xadj[i]; j<xadj[i+1]; j++) {
            k = ladjncy[j];
            if (htable[k] == 0 && SelectWhere(where[k]) != other) {
              htable[k] = 1;
              if (k<nvtxs)
                update[nlupd++] = k;
              else
                supdate[nsupd++] = k;
            }
          }
          nmoves++;
          if (graph->pexadj[i+1]-graph->pexadj[i] > 0)
            changed[nchanged++] = i;
        }
      }

      /* myprintf(ctrl, "nmoves: %d, nlupd: %d, nsupd: %d\n", nmoves, nlupd, nsupd); */

      /* Tell everybody interested what the new where[] info is for the interface vertices */
      CommChangedInterfaceData(ctrl, graph, nchanged, changed, where, swchanges, rwchanges, wspace->pv4); 


      IFSET(ctrl->dbglvl, DBG_RMOVEINFO, rprintf(ctrl, "\t[%d %d], [%d %d %d]\n", 
                pass, c, GlobalSESum(ctrl, nmoves), GlobalSESum(ctrl, nsupd), GlobalSESum(ctrl, nlupd)));


      /*-------------------------------------------------------------
      / Time to communicate with processors to send the vertices
      / whose degrees need to be update.
      /-------------------------------------------------------------*/
      /* Issue the receives first */
      for (i=0; i<nnbrs; i++) {
        MPI_Irecv((void *)(rupdate+sendptr[i]), sendptr[i+1]-sendptr[i], IDX_DATATYPE,
                  peind[i], 1, ctrl->comm, ctrl->rreq+i);
      }

      /* Issue the sends next. This needs some preporcessing */
      for (i=0; i<nsupd; i++) {
        htable[supdate[i]] = 0;
        supdate[i] = graph->imap[supdate[i]];
      }
      iidxsort(nsupd, supdate);

      for (j=i=0; i<nnbrs; i++) {
        otherlastvtx = vtxdist[peind[i]+1];
        for (k=j; k<nsupd && supdate[k] < otherlastvtx; k++); 
        MPI_Isend((void *)(supdate+j), k-j, IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
        j = k;
      }

      /* OK, now get into the loop waiting for the send/recv operations to finish */
      MPI_Waitall(nnbrs, ctrl->rreq, ctrl->statuses);
      for (i=0; i<nnbrs; i++) 
        MPI_Get_count(ctrl->statuses+i, IDX_DATATYPE, nupds_pe+i);
      MPI_Waitall(nnbrs, ctrl->sreq, ctrl->statuses);


      /*-------------------------------------------------------------
      / Place the received to-be updated vertices into update[] 
      /-------------------------------------------------------------*/
      for (i=0; i<nnbrs; i++) {
        pe_updates = rupdate+sendptr[i];
        for (j=0; j<nupds_pe[i]; j++) {
          k = pe_updates[j];
          if (htable[k-firstvtx] == 0) {
            htable[k-firstvtx] = 1;
            update[nlupd++] = k-firstvtx;
          }
        }
      }


      /*-------------------------------------------------------------
      / Update the where information of the vertices that are pulled
      / into the separator.
      /-------------------------------------------------------------*/
      nchanged = 0;
      for (ii=0; ii<nlupd; ii++) {
        i = update[ii];
        me = SelectWhere(where[i]);
        if (me < nparts && me%2 == (c+1)%2) { /* This vertex is pulled into the separator */
          lpwgts[me] -= vwgt[i];
          where[i] = PackWeightWhereInfo(vwgt[i], nparts+me-(me%2)); 
          sepind[nsep++] = i;  /* Put the vertex into the sepind array */
          if (graph->pexadj[i+1]-graph->pexadj[i] > 0)
            changed[nchanged++] = i;

          lpwgts[SelectWhere(where[i])] += vwgt[i];
          lpwgts[2*nparts-1] += vwgt[i];
          /* myprintf(ctrl, "Vertex %d moves into the separator from %d to %d\n", i+firstvtx, me, SelectWhere(where[i])); */
        }
      }

      /* Tell everybody interested what the new where[] info is for the interface vertices */
      CommChangedInterfaceData(ctrl, graph, nchanged, changed, where, swchanges, rwchanges, wspace->pv4); 


      /*-------------------------------------------------------------
      / Update the rinfo of the vertices in the update[] array
      /-------------------------------------------------------------*/
      for (ii=0; ii<nlupd; ii++) {
        i = update[ii];
        ASSERT(ctrl, htable[i] == 1);

        htable[i] = 0;

        me = SelectWhere(where[i]);
        if (me >= nparts) {  /* If it is a separator vertex */
          /* myprintf(ctrl, "Updating %d %d\n", i+firstvtx, me); */

          myrinfo = rinfo+i;
          myrinfo->edegrees[0] = myrinfo->edegrees[1] = 0;

          for (j=xadj[i]; j<xadj[i+1]; j++) {
            other = SelectWhere(where[ladjncy[j]]);
            otherwgt = SelectWeight(where[ladjncy[j]]);
            if (me != other)
              myrinfo->edegrees[other%2] += otherwgt;
          }
        }
      }

      /* Finally, sum-up the partition weights */
      MPI_Allreduce((void *)lpwgts, (void *)gpwgts, 2*nparts, IDX_DATATYPE, MPI_SUM, ctrl->comm);
      graph->mincut = gpwgts[2*nparts-1];

      IFSET(ctrl->dbglvl, DBG_REFINEINFO, PrintNodeBalanceInfo(ctrl, nparts, gpwgts, badminpwgt, badmaxpwgt, 0));
    }

    if (graph->mincut == oldcut)
      break;
  }

  /* Go and clear-up the where array */
  for (i=0; i<nvtxs+graph->nrecv; i++)
    where[i] = SelectWhere(where[i]);

  GKfree((void **)&update, (void **)&nupds_pe, (void **)&htable, (void **)&changed, LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->KWayTmr));
}




/*************************************************************************
* This function prints balance information for the parallel k-section 
* refinement algorithm
**************************************************************************/
void PrintNodeBalanceInfo(CtrlType *ctrl, int nparts, idxtype *gpwgts, idxtype *badminpwgt, idxtype *badmaxpwgt, int title)
{
  int i;

  if (ctrl->mype == 0) {
    if (title)
      printf("K-way sep-refinement: TotalSep: %d, ", gpwgts[2*nparts-1]);
    else
      printf("\tTotalSep: %d, ", gpwgts[2*nparts-1]);

    for (i=0; i<nparts; i+=2) 
      printf(" [%5d %5d %5d %5d %5d]", gpwgts[i], gpwgts[i+1], gpwgts[nparts+i], badminpwgt[i], badmaxpwgt[i]); 
    printf("\n");
  }
  MPI_Barrier(ctrl->comm);
}

