/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * initpart.c
 *
 * This file contains code that performs log(p) parallel multilevel
 * recursive bissection
 *
 * Started 3/4/96
 * George
 *
 * $Id: initpart.c,v 1.2 2003/07/21 17:18:49 karypis Exp $
 */

#include <parmetislib.h>


#define DEBUG_IPART_



/*************************************************************************
* This function is the entry point of the initial partition algorithm
* that does recursive bissection.
* This algorithm assembles the graph to all the processors and preceeds
* by parallelizing the recursive bisection step.
**************************************************************************/
void Moc_InitPartition_RB(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, j;
  int ncon, mype, npes, gnvtxs, ngroups;
  idxtype *xadj, *adjncy, *adjwgt, *vwgt;
  idxtype *part, *gwhere0, *gwhere1;
  idxtype *tmpwhere, *tmpvwgt, *tmpxadj, *tmpadjncy, *tmpadjwgt;
  GraphType *agraph;
  int lnparts, fpart, fpe, lnpes; 
  int twoparts=2, numflag = 0, wgtflag = 3, moptions[10], edgecut, max_cut;
  float *mytpwgts, mytpwgts2[2], lbvec[MAXNCON], lbsum, min_lbsum, wsum;
  MPI_Comm ipcomm;
  struct {
    float sum;
    int rank;
  } lpesum, gpesum;

  ncon = graph->ncon;
  ngroups = amax(amin(RIP_SPLIT_FACTOR, ctrl->npes), 1);

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->InitPartTmr));

  agraph = Moc_AssembleAdaptiveGraph(ctrl, graph, wspace);
  part = idxmalloc(agraph->nvtxs, "Moc_IP_RB: part");
  xadj = idxmalloc(agraph->nvtxs+1, "Moc_IP_RB: xadj");
  adjncy = idxmalloc(agraph->nedges, "Moc_IP_RB: adjncy");
  adjwgt = idxmalloc(agraph->nedges, "Moc_IP_RB: adjwgt");
  vwgt = idxmalloc(agraph->nvtxs*ncon, "Moc_IP_RB: vwgt");

  idxcopy(agraph->nvtxs*ncon, agraph->vwgt, vwgt);
  idxcopy(agraph->nvtxs+1, agraph->xadj, xadj);
  idxcopy(agraph->nedges, agraph->adjncy, adjncy);
  idxcopy(agraph->nedges, agraph->adjwgt, adjwgt);

  MPI_Comm_split(ctrl->gcomm, ctrl->mype % ngroups, 0, &ipcomm);
  MPI_Comm_rank(ipcomm, &mype);
  MPI_Comm_size(ipcomm, &npes);

  gnvtxs = agraph->nvtxs;

  gwhere0 = idxsmalloc(gnvtxs, 0, "Moc_IP_RB: gwhere0");
  gwhere1 = idxmalloc(gnvtxs, "Moc_IP_RB: gwhere1");

  /* ADD: this assumes that tpwgts for all constraints is the same */
  /* ADD: this is necessary because serial metis does not support the general case */
  mytpwgts = fsmalloc(ctrl->nparts, 0.0, "mytpwgts");
  for (i=0; i<ctrl->nparts; i++)
    for (j=0; j<ncon; j++)
      mytpwgts[i] += ctrl->tpwgts[i*ncon+j];
  for (i=0; i<ctrl->nparts; i++)
    mytpwgts[i] /= (float)ncon;

  /* Go into the recursive bisection */
  /* ADD: consider changing this to breadth-first type bisection */
  moptions[0] = 0;
  moptions[7] = ctrl->sync + (ctrl->mype % ngroups) + 1;

  lnparts = ctrl->nparts;
  fpart = fpe = 0;
  lnpes = npes;
  while (lnpes > 1 && lnparts > 1) {
    /* Determine the weights of the partitions */
    mytpwgts2[0] = ssum(lnparts/2, mytpwgts+fpart);
    mytpwgts2[1] = 1.0-mytpwgts2[0];

    if (ncon == 1)
      METIS_WPartGraphKway2(&agraph->nvtxs, agraph->xadj, agraph->adjncy,
        agraph->vwgt, agraph->adjwgt, &wgtflag, &numflag, &twoparts, mytpwgts2,
        moptions, &edgecut, part);
    else {
      METIS_mCPartGraphRecursive2(&agraph->nvtxs, &ncon, agraph->xadj,
        agraph->adjncy, agraph->vwgt, agraph->adjwgt, &wgtflag, &numflag,
        &twoparts, mytpwgts2, moptions, &edgecut, part);
    }

    wsum = ssum(lnparts/2, mytpwgts+fpart);
    sscale(lnparts/2, 1.0/wsum, mytpwgts+fpart);
    sscale(lnparts-lnparts/2, 1.0/(1.0-wsum), mytpwgts+fpart+lnparts/2);

    /* I'm picking the left branch */
    if (mype < fpe+lnpes/2) {
      Moc_KeepPart(agraph, wspace, part, 0);
      lnpes = lnpes/2;
      lnparts = lnparts/2;
    }
    else {
      Moc_KeepPart(agraph, wspace, part, 1);
      fpart = fpart + lnparts/2;
      fpe = fpe + lnpes/2;
      lnpes = lnpes - lnpes/2;
      lnparts = lnparts - lnparts/2;
    }
  }

  /* In case npes is greater than or equal to nparts */
  if (lnparts == 1) {
    /* Only the first process will assign labels (for the reduction to work) */
    if (mype == fpe) {
      for (i=0; i<agraph->nvtxs; i++) 
        gwhere0[agraph->label[i]] = fpart;
    }
  }
  /* In case npes is smaller than nparts */
  else {
    if (ncon == 1)
      METIS_WPartGraphKway2(&agraph->nvtxs, agraph->xadj, agraph->adjncy,
      agraph->vwgt, agraph->adjwgt, &wgtflag, &numflag, &lnparts, mytpwgts+fpart,
      moptions, &edgecut, part);
    else
      METIS_mCPartGraphRecursive2(&agraph->nvtxs, &ncon, agraph->xadj,
      agraph->adjncy, agraph->vwgt, agraph->adjwgt, &wgtflag, &numflag,
      &lnparts, mytpwgts+fpart, moptions, &edgecut, part);

    for (i=0; i<agraph->nvtxs; i++) 
      gwhere0[agraph->label[i]] = fpart + part[i];
  }

  MPI_Allreduce((void *)gwhere0, (void *)gwhere1, gnvtxs, IDX_DATATYPE, MPI_SUM, ipcomm);

  if (ngroups > 1) {
    tmpxadj = agraph->xadj;
    tmpadjncy = agraph->adjncy;
    tmpadjwgt = agraph->adjwgt;
    tmpvwgt = agraph->vwgt;
    tmpwhere = agraph->where;
    agraph->xadj = xadj;
    agraph->adjncy = adjncy;
    agraph->adjwgt = adjwgt;
    agraph->vwgt = vwgt;
    agraph->where = gwhere1;
    agraph->vwgt = vwgt;
    agraph->nvtxs = gnvtxs;
    Moc_ComputeSerialBalance(ctrl, agraph, gwhere1, lbvec);
    lbsum = ssum(ncon, lbvec);

    edgecut = ComputeSerialEdgeCut(agraph);
    MPI_Allreduce((void *)&edgecut, (void *)&max_cut, 1, MPI_INT, MPI_MAX, ctrl->gcomm);
    MPI_Allreduce((void *)&lbsum, (void *)&min_lbsum, 1, MPI_FLOAT, MPI_MIN, ctrl->gcomm);

    lpesum.sum = lbsum;
    if (min_lbsum < UNBALANCE_FRACTION * (float)(ncon)) {
      if (lbsum < UNBALANCE_FRACTION * (float)(ncon))
        lpesum.sum = (float) (edgecut);
      else
        lpesum.sum = (float) (max_cut);
    } 
    
    MPI_Comm_rank(ctrl->gcomm, &(lpesum.rank));
    MPI_Allreduce((void *)&lpesum, (void *)&gpesum, 1, MPI_FLOAT_INT, MPI_MINLOC, ctrl->gcomm);
    MPI_Bcast((void *)gwhere1, gnvtxs, IDX_DATATYPE, gpesum.rank, ctrl->gcomm);

    agraph->xadj = tmpxadj;
    agraph->adjncy = tmpadjncy;
    agraph->adjwgt = tmpadjwgt;
    agraph->vwgt = tmpvwgt;
    agraph->where = tmpwhere;
  }

  idxcopy(graph->nvtxs, gwhere1+graph->vtxdist[ctrl->mype], graph->where);

  FreeGraph(agraph);
  MPI_Comm_free(&ipcomm);
  GKfree((void **)&gwhere0, (void **)&gwhere1, (void **)&mytpwgts, (void **)&part, (void **)&xadj, (void **)&adjncy, (void **)&adjwgt, (void **)&vwgt, LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->InitPartTmr));

}


/*************************************************************************
* This function keeps one parts
**************************************************************************/
void Moc_KeepPart(GraphType *graph, WorkSpaceType *wspace, idxtype *part, int mypart)
{
  int h, i, j, k;
  int nvtxs, ncon, mynvtxs, mynedges;
  idxtype *xadj, *vwgt, *adjncy, *adjwgt, *label;
  idxtype *rename;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  label = graph->label;

  rename = idxmalloc(nvtxs, "Moc_KeepPart: rename");
 
  for (mynvtxs=0, i=0; i<nvtxs; i++) {
    if (part[i] == mypart)
      rename[i] = mynvtxs++;
  }

  for (mynvtxs=0, mynedges=0, j=xadj[0], i=0; i<nvtxs; i++) {
    if (part[i] == mypart) {
      for (; j<xadj[i+1]; j++) {
        k = adjncy[j];
        if (part[k] == mypart) {
          adjncy[mynedges] = rename[k];
          adjwgt[mynedges++] = adjwgt[j];
        }
      }
      j = xadj[i+1];  /* Save xadj[i+1] for later use */

      for (h=0; h<ncon; h++)
        vwgt[mynvtxs*ncon+h] = vwgt[i*ncon+h];
      label[mynvtxs] = label[i];
      xadj[++mynvtxs] = mynedges;

    }
    else {
      j = xadj[i+1];  /* Save xadj[i+1] for later use */
    }
  }

  graph->nvtxs = mynvtxs;
  graph->nedges = mynedges;

  free(rename);
}


