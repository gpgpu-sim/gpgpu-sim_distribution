/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mgrsetup.c
 *
 * This file contain various graph setting up routines
 *
 * Started 10/19/96
 * George
 *
 * $Id: grsetup.c,v 1.7 2003/07/23 00:54:55 karypis Exp $
 *
 */

#include <parmetislib.h>



/*************************************************************************
* This function setsup the CtrlType structure
**************************************************************************/
GraphType *Moc_SetUpGraph(CtrlType *ctrl, int ncon, idxtype *vtxdist, idxtype *xadj, 
                          idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt, int *wgtflag)
{
  int i, j;
  GraphType *graph;
  int ltvwgts[MAXNCON];

  graph = CreateGraph();
  graph->level   = 0;
  graph->gnvtxs  = vtxdist[ctrl->npes];
  graph->nvtxs   = vtxdist[ctrl->mype+1]-vtxdist[ctrl->mype];
  graph->ncon    = ncon;
  graph->nedges  = xadj[graph->nvtxs];
  graph->xadj    = xadj;
  graph->vwgt    = vwgt;
  graph->adjncy  = adjncy;
  graph->adjwgt  = adjwgt;
  graph->vtxdist = vtxdist;


  if (((*wgtflag)&2) == 0) 
    graph->vwgt = idxsmalloc(graph->nvtxs*ncon, 1, "Par_KMetis: vwgt");

  if (((*wgtflag)&1) == 0) 
    graph->adjwgt = idxsmalloc(graph->nedges, 1, "Par_KMetis: adjwgt");

  /* compute tvwgts */
  for (j=0; j<ncon; j++)
    ltvwgts[j] = 0;

  for (i=0; i<graph->nvtxs; i++)
    for (j=0; j<ncon; j++)
      ltvwgts[j] += graph->vwgt[i*ncon+j];

  for (j=0; j<ncon; j++)
    ctrl->tvwgts[j] = GlobalSESum(ctrl, ltvwgts[j]);

  /* check for zero wgt constraints */
  for (i=0; i<ncon; i++) {
    /* ADD: take care of the case in which tvwgts is zero */
    if (ctrl->tvwgts[i] == 0) {
      rprintf(ctrl, "ERROR: sum weight for constraint %d is zero\n", i);
      MPI_Finalize();
      exit(-1);
    }
  }

  /* compute nvwgts */
  graph->nvwgt = fmalloc(graph->nvtxs*ncon, "graph->nvwgt");
  for (i=0; i<graph->nvtxs; i++) {
    for (j=0; j<ncon; j++)
      graph->nvwgt[i*ncon+j] = (float)(graph->vwgt[i*ncon+j]) / (float)(ctrl->tvwgts[j]);
  }

  srand(ctrl->seed);

  return graph;
}


/*************************************************************************
* This function setsup the CtrlType structure
**************************************************************************/
void SetUpCtrl(CtrlType *ctrl, int nparts, int dbglvl, MPI_Comm comm)
{

  MPI_Comm_dup(comm, &(ctrl->gcomm));
  MPI_Comm_rank(ctrl->gcomm, &ctrl->mype);
  MPI_Comm_size(ctrl->gcomm, &ctrl->npes);

  ctrl->dbglvl  = dbglvl;
  ctrl->nparts  = nparts;    /* Set the # of partitions is de-coupled from the # of domains */
  ctrl->comm    = ctrl->gcomm;
  ctrl->xyztype = XYZ_SPFILL;

  srand(ctrl->mype);
}


/*************************************************************************
* This function changes the numbering from 1 to 0 or 0 to 1
**************************************************************************/
void ChangeNumbering(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *part, int npes, int mype, int from)
{
  int i, nvtxs, nedges;

  if (from == 1) {  /* Change it from 1 to 0 */
    for (i=0; i<npes+1; i++)
      vtxdist[i]--;

    nvtxs = vtxdist[mype+1]-vtxdist[mype];
    for (i=0; i<nvtxs+1; i++) 
      xadj[i]--;

    nedges = xadj[nvtxs];
    for (i=0; i<nedges; i++) 
      adjncy[i]--;
  }
  else {  /* Change it from 0 to 1 */
    nvtxs = vtxdist[mype+1]-vtxdist[mype];
    nedges = xadj[nvtxs];

    for (i=0; i<npes+1; i++) 
      vtxdist[i]++;

    for (i=0; i<nvtxs+1; i++) 
      xadj[i]++; 

    for (i=0; i<nedges; i++) 
      adjncy[i]++; 

    for (i=0; i<nvtxs; i++)
      part[i]++;

  }
}


/*************************************************************************
* This function changes the numbering from 1 to 0 or 0 to 1
**************************************************************************/
void ChangeNumberingMesh(idxtype *elmdist, idxtype *elements, idxtype *xadj, 
                         idxtype *adjncy, idxtype *part, int npes, int mype, 
			 int elmntlen, int from)
{
  int i, nelms, nedges;

  if (from == 1) {  /* Change it from 1 to 0 */
    for (i=0; i<npes+1; i++)
      elmdist[i]--;

    for (i=0; i<elmntlen; i++) 
      elements[i]--;
  }
  else {  /* Change it from 0 to 1 */
    nelms = elmdist[mype+1]-elmdist[mype];
    nedges = xadj[nelms];

    for (i=0; i<npes+1; i++) 
      elmdist[i]++;

    for (i=0; i<elmntlen; i++) 
      elements[i]++;

    for (i=0; i<nelms+1; i++) 
      xadj[i]++; 

    for (i=0; i<nedges; i++) 
      adjncy[i]++; 

    if (part != NULL)
      for (i=0; i<nelms; i++)
        part[i]++;
  }
}


/*************************************************************************
* This function changes the numbering from 1 to 0 or 0 to 1
**************************************************************************/
void ChangeNumberingMesh2(idxtype *elmdist, idxtype *eptr, idxtype *eind, 
                          idxtype *xadj, idxtype *adjncy, idxtype *part, 
			  int npes, int mype, int from)
{
  int i, nelms;

  nelms = elmdist[mype+1]-elmdist[mype];

  if (from == 1) {  /* Change it from 1 to 0 */
    for (i=0; i<npes+1; i++)
      elmdist[i]--;

    for (i=0; i<nelms+1; i++) 
      eptr[i]--;

    for (i=0; i<eptr[nelms]; i++) 
      eind[i]--;
  }
  else {  /* Change it from 0 to 1 */
    for (i=0; i<npes+1; i++) 
      elmdist[i]++;

    for (i=0; i<nelms+1; i++) 
      eptr[i]++;

    for (i=0; i<eptr[nelms]; i++) 
      eind[i]++;

    for (i=0; i<nelms+1; i++) 
      xadj[i]++; 

    for (i=0; i<xadj[nelms]; i++) 
      adjncy[i]++; 

    if (part != NULL)
      for (i=0; i<nelms; i++)
        part[i]++;
  }
}




/*************************************************************************
* This function randomly permutes the locally stored adjacency lists
**************************************************************************/
void GraphRandomPermute(GraphType *graph) 
{
  int i, j, k, tmp;

  for (i=0; i<graph->nvtxs; i++) {
    for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++) {
      k = graph->xadj[i] + RandomInRange(graph->xadj[i+1]-graph->xadj[i]);
      SWAP(graph->adjncy[j], graph->adjncy[k], tmp);
      SWAP(graph->adjwgt[j], graph->adjwgt[k], tmp);
    }
  }
}


/*************************************************************************
* This function computes movement statistics for adaptive refinement
* schemes
**************************************************************************/
void ComputeMoveStatistics(CtrlType *ctrl, GraphType *graph, int *nmoved, int *maxin, int *maxout)
{
  int i, j, nvtxs;
  idxtype *vwgt, *where;
  idxtype *lpvtxs, *gpvtxs;

  nvtxs = graph->nvtxs;
  vwgt = graph->vwgt;
  where = graph->where;

  lpvtxs = idxsmalloc(ctrl->nparts, 0, "ComputeMoveStatistics: lpvtxs");
  gpvtxs = idxsmalloc(ctrl->nparts, 0, "ComputeMoveStatistics: gpvtxs");

  for (j=i=0; i<nvtxs; i++) {
    lpvtxs[where[i]]++;
    if (where[i] != ctrl->mype)
      j++;
  }

  /* PrintVector(ctrl, ctrl->npes, 0, lpvtxs, "Lpvtxs: "); */

  MPI_Allreduce((void *)lpvtxs, (void *)gpvtxs, ctrl->nparts, IDX_DATATYPE, MPI_SUM, ctrl->comm);

  *nmoved = GlobalSESum(ctrl, j);
  *maxout = GlobalSEMax(ctrl, j);
  *maxin = GlobalSEMax(ctrl, gpvtxs[ctrl->mype]-(nvtxs-j));

  GKfree((void **)&lpvtxs, (void **)&gpvtxs, LTERM);
}
