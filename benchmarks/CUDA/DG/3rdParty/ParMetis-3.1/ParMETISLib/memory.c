/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * memory.c
 *
 * This file contains routines that deal with memory allocation
 *
 * Started 2/24/96
 * George
 *
 * $Id: memory.c,v 1.3 2003/07/30 18:37:59 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function allocate various pools of memory
**************************************************************************/
void PreAllocateMemory(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  wspace->nlarge  = 2*graph->nedges;

  wspace->maxcore = 8*graph->nedges+1;
  wspace->core    = idxmalloc(wspace->maxcore, "PreAllocateMemory: wspace->core");

  wspace->pairs   = (KeyValueType *)wspace->core;
  wspace->indices = (idxtype *)(wspace->pairs + wspace->nlarge);
  wspace->degrees = (EdgeType *)(wspace->indices + wspace->nlarge);


  wspace->pv1 = idxmalloc(ctrl->nparts+ctrl->npes+1, "PreAllocateMemory: wspace->pv?");
  wspace->pv2 = idxmalloc(ctrl->nparts+ctrl->npes+1, "PreAllocateMemory: wspace->pv?");
  wspace->pv3 = idxmalloc(ctrl->nparts+ctrl->npes+1, "PreAllocateMemory: wspace->pv?");
  wspace->pv4 = idxmalloc(ctrl->nparts+ctrl->npes+1, "PreAllocateMemory: wspace->pv?");

  wspace->pepairs1 = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*(ctrl->nparts+ctrl->npes+1), "PreAllocateMemory: wspace->pepairs?");
  wspace->pepairs2 = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*(ctrl->nparts+ctrl->npes+1), "PreAllocateMemory: wspace->pepairs?");

}


/*************************************************************************
* This function de-allocate various pools of memory
**************************************************************************/
void FreeWSpace(WorkSpaceType *wspace)
{

  GKfree((void **)&wspace->core, 
         (void **)&wspace->pv1, 
         (void **)&wspace->pv2, 
         (void **)&wspace->pv3,
         (void **)&wspace->pv4, 
         (void **)&wspace->pepairs1, 
         (void **)&wspace->pepairs2, 
         LTERM);
}


/*************************************************************************
* This function de-allocates memory allocated for the control structures
**************************************************************************/
void FreeCtrl(CtrlType *ctrl)
{
  MPI_Comm_free(&(ctrl->gcomm));
}


/*************************************************************************
* This function creates a CoarseGraphType data structure and initializes
* the various fields
**************************************************************************/
GraphType *CreateGraph(void)
{
  GraphType *graph;

  graph = (GraphType *)GKmalloc(sizeof(GraphType), "CreateCoarseGraph: graph");

  InitGraph(graph);

  return graph;
}


/*************************************************************************
* This function creates a CoarseGraphType data structure and initializes
* the various fields
**************************************************************************/
void InitGraph(GraphType *graph) 
{
  graph->gnvtxs = graph->nvtxs = graph->nedges = graph->nsep = -1;
  graph->nnbrs = graph->nrecv = graph->nsend = graph->nlocal = -1;
  graph->xadj = graph->vwgt = graph->vsize = graph->adjncy = graph->adjwgt = NULL;
  graph->nvwgt = NULL;
  graph->vtxdist = NULL;
  graph->match = graph->cmap = NULL;
  graph->label = NULL;

  graph->peind = NULL;
  graph->sendptr = graph->sendind = graph->recvptr = graph->recvind = NULL;
  graph->imap = NULL;
  graph->pexadj = graph->peadjncy = graph->peadjloc = NULL;
  graph->lperm = NULL;

  graph->slens = graph->rlens = NULL;
  graph->rcand = NULL;

  graph->where = graph->home = graph->lpwgts = graph->gpwgts = NULL;
  graph->lnpwgts = graph->gnpwgts = NULL;
  graph->rinfo = NULL;

  graph->nrinfo = NULL;
  graph->sepind = NULL;

  graph->coarser = graph->finer = NULL;

}

/*************************************************************************
* This function deallocates any memory stored in a graph
**************************************************************************/
void FreeGraph(GraphType *graph) 
{

  GKfree((void **)&graph->xadj, 
         (void **)&graph->vwgt,
         (void **)&graph->nvwgt,
         (void **)&graph->vsize,
         (void **)&graph->adjncy,
         (void **)&graph->adjwgt,
         (void **)&graph->vtxdist, 
         (void **)&graph->match, 
         (void **)&graph->cmap, 
         (void **)&graph->lperm, 
         (void **)&graph->label, 
         (void **)&graph->where, 
         (void **)&graph->home, 
         (void **)&graph->rinfo, 
         (void **)&graph->nrinfo, 
         (void **)&graph->sepind,
         (void **)&graph->lpwgts, 
         (void **)&graph->gpwgts, 
         (void **)&graph->lnpwgts, 
         (void **)&graph->gnpwgts, 
         (void **)&graph->peind, 
         (void **)&graph->sendptr, 
         (void **)&graph->sendind, 
         (void **)&graph->recvptr, 
         (void **)&graph->recvind, 
         (void **)&graph->imap,
         (void **)&graph->rlens,
         (void **)&graph->slens,
         (void **)&graph->rcand,
         (void **)&graph->pexadj,
         (void **)&graph->peadjncy,
         (void **)&graph->peadjloc,
         LTERM);

  free(graph);
}



/*************************************************************************
* This function deallocates any memory stored in a graph
**************************************************************************/
void FreeInitialGraphAndRemap(GraphType *graph, int wgtflag) 
{
  int i, nedges;
  idxtype *adjncy, *imap;

  nedges = graph->nedges;
  adjncy = graph->adjncy;
  imap = graph->imap;

  if (imap != NULL) {
    for (i=0; i<nedges; i++)
      adjncy[i] = imap[adjncy[i]];  /* Apply local to global transformation */
  }

  /* Free Metis's things */
  GKfree((void **)&graph->match, 
         (void **)&graph->cmap, 
         (void **)&graph->lperm, 
         (void **)&graph->where, 
         (void **)&graph->label, 
         (void **)&graph->rinfo, 
         (void **)&graph->nrinfo, 
         (void **)&graph->nvwgt, 
         (void **)&graph->lpwgts, 
         (void **)&graph->gpwgts, 
         (void **)&graph->lnpwgts, 
         (void **)&graph->gnpwgts, 
         (void **)&graph->sepind,
         (void **)&graph->peind, 
         (void **)&graph->sendptr, 
         (void **)&graph->sendind, 
         (void **)&graph->recvptr, 
         (void **)&graph->recvind, 
         (void **)&graph->imap,
         (void **)&graph->rlens,
         (void **)&graph->slens,
         (void **)&graph->rcand,
         (void **)&graph->pexadj,
         (void **)&graph->peadjncy,
         (void **)&graph->peadjloc,
         LTERM);

  if ((wgtflag&2) == 0) 
    GKfree((void **)&graph->vwgt, (void **)&graph->vsize, LTERM);
  if ((wgtflag&1) == 0) 
    GKfree((void **)&graph->adjwgt, LTERM);

  free(graph);
}
