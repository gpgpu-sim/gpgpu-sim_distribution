/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mdrivers.c
 *
 * This file contains the driving routines for the various parallel
 * multilevel partitioning and repartitioning algorithms
 *
 * Started 11/19/96
 * George
 *
 * $Id: mdrivers.c,v 1.3 2003/07/22 20:29:06 karypis Exp $
 *
 */

#include <parmetislib.h>



/*************************************************************************
* This function is the driver to the multi-constraint partitioning algorithm.
**************************************************************************/
void Moc_Global_Partition(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, ncon, nparts;
  float ftmp, ubavg, lbavg, lbvec[MAXNCON];
 
  ncon = graph->ncon;
  nparts = ctrl->nparts;
  ubavg = savg(graph->ncon, ctrl->ubvec);

  SetUp(ctrl, graph, wspace);

  if (ctrl->dbglvl&DBG_PROGRESS) {
    rprintf(ctrl, "[%6d %8d %5d %5d] [%d] [", graph->gnvtxs, GlobalSESum(ctrl, graph->nedges),
	    GlobalSEMin(ctrl, graph->nvtxs), GlobalSEMax(ctrl, graph->nvtxs), ctrl->CoarsenTo);
    for (i=0; i<ncon; i++)
      rprintf(ctrl, " %.3f", GlobalSEMinFloat(ctrl,graph->nvwgt[samin_strd(graph->nvtxs, graph->nvwgt+i, ncon)*ncon+i]));  
    rprintf(ctrl, "] [");
    for (i=0; i<ncon; i++)
      rprintf(ctrl, " %.3f", GlobalSEMaxFloat(ctrl, graph->nvwgt[samax_strd(graph->nvtxs, graph->nvwgt+i, ncon)*ncon+i]));  
    rprintf(ctrl, "]\n");
  }

  if (graph->gnvtxs < 1.3*ctrl->CoarsenTo ||
	(graph->finer != NULL &&
	graph->gnvtxs > graph->finer->gnvtxs*COARSEN_FRACTION)) {

    /* Done with coarsening. Find a partition */
    graph->where = idxmalloc(graph->nvtxs+graph->nrecv, "graph->where");
    Moc_InitPartition_RB(ctrl, graph, wspace);

    if (ctrl->dbglvl&DBG_PROGRESS) {
      Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
      rprintf(ctrl, "nvtxs: %10d, balance: ", graph->gnvtxs);
      for (i=0; i<graph->ncon; i++) 
        rprintf(ctrl, "%.3f ", lbvec[i]);
      rprintf(ctrl, "\n");
    }

    /* In case no coarsening took place */
    if (graph->finer == NULL) {
      Moc_ComputePartitionParams(ctrl, graph, wspace);
      Moc_KWayFM(ctrl, graph, wspace, NGR_PASSES);
    }
  }
  else {
    Moc_GlobalMatch_Balance(ctrl, graph, wspace);

    Moc_Global_Partition(ctrl, graph->coarser, wspace);

    Moc_ProjectPartition(ctrl, graph, wspace);
    Moc_ComputePartitionParams(ctrl, graph, wspace);

    if (graph->ncon > 1 && graph->level < 3) {
      for (i=0; i<ncon; i++) {
        ftmp = ssum_strd(nparts, graph->gnpwgts+i, ncon);
        if (ftmp != 0.0)
          lbvec[i] = (float)(nparts) *
          graph->gnpwgts[samax_strd(nparts, graph->gnpwgts+i, ncon)*ncon+i]/ftmp;
        else
          lbvec[i] = 1.0;
      }
      lbavg = savg(graph->ncon, lbvec);

      if (lbavg > ubavg + 0.035) {
        if (ctrl->dbglvl&DBG_PROGRESS) {
          Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
          rprintf(ctrl, "nvtxs: %10d, cut: %8d, balance: ", graph->gnvtxs, graph->mincut);
          for (i=0; i<graph->ncon; i++) 
            rprintf(ctrl, "%.3f ", lbvec[i]);
          rprintf(ctrl, "\n");
	}

        Moc_KWayBalance(ctrl, graph, wspace, graph->ncon);
      }
    }

    Moc_KWayFM(ctrl, graph, wspace, NGR_PASSES);

    if (ctrl->dbglvl&DBG_PROGRESS) {
      Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
      rprintf(ctrl, "nvtxs: %10d, cut: %8d, balance: ", graph->gnvtxs, graph->mincut);
      for (i=0; i<graph->ncon; i++) 
        rprintf(ctrl, "%.3f ", lbvec[i]);
      rprintf(ctrl, "\n");
    }

    if (graph->level != 0)
      GKfree((void **)&graph->lnpwgts, (void **)&graph->gnpwgts, LTERM);
  }

  return;
}


