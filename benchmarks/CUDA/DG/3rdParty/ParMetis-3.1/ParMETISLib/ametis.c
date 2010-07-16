/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * ametis.c
 *
 * This is the entry point of parallel difussive repartitioning routines
 *
 * Started 10/19/96
 * George
 *
 * $Id: ametis.c,v 1.6 2003/07/25 04:01:03 karypis Exp $
 *
 */

#include <parmetislib.h>



/***********************************************************************************
* This function is the entry point of the parallel multilevel local diffusion
* algorithm. It uses parallel undirected diffusion followed by adaptive k-way 
* refinement. This function utilizes local coarsening.
************************************************************************************/
void ParMETIS_V3_AdaptiveRepart(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
  idxtype *vwgt, idxtype *vsize, idxtype *adjwgt, int *wgtflag, int *numflag,
  int *ncon, int *nparts, float *tpwgts, float *ubvec, float *ipc2redist,
  int *options, int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int h, i;
  int npes, mype;
  CtrlType ctrl;
  WorkSpaceType wspace;
  GraphType *graph;
  int tewgt, tvsize, nmoved, maxin, maxout, vtx_factor;
  float gtewgt, gtvsize, avg, maximb;
  int ps_relation, seed, dbglvl = 0;
  int iwgtflag, inumflag, incon, inparts, ioptions[10];
  float iipc2redist, *itpwgts, iubvec[MAXNCON];

  MPI_Comm_size(*comm, &npes);
  MPI_Comm_rank(*comm, &mype);

  /********************************/
  /* Try and take care bad inputs */
  /********************************/
  if (options != NULL && options[0] == 1)
    dbglvl = options[PMV3_OPTION_DBGLVL];
  CheckInputs(ADAPTIVE_PARTITION, npes, dbglvl, wgtflag, &iwgtflag, numflag, &inumflag,
              ncon, &incon, nparts, &inparts, tpwgts, &itpwgts, ubvec, iubvec, 
	      ipc2redist, &iipc2redist, options, ioptions, part, comm);

  /* ADD: take care of disconnected graph */
  /* ADD: take care of highly unbalanced vtxdist */
  /*********************************/
  /* Take care the nparts = 1 case */
  /*********************************/
  if (inparts == 1) {
    idxset(vtxdist[mype+1]-vtxdist[mype], 0, part); 
    *edgecut = 0;
    return;
  }

  /**************************/
  /* Set up data structures */
  /**************************/
  if (inumflag == 1) 
    ChangeNumbering(vtxdist, xadj, adjncy, part, npes, mype, 1);

  /*****************************/
  /* Set up control structures */
  /*****************************/
  if (ioptions[0] == 1) {
    dbglvl      = ioptions[PMV3_OPTION_DBGLVL];
    seed        = ioptions[PMV3_OPTION_SEED];
    ps_relation = (npes == inparts ? ioptions[PMV3_OPTION_PSR] : DISCOUPLED);
  }
  else {
    dbglvl      = GLOBAL_DBGLVL;
    seed        = GLOBAL_SEED;
    ps_relation = (npes == inparts ? COUPLED : DISCOUPLED);
  }

  SetUpCtrl(&ctrl, inparts, dbglvl, *comm);
  vtx_factor         = (amax(npes, inparts) > 256) ? 20 : 50;
  ctrl.CoarsenTo     = amin(vtxdist[npes]+1, vtx_factor*incon*amax(npes, inparts));
  ctrl.ipc_factor    = iipc2redist;
  ctrl.redist_factor = 1.0;
  ctrl.redist_base   = 1.0;
  ctrl.seed          = (seed == 0 ? mype : seed*mype);
  ctrl.sync          = GlobalSEMax(&ctrl, seed);
  ctrl.partType      = ADAPTIVE_PARTITION;
  ctrl.ps_relation   = ps_relation;
  ctrl.tpwgts        = itpwgts;

  graph = Moc_SetUpGraph(&ctrl, incon, vtxdist, xadj, vwgt, adjncy, adjwgt, &iwgtflag);
  graph->vsize = (vsize == NULL ? idxsmalloc(graph->nvtxs, 1, "vsize") : vsize);

  graph->home = idxmalloc(graph->nvtxs, "home");
  if (ctrl.ps_relation == COUPLED)
    idxset(graph->nvtxs, mype, graph->home);
  else {
    /* Downgrade the partition numbers if part[] has more partitions that nparts */
    for (i=0; i<graph->nvtxs; i++)
      part[i] = (part[i] >= ctrl.nparts ? 0 : part[i]);

    idxcopy(graph->nvtxs, part, graph->home);
  }

  tewgt   = idxsum(graph->nedges, graph->adjwgt);
  tvsize  = idxsum(graph->nvtxs, graph->vsize);
  gtewgt  = (float) GlobalSESum(&ctrl, tewgt) + 1.0/graph->gnvtxs;  /* The +1/graph->gnvtxs were added to remove any FPE */
  gtvsize = (float) GlobalSESum(&ctrl, tvsize) + 1.0/graph->gnvtxs;
  ctrl.edge_size_ratio = gtewgt/gtvsize;
  scopy(incon, iubvec, ctrl.ubvec);

  PreAllocateMemory(&ctrl, graph, &wspace);

  /***********************/
  /* Partition and Remap */
  /***********************/
  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  Adaptive_Partition(&ctrl, graph, &wspace);
  ParallelReMapGraph(&ctrl, graph, &wspace);

  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));

  idxcopy(graph->nvtxs, graph->where, part);
  if (edgecut != NULL)
    *edgecut = graph->mincut;

  /***********************/
  /* Take care of output */
  /***********************/
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimingInfo(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));

  if (ctrl.dbglvl&DBG_INFO) {
    Mc_ComputeMoveStatistics(&ctrl, graph, &nmoved, &maxin, &maxout);
    rprintf(&ctrl, "Final %3d-way Cut: %6d \tBalance: ", inparts, graph->mincut);
    avg = 0.0;
    for (h=0; h<incon; h++) {
      maximb = 0.0;
      for (i=0; i<inparts; i++)
        maximb = amax(maximb, graph->gnpwgts[i*incon+h]/itpwgts[i*incon+h]);
      avg += maximb;
      rprintf(&ctrl, "%.3f ", maximb);
    }
    rprintf(&ctrl, "\nNMoved: %d %d %d %d\n", nmoved, maxin, maxout, maxin+maxout);
  }

  /*************************************/
  /* Free memory, renumber, and return */
  /*************************************/
  GKfree((void **)&graph->lnpwgts, (void **)&graph->gnpwgts, (void **)&graph->nvwgt, (void **)(&graph->home), LTERM);
  if (vsize == NULL)
    GKfree((void **)(&graph->vsize), LTERM);
  GKfree((void **)&itpwgts, LTERM);
  FreeInitialGraphAndRemap(graph, iwgtflag);
  FreeWSpace(&wspace);
  FreeCtrl(&ctrl);

  if (inumflag == 1)
    ChangeNumbering(vtxdist, xadj, adjncy, part, npes, mype, 0);

  return;
}




/*************************************************************************
* This function is the driver for the adaptive refinement mode of ParMETIS
**************************************************************************/
void Adaptive_Partition(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i;
  int tewgt, tvsize;
  float gtewgt, gtvsize;
  float ubavg, lbavg, lbvec[MAXNCON];

  /************************************/
  /* Set up important data structures */
  /************************************/
  SetUp(ctrl, graph, wspace);

  ubavg   = savg(graph->ncon, ctrl->ubvec);
  tewgt   = idxsum(graph->nedges, graph->adjwgt);
  tvsize  = idxsum(graph->nvtxs, graph->vsize);
  gtewgt  = (float) GlobalSESum(ctrl, tewgt) + 1.0/graph->gnvtxs;  /* The +1/graph->gnvtxs were added to remove any FPE */
  gtvsize = (float) GlobalSESum(ctrl, tvsize) + 1.0/graph->gnvtxs;
  ctrl->redist_factor = ctrl->redist_base * ((gtewgt/gtvsize)/ ctrl->edge_size_ratio);

  IFSET(ctrl->dbglvl, DBG_PROGRESS, rprintf(ctrl, "[%6d %8d %5d %5d][%d]\n", 
        graph->gnvtxs, GlobalSESum(ctrl, graph->nedges), GlobalSEMin(ctrl, graph->nvtxs), GlobalSEMax(ctrl, graph->nvtxs), ctrl->CoarsenTo));

  if (graph->gnvtxs < 1.3*ctrl->CoarsenTo ||
     (graph->finer != NULL && graph->gnvtxs > graph->finer->gnvtxs*COARSEN_FRACTION)) {

    /***********************************************/
    /* Balance the partition on the coarsest graph */
    /***********************************************/
    graph->where = idxsmalloc(graph->nvtxs+graph->nrecv, -1, "graph->where");
    idxcopy(graph->nvtxs, graph->home, graph->where);

    Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
    lbavg = savg(graph->ncon, lbvec);

    if (lbavg > ubavg + 0.035 && ctrl->partType != REFINE_PARTITION)
      Balance_Partition(ctrl, graph, wspace);

    if (ctrl->dbglvl&DBG_PROGRESS) {
      Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
      rprintf(ctrl, "nvtxs: %10d, balance: ", graph->gnvtxs);
      for (i=0; i<graph->ncon; i++) 
        rprintf(ctrl, "%.3f ", lbvec[i]);
      rprintf(ctrl, "\n");
    }

    /* check if no coarsening took place */
    if (graph->finer == NULL) {
      Moc_ComputePartitionParams(ctrl, graph, wspace);
      Moc_KWayBalance(ctrl, graph, wspace, graph->ncon);
      Moc_KWayAdaptiveRefine(ctrl, graph, wspace, NGR_PASSES);
    }
  }
  else {
    /*******************************/
    /* Coarsen it and partition it */
    /*******************************/
    switch (ctrl->ps_relation) {
      case COUPLED:
        Mc_LocalMatch_HEM(ctrl, graph, wspace);
        break;
      case DISCOUPLED:
      default:
        Moc_GlobalMatch_Balance(ctrl, graph, wspace);
        break;
    }

    Adaptive_Partition(ctrl, graph->coarser, wspace);

    /********************************/
    /* project partition and refine */
    /********************************/
    Moc_ProjectPartition(ctrl, graph, wspace);
    Moc_ComputePartitionParams(ctrl, graph, wspace);

    if (graph->ncon > 1 && graph->level < 4) {
      Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
      lbavg = savg(graph->ncon, lbvec);

      if (lbavg > ubavg + 0.025) {
        Moc_KWayBalance(ctrl, graph, wspace, graph->ncon);
      }
    }

    Moc_KWayAdaptiveRefine(ctrl, graph, wspace, NGR_PASSES);

    if (ctrl->dbglvl&DBG_PROGRESS) {
      Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
      rprintf(ctrl, "nvtxs: %10d, cut: %8d, balance: ", graph->gnvtxs, graph->mincut);
      for (i=0; i<graph->ncon; i++) 
        rprintf(ctrl, "%.3f ", lbvec[i]);
      rprintf(ctrl, "\n");
    }
  }
}

