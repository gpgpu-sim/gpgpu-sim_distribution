/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rmetis.c
 *
 * This is the entry point of the partitioning refinement routine
 *
 * Started 10/19/96
 * George
 *
 * $Id: rmetis.c,v 1.5 2003/07/25 04:01:05 karypis Exp $
 *
 */

#include <parmetislib.h>



/***********************************************************************************
* This function is the entry point of the parallel multilevel local diffusion
* algorithm. It uses parallel undirected diffusion followed by adaptive k-way 
* refinement. This function utilizes local coarsening.
************************************************************************************/
void ParMETIS_V3_RefineKway(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
              idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *ncon, 
	      int *nparts, float *tpwgts, float *ubvec, int *options, int *edgecut, 
	      idxtype *part, MPI_Comm *comm)
{
  int h, i;
  int npes, mype;
  CtrlType ctrl;
  WorkSpaceType wspace;
  GraphType *graph;
  int tewgt, tvsize, nmoved, maxin, maxout;
  float gtewgt, gtvsize, avg, maximb;
  int ps_relation, seed, dbglvl = 0;
  int iwgtflag, inumflag, incon, inparts, ioptions[10];
  float *itpwgts, iubvec[MAXNCON];

  MPI_Comm_size(*comm, &npes);
  MPI_Comm_rank(*comm, &mype);

  /********************************/
  /* Try and take care bad inputs */
  /********************************/
  if (options != NULL && options[0] == 1)
    dbglvl = options[PMV3_OPTION_DBGLVL];
  CheckInputs(REFINE_PARTITION, npes, dbglvl, wgtflag, &iwgtflag, numflag, &inumflag,
              ncon, &incon, nparts, &inparts, tpwgts, &itpwgts, ubvec, iubvec, 
              NULL, NULL, options, ioptions, part, comm);

  /* ADD: take care of disconnected graph */
  /* ADD: take care of highly unbalanced vtxdist */
  /*********************************/
  /* Take care the nparts = 1 case */
  /*********************************/
  if (inparts <= 1) {
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
    dbglvl = ioptions[PMV3_OPTION_DBGLVL];
    seed = ioptions[PMV3_OPTION_SEED];
    ps_relation = (npes == inparts) ? ioptions[PMV3_OPTION_PSR] : DISCOUPLED;
  }
  else {
    dbglvl = GLOBAL_DBGLVL;
    seed = GLOBAL_SEED;
    ps_relation = (npes == inparts) ? COUPLED : DISCOUPLED;
  }

  SetUpCtrl(&ctrl, inparts, dbglvl, *comm);
  ctrl.CoarsenTo = amin(vtxdist[npes]+1, 50*incon*amax(npes, inparts));
  ctrl.ipc_factor = 1000.0;
  ctrl.redist_factor = 1.0;
  ctrl.redist_base = 1.0;
  ctrl.seed = (seed == 0) ? mype : seed*mype;
  ctrl.sync = GlobalSEMax(&ctrl, seed);
  ctrl.partType = REFINE_PARTITION;
  ctrl.ps_relation = ps_relation;
  ctrl.tpwgts = itpwgts;

  graph = Moc_SetUpGraph(&ctrl, incon, vtxdist, xadj, vwgt, adjncy, adjwgt, &iwgtflag);
  graph->vsize = idxsmalloc(graph->nvtxs, 1, "vsize");

  graph->home = idxmalloc(graph->nvtxs, "home");
  if (ctrl.ps_relation == COUPLED)
    idxset(graph->nvtxs, mype, graph->home);
  else
    idxcopy(graph->nvtxs, part, graph->home);

  tewgt   = idxsum(graph->nedges, graph->adjwgt);
  tvsize  = idxsum(graph->nvtxs, graph->vsize);
  gtewgt  = (float) GlobalSESum(&ctrl, tewgt) + 1.0/graph->gnvtxs;
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
  GKfree((void **)&graph->lnpwgts, (void **)&graph->gnpwgts, (void **)&graph->nvwgt, (void **)(&graph->home), (void **)(&graph->vsize), LTERM);

  GKfree((void **)&itpwgts, LTERM);
  FreeInitialGraphAndRemap(graph, iwgtflag);
  FreeWSpace(&wspace);
  FreeCtrl(&ctrl);

  if (inumflag == 1)
    ChangeNumbering(vtxdist, xadj, adjncy, part, npes, mype, 0);

  return;
}


