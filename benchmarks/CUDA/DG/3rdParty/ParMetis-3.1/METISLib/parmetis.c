/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * parmetis.c
 *
 * This file contains top level routines that are used by ParMETIS
 *
 * Started 10/14/97
 * George
 *
 * $Id: parmetis.c,v 1.2 2003/07/24 18:39:11 karypis Exp $
 *
 */

#include <metis.h>


/*************************************************************************
* This function is the entry point for KMETIS with seed specification
* in options[7] 
**************************************************************************/
void METIS_PartGraphKway2(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
                         idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, 
                         int *options, int *edgecut, idxtype *part)
{
  int i;
  float *tpwgts;

  tpwgts = fmalloc(*nparts, "KMETIS: tpwgts");
  for (i=0; i<*nparts; i++) 
    tpwgts[i] = 1.0/(1.0*(*nparts));

  METIS_WPartGraphKway2(nvtxs, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, 
                       tpwgts, options, edgecut, part);

  free(tpwgts);
}


/*************************************************************************
* This function is the entry point for KWMETIS with seed specification
* in options[7] 
**************************************************************************/
void METIS_WPartGraphKway2(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
                          idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, 
                          float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  SetUpGraph(&graph, OP_KMETIS, *nvtxs, 1, xadj, adjncy, vwgt, adjwgt, *wgtflag);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = KMETIS_CTYPE;
    ctrl.IType = KMETIS_ITYPE;
    ctrl.RType = KMETIS_RTYPE;
    ctrl.dbglvl = KMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_KMETIS;
  ctrl.CoarsenTo = 20*(*nparts);
  ctrl.maxvwgt = 1.5*((graph.vwgt ? idxsum(*nvtxs, graph.vwgt) : (*nvtxs))/ctrl.CoarsenTo);

  InitRandom(options[7]);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  *edgecut = MlevelKWayPartitioning(&ctrl, &graph, *nparts, part, tpwgts, 1.000);

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);

  if (*numflag == 1)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);
}


/*************************************************************************
* This function is the entry point for the node ND code for ParMETIS
**************************************************************************/
void METIS_NodeNDP(int nvtxs, idxtype *xadj, idxtype *adjncy, int npes, 
                   int *options, idxtype *perm, idxtype *iperm, idxtype *sizes) 
{
  int i, ii, j, l, wflag, nflag;
  GraphType graph;
  CtrlType ctrl;
  idxtype *cptr, *cind;

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType   = ONMETIS_CTYPE;
    ctrl.IType   = ONMETIS_ITYPE;
    ctrl.RType   = ONMETIS_RTYPE;
    ctrl.dbglvl  = ONMETIS_DBGLVL;
    ctrl.oflags  = ONMETIS_OFLAGS;
    ctrl.pfactor = ONMETIS_PFACTOR;
    ctrl.nseps   = ONMETIS_NSEPS;
  }
  else {
    ctrl.CType   = options[OPTION_CTYPE];
    ctrl.IType   = options[OPTION_ITYPE];
    ctrl.RType   = options[OPTION_RTYPE];
    ctrl.dbglvl  = options[OPTION_DBGLVL];
    ctrl.oflags  = options[OPTION_OFLAGS];
    ctrl.pfactor = options[OPTION_PFACTOR];
    ctrl.nseps   = options[OPTION_NSEPS];
  }
  if (ctrl.nseps < 1)
    ctrl.nseps = 1;

  ctrl.optype = OP_ONMETIS;
  ctrl.CoarsenTo = 100;

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  InitRandom(-1);

  if (ctrl.oflags&OFLAG_COMPRESS) {
    /*============================================================
    * Compress the graph 
    ==============================================================*/
    cptr = idxmalloc(nvtxs+1, "ONMETIS: cptr");
    cind = idxmalloc(nvtxs, "ONMETIS: cind");

    CompressGraph(&ctrl, &graph, nvtxs, xadj, adjncy, cptr, cind);

    if (graph.nvtxs >= COMPRESSION_FRACTION*(nvtxs)) {
      ctrl.oflags--; /* We actually performed no compression */
      GKfree(&cptr, &cind, LTERM);
    }
    else if (2*graph.nvtxs < nvtxs && ctrl.nseps == 1)
      ctrl.nseps = 2;
  }
  else {
    SetUpGraph(&graph, OP_ONMETIS, nvtxs, 1, xadj, adjncy, NULL, NULL, 0);
  }


  /*=============================================================
  * Do the nested dissection ordering 
  --=============================================================*/
  ctrl.maxvwgt = 1.5*(idxsum(graph.nvtxs, graph.vwgt)/ctrl.CoarsenTo);
  AllocateWorkSpace(&ctrl, &graph, 2);

  idxset(2*npes-1, 0, sizes);
  MlevelNestedDissectionP(&ctrl, &graph, iperm, graph.nvtxs, npes, 0, sizes);

  FreeWorkSpace(&ctrl, &graph);

  if (ctrl.oflags&OFLAG_COMPRESS) { /* Uncompress the ordering */
    if (graph.nvtxs < COMPRESSION_FRACTION*(nvtxs)) { 
      /* construct perm from iperm */
      for (i=0; i<graph.nvtxs; i++)
        perm[iperm[i]] = i; 
      for (l=ii=0; ii<graph.nvtxs; ii++) {
        i = perm[ii];
        for (j=cptr[i]; j<cptr[i+1]; j++)
          iperm[cind[j]] = l++;
      }
    }

    GKfree(&cptr, &cind, LTERM);
  }


  for (i=0; i<nvtxs; i++)
    perm[iperm[i]] = i;

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

}



/*************************************************************************
* This function takes a graph and produces a bisection of it
**************************************************************************/
void MlevelNestedDissectionP(CtrlType *ctrl, GraphType *graph, idxtype *order, int lastvtx, 
                             int npes, int cpos, idxtype *sizes)
{
  int i, j, nvtxs, nbnd, tvwgt, tpwgts2[2];
  idxtype *label, *bndind;
  GraphType lgraph, rgraph;
  float ubfactor;

  nvtxs = graph->nvtxs;

  if (nvtxs == 0) {
    GKfree(&graph->gdata, &graph->rdata, &graph->label, LTERM);
    return;
  }

  /* Determine the weights of the partitions */
  tvwgt = idxsum(nvtxs, graph->vwgt);
  tpwgts2[0] = tvwgt/2;
  tpwgts2[1] = tvwgt-tpwgts2[0];

  if (cpos >= npes-1) 
    ubfactor = ORDER_UNBALANCE_FRACTION;
  else 
    ubfactor = 1.05;


  MlevelNodeBisectionMultiple(ctrl, graph, tpwgts2, ubfactor);

  IFSET(ctrl->dbglvl, DBG_SEPINFO, printf("Nvtxs: %6d, [%6d %6d %6d]\n", graph->nvtxs, graph->pwgts[0], graph->pwgts[1], graph->pwgts[2]));

  if (cpos < npes-1) {
    sizes[2*npes-2-cpos] = graph->pwgts[2];
    sizes[2*npes-2-(2*cpos+1)] = graph->pwgts[1];
    sizes[2*npes-2-(2*cpos+2)] = graph->pwgts[0];
  }

  /* Order the nodes in the separator */
  nbnd = graph->nbnd;
  bndind = graph->bndind;
  label = graph->label;
  for (i=0; i<nbnd; i++) 
    order[label[bndind[i]]] = --lastvtx;

  SplitGraphOrder(ctrl, graph, &lgraph, &rgraph);

  /* Free the memory of the top level graph */
  GKfree(&graph->gdata, &graph->rdata, &graph->label, LTERM);

  if (rgraph.nvtxs > MMDSWITCH || 2*cpos+1 < npes-1) 
    MlevelNestedDissectionP(ctrl, &rgraph, order, lastvtx, npes, 2*cpos+1, sizes);
  else {
    MMDOrder(ctrl, &rgraph, order, lastvtx); 
    GKfree(&rgraph.gdata, &rgraph.rdata, &rgraph.label, LTERM);
  }
  if (lgraph.nvtxs > MMDSWITCH || 2*cpos+2 < npes-1) 
    MlevelNestedDissectionP(ctrl, &lgraph, order, lastvtx-rgraph.nvtxs, npes, 2*cpos+2, sizes);
  else {
    MMDOrder(ctrl, &lgraph, order, lastvtx-rgraph.nvtxs); 
    GKfree(&lgraph.gdata, &lgraph.rdata, &lgraph.label, LTERM);
  }
}




/*************************************************************************
* This function is the entry point for ONWMETIS. It requires weights on the
* vertices. It is for the case that the matrix has been pre-compressed.
**************************************************************************/
void METIS_NodeComputeSeparator(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
           idxtype *adjwgt, int *options, int *sepsize, idxtype *part) 
{
  int i, j, tvwgt, tpwgts[2];
  GraphType graph;
  CtrlType ctrl;

  SetUpGraph(&graph, OP_ONMETIS, *nvtxs, 1, xadj, adjncy, vwgt, adjwgt, 3);
  tvwgt = idxsum(*nvtxs, graph.vwgt);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = ONMETIS_CTYPE;
    ctrl.IType = ONMETIS_ITYPE;
    ctrl.RType = ONMETIS_RTYPE;
    ctrl.dbglvl = ONMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }

  ctrl.oflags  = 0;
  ctrl.pfactor = 0;
  ctrl.nseps = 1;
  ctrl.optype = OP_ONMETIS;
  ctrl.CoarsenTo = amin(100, *nvtxs-1);
  ctrl.maxvwgt = 1.5*tvwgt/ctrl.CoarsenTo;

  InitRandom(options[7]);

  AllocateWorkSpace(&ctrl, &graph, 2);

  /*============================================================
   * Perform the bisection
   *============================================================*/ 
  tpwgts[0] = tvwgt/2;
  tpwgts[1] = tvwgt-tpwgts[0];

  MlevelNodeBisectionMultiple(&ctrl, &graph, tpwgts, 1.05);

  *sepsize = graph.pwgts[2];
  idxcopy(*nvtxs, graph.where, part);

  GKfree(&graph.gdata, &graph.rdata, &graph.label, LTERM);


  FreeWorkSpace(&ctrl, &graph);

}



/*************************************************************************
* This function is the entry point for ONWMETIS. It requires weights on the
* vertices. It is for the case that the matrix has been pre-compressed.
**************************************************************************/
void METIS_EdgeComputeSeparator(int *nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
           idxtype *adjwgt, int *options, int *sepsize, idxtype *part) 
{
  int i, j, tvwgt, tpwgts[2];
  GraphType graph;
  CtrlType ctrl;

  SetUpGraph(&graph, OP_ONMETIS, *nvtxs, 1, xadj, adjncy, vwgt, adjwgt, 3);
  tvwgt = idxsum(*nvtxs, graph.vwgt);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType = ONMETIS_CTYPE;
    ctrl.IType = ONMETIS_ITYPE;
    ctrl.RType = ONMETIS_RTYPE;
    ctrl.dbglvl = ONMETIS_DBGLVL;
  }
  else {
    ctrl.CType = options[OPTION_CTYPE];
    ctrl.IType = options[OPTION_ITYPE];
    ctrl.RType = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }

  ctrl.oflags  = 0;
  ctrl.pfactor = 0;
  ctrl.nseps = 1;
  ctrl.optype = OP_OEMETIS;
  ctrl.CoarsenTo = amin(100, *nvtxs-1);
  ctrl.maxvwgt = 1.5*tvwgt/ctrl.CoarsenTo;

  InitRandom(options[7]);

  AllocateWorkSpace(&ctrl, &graph, 2);

  /*============================================================
   * Perform the bisection
   *============================================================*/ 
  tpwgts[0] = tvwgt/2;
  tpwgts[1] = tvwgt-tpwgts[0];

  MlevelEdgeBisection(&ctrl, &graph, tpwgts, 1.05);
  ConstructMinCoverSeparator(&ctrl, &graph, 1.05);

  *sepsize = graph.pwgts[2];
  idxcopy(*nvtxs, graph.where, part);

  GKfree(&graph.gdata, &graph.rdata, &graph.label, LTERM);


  FreeWorkSpace(&ctrl, &graph);

}


/*************************************************************************
* This function is the entry point for PWMETIS that accepts exact weights
* for the target partitions
**************************************************************************/
void METIS_mCPartGraphRecursive2(int *nvtxs, int *ncon, idxtype *xadj, idxtype *adjncy, 
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, 
       float *tpwgts, int *options, int *edgecut, idxtype *part)
{
  int i, j;
  GraphType graph;
  CtrlType ctrl;
  float *mytpwgts;
  float avgwgt;

  if (*numflag == 1)
    Change2CNumbering(*nvtxs, xadj, adjncy);

  SetUpGraph(&graph, OP_PMETIS, *nvtxs, *ncon, xadj, adjncy, vwgt, adjwgt, *wgtflag);
  graph.npwgts = NULL;
  mytpwgts = fmalloc(*nparts, "mytpwgts");
  scopy(*nparts, tpwgts, mytpwgts);

  if (options[0] == 0) {  /* Use the default parameters */
    ctrl.CType  = McPMETIS_CTYPE;
    ctrl.IType  = McPMETIS_ITYPE;
    ctrl.RType  = McPMETIS_RTYPE;
    ctrl.dbglvl = McPMETIS_DBGLVL;
  }
  else {
    ctrl.CType  = options[OPTION_CTYPE];
    ctrl.IType  = options[OPTION_ITYPE];
    ctrl.RType  = options[OPTION_RTYPE];
    ctrl.dbglvl = options[OPTION_DBGLVL];
  }
  ctrl.optype = OP_PMETIS;
  ctrl.CoarsenTo = 100;

  ctrl.nmaxvwgt = 1.5/(1.0*ctrl.CoarsenTo);

  InitRandom(options[7]);

  AllocateWorkSpace(&ctrl, &graph, *nparts);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  ASSERT(CheckGraph(&graph));
  *edgecut = MCMlevelRecursiveBisection2(&ctrl, &graph, *nparts, mytpwgts, part, 1.000, 0);

/* 
{
idxtype wgt[2048], minwgt, maxwgt, sumwgt;

printf("nvtxs: %d, nparts: %d, ncon: %d\n", graph.nvtxs, *nparts, *ncon);
for (i=0; i<(*nparts)*(*ncon); i++)
  wgt[i] = 0;
for (i=0; i<graph.nvtxs; i++)
  for (j=0; j<*ncon; j++)
    wgt[part[i]*(*ncon)+j] += vwgt[i*(*ncon)+j];

for (j=0; j<*ncon; j++) {
 minwgt = maxwgt = sumwgt = 0;
 for (i=0; i<(*nparts); i++) {
   minwgt = (wgt[i*(*ncon)+j] < wgt[minwgt*(*ncon)+j]) ? i : minwgt;
   maxwgt = (wgt[i*(*ncon)+j] > wgt[maxwgt*(*ncon)+j]) ? i : maxwgt;
   sumwgt += wgt[i*(*ncon)+j];
 }
 avgwgt = (float)sumwgt / (float)*nparts;
 printf("min: %5d, max: %5d, avg: %5.2f, balance: %6.3f\n", wgt[minwgt*(*ncon)+j], wgt[maxwgt*(*ncon)+j], avgwgt, (float)wgt[maxwgt*(*ncon)+j] / avgwgt);
}
printf("\n");
}
*/

  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimers(&ctrl));

  FreeWorkSpace(&ctrl, &graph);
  GKfree((void *)&mytpwgts, LTERM);

  if (*numflag == 1)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);
}



/*************************************************************************
* This function takes a graph and produces a bisection of it
**************************************************************************/
int MCMlevelRecursiveBisection2(CtrlType *ctrl, GraphType *graph, int nparts,
    float *tpwgts, idxtype *part, float ubfactor, int fpart)
{
  int i, nvtxs, cut;
  float wsum, tpwgts2[2];
  idxtype *label, *where;
  GraphType lgraph, rgraph;

  nvtxs = graph->nvtxs;
  if (nvtxs == 0) 
    return 0;

  /* Determine the weights of the partitions */
  tpwgts2[0] = ssum(nparts/2, tpwgts);
  tpwgts2[1] = 1.0-tpwgts2[0];

  MCMlevelEdgeBisection(ctrl, graph, tpwgts2, ubfactor);
  cut = graph->mincut;

  label = graph->label;
  where = graph->where;
  for (i=0; i<nvtxs; i++)
    part[label[i]] = where[i] + fpart;

  if (nparts > 2) 
    SplitGraphPart(ctrl, graph, &lgraph, &rgraph);

  /* Free the memory of the top level graph */
  GKfree(&graph->gdata, &graph->nvwgt, &graph->rdata, &graph->label, &graph->npwgts, LTERM);

  /* Scale the fractions in the tpwgts according to the true weight */
  wsum = ssum(nparts/2, tpwgts);
  sscale(nparts/2, 1.0/wsum, tpwgts);
  sscale(nparts-nparts/2, 1.0/(1.0-wsum), tpwgts+nparts/2);

  /* Do the recursive call */
  if (nparts > 3) {
    cut += MCMlevelRecursiveBisection2(ctrl, &lgraph, nparts/2, tpwgts, part, ubfactor, fpart);
    cut += MCMlevelRecursiveBisection2(ctrl, &rgraph, nparts-nparts/2, tpwgts+nparts/2, part, ubfactor, fpart+nparts/2);
  }
  else if (nparts == 3) {
    cut += MCMlevelRecursiveBisection2(ctrl, &rgraph, nparts-nparts/2, tpwgts+nparts/2, part, ubfactor, fpart+nparts/2);
    GKfree(&lgraph.gdata, &lgraph.nvwgt, &lgraph.label, LTERM);
  }

  return cut;

}


