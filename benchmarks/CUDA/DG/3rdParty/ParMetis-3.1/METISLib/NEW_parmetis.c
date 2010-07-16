/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * parmetis.c
 *
 * This file contains the top level routines for the multilevel recursive
 * bisection algorithm PMETIS.
 *
 * Started 7/24/97
 * George
 *
 * $Id: NEW_parmetis.c,v 1.1 2003/07/16 15:55:14 karypis Exp $
 *
 */

#include <metis.h>


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
idxtype wgt[2048], minwgt, maxwgt, sumwgt;
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
  if (nvtxs == 0) {
/*    printf("\t***Cannot bisect a graph with 0 vertices!\n\t***You are trying to partition a graph into too many parts!\n"); */
    return 0;
  }

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


