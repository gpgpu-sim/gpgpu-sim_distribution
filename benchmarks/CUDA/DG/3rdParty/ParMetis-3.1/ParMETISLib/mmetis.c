/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mmetis.c
 *
 * This is the entry point of ParMETIS_V3_PartMeshKway
 *
 * Started 10/19/96
 * George
 *
 * $Id: mmetis.c,v 1.8 2003/07/25 04:01:04 karypis Exp $
 *
 */

#include <parmetislib.h>


/***********************************************************************************
* This function is the entry point of the parallel k-way multilevel mesh partitionioner. 
* This function assumes nothing about the mesh distribution.
* It is the general case.
************************************************************************************/
void ParMETIS_V3_PartMeshKway(idxtype *elmdist, idxtype *eptr, idxtype *eind, idxtype *elmwgt, 
                 int *wgtflag, int *numflag, int *ncon, int *ncommonnodes, int *nparts, 
		 float *tpwgts, float *ubvec, int *options, int *edgecut, idxtype *part, 
		 MPI_Comm *comm)
{
  int i, nvtxs, nedges, gnedges, npes, mype;
  idxtype *xadj, *adjncy;
  timer TotalTmr, Mesh2DualTmr, ParMETISTmr;
  CtrlType ctrl;

  /********************************/
  /* Try and take care bad inputs */
  /********************************/
  if (elmdist == NULL || eptr == NULL || eind == NULL || wgtflag == NULL || 
      numflag == NULL || ncon == NULL || ncommonnodes == NULL || nparts == NULL ||
      tpwgts == NULL || ubvec == NULL || options == NULL || edgecut == NULL || 
      part == NULL || comm == NULL) {
    printf("ERROR: One or more required parameters is NULL. Aborting.\n");
    abort();
  }
  if (((*wgtflag)&2) && elmwgt == NULL) {
    printf("ERROR: elmwgt == NULL when vertex weights were specified. Aborting.\n");
    abort();
  }

  
  SetUpCtrl(&ctrl, *nparts, (options[0] == 1 ? options[PMV3_OPTION_DBGLVL] : 0), *comm);
  npes = ctrl.npes;
  mype = ctrl.mype;

  cleartimer(TotalTmr);
  cleartimer(Mesh2DualTmr);
  cleartimer(ParMETISTmr);

  MPI_Barrier(ctrl.comm);
  starttimer(TotalTmr);
  starttimer(Mesh2DualTmr);

  ParMETIS_V3_Mesh2Dual(elmdist, eptr, eind, numflag, ncommonnodes, &xadj, &adjncy, &(ctrl.comm));

  if (ctrl.dbglvl&DBG_INFO) {
    nvtxs = elmdist[mype+1]-elmdist[mype];
    nedges = xadj[nvtxs] + (*numflag == 0 ? 0 : -1);
    rprintf(&ctrl, "Completed Dual Graph -- Nvtxs: %d, Nedges: %d \n", 
            elmdist[npes], GlobalSESum(&ctrl, nedges));
  }

  MPI_Barrier(ctrl.comm);
  stoptimer(Mesh2DualTmr);


  /***********************/
  /* Partition the graph */
  /***********************/
  starttimer(ParMETISTmr);

  ParMETIS_V3_PartKway(elmdist, xadj, adjncy, elmwgt, NULL, wgtflag, numflag, ncon, 
                       nparts, tpwgts, ubvec, options, edgecut, part, &(ctrl.comm));

  MPI_Barrier(ctrl.comm);
  stoptimer(ParMETISTmr);
  stoptimer(TotalTmr);

  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimer(&ctrl, Mesh2DualTmr,	"   Mesh2Dual"));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimer(&ctrl, ParMETISTmr,	"    ParMETIS"));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimer(&ctrl, TotalTmr,	"       Total"));

  GKfree((void **)&xadj, (void **)&adjncy, LTERM);

  FreeCtrl(&ctrl);

  return;
}
