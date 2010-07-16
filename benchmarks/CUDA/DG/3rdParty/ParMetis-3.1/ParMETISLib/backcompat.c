/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * backcompat.c
 *
 * This file ensures backwards compatability with previous ParMETIS releases
 *
 * Started 10/19/96
 * George
 *
 * $Id: backcompat.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 *
 */

#include <parmetislib.h>

/*****************************************************************************
*  This function computes a partitioning.
*****************************************************************************/
void ParMETIS_PartKway(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
       idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut,
       idxtype *part, MPI_Comm *comm)
{
  int i;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  int myoptions[10];

  tpwgts = fmalloc(*nparts*ncon, "tpwgts");
  for (i=0; i<*nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(*nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
  }

  ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag,
  &ncon, nparts, tpwgts, ubvec, myoptions, edgecut, part, comm);

  free(tpwgts);
}



/***********************************************************************************
 * * This function is the entry point of the parallel k-way multilevel partitionioner.
 * * This function assumes nothing about the graph distribution.
 * * It is the general case.
 * ************************************************************************************/
void PARKMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
		               idxtype *part, int *options, MPI_Comm comm)
{
  int wgtflag, numflag, edgecut, newoptions[5];
  int npes;

  MPI_Comm_size(comm, &npes);

  newoptions[0] = 1;
  newoptions[OPTION_IPART] = options[2];
  newoptions[OPTION_FOLDF] = options[1];
  newoptions[OPTION_DBGLVL] = options[4];

		        numflag = options[3];
  wgtflag = (vwgt == NULL ? 0 : 2) + (adjwgt == NULL ? 0 : 1);

  ParMETIS_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &npes,
	newoptions, &edgecut, part, &comm);

  options[0] = edgecut;

}



/*****************************************************************************
*  This function computes a partitioning using coordinate data.
*****************************************************************************/
void ParMETIS_PartGeomKway(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
       idxtype *adjwgt, int *wgtflag, int *numflag, int *ndims, float *xyz, int *nparts,
       int *options, int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  int myoptions[10];

  tpwgts = fmalloc(*nparts*ncon, "tpwgts");
  for (i=0; i<*nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(*nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
  }

  ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ndims, xyz,
  &ncon, nparts, tpwgts, ubvec, myoptions, edgecut, part, comm);

  free(tpwgts);
  return;
}


/***********************************************************************************
* This function is the entry point of the parallel ordering algorithm.
* This function assumes that the graph is already nice partitioned among the
* processors and then proceeds to perform recursive bisection.
************************************************************************************/
void ParMETIS_PartGeom(idxtype *vtxdist, int *ndims, float *xyz, idxtype *part, MPI_Comm *comm)
{
  ParMETIS_V3_PartGeom(vtxdist, ndims, xyz, part, comm);
}


/*****************************************************************************
*  This function computes a partitioning using coordinate data.
*****************************************************************************/
void ParMETIS_PartGeomRefine(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
  idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *ndims,
  float *xyz, int *options, int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int npes, nparts, ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  int myoptions[10];

  MPI_Comm_size(*comm, &npes);
  nparts = npes;

  tpwgts = fmalloc(nparts*ncon, "tpwgts");
  for (i=0; i<nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
  }

  ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ndims, xyz,
  &ncon, &nparts, tpwgts, ubvec, myoptions, edgecut, part, comm);

  free(tpwgts);
  return;
}


/***********************************************************************************
* This function is the entry point of the parallel kmetis algorithm that uses
* coordinates to compute an initial graph distribution.
************************************************************************************/
void PARGKMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
                int ndims, float *xyz, idxtype *part, int *options, MPI_Comm comm)
{
  int npes, wgtflag, numflag, edgecut, newoptions[5];

  MPI_Comm_size(comm, &npes);

  newoptions[0] = 1;
  newoptions[OPTION_IPART] = options[2];
  newoptions[OPTION_FOLDF] = options[1];
  newoptions[OPTION_DBGLVL] = options[4];

  numflag = options[3];
  wgtflag = (vwgt == NULL ? 0 : 2) + (adjwgt == NULL ? 0 : 1);

  ParMETIS_PartGeomKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag,
     &ndims, xyz, &npes, newoptions, &edgecut, part, &comm);

  options[0] = edgecut;

}


/***********************************************************************************
* This function is the entry point of the parallel rmetis algorithm that uses
* coordinates to compute an initial graph distribution.
************************************************************************************/
void PARGRMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
                int ndims, float *xyz, idxtype *part, int *options, MPI_Comm comm)
{
  int wgtflag, numflag, edgecut, newoptions[5];

  newoptions[0] = 1;
  newoptions[OPTION_IPART] = options[2];
  newoptions[OPTION_FOLDF] = options[1];
  newoptions[OPTION_DBGLVL] = options[4];

  numflag = options[3];
  wgtflag = (vwgt == NULL ? 0 : 2) + (adjwgt == NULL ? 0 : 1);

  ParMETIS_PartGeomRefine(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag,
     &ndims, xyz, newoptions, &edgecut, part, &comm);

  options[0] = edgecut;

}

/***********************************************************************************
* This function is the entry point of the parallel ordering algorithm.
* This function assumes that the graph is already nice partitioned among the
* processors and then proceeds to perform recursive bisection.
************************************************************************************/
void PARGMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int ndims, float *xyz,
               idxtype *part, int *options, MPI_Comm comm)
{

  ParMETIS_PartGeom(vtxdist, &ndims, xyz, part, &comm);

  options[0] = -1;

}

/*****************************************************************************
*  This function performs refinement on a partitioning.
*****************************************************************************/
void ParMETIS_RefineKway(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *options,
       int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int nparts;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  int myoptions[10];

  MPI_Comm_size(*comm, &nparts);
  tpwgts = fmalloc(nparts*ncon, "tpwgts");
  for (i=0; i<nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
    myoptions[PMV3_OPTION_PSR] = COUPLED;
  }

  ParMETIS_V3_RefineKway(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag,
  &ncon, &nparts, tpwgts, ubvec, myoptions, edgecut, part, comm);

  free(tpwgts);
}


/***********************************************************************************
* This function is the entry point of the parallel k-way multilevel partitionioner.
* This function assumes nothing about the graph distribution.
* It is the general case.
************************************************************************************/
void PARRMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
               idxtype *part, int *options, MPI_Comm comm)
{
  int wgtflag, numflag, edgecut, newoptions[5];

  newoptions[0] = 1;
  newoptions[OPTION_IPART] = options[2];
  newoptions[OPTION_FOLDF] = options[1];
  newoptions[OPTION_DBGLVL] = options[4];

  numflag = options[3];
  wgtflag = (vwgt == NULL ? 0 : 2) + (adjwgt == NULL ? 0 : 1);

  ParMETIS_RefineKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag,
     newoptions, &edgecut, part, &comm);

  options[0] = edgecut;

}


/*****************************************************************************
*  This function computes a repartitioning by local diffusion.
*****************************************************************************/
void ParMETIS_RepartLDiffusion(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, 
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *options,
       int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int nparts;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  float ipc_factor = 1.0;
  int myoptions[10];

  MPI_Comm_size(*comm, &nparts);
  tpwgts = fmalloc(nparts*ncon, "tpwgts");
  for (i=0; i<nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
    myoptions[PMV3_OPTION_PSR] = COUPLED;
  }

  ParMETIS_V3_AdaptiveRepart(vtxdist, xadj, adjncy, vwgt, NULL, adjwgt, wgtflag, numflag,
  &ncon, &nparts, tpwgts, ubvec, &ipc_factor, myoptions, edgecut, part, comm);

  free(tpwgts);
}


/***********************************************************************************
* This function is the entry point of the parallel multilevel undirected diffusion
* algorithm. It uses parallel undirected diffusion followed by adaptive k-way
* refinement. This function utilizes local coarsening.
************************************************************************************/
void PARUAMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
               idxtype *part, int *options, MPI_Comm comm)
{
  int wgtflag, numflag, edgecut, newoptions[5];

  newoptions[0] = 1;
  newoptions[OPTION_IPART] = options[2];
  newoptions[OPTION_FOLDF] = options[1];
  newoptions[OPTION_DBGLVL] = options[4];

  numflag = options[3];
  wgtflag = (vwgt == NULL ? 0 : 2) + (adjwgt == NULL ? 0 : 1);

  ParMETIS_RepartLDiffusion(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag,
     newoptions, &edgecut, part, &comm);

  options[0] = edgecut;

}

/*****************************************************************************
*  This function computes a repartitioning by global diffusion.
*****************************************************************************/
void ParMETIS_RepartGDiffusion(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, 
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *options,
       int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int nparts;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  float ipc_factor = 100.0;
  int myoptions[10];

  MPI_Comm_size(*comm, &nparts);
  tpwgts = fmalloc(nparts*ncon, "tpwgts");
  for (i=0; i<nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
    myoptions[PMV3_OPTION_PSR] = COUPLED;
  }

  ParMETIS_V3_AdaptiveRepart(vtxdist, xadj, adjncy, vwgt, NULL, adjwgt, wgtflag, numflag,
  &ncon, &nparts, tpwgts, ubvec, &ipc_factor, myoptions, edgecut, part, comm);

  free(tpwgts);
}

/***********************************************************************************
* This function is the entry point of the parallel multilevel directed diffusion
* algorithm. It uses parallel undirected diffusion followed by adaptive k-way
* refinement. This function utilizes local coarsening.
************************************************************************************/
void PARDAMETIS(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
               idxtype *part, int *options, MPI_Comm comm)
{
  int wgtflag, numflag, edgecut, newoptions[5];

  newoptions[0] = 1;
  newoptions[OPTION_IPART] = options[2];
  newoptions[OPTION_FOLDF] = options[1];
  newoptions[OPTION_DBGLVL] = options[4];

  numflag = options[3];
  wgtflag = (vwgt == NULL ? 0 : 2) + (adjwgt == NULL ? 0 : 1);

  ParMETIS_RepartGDiffusion(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag,
     newoptions, &edgecut, part, &comm);

  options[0] = edgecut;

}

/*****************************************************************************
*  This function computes a repartitioning by scratch-remap.
*****************************************************************************/
void ParMETIS_RepartRemap(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *options,
       int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int nparts;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  float ipc_factor = 1000.0;
  int myoptions[10];

  MPI_Comm_size(*comm, &nparts);
  tpwgts = fmalloc(nparts*ncon, "tpwgts");
  for (i=0; i<nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
    myoptions[PMV3_OPTION_PSR] = COUPLED;
  }

  ParMETIS_V3_AdaptiveRepart(vtxdist, xadj, adjncy, vwgt, NULL, adjwgt, wgtflag, numflag,
  &ncon, &nparts, tpwgts, ubvec, &ipc_factor, myoptions, edgecut, part, comm);

  free(tpwgts);
}


/*****************************************************************************
*  This function computes a repartitioning by LMSR scratch-remap.
*****************************************************************************/
void ParMETIS_RepartMLRemap(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
       idxtype *vwgt, idxtype *adjwgt, int *wgtflag, int *numflag, int *options,
       int *edgecut, idxtype *part, MPI_Comm *comm)
{
  int i;
  int nparts;
  int ncon = 1;
  float *tpwgts, ubvec[MAXNCON];
  float ipc_factor = 1000.0;
  int myoptions[10];

  MPI_Comm_size(*comm, &nparts);
  tpwgts = fmalloc(nparts*ncon, "tpwgts");
  for (i=0; i<nparts*ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);
  for (i=0; i<ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  if (options[0] == 0) {
    myoptions[0] = 0;
  }
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
    myoptions[PMV3_OPTION_PSR] = COUPLED;
  }

  ParMETIS_V3_AdaptiveRepart(vtxdist, xadj, adjncy, vwgt, NULL, adjwgt, wgtflag, numflag,
  &ncon, &nparts, tpwgts, ubvec, &ipc_factor, myoptions, edgecut, part, comm);

  free(tpwgts);
}

/***********************************************************************************
* This function is the entry point of the parallel ordering algorithm.
* This function assumes that the graph is already nice partitioned among the
* processors and then proceeds to perform recursive bisection.
************************************************************************************/
void ParMETIS_NodeND(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag,
  int *options, idxtype *order, idxtype *sizes, MPI_Comm *comm)
{
  int myoptions[10];

  if (options[0] == 0) {
    myoptions[0] = 0;
  } 
  else {
    myoptions[0] = 1;
    myoptions[PMV3_OPTION_DBGLVL] = options[OPTION_DBGLVL];
    myoptions[PMV3_OPTION_SEED] = GLOBAL_SEED;
    myoptions[PMV3_OPTION_IPART] = options[OPTION_IPART];
  }

  ParMETIS_V3_NodeND(vtxdist, xadj, adjncy, numflag, myoptions, order, sizes, comm);
}

