/*
 * pspases.c
 *
 * This file contains ordering routines that are to be used with the
 * parallel Cholesky factorization code PSPASES
 *
 * Started 10/14/97
 * George
 *
 * $Id: pspases.c,v 1.3 2003/07/21 17:18:53 karypis Exp $
 *
 */

#include <parmetislib.h>


/***********************************************************************************
* This function is the entry point of the serial ordering algorithm.
************************************************************************************/
void ParMETIS_SerialNodeND(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag,
                int *options, idxtype *order, idxtype *sizes, MPI_Comm *comm)
{
  int i, npes, mype, seroptions[10];
  CtrlType ctrl;
  GraphType *agraph;
  idxtype *perm=NULL, *iperm=NULL;
  int *sendcount, *displs;

  MPI_Comm_size(*comm, &npes);
  MPI_Comm_rank(*comm, &mype);

  if (!ispow2(npes)) {
    if (mype == 0)
      printf("Error: The number of processors must be a power of 2!\n");
    return;
  }

  if (*numflag == 1) 
    ChangeNumbering(vtxdist, xadj, adjncy, order, npes, mype, 1);

  SetUpCtrl(&ctrl, npes, options[OPTION_DBGLVL], *comm);

  IFSET(ctrl.dbglvl, DBG_TIME, InitTimers(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.TotalTmr));

  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.MoveTmr));

  agraph = AssembleEntireGraph(&ctrl, vtxdist, xadj, adjncy);

  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.MoveTmr));


  if (mype == 0) {
    perm = idxmalloc(agraph->nvtxs, "PAROMETISS: perm");
    iperm = idxmalloc(agraph->nvtxs, "PAROMETISS: iperm");

    seroptions[0] = 0;
    /*
    seroptions[1] = 3;
    seroptions[2] = 1;
    seroptions[3] = 2;
    seroptions[4] = 128;
    seroptions[5] = 1;
    seroptions[6] = 0;
    seroptions[7] = 1;
    */

    METIS_NodeNDP(agraph->nvtxs, agraph->xadj, agraph->adjncy, npes, seroptions, perm, iperm, sizes);
  }

  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, starttimer(ctrl.MoveTmr));

  /* Broadcast the sizes array */
  MPI_Bcast((void *)sizes, 2*npes, IDX_DATATYPE, 0, ctrl.gcomm);

  /* Scatter the iperm */
  sendcount = imalloc(npes, "PAROMETISS: sendcount");
  displs = imalloc(npes, "PAROMETISS: displs");
  for (i=0; i<npes; i++) {
    sendcount[i] = vtxdist[i+1]-vtxdist[i];
    displs[i] = vtxdist[i];
  }

  MPI_Scatterv((void *)iperm, sendcount, displs, IDX_DATATYPE, (void *)order, vtxdist[mype+1]-vtxdist[mype], IDX_DATATYPE, 0, ctrl.gcomm);

  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.MoveTmr));

  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));
  IFSET(ctrl.dbglvl, DBG_TIME, stoptimer(ctrl.TotalTmr));
  IFSET(ctrl.dbglvl, DBG_TIME, PrintTimingInfo(&ctrl));
  IFSET(ctrl.dbglvl, DBG_TIME, MPI_Barrier(ctrl.gcomm));

  GKfree((void **)&agraph->xadj, (void **)&agraph->adjncy, (void **)&perm, (void **)&iperm, (void **)&sendcount, (void **)&displs, LTERM);
  free(agraph);
  FreeCtrl(&ctrl);

  if (*numflag == 1) 
    ChangeNumbering(vtxdist, xadj, adjncy, order, npes, mype, 0);

}



/*************************************************************************
* This function assembles the graph into a single processor
**************************************************************************/
GraphType *AssembleEntireGraph(CtrlType *ctrl, idxtype *vtxdist, idxtype *xadj, idxtype *adjncy)
{
  int i, gnvtxs, nvtxs, gnedges, nedges;
  int npes = ctrl->npes, mype = ctrl->mype;
  idxtype *axadj, *aadjncy;
  int *recvcounts, *displs;
  GraphType *agraph;

  gnvtxs = vtxdist[npes];
  nvtxs = vtxdist[mype+1]-vtxdist[mype];
  nedges = xadj[nvtxs];

  recvcounts = imalloc(npes, "AssembleGraph: recvcounts");
  displs = imalloc(npes+1, "AssembleGraph: displs");

  /* Gather all the xadj arrays first */
  for (i=0; i<nvtxs; i++)
    xadj[i] = xadj[i+1]-xadj[i];

  axadj = idxmalloc(gnvtxs+1, "AssembleEntireGraph: axadj");

  for (i=0; i<npes; i++) {
    recvcounts[i] = vtxdist[i+1]-vtxdist[i];
    displs[i] = vtxdist[i];
  }

  /* Assemble the xadj and then the adjncy */
  MPI_Gatherv((void *)xadj, nvtxs, IDX_DATATYPE, axadj, recvcounts, displs, IDX_DATATYPE, 0, ctrl->comm);

  MAKECSR(i, nvtxs, xadj);
  MAKECSR(i, gnvtxs, axadj);

  /* Gather all the adjncy arrays next */
  /* Determine the # of edges stored at each processor */
  MPI_Allgather((void *)(&nedges), 1, MPI_INT, (void *)recvcounts, 1, MPI_INT, ctrl->comm);
  
  displs[0] = 0;
  for (i=1; i<npes+1; i++) 
    displs[i] = displs[i-1] + recvcounts[i-1];
  gnedges = displs[npes];

  aadjncy = idxmalloc(gnedges, "AssembleEntireGraph: aadjncy");

  /* Assemble the xadj and then the adjncy */
  MPI_Gatherv((void *)adjncy, nedges, IDX_DATATYPE, aadjncy, recvcounts, displs, IDX_DATATYPE, 0, ctrl->comm);

  /* myprintf(ctrl, "Gnvtxs: %d, Gnedges: %d\n", gnvtxs, gnedges); */

  agraph = CreateGraph();
  agraph->nvtxs = gnvtxs;
  agraph->nedges = gnedges;
  agraph->xadj = axadj;
  agraph->adjncy = aadjncy; 

  return agraph;
}
