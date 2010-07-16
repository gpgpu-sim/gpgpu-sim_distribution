/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * weird.c
 *
 * This file contain various graph setting up routines
 *
 * Started 10/19/96
 * George
 *
 * $Id: weird.c,v 1.9 2003/07/31 16:27:28 karypis Exp $
 *
 */

#include <parmetislib.h>



/*************************************************************************
* This function computes a partitioning of a small graph
**************************************************************************/
void PartitionSmallGraph(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, h, ncon, nparts, npes, mype;
  int moptions[10];
  int mynumflag, mywgtflag, me;
  idxtype *mypart;
  int lpecut[2], gpecut[2];
  GraphType *agraph;
  int *sendcounts, *displs;
  float *mytpwgts, *gnpwgts, *lnpwgts;

  ncon = graph->ncon;
  nparts = ctrl->nparts;

  MPI_Comm_size(ctrl->comm, &npes);
  MPI_Comm_rank(ctrl->comm, &mype);

  SetUp(ctrl, graph, wspace);
  graph->where = idxmalloc(graph->nvtxs+graph->nrecv, "PartitionSmallGraph: where");
  agraph       = Moc_AssembleAdaptiveGraph(ctrl, graph, wspace);
  mypart       = idxmalloc(agraph->nvtxs, "mypart");

  moptions[0] = 0;
  moptions[7] = ctrl->sync + mype;
  mynumflag = 0;
  mywgtflag = 3;
  if (ncon == 1) {
    METIS_WPartGraphKway2(&agraph->nvtxs, agraph->xadj, agraph->adjncy, agraph->vwgt, 
          agraph->adjwgt, &mywgtflag, &mynumflag, &nparts, ctrl->tpwgts, moptions, 
	  &graph->mincut, mypart);
  }
  else {
    mytpwgts = fmalloc(nparts, "mytpwgts");
    for (i=0; i<nparts; i++)
      mytpwgts[i] = ctrl->tpwgts[i*ncon];

    METIS_mCPartGraphRecursive2(&agraph->nvtxs, &ncon, agraph->xadj, agraph->adjncy, 
          agraph->vwgt, agraph->adjwgt, &mywgtflag, &mynumflag, &nparts, mytpwgts, 
	  moptions, &graph->mincut, mypart);

    free(mytpwgts);
  }

  lpecut[0] = graph->mincut;
  lpecut[1] = mype;
  MPI_Allreduce(lpecut, gpecut, 1, MPI_2INT, MPI_MINLOC, ctrl->comm);
  graph->mincut = gpecut[0];

  if (lpecut[1] == gpecut[1] && gpecut[1] != 0)
    MPI_Send((void *)mypart, agraph->nvtxs, IDX_DATATYPE, 0, 1, ctrl->comm);
  if (lpecut[1] == 0 && gpecut[1] != 0)
    MPI_Recv((void *)mypart, agraph->nvtxs, IDX_DATATYPE, gpecut[1], 1, ctrl->comm, &ctrl->status);

  sendcounts = imalloc(npes, "sendcounts");
  displs     = imalloc(npes, "displs");

  for (i=0; i<npes; i++) {
    sendcounts[i] = graph->vtxdist[i+1]-graph->vtxdist[i];
    displs[i] = graph->vtxdist[i];
  }

  MPI_Scatterv((void *)mypart, sendcounts, displs, IDX_DATATYPE,
               (void *)graph->where, graph->nvtxs, IDX_DATATYPE, 0, ctrl->comm);

  lnpwgts = graph->lnpwgts = fmalloc(nparts*ncon, "lnpwgts");
  gnpwgts = graph->gnpwgts = fmalloc(nparts*ncon, "gnpwgts");
  sset(nparts*ncon, 0, lnpwgts);
  for (i=0; i<graph->nvtxs; i++) {
    me = graph->where[i];
    for (h=0; h<ncon; h++)
      lnpwgts[me*ncon+h] += graph->nvwgt[i*ncon+h];
  }
  MPI_Allreduce((void *)lnpwgts, (void *)gnpwgts, nparts*ncon, MPI_FLOAT, MPI_SUM, ctrl->comm);
  GKfree((void**)&mypart, (void**)&sendcounts, (void**)&displs, LTERM);
  FreeGraph(agraph);

  return;
}



/*************************************************************************
* This function checks the inputs for the partitioning routines
**************************************************************************/
void CheckInputs(int partType, int npes, int dbglvl, int *wgtflag, int *iwgtflag,
                 int *numflag, int *inumflag, int *ncon, int *incon, int *nparts, 
		 int *inparts, float *tpwgts, float **itpwgts, float *ubvec, 
		 float *iubvec, float *ipc2redist, float *iipc2redist, int *options, 
		 int *ioptions, idxtype *part, MPI_Comm *comm)
{
  int i, j;
  int doweabort, doiabort = 0;
  float tsum, *myitpwgts;
  int mgcnums[5] = {-1, 2, 3, 4, 2};

  /**************************************/
  if (part == NULL) {
    doiabort = 1;
    IFSET(dbglvl, DBG_INFO, printf("ERROR: part array is set to NULL.\n"));
  }
  /**************************************/


  /**************************************/
  if (wgtflag == NULL) {
    *iwgtflag = 0;
    IFSET(dbglvl, DBG_INFO, printf("WARNING: wgtflag is NULL.  Using a value of 0.\n"));
  }
  else {
    *iwgtflag = *wgtflag;
  }
  /**************************************/


  /**************************************/
  if (numflag == NULL) {
    *inumflag = 0;
    IFSET(dbglvl, DBG_INFO, printf("WARNING: numflag is NULL.  Using a value of 0.\n"));
  }
  else {
    if (*numflag != 0 && *numflag != 1) {
      IFSET(dbglvl, DBG_INFO, printf("WARNING: bad value for numflag %d.  Using a value of 0.\n", *numflag));
      *inumflag = 0;
    }
    else {
      *inumflag = *numflag;
    }
  }
  /**************************************/


  /**************************************/
  if (ncon == NULL) {
    *incon = 1;
    IFSET(dbglvl, DBG_INFO, printf("WARNING: ncon is NULL.  Using a value of 1.\n"));
  }
  else {
    if (*ncon < 1 || *ncon > MAXNCON) {
      IFSET(dbglvl, DBG_INFO, printf("WARNING: bad value for ncon %d.  Using a value of 1.\n", *ncon));
      *incon = 1;
    }
    else {
      *incon = *ncon;
    }
  }
  /**************************************/


  /**************************************/
  if (nparts == NULL) {
    *inparts = npes;
    IFSET(dbglvl, DBG_INFO, printf("WARNING: nparts is NULL.  Using a value of %d.\n", npes));
  }
  else {
    if (*nparts < 1 || *nparts > MAX_NPARTS) {
      IFSET(dbglvl, DBG_INFO, printf("WARNING: bad value for nparts %d.  Using a value of %d.\n", *nparts, npes));
      *inparts = npes;
    }
    else {
      *inparts = *nparts;
    }
  }
  /**************************************/


  /**************************************/
  myitpwgts = *itpwgts = fmalloc((*inparts)*(*incon), "CheckInputs: itpwgts");
  if (tpwgts == NULL) {
    sset((*inparts)*(*incon), 1.0/(float)(*inparts), myitpwgts);
    IFSET(dbglvl, DBG_INFO, printf("WARNING: tpwgts is NULL.  Setting all array elements to %.3f.\n", 1.0/(float)(*inparts)));
  }
  else {
    for (i=0; i<*incon; i++) {
      tsum = 0.0;
      for (j=0; j<*inparts; j++) {
        tsum += tpwgts[j*(*incon)+i];
      } 
      if (fabs(1.0-tsum) < SMALLFLOAT)
        tsum = 1.0;
      for (j=0; j<*inparts; j++)
       myitpwgts[j*(*incon)+i] = tpwgts[j*(*incon)+i] / tsum;
    }
  }
  /**************************************/


  /**************************************/
  if (ubvec == NULL) {
    sset(*incon, 1.05, iubvec);
    IFSET(dbglvl, DBG_INFO, printf("WARNING: ubvec is NULL.  Setting all array elements to 1.05.\n"));
  }
  else {
    for (i=0; i<*incon; i++) {
      if (ubvec[i] < 1.0 || ubvec[i] > (float)(*inparts)) {
        iubvec[i] = 1.05;
        IFSET(dbglvl, DBG_INFO, printf("WARNING: bad value for ubvec[%d]: %.3f.  Setting value to 1.05.[%d]\n", i, ubvec[i], *inparts));
      }
      else {
        iubvec[i] = ubvec[i];
      }
    }
  }
  /**************************************/


  /**************************************/
  if (partType == ADAPTIVE_PARTITION) {
    if (ipc2redist != NULL) {
      if (*ipc2redist < SMALLFLOAT || *ipc2redist > 1000000.0) {
        IFSET(dbglvl, DBG_INFO, printf("WARNING: bad value for ipc2redist %.3f.  Using a value of 1000.\n", *ipc2redist));
        *iipc2redist = 1000.0;
      }
      else {
        *iipc2redist = *ipc2redist;
      }
    }
    else {
      IFSET(dbglvl, DBG_INFO, printf("WARNING: ipc2redist is NULL.  Using a value of 1000.\n"));
      *iipc2redist = 1000.0;
    }
  }
  /**************************************/


  /**************************************/
  if (options == NULL) {
    ioptions[0] = 0;
    IFSET(dbglvl, DBG_INFO, printf("WARNING: options is NULL.  Using defaults\n"));
  }
  else {
    ioptions[0] = options[0];
    ioptions[1] = options[1];
    ioptions[2] = options[2];
    if (partType == ADAPTIVE_PARTITION || partType == REFINE_PARTITION)
      ioptions[3] = options[3];
  }
  /**************************************/


  /**************************************/
  if (comm == NULL) {
    IFSET(dbglvl, DBG_INFO, printf("ERROR: comm is NULL.  Aborting\n"));
    abort();
  }
  else {
    MPI_Allreduce((void *)&doiabort, (void *)&doweabort, 1, MPI_INT, MPI_MAX, *comm);
    if (doweabort > 0)
      abort();
  }
  /**************************************/

}


