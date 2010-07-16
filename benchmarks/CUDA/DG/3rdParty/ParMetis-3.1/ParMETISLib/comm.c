/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * comm.c
 *
 * This function provides various high level communication functions 
 *
 * $Id: comm.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 */

#include <parmetislib.h>



/*************************************************************************
* This function performs the gather/scatter for the boundary vertices
**************************************************************************/
void CommInterfaceData(CtrlType *ctrl, GraphType *graph, idxtype *data, 
                       idxtype *sendvector, idxtype *recvvector)
{
  int i, k, nnbrs, firstvtx;
  idxtype *peind, *sendptr, *sendind, *recvptr, *recvind;

  firstvtx = graph->vtxdist[ctrl->mype];
  nnbrs = graph->nnbrs;
  peind = graph->peind;
  sendptr = graph->sendptr;
  sendind = graph->sendind;
  recvptr = graph->recvptr;
  recvind = graph->recvind;

  /* Issue the receives first */
  for (i=0; i<nnbrs; i++) {
    MPI_Irecv((void *)(recvvector+recvptr[i]), recvptr[i+1]-recvptr[i], IDX_DATATYPE, 
              peind[i], 1, ctrl->comm, ctrl->rreq+i);
  }

  /* Issue the sends next */
  k = sendptr[nnbrs];
  for (i=0; i<k; i++) 
    sendvector[i] = data[sendind[i]-firstvtx];

  for (i=0; i<nnbrs; i++) {
    MPI_Isend((void *)(sendvector+sendptr[i]), sendptr[i+1]-sendptr[i], IDX_DATATYPE, 
              peind[i], 1, ctrl->comm, ctrl->sreq+i); 
  }

  /* OK, now get into the loop waiting for the operations to finish */
  MPI_Waitall(nnbrs, ctrl->rreq, ctrl->statuses); 
  MPI_Waitall(nnbrs, ctrl->sreq, ctrl->statuses); 

}



/*************************************************************************
* This function performs the gather/scatter for the boundary vertices
**************************************************************************/
void CommChangedInterfaceData(CtrlType *ctrl, GraphType *graph, 
        int nchanged, idxtype *changed, idxtype *data,
        KeyValueType *sendpairs, KeyValueType *recvpairs, idxtype *psendptr)
{
  int i, j, k, n, penum, nnbrs, firstvtx, nrecv;
  idxtype *peind, *sendptr, *recvptr, *recvind, *pexadj, *peadjncy, *peadjloc;
  KeyValueType *pairs;

  firstvtx = graph->vtxdist[ctrl->mype];
  nnbrs = graph->nnbrs;
  nrecv = graph->nrecv;
  peind = graph->peind;
  sendptr = graph->sendptr;
  recvptr = graph->recvptr;
  recvind = graph->recvind;
  pexadj = graph->pexadj;
  peadjncy = graph->peadjncy;
  peadjloc = graph->peadjloc;

  /* Issue the receives first */
  for (i=0; i<nnbrs; i++) {
    MPI_Irecv((void *)(recvpairs+recvptr[i]), 2*(recvptr[i+1]-recvptr[i]), IDX_DATATYPE, 
              peind[i], 1, ctrl->comm, ctrl->rreq+i);
  }

  if (nchanged != 0) {
    idxcopy(ctrl->npes, sendptr, psendptr);

    /* Copy the changed values into the sendvector */
    for (i=0; i<nchanged; i++) {
      j = changed[i];
      for (k=pexadj[j]; k<pexadj[j+1]; k++) {
        penum = peadjncy[k];
        sendpairs[psendptr[penum]].key = peadjloc[k];
        sendpairs[psendptr[penum]].val = data[j];
        psendptr[penum]++;
      }
    }

    for (i=0; i<nnbrs; i++) {
      MPI_Isend((void *)(sendpairs+sendptr[i]), 2*(psendptr[i]-sendptr[i]), IDX_DATATYPE, 
                peind[i], 1, ctrl->comm, ctrl->sreq+i);
    }
  }
  else {
    for (i=0; i<nnbrs; i++) 
      MPI_Isend((void *)(sendpairs), 0, IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
  }

  /* OK, now get into the loop waiting for the operations to finish */
  for (i=0; i<nnbrs; i++) {
    MPI_Wait(ctrl->rreq+i, &(ctrl->status));
    MPI_Get_count(&ctrl->status, IDX_DATATYPE, &n);
    if (n != 0) {
      n = n/2;
      pairs = recvpairs+graph->recvptr[i];
      for (k=0; k<n; k++) 
        data[pairs[k].key] = pairs[k].val;
    }
  }

  MPI_Waitall(nnbrs, ctrl->sreq, ctrl->statuses);
}



/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int GlobalSEMax(CtrlType *ctrl, int value)
{
  int max;

  MPI_Allreduce((void *)&value, (void *)&max, 1, MPI_INT, MPI_MAX, ctrl->comm);

  return max;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
double GlobalSEMaxDouble(CtrlType *ctrl, double value)
{
  double max;

  MPI_Allreduce((void *)&value, (void *)&max, 1, MPI_DOUBLE, MPI_MAX, ctrl->comm);

  return max;
}



/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int GlobalSEMin(CtrlType *ctrl, int value)
{
  int min;

  MPI_Allreduce((void *)&value, (void *)&min, 1, MPI_INT, MPI_MIN, ctrl->comm);

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
int GlobalSESum(CtrlType *ctrl, int value)
{
  int sum;

  MPI_Allreduce((void *)&value, (void *)&sum, 1, MPI_INT, MPI_SUM, ctrl->comm);

  return sum;
}


/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
float GlobalSEMaxFloat(CtrlType *ctrl, float value)
{
  float max;

  MPI_Allreduce((void *)&value, (void *)&max, 1, MPI_FLOAT, MPI_MAX, ctrl->comm);

  return max;
}



/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
float GlobalSEMinFloat(CtrlType *ctrl, float value)
{
  float min;

  MPI_Allreduce((void *)&value, (void *)&min, 1, MPI_FLOAT, MPI_MIN, ctrl->comm);

  return min;
}

/*************************************************************************
* This function computes the max of a single element
**************************************************************************/
float GlobalSESumFloat(CtrlType *ctrl, float value)
{
  float sum;

  MPI_Allreduce((void *)&value, (void *)&sum, 1, MPI_FLOAT, MPI_SUM, ctrl->comm);

  return sum;
}

