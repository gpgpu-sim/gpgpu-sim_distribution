/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * debug.c
 *
 * This file contains various functions that are used to display debuging 
 * information
 *
 * Started 10/20/96
 * George
 *
 * $Id: debug.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function prints a vector stored in each processor 
**************************************************************************/
void PrintVector(CtrlType *ctrl, int n, int first, idxtype *vec, char *title)
{
  int i, penum;

  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      if (ctrl->mype == 0)
        printf("%s\n", title);
      printf("\t%3d. ", ctrl->mype);
      for (i=0; i<n; i++)
        printf("[%d %hd] ", first+i, vec[i]);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(ctrl->comm);
  }
}


/*************************************************************************
* This function prints a vector stored in each processor 
**************************************************************************/
void PrintVector2(CtrlType *ctrl, int n, int first, idxtype *vec, char *title)
{
  int i, penum;

  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      if (ctrl->mype == 0)
        printf("%s\n", title);
      printf("\t%3d. ", ctrl->mype);
      for (i=0; i<n; i++)
        printf("[%d %d.%hd] ", first+i, (vec[i]>=KEEP_BIT ? 1 : 0), (vec[i]>=KEEP_BIT ? vec[i]-KEEP_BIT : vec[i]));
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(ctrl->comm);
  }
}


/*************************************************************************
* This function prints a vector stored in each processor 
**************************************************************************/
void PrintPairs(CtrlType *ctrl, int n, KeyValueType *pairs, char *title)
{
  int i, penum;

  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      if (ctrl->mype == 0)
        printf("%s\n", title);
      printf("\t%3d. ", ctrl->mype);
      for (i=0; i<n; i++)
        printf("[%d %hd,%hd] ", i, pairs[i].key, pairs[i].val);
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(ctrl->comm);
  }
}



/*************************************************************************
* This function prints the local portion of the graph stored at each 
* processor
**************************************************************************/
void PrintGraph(CtrlType *ctrl, GraphType *graph)
{
  int i, j, penum;
  int firstvtx;

  MPI_Barrier(ctrl->comm);

  firstvtx = graph->vtxdist[ctrl->mype];

  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      printf("\t%d", penum);
      for (i=0; i<graph->nvtxs; i++) {
        if (i==0)
          printf("\t%2d %2d\t", firstvtx+i, graph->vwgt[i]);
        else
          printf("\t\t%2d %2d\t", firstvtx+i, graph->vwgt[i]);
        for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++)
          printf("[%d %d] ", graph->adjncy[j], graph->adjwgt[j]);
        printf("\n");
      }
      fflush(stdout);
    }
    MPI_Barrier(ctrl->comm);
  }
}


/*************************************************************************
* This function prints the local portion of the graph stored at each 
* processor along with degree information during refinement
**************************************************************************/
void PrintGraph2(CtrlType *ctrl, GraphType *graph)
{
  int i, j, penum;
  int firstvtx;

  MPI_Barrier(ctrl->comm);

  firstvtx = graph->vtxdist[ctrl->mype];

  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      printf("\t%d", penum);
      for (i=0; i<graph->nvtxs; i++) {
        if (i==0)
          printf("\t%2d %2d [%d %d %d]\t", firstvtx+i, graph->vwgt[i], graph->where[i], graph->rinfo[i].id, graph->rinfo[i].ed);
        else
          printf("\t\t%2d %2d [%d %d %d]\t", firstvtx+i, graph->vwgt[i], graph->where[i], graph->rinfo[i].id, graph->rinfo[i].ed);
        for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++)
          printf("[%d %d] ", graph->adjncy[j], graph->adjwgt[j]);
        printf("\n");
      }
      fflush(stdout);
    }
    MPI_Barrier(ctrl->comm);
  }
}


/*************************************************************************
* This function prints the information computed during setup
**************************************************************************/
void PrintSetUpInfo(CtrlType *ctrl, GraphType *graph)
{
  int i, j, penum;

  MPI_Barrier(ctrl->comm);

  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      printf("PE: %d, nnbrs: %d\n", ctrl->mype, graph->nnbrs);
      printf("\tSending...\n");
      for (i=0; i<graph->nnbrs; i++) {
        printf("\t\tTo: %d: ", graph->peind[i]);
        for (j=graph->sendptr[i]; j<graph->sendptr[i+1]; j++)
          printf("%d ", graph->sendind[j]);
        printf("\n");
      }
      printf("\tReceiving...\n");
      for (i=0; i<graph->nnbrs; i++) {
        printf("\t\tFrom: %d: ", graph->peind[i]);
        for (j=graph->recvptr[i]; j<graph->recvptr[i+1]; j++)
          printf("%d ", graph->recvind[j]);
        printf("\n");
      }
      printf("\n");
    }
    MPI_Barrier(ctrl->comm);
  }

}


/*************************************************************************
* This function prints information about the graphs that were sent/received
**************************************************************************/
void PrintTransferedGraphs(CtrlType *ctrl, int nnbrs, idxtype *peind, idxtype *slens, 
      idxtype *rlens, idxtype *sgraph, idxtype *rgraph)
{
  int i, ii, jj, ll, penum;

  MPI_Barrier(ctrl->comm);
  for (penum=0; penum<ctrl->npes; penum++) {
    if (ctrl->mype == penum) {
      printf("PE: %d, nnbrs: %d", ctrl->mype, nnbrs);
      for (ll=i=0; i<nnbrs; i++) {
        if (slens[i+1]-slens[i] > 0) {
          printf("\n\tTo %d\t", peind[i]);
          for (ii=slens[i]; ii<slens[i+1]; ii++) {
            printf("%d %d %d, ", sgraph[ll], sgraph[ll+1], sgraph[ll+2]);
            for (jj=0; jj<sgraph[ll+1]; jj++)
              printf("[%d %d] ", sgraph[ll+3+2*jj], sgraph[ll+3+2*jj+1]);
            printf("\n\t\t");
            ll += 3+2*sgraph[ll+1];
          }
        }
      }

      for (ll=i=0; i<nnbrs; i++) {
        if (rlens[i+1]-rlens[i] > 0) {
          printf("\n\tFrom %d\t", peind[i]);
          for (ii=rlens[i]; ii<rlens[i+1]; ii++) {
            printf("%d %d %d, ", rgraph[ll], rgraph[ll+1], rgraph[ll+2]);
            for (jj=0; jj<rgraph[ll+1]; jj++)
              printf("[%d %d] ", rgraph[ll+3+2*jj], rgraph[ll+3+2*jj+1]);
            printf("\n\t\t");
            ll += 3+2*rgraph[ll+1];
          }
        }
      }
      printf("\n");
    }
    MPI_Barrier(ctrl->comm);
  }

}


/*************************************************************************
* This function writes a graph in the format used by serial METIS
**************************************************************************/
void WriteMetisGraph(int nvtxs, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt)
{
  int i, j;
  FILE *fp;

  fp = fopen("test.graph", "w");

  fprintf(fp, "%d %d 11", nvtxs, xadj[nvtxs]/2);
  for (i=0; i<nvtxs; i++) {
    fprintf(fp, "\n%d ", vwgt[i]);
    for (j=xadj[i]; j<xadj[i+1]; j++)
      fprintf(fp, " %d %d", adjncy[j]+1, adjwgt[j]);
  }
  fclose(fp);
}

