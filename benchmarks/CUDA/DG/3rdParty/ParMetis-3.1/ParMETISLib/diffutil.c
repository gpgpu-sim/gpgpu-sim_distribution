/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * wavefrontK.c 
 *
 * This file contains code for the initial directed diffusion at the coarsest
 * graph
 *
 * Started 5/19/97, Kirk, George
 *
 * $Id: diffutil.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
*  This function computes the load for each subdomain
**************************************************************************/
void SetUpConnectGraph(GraphType *graph, MatrixType *matrix, idxtype *workspace)
{
  int i, ii, j, jj, k, l;
  int nvtxs, nrows;
  idxtype *xadj, *adjncy, *where;
  idxtype *rowptr, *colind;
  idxtype *pcounts, *perm, *marker;
  float *values;

  nvtxs = graph->nvtxs;
  xadj = graph->xadj;
  adjncy = graph->adjncy;
  where = graph->where;

  nrows = matrix->nrows;
  rowptr = matrix->rowptr;
  colind = matrix->colind;
  values = matrix->values;

  perm = workspace;
  marker = idxset(nrows, -1, workspace+nvtxs);
  pcounts = idxset(nrows+1, 0, workspace+nvtxs+nrows);

  for (i=0; i<nvtxs; i++)
    pcounts[where[i]]++;
  MAKECSR(i, nrows, pcounts);

  for (i=0; i<nvtxs; i++)
    perm[pcounts[where[i]]++] = i;

  for (i=nrows; i>0; i--)
    pcounts[i] = pcounts[i-1];
  pcounts[0] = 0;

  /************************/
  /* Construct the matrix */
  /************************/
  rowptr[0] = k = 0;
  for (ii=0; ii<nrows; ii++) {
    colind[k++] = ii;
    marker[ii] = ii;

    for (jj=pcounts[ii]; jj<pcounts[ii+1]; jj++) {
      i = perm[jj];
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        l = where[adjncy[j]];
        if (marker[l] != ii) {
          colind[k] = l;
          values[k++] = -1.0;
          marker[l] = ii;
        }
      }
    }
    values[rowptr[ii]] = (float)(k-rowptr[ii]-1);
    rowptr[ii+1] = k;
  }
  matrix->nnzs = rowptr[nrows];

  return;
}


/*************************************************************************
* This function computes movement statistics for adaptive refinement
* schemes
**************************************************************************/
void Mc_ComputeMoveStatistics(CtrlType *ctrl, GraphType *graph, int *nmoved, int *maxin, int *maxout)
{
  int i, nvtxs, nparts, myhome;
  idxtype *vwgt, *where;
  idxtype *lend, *gend, *lleft, *gleft, *lstart, *gstart;

  nvtxs = graph->nvtxs;
  vwgt = graph->vwgt;
  where = graph->where;
  nparts = ctrl->nparts;

  lstart = idxsmalloc(nparts, 0, "ComputeMoveStatistics: lstart");
  gstart = idxsmalloc(nparts, 0, "ComputeMoveStatistics: gstart");
  lleft = idxsmalloc(nparts, 0, "ComputeMoveStatistics: lleft");
  gleft = idxsmalloc(nparts, 0, "ComputeMoveStatistics: gleft");
  lend = idxsmalloc(nparts, 0, "ComputeMoveStatistics: lend");
  gend = idxsmalloc(nparts, 0, "ComputeMoveStatistics: gend");

  for (i=0; i<nvtxs; i++) {
    myhome = (ctrl->ps_relation == COUPLED) ? ctrl->mype : graph->home[i];
    lstart[myhome] += (graph->vsize == NULL) ? 1 : graph->vsize[i];
    lend[where[i]] += (graph->vsize == NULL) ? 1 : graph->vsize[i];
    if (where[i] != myhome)
      lleft[myhome] += (graph->vsize == NULL) ? 1 : graph->vsize[i];
  }

  /* PrintVector(ctrl, ctrl->npes, 0, lend, "Lend: "); */

  MPI_Allreduce((void *)lstart, (void *)gstart, nparts, IDX_DATATYPE, MPI_SUM, ctrl->comm);
  MPI_Allreduce((void *)lleft, (void *)gleft, nparts, IDX_DATATYPE, MPI_SUM, ctrl->comm);
  MPI_Allreduce((void *)lend, (void *)gend, nparts, IDX_DATATYPE, MPI_SUM, ctrl->comm);

  *nmoved = idxsum(nparts, gleft);
  *maxout = gleft[idxamax(nparts, gleft)];
  for (i=0; i<nparts; i++)
    lstart[i] = gend[i]+gleft[i]-gstart[i];
  *maxin = lstart[idxamax(nparts, lstart)];

  GKfree((void **)&lstart, (void **)&gstart, (void **)&lleft, (void **)&gleft, (void **)&lend, (void **)&gend, LTERM);
}

/*************************************************************************
*  This function computes the TotalV of a serial graph.
**************************************************************************/
int Mc_ComputeSerialTotalV(GraphType *graph, idxtype *home)
{
  int i;
  int totalv = 0;

  for (i=0; i<graph->nvtxs; i++) {
    if (graph->where[i] != home[i])
      totalv += (graph->vsize == NULL) ? graph->vwgt[i*graph->ncon] : graph->vsize[i];
  }

  return totalv;
}



/*************************************************************************
*  This function computes the load for each subdomain
**************************************************************************/
void ComputeLoad(GraphType *graph, int nparts, float *load, float *tpwgts, int index)
{
  int i;
  int nvtxs, ncon;
  idxtype *where;
  float *nvwgt;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  where = graph->where;
  nvwgt = graph->nvwgt;

  sset(nparts, 0.0, load);

  for (i=0; i<nvtxs; i++)
    load[where[i]] += nvwgt[i*ncon+index];

  ASSERTS(fabs(ssum(nparts, load)-1.0) < 0.001);

  for (i=0; i<nparts; i++) {
    load[i] -= tpwgts[i*ncon+index];
  }

  return;
}


/*************************************************************************
* This function implements the CG solver used during the directed diffusion
**************************************************************************/
void ConjGrad2(MatrixType *A, float *b, float *x, float tol, float *workspace)
{
  int i, k, n;
  float *p, *r, *q, *z, *M;
  float alpha, beta, rho, rho_1 = -1.0, error, bnrm2, tmp;
  idxtype *rowptr, *colind;
  float *values;

  n = A->nrows;
  rowptr = A->rowptr;
  colind = A->colind;
  values = A->values;

  /* Initial Setup */
  p = workspace;
  r = workspace + n;
  q = workspace + 2*n;
  z = workspace + 3*n;
  M = workspace + 4*n;

  for (i=0; i<n; i++) {
    x[i] = 0.0;
    if (values[rowptr[i]] != 0.0)
      M[i] = 1.0/values[rowptr[i]];
    else
      M[i] = 0.0;
  }

  /* r = b - Ax */
  mvMult2(A, x, r);
  for (i=0; i<n; i++)
    r[i] = b[i]-r[i];

  bnrm2 = snorm2(n, b);
  if (bnrm2 > 0.0) {
    error = snorm2(n, r) / bnrm2;

    if (error > tol) {
      /* Begin Iterations */
      for (k=0; k<n; k++) {
        for (i=0; i<n; i++)
          z[i] = r[i]*M[i];

        rho = sdot(n, r, z);

        if (k == 0)
          scopy(n, z, p);
        else {
          if (rho_1 != 0.0)
            beta = rho/rho_1;
          else
            beta = 0.0;
          for (i=0; i<n; i++)
            p[i] = z[i] + beta*p[i];
        }

        mvMult2(A, p, q); /* q = A*p */

        tmp = sdot(n, p, q);
        if (tmp != 0.0)
          alpha = rho/tmp;
        else
          alpha = 0.0;
        saxpy(n, alpha, p, x);    /* x = x + alpha*p */
        saxpy(n, -alpha, q, r);   /* r = r - alpha*q */
        error = snorm2(n, r) / bnrm2;
        if (error < tol)
          break;

        rho_1 = rho;
      }
    }
  }
}


/*************************************************************************
* This function performs Matrix-Vector multiplication
**************************************************************************/
void mvMult2(MatrixType *A, float *v, float *w)
{
  int i, j;

  for (i = 0; i < A->nrows; i++)
    w[i] = 0.0;

  for (i = 0; i < A->nrows; i++)
    for (j = A->rowptr[i]; j < A->rowptr[i+1]; j++)
      w[i] += A->values[j] * v[A->colind[j]];

  return;
  }


/*************************************************************************
* This function sets up the transfer vectors
**************************************************************************/
void ComputeTransferVector(int ncon, MatrixType *matrix, float *solution,
  float *transfer, int index)
{
  int j, k;
  int nrows;
  idxtype *rowptr, *colind;

  nrows = matrix->nrows;
  rowptr = matrix->rowptr;
  colind = matrix->colind;

  for (j=0; j<nrows; j++) {
    for (k=rowptr[j]+1; k<rowptr[j+1]; k++) {
      if (solution[j] > solution[colind[k]]) {
        transfer[k*ncon+index] = solution[j] - solution[colind[k]];
      }
      else {
        transfer[k*ncon+index] = 0.0;
      }
    }
  }
}

