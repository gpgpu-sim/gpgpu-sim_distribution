/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * csrmatch.c
 *
 * This file contains the code that computes matchings
 *
 * Started 7/23/97
 * George
 *
 * $Id: csrmatch.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 *
 */

#include <parmetislib.h>




/*************************************************************************
* This function finds a matching using the HEM heuristic
**************************************************************************/
void CSR_Match_SHEM(MatrixType *matrix, idxtype *match, idxtype *mlist,
     idxtype *skip, int ncon)
{
  int h, i, ii, j;
  int nrows, edge, maxidx, count;
  float maxwgt;
  idxtype *rowptr, *colind;
  float *transfer;
  KVType *links;

  nrows = matrix->nrows;
  rowptr = matrix->rowptr;
  colind = matrix->colind;
  transfer = matrix->transfer;

  idxset(nrows, UNMATCHED, match);

  links = (KVType *)GKmalloc(sizeof(KVType)*nrows, "links");
  for (i=0; i<nrows; i++) { 
    links[i].key = i; 
    links[i].val = 0.0;
  }

  for (i=0; i<nrows; i++)
    for (j=rowptr[i]; j<rowptr[i+1]; j++) 
      for (h=0; h<ncon; h++)
        if (links[i].val < fabs(transfer[j*ncon+h]))
          links[i].val = fabs(transfer[j*ncon+h]);

  qsort(links, nrows, sizeof(KVType), myvalkeycompare);

  count = 0;
  for (ii=0; ii<nrows; ii++) {
    i = links[ii].key;

    if (match[i] == UNMATCHED) {
      maxidx = i;
      maxwgt = 0.0;

      /* Find a heavy-edge matching */
      for (j=rowptr[i]; j<rowptr[i+1]; j++) {
        edge = colind[j];
        if (match[edge] == UNMATCHED && edge != i && skip[j] == 0) {
          for (h=0; h<ncon; h++)
            if (maxwgt < fabs(transfer[j*ncon+h]))
              break;

          if (h != ncon) {
            maxwgt = fabs(transfer[j*ncon+h]);
            maxidx = edge;
          }
        }
      }

      if (maxidx != i) {
        match[i] = maxidx;
        match[maxidx] = i;
        mlist[count++] = amax(i, maxidx);
        mlist[count++] = amin(i, maxidx);
      }
    }
  }

  GKfree((void **)&links, LTERM);
}

