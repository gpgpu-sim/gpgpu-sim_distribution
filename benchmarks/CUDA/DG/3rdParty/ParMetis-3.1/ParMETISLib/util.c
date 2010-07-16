/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * util.c
 *
 * This function contains various utility routines
 *
 * Started 9/28/95
 * George
 *
 * $Id: util.c,v 1.2 2003/07/21 17:18:54 karypis Exp $
 */

#include <parmetislib.h>


/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void errexit(char *f_str,...)
{
  va_list argp;
  char out1[256], out2[256];

  va_start(argp, f_str);
  vsprintf(out1, f_str, argp);
  va_end(argp);

  sprintf(out2, "Error! %s", out1);

  fprintf(stdout, out2);
  fflush(stdout);

  abort();
}


/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void myprintf(CtrlType *ctrl, char *f_str,...)
{
  va_list argp;
  char out1[256], out2[256];

  va_start(argp, f_str);
  vsprintf(out1, f_str, argp);
  va_end(argp);

  sprintf(out2, "[%2d] %s", ctrl->mype, out1);

  fprintf(stdout, out2);
  fflush(stdout);

}



/*************************************************************************
* This function prints an error message and exits
**************************************************************************/
void rprintf(CtrlType *ctrl, char *f_str,...)
{
  va_list argp;

  if (ctrl->mype == 0) {
    va_start(argp, f_str);
    vfprintf(stdout, f_str, argp);
    va_end(argp);
  }

  fflush(stdout);

  MPI_Barrier(ctrl->comm);

}


#ifndef DMALLOC
/*************************************************************************
* The following function allocates an array of integers
**************************************************************************/
int *imalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (int *)GKmalloc(sizeof(int)*n, msg);
}


/*************************************************************************
* The following function allocates an array of integers
**************************************************************************/
idxtype *idxmalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (idxtype *)GKmalloc(sizeof(idxtype)*n, msg);
}


/*************************************************************************
* The following function allocates an array of float 
**************************************************************************/
float *fmalloc(int n, char *msg)
{
  if (n == 0)
    return NULL;

  return (float *)GKmalloc(sizeof(float)*n, msg);
}


/*************************************************************************
* The follwoing function allocates an array of integers
**************************************************************************/
int *ismalloc(int n, int ival, char *msg)
{
  if (n == 0)
    return NULL;

  return iset(n, ival, (int *)GKmalloc(sizeof(int)*n, msg));
}



/*************************************************************************
* The follwoing function allocates an array of integers
**************************************************************************/
idxtype *idxsmalloc(int n, idxtype ival, char *msg)
{
  if (n == 0)
    return NULL;

  return idxset(n, ival, (idxtype *)GKmalloc(sizeof(idxtype)*n, msg));
}


/*************************************************************************
* This function is my wrapper around malloc
**************************************************************************/
void *GKmalloc(int nbytes, char *msg)
{
  void *ptr;

  if (nbytes == 0)
    return NULL;

  ptr = (void *)malloc(nbytes);
  if (ptr == NULL) 
    errexit("***Memory allocation failed for %s. Requested size: %d bytes", msg, nbytes);

  return ptr;
}
#endif

/*************************************************************************
* This function is my wrapper around free, allows multiple pointers    
**************************************************************************/
void GKfree(void **ptr1,...)
{
  va_list plist;
  void **ptr;

  if (*ptr1 != NULL)
    free(*ptr1);
  *ptr1 = NULL;

  va_start(plist, ptr1);

  while ((ptr = va_arg(plist, void **)) != LTERM) {
    if (*ptr != NULL)
      free(*ptr);
    *ptr = NULL;
  }

  va_end(plist);
}            


/*************************************************************************
* These functions set the values of a vector
**************************************************************************/
int *iset(int n, int val, int *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = val;

  return x;
}


/*************************************************************************
* These functions set the values of a vector
**************************************************************************/
idxtype *idxset(int n, idxtype val, idxtype *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = val;

  return x;
}



/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int idxamax(int n, idxtype *x)
{
  int i, max=0;

  for (i=1; i<n; i++)
    max = (x[i] > x[max] ? i : max);

  return max;
}


/*************************************************************************
* These functions return the index of the minimum element in a vector
**************************************************************************/
int idxamin(int n, idxtype *x)
{
  int i, min=0;

  for (i=1; i<n; i++)
    min = (x[i] < x[min] ? i : min);

  return min;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int idxsum(int n, idxtype *x)
{
  int i, sum = 0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int charsum(int n, char *x)
{
  int i, sum = 0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int isum(int n, int *x)
{
  int i, sum = 0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
float snorm2(int n, float *v)
{
  int i;
  float partial = 0;
 
  for (i = 0; i<n; i++)
    partial += v[i] * v[i];

  return sqrt(partial);
}



/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
float sdot(int n, float *x, float *y)
{
  int i;
  float partial = 0;
 
  for (i = 0; i<n; i++)
    partial += x[i] * y[i];

  return partial;
}


/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
void saxpy(int n, float alpha, float *x, float *y)
{
  int i;
 
  for (i=0; i<n; i++)
    y[i] += alpha*x[i];
}






/*************************************************************************
* This function sorts an array of type KeyValueType in increasing order
**************************************************************************/
void ikeyvalsort_org(int n, KeyValueType *nodes)
{
  qsort((void *)nodes, (size_t)n, (size_t)sizeof(KeyValueType), IncKeyValueCmp);
}


/*************************************************************************
* This function compares 2 KeyValueType variables for sorting in inc order
**************************************************************************/
int IncKeyValueCmp(const void *v1, const void *v2)
{
  KeyValueType *n1, *n2;

  n1 = (KeyValueType *)v1;
  n2 = (KeyValueType *)v2;

  return (n1->key != n2->key ? n1->key - n2->key : n1->val - n2->val);
}



/*************************************************************************
* This function sorts an array of type KeyValueType in increasing order
**************************************************************************/
void dkeyvalsort(int n, KeyValueType *nodes)
{
  qsort((void *)nodes, (size_t)n, (size_t)sizeof(KeyValueType), DecKeyValueCmp);
}


/*************************************************************************
* This function compares 2 KeyValueType variables for sorting in inc order
**************************************************************************/
int DecKeyValueCmp(const void *v1, const void *v2)
{
  KeyValueType *n1, *n2;

  n1 = (KeyValueType *)v1;
  n2 = (KeyValueType *)v2;

  return n2->key - n1->key;

}



/*************************************************************************
* This function does a binary search on an array for a key and returns
* the index
**************************************************************************/
int BSearch(int n, idxtype *array, int key)
{
  int a=0, b=n, c;

  while (b-a > 8) {
    c = (a+b)>>1;
    if (array[c] > key)
      b = c;
    else
      a = c;
  }

  for (c=a; c<b; c++) {
    if (array[c] == key)
      return c;
  }

  errexit("Key %d not found!\n", key);

  return 0;
}



/*************************************************************************
* This file randomly permutes the contents of an array.
* flag == 0, don't initialize perm
* flag == 1, set p[i] = i 
**************************************************************************/
void RandomPermute(int n, idxtype *p, int flag)
{
  int i, u, v;
  idxtype tmp;

  if (flag == 1) {
    for (i=0; i<n; i++)
      p[i] = i;
  }

  for (i=0; i<n; i++) {
    v = RandomInRange(n);
    u = RandomInRange(n);
    SWAP(p[v], p[u], tmp);
  }
}


/*************************************************************************
* This file randomly permutes the contents of an array.
* flag == 0, don't initialize perm
* flag == 1, set p[i] = i 
**************************************************************************/
void FastRandomPermute(int n, idxtype *p, int flag)
{
  int i, u, v;
  idxtype tmp;

  /* this is for very small arrays */
  if (n < 25) {
    RandomPermute(n, p, flag);
    return;
  }

  if (flag == 1) {
    for (i=0; i<n; i++)
      p[i] = i;
  }

  for (i=0; i<n; i+=8) {
    v = RandomInRange(n-4);
    u = RandomInRange(n-4);
    SWAP(p[v], p[u], tmp);
    SWAP(p[v+1], p[u+1], tmp);
    SWAP(p[v+2], p[u+2], tmp);
    SWAP(p[v+3], p[u+3], tmp);
  }
}

/*************************************************************************
* This function returns true if the a is a power of 2
**************************************************************************/
int ispow2(int a)
{
  for (; a%2 != 1; a = a>>1);
  return (a > 1 ? 0 : 1);
}

/*************************************************************************
* This function returns the log2(x)
**************************************************************************/
int log2Int(int a)
{
  int i;

  for (i=1; a > 1; i++, a = a>>1);
  return i-1;
}


/*************************************************************************
* These functions set the values of a vector
**************************************************************************/
float *sset(int n, float val, float *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = val;

  return x;
}



/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int iamax(int n, int *x)
{
  int i, max=0;

  for (i=1; i<n; i++)
    max = (x[i] > x[max] ? i : max);

  return max;
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int samax_strd(int n, float *x, int incx)
{
  int i;
  int max=0;

  n *= incx;
  for (i=incx; i<n; i+=incx)
    max = (x[i] > x[max] ? i : max);

  return max/incx;
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int sfamax(int n, float *x)
{
  int i;
  int max=0;

  for (i=1; i<n; i++)
    max = (fabs(x[i]) > fabs(x[max]) ? i : max);

  return max;
}



/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int samin_strd(int n, float *x, int incx)
{
  int i;
  int min=0;

  n *= incx;
  for (i=incx; i<n; i+=incx)
    min = (x[i] < x[min] ? i : min);

  return min/incx;
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int idxamax_strd(int n, idxtype *x, int incx)
{
  int i, max=0;

  n *= incx;
  for (i=incx; i<n; i+=incx)
    max = (x[i] > x[max] ? i : max);

  return max/incx;
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int idxamin_strd(int n, idxtype *x, int incx)
{
  int i, min=0;

  n *= incx;
  for (i=incx; i<n; i+=incx)
    min = (x[i] < x[min] ? i : min);

  return min/incx;
}


/*************************************************************************
* This function returns the average value of an array
**************************************************************************/
float idxavg(int n, idxtype *x)
{
  int i;
  float retval = 0.0;

  for (i=0; i<n; i++)
    retval += (float)(x[i]);

  return retval / (float)(n);
}


/*************************************************************************
* This function returns the average value of an array
**************************************************************************/
float savg(int n, float *x)
{
  int i;
  float retval = 0.0;

  for (i=0; i<n; i++)
    retval += x[i];

  return retval / (float)(n);
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int samax(int n, float *x)
{
  int i, max=0;

  for (i=1; i<n; i++)
    max = (x[i] > x[max] ? i : max);

  return max;
}


/*************************************************************************
* These functions return the index of the maximum element in a vector
**************************************************************************/
int sfavg(int n, float *x)
{
  int i;
  float total = 0.0;

  if (n == 0)
    return 0.0;

  for (i=0; i<n; i++)
    total += fabs(x[i]);

  return total / (float) n;
}


/*************************************************************************
* These functions return the index of the almost maximum element in a vector
**************************************************************************/
int samax2(int n, float *x)
{
  int i, max1, max2;

  if (x[0] > x[1]) {
    max1 = 0;
    max2 = 1;
  }
  else {
    max1 = 1;
    max2 = 0;
  }

  for (i=2; i<n; i++) {
    if (x[i] > x[max1]) {
      max2 = max1;
      max1 = i;
    }
    else if (x[i] > x[max2])
      max2 = i;
  }

  return max2;
}


/*************************************************************************
* These functions return the index of the minimum element in a vector
**************************************************************************/
int samin(int n, float *x)
{
  int i, min=0;

  for (i=1; i<n; i++)
    min = (x[i] < x[min] ? i : min);

  return min;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
int idxsum_strd(int n, idxtype *x, int incx)
{
  int i, sum = 0;

  for (i=0; i<n; i++, x+=incx) {
    sum += *x;
  }

  return sum;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
void idxadd(int n, idxtype *x, idxtype *y)
{
  for (n--; n>=0; n--)
    y[n] += x[n];
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
float ssum(int n, float *x)
{
  int i;
  float sum = 0.0;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
float ssum_strd(int n, float *x, int incx)
{
  int i;
  float sum = 0.0;

  for (i=0; i<n; i++, x+=incx)
    sum += *x;

  return sum;
}

/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
void sscale(int n, float alpha, float *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] *= alpha;
}


/*************************************************************************
* This function negates the entries in an array
**************************************************************************/
void saneg(int n, float *x)
{
  int i;

  for (i=0; i<n; i++)
    x[i] = -1.0*x[i];
}



/*************************************************************************
* This function checks if v+u2 provides a better balance in the weight
* vector that v+u1
**************************************************************************/
float BetterVBalance(int ncon, float *vwgt, float *u1wgt, float *u2wgt)
{
  int i;
  float sum1, sum2, diff1, diff2;

  if (ncon == 1)
    return u1wgt[0] - u1wgt[0];

  sum1 = sum2 = 0.0;
  for (i=0; i<ncon; i++) {
    sum1 += vwgt[i]+u1wgt[i];
    sum2 += vwgt[i]+u2wgt[i];
  }
  sum1 = sum1/(1.0*ncon);
  sum2 = sum2/(1.0*ncon);

  diff1 = diff2 = 0.0;
  for (i=0; i<ncon; i++) {
    diff1 += fabs(sum1 - (vwgt[i]+u1wgt[i]));
    diff2 += fabs(sum2 - (vwgt[i]+u2wgt[i]));
  }

  return diff1 - diff2;

}


/*************************************************************************
* This function checks if the pairwise balance of the between the two
* partitions will improve by moving the vertex v from pfrom to pto,
* subject to the target partition weights of tfrom, and tto respectively
**************************************************************************/
int IsHBalanceBetterFT(int ncon, float *pfrom, float *pto, float *nvwgt, float *ubvec)
{
  int i;
  float blb1=0.0, alb1=0.0, sblb=0.0, salb=0.0;
  float blb2=0.0, alb2=0.0;
  float temp;

  for (i=0; i<ncon; i++) {
    temp = amax(pfrom[i], pto[i])/ubvec[i];
    if (blb1 < temp) {
      blb2 = blb1;
      blb1 = temp;
    }
    else if (blb2 < temp)
      blb2 = temp;
    sblb += temp;

    temp = amax(pfrom[i]-nvwgt[i], pto[i]+nvwgt[i])/ubvec[i];
    if (alb1 < temp) {
      alb2 = alb1;
      alb1 = temp;
    }
    else if (alb2 < temp)
      alb2 = temp;
    salb += temp;
  }

  if (alb1 < blb1)
    return 1;
  if (blb1 < alb1)
    return 0;
  if (alb2 < blb2)
    return 1;
  if (blb2 < alb2)
    return 0;

  return salb < sblb;

}

/*************************************************************************
* This function checks if it will be better to move a vertex to pt2 than
* to pt1 subject to their target weights of tt1 and tt2, respectively
* This routine takes into account the weight of the vertex in question
**************************************************************************/
int IsHBalanceBetterTT(int ncon, float *pt1, float *pt2, float *nvwgt, float *ubvec)
{
  int i;
  float m11=0.0, m12=0.0, m21=0.0, m22=0.0, sm1=0.0, sm2=0.0, temp;

  for (i=0; i<ncon; i++) {
    temp = (pt1[i]+nvwgt[i])/ubvec[i];
    if (m11 < temp) {
      m12 = m11;
      m11 = temp;
    }
    else if (m12 < temp)
      m12 = temp;
    sm1 += temp;
    temp = (pt2[i]+nvwgt[i])/ubvec[i];
    if (m21 < temp) {
      m22 = m21;
      m21 = temp;
    }
    else if (m22 < temp)
      m22 = temp;
    sm2 += temp;
  }
  if (m21 < m11)
    return 1;
  if (m21 > m11)
    return 0;
  if (m22 < m12)
    return 1;
  if (m22 > m12)
    return 0;

  return sm2 < sm1;
}

/*************************************************************************
*  This is a comparison function
**************************************************************************/
int myvalkeycompare(const void *fptr, const void *sptr)
{
  KVType *first, *second;

  first = (KVType *)(fptr);
  second = (KVType *)(sptr);

  if (first->val > second->val)
    return 1;

  if (first->val < second->val)
    return -1;

  return 0;
}

/*************************************************************************
*  This is the inverse comparison function
**************************************************************************/
int imyvalkeycompare(const void *fptr, const void *sptr)
{
  KVType *first, *second;

  first = (KVType *)(fptr);
  second = (KVType *)(sptr);

  if (first->val > second->val)
    return -1;

  if (first->val < second->val)
    return 1;

  return 0;
}


/*************************************************************************
* The following function allocates and sets an array of floats
**************************************************************************/
float *fsmalloc(int n, float fval, char *msg)
{
  if (n == 0)
    return NULL;

  return sset(n, fval, (float *)GKmalloc(sizeof(float)*n, msg));
}


/*************************************************************************
* This function computes a 2-norm
**************************************************************************/
void saxpy2(int n, float alpha, float *x, int incx, float *y, int incy)
{
  int i;

  for (i=0; i<n; i++, x+=incx, y+=incy)
    *y += alpha*(*x);
}


/*************************************************************************
* This function computes the top three values of a float array
**************************************************************************/
void GetThreeMax(int n, float *x, int *first, int *second, int *third)
{
  int i;

  if (n <= 0) {
    *first = *second = *third = -1;
    return;
  }

  *second = *third = -1;
  *first = 0;

  for (i=1; i<n; i++) {
    if (x[i] > x[*first]) {
      *third = *second;
      *second = *first;
      *first = i;
      continue;
    }

    if (*second == -1 || x[i] > x[*second]) {
      *third = *second;
      *second = i;
      continue;
    }

    if (*third == -1 || x[i] > x[*third])
      *third = i;
  }

  return;
}
