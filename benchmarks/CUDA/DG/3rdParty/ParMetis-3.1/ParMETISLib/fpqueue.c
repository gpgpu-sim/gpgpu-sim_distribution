/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * fpqueue.c
 *
 * This file contains functions for manipulating the bucket list
 * representation of the gains associated with each vertex in a graph.
 * These functions are used by the refinement algorithms
 *
 * Started 9/2/94
 * George
 *
 * $Id: fpqueue.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function initializes the data structures of the priority queue
**************************************************************************/
void FPQueueInit(FPQueueType *queue, int maxnodes)
{
  queue->nnodes = 0;
  queue->maxnodes = maxnodes;
  queue->heap = NULL;
  queue->locator = NULL;

  queue->heap = (FKeyValueType *) malloc(sizeof(FKeyValueType)*maxnodes);
  queue->locator = (idxtype *) malloc(sizeof(idxtype)*maxnodes);

  idxset(maxnodes, -1, queue->locator);

}


/*************************************************************************
* This function resets the buckets
**************************************************************************/
void FPQueueReset(FPQueueType *queue)
{
  queue->nnodes = 0;

  idxset(queue->maxnodes, -1, queue->locator);

}


/*************************************************************************
* This function frees the buckets
**************************************************************************/
void FPQueueFree(FPQueueType *queue)
{

  free(queue->heap);
  free(queue->locator);

  queue->maxnodes = 0;
}


/*************************************************************************
* This function returns the number of nodes in the queue
**************************************************************************/
int FPQueueGetSize(FPQueueType *queue)
{
  return queue->nnodes;
}


/*************************************************************************
* This function adds a node of certain gain into a partition
**************************************************************************/
int FPQueueInsert(FPQueueType *queue, int node, float gain)
{
  int i, j;
  idxtype *locator;
  FKeyValueType *heap;

  ASSERTS(CheckHeapFloat(queue));

  heap = queue->heap;
  locator = queue->locator;

  ASSERTS(locator[node] == -1);

  i = queue->nnodes++;
  while (i > 0) {
    j = (i-1)/2;
    if (heap[j].key < gain) {
      heap[i] = heap[j];
      locator[heap[i].val] = i;
      i = j;
    }
    else 
      break;
  }
  ASSERTS(i >= 0);
  heap[i].key = gain;
  heap[i].val = node;
  locator[node] = i;

  ASSERTS(CheckHeapFloat(queue));

  return 0;
}


/*************************************************************************
* This function deletes a node from a partition and reinserts it with
* an updated gain
**************************************************************************/
int FPQueueDelete(FPQueueType *queue, int node)
{
  int i, j;
  float newgain, oldgain;
  idxtype *locator;
  FKeyValueType *heap;

  heap = queue->heap;
  locator = queue->locator;

  ASSERTS(locator[node] != -1);
  ASSERTS(heap[locator[node]].val == node);

  ASSERTS(CheckHeapFloat(queue));

  i = locator[node];
  locator[node] = -1;

  if (--queue->nnodes > 0 && heap[queue->nnodes].val != node) {
    node = heap[queue->nnodes].val;
    newgain = heap[queue->nnodes].key;
    oldgain = heap[i].key;

    if (oldgain < newgain) {
      /* Filter-up */
      while (i > 0) {
        j = (i-1)>>1;
        if (heap[j].key < newgain) {
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else 
          break;
      }
    }
    else {
      /* Filter down */
      while ((j=2*i+1) < queue->nnodes) {
        if (heap[j].key > newgain) {
          if (j+1 < queue->nnodes && heap[j+1].key > heap[j].key)
            j = j+1;
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else if (j+1 < queue->nnodes && heap[j+1].key > newgain) {
          j = j+1;
          heap[i] = heap[j];
          locator[heap[i].val] = i;
          i = j;
        }
        else
          break;
      }
    }

    heap[i].key = newgain;
    heap[i].val = node;
    locator[node] = i;
  }

  ASSERTS(CheckHeapFloat(queue));

  return 0;
}



/*************************************************************************
* This function deletes a node from a partition and reinserts it with
* an updated gain
**************************************************************************/
int FPQueueUpdate(FPQueueType *queue, int node, float oldgain, float newgain)
{
  int i, j;
  idxtype *locator;
  FKeyValueType *heap;

  if (oldgain == newgain) 
    return 0;

  heap = queue->heap;
  locator = queue->locator;

  ASSERTS(locator[node] != -1);
  ASSERTS(heap[locator[node]].val == node);
  ASSERTS(fabs(heap[locator[node]].key - oldgain) < SMALLFLOAT);
  ASSERTS(CheckHeapFloat(queue));

  i = locator[node];

  if (oldgain < newgain) {
    /* Filter-up */
    while (i > 0) {
      j = (i-1)>>1;
      if (heap[j].key < newgain) {
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else 
        break;
    }
  }
  else {
    /* Filter down */
    while ((j=2*i+1) < queue->nnodes) {
      if (heap[j].key > newgain) {
        if (j+1 < queue->nnodes && heap[j+1].key > heap[j].key)
          j = j+1;
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else if (j+1 < queue->nnodes && heap[j+1].key > newgain) {
        j = j+1;
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else
        break;
    }
  }

  heap[i].key = newgain;
  heap[i].val = node;
  locator[node] = i;

  ASSERTS(CheckHeapFloat(queue));

  return 0;
}



/*************************************************************************
* This function deletes a node from a partition and reinserts it with
* an updated gain
**************************************************************************/
void FPQueueUpdateUp(FPQueueType *queue, int node, float oldgain, float newgain)
{
  int i, j;
  idxtype *locator;
  FKeyValueType *heap;

  if (oldgain == newgain) 
    return;

  heap = queue->heap;
  locator = queue->locator;

  ASSERTS(locator[node] != -1);
  ASSERTS(heap[locator[node]].val == node);
  ASSERTS(heap[locator[node]].key == oldgain);
  ASSERTS(CheckHeapFloat(queue));


  /* Here we are just filtering up since the newgain is greater than the oldgain */
  i = locator[node];
  while (i > 0) {
    j = (i-1)>>1;
    if (heap[j].key < newgain) {
      heap[i] = heap[j];
      locator[heap[i].val] = i;
      i = j;
    }
    else 
      break;
  }

  heap[i].key = newgain;
  heap[i].val = node;
  locator[node] = i;

  ASSERTS(CheckHeapFloat(queue));

}


/*************************************************************************
* This function returns the vertex with the largest gain from a partition
* and removes the node from the bucket list
**************************************************************************/
int FPQueueGetMax(FPQueueType *queue)
{
  int vtx, i, j, node;
  float gain;
  idxtype *locator;
  FKeyValueType *heap;

  if (queue->nnodes == 0)
    return -1;

  queue->nnodes--;

  heap = queue->heap;
  locator = queue->locator;

  vtx = heap[0].val;
  locator[vtx] = -1;

  if ((i = queue->nnodes) > 0) {
    gain = heap[i].key;
    node = heap[i].val;
    i = 0;
    while ((j=2*i+1) < queue->nnodes) {
      if (heap[j].key > gain) {
        if (j+1 < queue->nnodes && heap[j+1].key > heap[j].key)
          j = j+1;
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else if (j+1 < queue->nnodes && heap[j+1].key > gain) {
        j = j+1;
        heap[i] = heap[j];
        locator[heap[i].val] = i;
        i = j;
      }
      else
        break;
    }

    heap[i].key = gain;
    heap[i].val = node;
    locator[node] = i;
  }

  ASSERTS(CheckHeapFloat(queue));
  return vtx;
}
      

/*************************************************************************
* This function returns the vertex with the largest gain from a partition
**************************************************************************/
int FPQueueSeeMaxVtx(FPQueueType *queue)
{
  int vtx;

  if (queue->nnodes == 0)
    return -1;

  vtx = queue->heap[0].val;

  return vtx;
}
      

/*************************************************************************
* This function returns the vertex with the largest gain from a partition
**************************************************************************/
float FPQueueSeeMaxGain(FPQueueType *queue)
{
  float gain;

  if (queue->nnodes == 0)
    return 0.0;

  gain = queue->heap[0].key;

  return gain;
}


/*************************************************************************
* This function returns the vertex with the largest gain from a partition
**************************************************************************/
float FPQueueGetKey(FPQueueType *queue)
{
  int key;

  if (queue->nnodes == 0)
    return -1;

  key = queue->heap[0].key;

  return key;
}
      
/*************************************************************************
* This function returns the number of nodes in the queue
**************************************************************************/
int FPQueueGetQSize(FPQueueType *queue)
{
  return queue->nnodes;
}






/*************************************************************************
* This functions checks the consistency of the heap
**************************************************************************/
int CheckHeapFloat(FPQueueType *queue)
{
  int i, j, nnodes;
  idxtype *locator;
  FKeyValueType *heap;

  heap = queue->heap;
  locator = queue->locator;
  nnodes = queue->nnodes;

  if (nnodes == 0)
    return 1;

  ASSERTS(locator[heap[0].val] == 0);
  for (i=1; i<nnodes; i++) {
    ASSERTS(locator[heap[i].val] == i);
    ASSERTS(heap[i].key <= heap[(i-1)/2].key);
  }
  for (i=1; i<nnodes; i++)
    ASSERTS(heap[i].key <= heap[0].key);

  for (j=i=0; i<queue->maxnodes; i++) {
    if (locator[i] != -1)
      j++;
  }
  ASSERTS(j == nnodes);

  return 1;
}
