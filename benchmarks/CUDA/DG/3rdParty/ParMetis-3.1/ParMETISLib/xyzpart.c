/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * xyzpart.c
 *
 * This file contains code that implements a coordinate based partitioning
 *
 * Started 7/11/97
 * George
 *
 * $Id: xyzpart.c,v 1.3 2003/07/30 18:37:59 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function implements a simple coordinate based partitioning
**************************************************************************/
void Coordinate_Partition(CtrlType *ctrl, GraphType *graph, int ndims, float *xyz, 
                          int setup, WorkSpaceType *wspace)
{
  int i, j, k, nvtxs, firstvtx, icoord, coords[3];
  idxtype *vtxdist;
  float max[3], min[3], gmin[3], gmax[3], shift[3], scale[3];
  KeyValueType *cand;

  if (setup)
    SetUp(ctrl, graph, wspace);
  else
    graph->nrecv = 0;

  nvtxs = graph->nvtxs;
  vtxdist = graph->vtxdist;

  firstvtx = vtxdist[ctrl->mype];

  cand = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "Coordinate_Partition: cand");

  /* Compute parameters for coordinate transformation */
  for (k=0; k<ndims; k++) {
    min[k] = +10000000;
    max[k] = -10000000;
  }
  for (i=0; i<nvtxs; i++) {
    for (k=0; k<ndims; k++) {
      if (xyz[i*ndims+k] < min[k])
        min[k] = xyz[i*ndims+k];
      if (xyz[i*ndims+k] > max[k])
        max[k] = xyz[i*ndims+k];
    }
  }

  /* Compute global min and max */
  MPI_Allreduce((void *)min, (void *)gmin, ndims, MPI_FLOAT, MPI_MIN, ctrl->comm);
  MPI_Allreduce((void *)max, (void *)gmax, ndims, MPI_FLOAT, MPI_MAX, ctrl->comm);

  /* myprintf(ctrl, "Coordinate Range: %e %e, Global %e %e\n", min[0], max[0], gmin[0], gmax[0]); */

  for (k=0; k<ndims; k++) {
    /* rprintf(ctrl, "Dim#%d: %e %e, span: %e\n", k, gmin[k], gmax[k], gmax[k]-gmin[k]); */
    shift[k] = -gmin[k];
    if (gmax[k] != gmin[k])
      scale[k] = 1.0/(gmax[k]-gmin[k]);
    else
      scale[k] = 1.0;
  }

  switch (ctrl->xyztype) {
    case XYZ_XCOORD:
      for (i=0; i<nvtxs; i++) {
        cand[i].key = 1000000*((xyz[i*ndims]+shift[0])*scale[0]);
        ASSERT(ctrl, cand[i].key>=0 && cand[i].key<=1000000);
        cand[i].val = firstvtx+i;
      }
      break;
    case XYZ_SPFILL:
      for (i=0; i<nvtxs; i++) {
        for (k=0; k<ndims; k++)
          coords[k] = 1024*((xyz[i*ndims+k]+shift[k])*scale[k]);
        for (icoord=0, j=9; j>=0; j--) {
          for (k=0; k<ndims; k++)
            icoord = (icoord<<1) + (coords[k]&(1<<j) ? 1 : 0);
        }
        cand[i].key = icoord;
        cand[i].val = firstvtx+i;
      }
      break;
    default:
      errexit("Unknown XYZ_Type type!\n");
  }


  /* Partition using sorting */
  PartSort(ctrl, graph, cand, wspace);

  free(cand);

}



/*************************************************************************
* This function sorts a distributed list of KeyValueType in increasing 
* order, and uses it to compute a partition. It uses samplesort. 
**************************************************************************/
void PartSort(CtrlType *ctrl, GraphType *graph, KeyValueType *elmnts, WorkSpaceType *wspace)
{
  int i, j, k, nvtxs, nrecv, npes=ctrl->npes, mype=ctrl->mype, firstvtx, lastvtx;
  idxtype *scounts, *rcounts, *vtxdist, *perm;
  KeyValueType *relmnts, *mypicks, *allpicks;

  nvtxs   = graph->nvtxs;
  vtxdist = graph->vtxdist;

  scounts = wspace->pv1;
  rcounts = wspace->pv2;

  /* Allocate memory for the splitters */
  mypicks  = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*(npes+1), "ParSort: mypicks");
  allpicks = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*npes*npes, "ParSort: allpicks");

  /* Sort the local elements */
  ikeysort(nvtxs, elmnts);

  /* Select the local npes-1 equally spaced elements */
  for (i=1; i<npes; i++) { 
    mypicks[i-1].key = elmnts[i*(nvtxs/npes)].key;
    mypicks[i-1].val = elmnts[i*(nvtxs/npes)].val;
  }

  /* PrintPairs(ctrl, npes-1, mypicks, "Mypicks"); */

  /* Gather the picks to all the processors */
  MPI_Allgather((void *)mypicks, 2*(npes-1), IDX_DATATYPE, (void *)allpicks, 2*(npes-1), IDX_DATATYPE, ctrl->comm);

  /* PrintPairs(ctrl, npes*(npes-1), allpicks, "Allpicks"); */

  /* Sort all the picks */
  ikeyvalsort(npes*(npes-1), allpicks);

  /* PrintPairs(ctrl, npes*(npes-1), allpicks, "Allpicks"); */

  /* Select the final splitters. Set the boundaries to simplify coding */
  for (i=1; i<npes; i++)
    mypicks[i] = allpicks[i*(npes-1)];
  mypicks[0].key    = MIN_INT;
  mypicks[npes].key = MAX_INT;

  /* PrintPairs(ctrl, npes+1, mypicks, "Mypicks"); */

  /* Compute the number of elements that belong to each bucket */
  idxset(npes, 0, scounts);
  for (j=i=0; i<nvtxs; i++) {
    if (elmnts[i].key < mypicks[j+1].key || (elmnts[i].key == mypicks[j+1].key && elmnts[i].val < mypicks[j+1].val))
      scounts[j]++;
    else
      scounts[++j]++;
  }
  MPI_Alltoall(scounts, 1, IDX_DATATYPE, rcounts, 1, IDX_DATATYPE, ctrl->comm);

/*
  PrintVector(ctrl, npes, 0, scounts, "Scounts");
  PrintVector(ctrl, npes, 0, rcounts, "Rcounts");
*/

  /* Allocate memory for sorted elements and receive them */
  MAKECSR(i, npes, scounts);
  MAKECSR(i, npes, rcounts);
  nrecv = rcounts[npes];
  if (wspace->nlarge >= nrecv)
    relmnts = (KeyValueType *)wspace->pairs;
  else
    relmnts = (KeyValueType *)GKmalloc(sizeof(KeyValueType)*nrecv, "ParSort: relmnts");

  /* Issue the receives first */
  for (i=0; i<npes; i++) 
    MPI_Irecv((void *)(relmnts+rcounts[i]), 2*(rcounts[i+1]-rcounts[i]), IDX_DATATYPE, i, 1, ctrl->comm, ctrl->rreq+i);

  /* Issue the sends next */
  for (i=0; i<npes; i++) 
    MPI_Isend((void *)(elmnts+scounts[i]), 2*(scounts[i+1]-scounts[i]), IDX_DATATYPE, i, 1, ctrl->comm, ctrl->sreq+i);

  MPI_Waitall(npes, ctrl->rreq, ctrl->statuses);
  MPI_Waitall(npes, ctrl->sreq, ctrl->statuses);


  /* OK, now do the local sort of the relmnts. Use perm to keep track original order */
  perm = idxmalloc(nrecv, "ParSort: perm");
  for (i=0; i<nrecv; i++) {
    perm[i] = relmnts[i].val;
    relmnts[i].val = i;
  }
  ikeysort(nrecv, relmnts);


  /* Compute what needs to be shifted */
  MPI_Scan((void *)(&nrecv), (void *)(&lastvtx), 1, MPI_INT, MPI_SUM, ctrl->comm);
  firstvtx = lastvtx-nrecv;  

  /*myprintf(ctrl, "first, last: %d %d\n", firstvtx, lastvtx); */

  for (j=0, i=0; i<npes; i++) {
    if (vtxdist[i+1] > firstvtx) {  /* Found the first PE that is passed me */
      if (vtxdist[i+1] >= lastvtx) {
        /* myprintf(ctrl, "Shifting %d elements to processor %d\n", lastvtx-firstvtx, i); */
        for (k=0; k<lastvtx-firstvtx; k++, j++) 
          relmnts[relmnts[j].val].key = i;
      }
      else {
        /* myprintf(ctrl, "Shifting %d elements to processor %d\n", vtxdist[i+1]-firstvtx, i); */
        for (k=0; k<vtxdist[i+1]-firstvtx; k++, j++) 
          relmnts[relmnts[j].val].key = i;

        firstvtx = vtxdist[i+1];
      }
    }
    if (vtxdist[i+1] >= lastvtx)
      break;
  }

  /* Reverse the ordering on the relmnts[].val */
  for (i=0; i<nrecv; i++) {
    ASSERTP(ctrl, relmnts[i].key>=0 && relmnts[i].key<npes, (ctrl, "%d %d\n", i, relmnts[i].key));
    relmnts[i].val = perm[i];
  }

  /* OK, now sent it back */
  /* Issue the receives first */
  for (i=0; i<npes; i++) 
    MPI_Irecv((void *)(elmnts+scounts[i]), 2*(scounts[i+1]-scounts[i]), IDX_DATATYPE, i, 1, ctrl->comm, ctrl->rreq+i);

  /* Issue the sends next */
  for (i=0; i<npes; i++) 
    MPI_Isend((void *)(relmnts+rcounts[i]), 2*(rcounts[i+1]-rcounts[i]), IDX_DATATYPE, i, 1, ctrl->comm, ctrl->sreq+i);

  MPI_Waitall(npes, ctrl->rreq, ctrl->statuses);
  MPI_Waitall(npes, ctrl->sreq, ctrl->statuses);


  /* Construct a partition for the graph */
  graph->where = idxmalloc(graph->nvtxs+graph->nrecv, "PartSort: graph->where");
  firstvtx = vtxdist[mype];
  for (i=0; i<nvtxs; i++) {
    ASSERTP(ctrl, elmnts[i].key>=0 && elmnts[i].key<npes, (ctrl, "%d %d\n", i, elmnts[i].key));
    ASSERTP(ctrl, elmnts[i].val>=vtxdist[mype] && elmnts[i].val<vtxdist[mype+1], (ctrl, "%d %d %d %d\n", i, vtxdist[mype], vtxdist[mype+1], elmnts[i].val));
    graph->where[elmnts[i].val-firstvtx] = elmnts[i].key;
  }


  GKfree((void **)&mypicks, (void **)&allpicks, (void **)&perm, LTERM);
  if (wspace->nlarge < nrecv)
    free(relmnts);

}

