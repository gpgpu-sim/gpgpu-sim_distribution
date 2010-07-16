/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mcoarsen.c
 *
 * This file contains code that performs graph coarsening
 *
 * Started 2/22/96
 * George
 *
 * $Id: coarsen.c,v 1.2 2003/07/21 17:18:48 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function creates the coarser graph
**************************************************************************/
void Moc_Global_CreateCoarseGraph(CtrlType *ctrl, GraphType *graph,
	WorkSpaceType *wspace, int cnvtxs)
{
  int h, i, j, k, l, ii, jj, ll, nnbrs, nvtxs, nedges, ncon;
  int firstvtx, lastvtx, cfirstvtx, clastvtx, otherlastvtx;
  int npes=ctrl->npes, mype=ctrl->mype;
  int cnedges, nsend, nrecv, nkeepsize, nrecvsize, nsendsize, v, u;
  idxtype *xadj, *ladjncy, *adjwgt, *vwgt, *vsize, *vtxdist, *home;
  idxtype *match, *cmap, *rcmap, *scmap;
  idxtype *cxadj, *cadjncy, *cadjwgt, *cvwgt, *cvsize = NULL, *chome = NULL, *cvtxdist;
  idxtype *rsizes, *ssizes, *rlens, *slens, *rgraph, *sgraph, *perm;
  idxtype *peind, *recvptr, *recvind;
  float *nvwgt, *cnvwgt;
  GraphType *cgraph;
  KeyValueType *scand, *rcand;
  int mask=(1<<13)-1, htable[8192], htableidx[8192];

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;

  vtxdist = graph->vtxdist;
  xadj = graph->xadj;
  vwgt = graph->vwgt;
  vsize = graph->vsize;
  nvwgt = graph->nvwgt;
  home = graph->home;
  ladjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  match = graph->match;

  firstvtx = vtxdist[mype];
  lastvtx = vtxdist[mype+1];

  cmap = graph->cmap = idxmalloc(nvtxs+graph->nrecv, "CreateCoarseGraph: cmap");

  nnbrs = graph->nnbrs;
  peind = graph->peind;
  recvind = graph->recvind;
  recvptr = graph->recvptr;

  /* Use wspace->indices as the tmp space for map of the boundary
   * vertices that are sent and received */
  scmap = wspace->indices;
  rcmap = cmap + nvtxs;


  /* Initialize the coarser graph */
  cgraph = CreateGraph();
  cgraph->nvtxs = cnvtxs;
  cgraph->ncon = ncon;
  cgraph->level = graph->level+1;
  cgraph->finer = graph;
  graph->coarser = cgraph;



  /*************************************************************
  * Obtain the vtxdist of the coarser graph 
  **************************************************************/
  cvtxdist = cgraph->vtxdist = idxmalloc(npes+1, "CreateCoarseGraph: cvtxdist");
  cvtxdist[npes] = cnvtxs;  /* Use last position in the cvtxdist as a temp buffer */

  MPI_Allgather((void *)(cvtxdist+npes), 1, IDX_DATATYPE, (void *)cvtxdist, 1, IDX_DATATYPE, ctrl->comm);

  MAKECSR(i, npes, cvtxdist);

  cgraph->gnvtxs = cvtxdist[npes];

#ifdef DEBUG_CONTRACT
  PrintVector(ctrl, npes+1, 0, cvtxdist, "cvtxdist");
#endif


  /*************************************************************
  * Construct the cmap vector 
  **************************************************************/
  cfirstvtx = cvtxdist[mype];
  clastvtx = cvtxdist[mype+1];

  /* Create the cmap of what you know so far locally */
  cnvtxs = 0;
  for (i=0; i<nvtxs; i++) {
    if (match[i] >= KEEP_BIT) {
      k = match[i] - KEEP_BIT;
      if (k>=firstvtx && k<firstvtx+i)
        continue;  /* Both (i,k) are local and i has been matched via the (k,i) side */

      cmap[i] = cfirstvtx + cnvtxs++;
      if (k != firstvtx+i && (k>=firstvtx && k<lastvtx)) { /* I'm matched locally */
        cmap[k-firstvtx] = cmap[i];
        match[k-firstvtx] += KEEP_BIT;  /* Add the KEEP_BIT to simplify coding */
      }
    }
  }
  ASSERT(ctrl, cnvtxs == clastvtx-cfirstvtx);

  CommInterfaceData(ctrl, graph, cmap, scmap, rcmap);

  /* Update the cmap of the locally stored vertices that will go away. 
   * The remote processor assigned cmap for them */
  for (i=0; i<nvtxs; i++) {
    if (match[i] < KEEP_BIT) { /* Only vertices that go away satisfy this*/
      cmap[i] = rcmap[BSearch(graph->nrecv, recvind, match[i])];
    }
  }

  CommInterfaceData(ctrl, graph, cmap, scmap, rcmap);


#ifdef DEBUG_CONTRACT
  PrintVector(ctrl, nvtxs, firstvtx, cmap, "Cmap");
#endif


  /*************************************************************
  * Determine how many adjcency lists you need to send/receive.
  **************************************************************/
  /* Use wspace->pairs as the tmp space for the boundary vertices that are sent and received */
  scand = wspace->pairs;
  rcand = graph->rcand = (KeyValueType *)GKmalloc(recvptr[nnbrs]*sizeof(KeyValueType), "CreateCoarseGraph: rcand");

  nkeepsize = nsend = nrecv = 0;
  for (i=0; i<nvtxs; i++) {
    if (match[i] < KEEP_BIT) { /* This is going away */
      scand[nsend].key = match[i];
      scand[nsend].val = i;
      nsend++;
    }
    else {
      nkeepsize += (xadj[i+1]-xadj[i]);

      k = match[i]-KEEP_BIT;
      if (k<firstvtx || k>=lastvtx) { /* This is comming from afar */
        rcand[nrecv].key = k;
        rcand[nrecv].val = cmap[i] - cfirstvtx;  /* Set it for use during the partition projection */
        ASSERT(ctrl, rcand[nrecv].val>=0 && rcand[nrecv].val<cnvtxs);
        nrecv++;
      }
    }
  }


#ifdef DEBUG_CONTRACT
  PrintPairs(ctrl, nsend, scand, "scand");
  PrintPairs(ctrl, nrecv, rcand, "rcand");
#endif

  /***************************************************************
  * Determine how many lists and their sizes  you will send and 
  * received for each of the neighboring PEs
  ****************************************************************/
  rsizes = wspace->pv1;
  ssizes = wspace->pv2;
  idxset(nnbrs, 0, ssizes);
  idxset(nnbrs, 0, rsizes);
  rlens = graph->rlens = idxmalloc(nnbrs+1, "CreateCoarseGraph: graph->rlens");
  slens = graph->slens = idxmalloc(nnbrs+1, "CreateCoarseGraph: graph->slens");

  /* Take care the sending data first */
  ikeyvalsort(nsend, scand);
  slens[0] = 0;
  for (k=i=0; i<nnbrs; i++) {
    otherlastvtx = vtxdist[peind[i]+1];
    for (; k<nsend && scand[k].key < otherlastvtx; k++)
      ssizes[i] += (xadj[scand[k].val+1]-xadj[scand[k].val]);
    slens[i+1] = k;
  }

  /* Take care the receiving data next. You cannot yet determine the rsizes[] */
  ikeyvalsort(nrecv, rcand);
  rlens[0] = 0;
  for (k=i=0; i<nnbrs; i++) {
    otherlastvtx = vtxdist[peind[i]+1];
    for (; k<nrecv && rcand[k].key < otherlastvtx; k++);
    rlens[i+1] = k;
  }

#ifdef DEBUG_CONTRACT
  PrintVector(ctrl, nnbrs+1, 0, slens, "slens");
  PrintVector(ctrl, nnbrs+1, 0, rlens, "rlens");
#endif

  /***************************************************************
  * Exchange size information
  ****************************************************************/
  /* Issue the receives first. */
  for (i=0; i<nnbrs; i++) {
    if (rlens[i+1]-rlens[i] > 0)  /* Issue a receive only if you are getting something */
      MPI_Irecv((void *)(rsizes+i), 1, IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->rreq+i);
  }

  /* Take care the sending data next */
  for (i=0; i<nnbrs; i++) {
    if (slens[i+1]-slens[i] > 0)  /* Issue a send only if you are sending something */
      MPI_Isend((void *)(ssizes+i), 1, IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
  }

  /* OK, now get into the loop waiting for the operations to finish */
  for (i=0; i<nnbrs; i++) {
    if (rlens[i+1]-rlens[i] > 0)  
      MPI_Wait(ctrl->rreq+i, &ctrl->status);
  }
  for (i=0; i<nnbrs; i++) {
    if (slens[i+1]-slens[i] > 0)  
      MPI_Wait(ctrl->sreq+i, &ctrl->status);
  }


#ifdef DEBUG_CONTRACT
  PrintVector(ctrl, nnbrs, 0, rsizes, "rsizes");
  PrintVector(ctrl, nnbrs, 0, ssizes, "ssizes");
#endif

  /*************************************************************
  * Allocate memory for received/sent graphs and start sending 
  * and receiving data.
  * rgraph and sgraph is a different data structure than CSR
  * to facilitate single message exchange.
  **************************************************************/
  nrecvsize = idxsum(nnbrs, rsizes);
  nsendsize = idxsum(nnbrs, ssizes);
  if ((4+ncon)*(nrecv+nsend) + 2*(nrecvsize+nsendsize) <= wspace->nlarge) {  
    rgraph = (idxtype *)wspace->degrees;
    sgraph = rgraph + (4+ncon)*nrecv+2*nrecvsize;
  }
  else {
    rgraph = idxmalloc((4+ncon)*nrecv+2*nrecvsize, "CreateCoarseGraph: rgraph");
    sgraph = idxmalloc((4+ncon)*nsend+2*nsendsize, "CreateCoarseGraph: sgraph");
  }

  /* Deal with the received portion first */
  for (l=i=0; i<nnbrs; i++) {
    /* Issue a receive only if you are getting something */
    if (rlens[i+1]-rlens[i] > 0) {
      MPI_Irecv((void *)(rgraph+l), (4+ncon)*(rlens[i+1]-rlens[i])+2*rsizes[i], IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->rreq+i);
      l += (4+ncon)*(rlens[i+1]-rlens[i])+2*rsizes[i];
    }
  }


  /* Deal with the sent portion now */
  for (ll=l=i=0; i<nnbrs; i++) {
    if (slens[i+1]-slens[i] > 0) {  /* Issue a send only if you are sending something */
      for (k=slens[i]; k<slens[i+1]; k++) {
        ii = scand[k].val;
        sgraph[ll++] = firstvtx+ii;
        sgraph[ll++] = xadj[ii+1]-xadj[ii];
        for (h=0; h<ncon; h++)
          sgraph[ll++] = vwgt[ii*ncon+h];
        sgraph[ll++] = (ctrl->partType == STATIC_PARTITION) ? -1 : vsize[ii];
        sgraph[ll++] = (ctrl->partType == STATIC_PARTITION) ? -1 : home[ii];
        for (jj=xadj[ii]; jj<xadj[ii+1]; jj++) {
          sgraph[ll++] = cmap[ladjncy[jj]];
          sgraph[ll++] = adjwgt[jj];
        }
      }

      ASSERT(ctrl, ll-l == (4+ncon)*(slens[i+1]-slens[i])+2*ssizes[i]);

      /* myprintf(ctrl, "Sending to pe:%d, %d lists of size %d\n", peind[i], slens[i+1]-slens[i], ssizes[i]); */
      MPI_Isend((void *)(sgraph+l), ll-l, IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
      l = ll;
    }
  }

  /* OK, now get into the loop waiting for the operations to finish */
  for (i=0; i<nnbrs; i++) {
    if (rlens[i+1]-rlens[i] > 0)  
      MPI_Wait(ctrl->rreq+i, &ctrl->status);
  }
  for (i=0; i<nnbrs; i++) {
    if (slens[i+1]-slens[i] > 0)  
      MPI_Wait(ctrl->sreq+i, &ctrl->status);
  }


#ifdef DEBUG_CONTRACT
  rprintf(ctrl, "Graphs were sent!\n");
  PrintTransferedGraphs(ctrl, nnbrs, peind, slens, rlens, sgraph, rgraph);
#endif

  /*************************************************************
  * Setup the mapping from indices returned by BSearch to 
  * those that are actually stored 
  **************************************************************/
  perm = idxsmalloc(recvptr[nnbrs], -1, "CreateCoarseGraph: perm");
  for (j=i=0; i<nrecv; i++) {
  /*   myprintf(ctrl, "For received vertex %d, set perm[%d]=%d\n", rgraph[j], BSearch(graph->nrecv, recvind, rgraph[j]), j+ncon); */
    perm[BSearch(graph->nrecv, recvind, rgraph[j])] = j+1;
    j += (4+ncon)+2*rgraph[j+1];
  }

  /*************************************************************
  * Finally, create the coarser graph
  **************************************************************/
  /* Allocate memory for the coarser graph, and fire up coarsening */
  cxadj = cgraph->xadj = idxmalloc(cnvtxs+1, "CreateCoarserGraph: cxadj");
  cvwgt = cgraph->vwgt = idxmalloc(cnvtxs*ncon, "CreateCoarserGraph: cvwgt");
  if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION) {
    cvsize = cgraph->vsize = idxmalloc(cnvtxs, "CreateCoarserGraph: cvsize");
    chome = cgraph->home = idxmalloc(cnvtxs, "CreateCoarserGraph: chome");
  }
  cnvwgt = cgraph->nvwgt = fmalloc(cnvtxs*ncon, "CreateCoarserGraph: cnvwgt");
  cadjncy = idxmalloc(2*(nkeepsize+nrecvsize), "CreateCoarserGraph: cadjncy");
  cadjwgt = cadjncy + nkeepsize+nrecvsize;

  iset(8192, -1, htable);

  cxadj[0] = cnvtxs = cnedges = 0;
  for (i=0; i<nvtxs; i++) {
    if (match[i] >= KEEP_BIT) {
      v = firstvtx+i; 
      u = match[i]-KEEP_BIT;

      if (u>=firstvtx && u<lastvtx && v > u) 
        continue;  /* I have already collapsed it as (u,v) */

      /* Collapse the v vertex first, which you know is local */
      for (h=0; h<ncon; h++)
        cvwgt[cnvtxs*ncon+h] = vwgt[i*ncon+h];
      if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION) {
        cvsize[cnvtxs] = vsize[i];
        chome[cnvtxs] = home[i];
      }
      nedges = 0;

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = cmap[ladjncy[j]];
        if (k != cfirstvtx+cnvtxs) {  /* If this is not an internal edge */
          l = k&mask;
          if (htable[l] == -1) { /* Seeing this for first time */
            htable[l] = k;
            htableidx[l] = cnedges+nedges;
            cadjncy[cnedges+nedges] = k; 
            cadjwgt[cnedges+nedges++] = adjwgt[j];
          }
          else if (htable[l] == k) {
            cadjwgt[htableidx[l]] += adjwgt[j];
          }
          else { /* Now you have to go and do a search. Expensive case */
            for (l=0; l<nedges; l++) {
              if (cadjncy[cnedges+l] == k)
                break;
            }
            if (l < nedges) {
              cadjwgt[cnedges+l] += adjwgt[j];
            }
            else {
              cadjncy[cnedges+nedges] = k; 
              cadjwgt[cnedges+nedges++] = adjwgt[j];
            }
          }
        }
      }

      /* Collapse the u vertex next */
      if (v != u) { 
        if (u>=firstvtx && u<lastvtx) { /* Local vertex */
          u -= firstvtx;
          for (h=0; h<ncon; h++)
            cvwgt[cnvtxs*ncon+h] += vwgt[u*ncon+h];
          if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION) {
            cvsize[cnvtxs] += vsize[u];
            /* chome[cnvtxs] = home[u]; */
          }

          for (j=xadj[u]; j<xadj[u+1]; j++) {
            k = cmap[ladjncy[j]];
            if (k != cfirstvtx+cnvtxs) {  /* If this is not an internal edge */
              l = k&mask;
              if (htable[l] == -1) { /* Seeing this for first time */
                htable[l] = k;
                htableidx[l] = cnedges+nedges;
                cadjncy[cnedges+nedges] = k; 
                cadjwgt[cnedges+nedges++] = adjwgt[j];
              }
              else if (htable[l] == k) {
                cadjwgt[htableidx[l]] += adjwgt[j];
              }
              else { /* Now you have to go and do a search. Expensive case */
                for (l=0; l<nedges; l++) {
                  if (cadjncy[cnedges+l] == k)
                    break;
                }
                if (l < nedges) {
                  cadjwgt[cnedges+l] += adjwgt[j];
                }
                else {
                  cadjncy[cnedges+nedges] = k; 
                  cadjwgt[cnedges+nedges++] = adjwgt[j];
                }
              }
            }
          }
        }
        else { /* Remote vertex */
          u = perm[BSearch(graph->nrecv, recvind, u)];
          for (h=0; h<ncon; h++)
            /* Remember that the +1 stores the vertex weight */
            cvwgt[cnvtxs*ncon+h] += rgraph[(u+1)+h];
            if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION) {
              cvsize[cnvtxs] += rgraph[u+1+ncon];
              chome[cnvtxs] = rgraph[u+2+ncon];
            }
          for (j=0; j<rgraph[u]; j++) {
            k = rgraph[u+3+ncon+2*j];
            if (k != cfirstvtx+cnvtxs) {  /* If this is not an internal edge */
              l = k&mask;
              if (htable[l] == -1) { /* Seeing this for first time */
                htable[l] = k;
                htableidx[l] = cnedges+nedges;
                cadjncy[cnedges+nedges] = k; 
                cadjwgt[cnedges+nedges++] = rgraph[u+3+ncon+2*j+1];
              }
              else if (htable[l] == k) {
                cadjwgt[htableidx[l]] += rgraph[u+3+ncon+2*j+1];
              }
              else { /* Now you have to go and do a search. Expensive case */
                for (l=0; l<nedges; l++) {
                  if (cadjncy[cnedges+l] == k)
                    break;
                }
                if (l < nedges) {
                  cadjwgt[cnedges+l] += rgraph[u+3+ncon+2*j+1];
                }
                else {
                  cadjncy[cnedges+nedges] = k; 
                  cadjwgt[cnedges+nedges++] = rgraph[u+3+ncon+2*j+1];
                }
              }
            }
          }
        }
      }

      cnedges += nedges;
      for (j=cxadj[cnvtxs]; j<cnedges; j++)
        htable[cadjncy[j]&mask] = -1;  /* reset the htable */
      cxadj[++cnvtxs] = cnedges;
    }
  }

  cgraph->nedges = cnedges;

  /* ADD:  In order to keep from having to change this too much */
  /* ADD:  I kept vwgt array and recomputed nvwgt for each coarser graph */
  for (j=0; j<cnvtxs; j++)
    for (h=0; h<ncon; h++)
      cgraph->nvwgt[j*ncon+h] = (float)(cvwgt[j*ncon+h])/(float)(ctrl->tvwgts[h]);

  cgraph->adjncy = idxmalloc(cnedges, "CreateCoarserGraph: cadjncy");
  cgraph->adjwgt = idxmalloc(cnedges, "CreateCoarserGraph: cadjwgt");
  idxcopy(cnedges, cadjncy, cgraph->adjncy);
  idxcopy(cnedges, cadjwgt, cgraph->adjwgt);
  free(cadjncy);

  free(perm);

  if (rgraph != (idxtype *)wspace->degrees) 
    GKfree((void **)&rgraph, (void **)&sgraph, LTERM);

}


