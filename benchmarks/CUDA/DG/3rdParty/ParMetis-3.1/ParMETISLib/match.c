/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mmatch.c
 *
 * This file contains code that finds a matching
 *
 * Started 2/22/96
 * George
 *
 * $Id: match.c,v 1.2 2003/07/21 17:18:50 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function finds a matching
**************************************************************************/
void Moc_GlobalMatch_Balance(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int h, i, ii, j, k;
  int nnbrs, nvtxs, ncon, cnvtxs, firstvtx, lastvtx, maxi, maxidx, nkept;
  int otherlastvtx, nrequests, nchanged, pass, nmatched, wside;
  idxtype *xadj, *ladjncy, *adjwgt, *vtxdist, *home, *myhome, *shome, *rhome;
  idxtype *match, *rmatch, *smatch;
  idxtype *peind, *sendptr, *recvptr;
  idxtype *perm, *iperm, *nperm, *changed;
  float *nvwgt, maxnvwgt;
  int *nreqs_pe;
  KeyValueType *match_requests, *match_granted, *pe_requests;

  maxnvwgt = 1.0/((float)(ctrl->nparts)*MAXNVWGT_FACTOR);

  graph->match_type = MATCH_GLOBAL;

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->MatchTmr));

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  ladjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  home = graph->home;
  nvwgt = graph->nvwgt;

  vtxdist = graph->vtxdist;
  firstvtx = vtxdist[ctrl->mype];
  lastvtx = vtxdist[ctrl->mype+1];

  match = graph->match = idxsmalloc(nvtxs+graph->nrecv, UNMATCHED, "HEM_Match: match");
  myhome = idxsmalloc(nvtxs+graph->nrecv, UNMATCHED, "HEM_Match: myhome");

  /*------------------------------------------------------------
  / Send/Receive the home information of interface vertices
  /------------------------------------------------------------*/
  if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION) {
    idxcopy(nvtxs, home, myhome);
    shome = wspace->indices;
    rhome = myhome + nvtxs;
    CommInterfaceData(ctrl, graph, myhome, shome, rhome);
  }

  nnbrs = graph->nnbrs;
  peind = graph->peind;
  sendptr = graph->sendptr;
  recvptr = graph->recvptr;

  /* Use wspace->indices as the tmp space for matching info of the boundary
   * vertices that are sent and received */
  rmatch = match + nvtxs;
  smatch = wspace->indices;
  changed = smatch+graph->nsend;

  /* Use wspace->indices as the tmp space for match requests of the boundary
   * vertices that are sent and received */
  match_requests = wspace->pairs;
  match_granted = match_requests + graph->nsend;

  nreqs_pe = ismalloc(nnbrs, 0, "Match_HEM: nreqs_pe");

  nkept = graph->gnvtxs/ctrl->npes - nvtxs;

  perm = (idxtype *)wspace->degrees;
  iperm = perm + nvtxs;
  FastRandomPermute(nvtxs, perm, 1);
  for (i=0; i<nvtxs; i++)
    iperm[perm[i]] = i;

  nperm = iperm + nvtxs;
  for (i=0; i<nnbrs; i++)
    nperm[i] = i;

  /*************************************************************
   * Go now and find a matching by doing multiple iterations
   *************************************************************/
  /* First nullify the heavy vertices */
  for (nchanged=i=0; i<nvtxs; i++) {
    for (h=0; h<ncon; h++)
      if (nvwgt[i*ncon+h] > maxnvwgt) {
        break;
      }

    if (h != ncon) {
      match[i] = TOO_HEAVY;
      nchanged++;
    }
  }
  if (GlobalSESum(ctrl, nchanged) > 0) {
    IFSET(ctrl->dbglvl, DBG_PROGRESS,
    rprintf(ctrl, "We found %d heavy vertices!\n", GlobalSESum(ctrl, nchanged)));
    CommInterfaceData(ctrl, graph, match, smatch, rmatch);
  }


  for (nmatched=pass=0; pass<NMATCH_PASSES; pass++) {
    wside = (graph->level+pass)%2;
    nchanged = nrequests = 0;
    for (ii=nmatched; ii<nvtxs; ii++) {
      i = perm[ii];
      if (match[i] == UNMATCHED) {  /* Unmatched */
        maxidx = i;
        maxi = -1;

        /* Find a heavy-edge matching */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          k = ladjncy[j];
          if (match[k] == UNMATCHED &&
               myhome[k] == myhome[i] &&
               (maxi == -1 ||
               adjwgt[maxi] < adjwgt[j] ||
               (maxidx < nvtxs &&
               k < nvtxs &&
               adjwgt[maxi] == adjwgt[j] &&
               BetterVBalance(ncon,nvwgt+i*ncon,nvwgt+maxidx*ncon,nvwgt+k*ncon) >= 0))) {
            maxi = j;
            maxidx = k;
          }
        }

        if (maxi != -1) {
          k = ladjncy[maxi];
          if (k < nvtxs) { /* Take care the local vertices first */
            /* Here we give preference the local matching by granting it right away */
            if (i <= k) {
              match[i] = firstvtx+k + KEEP_BIT;
              match[k] = firstvtx+i;
            }
            else {
              match[i] = firstvtx+k;
              match[k] = firstvtx+i + KEEP_BIT;
            }
            changed[nchanged++] = i;
            changed[nchanged++] = k;
          }
          else { /* Take care any remote boundary vertices */
            match[k] = MAYBE_MATCHED;
            /* Alternate among which vertices will issue the requests */
            if ((wside ==0 && firstvtx+i < graph->imap[k]) || (wside == 1 && firstvtx+i > graph->imap[k])) { 
              match[i] = MAYBE_MATCHED;
              match_requests[nrequests].key = graph->imap[k];
              match_requests[nrequests].val = firstvtx+i;
              nrequests++;
            }
          }
        }
      }
    }


#ifdef DEBUG_MATCH
    PrintVector2(ctrl, nvtxs, firstvtx, match, "Match1");
    myprintf(ctrl, "[c: %2d] Nlocal: %d, Nrequests: %d\n", c, nlocal, nrequests);
#endif


    /***********************************************************
    * Exchange the match_requests, requests for me are stored in
    * match_granted 
    ************************************************************/
    /* Issue the receives first. Note that from each PE can receive a maximum
       of the interface node that it needs to send it in the case of a mat-vec */
    for (i=0; i<nnbrs; i++) {
      MPI_Irecv((void *)(match_granted+recvptr[i]), 2*(recvptr[i+1]-recvptr[i]), IDX_DATATYPE,
                peind[i], 1, ctrl->comm, ctrl->rreq+i);
    }

    /* Issue the sends next. This needs some work */
    ikeysort(nrequests, match_requests);
    for (j=i=0; i<nnbrs; i++) {
      otherlastvtx = vtxdist[peind[i]+1];
      for (k=j; k<nrequests && match_requests[k].key < otherlastvtx; k++);
      MPI_Isend((void *)(match_requests+j), 2*(k-j), IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
      j = k;
    }

    /* OK, now get into the loop waiting for the operations to finish */
    MPI_Waitall(nnbrs, ctrl->rreq, ctrl->statuses);
    for (i=0; i<nnbrs; i++) {
      MPI_Get_count(ctrl->statuses+i, IDX_DATATYPE, nreqs_pe+i);
      nreqs_pe[i] = nreqs_pe[i]/2;  /* Adjust for pairs of IDX_DATATYPE */
    }
    MPI_Waitall(nnbrs, ctrl->sreq, ctrl->statuses);


    /***********************************************************
    * Now, go and service the requests that you received in 
    * match_granted 
    ************************************************************/
    RandomPermute(nnbrs, nperm, 0);
    for (ii=0; ii<nnbrs; ii++) {
      i = nperm[ii];
      pe_requests = match_granted+recvptr[i];
      for (j=0; j<nreqs_pe[i]; j++) {
        k = pe_requests[j].key;
        ASSERTP(ctrl, k >= firstvtx && k < lastvtx, (ctrl, "%d %d %d %d %d\n", firstvtx, lastvtx, k, j, peind[i]));
        /* myprintf(ctrl, "Requesting a match %d %d\n", pe_requests[j].key, pe_requests[j].val); */
        if (match[k-firstvtx] == UNMATCHED) { /* Bingo, lets grant this request */
          changed[nchanged++] = k-firstvtx;
          if (nkept >= 0) { /* Flip a coin for who gets it */
            match[k-firstvtx] = pe_requests[j].val + KEEP_BIT;
            nkept--;
          }
          else {
            match[k-firstvtx] = pe_requests[j].val;
            pe_requests[j].key += KEEP_BIT;
            nkept++;
          }
          /* myprintf(ctrl, "Request from pe:%d (%d %d) granted!\n", peind[i], pe_requests[j].val, pe_requests[j].key); */ 
        }
        else { /* We are not granting the request */
          /* myprintf(ctrl, "Request from pe:%d (%d %d) not granted!\n", peind[i], pe_requests[j].val, pe_requests[j].key); */ 
          pe_requests[j].key = UNMATCHED;
        }
      }
    }


    /***********************************************************
    * Exchange the match_granted information. It is stored in
    * match_requests 
    ************************************************************/
    /* Issue the receives first. Note that from each PE can receive a maximum
       of the interface node that it needs to send during the case of a mat-vec */
    for (i=0; i<nnbrs; i++) {
      MPI_Irecv((void *)(match_requests+sendptr[i]), 2*(sendptr[i+1]-sendptr[i]), IDX_DATATYPE,
                peind[i], 1, ctrl->comm, ctrl->rreq+i);
    }

    /* Issue the sends next. */
    for (i=0; i<nnbrs; i++) {
      MPI_Isend((void *)(match_granted+recvptr[i]), 2*nreqs_pe[i], IDX_DATATYPE, 
                peind[i], 1, ctrl->comm, ctrl->sreq+i);
    }

    /* OK, now get into the loop waiting for the operations to finish */
    MPI_Waitall(nnbrs, ctrl->rreq, ctrl->statuses);
    for (i=0; i<nnbrs; i++) {
      MPI_Get_count(ctrl->statuses+i, IDX_DATATYPE, nreqs_pe+i);
      nreqs_pe[i] = nreqs_pe[i]/2;  /* Adjust for pairs of IDX_DATATYPE */
    }
    MPI_Waitall(nnbrs, ctrl->sreq, ctrl->statuses);


    /***********************************************************
    * Now, go and through the match_requests and update local
    * match information for the matchings that were granted.
    ************************************************************/
    for (i=0; i<nnbrs; i++) {
      pe_requests = match_requests+sendptr[i];
      for (j=0; j<nreqs_pe[i]; j++) {
        match[pe_requests[j].val-firstvtx] = pe_requests[j].key;
        if (pe_requests[j].key != UNMATCHED)
          changed[nchanged++] = pe_requests[j].val-firstvtx;
      }
    }

    for (i=0; i<nchanged; i++) {
      ii = iperm[changed[i]];
      perm[ii] = perm[nmatched];
      iperm[perm[nmatched]] = ii;
      nmatched++;
    }

    CommChangedInterfaceData(ctrl, graph, nchanged, changed, match, match_requests, match_granted, wspace->pv4);
  }

  /* Traverse the vertices and those that were unmatched, match them with themselves */
  cnvtxs = 0;
  for (i=0; i<nvtxs; i++) {
    if (match[i] == UNMATCHED || match[i] == TOO_HEAVY) {
      match[i] = (firstvtx+i) + KEEP_BIT;
      cnvtxs++;
    }
    else if (match[i] >= KEEP_BIT) {  /* A matched vertex which I get to keep */
      cnvtxs++;
    }
  }

  if (ctrl->dbglvl&DBG_MATCHINFO) {
    PrintVector2(ctrl, nvtxs, firstvtx, match, "Match");
    myprintf(ctrl, "Cnvtxs: %d\n", cnvtxs);
    rprintf(ctrl, "Done with matching...\n");
  }

  GKfree((void **)(&myhome), (void **)(&nreqs_pe), LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->MatchTmr));
  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->ContractTmr));

  Moc_Global_CreateCoarseGraph(ctrl, graph, wspace, cnvtxs);

  IFSET(ctrl->dbglvl, DBG_TIME, MPI_Barrier(ctrl->comm));
  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->ContractTmr));

}

