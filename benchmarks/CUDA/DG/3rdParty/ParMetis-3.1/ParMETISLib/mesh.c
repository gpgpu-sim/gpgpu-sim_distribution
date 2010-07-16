/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * mesh.c
 *
 * This file contains routines for constructing the dual graph of a mesh.
 * Assumes that each processor has at least one mesh element.
 *
 * Started 10/19/94
 * George
 *
 * $Id: mesh.c,v 1.11 2003/07/25 04:01:04 karypis Exp $
 *
 */

#include <parmetislib.h>


/*************************************************************************
* This function converts a mesh into a dual graph
**************************************************************************/
void ParMETIS_V3_Mesh2Dual(idxtype *elmdist, idxtype *eptr, idxtype *eind, 
                 int *numflag, int *ncommonnodes, idxtype **xadj, 
		 idxtype **adjncy, MPI_Comm *comm)
{
  int i, j, jj, k, kk, m;
  int npes, mype, pe, count, mask, pass;
  int nelms, lnns, my_nns, node;
  int firstelm, firstnode, lnode, nrecv, nsend;
  int *scounts, *rcounts, *sdispl, *rdispl;
  idxtype *nodedist, *nmap, *auxarray;
  idxtype *gnptr, *gnind, *nptr, *nind, *myxadj, *myadjncy = NULL;
  idxtype *sbuffer, *rbuffer, *htable;
  KeyValueType *nodelist, *recvbuffer;
  idxtype ind[200], wgt[200];
  int gmaxnode, gminnode;
  CtrlType ctrl;


  SetUpCtrl(&ctrl, -1, 0, *comm);

  npes = ctrl.npes;
  mype = ctrl.mype;

  nelms = elmdist[mype+1]-elmdist[mype];

  if (*numflag == 1) 
    ChangeNumberingMesh2(elmdist, eptr, eind, NULL, NULL, NULL, npes, mype, 1);

  mask = (1<<11)-1;

  /*****************************/
  /* Determine number of nodes */
  /*****************************/
  gminnode = GlobalSEMin(&ctrl, eind[idxamin(eptr[nelms], eind)]);
  for (i=0; i<eptr[nelms]; i++)
    eind[i] -= gminnode;

  gmaxnode = GlobalSEMax(&ctrl, eind[idxamax(eptr[nelms], eind)]);


  /**************************/
  /* Check for input errors */
  /**************************/
  ASSERTS(nelms > 0);

  /* construct node distribution array */
  nodedist = idxsmalloc(npes+1, 0, "nodedist");
  for (nodedist[0]=0, i=0,j=gmaxnode+1; i<npes; i++) {
    k = j/(npes-i);
    nodedist[i+1] = nodedist[i]+k;
    j -= k;
  }
  my_nns = nodedist[mype+1]-nodedist[mype];
  firstnode = nodedist[mype];

  nodelist = (KeyValueType *)GKmalloc(eptr[nelms]*sizeof(KeyValueType), "nodelist");
  auxarray = idxmalloc(eptr[nelms], "auxarray");
  htable   = idxsmalloc(amax(my_nns, mask+1), -1, "htable");
  scounts  = imalloc(4*npes+2, "scounts");
  rcounts  = scounts+npes;
  sdispl   = scounts+2*npes;
  rdispl   = scounts+3*npes+1;


  /*********************************************/
  /* first find a local numbering of the nodes */
  /*********************************************/
  for (i=0; i<nelms; i++) {
    for (j=eptr[i]; j<eptr[i+1]; j++) {
      nodelist[j].key = eind[j];
      nodelist[j].val = j;
      auxarray[j]     = i; /* remember the local element ID that uses this node */
    }
  }
  ikeysort(eptr[nelms], nodelist);

  for (count=1, i=1; i<eptr[nelms]; i++) {
    if (nodelist[i].key > nodelist[i-1].key)
      count++;
  }

  lnns = count;
  nmap = idxmalloc(lnns, "nmap");

  /* renumber the nodes of the elements array */
  count = 1;
  nmap[0] = nodelist[0].key;
  eind[nodelist[0].val] = 0;
  nodelist[0].val = auxarray[nodelist[0].val];  /* Store the local element ID */
  for (i=1; i<eptr[nelms]; i++) {
    if (nodelist[i].key > nodelist[i-1].key) {
      nmap[count] = nodelist[i].key;
      count++;
    }
    eind[nodelist[i].val] = count-1;
    nodelist[i].val = auxarray[nodelist[i].val];  /* Store the local element ID */
  }
  MPI_Barrier(*comm);

  /**********************************************************/
  /* perform comms necessary to construct node-element list */
  /**********************************************************/
  iset(npes, 0, scounts);
  for (pe=i=0; i<eptr[nelms]; i++) {
    while (nodelist[i].key >= nodedist[pe+1])
      pe++;
    scounts[pe] += 2;
  }
  ASSERTS(pe < npes);

  MPI_Alltoall((void *)scounts, 1, MPI_INT, (void *)rcounts, 1, MPI_INT, *comm);

  icopy(npes, scounts, sdispl);
  MAKECSR(i, npes, sdispl);

  icopy(npes, rcounts, rdispl);
  MAKECSR(i, npes, rdispl);

  ASSERTS(sdispl[npes] == eptr[nelms]*2);

  nrecv = rdispl[npes]/2;
  recvbuffer = (KeyValueType *)GKmalloc(amax(1, nrecv)*sizeof(KeyValueType), "recvbuffer");

  MPI_Alltoallv((void *)nodelist, scounts, sdispl, IDX_DATATYPE, (void *)recvbuffer, 
                rcounts, rdispl, IDX_DATATYPE, *comm);

  /**************************************/
  /* construct global node-element list */
  /**************************************/
  gnptr = idxsmalloc(my_nns+1, 0, "gnptr");

  for (i=0; i<npes; i++) {
    for (j=rdispl[i]/2; j<rdispl[i+1]/2; j++) {
      lnode = recvbuffer[j].key-firstnode;
      ASSERTS(lnode >= 0 && lnode < my_nns)

      gnptr[lnode]++;
    }
  }
  MAKECSR(i, my_nns, gnptr);

  gnind = idxmalloc(amax(1, gnptr[my_nns]), "gnind");
  for (pe=0; pe<npes; pe++) {
    firstelm = elmdist[pe];
    for (j=rdispl[pe]/2; j<rdispl[pe+1]/2; j++) {
      lnode = recvbuffer[j].key-firstnode;
      gnind[gnptr[lnode]++] = recvbuffer[j].val+firstelm;
    }
  }
  SHIFTCSR(i, my_nns, gnptr);


  /*********************************************************/
  /* send the node-element info to the relevant processors */
  /*********************************************************/
  iset(npes, 0, scounts);

  /* use a hash table to ensure that each node is sent to a proc only once */
  for (pe=0; pe<npes; pe++) {
    for (j=rdispl[pe]/2; j<rdispl[pe+1]/2; j++) {
      lnode = recvbuffer[j].key-firstnode;
      if (htable[lnode] == -1) {
        scounts[pe] += gnptr[lnode+1]-gnptr[lnode];
        htable[lnode] = 1;
      }
    }

    /* now reset the hash table */
    for (j=rdispl[pe]/2; j<rdispl[pe+1]/2; j++) {
      lnode = recvbuffer[j].key-firstnode;
      htable[lnode] = -1;
    }
  }


  MPI_Alltoall((void *)scounts, 1, MPI_INT, (void *)rcounts, 1, MPI_INT, *comm);

  icopy(npes, scounts, sdispl);
  MAKECSR(i, npes, sdispl);

  /* create the send buffer */
  nsend = sdispl[npes];
  sbuffer = (idxtype *)realloc(nodelist, sizeof(idxtype)*amax(1, nsend));

  count = 0;
  for (pe=0; pe<npes; pe++) {
    for (j=rdispl[pe]/2; j<rdispl[pe+1]/2; j++) {
      lnode = recvbuffer[j].key-firstnode;
      if (htable[lnode] == -1) {
        for (k=gnptr[lnode]; k<gnptr[lnode+1]; k++) {
          if (k == gnptr[lnode])
            sbuffer[count++] = -1*(gnind[k]+1);
          else
            sbuffer[count++] = gnind[k];
        }
        htable[lnode] = 1;
      }
    }
    ASSERTS(count == sdispl[pe+1]);

    /* now reset the hash table */
    for (j=rdispl[pe]/2; j<rdispl[pe+1]/2; j++) {
      lnode = recvbuffer[j].key-firstnode;
      htable[lnode] = -1;
    }
  }

  icopy(npes, rcounts, rdispl);
  MAKECSR(i, npes, rdispl);

  nrecv = rdispl[npes];
  rbuffer = (idxtype *)realloc(recvbuffer, sizeof(idxtype)*amax(1, nrecv));

  MPI_Alltoallv((void *)sbuffer, scounts, sdispl, IDX_DATATYPE, (void *)rbuffer, 
                rcounts, rdispl, IDX_DATATYPE, *comm);

  k = -1;
  nptr = idxsmalloc(lnns+1, 0, "nptr");
  nind = rbuffer;
  for (pe=0; pe<npes; pe++) {
    for (j=rdispl[pe]; j<rdispl[pe+1]; j++) {
      if (nind[j] < 0) {
        k++;
        nind[j] = (-1*nind[j])-1;
      }
      nptr[k]++;
    }
  }
  MAKECSR(i, lnns, nptr);

  ASSERTS(k+1 == lnns);
  ASSERTS(nptr[lnns] == nrecv)

  myxadj = *xadj = idxsmalloc(nelms+1, 0, "xadj");
  idxset(mask+1, -1, htable);

  firstelm = elmdist[mype];

  /* Two passes -- in first pass, simply find out the memory requirements */
  for (pass=0; pass<2; pass++) {
    for (i=0; i<nelms; i++) {
      for (count=0, j=eptr[i]; j<eptr[i+1]; j++) {
        node = eind[j];

        for (k=nptr[node]; k<nptr[node+1]; k++) {
          if ((kk=nind[k]) == firstelm+i) 
	    continue;
	    
          m = htable[(kk&mask)];

          if (m == -1) {
            ind[count] = kk;
            wgt[count] = 1;
            htable[(kk&mask)] = count++;
          }
          else {
            if (ind[m] == kk) { 
              wgt[m]++;
            }
            else {
              for (jj=0; jj<count; jj++) {
                if (ind[jj] == kk) {
                  wgt[jj]++;
                  break;
	        }
              }
              if (jj == count) {
                ind[count]   = kk;
                wgt[count++] = 1;
              }
	    }
          }
        }
      }

      for (j=0; j<count; j++) {
        htable[(ind[j]&mask)] = -1;
        if (wgt[j] >= *ncommonnodes) {
          if (pass == 0) 
            myxadj[i]++;
          else 
            myadjncy[myxadj[i]++] = ind[j];
	}
      }
    }

    if (pass == 0) {
      MAKECSR(i, nelms, myxadj);
      myadjncy = *adjncy = idxmalloc(myxadj[nelms], "adjncy");
    }
    else {
      SHIFTCSR(i, nelms, myxadj);
    }
  }

  /*****************************************/
  /* correctly renumber the elements array */
  /*****************************************/
  for (i=0; i<eptr[nelms]; i++)
    eind[i] = nmap[eind[i]] + gminnode;

  if (*numflag == 1) 
    ChangeNumberingMesh2(elmdist, eptr, eind, myxadj, myadjncy, NULL, npes, mype, 0);

  /* do not free nodelist, recvbuffer, rbuffer */
  GKfree((void **)&scounts, (void **)&nodedist, (void **)&nmap, (void **)&sbuffer, 
         (void **)&htable, (void **)&nptr, (void **)&nind, (void **)&gnptr, 
	 (void **)&gnind, (void **)&auxarray, LTERM);

  FreeCtrl(&ctrl);

  return;
}

