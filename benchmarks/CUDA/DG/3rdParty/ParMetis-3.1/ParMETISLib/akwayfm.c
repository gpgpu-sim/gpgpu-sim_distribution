/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * makwayfm.c
 *
 * This file contains code that performs the k-way refinement
 *
 * Started 3/1/96
 * George
 *
 * $Id: akwayfm.c,v 1.3 2003/07/22 22:58:18 karypis Exp $
 */

#include <parmetislib.h>

#define ProperSide(c, from, other) \
              (((c) == 0 && (from)-(other) < 0) || ((c) == 1 && (from)-(other) > 0))


/*************************************************************************
* This function performs k-way refinement
**************************************************************************/
void Moc_KWayAdaptiveRefine(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace, int npasses)
{
  int h, i, ii, iii, j, k, c;
  int pass, nvtxs, nedges, ncon;
  int nmoves, nmoved;
  int me, firstvtx, lastvtx, yourlastvtx;
  int from, to = -1, oldto, oldcut, mydomain, yourdomain, imbalanced, overweight;
  int npes = ctrl->npes, mype = ctrl->mype, nparts = ctrl->nparts;
  int nlupd, nsupd, nnbrs, nchanged;
  idxtype *xadj, *ladjncy, *adjwgt, *vtxdist;
  idxtype *where, *tmp_where, *moved;
  float *lnpwgts, *gnpwgts, *ognpwgts, *pgnpwgts, *movewgts, *overfill;
  idxtype *update, *supdate, *rupdate, *pe_updates;
  idxtype *changed, *perm, *pperm, *htable;
  idxtype *peind, *recvptr, *sendptr;
  KeyValueType *swchanges, *rwchanges;
  RInfoType *rinfo, *myrinfo, *tmp_myrinfo, *tmp_rinfo;
  EdgeType *tmp_edegrees, *my_edegrees, *your_edegrees;
  float lbvec[MAXNCON], *nvwgt, *badmaxpwgt, *ubvec, *tpwgts, lbavg, ubavg;
  float oldgain, gain;
  float ipc_factor, redist_factor, vsize;
  int *nupds_pe, ndirty, nclean, dptr;
  int better, worse;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->KWayTmr));

  /*************************/
  /* set up common aliases */
  /*************************/
  nvtxs = graph->nvtxs;
  nedges = graph->nedges;
  ncon = graph->ncon;

  vtxdist = graph->vtxdist;
  xadj = graph->xadj;
  ladjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  firstvtx = vtxdist[mype];
  lastvtx = vtxdist[mype+1];

  where = graph->where;
  rinfo = graph->rinfo;
  lnpwgts = graph->lnpwgts;
  gnpwgts = graph->gnpwgts;
  ubvec = ctrl->ubvec;
  tpwgts = ctrl->tpwgts;
  ipc_factor = ctrl->ipc_factor;
  redist_factor = ctrl->redist_factor;

  nnbrs = graph->nnbrs;
  peind = graph->peind;
  recvptr = graph->recvptr;
  sendptr = graph->sendptr;

  changed = idxmalloc(nvtxs, "AKWR: changed");
  rwchanges = wspace->pairs;
  swchanges = rwchanges + recvptr[nnbrs];

  /************************************/
  /* set up important data structures */
  /************************************/
  perm = idxmalloc(nvtxs, "AKWR: perm");
  pperm = idxmalloc(nparts, "AKWR: pperm");

  update = idxmalloc(nvtxs, "AKWR: update");
  supdate = wspace->indices;
  rupdate = supdate + recvptr[nnbrs];
  nupds_pe = imalloc(npes, "AKWR: nupds_pe");
  htable = idxsmalloc(nvtxs+graph->nrecv, 0, "AKWR: lhtable");
  badmaxpwgt = fmalloc(nparts*ncon, "badmaxpwgt");

  for (i=0; i<nparts; i++) {
    for (h=0; h<ncon; h++) {
      badmaxpwgt[i*ncon+h] = ubvec[h]*tpwgts[i*ncon+h];
    }
  }

  movewgts = fmalloc(ncon*nparts, "AKWR: movewgts");
  ognpwgts = fmalloc(nparts*ncon, "AKWR: ognpwgts");
  pgnpwgts = fmalloc(nparts*ncon, "AKWR: pgnpwgts");
  overfill = fmalloc(nparts*ncon, "AKWR: overfill");
  moved = idxmalloc(nvtxs, "AKWR: moved");
  tmp_where = idxmalloc(nvtxs+graph->nrecv, "AKWR: tmp_where");
  tmp_rinfo = (RInfoType *)GKmalloc(sizeof(RInfoType)*nvtxs, "AKWR: tmp_rinfo");
  tmp_edegrees = (EdgeType *)GKmalloc(sizeof(EdgeType)*nedges, "AKWR: tmp_edegrees");

  idxcopy(nvtxs+graph->nrecv, where, tmp_where);
  for (i=0; i<nvtxs; i++) {
    tmp_rinfo[i].id = rinfo[i].id;
    tmp_rinfo[i].ed = rinfo[i].ed;
    tmp_rinfo[i].ndegrees = rinfo[i].ndegrees;
    tmp_rinfo[i].degrees = tmp_edegrees+xadj[i];

    for (j=0; j<rinfo[i].ndegrees; j++) {
      tmp_rinfo[i].degrees[j].edge = rinfo[i].degrees[j].edge;
      tmp_rinfo[i].degrees[j].ewgt = rinfo[i].degrees[j].ewgt;
    }
  }

  /*********************************************************/
  /* perform a small number of passes through the vertices */
  /*********************************************************/
  for (pass=0; pass<npasses; pass++) {
    oldcut = graph->mincut;
    if (mype == 0)
      RandomPermute(nparts, pperm, 1);
    MPI_Bcast((void *)pperm, nparts, IDX_DATATYPE, 0, ctrl->comm);
/*    FastRandomPermute(nvtxs, perm, 1); */

    /*****************************/
    /* move dirty vertices first */
    /*****************************/
    ndirty = 0;
    for (i=0; i<nvtxs; i++)
      if (where[i] != mype)
        ndirty++;

    dptr = 0;
    for (i=0; i<nvtxs; i++)
      if (where[i] != mype)
        perm[dptr++] = i;
      else
        perm[ndirty++] = i;

    ASSERT(ctrl, ndirty == nvtxs);
    ndirty = dptr;
    nclean = nvtxs-dptr;
    FastRandomPermute(ndirty, perm, 0);
    FastRandomPermute(nclean, perm+ndirty, 0);

    /* check to see if the partitioning is imbalanced */
    Moc_ComputeParallelBalance(ctrl, graph, graph->where, lbvec);
    ubavg = savg(ncon, ubvec);
    lbavg = savg(ncon, lbvec);
    imbalanced = (lbavg > ubavg) ? 1 : 0;

    for (c=0; c<2; c++) {
      scopy(ncon*nparts, gnpwgts, ognpwgts);
      sset(ncon*nparts, 0.0, movewgts);
      nmoved = 0;

      /**********************************************/
      /* PASS ONE -- record stats for desired moves */
      /**********************************************/
      for (iii=0; iii<nvtxs; iii++) {
        i     = perm[iii];
        from  = tmp_where[i];
        nvwgt = graph->nvwgt+i*ncon;
        vsize = (float)(graph->vsize[i]);

        for (h=0; h<ncon; h++) {
          if (fabs(nvwgt[h]-gnpwgts[from*ncon+h]) < SMALLFLOAT)
            break;
        }
        if (h < ncon) 
          continue;

        /* only check border vertices */
        if (tmp_rinfo[i].ed <= 0)
	  continue;

        my_edegrees = tmp_rinfo[i].degrees;

        for (k=0; k<tmp_rinfo[i].ndegrees; k++) {
          to = my_edegrees[k].edge;
          if (ProperSide(c, pperm[from], pperm[to])) {
            for (h=0; h<ncon; h++) {
              if (gnpwgts[to*ncon+h]+nvwgt[h] > badmaxpwgt[to*ncon+h] && nvwgt[h] > 0.0)
                break;
            }
            if (h == ncon)
              break;
          }
        }
        oldto = to;

        /* check if a subdomain was found that fits */
        if (k < tmp_rinfo[i].ndegrees) {
          /**************************/
          /**************************/
          switch (ctrl->ps_relation) {
            case COUPLED:
              better = (oldto == mype) ? 1 : 0;
              worse = (from == mype) ? 1 : 0;
              break;
            case DISCOUPLED:
            default:
              better = (oldto == graph->home[i]) ? 1 : 0;
              worse = (from == graph->home[i]) ? 1 : 0;
              break;
          }
          /**************************/
          /**************************/

          oldgain = ipc_factor * (float)(my_edegrees[k].ewgt-tmp_rinfo[i].id);
          if (better) oldgain += redist_factor * vsize;
          if (worse) oldgain -= redist_factor * vsize;

          for (j=k+1; j<tmp_rinfo[i].ndegrees; j++) {
            to = my_edegrees[j].edge;
            if (ProperSide(c, pperm[from], pperm[to])) {
              /**************************/
              /**************************/
              switch (ctrl->ps_relation) {
                case COUPLED:
                  better = (to == mype) ? 1 : 0;
                  break;
                case DISCOUPLED:
                default:
                  better = (to == graph->home[i]) ? 1 : 0;
                  break;
              }
              /**************************/
              /**************************/

              gain = ipc_factor * (float)(my_edegrees[j].ewgt-tmp_rinfo[i].id);
              if (better) gain += redist_factor * vsize;
              if (worse) gain -= redist_factor * vsize;

              for (h=0; h<ncon; h++)
                if (gnpwgts[to*ncon+h]+nvwgt[h] > badmaxpwgt[to*ncon+h] && nvwgt[h] > 0.0)
                  break;

              if (h == ncon) {
                if (gain > oldgain ||
                 (fabs(gain-oldgain) < SMALLFLOAT &&
                 IsHBalanceBetterTT(ncon,gnpwgts+oldto*ncon,gnpwgts+to*ncon,nvwgt,ubvec))){
                  oldgain = gain;
                  oldto = to;
                  k = j;
                }
              }
            }
          }
          to = oldto;
          gain = oldgain;

          if (gain > 0.0 ||  
          (gain > -1.0*SMALLFLOAT &&
          (imbalanced ||  graph->level > 3  || iii % 8 == 0) &&
          IsHBalanceBetterFT(ncon,gnpwgts+from*ncon,gnpwgts+to*ncon,nvwgt,ubvec))){

            /****************************************/
            /* Update tmp arrays of the moved vertex */
            /****************************************/
            tmp_where[i] = to;
            moved[nmoved++] = i;
            for (h=0; h<ncon; h++) {
	      INC_DEC(lnpwgts[to*ncon+h], lnpwgts[from*ncon+h], nvwgt[h]);
	      INC_DEC(gnpwgts[to*ncon+h], gnpwgts[from*ncon+h], nvwgt[h]);
	      INC_DEC(movewgts[to*ncon+h], movewgts[from*ncon+h], nvwgt[h]);
            }

            tmp_rinfo[i].ed += tmp_rinfo[i].id-my_edegrees[k].ewgt;
            SWAP(tmp_rinfo[i].id, my_edegrees[k].ewgt, j);
            if (my_edegrees[k].ewgt == 0) {
              tmp_rinfo[i].ndegrees--;
              my_edegrees[k].edge = my_edegrees[tmp_rinfo[i].ndegrees].edge;
              my_edegrees[k].ewgt = my_edegrees[tmp_rinfo[i].ndegrees].ewgt;
            }
            else {
              my_edegrees[k].edge = from;
            }

            /* Update the degrees of adjacent vertices */
            for (j=xadj[i]; j<xadj[i+1]; j++) {
              /* no need to bother about vertices on different pe's */
              if (ladjncy[j] >= nvtxs)
                continue;

              me = ladjncy[j];
              mydomain = tmp_where[me];

              myrinfo = tmp_rinfo+me;
              your_edegrees = myrinfo->degrees;

              if (mydomain == from) {
                INC_DEC(myrinfo->ed, myrinfo->id, adjwgt[j]);
              }
              else {
                if (mydomain == to) {
                  INC_DEC(myrinfo->id, myrinfo->ed, adjwgt[j]);
                }
              }

              /* Remove contribution from the .ed of 'from' */
              if (mydomain != from) {
                for (k=0; k<myrinfo->ndegrees; k++) {
                  if (your_edegrees[k].edge == from) {
                    if (your_edegrees[k].ewgt == adjwgt[j]) {
                      myrinfo->ndegrees--;
                      your_edegrees[k].edge = your_edegrees[myrinfo->ndegrees].edge;
                      your_edegrees[k].ewgt = your_edegrees[myrinfo->ndegrees].ewgt;
                    }
                    else {
                      your_edegrees[k].ewgt -= adjwgt[j];
                    }
                    break;
                  }
                }
              }

              /* Add contribution to the .ed of 'to' */
              if (mydomain != to) {
                for (k=0; k<myrinfo->ndegrees; k++) {
                  if (your_edegrees[k].edge == to) {
                    your_edegrees[k].ewgt += adjwgt[j];
                    break;
                  }
                }
                if (k == myrinfo->ndegrees) {
                  your_edegrees[myrinfo->ndegrees].edge = to;
                  your_edegrees[myrinfo->ndegrees++].ewgt = adjwgt[j];
                }
              }
            }
          }
        }
      }

      /******************************************/
      /* Let processors know the subdomain wgts */
      /* if all proposed moves commit.          */
      /******************************************/
      MPI_Allreduce((void *)lnpwgts, (void *)pgnpwgts, nparts*ncon, MPI_FLOAT, MPI_SUM, ctrl->comm);

      /**************************/
      /* compute overfill array */
      /**************************/
      overweight = 0;
      for (j=0; j<nparts; j++) {
        for (h=0; h<ncon; h++) {
          if (pgnpwgts[j*ncon+h] > ognpwgts[j*ncon+h]) 
            overfill[j*ncon+h] = (pgnpwgts[j*ncon+h]-badmaxpwgt[j*ncon+h]) / (pgnpwgts[j*ncon+h]-ognpwgts[j*ncon+h]);
          else 
            overfill[j*ncon+h] = 0.0;

          overfill[j*ncon+h] = amax(overfill[j*ncon+h], 0.0);
          overfill[j*ncon+h] *= movewgts[j*ncon+h];

          if (overfill[j*ncon+h] > 0.0)
            overweight = 1;

          ASSERTP(ctrl, ognpwgts[j*ncon+h] <= badmaxpwgt[j*ncon+h] || pgnpwgts[j*ncon+h] <= ognpwgts[j*ncon+h],
                  (ctrl, "%.4f %.4f %.4f\n", ognpwgts[j*ncon+h], badmaxpwgt[j*ncon+h], pgnpwgts[j*ncon+h]));
        }
      }

      /****************************************************/
      /* select moves to undo according to overfill array */
      /****************************************************/
      if (overweight == 1) {
        for (iii=0; iii<nmoved; iii++) {
          i = moved[iii];
          oldto = tmp_where[i];
          nvwgt = graph->nvwgt+i*ncon;
          my_edegrees = tmp_rinfo[i].degrees;

          for (k=0; k<tmp_rinfo[i].ndegrees; k++)
            if (my_edegrees[k].edge == where[i])
              break;

          for (h=0; h<ncon; h++)
            if (nvwgt[h] > 0.0 && overfill[oldto*ncon+h] > nvwgt[h]/4.0)
              break;

          /**********************************/
          /* nullify this move if necessary */
          /**********************************/
          if (k != tmp_rinfo[i].ndegrees && h != ncon) {
            moved[iii] = -1;
            from = oldto;
            to = where[i];

            for (h=0; h<ncon; h++) 
              overfill[oldto*ncon+h] = amax(overfill[oldto*ncon+h]-nvwgt[h], 0.0);

            tmp_where[i] = to;
            tmp_rinfo[i].ed += tmp_rinfo[i].id-my_edegrees[k].ewgt;
            SWAP(tmp_rinfo[i].id, my_edegrees[k].ewgt, j);
            if (my_edegrees[k].ewgt == 0) {
              tmp_rinfo[i].ndegrees--;
              my_edegrees[k].edge = my_edegrees[tmp_rinfo[i].ndegrees].edge;
              my_edegrees[k].ewgt = my_edegrees[tmp_rinfo[i].ndegrees].ewgt;
            }
            else {
              my_edegrees[k].edge = from;
            }

            for (h=0; h<ncon; h++) 
	      INC_DEC(lnpwgts[to*ncon+h], lnpwgts[from*ncon+h], nvwgt[h]);

            /* Update the degrees of adjacent vertices */
            for (j=xadj[i]; j<xadj[i+1]; j++) {
              /* no need to bother about vertices on different pe's */
              if (ladjncy[j] >= nvtxs)
                continue;

              me = ladjncy[j];
              mydomain = tmp_where[me];

              myrinfo = tmp_rinfo+me;
              your_edegrees = myrinfo->degrees;

              if (mydomain == from) {
                INC_DEC(myrinfo->ed, myrinfo->id, adjwgt[j]);
              }
              else {
                if (mydomain == to) {
                  INC_DEC(myrinfo->id, myrinfo->ed, adjwgt[j]);
                }
              }

              /* Remove contribution from the .ed of 'from' */
              if (mydomain != from) {
                for (k=0; k<myrinfo->ndegrees; k++) {
                  if (your_edegrees[k].edge == from) {
                    if (your_edegrees[k].ewgt == adjwgt[j]) {
                      myrinfo->ndegrees--;
                      your_edegrees[k].edge = your_edegrees[myrinfo->ndegrees].edge;
                      your_edegrees[k].ewgt = your_edegrees[myrinfo->ndegrees].ewgt;
                    }
                    else {
                      your_edegrees[k].ewgt -= adjwgt[j];
                    }
                    break;
                  }
                }
              }

              /* Add contribution to the .ed of 'to' */
              if (mydomain != to) {
                for (k=0; k<myrinfo->ndegrees; k++) {
                  if (your_edegrees[k].edge == to) {
                    your_edegrees[k].ewgt += adjwgt[j];
                    break;
                  }
                }
                if (k == myrinfo->ndegrees) {
                  your_edegrees[myrinfo->ndegrees].edge = to;
                  your_edegrees[myrinfo->ndegrees++].ewgt = adjwgt[j];
                }
              }
            }
          }
        }
      }

      /*************************************************/
      /* PASS TWO -- commit the remainder of the moves */
      /*************************************************/
      nlupd = nsupd = nmoves = nchanged = 0;
      for (iii=0; iii<nmoved; iii++) {
        i = moved[iii];
        if (i == -1)
          continue;

        where[i] = tmp_where[i]; 

        /* Make sure to update the vertex information */
        if (htable[i] == 0) {
          /* make sure you do the update */
          htable[i] = 1;
          update[nlupd++] = i;
        }

        /* Put the vertices adjacent to i into the update array */
        for (j=xadj[i]; j<xadj[i+1]; j++) {
          k = ladjncy[j];
          if (htable[k] == 0) {
            htable[k] = 1;
            if (k<nvtxs)
              update[nlupd++] = k;
            else
              supdate[nsupd++] = k;
          }
        }
        nmoves++;

        if (graph->pexadj[i+1]-graph->pexadj[i] > 0)
          changed[nchanged++] = i;
      }

      /* Tell interested pe's the new where[] info for the interface vertices */
      CommChangedInterfaceData(ctrl, graph, nchanged, changed, where, swchanges, rwchanges, wspace->pv4); 


      IFSET(ctrl->dbglvl, DBG_RMOVEINFO, rprintf(ctrl, "\t[%d %d], [%.4f],  [%d %d %d]\n",
            pass, c, badmaxpwgt[0], GlobalSESum(ctrl, nmoves), GlobalSESum(ctrl, nsupd), GlobalSESum(ctrl, nlupd)));

      /*-------------------------------------------------------------
      / Time to communicate with processors to send the vertices
      / whose degrees need to be update.
      /-------------------------------------------------------------*/
      /* Issue the receives first */
      for (i=0; i<nnbrs; i++) 
        MPI_Irecv((void *)(rupdate+sendptr[i]), sendptr[i+1]-sendptr[i], IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->rreq+i);

      /* Issue the sends next. This needs some preporcessing */
      for (i=0; i<nsupd; i++) {
        htable[supdate[i]] = 0;
        supdate[i] = graph->imap[supdate[i]];
      }
      iidxsort(nsupd, supdate);

      for (j=i=0; i<nnbrs; i++) {
        yourlastvtx = vtxdist[peind[i]+1];
        for (k=j; k<nsupd && supdate[k] < yourlastvtx; k++); 
        MPI_Isend((void *)(supdate+j), k-j, IDX_DATATYPE, peind[i], 1, ctrl->comm, ctrl->sreq+i);
        j = k;
      }

      /* OK, now get into the loop waiting for the send/recv operations to finish */
      MPI_Waitall(nnbrs, ctrl->rreq, ctrl->statuses);
      for (i=0; i<nnbrs; i++) 
        MPI_Get_count(ctrl->statuses+i, IDX_DATATYPE, nupds_pe+i);
      MPI_Waitall(nnbrs, ctrl->sreq, ctrl->statuses);


      /*-------------------------------------------------------------
      / Place the recieved to-be updated vertices into update[] 
      /-------------------------------------------------------------*/
      for (i=0; i<nnbrs; i++) {
        pe_updates = rupdate+sendptr[i];
        for (j=0; j<nupds_pe[i]; j++) {
          k = pe_updates[j];
          if (htable[k-firstvtx] == 0) {
            htable[k-firstvtx] = 1;
            update[nlupd++] = k-firstvtx;
          }
        }
      }


      /*-------------------------------------------------------------
      / Update the rinfo of the vertices in the update[] array
      /-------------------------------------------------------------*/
      for (ii=0; ii<nlupd; ii++) {
        i = update[ii];
        ASSERT(ctrl, htable[i] == 1);

        htable[i] = 0;

        mydomain = where[i];
        myrinfo = rinfo+i;
        tmp_myrinfo = tmp_rinfo+i;
        my_edegrees = myrinfo->degrees;
        your_edegrees = tmp_myrinfo->degrees;

        graph->lmincut -= myrinfo->ed;
        myrinfo->ndegrees = 0;
        myrinfo->id = 0;
        myrinfo->ed = 0;

        for (j=xadj[i]; j<xadj[i+1]; j++) {
          yourdomain = where[ladjncy[j]];
          if (mydomain != yourdomain) {
            myrinfo->ed += adjwgt[j];

            for (k=0; k<myrinfo->ndegrees; k++) {
              if (my_edegrees[k].edge == yourdomain) {
                my_edegrees[k].ewgt += adjwgt[j];
                your_edegrees[k].ewgt += adjwgt[j];
                break;
              }
            }
            if (k == myrinfo->ndegrees) {
              my_edegrees[k].edge = yourdomain;
              my_edegrees[k].ewgt = adjwgt[j];
              your_edegrees[k].edge = yourdomain;
              your_edegrees[k].ewgt = adjwgt[j];
              myrinfo->ndegrees++;
            }
            ASSERT(ctrl, myrinfo->ndegrees <= xadj[i+1]-xadj[i]);
            ASSERT(ctrl, tmp_myrinfo->ndegrees <= xadj[i+1]-xadj[i]);

          }
          else {
            myrinfo->id += adjwgt[j];
          }
        }
        graph->lmincut += myrinfo->ed;

        tmp_myrinfo->id = myrinfo->id;
        tmp_myrinfo->ed = myrinfo->ed;
        tmp_myrinfo->ndegrees = myrinfo->ndegrees;
      }

      /* finally, sum-up the partition weights */
      MPI_Allreduce((void *)lnpwgts, (void *)gnpwgts, nparts*ncon, MPI_FLOAT, MPI_SUM, ctrl->comm);
    }
    graph->mincut = GlobalSESum(ctrl, graph->lmincut)/2;

    if (graph->mincut == oldcut)
      break;
  }

  GKfree((void **)&badmaxpwgt, (void **)&update, (void **)&nupds_pe, (void **)&htable, LTERM);
  GKfree((void **)&changed, (void **)&pperm, (void **)&perm, (void **)&moved, LTERM);
  GKfree((void **)&pgnpwgts, (void **)&ognpwgts, (void **)&overfill, (void **)&movewgts, LTERM);
  GKfree((void **)&tmp_where, (void **)&tmp_rinfo, (void **)&tmp_edegrees, LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->KWayTmr));
}


