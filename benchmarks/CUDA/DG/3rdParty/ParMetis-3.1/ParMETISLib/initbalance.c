/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * initbalance.c
 *
 * This file contains code that computes an initial partitioning
 *
 * Started 3/4/96
 * George
 *
 * $Id: initbalance.c,v 1.4 2003/07/30 21:18:52 karypis Exp $
 */

#include <parmetislib.h>


/*************************************************************************
* This function is the entry point of the initial balancing algorithm.
* This algorithm assembles the graph to all the processors and preceeds
* with the balancing step.
**************************************************************************/
void Balance_Partition(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, j, mype, npes, nvtxs, nedges, ncon;
  idxtype *vtxdist, *xadj, *adjncy, *adjwgt, *vwgt, *vsize;
  idxtype *part, *lwhere, *home;
  GraphType *agraph, cgraph;
  CtrlType myctrl;
  int lnparts, fpart, fpe, lnpes, ngroups, srnpes, srmype; 
  int twoparts=2, numflag = 0, wgtflag = 3, moptions[10], edgecut, max_cut;
  int sr_pe, gd_pe, sr, gd, who_wins, *rcounts, *rdispls;
  float my_cut, my_totalv, my_cost = -1.0, my_balance = -1.0, wsum;
  float rating, max_rating, your_cost = -1.0, your_balance = -1.0;
  float lbvec[MAXNCON], lbsum, min_lbsum, *mytpwgts, mytpwgts2[2], buffer[2];
  MPI_Status status;
  MPI_Comm ipcomm, srcomm;
  struct {
    float cost;
    int rank;
  } lpecost, gpecost;

  IFSET(ctrl->dbglvl, DBG_TIME, starttimer(ctrl->InitPartTmr));

  vtxdist = graph->vtxdist;
  agraph = Moc_AssembleAdaptiveGraph(ctrl, graph, wspace);
  nvtxs = cgraph.nvtxs = agraph->nvtxs;
  nedges = cgraph.nedges = agraph->nedges;
  ncon = cgraph.ncon = agraph->ncon;

  xadj = cgraph.xadj = idxmalloc(nvtxs*(5+ncon)+1+nedges*2, "U_IP: xadj");
  vwgt = cgraph.vwgt = xadj + nvtxs+1;
  vsize = cgraph.vsize = xadj + nvtxs*(1+ncon)+1;
  cgraph.where = agraph->where = part = xadj + nvtxs*(2+ncon)+1;
  lwhere = xadj + nvtxs*(3+ncon)+1;
  home = xadj + nvtxs*(4+ncon)+1;
  adjncy = cgraph.adjncy = xadj + nvtxs*(5+ncon)+1;
  adjwgt = cgraph.adjwgt = xadj + nvtxs*(5+ncon)+1 + nedges;

  /* ADD: this assumes that tpwgts for all constraints is the same */
  /* ADD: this is necessary because serial metis does not support the general case */
  mytpwgts = fsmalloc(ctrl->nparts, 0.0, "mytpwgts");
  for (i=0; i<ctrl->nparts; i++)
    for (j=0; j<ncon; j++)
      mytpwgts[i] += ctrl->tpwgts[i*ncon+j];
  for (i=0; i<ctrl->nparts; i++)
    mytpwgts[i] /= (float)ncon;

  idxcopy(nvtxs+1, agraph->xadj, xadj);
  idxcopy(nvtxs*ncon, agraph->vwgt, vwgt);
  idxcopy(nvtxs, agraph->vsize, vsize);
  idxcopy(nedges, agraph->adjncy, adjncy);
  idxcopy(nedges, agraph->adjwgt, adjwgt);

  /****************************************/
  /****************************************/
  if (ctrl->ps_relation == DISCOUPLED) {
    rcounts = imalloc(ctrl->npes, "rcounts");
    rdispls = imalloc(ctrl->npes+1, "rdispls");

    for (i=0; i<ctrl->npes; i++) {
      rdispls[i] = rcounts[i] = vtxdist[i+1]-vtxdist[i];
    }
    MAKECSR(i, ctrl->npes, rdispls);

    MPI_Allgatherv((void *)graph->home, graph->nvtxs, IDX_DATATYPE,
    (void *)part, rcounts, rdispls, IDX_DATATYPE, ctrl->comm);

    for (i=0; i<agraph->nvtxs; i++)
      home[i] = part[i];

    GKfree((void **)&rcounts, (void **)&rdispls, LTERM);
  }
  else {
    for (i=0; i<ctrl->npes; i++)
      for (j=vtxdist[i]; j<vtxdist[i+1]; j++)
        part[j] = home[j] = i;
  }

  /* Ensure that the initial partitioning is legal */
  for (i=0; i<agraph->nvtxs; i++) {
    if (part[i] >= ctrl->nparts)
      part[i] = home[i] = part[i] % ctrl->nparts;
    if (part[i] < 0)
      part[i] = home[i] = (-1*part[i]) % ctrl->nparts;
  }
  /****************************************/
  /****************************************/

  IFSET(ctrl->dbglvl, DBG_REFINEINFO, Moc_ComputeSerialBalance(ctrl, agraph, agraph->where, lbvec));
  IFSET(ctrl->dbglvl, DBG_REFINEINFO, rprintf(ctrl, "input cut: %d, balance: ", ComputeSerialEdgeCut(agraph)));
  for (i=0; i<agraph->ncon; i++)
    IFSET(ctrl->dbglvl, DBG_REFINEINFO, rprintf(ctrl, "%.3f ", lbvec[i]));
  IFSET(ctrl->dbglvl, DBG_REFINEINFO, rprintf(ctrl, "\n"));

  /****************************************/
  /* Split the processors into two groups */
  /****************************************/
  sr = (ctrl->mype % 2 == 0) ? 1 : 0;
  gd = (ctrl->mype % 2 == 1) ? 1 : 0;

  if (graph->ncon > MAX_NCON_FOR_DIFFUSION || ctrl->npes == 1) {
    sr = 1;
    gd = 0;
  }

  sr_pe = 0;
  gd_pe = 1;

  MPI_Comm_split(ctrl->gcomm, sr, 0, &ipcomm);
  MPI_Comm_rank(ipcomm, &mype);
  MPI_Comm_size(ipcomm, &npes);

  myctrl.dbglvl = 0;
  myctrl.mype = mype;
  myctrl.npes = npes;
  myctrl.comm = ipcomm;
  myctrl.sync = ctrl->sync;
  myctrl.seed = ctrl->seed;
  myctrl.nparts = ctrl->nparts;
  myctrl.ipc_factor = ctrl->ipc_factor;
  myctrl.redist_factor = ctrl->redist_base;
  myctrl.partType = ADAPTIVE_PARTITION;
  myctrl.ps_relation = DISCOUPLED;
  myctrl.tpwgts = ctrl->tpwgts;
  icopy(ncon, ctrl->tvwgts, myctrl.tvwgts);
  icopy(ncon, ctrl->ubvec, myctrl.ubvec);

  if (sr == 1) {
    /*******************************************/
    /* Half of the processors do scratch-remap */
    /*******************************************/
    ngroups = amax(amin(RIP_SPLIT_FACTOR, npes), 1);
    MPI_Comm_split(ipcomm, mype % ngroups, 0, &srcomm);
    MPI_Comm_rank(srcomm, &srmype);
    MPI_Comm_size(srcomm, &srnpes);

    moptions[0] = 0;
    moptions[7] = ctrl->sync + (mype % ngroups) + 1;

    idxset(nvtxs, 0, lwhere);
    lnparts = ctrl->nparts;
    fpart = fpe = 0;
    lnpes = srnpes;
    while (lnpes > 1 && lnparts > 1) {
      ASSERT(ctrl, agraph->nvtxs > 1);
      /* Determine the weights of the partitions */
      mytpwgts2[0] = ssum(lnparts/2, mytpwgts+fpart);
      mytpwgts2[1] = 1.0-mytpwgts2[0];


      if (agraph->ncon == 1) {
        METIS_WPartGraphKway2(&agraph->nvtxs, agraph->xadj, agraph->adjncy, agraph->vwgt, 
	      agraph->adjwgt, &wgtflag, &numflag, &twoparts, mytpwgts2, moptions, &edgecut, 
	      part);
      }
      else {
        METIS_mCPartGraphRecursive2(&agraph->nvtxs, &ncon, agraph->xadj, agraph->adjncy, 
	      agraph->vwgt, agraph->adjwgt, &wgtflag, &numflag, &twoparts, mytpwgts2, 
	      moptions, &edgecut, part);
      }

      wsum = ssum(lnparts/2, mytpwgts+fpart);
      sscale(lnparts/2, 1.0/wsum, mytpwgts+fpart);
      sscale(lnparts-lnparts/2, 1.0/(1.0-wsum), mytpwgts+fpart+lnparts/2);

      /* I'm picking the left branch */
      if (srmype < fpe+lnpes/2) {
        Moc_KeepPart(agraph, wspace, part, 0);
        lnpes = lnpes/2;
        lnparts = lnparts/2;
      }
      else {
        Moc_KeepPart(agraph, wspace, part, 1);
        fpart = fpart + lnparts/2;
        fpe = fpe + lnpes/2;
        lnpes = lnpes - lnpes/2;
        lnparts = lnparts - lnparts/2;
      }
    }

    /* In case srnpes is greater than or equal to nparts */
    if (lnparts == 1) {
      /* Only the first process will assign labels (for the reduction to work) */
      if (srmype == fpe) {
        for (i=0; i<agraph->nvtxs; i++) 
          lwhere[agraph->label[i]] = fpart;
      }
    }
    /* In case srnpes is smaller than nparts */
    else {
      if (ncon == 1)
        METIS_WPartGraphKway2(&agraph->nvtxs, agraph->xadj, agraph->adjncy, agraph->vwgt, 
	      agraph->adjwgt, &wgtflag, &numflag, &lnparts, mytpwgts+fpart, moptions, 
	      &edgecut, part);
      else
        METIS_mCPartGraphRecursive2(&agraph->nvtxs, &ncon, agraph->xadj, agraph->adjncy, 
	      agraph->vwgt, agraph->adjwgt, &wgtflag, &numflag, &lnparts, mytpwgts+fpart, 
	      moptions, &edgecut, part);

      for (i=0; i<agraph->nvtxs; i++) 
        lwhere[agraph->label[i]] = fpart + part[i];
    }

    MPI_Allreduce((void *)lwhere, (void *)part, nvtxs, IDX_DATATYPE, MPI_SUM, srcomm);

    edgecut = ComputeSerialEdgeCut(&cgraph);
    Moc_ComputeSerialBalance(ctrl, &cgraph, part, lbvec);
    lbsum = ssum(ncon, lbvec);
    MPI_Allreduce((void *)&edgecut, (void *)&max_cut, 1, MPI_INT, MPI_MAX, ipcomm);
    MPI_Allreduce((void *)&lbsum, (void *)&min_lbsum, 1, MPI_FLOAT, MPI_MIN, ipcomm);
    lpecost.rank = ctrl->mype;
    lpecost.cost = lbsum;
    if (min_lbsum < UNBALANCE_FRACTION * (float)(ncon)) {
      if (lbsum < UNBALANCE_FRACTION * (float)(ncon))
        lpecost.cost = (float)edgecut;
      else
        lpecost.cost = (float)max_cut + lbsum;
    }
    MPI_Allreduce((void *)&lpecost, (void *)&gpecost, 1, MPI_FLOAT_INT, MPI_MINLOC, ipcomm);

    if (ctrl->mype == gpecost.rank && ctrl->mype != sr_pe) {
      MPI_Send((void *)part, nvtxs, IDX_DATATYPE, sr_pe, 1, ctrl->comm);
    }

    if (ctrl->mype != gpecost.rank && ctrl->mype == sr_pe) {
      MPI_Recv((void *)part, nvtxs, IDX_DATATYPE, gpecost.rank, 1, ctrl->comm, &status);
    }

    if (ctrl->mype == sr_pe) {
      idxcopy(nvtxs, part, lwhere);
      SerialRemap(&cgraph, ctrl->nparts, home, lwhere, part, ctrl->tpwgts);
    }

    MPI_Comm_free(&srcomm);
  }
  /**************************************/
  /* The other half do global diffusion */
  /**************************************/
  else {
    /******************************************************************/
    /* The next stmt is required to balance out the sr MPI_Comm_split */
    /******************************************************************/
    MPI_Comm_split(ipcomm, MPI_UNDEFINED, 0, &srcomm);

    if (ncon == 1) {
      rating = WavefrontDiffusion(&myctrl, agraph, home);
      Moc_ComputeSerialBalance(ctrl, &cgraph, part, lbvec);
      lbsum = ssum(ncon, lbvec);

      /* Determine which PE computed the best partitioning */
      MPI_Allreduce((void *)&rating, (void *)&max_rating, 1, MPI_FLOAT, MPI_MAX, ipcomm);
      MPI_Allreduce((void *)&lbsum, (void *)&min_lbsum, 1, MPI_FLOAT, MPI_MIN, ipcomm);

      lpecost.rank = ctrl->mype;
      lpecost.cost = lbsum;
      if (min_lbsum < UNBALANCE_FRACTION * (float)(ncon)) {
        if (lbsum < UNBALANCE_FRACTION * (float)(ncon))
          lpecost.cost = rating;
        else
          lpecost.cost = max_rating + lbsum;
      }

      MPI_Allreduce((void *)&lpecost, (void *)&gpecost, 1, MPI_FLOAT_INT, MPI_MINLOC, ipcomm);

      /* Now send this to the coordinating processor */
      if (ctrl->mype == gpecost.rank && ctrl->mype != gd_pe)
        MPI_Send((void *)part, nvtxs, IDX_DATATYPE, gd_pe, 1, ctrl->comm);

      if (ctrl->mype != gpecost.rank && ctrl->mype == gd_pe)
        MPI_Recv((void *)part, nvtxs, IDX_DATATYPE, gpecost.rank, 1, ctrl->comm, &status);

      if (ctrl->mype == gd_pe) {
        idxcopy(nvtxs, part, lwhere);
        SerialRemap(&cgraph, ctrl->nparts, home, lwhere, part, ctrl->tpwgts);
      }
    }
    else {
      Moc_Diffusion(&myctrl, agraph, graph->vtxdist, agraph->where, home, wspace, N_MOC_GD_PASSES);
    }
  }

  if (graph->ncon <= MAX_NCON_FOR_DIFFUSION) {
    if (ctrl->mype == sr_pe  || ctrl->mype == gd_pe) {
      /********************************************************************/
      /* The coordinators from each group decide on the best partitioning */
      /********************************************************************/
      my_cut = (float) ComputeSerialEdgeCut(&cgraph);
      my_totalv = (float) Mc_ComputeSerialTotalV(&cgraph, home);
      Moc_ComputeSerialBalance(ctrl, &cgraph, part, lbvec);
      my_balance = ssum(cgraph.ncon, lbvec);
      my_balance /= (float) cgraph.ncon;
      my_cost = ctrl->ipc_factor * my_cut + REDIST_WGT * ctrl->redist_base * my_totalv;

      IFSET(ctrl->dbglvl, DBG_REFINEINFO, printf("%s initial cut: %.1f, totalv: %.1f, balance: %.3f\n",
      (ctrl->mype == sr_pe ? "scratch-remap" : "diffusion"), my_cut, my_totalv, my_balance));

      if (ctrl->mype == gd_pe) {
        buffer[0] = my_cost;
        buffer[1] = my_balance;
        MPI_Send((void *)buffer, 2, MPI_FLOAT, sr_pe, 1, ctrl->comm);
      }
      else {
        MPI_Recv((void *)buffer, 2, MPI_FLOAT, gd_pe, 1, ctrl->comm, &status);
        your_cost = buffer[0];
        your_balance = buffer[1];
      }
    }

    if (ctrl->mype == sr_pe) {
      who_wins = gd_pe;
      if ((my_balance < 1.1 && your_balance > 1.1) ||
          (my_balance < 1.1 && your_balance < 1.1 && my_cost < your_cost) ||
          (my_balance > 1.1 && your_balance > 1.1 && my_balance < your_balance)) {
        who_wins = sr_pe;
      }
    }

    MPI_Bcast((void *)&who_wins, 1, MPI_INT, sr_pe, ctrl->comm);
  }
  else {
    who_wins = sr_pe;
  }

  MPI_Bcast((void *)part, nvtxs, IDX_DATATYPE, who_wins, ctrl->comm);
  idxcopy(graph->nvtxs, part+vtxdist[ctrl->mype], graph->where);

  MPI_Comm_free(&ipcomm);
  GKfree((void **)&xadj, (void **)&mytpwgts, LTERM);

/* For whatever reason, FreeGraph crashes here...so explicitly free the memory.
  FreeGraph(agraph);
*/
  GKfree((void **)&agraph->xadj, (void **)&agraph->adjncy, (void **)&agraph->vwgt, (void **)&agraph->nvwgt, LTERM);
  GKfree((void **)&agraph->vsize, (void **)&agraph->adjwgt, (void **)&agraph->label, LTERM);
  GKfree((void **)&agraph, LTERM);

  IFSET(ctrl->dbglvl, DBG_TIME, stoptimer(ctrl->InitPartTmr));

}


/* NOTE: this subroutine should work for static, adaptive, single-, and multi-contraint */
/*************************************************************************
* This function assembles the graph into a single processor
**************************************************************************/
GraphType *Moc_AssembleAdaptiveGraph(CtrlType *ctrl, GraphType *graph, WorkSpaceType *wspace)
{
  int i, j, k, l, gnvtxs, nvtxs, ncon, gnedges, nedges, gsize;
  idxtype *xadj, *vwgt, *vsize, *adjncy, *adjwgt, *vtxdist, *imap;
  idxtype *axadj, *aadjncy, *aadjwgt, *avwgt, *avsize = NULL, *alabel;
  idxtype *mygraph, *ggraph;
  int *rcounts, *rdispls, mysize;
  float *anvwgt;
  GraphType *agraph;

  gnvtxs  = graph->gnvtxs;
  nvtxs   = graph->nvtxs;
  ncon    = graph->ncon;
  nedges  = graph->xadj[nvtxs];
  xadj    = graph->xadj;
  vwgt    = graph->vwgt;
  vsize   = graph->vsize;
  adjncy  = graph->adjncy;
  adjwgt  = graph->adjwgt;
  vtxdist = graph->vtxdist;
  imap    = graph->imap;

  /*************************************************************/
  /* Determine the # of idxtype to receive from each processor */
  /*************************************************************/
  rcounts = imalloc(ctrl->npes, "AssembleGraph: rcounts");
  switch (ctrl->partType) {
    case STATIC_PARTITION:
      mysize = (1+ncon)*nvtxs + 2*nedges;
      break;
    case ADAPTIVE_PARTITION:
    case REFINE_PARTITION:
      mysize = (2+ncon)*nvtxs + 2*nedges;
      break;
    default:
      printf("WARNING: bad value for ctrl->partType %d\n", ctrl->partType);
      break;
  }
  MPI_Allgather((void *)(&mysize), 1, MPI_INT, (void *)rcounts, 1, MPI_INT, ctrl->comm);

  rdispls = imalloc(ctrl->npes+1, "AssembleGraph: rdispls");
  rdispls[0] = 0;
  for (i=1; i<ctrl->npes+1; i++)
    rdispls[i] = rdispls[i-1] + rcounts[i-1];

  /* Construct the one-array storage format of the assembled graph */
  mygraph = (mysize <= wspace->maxcore ? wspace->core : idxmalloc(mysize, "AssembleGraph: mygraph"));
  for (k=i=0; i<nvtxs; i++) {
    mygraph[k++] = xadj[i+1]-xadj[i];
    for (j=0; j<ncon; j++)
      mygraph[k++] = vwgt[i*ncon+j];
    if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION)
      mygraph[k++] = vsize[i];
    for (j=xadj[i]; j<xadj[i+1]; j++) {
      mygraph[k++] = imap[adjncy[j]];
      mygraph[k++] = adjwgt[j];
    }
  }
  ASSERT(ctrl, mysize == k);

  /**************************************/
  /* Assemble and send the entire graph */
  /**************************************/
  gsize = rdispls[ctrl->npes];
  ggraph = (gsize <= wspace->maxcore-mysize ? wspace->core+mysize : idxmalloc(gsize, "AssembleGraph: ggraph"));
  MPI_Allgatherv((void *)mygraph, mysize, IDX_DATATYPE, (void *)ggraph, rcounts, rdispls, IDX_DATATYPE, ctrl->comm);

  GKfree((void **)&rcounts, (void **)&rdispls, LTERM);
  if (mysize > wspace->maxcore)
    free(mygraph);

  agraph = CreateGraph();
  agraph->nvtxs = gnvtxs;
  switch (ctrl->partType) {
    case STATIC_PARTITION:
      agraph->nedges = gnedges = (gsize-(1+ncon)*gnvtxs)/2;
      break;
    case ADAPTIVE_PARTITION:
    case REFINE_PARTITION:
      agraph->nedges = gnedges = (gsize-(2+ncon)*gnvtxs)/2;
      break;
    default:
      printf("WARNING: bad value for ctrl->partType %d\n", ctrl->partType);
      agraph->nedges = gnedges = -1;
      break;
  }

  agraph->ncon = ncon;

  /*******************************************/
  /* Allocate memory for the assembled graph */
  /*******************************************/
  axadj = agraph->xadj = idxmalloc(gnvtxs+1, "AssembleGraph: axadj");
  avwgt = agraph->vwgt = idxmalloc(gnvtxs*ncon, "AssembleGraph: avwgt");
  anvwgt = agraph->nvwgt = fmalloc(gnvtxs*ncon, "AssembleGraph: anvwgt");
  aadjncy = agraph->adjncy = idxmalloc(gnedges, "AssembleGraph: adjncy");
  aadjwgt = agraph->adjwgt = idxmalloc(gnedges, "AssembleGraph: adjwgt");
  alabel = agraph->label = idxmalloc(gnvtxs, "AssembleGraph: alabel");
  if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION)
    avsize = agraph->vsize = idxmalloc(gnvtxs, "AssembleGraph: avsize");

  for (k=j=i=0; i<gnvtxs; i++) {
    axadj[i] = ggraph[k++];
    for (l=0; l<ncon; l++)
      avwgt[i*ncon+l] = ggraph[k++];
    if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION)
      avsize[i] = ggraph[k++];
    for (l=0; l<axadj[i]; l++) {
      aadjncy[j] = ggraph[k++];
      aadjwgt[j] = ggraph[k++];
      j++;
    }
  }

  /*********************************/
  /* Now fix up the received graph */
  /*********************************/
  MAKECSR(i, gnvtxs, axadj);

  for (i=0; i<gnvtxs; i++)
    for (j=0; j<ncon; j++)
      anvwgt[i*ncon+j] = (float)(agraph->vwgt[i*ncon+j]) / (float)(ctrl->tvwgts[j]);

  for (i=0; i<gnvtxs; i++)
    alabel[i] = i;

  if (gsize > wspace->maxcore-mysize)
    free(ggraph);

  return agraph;
}


