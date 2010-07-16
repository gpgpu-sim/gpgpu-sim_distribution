/* * Copyright 1997, Regents of the University of Minnesota
 *
 * mdiffusion.c
 *
 * This file contains code that performs mc-diffusion
 *
 * Started 9/16/99
 * George
 *
 * $Id: mdiffusion.c,v 1.2 2003/07/21 17:18:50 karypis Exp $
 */

#include <parmetislib.h>

#define PE	-1

/*************************************************************************
* This function is the entry point of the initial partitioning algorithm.
* This algorithm assembles the graph to all the processors and preceed
* serially.
**************************************************************************/
int Moc_Diffusion(CtrlType *ctrl, GraphType *graph, idxtype *vtxdist,
  idxtype *where, idxtype *home, WorkSpaceType *wspace, int npasses)
{
  int h, i, j;
  int nvtxs, nedges, ncon, pass, iter, domain, processor;
  int nparts, mype, npes, nlinks, me, you, wsize;
  int nvisited, nswaps = -1, tnswaps, done, alldone = -1;
  idxtype *rowptr, *colind, *diff_where, *sr_where, *ehome, *map, *rmap;
  idxtype *pack, *unpack, *match, *proc2sub, *sub2proc;
  idxtype *visited, *gvisited;
  float *transfer, *npwgts, maxdiff, minflow, maxflow;
  float lbavg, oldlbavg, ubavg, lbvec[MAXNCON];
  float diff_flows[MAXNCON], sr_flows[MAXNCON];
  float diff_lbavg, sr_lbavg, diff_cost, sr_cost;
  idxtype *rbuffer, *sbuffer; 
  int *rcount, *rdispl;
  float *solution, *load, *workspace;
  EdgeType *degrees;
  MatrixType matrix;
  GraphType *egraph;
  RInfoType *rinfo;

  if (graph->ncon > 3)
    return 0;

  nvtxs = graph->nvtxs;
  nedges = graph->nedges;
  ncon = graph->ncon;

  nparts = ctrl->nparts;
  mype = ctrl->mype;
  npes = ctrl->npes;
  ubavg = savg(ncon, ctrl->ubvec);

  /********************************************/
  /* initialize variables and allocate memory */
  /********************************************/
  load = fmalloc(nparts*(2+ncon)+nedges*(1+ncon), "load");
  solution =                   load + nparts;
  npwgts = graph->gnpwgts =    load + 2*nparts;
  matrix.values =              load + (2+ncon)*nparts;
  transfer = matrix.transfer = load + (2+ncon)*nparts + nedges;

  proc2sub = idxmalloc(amax(nparts, npes*2), "Mc_Diffusion: proc2sub");
  sub2proc = idxmalloc(nparts*3+nedges+1, "Mc_Diffusion: match");
  match =                  sub2proc + nparts;
  rowptr = matrix.rowptr = sub2proc + 2*nparts;
  colind = matrix.colind = sub2proc + 3*nparts + 1;

  rcount = imalloc(2*npes+1, "Mc_Diffusion: rcount");
  rdispl = rcount + npes;

  pack = idxmalloc(nvtxs*8, "Mc_Diffusion: pack");
  unpack =     pack + nvtxs;
  rbuffer =    pack + 2*nvtxs;
  sbuffer =    pack + 3*nvtxs;
  map =        pack + 4*nvtxs;
  rmap =       pack + 5*nvtxs;
  diff_where = pack + 6*nvtxs;
  ehome =      pack + 7*nvtxs;

  wsize = amax(sizeof(float)*nparts*6, sizeof(idxtype)*(nvtxs+nparts*2+1));
  workspace = (float *)GKmalloc(wsize, "Moc_Diffusion: workspace");
  degrees = GKmalloc(nedges*sizeof(EdgeType), "Mc_Diffusion: degrees");
  rinfo = graph->rinfo = GKmalloc(nvtxs*sizeof(RInfoType), "Mc_Diffusion: rinfo");

  /******************************************/
  /* construct subdomain connectivity graph */
  /******************************************/
  matrix.nrows = nparts;
  SetUpConnectGraph(graph, &matrix, (idxtype *)workspace);
  nlinks = (matrix.nnzs-nparts) / 2;

  visited = idxmalloc(matrix.nnzs*2, "visited");
  gvisited = visited + matrix.nnzs;

  for (pass=0; pass<npasses; pass++) {
    sset(matrix.nnzs*ncon, 0.0, transfer);
    idxset(matrix.nnzs, 0, gvisited);
    idxset(matrix.nnzs, 0, visited);
    iter = nvisited = 0;

    /*******************************/
    /* compute ncon flow solutions */
    /*******************************/
    for (h=0; h<ncon; h++) {
      sset(nparts, 0.0, solution);
      ComputeLoad(graph, nparts, load, ctrl->tpwgts, h);

      lbvec[h] = (load[samax(nparts, load)]+1.0/(float)nparts) * (float)nparts;

      ConjGrad2(&matrix, load, solution, 0.001, workspace);
      ComputeTransferVector(ncon, &matrix, solution, transfer, h);
    }

    oldlbavg = savg(ncon, lbvec);
    tnswaps = 0;
    maxdiff = 0.0;
    for (i=0; i<nparts; i++) {
      for (j=rowptr[i]; j<rowptr[i+1]; j++) {
        minflow = transfer[j*ncon+samin(ncon, transfer+j*ncon)];
        maxflow = transfer[j*ncon+samax(ncon, transfer+j*ncon)];
        maxdiff = (maxflow - minflow > maxdiff) ? maxflow - minflow : maxdiff;
      }
    }

    while (nvisited < nlinks) {

      /******************************************/
      /* compute independent sets of subdomains */
      /******************************************/
      idxset(amax(nparts, npes*2), UNMATCHED, proc2sub);
      CSR_Match_SHEM(&matrix, match, proc2sub, gvisited, ncon);

      /*****************************/
      /* Set up the packing arrays */
      /*****************************/
      idxset(nparts, UNMATCHED, sub2proc);
      for (i=0; i<npes*2; i++) {
        if (proc2sub[i] == UNMATCHED)
          break;

        sub2proc[proc2sub[i]] = i/2;
      }

      iset(npes, 0, rcount);
      for (i=0; i<nvtxs; i++) {
        domain = where[i];
        processor = sub2proc[domain];
        if (processor != UNMATCHED) {
          rcount[processor]++;
        }
      }

      rdispl[0] = 0;
      for (i=1; i<npes+1; i++)
        rdispl[i] = rdispl[i-1] + rcount[i-1];

      idxset(nvtxs, UNMATCHED, unpack);
      for (i=0; i<nvtxs; i++) {
        domain = where[i];
        processor = sub2proc[domain];
        if (processor != UNMATCHED) {
          unpack[rdispl[processor]++] = i;
        }
      }

      for (i=npes; i>0; i--)
        rdispl[i] = rdispl[i-1];
      rdispl[0] = 0;

      idxset(nvtxs, UNMATCHED, pack);
      for (i=0; i<rdispl[npes]; i++) {
        ASSERTS(unpack[i] != UNMATCHED);
        domain = where[unpack[i]];
        processor = sub2proc[domain];
        if (processor != UNMATCHED) {
          pack[unpack[i]] = i;
        }
      }

      /*********************/
      /* Compute the flows */
      /*********************/
      if (proc2sub[mype*2] != UNMATCHED) {
        me = proc2sub[mype*2];
        you = proc2sub[mype*2+1];
        ASSERTS(me != you);

        for (j=rowptr[me]; j<rowptr[me+1]; j++) {
          if (colind[j] == you) {
            visited[j] = 1;
            scopy(ncon, transfer+j*ncon, diff_flows);
            break;
          }
        }

        for (j=rowptr[you]; j<rowptr[you+1]; j++) {
          if (colind[j] == me) {
            visited[j] = 1;
            for (h=0; h<ncon; h++)
              if (transfer[j*ncon+h] > 0.0)
                diff_flows[h] = -1.0 * transfer[j*ncon+h];
            break;
          }
        } 

        nswaps = 1;
        scopy(ncon, diff_flows, sr_flows);

        idxset(nvtxs, 0, sbuffer);
        for (i=0; i<nvtxs; i++)
          if (where[i] == me || where[i] == you)
            sbuffer[i] = 1;

        egraph = ExtractGraph(ctrl, graph, sbuffer, map, rmap);

        if (egraph != NULL) {
          idxcopy(egraph->nvtxs, egraph->where, diff_where);
          for (j=0; j<egraph->nvtxs; j++)
            ehome[j] = home[map[j]];
 
          RedoMyLink(ctrl, egraph, ehome, me, you, sr_flows, &sr_cost, &sr_lbavg);

          if (ncon <= 4) {
            sr_where = egraph->where;
            egraph->where = diff_where;

            nswaps = BalanceMyLink(ctrl, egraph, ehome, me, you, diff_flows, maxdiff, &diff_cost, &diff_lbavg, 1.0/(float)nvtxs);

            if ((sr_lbavg < diff_lbavg &&
            (diff_lbavg >= ubavg-1.0 || sr_cost == diff_cost)) ||
            (sr_lbavg < ubavg-1.0 && sr_cost < diff_cost)) {
              for (i=0; i<egraph->nvtxs; i++)
                where[map[i]] = sr_where[i];
            }
            else {
              for (i=0; i<egraph->nvtxs; i++)
                where[map[i]] = diff_where[i];
            }
          }
          else {
              for (i=0; i<egraph->nvtxs; i++)
                where[map[i]] = egraph->where[i];
          }

          GKfree((void **)&egraph->xadj, (void **)&egraph->nvwgt, (void **)&egraph->adjncy, LTERM);
          GKfree((void **)&egraph, LTERM);
        }

        /**********************/
        /* Pack the flow data */
        /**********************/
        idxset(nvtxs, UNMATCHED, sbuffer);
        for (i=0; i<nvtxs; i++) {
          domain = where[i];
          if (domain == you || domain == me) {
            sbuffer[pack[i]] = where[i];
          }
        }
      }

      /***************************/
      /* Broadcast the flow data */
      /***************************/
      MPI_Allgatherv((void *)&sbuffer[rdispl[mype]], rcount[mype], IDX_DATATYPE, (void *)rbuffer, rcount, rdispl, IDX_DATATYPE, ctrl->comm);


      /************************/
      /* Unpack the flow data */
      /************************/
      for (i=0; i<rdispl[npes]; i++) {
        if (rbuffer[i] != UNMATCHED) {
          where[unpack[i]] = rbuffer[i];
        }
      }


      /******************/
      /* Do other stuff */
      /******************/
      MPI_Allreduce((void *)visited, (void *)gvisited, matrix.nnzs,
      IDX_DATATYPE, MPI_MAX, ctrl->comm);
      nvisited = idxsum(matrix.nnzs, gvisited)/2;
      tnswaps += GlobalSESum(ctrl, nswaps);

      if (iter++ == NGD_PASSES)
        break;
    }

    /*****************************/
    /* perform serial refinement */
    /*****************************/
    Moc_ComputeSerialPartitionParams(graph, nparts, degrees);
    Moc_SerialKWayAdaptRefine(graph, nparts, home, ctrl->ubvec, 10);


    /****************************/
    /* check for early breakout */
    /****************************/
    for (h=0; h<ncon; h++) {
      lbvec[h] = (float)(nparts) *
        npwgts[samax_strd(nparts,npwgts+h,ncon)*ncon+h];
    }
    lbavg = savg(ncon, lbvec);

    done = 0;
    if (
      tnswaps == 0 ||
      lbavg >= oldlbavg ||
      lbavg <= ubavg + 0.035
    )
      done = 1;

    alldone = GlobalSEMax(ctrl, done);
    if (alldone == 1)
      break;
  }

  /*******************************************************/
  /* ensure that all subdomains have at least one vertex */
  /*******************************************************/
/*
  idxset(nparts, 0, match);
  for (i=0; i<nvtxs; i++)
    match[where[i]]++;

  done = 0;
  while (done == 0) {
    done = 1;

    me = idxamin(nparts, match);  
    if (match[me] == 0) {
if (ctrl->mype == PE) printf("WARNING: empty subdomain %d in Moc_Diffusion\n", me);
      you = idxamax(nparts, match);  
      for (i=0; i<nvtxs; i++) {
        if (where[i] == you) {
          where[i] = me;
          match[you]--;
          match[me]++;
          done = 0;
          break;
        }
      }
    }
  }
*/
 
  /******************************/
  /* now free memory and return */
  /******************************/
  GKfree((void **)&load, (void **)&proc2sub, (void **)&sub2proc, (void **)&rcount, LTERM);
  GKfree((void **)&pack, (void **)&workspace, (void **)&degrees, (void **)&rinfo, LTERM);
  GKfree((void **)&visited, LTERM);
  graph->gnpwgts = NULL;
  graph->rinfo = NULL;

  return 0;
}


/*************************************************************************
* This function extracts a subgraph from a graph given an indicator array.
**************************************************************************/
GraphType *ExtractGraph(CtrlType *ctrl, GraphType *graph, idxtype *indicator,
  idxtype *map, idxtype *rmap)
{
  int h, i, j;
  int nvtxs, envtxs, enedges, ncon;
  int vtx, count;
  idxtype *xadj, *vsize, *adjncy, *adjwgt, *where;
  idxtype *exadj, *evsize, *eadjncy, *eadjwgt, *ewhere;
  float *nvwgt, *envwgt;
  GraphType *egraph;

  nvtxs = graph->nvtxs;
  ncon = graph->ncon;
  xadj = graph->xadj;
  nvwgt = graph->nvwgt;
  vsize = graph->vsize;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;
  where = graph->where;

  count = 0;
  for (i=0; i<nvtxs; i++) {
    if (indicator[i] == 1) {
      map[count] = i;
      rmap[i] = count;
      count++;
    }
  }

  if (count == 0) {
    return NULL;
  }

  /*******************/
  /* allocate memory */
  /*******************/
  egraph = CreateGraph();
  envtxs = egraph->nvtxs = count;
  egraph->ncon = graph->ncon;

  exadj = egraph->xadj = idxmalloc(envtxs*3+1, "exadj");
  ewhere = egraph->where = exadj + envtxs + 1;
  evsize = egraph->vsize = exadj + 2*envtxs + 1;

  envwgt = egraph->nvwgt = fmalloc(envtxs*ncon, "envwgt");

  /************************************************/
  /* compute xadj, where, nvwgt, and vsize arrays */
  /************************************************/
  idxset(envtxs+1, 0, exadj);
  for (i=0; i<envtxs; i++) {
    vtx = map[i];

    ewhere[i] = where[vtx];
    for (h=0; h<ncon; h++)
      envwgt[i*ncon+h] = nvwgt[vtx*ncon+h];

    if (ctrl->partType == ADAPTIVE_PARTITION || ctrl->partType == REFINE_PARTITION) 
      evsize[i] = vsize[vtx];

    for (j=xadj[vtx]; j<xadj[vtx+1]; j++)
      if (indicator[adjncy[j]] == 1)
        exadj[i]++;

  }
  MAKECSR(i, envtxs, exadj);

  /************************************/
  /* compute adjncy and adjwgt arrays */
  /************************************/
  enedges = egraph->nedges = exadj[envtxs];
  eadjncy = egraph->adjncy = idxmalloc(enedges*2, "eadjncy");
  eadjwgt = egraph->adjwgt = eadjncy + enedges;

  for (i=0; i<envtxs; i++) {
    vtx = map[i];
    for (j=xadj[vtx]; j<xadj[vtx+1]; j++) {
      if (indicator[adjncy[j]] == 1) {
        eadjncy[exadj[i]] = rmap[adjncy[j]];
        eadjwgt[exadj[i]++] = adjwgt[j];
      }
    }
  }

  for (i=envtxs; i>0; i--)
    exadj[i] = exadj[i-1];
  exadj[0] = 0;

  return egraph;
}
