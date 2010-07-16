/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * wave.c 
 *
 * This file contains code for directed diffusion at the coarsest graph
 *
 * Started 5/19/97, Kirk, George
 *
 * $Id: wave.c,v 1.3 2003/07/22 21:47:18 karypis Exp $
 *
 */

#include <parmetislib.h>

/*************************************************************************
* This function performs a k-way directed diffusion
**************************************************************************/
float WavefrontDiffusion(CtrlType *ctrl, GraphType *graph, idxtype *home)
{
  int ii, i, j, k, l, nvtxs, nedges, nparts;
  int from, to, edge, done, nswaps, noswaps, totalv, wsize;
  int npasses, first, second, third, mind, maxd;
  idxtype *xadj, *adjncy, *adjwgt, *where, *perm;
  idxtype *rowptr, *colind, *ed, *psize;
  float *transfer, *tmpvec;
  float balance = -1.0, *load, *solution, *workspace;
  float *nvwgt, *npwgts, flowFactor, cost, ubfactor;
  MatrixType matrix;
  KeyValueType *cand;
  int ndirty, nclean, dptr, clean;

  nvtxs        = graph->nvtxs;
  nedges       = graph->nedges;
  xadj         = graph->xadj;
  nvwgt        = graph->nvwgt;
  adjncy       = graph->adjncy;
  adjwgt       = graph->adjwgt;
  where        = graph->where;
  nparts       = ctrl->nparts;
  ubfactor     = ctrl->ubvec[0];
  matrix.nrows = nparts;

  flowFactor = 0.35;
  flowFactor = (ctrl->mype == 2) ? 0.50 : flowFactor;
  flowFactor = (ctrl->mype == 3) ? 0.75 : flowFactor;
  flowFactor = (ctrl->mype == 4) ? 1.00 : flowFactor;

  /* allocate memory */
  solution                   = fmalloc(4*nparts+2*nedges, "WavefrontDiffusion: solution");
  tmpvec                     = solution + nparts;
  npwgts                     = solution + 2*nparts;
  load                       = solution + 3*nparts;
  matrix.values              = solution + 4*nparts;
  transfer = matrix.transfer = solution + 4*nparts + nedges;

  perm                   = idxmalloc(2*nvtxs+2*nparts+nedges+1, "WavefrontDiffusion: perm");
  ed                     = perm + nvtxs;
  psize                  = perm + 2*nvtxs;
  rowptr = matrix.rowptr = perm + 2*nvtxs + nparts;
  colind = matrix.colind = perm + 2*nvtxs + 2*nparts + 1;

  wsize     = amax(sizeof(float)*nparts*6, sizeof(idxtype)*(nvtxs+nparts*2+1));
  workspace = (float *)GKmalloc(wsize, "WavefrontDiffusion: workspace");
  cand      = (KeyValueType *)GKmalloc(nvtxs*sizeof(KeyValueType), "WavefrontDiffusion: cand");


  /*****************************/
  /* Populate empty subdomains */
  /*****************************/
  idxset(nparts, 0, psize);
  for (i=0; i<nvtxs; i++) 
    psize[where[i]]++;

  mind = idxamin(nparts, psize);
  maxd = idxamax(nparts, psize);
  if (psize[mind] == 0) {
    for (i=0; i<nvtxs; i++) {
      k = (RandomInRange(nvtxs)+i)%nvtxs; 
      if (where[k] == maxd) {
        where[k] = mind;
        psize[mind]++;
        psize[maxd]--;
        break;
      }
    }
  }
  idxset(nvtxs, 0, ed);
  sset(nparts, 0.0, npwgts);
  for (i=0; i<nvtxs; i++) {
    npwgts[where[i]] += nvwgt[i];
    for (j=xadj[i]; j<xadj[i+1]; j++)
      ed[i] += (where[i] != where[adjncy[j]] ? adjwgt[j] : 0);
  }

  ComputeLoad(graph, nparts, load, ctrl->tpwgts, 0);
  done = 0;

  npasses = amin(nparts/2, NGD_PASSES);
  for (l=0; l<npasses; l++) {
    /* Set-up and solve the diffusion equation */
    nswaps = 0;

    /************************/
    /* Solve flow equations */
    /************************/
    SetUpConnectGraph(graph, &matrix, (idxtype *)workspace);

    /* check for disconnected subdomains */
    for(i=0; i<matrix.nrows; i++) {
      if (matrix.rowptr[i]+1 == matrix.rowptr[i+1]) {
        cost = (float)(ctrl->mype); 
	goto CleanUpAndExit;
      }
    }

    ConjGrad2(&matrix, load, solution, 0.001, workspace);
    ComputeTransferVector(1, &matrix, solution, transfer, 0);

    GetThreeMax(nparts, load, &first, &second, &third);

    if (l%3 == 0) {
      FastRandomPermute(nvtxs, perm, 1);
    }
    else {
      /*****************************/
      /* move dirty vertices first */
      /*****************************/
      ndirty = 0;
      for (i=0; i<nvtxs; i++)
        if (where[i] != home[i])
          ndirty++;

      dptr = 0;
      for (i=0; i<nvtxs; i++)
        if (where[i] != home[i])
          perm[dptr++] = i;
        else
          perm[ndirty++] = i;

      ASSERT(ctrl, ndirty == nvtxs);
      ndirty = dptr;
      nclean = nvtxs-dptr;
      FastRandomPermute(ndirty, perm, 0);
      FastRandomPermute(nclean, perm+ndirty, 0);
    }

    if (ctrl->mype == 0) {
      for (j=nvtxs, k=0, ii=0; ii<nvtxs; ii++) {
        i = perm[ii];
        if (ed[i] != 0) {
          cand[k].key = -ed[i];
          cand[k++].val = i;
        }
        else {
          cand[--j].key = 0;
          cand[j].val = i;
        }
      }
      ikeysort(k, cand);
    }

    for (ii=0; ii<nvtxs/3; ii++) {
      i = (ctrl->mype == 0) ? cand[ii].val : perm[ii];
      from = where[i];

      /* don't move out the last vertex in a subdomain */
      if (psize[from] == 1)
        continue;

      clean = (from == home[i]) ? 1 : 0;

      /* only move from top three or dirty vertices */
      if (from != first && from != second && from != third && clean)
        continue;

      /* Scatter the sparse transfer row into the dense tmpvec row */
      for (j=rowptr[from]+1; j<rowptr[from+1]; j++)
        tmpvec[colind[j]] = transfer[j];

      for (j=xadj[i]; j<xadj[i+1]; j++) {
        to = where[adjncy[j]];
        if (from != to) {
          if (tmpvec[to] > (flowFactor * nvwgt[i])) {
            tmpvec[to] -= nvwgt[i];
            INC_DEC(psize[to], psize[from], 1);
            INC_DEC(npwgts[to], npwgts[from], nvwgt[i]);
            INC_DEC(load[to], load[from], nvwgt[i]);
            where[i] = to;
            nswaps++;

            /* Update external degrees */
            ed[i] = 0;
            for (k=xadj[i]; k<xadj[i+1]; k++) {
              edge = adjncy[k];
              ed[i] += (to != where[edge] ? adjwgt[k] : 0);

              if (where[edge] == from)
                ed[edge] += adjwgt[k];
              if (where[edge] == to)
                ed[edge] -= adjwgt[k];
            }
            break;
          }
        }
      }

      /* Gather the dense tmpvec row into the sparse transfer row */
      for (j=rowptr[from]+1; j<rowptr[from+1]; j++) {
        transfer[j] = tmpvec[colind[j]];
        tmpvec[colind[j]] = 0.0;
      }
      ASSERTS(fabs(ssum(nparts, tmpvec)) < .0001)
    }

    if (l % 2 == 1) {
      balance = npwgts[samax(nparts, npwgts)] * (float)nparts;
      if (balance < ubfactor + 0.035)
        done = 1;

      if (GlobalSESum(ctrl, done) > 0)
        break;

      noswaps = (nswaps > 0) ? 0 : 1;
      if (GlobalSESum(ctrl, noswaps) > ctrl->npes/2)
        break;

    }
  }

  graph->mincut = ComputeSerialEdgeCut(graph);
  totalv        = Mc_ComputeSerialTotalV(graph, home);
  cost          = ctrl->ipc_factor * (float)graph->mincut + ctrl->redist_factor * (float)totalv;


CleanUpAndExit:
  GKfree((void **)&solution, (void **)&perm, (void **)&workspace, (void **)&cand, LTERM);

  return cost;
}

