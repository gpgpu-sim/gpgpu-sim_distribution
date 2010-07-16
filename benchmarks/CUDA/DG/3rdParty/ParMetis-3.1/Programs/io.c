/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * pio.c
 *
 * This file contains routines related to I/O
 *
 * Started 10/19/94
 * George
 *
 * $Id: io.c,v 1.1 2003/07/22 21:47:18 karypis Exp $
 *
 */

#include <parmetisbin.h>
#define	MAXLINE	8192

/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void ParallelReadGraph(GraphType *graph, char *filename, MPI_Comm comm)
{
  int i, k, l, pe;
  int npes, mype, ier;
  int gnvtxs, nvtxs, your_nvtxs, your_nedges, gnedges;
  int maxnvtxs = -1, maxnedges = -1;
  int readew = -1, readvw = -1, dummy, edge;
  idxtype *vtxdist, *xadj, *adjncy, *vwgt, *adjwgt;
  idxtype *your_xadj, *your_adjncy, *your_vwgt, *your_adjwgt, graphinfo[4];
  int fmt, ncon, nobj;
  MPI_Status stat;
  char *line = NULL, *oldstr, *newstr;
  FILE *fpin = NULL;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = idxsmalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL){
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    MPI_Bcast(&ier, 1, MPI_INT, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    line = (char *)GKmalloc(sizeof(char)*(MAXLINE+1), "line");

    do {
      fgets(line, MAXLINE, fpin);
    } while (line[0] == '%' && !feof(fpin));

    fmt = ncon = nobj = 0;
    sscanf(line, "%d %d %d %d %d", &gnvtxs, &gnedges, &fmt, &ncon, &nobj);
    gnedges *=2;
    readew = (fmt%10 > 0);
    readvw = ((fmt/10)%10 > 0);
    graph->ncon = ncon = (ncon == 0 ? 1 : ncon);
    graph->nobj = nobj = (nobj == 0 ? 1 : nobj);

/*    printf("Nvtxs: %d, Nedges: %d, Ncon: %d\n", gnvtxs, gnedges, ncon); */

    graphinfo[0] = ncon;
    graphinfo[1] = nobj;
    graphinfo[2] = readvw;
    graphinfo[3] = readew;
    MPI_Bcast((void *)graphinfo, 4, IDX_DATATYPE, npes-1, comm);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=gnvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }

    MPI_Bcast((void *)vtxdist, npes+1, IDX_DATATYPE, npes-1, comm);
  }
  else {
    MPI_Bcast(&ier, 1, MPI_INT, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    MPI_Bcast((void *)graphinfo, 4, IDX_DATATYPE, npes-1, comm);
    graph->ncon = ncon = graphinfo[0];
    graph->nobj = nobj = graphinfo[1];
    readvw = graphinfo[2];
    readew = graphinfo[3];

    MPI_Bcast((void *)vtxdist, npes+1, IDX_DATATYPE, npes-1, comm);
  }

  if ((ncon > 1 && !readvw) || (nobj > 1 && !readew)) {
    printf("fmt and ncon/nobj are inconsistant.  Exiting...\n");
    MPI_Finalize();
    exit(-1);
  }

  graph->gnvtxs = vtxdist[npes];
  nvtxs = graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  xadj = graph->xadj = idxmalloc(graph->nvtxs+1, "ParallelReadGraph: xadj");
  vwgt = graph->vwgt = idxmalloc(graph->nvtxs*ncon, "ParallelReadGraph: vwgt");
  /*******************************************/
  /* Go through first time and generate xadj */
  /*******************************************/
  if (mype == npes-1) {
    maxnvtxs = 0;
    for (i=0; i<npes; i++) {
      maxnvtxs = (maxnvtxs < vtxdist[i+1]-vtxdist[i]) ?
      vtxdist[i+1]-vtxdist[i] : maxnvtxs;
    }

    your_xadj = idxmalloc(maxnvtxs+1, "your_xadj");
    your_vwgt = idxmalloc(maxnvtxs*ncon, "your_vwgt");

    maxnedges = 0;
    for (pe=0; pe<npes; pe++) {
      idxset(maxnvtxs*ncon, 1, your_vwgt);
      your_nvtxs = vtxdist[pe+1]-vtxdist[pe];
      for (i=0; i<your_nvtxs; i++) {
        your_nedges = 0;

        do {
          fgets(line, MAXLINE, fpin);
        } while (line[0] == '%' && !feof(fpin));

        oldstr = line;
        newstr = NULL;

        if (readvw) {
          for (l=0; l<ncon; l++) {
            your_vwgt[i*ncon+l] = (int)strtol(oldstr, &newstr, 10);
            oldstr = newstr;
          }
        }

        for (;;) {
          edge = (int)strtol(oldstr, &newstr, 10) -1;
          oldstr = newstr;

          if (edge < 0)
            break;

          if (readew) {
            for (l=0; l<nobj; l++) {
              dummy = (int)strtol(oldstr, &newstr, 10);
              oldstr = newstr;
            }
          }
          your_nedges++;
        }
        your_xadj[i] = your_nedges;
      }

      MAKECSR(i, your_nvtxs, your_xadj);
      maxnedges = (maxnedges < your_xadj[your_nvtxs]) ?
      your_xadj[your_nvtxs] : maxnedges;

      if (pe < npes-1) {
        MPI_Send((void *)your_xadj, your_nvtxs+1, IDX_DATATYPE, pe, 0, comm);
        MPI_Send((void *)your_vwgt, your_nvtxs*ncon, IDX_DATATYPE, pe, 1, comm);
      }
      else {
        for (i=0; i<your_nvtxs+1; i++)
          xadj[i] = your_xadj[i];
        for (i=0; i<your_nvtxs*ncon; i++)
          vwgt[i] = your_vwgt[i];
      }
    }
    fclose(fpin);
    GKfree(&your_xadj, &your_vwgt, LTERM);
  }
  else {
    MPI_Recv((void *)xadj, nvtxs+1, IDX_DATATYPE, npes-1, 0, comm, &stat);
    MPI_Recv((void *)vwgt, nvtxs*ncon, IDX_DATATYPE, npes-1, 1, comm, &stat);
  }

  graph->nedges = xadj[nvtxs];
  adjncy = graph->adjncy = idxmalloc(xadj[nvtxs], "ParallelReadGraph: adjncy");
  adjwgt = graph->adjwgt = idxmalloc(xadj[nvtxs]*nobj, "ParallelReadGraph: adjwgt");
  /***********************************************/
  /* Now go through again and record adjncy data */
  /***********************************************/
  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL){
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    MPI_Bcast(&ier, 1, MPI_INT, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    /* get first line again */
    do {
      fgets(line, MAXLINE, fpin);
    } while (line[0] == '%' && !feof(fpin));

    your_adjncy = idxmalloc(maxnedges, "your_adjncy");
    your_adjwgt = idxmalloc(maxnedges*nobj, "your_adjwgt");

    for (pe=0; pe<npes; pe++) {
      your_nedges = 0;
      idxset(maxnedges*nobj, 1, your_adjwgt);
      your_nvtxs = vtxdist[pe+1]-vtxdist[pe];
      for (i=0; i<your_nvtxs; i++) {
        do {
          fgets(line, MAXLINE, fpin);
        } while (line[0] == '%' && !feof(fpin));

        oldstr = line;
        newstr = NULL;

        if (readvw) {
          for (l=0; l<ncon; l++) {
            dummy = (int)strtol(oldstr, &newstr, 10);
            oldstr = newstr;
          }
        }

        for (;;) {
          edge = (int)strtol(oldstr, &newstr, 10) -1;
          oldstr = newstr;

          if (edge < 0)
            break;

          your_adjncy[your_nedges] = edge;
          if (readew) {
            for (l=0; l<nobj; l++) {
              your_adjwgt[your_nedges*nobj+l] = (int)strtol(oldstr, &newstr, 10);
              oldstr = newstr;
            }
          }
          your_nedges++;

        }
      }
      if (pe < npes-1) {
        MPI_Send((void *)your_adjncy, your_nedges, IDX_DATATYPE, pe, 0, comm);
        MPI_Send((void *)your_adjwgt, your_nedges*nobj, IDX_DATATYPE, pe, 1, comm);
      }
      else {
        for (i=0; i<your_nedges; i++)
          adjncy[i] = your_adjncy[i];
        for (i=0; i<your_nedges*nobj; i++)
          adjwgt[i] = your_adjwgt[i];
      }
    }
    fclose(fpin);
    GKfree(&your_adjncy, &your_adjwgt, &line, LTERM);
  }
  else {
    MPI_Bcast(&ier, 1, MPI_INT, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    MPI_Recv((void *)adjncy, xadj[nvtxs], IDX_DATATYPE, npes-1, 0, comm, &stat);
    MPI_Recv((void *)adjwgt, xadj[nvtxs]*nobj, IDX_DATATYPE, npes-1, 1, comm, &stat);
  }

}



/*************************************************************************
* This function writes a distributed graph to file
**************************************************************************/
void Moc_ParallelWriteGraph(CtrlType *ctrl, GraphType *graph, char *filename,
     int nparts, int testset)
{
  int h, i, j;
  int npes, mype, penum, gnedges;
  char partfile[256];
  FILE *fpin;
  MPI_Comm comm;

  comm = ctrl->comm;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  gnedges = GlobalSESum(ctrl, graph->nedges);
  sprintf(partfile, "%s.%d.%d.%d", filename, testset, graph->ncon, nparts);

  if (mype == 0) {
    if ((fpin = fopen(partfile, "w")) == NULL)
      errexit("Failed to open file %s", partfile);

    fprintf(fpin, "%d %d %d %d %d\n", graph->gnvtxs, gnedges/2, 11, graph->ncon, 1);
    fclose(fpin);
  }

  MPI_Barrier(comm);
  for (penum=0; penum<npes; penum++) {
    if (mype == penum) {

      if ((fpin = fopen(partfile, "a")) == NULL)
        errexit("Failed to open file %s", partfile);

      for (i=0; i<graph->nvtxs; i++) {
        for (h=0; h<graph->ncon; h++)
          fprintf(fpin, "%d ", graph->vwgt[i*graph->ncon+h]);

        for (j=graph->xadj[i]; j<graph->xadj[i+1]; j++) {
          fprintf(fpin, "%d ", graph->adjncy[j]+1);
          fprintf(fpin, "%d ", graph->adjwgt[j]);
        }
      fprintf(fpin, "\n");
      }
      fclose(fpin);
    }
    MPI_Barrier(comm);
  }

  return;
}


/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void ReadTestGraph(GraphType *graph, char *filename, MPI_Comm comm)
{
  int i, k, l, npes, mype;
  int nvtxs, penum, snvtxs;
  idxtype *gxadj, *gadjncy;  
  idxtype *vtxdist, *sxadj, *ssize = NULL;
  MPI_Status status;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = idxsmalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == 0) {
    ssize = idxsmalloc(npes, 0, "ReadGraph: ssize");

    ReadMetisGraph(filename, &nvtxs, &gxadj, &gadjncy);

    printf("Nvtxs: %d, Nedges: %d\n", nvtxs, gxadj[nvtxs]);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=nvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }
  }

  MPI_Bcast((void *)vtxdist, npes+1, IDX_DATATYPE, 0, comm);

  graph->gnvtxs = vtxdist[npes];
  graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  graph->xadj = idxmalloc(graph->nvtxs+1, "ReadGraph: xadj");

  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      snvtxs = vtxdist[penum+1]-vtxdist[penum];
      sxadj = idxmalloc(snvtxs+1, "ReadGraph: sxadj");

      idxcopy(snvtxs+1, gxadj+vtxdist[penum], sxadj);
      for (i=snvtxs; i>=0; i--)
        sxadj[i] -= sxadj[0];

      ssize[penum] = gxadj[vtxdist[penum+1]] - gxadj[vtxdist[penum]];

      if (penum == mype) 
        idxcopy(snvtxs+1, sxadj, graph->xadj);
      else
        MPI_Send((void *)sxadj, snvtxs+1, IDX_DATATYPE, penum, 1, comm); 

      free(sxadj);
    }
  }
  else 
    MPI_Recv((void *)graph->xadj, graph->nvtxs+1, IDX_DATATYPE, 0, 1, comm, &status);


  graph->nedges = graph->xadj[graph->nvtxs];
  graph->adjncy = idxmalloc(graph->nedges, "ReadGraph: graph->adjncy");

  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      if (penum == mype) 
        idxcopy(ssize[penum], gadjncy+gxadj[vtxdist[penum]], graph->adjncy);
      else
        MPI_Send((void *)(gadjncy+gxadj[vtxdist[penum]]), ssize[penum], IDX_DATATYPE, penum, 1, comm); 
    }

    free(ssize);
  }
  else 
    MPI_Recv((void *)graph->adjncy, graph->nedges, IDX_DATATYPE, 0, 1, comm, &status);

  graph->vwgt = NULL;
  graph->adjwgt = NULL;

  if (mype == 0) 
    GKfree(&gxadj, &gadjncy, LTERM);

  MALLOC_CHECK(NULL);
}



/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
float *ReadTestCoordinates(GraphType *graph, char *filename, int ndims, MPI_Comm comm)
{
  int i, j, k, npes, mype, penum;
  float *xyz, *txyz;
  FILE *fpin;
  idxtype *vtxdist;
  MPI_Status status;
  char xyzfile[256];

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist;

  xyz = fmalloc(graph->nvtxs*ndims, "io");

  if (mype == 0) {
    sprintf(xyzfile, "%s.xyz", filename);
    if ((fpin = fopen(xyzfile, "r")) == NULL) 
      errexit("Failed to open file %s\n", xyzfile);
  }

  if (mype == 0) {
    txyz = fmalloc(2*graph->nvtxs*ndims, "io");

    for (penum=0; penum<npes; penum++) {
      for (k=0, i=vtxdist[penum]; i<vtxdist[penum+1]; i++, k++) {
        for (j=0; j<ndims; j++)
          fscanf(fpin, "%e ", txyz+k*ndims+j);
      }

      if (penum == mype) 
        memcpy((void *)xyz, (void *)txyz, sizeof(float)*ndims*k);
      else {
        MPI_Send((void *)txyz, ndims*k, MPI_FLOAT, penum, 1, comm); 
      }
    }
    free(txyz);
    fclose(fpin);
  }
  else 
    MPI_Recv((void *)xyz, ndims*graph->nvtxs, MPI_FLOAT, 0, 1, comm, &status);

  return xyz;
}



/*************************************************************************
* This function reads the spd matrix
**************************************************************************/
void ReadMetisGraph(char *filename, int *r_nvtxs, idxtype **r_xadj, idxtype **r_adjncy)
{
  int i, k, edge, nvtxs, nedges;
  idxtype *xadj, *adjncy;
  char *line, *oldstr, *newstr;
  FILE *fpin;

  line = (char *)malloc(sizeof(char)*(8192+1));

  if ((fpin = fopen(filename, "r")) == NULL) {
    printf("Failed to open file %s\n", filename);
    exit(0);
  }

  fgets(line, 8192, fpin);
  sscanf(line, "%d %d", &nvtxs, &nedges);
  nedges *=2;

  xadj = idxmalloc(nvtxs+1, "ReadGraph: xadj");
  adjncy = idxmalloc(nedges, "ReadGraph: adjncy");

  /* Start reading the graph file */
  for (xadj[0]=0, k=0, i=0; i<nvtxs; i++) {
    fgets(line, 8192, fpin);
    oldstr = line;
    newstr = NULL;

    for (;;) {
      edge = (int)strtol(oldstr, &newstr, 10) -1;
      oldstr = newstr;

      if (edge < 0)
        break;

      adjncy[k++] = edge;
    } 
    xadj[i+1] = k;
  }

  fclose(fpin);

  free(line);

  *r_nvtxs = nvtxs;
  *r_xadj = xadj;
  *r_adjncy = adjncy;
}


/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void Moc_SerialReadGraph(GraphType *graph, char *filename, int *wgtflag, MPI_Comm comm)
{
  int i, k, l, npes, mype;
  int nvtxs, ncon, nobj, fmt;
  int penum, snvtxs;
  idxtype *gxadj, *gadjncy, *gvwgt, *gadjwgt;  
  idxtype *vtxdist, *sxadj, *ssize = NULL;
  MPI_Status status;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = idxsmalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == 0) {
    ssize = idxsmalloc(npes, 0, "ReadGraph: ssize");

    Moc_SerialReadMetisGraph(filename, &nvtxs, &ncon, &nobj, &fmt, &gxadj, &gvwgt,
	&gadjncy, &gadjwgt, wgtflag);

    printf("Nvtxs: %d, Nedges: %d\n", nvtxs, gxadj[nvtxs]);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=nvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }
  }

  MPI_Bcast((void *)(&fmt), 1, MPI_INT, 0, comm);
  MPI_Bcast((void *)(&ncon), 1, MPI_INT, 0, comm);
  MPI_Bcast((void *)(&nobj), 1, MPI_INT, 0, comm);
  MPI_Bcast((void *)(wgtflag), 1, MPI_INT, 0, comm);
  MPI_Bcast((void *)vtxdist, npes+1, IDX_DATATYPE, 0, comm);

  graph->gnvtxs = vtxdist[npes];
  graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  graph->ncon = ncon;
  graph->xadj = idxmalloc(graph->nvtxs+1, "ReadGraph: xadj");
  /*************************************************/
  /* distribute xadj array */
  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      snvtxs = vtxdist[penum+1]-vtxdist[penum];
      sxadj = idxmalloc(snvtxs+1, "ReadGraph: sxadj");

      idxcopy(snvtxs+1, gxadj+vtxdist[penum], sxadj);
      for (i=snvtxs; i>=0; i--)
        sxadj[i] -= sxadj[0];

      ssize[penum] = gxadj[vtxdist[penum+1]] - gxadj[vtxdist[penum]];

      if (penum == mype) 
        idxcopy(snvtxs+1, sxadj, graph->xadj);
      else
        MPI_Send((void *)sxadj, snvtxs+1, IDX_DATATYPE, penum, 1, comm); 

      free(sxadj);
    }
  }
  else 
    MPI_Recv((void *)graph->xadj, graph->nvtxs+1, IDX_DATATYPE, 0, 1, comm,
		&status);



  graph->nedges = graph->xadj[graph->nvtxs];
  graph->adjncy = idxmalloc(graph->nedges, "ReadGraph: graph->adjncy");
  /*************************************************/
  /* distribute adjncy array */
  if (mype == 0) {
    for (penum=0; penum<npes; penum++) {
      if (penum == mype) 
        idxcopy(ssize[penum], gadjncy+gxadj[vtxdist[penum]], graph->adjncy);
      else
        MPI_Send((void *)(gadjncy+gxadj[vtxdist[penum]]), ssize[penum],
		IDX_DATATYPE, penum, 1, comm); 
    }

  }
  else 
    MPI_Recv((void *)graph->adjncy, graph->nedges, IDX_DATATYPE, 0, 1, comm,
		&status);


  graph->adjwgt = idxmalloc(graph->nedges*nobj, "ReadGraph: graph->adjwgt");
  if (fmt%10 > 0) {
    /*************************************************/
    /* distribute adjwgt array */
    if (mype == 0) {
      for (penum=0; penum<npes; penum++) {
        ssize[penum] *= nobj;
        if (penum == mype)
          idxcopy(ssize[penum], gadjwgt+(gxadj[vtxdist[penum]]*nobj), graph->adjwgt);
        else
          MPI_Send((void *)(gadjwgt+(gxadj[vtxdist[penum]]*nobj)), ssize[penum],
                IDX_DATATYPE, penum, 1, comm);
      }

    }
    else
      MPI_Recv((void *)graph->adjwgt, graph->nedges*nobj, IDX_DATATYPE, 0, 1,
		comm, &status);

  }
  else {
    for (i=0; i<graph->nedges*nobj; i++)
      graph->adjwgt[i] = 1;
  }

  graph->vwgt = idxmalloc(graph->nvtxs*ncon, "ReadGraph: graph->vwgt");
  if ((fmt/10)%10 > 0) {
    /*************************************************/
    /* distribute vwgt array */

    if (mype == 0) {
      for (penum=0; penum<npes; penum++) {
        ssize[penum] = (vtxdist[penum+1]-vtxdist[penum])*ncon;

        if (penum == mype) 
          idxcopy(ssize[penum], gvwgt+(vtxdist[penum]*ncon), graph->vwgt);
        else
          MPI_Send((void *)(gvwgt+(vtxdist[penum]*ncon)), ssize[penum],
		IDX_DATATYPE, penum, 1, comm);
      }

      free(ssize);
    }
    else
      MPI_Recv((void *)graph->vwgt, graph->nvtxs*ncon, IDX_DATATYPE, 0, 1,
		comm, &status);

  }
  else {
    for (i=0; i<graph->nvtxs*ncon; i++)
      graph->vwgt[i] = 1;
  }

  if (mype == 0) 
    GKfree((void *)&gxadj, (void *)&gadjncy, (void *)&gvwgt, (void *)&gadjwgt, LTERM);

  MALLOC_CHECK(NULL);
}



/*************************************************************************
* This function reads the spd matrix
**************************************************************************/
void Moc_SerialReadMetisGraph(char *filename, int *r_nvtxs, int *r_ncon, int *r_nobj,
	int *r_fmt, idxtype **r_xadj, idxtype **r_vwgt, idxtype **r_adjncy,
	idxtype **r_adjwgt, int *wgtflag)
{
  int i, k, l;
  int ncon, nobj, edge, nvtxs, nedges;
  idxtype *xadj, *adjncy, *vwgt, *adjwgt;
  char *line, *oldstr, *newstr;
  int fmt, readew, readvw;
  int ewgt[MAXNOBJ];
  FILE *fpin;

  line = (char *)GKmalloc(sizeof(char)*(8192+1), "line");

  if ((fpin = fopen(filename, "r")) == NULL) {
    printf("Failed to open file %s\n", filename);
    exit(-1);
  }

  fgets(line, 8192, fpin);
  fmt = ncon = nobj = 0;
  sscanf(line, "%d %d %d %d %d", &nvtxs, &nedges, &fmt, &ncon, &nobj);
  readew = (fmt%10 > 0);
  readvw = ((fmt/10)%10 > 0);

  *wgtflag = 0;
  if (readew)
    *wgtflag += 1;
  if (readvw)
    *wgtflag += 2;

  if ((ncon > 0 && !readvw) || (nobj > 0 && !readew)) {
    printf("fmt and ncon/nobj are inconsistant.\n");
    exit(-1);
  }

  nedges *=2;
  ncon = (ncon == 0 ? 1 : ncon);
  nobj = (nobj == 0 ? 1 : nobj);

  xadj = idxmalloc(nvtxs+1, "ReadGraph: xadj");
  adjncy = idxmalloc(nedges, "Moc_ReadGraph: adjncy");
  vwgt = (readvw ? idxmalloc(ncon*nvtxs, "RG: vwgt") : NULL);
  adjwgt = (readew ? idxmalloc(nobj*nedges, "RG: adjwgt") : NULL);

  /* Start reading the graph file */
  for (xadj[0]=0, k=0, i=0; i<nvtxs; i++) {
    do {
      fgets(line, 8192, fpin);
    } while (line[0] == '%' && !feof(fpin));
    oldstr = line;
    newstr = NULL;

    if (readvw) {
      for (l=0; l<ncon; l++) {
        vwgt[i*ncon+l] = (int)strtol(oldstr, &newstr, 10);
        oldstr = newstr;
      }
    }

    for (;;) {
      edge = (int)strtol(oldstr, &newstr, 10) -1;
      oldstr = newstr;

      if (readew) {
        for (l=0; l<nobj; l++) {
          ewgt[l] = (float)strtod(oldstr, &newstr);
          oldstr = newstr;
        }
      }

      if (edge < 0)
        break;

      adjncy[k] = edge;
      if (readew)
        for (l=0; l<nobj; l++)
          adjwgt[k*nobj+l] = ewgt[l];
      k++;
    }
    xadj[i+1] = k;
  }

  fclose(fpin);

  free(line);

  *r_nvtxs = nvtxs;
  *r_ncon = ncon;
  *r_nobj = nobj;
  *r_fmt = fmt;
  *r_xadj = xadj;
  *r_vwgt = vwgt;
  *r_adjncy = adjncy;
  *r_adjwgt = adjwgt;
}




/*************************************************************************
* This function writes out a partition vector
**************************************************************************/
void WritePVector(char *gname, idxtype *vtxdist, idxtype *part, MPI_Comm comm)
{
  int i, j, k, l, rnvtxs, npes, mype, penum;
  FILE *fpin;
  idxtype *rpart;
  char partfile[256];
  MPI_Status status;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  if (mype == 0) {
    sprintf(partfile, "%s.part", gname);
    if ((fpin = fopen(partfile, "w")) == NULL) 
      errexit("Failed to open file %s", partfile);

    for (i=0; i<vtxdist[1]; i++)
      fprintf(fpin, "%d\n", part[i]);

    for (penum=1; penum<npes; penum++) {
      rnvtxs = vtxdist[penum+1]-vtxdist[penum];
      rpart = idxmalloc(rnvtxs, "rpart");
      MPI_Recv((void *)rpart, rnvtxs, IDX_DATATYPE, penum, 1, comm, &status);

      for (i=0; i<rnvtxs; i++)
        fprintf(fpin, "%d\n", rpart[i]);

      free(rpart);
    }
    fclose(fpin);
  }
  else
    MPI_Send((void *)part, vtxdist[mype+1]-vtxdist[mype], IDX_DATATYPE, 0, 1, comm); 

}


/*************************************************************************
* This function reads a mesh from a file
**************************************************************************/
void ParallelReadMesh(MeshType *mesh, char *filename, MPI_Comm comm)
{
  int i, j, k, pe;
  int npes, mype, ier;
  int gnelms, nelms, your_nelms, etype, maxnelms;
  int maxnode, gmaxnode, minnode, gminnode;
  idxtype *elmdist, *elements;
  idxtype *your_elements;
  MPI_Status stat;
  char *line = NULL, *oldstr, *newstr;
  FILE *fpin = NULL;
  int esize, esizes[5] = {-1, 3, 4, 8, 4};
  int mgcnum, mgcnums[5] = {-1, 2, 3, 4, 2};

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  elmdist = mesh->elmdist = idxsmalloc(npes+1, 0, "ReadGraph: elmdist");

  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL){
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    MPI_Bcast(&ier, 1, MPI_INT, npes-1, comm);
    if (ier > 0){
      fclose(fpin);
      MPI_Finalize();
      exit(0);
    }

    line = (char *)GKmalloc(sizeof(char)*(MAXLINE+1), "line");

    fgets(line, MAXLINE, fpin);
    sscanf(line, "%d %d", &gnelms, &etype);

    /* Construct elmdist and send it to all the processors */
    elmdist[0] = 0;
    for (i=0,j=gnelms; i<npes; i++) {
      k = j/(npes-i);
      elmdist[i+1] = elmdist[i]+k;
      j -= k;
    }

    MPI_Bcast((void *)elmdist, npes+1, IDX_DATATYPE, npes-1, comm);
  }
  else {
    MPI_Bcast(&ier, 1, MPI_INT, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    MPI_Bcast((void *)elmdist, npes+1, IDX_DATATYPE, npes-1, comm);
  }

  MPI_Bcast((void *)(&etype), 1, MPI_INT, npes-1, comm);

  gnelms = mesh->gnelms = elmdist[npes];
  nelms = mesh->nelms = elmdist[mype+1]-elmdist[mype];
  mesh->etype = etype;
  esize = esizes[etype];
  mgcnum = mgcnums[etype];

  elements = mesh->elements = idxmalloc(nelms*esize, "ParallelReadMesh: elements");

  if (mype == npes-1) {
    maxnelms = 0;
    for (i=0; i<npes; i++) {
      maxnelms = (maxnelms > elmdist[i+1]-elmdist[i]) ?
      maxnelms : elmdist[i+1]-elmdist[i];
    }

    your_elements = idxmalloc(maxnelms*esize, "your_elements");

    for (pe=0; pe<npes; pe++) {
      your_nelms = elmdist[pe+1]-elmdist[pe];
      for (i=0; i<your_nelms; i++) {

        fgets(line, MAXLINE, fpin);
        oldstr = line;
        newstr = NULL;

        /*************************************/
        /* could get element weigts here too */
        /*************************************/

        for (j=0; j<esize; j++) {
          your_elements[i*esize+j] = (int)strtol(oldstr, &newstr, 10);
          oldstr = newstr;
        }
      }

      if (pe < npes-1) {
        MPI_Send((void *)your_elements, your_nelms*esize, IDX_DATATYPE, pe, 0, comm);
      }
      else {
        for (i=0; i<your_nelms*esize; i++)
          elements[i] = your_elements[i];
      }
    }
    fclose(fpin);
    free(your_elements);
  }
  else {
    MPI_Recv((void *)elements, nelms*esize, IDX_DATATYPE, npes-1, 0, comm, &stat);
  }

  /*********************************/
  /* now check for number of nodes */
  /*********************************/
  minnode = elements[idxamin(nelms*esize, elements)];
  MPI_Allreduce((void *)&minnode, (void *)&gminnode, 1, MPI_INT, MPI_MIN, comm);
  for (i=0; i<nelms*esize; i++)
    elements[i] -= gminnode;

  maxnode = elements[idxamax(nelms*esize, elements)];
  MPI_Allreduce((void *)&maxnode, (void *)&gmaxnode, 1, MPI_INT, MPI_MAX, comm);
  mesh->gnns = gmaxnode+1;

  if (mype==0) printf("Nelements: %d, Nnodes: %d, EType: %d\n", gnelms, mesh->gnns, etype);
}


