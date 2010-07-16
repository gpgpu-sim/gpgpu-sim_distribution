/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * main.c
 * 
 * This file contains code for testing teh adaptive partitioning routines
 *
 * Started 5/19/97
 * George
 *
 * $Id: mtest.c,v 1.3 2003/07/25 14:31:47 karypis Exp $
 *
 */

#include <parmetisbin.h>


/*************************************************************************
* Let the game begin
**************************************************************************/
int main(int argc, char *argv[])
{
  int i, mype, npes, nelms;
  idxtype *part, *eptr;
  MeshType mesh;
  MPI_Comm comm;
  int wgtflag, numflag, edgecut, nparts, options[10];
  int mgcnum = -1, mgcnums[5] = {-1, 2, 3, 4, 2}, esizes[5] = {-1, 3, 4, 8, 4};
  float *tpwgts, ubvec[MAXNCON];

  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &mype);

  if (argc < 2) {
    if (mype == 0)
      printf("Usage: %s <mesh-file> [NCommonNodes]\n", argv[0]);

    MPI_Finalize();
    exit(0);
  }

  ParallelReadMesh(&mesh, argv[1], comm); 
  mgcnum = mgcnums[mesh.etype];
  mesh.ncon = 1;

  if (argc > 2)
    mgcnum = atoi(argv[2]);

  if (mype == 0) printf("MGCNUM: %d\n", mgcnum);

  nparts = npes;
  tpwgts = fmalloc(nparts*mesh.ncon, "tpwgts");
  for (i=0; i<nparts*mesh.ncon; i++)
    tpwgts[i] = 1.0/(float)(nparts);

  for (i=0; i<mesh.ncon; i++)
    ubvec[i] = UNBALANCE_FRACTION;

  part = idxmalloc(mesh.nelms, "part");

  numflag = wgtflag = 0;
  options[0] = 1;
  options[PMV3_OPTION_DBGLVL] = 7;
  options[PMV3_OPTION_SEED] = 0;

  nelms = mesh.elmdist[mype+1]-mesh.elmdist[mype];
  eptr = idxsmalloc(nelms+1, esizes[mesh.etype], "main; eptr");
  MAKECSR(i, nelms, eptr);
  eptr[nelms]--; /* make the last element different */
  ParMETIS_V3_PartMeshKway(mesh.elmdist, eptr, mesh.elements, NULL, &wgtflag, 
              &numflag, &(mesh.ncon), &mgcnum, &nparts, tpwgts, ubvec, options, 
	      &edgecut, part, &comm);
 
/* 
  graph = ParallelMesh2Dual(&mesh, mgcnum, comm);
  MPI_Barrier(comm);

  MPI_Allreduce((void *)&(graph->nedges), (void *)&gnedges, 1, MPI_INT, MPI_SUM, comm);
  if (mype == 0)
    printf("Completed Dual Graph -- Nvtxs: %d, Nedges: %d\n", graph->gnvtxs, gnedges/2);

  numflag = wgtflag = 0;
  ParMETIS_V3_PartKway(graph->vtxdist, graph->xadj, graph->adjncy, NULL, NULL, &wgtflag,
  &numflag, &(graph->ncon), &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
  GKfree((void *)&(graph.vtxdist), (void *)&(graph.xadj), (void *)&(graph.vwgt), (void *)&(graph.adjncy), (void *)&(graph.adjwgt), LTERM);
*/ 

  GKfree((void *)&part, (void *)&tpwgts, (void *)&eptr, LTERM);
  MPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}


