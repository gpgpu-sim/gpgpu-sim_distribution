/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * msetup.c
 *
 * This file contain various routines for setting up a mesh
 *
 * Started 10/19/96
 * George
 *
 * $Id: msetup.c,v 1.3 2003/07/31 06:14:01 karypis Exp $
 *
 */

#include <parmetis.h>



/*************************************************************************
* This function setsup the CtrlType structure
**************************************************************************/
MeshType *SetUpMesh(int *etype, int *ncon, idxtype *elmdist, idxtype *elements,
  idxtype *elmwgt, int *wgtflag, MPI_Comm *comm)
{
  MeshType *mesh;
  int i, npes, mype;
  int esizes[5] = {-1, 3, 4, 8, 4};
  int maxnode, gmaxnode, minnode, gminnode;

  MPI_Comm_size(*comm, &npes);
  MPI_Comm_rank(*comm, &mype);

  mesh = CreateMesh();
  mesh->elmdist = elmdist;
  mesh->gnelms = elmdist[npes];
  mesh->nelms = elmdist[mype+1]-elmdist[mype];
  mesh->elements = elements;
  mesh->elmwgt = elmwgt;
  mesh->etype = *etype;
  mesh->ncon = *ncon;
  mesh->esize = esizes[*etype];

  if (((*wgtflag)&1) == 0) {
    mesh->elmwgt = idxsmalloc(mesh->nelms*mesh->ncon, 1, "SetUpMesh: elmwgt");
  }

  minnode = elements[idxamin(mesh->nelms*mesh->esize, elements)];
  MPI_Allreduce((void *)&minnode, (void *)&gminnode, 1, MPI_INT, MPI_MIN, *comm);
  for (i=0; i<mesh->nelms*mesh->esize; i++)
    elements[i] -= gminnode;
  mesh->gminnode = gminnode;

  maxnode = elements[idxamax(mesh->nelms*mesh->esize, elements)];
  MPI_Allreduce((void *)&maxnode, (void *)&gmaxnode, 1, MPI_INT, MPI_MAX, *comm);
  mesh->gnns = gmaxnode+1;

  return mesh;
}

/*************************************************************************
* This function creates a MeshType data structure and initializes
* the various fields
**************************************************************************/
MeshType *CreateMesh(void)
{
  MeshType *mesh;

  mesh = (MeshType *)GKmalloc(sizeof(MeshType), "CreateMesh: mesh");

  InitMesh(mesh);

  return mesh;
}

/*************************************************************************
* This function initializes the various fields of a MeshType.
**************************************************************************/
void InitMesh(MeshType *mesh)
{

  mesh->etype = -1;
  mesh->gnelms = -1;
  mesh->gnns = -1;
  mesh->nelms = -1;
  mesh->nns = -1;
  mesh->ncon = -1;
  mesh->esize = -1;
  mesh->gminnode = 0;
  mesh->elmdist = NULL;
  mesh->elements = NULL;
  mesh->elmwgt = NULL;

  return;
}

