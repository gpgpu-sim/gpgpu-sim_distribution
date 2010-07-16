#include "mpi.h"
#include "fem.h"

typedef struct foob {
  int p1, k1, f1, p2, k2, f2, va, vb, g; 
}face;

int compare_pairs(const void *obj1, const void *obj2){
  
  face *e1 = (face*) obj1;
  face *e2 = (face*) obj2;
  
  int a1 = e1->va, b1 = e1->vb;
  int a2 = e2->va, b2 = e2->vb;

  int va1, vb1, va2, vb2;

  va1 = min(a1, b1);
  vb1 = max(a1, b1);

  va2 = min(a2, b2);
  vb2 = max(a2, b2);

  if(vb1<vb2)
    return -1;
  else if(vb1>vb2)
    return 1;
  else if(va1<va2)
    return -1;
  else if(va1>va2)
    return 1;
  
  return 0;
}

int pairprocget(const void *obj1){
  face *e1 = (face*) obj1;
  return (e1->p1);
}


int pairnumget(const void *obj1){
  face *e1 = (face*) obj1;
  return (e1->g);
}

void pairnumset(const void *obj1, int g){
  face *e1 = (face*) obj1;
  e1->g = g;
}

void pairmarry(const void *obj1, const void *obj2){
  
  face *e1 = (face*) obj1;
  face *e2 = (face*) obj2;
  e1->p2 = e2->p1;  e1->k2 = e2->k1;  e1->f2 = e2->f1;
  e2->p2 = e1->p1;  e2->k2 = e1->k1;  e2->f2 = e1->f1;
}

void FacePair2d(Mesh *mesh, int *maxNv){

  int procid = mesh->procid;
  int nprocs = mesh->nprocs;

  int Klocal = mesh->K;
  int Nfaces = mesh->Nfaces;
  int Nverts = mesh->Nverts;
  
  int **EToV = mesh->EToV;

  const int vnum[3][2] = { {0,1}, {1,2}, {2,0} };

  int n, k, e, sk, v;

  face *myfaces = (face*) calloc(Klocal*Nfaces, sizeof(face));

  /* find maximum local vertex number */
  int localmaxgnum = 0;
  for(k=0;k<Klocal;++k)
    for(v=0;v<Nverts;++v)
      localmaxgnum = max(localmaxgnum, EToV[k][v]);
  ++localmaxgnum;
  
  int maxgnum;
  MPI_Allreduce(&localmaxgnum, &maxgnum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  *maxNv = maxgnum;

  sk = 0;
  for(k=0;k<Klocal;++k){
    for(e=0;e<Nfaces;++e){
      int n1 = EToV[k][vnum[e][0]];
      int n2 = EToV[k][vnum[e][1]];

      myfaces[sk].p1 = procid; myfaces[sk].k1 = k; myfaces[sk].f1 = e;
      myfaces[sk].p2 = procid; myfaces[sk].k2 = k; myfaces[sk].f2 = e;

      myfaces[sk].va = max(n1,n2);
      myfaces[sk].vb = min(n1,n2);
      myfaces[sk].g  = max(n1,n2); /* marker for sorting into bins */
      ++sk;
    }
  }

  ParallelPairs(myfaces, Klocal*Nfaces, sizeof(face),
		pairnumget, pairnumset, pairprocget,
		pairmarry, compare_pairs);

  mesh->Npar = BuildIntVector(nprocs);

  mesh->EToE = BuildIntMatrix(Klocal, Nfaces);
  mesh->EToF = BuildIntMatrix(Klocal, Nfaces);
  mesh->EToP = BuildIntMatrix(Klocal, Nfaces);
  
  int id, k1, k2, f1, f2, p1, p2;
  sk = 0;

  for(n=0;n<Klocal*Nfaces;++n){

    k1 = myfaces[n].k1; f1 = myfaces[n].f1; p1 = myfaces[n].p1;
    k2 = myfaces[n].k2; f2 = myfaces[n].f2; p2 = myfaces[n].p2;
    
    if(p1!=procid)
      fprintf(stderr, "WARNING WRONG proc\n");

    mesh->EToE[k1][f1] = k2;
    mesh->EToF[k1][f1] = f2;
    mesh->EToP[k1][f1] = p2;

    if(p1!=p2){
      /* increment number of links */
      ++mesh->Npar[p2];
    }
  }

  mesh->parK = (int**) calloc(nprocs, sizeof(int*));
  mesh->parF = (int**) calloc(nprocs, sizeof(int*));
  for(p2=0;p2<nprocs;++p2){
    mesh->parK[p2] = BuildIntVector(mesh->Npar[p2]);
    mesh->parF[p2] = BuildIntVector(mesh->Npar[p2]);
    mesh->Npar[p2] = 0;
    for(n=0;n<Klocal*Nfaces;++n){
      if(myfaces[n].p2==p2 && p2!=procid){
	int k1 = myfaces[n].k1, f1 = myfaces[n].f1;
	int k2 = myfaces[n].k2, f2 = myfaces[n].f2;
	mesh->parK[p2][mesh->Npar[p2]  ] = k1;
	mesh->parF[p2][mesh->Npar[p2]++] = f1;
      }
    }
  }

  free(myfaces);
}


