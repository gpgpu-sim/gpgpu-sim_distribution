#include "mpi.h"
#include "fem.h"

typedef struct foob {
  int p1, k1, f1, p2, k2, f2;
  int va, vb, vc, g; 
}face3d;

int compare_pairs3d(const void *obj1, const void *obj2){
  
  face3d *e1 = (face3d*) obj1;
  face3d *e2 = (face3d*) obj2;
  
  int a1 = e1->va, b1 = e1->vb, c1 = e1->vc;
  int a2 = e2->va, b2 = e2->vb, c2 = e2->vc;

  int va1, vb1, vc1, va2, vb2, vc2;

  va1 = min(a1, min(b1, c1));
  vc1 = max(a1, max(b1, c1));

  if(va1!=a1 && vc1!=a1) vb1=a1;
  if(va1!=b1 && vc1!=b1) vb1=b1;
  if(va1!=c1 && vc1!=c1) vb1=c1;

  va2 = min(a2, min(b2, c2));
  vc2 = max(a2, max(b2, c2));

  if(va2!=a2 && vc2!=a2) vb2=a2;
  if(va2!=b2 && vc2!=b2) vb2=b2;
  if(va2!=c2 && vc2!=c2) vb2=c2;

  if(vc1<vc2)
    return -1;
  else if(vc1>vc2)
    return 1;
  else if(vb1<vb2)
    return -1;
  else if(vb1>vb2)
    return 1;
  else if(va1<va2)
    return -1;
  else if(va1>va2)
    return 1;
  
  return 0;

}


int pairprocget3d(const void *obj1){
  face3d *e1 = (face3d*) obj1;
  return (e1->p1);
}


int pairnumget3d(const void *obj1){
  face3d *e1 = (face3d*) obj1;
  return (e1->g);
}

void pairnumset3d(const void *obj1, int g){
  face3d *e1 = (face3d*) obj1;
  e1->g = g;
}

void pairmarry3d(const void *obj1, const void *obj2){
  
  face3d *e1 = (face3d*) obj1;
  face3d *e2 = (face3d*) obj2;
  e1->p2 = e2->p1;  e1->k2 = e2->k1;  e1->f2 = e2->f1;
  e2->p2 = e1->p1;  e2->k2 = e1->k1;  e2->f2 = e1->f1;
}

void FacePair3d(Mesh *mesh, int *maxNv){

  int procid = mesh->procid;
  int nprocs = mesh->nprocs;

  int Klocal = mesh->K;
  int Nfaces = mesh->Nfaces;
  int Nverts = mesh->Nverts;
  
  int **EToV = mesh->EToV;

  const int vnum[4][3] = { {0,1,2}, {0,1,3}, {1,2,3}, {0,2,3} };

  int n, k, e, sk, v;

  face3d *myfaces = (face3d*) calloc(Klocal*Nfaces, sizeof(face3d));

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
      int a1 = EToV[k][vnum[e][0]];
      int b1 = EToV[k][vnum[e][1]];
      int c1 = EToV[k][vnum[e][2]];

      myfaces[sk].p1 = procid; myfaces[sk].k1 = k; myfaces[sk].f1 = e;
      myfaces[sk].p2 = procid; myfaces[sk].k2 = k; myfaces[sk].f2 = e;

      int va1, vb1, vc1, va2, vb2, vc2;
      
      va1 = min(a1, min(b1, c1));
      vc1 = max(a1, max(b1, c1));
      
      if(va1!=a1 && vc1!=a1) vb1=a1;
      if(va1!=b1 && vc1!=b1) vb1=b1;
      if(va1!=c1 && vc1!=c1) vb1=c1;
      
      myfaces[sk].va = va1;
      myfaces[sk].vb = vb1;
      myfaces[sk].vc = vc1;
      myfaces[sk].g  = max(va1,max(vb1,vc1)); /* marker for sorting into bins */
      ++sk;
    }
  }

  ParallelPairs(myfaces, Klocal*Nfaces, sizeof(face3d),
		pairnumget3d, pairnumset3d, pairprocget3d,
		pairmarry3d, compare_pairs3d);

  mesh->Npar = BuildIntVector(nprocs);

  mesh->EToE = BuildIntMatrix(Klocal, Nfaces);
  mesh->EToF = BuildIntMatrix(Klocal, Nfaces);
  mesh->EToP = BuildIntMatrix(Klocal, Nfaces);
  
  int id, k1, k2, f1, f2, p1, p2;
  sk = 0;

  for(n=0;n<Klocal*Nfaces;++n){

    k1 = myfaces[n].k1; f1 = myfaces[n].f1; p1 = myfaces[n].p1;
    k2 = myfaces[n].k2; f2 = myfaces[n].f2; p2 = myfaces[n].p2;
    
    if(p1!=procid){
      fprintf(stderr, "WARNING WRONG proc\n");
      exit(-1);
    }

    mesh->EToE[k1][f1] = k2;
    mesh->EToF[k1][f1] = f2;
    mesh->EToP[k1][f1] = p2;

    if(p1!=p2){
      /* increment number of links */
      ++mesh->Npar[p2];
    }
  }

#if 0
  char fname[BUFSIZ];
  sprintf(fname, "proc%d.dat", mesh->procid);
  FILE *fp = fopen(fname, "w");
  for(k1=0;k1<mesh->K;++k1){
    for(f1=0;f1<mesh->Nfaces;++f1){
      fprintf(fp, "p: (%d,%d,%d)->(%d,%d,%d)\n",
	     k1,f1,mesh->procid, 
	     mesh->EToE[k1][f1],
	     mesh->EToF[k1][f1],
	     mesh->EToP[k1][f1]);

    }
  }
  fclose(fp);
#endif

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
    printf("proc: %d sends %d to proc: %d\n",
	   mesh->procid, mesh->Npar[p2], p2);
  }

  free(myfaces);
}


