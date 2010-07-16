#include "mpi.h"
#include "fem.h"

Mesh *ReadMesh3d(char *filename){

  int n;

  Mesh *mesh = (Mesh*) calloc(1, sizeof(Mesh));

  char buf[BUFSIZ];
  
  FILE *fp = fopen(filename, "r");

  /* assume modified Gambit neutral format */
  for(n=0;n<6;++n)
    fgets(buf, BUFSIZ, fp);

  fgets(buf, BUFSIZ, fp);
  sscanf(buf, "%d %d \n", &(mesh->Nv), &(mesh->K));
  mesh->Nverts = 4; /* assume tets */
  mesh->Nedges = 6; /* assume tets */
  mesh->Nfaces = 4; /* assume tets */

  fgets(buf, BUFSIZ, fp);
  fgets(buf, BUFSIZ, fp);

  /* read vertex coordinates */
  double *VX = BuildVector(mesh->Nv);
  double *VY = BuildVector(mesh->Nv);
  double *VZ = BuildVector(mesh->Nv);
  for(n=0;n<mesh->Nv;++n){
    fgets(buf, BUFSIZ, fp);
    sscanf(buf, "%*d %lf %lf %lf", VX+n, VY+n, VZ+n);
  }

  /* decide on parition */
  int procid, nprocs;

  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  mesh->procid = procid;
  mesh->nprocs = nprocs;

  /* assume this proc owns a block of elements */

  int Klocal, Kstart;
  int *Kprocs = (int*) calloc(nprocs, sizeof(int));
  int p;
  
  int **newEToV, *newKprocs;
  double **newVX, **newVY;
  
  Klocal = (int) ( (double)(mesh->K)/(double)nprocs );
  
  for(p=0;p<nprocs-1;++p){
    Kprocs[p] = Klocal;
  }
  Kprocs[p] = Klocal + mesh->K - nprocs*Klocal;
  
  
  Kstart= 0;
  for(p=0;p<procid;++p)
    Kstart += Kprocs[p];
  
  Klocal = Kprocs[procid];

  /* read element to vertex connectivity */
  fgets(buf, BUFSIZ, fp);
  fgets(buf, BUFSIZ, fp);
  mesh->EToV = BuildIntMatrix(Klocal, mesh->Nverts);
  mesh->GX = BuildMatrix(Klocal, mesh->Nverts);
  mesh->GY = BuildMatrix(Klocal, mesh->Nverts);
  mesh->GZ = BuildMatrix(Klocal, mesh->Nverts);

  int sk = 0, v;
  for(n=0;n<mesh->K;++n){
    fgets(buf, BUFSIZ, fp);
    if(n>=Kstart && n<Kstart+Klocal){
      sscanf(buf, "%*d %*d %*d %d %d %d %d", 
	     mesh->EToV[sk]+0, mesh->EToV[sk]+1,
	     mesh->EToV[sk]+2, mesh->EToV[sk]+3);
      
      /* correct to 0-index */
      --(mesh->EToV[sk][0]);
      --(mesh->EToV[sk][1]);
      --(mesh->EToV[sk][2]);
      --(mesh->EToV[sk][3]);

      for(v=0;v<mesh->Nverts;++v){
	mesh->GX[sk][v] = VX[mesh->EToV[sk][v]];
	mesh->GY[sk][v] = VY[mesh->EToV[sk][v]];
	mesh->GZ[sk][v] = VZ[mesh->EToV[sk][v]];
      }

      ++sk;
    }
  }
  fgets(buf, BUFSIZ, fp);
  fgets(buf, BUFSIZ, fp);

  mesh->K = Klocal;

  fclose(fp);

  return mesh;
  
}

void PrintMesh ( Mesh *mesh ){
  int n;
  printf("Mesh data: \n");
  printf("\n K = %d\n", mesh->K);
  printf("\n Nv = %d\n", mesh->Nv);
  printf("\n Nverts = %d\n", mesh->Nverts);
  printf("\n Node coordinates = \n");
  printf("\n Element to vertex connectivity = \n");
  for(n=0;n<mesh->K;++n){
    printf("%d: %d %d %d %d\n", n, 
	   mesh->EToV[n][0], mesh->EToV[n][1], 
	   mesh->EToV[n][2], mesh->EToV[n][3]);
  }

}

void GeometricFactors3d(Mesh *mesh, int k,
		      double *drdx, double *dsdx, double *dtdx, 
		      double *drdy, double *dsdy, double *dtdy, 
		      double *drdz, double *dsdz, double *dtdz, 
		      double *J){

  double x1 = mesh->GX[k][0], y1 =  mesh->GY[k][0], z1 =  mesh->GZ[k][0];
  double x2 = mesh->GX[k][1], y2 =  mesh->GY[k][1], z2 =  mesh->GZ[k][1];
  double x3 = mesh->GX[k][2], y3 =  mesh->GY[k][2], z3 =  mesh->GZ[k][2];
  double x4 = mesh->GX[k][3], y4 =  mesh->GY[k][3], z4 =  mesh->GZ[k][3];
  
  /* compute geometric factors of the following afine map */
  /* x = 0.5*( (-1-r-s-t)*x1 + (1+r)*x2 + (1+s)*x3 + (1+t)*x4) */
  /* y = 0.5*( (-1-r-s-t)*y1 + (1+r)*y2 + (1+s)*y3 + (1+t)*y4) */
  /* z = 0.5*( (-1-r-s-t)*z1 + (1+r)*z2 + (1+s)*z3 + (1+t)*z4) */

  double dxdr = (x2-x1)/2,  dxds = (x3-x1)/2, dxdt = (x4-x1)/2;
  double dydr = (y2-y1)/2,  dyds = (y3-y1)/2, dydt = (y4-y1)/2;
  double dzdr = (z2-z1)/2,  dzds = (z3-z1)/2, dzdt = (z4-z1)/2;

  *J = 
     dxdr*(dyds*dzdt-dzds*dydt) 
    -dydr*(dxds*dzdt-dzds*dxdt) 
    +dzdr*(dxds*dydt-dyds*dxdt);

  *drdx  =  (dyds*dzdt - dzds*dydt)/(*J);
  *drdy  = -(dxds*dzdt - dzds*dxdt)/(*J);
  *drdz  =  (dxds*dydt - dyds*dxdt)/(*J);
   	       	    	             
  *dsdx  = -(dydr*dzdt - dzdr*dydt)/(*J);
  *dsdy  =  (dxdr*dzdt - dzdr*dxdt)/(*J);
  *dsdz  = -(dxdr*dydt - dydr*dxdt)/(*J);
   	       	    	             
  *dtdx  =  (dydr*dzds - dzdr*dyds)/(*J);
  *dtdy  = -(dxdr*dzds - dzdr*dxds)/(*J);
  *dtdz  =  (dxdr*dyds - dydr*dxds)/(*J);
  
  if(*J<1e-10)
    printf("warning: J = %lg\n", *J);
  
}

void Normals3d(Mesh *mesh, int k, 
	       double *nx, double *ny, double *nz, double *sJ){
  
  int f;

  double drdx, dsdx, dtdx;
  double drdy, dsdy, dtdy;
  double drdz, dsdz, dtdz;
  double J;

  GeometricFactors3d(mesh, k,
		     &drdx, &dsdx, &dtdx, 
		     &drdy, &dsdy, &dtdy, 
		     &drdz, &dsdz, &dtdz, 
		     &J);

  nx[0] = -dtdx; nx[1] = -dsdx; nx[2] =  drdx + dsdx + dtdx; nx[3] = -drdx;
  ny[0] = -dtdy; ny[1] = -dsdy; ny[2] =  drdy + dsdy + dtdy; ny[3] = -drdy;
  nz[0] = -dtdz; nz[1] = -dsdz; nz[2] =  drdz + dsdz + dtdz; nz[3] = -drdz;
    
  for(f=0;f<4;++f){
    sJ[f] = sqrt(nx[f]*nx[f]+ny[f]*ny[f]+nz[f]*nz[f]);
    nx[f] /= sJ[f];
    ny[f] /= sJ[f];
    nz[f] /= sJ[f];
    sJ[f] *= J;
  }
}
