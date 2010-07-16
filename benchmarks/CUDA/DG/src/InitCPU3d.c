#include "mpi.h"
#include "fem.h"

double InitCPU3d(Mesh *mesh, int Nfields){

  printf("Np = %d, BSIZE = %d\n", p_Np, BSIZE);

  /* Q  */
  int sz = mesh->K*(p_Np)*Nfields*sizeof(float);  /* TW BLOCK */

  mesh->f_Q    = (float*) calloc(mesh->K*p_Np*Nfields, sizeof(float));
  mesh->f_rhsQ = (float*) calloc(mesh->K*p_Np*Nfields, sizeof(float));
  mesh->f_resQ = (float*) calloc(mesh->K*p_Np*Nfields, sizeof(float));

  /*  float LIFT  */
  sz = p_Np*(p_Nfp)*(p_Nfaces)*sizeof(float);
  mesh->f_LIFT = (float*) malloc(sz);
  int sk = 0, n, m, f, k;

  for(n=0;n<p_Np;++n){
    for(m=0;m<p_Nfp*p_Nfaces;++m){    
      mesh->f_LIFT[sk++] = mesh->LIFT[n][m];
    }
  }

  /*  float Dr & Ds */
  sz = p_Np*p_Np*sizeof(float);
  mesh->f_Dr = (float*) malloc(sz);
  mesh->f_Ds = (float*) malloc(sz);
  mesh->f_Dt = (float*) malloc(sz);

  sk = 0;
  for(n=0;n<p_Np;++n){
    for(m=0;m<p_Np;++m){    
      mesh->f_Dr[sk] = mesh->Dr[n][m];
      mesh->f_Ds[sk] = mesh->Ds[n][m];
      mesh->f_Dt[sk] = mesh->Dt[n][m];
      ++sk;
    }
  }

  /* vgeo */
  double drdx, dsdx, dtdx;
  double drdy, dsdy, dtdy;
  double drdz, dsdz, dtdz, J;
  mesh->vgeo = (float*) calloc(12*mesh->K, sizeof(float));
  
  for(k=0;k<mesh->K;++k){
    GeometricFactors3d(mesh, k, 
		       &drdx, &dsdx, &dtdx,
		       &drdy, &dsdy, &dtdy,
		       &drdz, &dsdz, &dtdz, &J);
    
    mesh->vgeo[k*12+0] = drdx; mesh->vgeo[k*12+1] = drdy; mesh->vgeo[k*12+2] = drdz;
    mesh->vgeo[k*12+4] = dsdx; mesh->vgeo[k*12+5] = dsdy; mesh->vgeo[k*12+6] = dsdz;
    mesh->vgeo[k*12+8] = dtdx; mesh->vgeo[k*12+9] = dtdy; mesh->vgeo[k*12+10] = dtdz;
  }
  
  /* surfinfo (vmapM, vmapP, Fscale, Bscale, nx, ny, nz, 0) */
  sz = mesh->K*p_Nfp*p_Nfaces*7*sizeof(float); 
  
  mesh->surfinfo = (float*) malloc(sz); 
  
  /* local-local info */
  sk = 0;
  int skP = -1;
  double *nxk = BuildVector(mesh->Nfaces);
  double *nyk = BuildVector(mesh->Nfaces);
  double *nzk = BuildVector(mesh->Nfaces);
  double *sJk = BuildVector(mesh->Nfaces);

  double dt = 1e6;

  sk = 0;
  for(k=0;k<mesh->K;++k){
    
    GeometricFactors3d(mesh, k, 
		       &drdx, &dsdx, &dtdx,
		       &drdy, &dsdy, &dtdy,
		       &drdz, &dsdz, &dtdz, &J);
    
    Normals3d(mesh, k, nxk, nyk, nzk, sJk);
    
    for(f=0;f<mesh->Nfaces;++f){

      dt = min(dt, J/sJk[f]);
      
      for(m=0;m<p_Nfp;++m){
	int id = m + f*p_Nfp + p_Nfp*p_Nfaces*k;
	int idM = mesh->vmapM[id];
	int idP = mesh->vmapP[id];
	int  nM = idM%p_Np; 
	int  nP = idP%p_Np; 
	int  kM = (idM-nM)/p_Np;
	int  kP = (idP-nP)/p_Np;
	idM = Nfields*(nM + p_Np*kM);
	idP = Nfields*(nP + p_Np*kP);
	
	/* stub resolve some other way */
	if(mesh->vmapP[id]<0){
	  idP = mesh->vmapP[id]; /* -ve numbers */
	}
	
	mesh->surfinfo[sk++] = idM;
	mesh->surfinfo[sk++] = idP;
	mesh->surfinfo[sk++] = sJk[f]/(2.*J);
	mesh->surfinfo[sk++] = (idM==idP)?-1.:1.;
	mesh->surfinfo[sk++] = nxk[f];
	mesh->surfinfo[sk++] = nyk[f];
	mesh->surfinfo[sk++] = nzk[f];
      }
    }
  }
}

void cpu_set_data3d(Mesh *mesh, double *Hx, double *Hy, double *Hz, 
		    double *Ex, double *Ey, double *Ez){

  const int K = mesh->K;
  int k, n, sk=0;
  
  for(k=0;k<K;++k){
    for(n=0;n<p_Np;++n){
      mesh->f_Q[sk++] = Hx[n+k*p_Np];
      mesh->f_Q[sk++] = Hy[n+k*p_Np];
      mesh->f_Q[sk++] = Hz[n+k*p_Np];
      mesh->f_Q[sk++] = Ex[n+k*p_Np];
      mesh->f_Q[sk++] = Ey[n+k*p_Np];
      mesh->f_Q[sk++] = Ez[n+k*p_Np];
    }
  }

}


void cpu_get_data3d(Mesh *mesh, double *Hx, double *Hy, double *Hz,
		                double *Ex, double *Ey, double *Ez){
  const int K = mesh->K;
  int k, n, sk=0;
  
  for(k=0;k<K;++k){
    for(n=0;n<p_Np;++n){
      Hx[n+k*p_Np] = mesh->f_Q[sk++];
      Hy[n+k*p_Np] = mesh->f_Q[sk++];
      Hz[n+k*p_Np] = mesh->f_Q[sk++];
      Ex[n+k*p_Np] = mesh->f_Q[sk++];
      Ey[n+k*p_Np] = mesh->f_Q[sk++];
      Ez[n+k*p_Np] = mesh->f_Q[sk++];
    }
  }
  
}
