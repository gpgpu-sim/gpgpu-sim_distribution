#include "mpi.h"
#include "fem.h"

void MaxwellsRHS3d(Mesh *mesh, float frka, float frkb, float fdt){

  /* registers and temporary */
  register unsigned int k, n;

  /* mesh parameters */
  const int K = mesh->K;

  float *vgeo     = mesh->vgeo;
  float *surfinfo = mesh->surfinfo;
  float *f_Dr    = mesh->f_Dr;
  float *f_Ds    = mesh->f_Ds;
  float *f_Dt    = mesh->f_Dt;
  float *f_LIFT  = mesh->f_LIFT;

  float *f_Q    = mesh->f_Q;
  float *f_rhsQ = mesh->f_rhsQ;
  float *f_resQ = mesh->f_resQ;

  float *f_inQ  = mesh->f_inQ;
  float *f_outQ  = mesh->f_outQ;

  int p;
  
  /* mpi request buffer */
  MPI_Request *mpi_out_requests = (MPI_Request*) calloc(mesh->nprocs, sizeof(MPI_Request));
  MPI_Request *mpi_in_requests  = (MPI_Request*) calloc(mesh->nprocs, sizeof(MPI_Request));

  /* buffer outgoing node data */
  for(n=0;n<mesh->parNtotalout;++n)
    mesh->f_outQ[n] = f_Q[mesh->parmapOUT[n]];

  /* do sends */
  int sk = 0, Nmess = 0;
  for(p=0;p<mesh->nprocs;++p){
    if(p!=mesh->procid){
      int Nout = mesh->Npar[p]*p_Nfields*p_Nfp;
      if(Nout){
	/* symmetric communications (different ordering) */
	MPI_Isend(f_outQ+sk, Nout, MPI_FLOAT, p, 6666+p,            MPI_COMM_WORLD, mpi_out_requests +Nmess);
	MPI_Irecv(f_inQ+sk,  Nout, MPI_FLOAT, p, 6666+mesh->procid, MPI_COMM_WORLD,  mpi_in_requests +Nmess);
	sk+=Nout;
	++Nmess;
      }
    }
  }

  for(k=0;k<K;++k){

    /* NOTE: once k is known, all other indexing variables etc are derived */
    register unsigned int n, m;

    /* NOTE: should be local memory */
    float    Qk[p_Np*p_Nfields];

    /* NOTE: index into geometric factors */
    int geoid=k*12;

    const float drdx = vgeo[geoid++], drdy = vgeo[geoid++], drdz = vgeo[geoid++]; ++geoid;
    const float dsdx = vgeo[geoid++], dsdy = vgeo[geoid++], dsdz = vgeo[geoid++]; ++geoid;
    const float dtdx = vgeo[geoid++], dtdy = vgeo[geoid++], dtdz = vgeo[geoid++]; ++geoid;
    
    int id = k*p_Nfp*p_Nfaces;

    /* NOTE: buffer element k into local storage */
    float *qpt = f_Q+p_Nfields*p_Np*k;
    for(m=0;m<p_Nfields*p_Np;++m){
      Qk[m] = qpt[m];
    }

    int sk = 0;

    for(n=0;n<p_Np;++n){

      const float *ptDr = f_Dr+n*p_Np;
      const float *ptDs = f_Ds+n*p_Np;
      const float *ptDt = f_Dt+n*p_Np;

      int skV = 0;

      float rhsHx = 0, rhsHy = 0, rhsHz = 0;
      float rhsEx = 0, rhsEy = 0, rhsEz = 0;

      int sk3 = 0;
      for(m=0;m<p_Np;++m){
	const float dr = ptDr[m], ds = ptDs[m], dt = ptDt[m];
	const float dx = drdx*dr+dsdx*ds+dtdx*dt;
	const float dy = drdy*dr+dsdy*ds+dtdy*dt;
	const float dz = drdz*dr+dsdz*ds+dtdz*dt;
	const float nHx = Qk[sk3++];
	const float nHy = Qk[sk3++];
	const float nHz = Qk[sk3++];
	const float nEx = Qk[sk3++];
	const float nEy = Qk[sk3++];
	const float nEz = Qk[sk3++];

	rhsHx += dz*nEy-dy*nEz;
	rhsHy += dx*nEz-dz*nEx;
	rhsHz += dy*nEx-dx*nEy;

	rhsEx += dy*nHz-dz*nHy;
	rhsEy += dz*nHx-dx*nHz;
	rhsEz += dx*nHy-dy*nHx;
      }
      
      int skR = p_Nfields*(n+k*p_Np);
      f_rhsQ[skR++] = rhsHx;
      f_rhsQ[skR++] = rhsHy;
      f_rhsQ[skR++] = rhsHz;
      f_rhsQ[skR++] = rhsEx;
      f_rhsQ[skR++] = rhsEy;
      f_rhsQ[skR++] = rhsEz;
    }
  }

  /* DO RECV */
  MPI_Status *instatus  = (MPI_Status*) calloc(mesh->nprocs, sizeof(MPI_Status));
  MPI_Waitall(Nmess, mpi_in_requests, instatus);
  free(instatus);

  for(k=0;k<K;++k){

    /* NOTE: once k is known, all other indexing variables etc are derived */
    register unsigned int n, m;

    /* NOTE: should be local memory */
    float fluxQ[p_Nfaces*p_Nfp*p_Nfields];

    /* NOTE: index into geometric factors */
    int surfid=k*7*p_Nfp*p_Nfaces;

    int id = k*p_Nfp*p_Nfaces;

    int sk = 0;

    for(m=0;m<p_Nfp*p_Nfaces;++m){
      int   idM       = surfinfo[surfid++]; 
      int   idP       = surfinfo[surfid++];
      const float FSc = surfinfo[surfid++];
      const float BSc = surfinfo[surfid++];
      const float NXf = surfinfo[surfid++];
      const float NYf = surfinfo[surfid++];     
      const float NZf = surfinfo[surfid++];
      
      float dHx, dHy, dHz, dEx, dEy, dEz;
      if(idP<0){
	idP = p_Nfields*(-1-idP);
	dHx = FSc*(f_inQ[idP++]  -f_Q[idM++]);
	dHy = FSc*(f_inQ[idP++]  -f_Q[idM++]);
	dHz = FSc*(f_inQ[idP++]  -f_Q[idM++]);
	dEx = FSc*(f_inQ[idP++]  -f_Q[idM++]);
	dEy = FSc*(f_inQ[idP++]  -f_Q[idM++]);
	dEz = FSc*(f_inQ[idP]    -f_Q[idM]);
      }else{
	dHx = FSc*(    f_Q[idP++]  -f_Q[idM++]);
	dHy = FSc*(    f_Q[idP++]  -f_Q[idM++]);
	dHz = FSc*(    f_Q[idP++]  -f_Q[idM++]);
	dEx = FSc*(BSc*f_Q[idP++]  -f_Q[idM++]);
	dEy = FSc*(BSc*f_Q[idP++]  -f_Q[idM++]);
	dEz = FSc*(BSc*f_Q[idP]    -f_Q[idM]);
      }
	
      const float ndotdH = NXf*dHx + NYf*dHy + NZf*dHz;
      const float ndotdE = NXf*dEx + NYf*dEy + NZf*dEz;
      
      fluxQ[sk++] = -NYf*dEz + NZf*dEy + dHx - ndotdH*NXf;
      fluxQ[sk++] = -NZf*dEx + NXf*dEz + dHy - ndotdH*NYf;
      fluxQ[sk++] = -NXf*dEy + NYf*dEx + dHz - ndotdH*NZf;
      
      fluxQ[sk++] =  NYf*dHz - NZf*dHy + dEx - ndotdE*NXf;
      fluxQ[sk++] =  NZf*dHx - NXf*dHz + dEy - ndotdE*NYf;
      fluxQ[sk++] =  NXf*dHy - NYf*dHx + dEz - ndotdE*NZf; 

    }
    
    for(n=0;n<p_Np;++n){    

      const float *ptLIFT = f_LIFT+n*p_Nfp*p_Nfaces;

      float rhsHx = 0, rhsHy = 0, rhsHz = 0;
      float rhsEx = 0, rhsEy = 0, rhsEz = 0;
      int sk3 = 0;

      /* NOTE: hint for compiler (nvcc) to unroll */
      for(m=0;m<p_Nfp*p_Nfaces;++m){
	const float L = ptLIFT[m];
	rhsHx += L*fluxQ[sk3++];
	rhsHy += L*fluxQ[sk3++];
	rhsHz += L*fluxQ[sk3++];
	rhsEx += L*fluxQ[sk3++];
	rhsEy += L*fluxQ[sk3++];
	rhsEz += L*fluxQ[sk3++];
      }
    
      int skR = p_Nfields*(n+k*p_Np);
      f_rhsQ[skR++] += rhsHx;
      f_rhsQ[skR++] += rhsHy;
      f_rhsQ[skR++] += rhsHz;
      f_rhsQ[skR++] += rhsEx;
      f_rhsQ[skR++] += rhsEy;
      f_rhsQ[skR++] += rhsEz;
    }
  }

  for(n=0;n<K*p_Np*p_Nfields;++n){
    f_resQ[n] = frka*f_resQ[n]+fdt*f_rhsQ[n];
    f_Q[n]   += frkb*f_resQ[n];
  }

  /* make sure all messages went out */
  MPI_Status *outstatus  = (MPI_Status*) calloc(mesh->nprocs, sizeof(MPI_Status));
  MPI_Waitall(Nmess, mpi_out_requests, outstatus);
  free(outstatus);


}

