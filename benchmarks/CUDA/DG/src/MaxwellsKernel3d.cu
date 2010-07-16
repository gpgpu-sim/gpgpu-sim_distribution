/* -*- mode: C; c-basic-offset: 8; c-indent-level: 8; c-continued-statement-offset: 8; c-label-offset: -8; -*- */

#include <stdio.h>
#include <cuda.h>

texture<float4, 1, cudaReadModeElementType> t_LIFT;
texture<float4, 1, cudaReadModeElementType> t_DrDsDt;
texture<float, 1, cudaReadModeElementType> t_Dr;
texture<float, 1, cudaReadModeElementType> t_Ds;
texture<float, 1, cudaReadModeElementType> t_Dt;
texture<float, 1, cudaReadModeElementType> t_vgeo;
texture<float4, 1, cudaReadModeElementType> t_vgeo4;
texture<float, 1, cudaReadModeElementType> t_Q;
texture<float, 1, cudaReadModeElementType> t_partQ;
texture<float, 1, cudaReadModeElementType> t_surfinfo;

static float *c_LIFT;
static float *c_DrDsDt;
static float *c_surfinfo;
static float *c_vgeo;
static float *c_Q; 
static float *c_partQ; 
static float *c_rhsQ; 
static float *c_resQ; 
static float *c_tmp;

extern "C"
{

#include "fem.h"

double InitGPU3d(Mesh *mesh, int Nfields){

  /* Q  */
  int sz = mesh->K*(BSIZE)*p_Nfields*sizeof(float); 

  float *f_Q = (float*) calloc(mesh->K*BSIZE*p_Nfields, sizeof(float));
  cudaMalloc  ((void**) &c_Q, sz);
  cudaMalloc  ((void**) &c_rhsQ, sz);
  cudaMalloc  ((void**) &c_resQ, sz);
  cudaMalloc  ((void**) &c_tmp, sz);
  cudaMemcpy( c_Q,    f_Q, sz, cudaMemcpyHostToDevice);
  cudaMemcpy( c_rhsQ, f_Q, sz, cudaMemcpyHostToDevice);
  cudaMemcpy( c_resQ, f_Q, sz, cudaMemcpyHostToDevice);
  cudaMemcpy( c_tmp,  f_Q, sz, cudaMemcpyHostToDevice);

  cudaBindTexture(0,  t_Q, c_Q, sz);

  sz = mesh->parNtotalout*sizeof(float);
  cudaMalloc((void**) &c_partQ, sz);
  cudaBindTexture(0,  t_partQ, c_partQ, sz);

  /*  LIFT  */
   sz = p_Np*(p_Nfp)*p_Nfaces*sizeof(float);
#if 0
   float *f_LIFT = (float*) malloc(sz);
   int skL = 0;
   for(int m=0;m<p_Nfp*p_Nfaces;++m){
     for(int n=0;n<p_Np;++n){
       f_LIFT[skL++] = d_LIFT[n+p_Np*m];
     }
   }
#else
   float *f_LIFT = (float*) malloc(sz);
   int skL = 0;
   for(int m=0;m<p_Nfp;++m){
     for(int n=0;n<p_Np;++n){
       for(int f=0;f<p_Nfaces;++f){
	 f_LIFT[skL++] = mesh->LIFT[0][p_Nfp*p_Nfaces*n+(f+p_Nfaces*m)];
       }
     }
   }
#endif
   cudaMalloc  ((void**) &c_LIFT, sz);
   cudaMemcpy( c_LIFT, f_LIFT, sz, cudaMemcpyHostToDevice);
   
   /* Bind the array to the texture */
   cudaBindTexture(0,  t_LIFT, c_LIFT, sz);

   /* DrDsDt */
   sz = BSIZE*BSIZE*4*sizeof(float);

   float* h_DrDsDt = (float*) calloc(BSIZE*BSIZE, sizeof(float4));
   int sk = 0;
   /* note transposed arrays to avoid "bank conflicts" */
   for(int n=0;n<p_Np;++n){
     for(int m=0;m<p_Np;++m){
       h_DrDsDt[4*(m+n*BSIZE)+0] = mesh->Dr[0][n+m*p_Np];
       h_DrDsDt[4*(m+n*BSIZE)+1] = mesh->Ds[0][n+m*p_Np];
       h_DrDsDt[4*(m+n*BSIZE)+2] = mesh->Dt[0][n+m*p_Np];
     }
   }
	   
   cudaMalloc  ((void**) &c_DrDsDt, sz);
   cudaMemcpy( c_DrDsDt, h_DrDsDt, sz, cudaMemcpyHostToDevice);
   
   /* Bind the array to the texture */
   cudaBindTexture(0,  t_DrDsDt, c_DrDsDt, sz);

   free(h_DrDsDt);

   /* vgeo */
   double drdx, dsdx, dtdx;
   double drdy, dsdy, dtdy;
   double drdz, dsdz, dtdz, J;
   float *vgeo = (float*) calloc(12*mesh->K, sizeof(float));

   for(int k=0;k<mesh->K;++k){
     GeometricFactors3d(mesh, k, 
			&drdx, &dsdx, &dtdx,
			&drdy, &dsdy, &dtdy,
			&drdz, &dsdz, &dtdz, &J);

     vgeo[k*12+0] = drdx; vgeo[k*12+1] = drdy; vgeo[k*12+2] = drdz;
     vgeo[k*12+4] = dsdx; vgeo[k*12+5] = dsdy; vgeo[k*12+6] = dsdz;
     vgeo[k*12+8] = dtdx; vgeo[k*12+9] = dtdy; vgeo[k*12+10] = dtdz;

   }

   sz = mesh->K*12*sizeof(float);
   cudaMalloc  ((void**) &c_vgeo, sz);
   cudaMemcpy( c_vgeo, vgeo, sz, cudaMemcpyHostToDevice);
   cudaBindTexture(0,  t_vgeo, c_vgeo, sz);
   
   /* surfinfo (vmapM, vmapP, Fscale, Bscale, nx, ny, nz, 0) */
   sz = mesh->K*p_Nfp*p_Nfaces*7*sizeof(float); 
   float* h_surfinfo = (float*) malloc(sz); 
   
   /* local-local info */
   sk = 0;
   int skP = -1;
   double *nxk = BuildVector(mesh->Nfaces);
   double *nyk = BuildVector(mesh->Nfaces);
   double *nzk = BuildVector(mesh->Nfaces);
   double *sJk = BuildVector(mesh->Nfaces);

   double dt = 1e6;

   for(int k=0;k<mesh->K;++k){

     GeometricFactors3d(mesh, k, 
			&drdx, &dsdx, &dtdx,
			&drdy, &dsdy, &dtdy,
			&drdz, &dsdz, &dtdz, &J);

     Normals3d(mesh, k, nxk, nyk, nzk, sJk);
     
     for(int f=0;f<mesh->Nfaces;++f){

	     dt = min(dt, J/sJk[f]);
       
       for(int m=0;m<p_Nfp;++m){
	 int n = m + f*p_Nfp + p_Nfp*p_Nfaces*k;
	 int idM = mesh->vmapM[n];
	 int idP = mesh->vmapP[n];
	 int  nM = idM%p_Np; 
	 int  nP = idP%p_Np; 
	 int  kM = (idM-nM)/p_Np;
	 int  kP = (idP-nP)/p_Np;
	 idM = nM + Nfields*BSIZE*kM;
	 idP = nP + Nfields*BSIZE*kP;
	 
	 /* stub resolve some other way */
	 if(mesh->vmapP[n]<0){
	   idP = mesh->vmapP[n]; /* -ve numbers */
	 }
 
	 sk = 7*p_Nfp*p_Nfaces*k+m+f*p_Nfp;
	 h_surfinfo[sk + 0*p_Nfp*p_Nfaces] = idM;
	 h_surfinfo[sk + 1*p_Nfp*p_Nfaces] = idP;
	 h_surfinfo[sk + 2*p_Nfp*p_Nfaces] = sJk[f]/(2.*J);
	 h_surfinfo[sk + 3*p_Nfp*p_Nfaces] = (idM==idP)?-1.:1.;
	 h_surfinfo[sk + 4*p_Nfp*p_Nfaces] = nxk[f];
	 h_surfinfo[sk + 5*p_Nfp*p_Nfaces] = nyk[f];
	 h_surfinfo[sk + 6*p_Nfp*p_Nfaces] = nzk[f];
       }
     }
   }
	   
   cudaMalloc  ((void**) &c_surfinfo, sz);
   cudaMemcpy( c_surfinfo, h_surfinfo, sz, cudaMemcpyHostToDevice);

   cudaBindTexture(0,  t_surfinfo, c_surfinfo, sz);

   free(h_surfinfo);

   sz = mesh->parNtotalout*sizeof(int);
   cudaMalloc((void**) &(mesh->c_parmapOUT), sz);
   cudaMemcpy(mesh->c_parmapOUT,  mesh->parmapOUT, sz, cudaMemcpyHostToDevice);

   return dt;
}



__global__ void MaxwellsGPU_VOL_Kernel3D(float *g_rhsQ){

  /* fastest */
  __device__ __shared__ float s_Q[p_Nfields*BSIZE];
  __device__ __shared__ float s_facs[12];

  const int n = threadIdx.x;
  const int k = blockIdx.x;
  
  /* "coalesced"  */
  int m = n+k*p_Nfields*BSIZE;
  int id = n;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch(t_Q, m); 

  if(p_Np<12 && n==0)
    for(m=0;m<12;++m)
      s_facs[m] = tex1Dfetch(t_vgeo, 12*k+m);
  else if(n<12 && p_Np>=12)
    s_facs[n] = tex1Dfetch(t_vgeo, 12*k+n);

  __syncthreads();

  float dHxdr=0,dHxds=0,dHxdt=0;
  float dHydr=0,dHyds=0,dHydt=0;
  float dHzdr=0,dHzds=0,dHzdt=0;
  float dExdr=0,dExds=0,dExdt=0;
  float dEydr=0,dEyds=0,dEydt=0;
  float dEzdr=0,dEzds=0,dEzdt=0;
  float Q;

  for(m=0;p_Np-m;){
    float4 D = tex1Dfetch(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;
#if ( (p_Np) % 2 )==0
    D = tex1Dfetch(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;

#if ( (p_Np)%3 )==0
    D = tex1Dfetch(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;
#endif
#endif
  }
  
  const float drdx= s_facs[0];
  const float drdy= s_facs[1];
  const float drdz= s_facs[2];
  const float dsdx= s_facs[4];
  const float dsdy= s_facs[5];
  const float dsdz= s_facs[6];
  const float dtdx= s_facs[8];
  const float dtdy= s_facs[9];
  const float dtdz= s_facs[10];
  
  m = n+p_Nfields*BSIZE*k;

  g_rhsQ[m] = -(drdy*dEzdr+dsdy*dEzds+dtdy*dEzdt - drdz*dEydr-dsdz*dEyds-dtdz*dEydt); m += BSIZE;
  g_rhsQ[m] = -(drdz*dExdr+dsdz*dExds+dtdz*dExdt - drdx*dEzdr-dsdx*dEzds-dtdx*dEzdt); m += BSIZE;
  g_rhsQ[m] = -(drdx*dEydr+dsdx*dEyds+dtdx*dEydt - drdy*dExdr-dsdy*dExds-dtdy*dExdt); m += BSIZE;
  g_rhsQ[m] =  (drdy*dHzdr+dsdy*dHzds+dtdy*dHzdt - drdz*dHydr-dsdz*dHyds-dtdz*dHydt); m += BSIZE;
  g_rhsQ[m] =  (drdz*dHxdr+dsdz*dHxds+dtdz*dHxdt - drdx*dHzdr-dsdx*dHzds-dtdx*dHzdt); m += BSIZE;
  g_rhsQ[m] =  (drdx*dHydr+dsdx*dHyds+dtdx*dHydt - drdy*dHxdr-dsdy*dHxds-dtdy*dHxdt); 
}

__global__ void MaxwellsGPU_SURF_Kernel3D(float *g_Q, float *g_rhsQ){

  __device__ __shared__ float s_fluxQ[p_Nfields*p_Nfp*p_Nfaces];

  const int n = threadIdx.x;
  const int k = blockIdx.x;
  int m;

  /* grab surface nodes and store flux in shared memory */
  if(n< (p_Nfp*p_Nfaces) ){
    /* coalesced reads (maybe) */
    m = 7*(k*p_Nfp*p_Nfaces)+n;
    const  int idM   = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
           int idP   = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float Fsc = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float Bsc = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float nx  = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float ny  = tex1Dfetch(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float nz  = tex1Dfetch(t_surfinfo, m);

    /* check if idP<0  */
    double dHx, dHy, dHz, dEx, dEy, dEz;
    if(idP<0){
      idP = p_Nfields*(-1-idP);
      
      dHx = Fsc*(tex1Dfetch(t_partQ, idP+0) - tex1Dfetch(t_Q, idM+0*BSIZE));
      dHy = Fsc*(tex1Dfetch(t_partQ, idP+1) - tex1Dfetch(t_Q, idM+1*BSIZE));
      dHz = Fsc*(tex1Dfetch(t_partQ, idP+2) - tex1Dfetch(t_Q, idM+2*BSIZE));
      
      dEx = Fsc*(tex1Dfetch(t_partQ, idP+3) - tex1Dfetch(t_Q, idM+3*BSIZE));
      dEy = Fsc*(tex1Dfetch(t_partQ, idP+4) - tex1Dfetch(t_Q, idM+4*BSIZE));
      dEz = Fsc*(tex1Dfetch(t_partQ, idP+5) - tex1Dfetch(t_Q, idM+5*BSIZE));
    }
    else{
      dHx = Fsc*(tex1Dfetch(t_Q, idP+0*BSIZE) - tex1Dfetch(t_Q, idM+0*BSIZE));
      dHy = Fsc*(tex1Dfetch(t_Q, idP+1*BSIZE) - tex1Dfetch(t_Q, idM+1*BSIZE));
      dHz = Fsc*(tex1Dfetch(t_Q, idP+2*BSIZE) - tex1Dfetch(t_Q, idM+2*BSIZE));
      
      dEx = Fsc*(Bsc*tex1Dfetch(t_Q, idP+3*BSIZE) - tex1Dfetch(t_Q, idM+3*BSIZE));
      dEy = Fsc*(Bsc*tex1Dfetch(t_Q, idP+4*BSIZE) - tex1Dfetch(t_Q, idM+4*BSIZE));
      dEz = Fsc*(Bsc*tex1Dfetch(t_Q, idP+5*BSIZE) - tex1Dfetch(t_Q, idM+5*BSIZE));
    }

    const double ndotdH = nx*dHx + ny*dHy + nz*dHz;
    const double ndotdE = nx*dEx + ny*dEy + nz*dEz;

    m = n;
    s_fluxQ[m] = -ny*dEz + nz*dEy + dHx - ndotdH*nx; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] = -nz*dEx + nx*dEz + dHy - ndotdH*ny; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] = -nx*dEy + ny*dEx + dHz - ndotdH*nz; m += p_Nfp*p_Nfaces;

    s_fluxQ[m] =  ny*dHz - nz*dHy + dEx - ndotdE*nx; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] =  nz*dHx - nx*dHz + dEy - ndotdE*ny; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] =  nx*dHy - ny*dHx + dEz - ndotdE*nz; 
  }

  /* make sure all element data points are cached */
  __syncthreads();

  if(n< (p_Np))
  {
    float rhsHx = 0, rhsHy = 0, rhsHz = 0;
    float rhsEx = 0, rhsEy = 0, rhsEz = 0;
    
    int sk = n;
    /* can manually unroll to 4 because there are 4 faces */
    for(m=0;p_Nfaces*p_Nfp-m;){
      const float4 L = tex1Dfetch(t_LIFT, sk); sk+=p_Np;

      /* broadcast */
      int sk1 = m;
      rhsHx += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

    }
    
    m = n+p_Nfields*k*BSIZE;
    g_rhsQ[m] += rhsHx; m += BSIZE;
    g_rhsQ[m] += rhsHy; m += BSIZE;
    g_rhsQ[m] += rhsHz; m += BSIZE;
    g_rhsQ[m] += rhsEx; m += BSIZE;
    g_rhsQ[m] += rhsEy; m += BSIZE;
    g_rhsQ[m] += rhsEz; m += BSIZE;

  }
}


__global__ void MaxwellsGPU_RK_Kernel3D(int Ntotal, float *g_resQ, float *g_rhsQ, float *g_Q, float fa, float fb, float fdt){
  
  int n = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(n<Ntotal){
    float rhs = g_rhsQ[n];
    float res = g_resQ[n];
    res = fa*res + fdt*rhs;
    
    g_resQ[n] = res;
    g_Q[n]    += fb*res;
  }

} 


/* assumes data resides on device */
void MaxwellsKernel3d(Mesh *mesh, float frka, float frkb, float fdt){

  /* grab data from device and initiate sends */
  MaxwellsMPISend3d(mesh);
   
  int ThreadsPerBlock, BlocksPerGrid;	

  BlocksPerGrid    = mesh->K; 
  ThreadsPerBlock = p_Np;

  /* evaluate volume derivatives */
  MaxwellsGPU_VOL_Kernel3D <<< BlocksPerGrid, ThreadsPerBlock >>>  (c_rhsQ);

  /* finalize sends and recvs, and transfer to device */
  MaxwellsMPIRecv3d(mesh, c_partQ);

  BlocksPerGrid = mesh->K;

  if( ( p_Nfp*p_Nfaces ) > (p_Np) )
    ThreadsPerBlock = p_Nfp*p_Nfaces;
  else
    ThreadsPerBlock = p_Np;

  /* evaluate surface contributions */
  MaxwellsGPU_SURF_Kernel3D <<< BlocksPerGrid, ThreadsPerBlock >>> (c_Q, c_rhsQ);

  int Ntotal = mesh->K*BSIZE*p_Nfields;
  
  ThreadsPerBlock = 256;
  BlocksPerGrid = (Ntotal+ThreadsPerBlock-1)/ThreadsPerBlock;

  /* update RK Step */
  MaxwellsGPU_RK_Kernel3D<<< BlocksPerGrid, ThreadsPerBlock >>> 
	  (Ntotal, c_resQ, c_rhsQ, c_Q, frka, frkb, fdt);

}




void gpu_set_data3d(int K,
		  double *d_Hx, double *d_Hy, double *d_Hz,
		  double *d_Ex, double *d_Ey, double *d_Ez){


  float *f_Q = (float*) calloc(K*p_Nfields*BSIZE,sizeof(float));
  
  /* also load into usual data matrices */
  
  for(int k=0;k<K;++k){
    int gk = k;
    for(int n=0;n<p_Np;++n)
      f_Q[n        +k*BSIZE*p_Nfields] = d_Hx[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n  +BSIZE+k*BSIZE*p_Nfields] = d_Hy[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+2*BSIZE+k*BSIZE*p_Nfields] = d_Hz[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+3*BSIZE+k*BSIZE*p_Nfields] = d_Ex[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+4*BSIZE+k*BSIZE*p_Nfields] = d_Ey[n+gk*p_Np];
    for(int n=0;n<p_Np;++n)
      f_Q[n+5*BSIZE+k*BSIZE*p_Nfields] = d_Ez[n+gk*p_Np];
  }
  
  cudaMemcpy(c_Q, f_Q, BSIZE*K*p_Nfields*sizeof(float), cudaMemcpyHostToDevice);
  
  free(f_Q);
}
  
void gpu_get_data3d(int K,
		  double *d_Hx, double *d_Hy, double *d_Hz,
		  double *d_Ex, double *d_Ey, double *d_Ez){

  float *f_Q = (float*) calloc(K*p_Nfields*BSIZE,sizeof(float));
  
  cudaMemcpy(f_Q, c_Q, K*BSIZE*p_Nfields*sizeof(float), cudaMemcpyDeviceToHost);

  /* also load into usual data matrices */
  
  for(int k=0;k<K;++k){
    int gk = k;
    for(int n=0;n<p_Np;++n)
      d_Hx[n+gk*p_Np] = f_Q[n        +k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n) 
      d_Hy[n+gk*p_Np] = f_Q[n  +BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Hz[n+gk*p_Np] = f_Q[n+2*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ex[n+gk*p_Np] = f_Q[n+3*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n) 
      d_Ey[n+gk*p_Np] = f_Q[n+4*BSIZE+k*BSIZE*p_Nfields];
    for(int n=0;n<p_Np;++n)
      d_Ez[n+gk*p_Np] = f_Q[n+5*BSIZE+k*BSIZE*p_Nfields];

  }

  free(f_Q);
}

__global__ void partial_get_kernel3d(int Ntotal, int *g_index, float *g_partQ){
  
  int n = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(n<Ntotal)
    g_partQ[n] = tex1Dfetch(t_Q, g_index[n]);
  
} 

void get_partial_gpu_data3d(int Ntotal, int *g_index, float *h_partQ){

  int ThreadsPerBlock = 256;
  int BlocksPerGrid = (Ntotal+ThreadsPerBlock-1)/ThreadsPerBlock;

  partial_get_kernel3d <<< BlocksPerGrid, ThreadsPerBlock >>> (Ntotal, g_index, c_tmp);

  cudaMemcpy(h_partQ, c_tmp, Ntotal*sizeof(float), cudaMemcpyDeviceToHost);
}

}
