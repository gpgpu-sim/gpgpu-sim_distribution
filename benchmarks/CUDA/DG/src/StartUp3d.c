
#include "fem.h"

void StartUp3d(Mesh *mesh){

#if p_N==1
#include "data3dN01.h"
#elif p_N==2
#include "data3dN02.h"
#elif p_N==3
#include "data3dN03.h"
#elif p_N==4
#include "data3dN04.h"
#elif p_N==5
#include "data3dN05.h"
#elif p_N==6
#include "data3dN06.h"
#elif p_N==7
#include "data3dN07.h"
#elif p_N==8
#include "data3dN08.h"
#elif p_N==9
#include "data3dN09.h"
#endif

  int n, m, k;

  /* load r, s, t */
  mesh->r = BuildVector(p_Np);
  mesh->s = BuildVector(p_Np);
  mesh->t = BuildVector(p_Np);
  for(n=0;n<p_Np;++n){
    mesh->r[n] = p_r[n];
    mesh->s[n] = p_s[n];
    mesh->t[n] = p_t[n];
  }

  /* load Dr, Ds, Dt */
  mesh->Dr = BuildMatrix(p_Np, p_Np);
  mesh->Ds = BuildMatrix(p_Np, p_Np);
  mesh->Dt = BuildMatrix(p_Np, p_Np);
  for(n=0;n<p_Np;++n){
    for(m=0;m<p_Np;++m){
      mesh->Dr[n][m] = p_Dr[n][m];
      mesh->Ds[n][m] = p_Ds[n][m];
      mesh->Dt[n][m] = p_Dt[n][m];
    }
  }

  /* load LIFT */
  mesh->LIFT = BuildMatrix(p_Np, p_Nfp*p_Nfaces);
  for(n=0;n<p_Np;++n){
    for(m=0;m<p_Nfp*p_Nfaces;++m){
      mesh->LIFT[n][m] = p_LIFT[n][m];
    }
  }

  mesh->Fmask = BuildIntMatrix(p_Nfaces, p_Nfp);
  for(n=0;n<p_Nfaces;++n){
    for(m=0;m<p_Nfp;++m){
      mesh->Fmask[n][m] = p_Fmask[n][m];
    }
  }

  /* low storage RK coefficients */
  mesh->rk4a = BuildVector(5);
  mesh->rk4a[0] =              0.0;
  mesh->rk4a[1] =  -567301805773.0 / 1357537059087.0;
  mesh->rk4a[2] = -2404267990393.0 / 2016746695238.0;
  mesh->rk4a[3] = -3550918686646.0 / 2091501179385.0;
  mesh->rk4a[4] = -1275806237668.0 /  842570457699.0;

  mesh->rk4b = BuildVector(5);
  mesh->rk4b[0] =  1432997174477.0 /  9575080441755.0;
  mesh->rk4b[1] =  5161836677717.0 / 13612068292357.0;
  mesh->rk4b[2] =  1720146321549.0 /  2090206949498.0;
  mesh->rk4b[3] =  3134564353537.0 /  4481467310338.0;
  mesh->rk4b[4] =  2277821191437.0 / 14882151754819.0;

  mesh->rk4c = BuildVector(6);
  mesh->rk4c[0] =              0.0;
  mesh->rk4c[1] =  1432997174477.0 / 9575080441755.0;
  mesh->rk4c[2] =  2526269341429.0 / 6820363962896.0;
  mesh->rk4c[3] =  2006345519317.0 / 3224310063776.0;
  mesh->rk4c[4] =  2802321613138.0 / 2924317926251.0;
  mesh->rk4c[5] =              1.0;

  /* build coordinates */
  mesh->x = BuildMatrix(mesh->K, p_Np);
  mesh->y = BuildMatrix(mesh->K, p_Np);
  mesh->z = BuildMatrix(mesh->K, p_Np);
  
  for(k=0;k<mesh->K;++k){
    for(n=0;n<p_Np;++n){
      double r = mesh->r[n];
      double s = mesh->s[n];
      double t = mesh->t[n];

      mesh->x[k][n] = 0.5*( -mesh->GX[k][0]*(r+s+t+1) + 
			     mesh->GX[k][1]*(1.+r) + 
			     mesh->GX[k][2]*(1.+s) +
			     mesh->GX[k][3]*(1.+t) 
			    );

      mesh->y[k][n] = 0.5*( -mesh->GY[k][0]*(r+s+t+1) + 
			     mesh->GY[k][1]*(1.+r) + 
			     mesh->GY[k][2]*(1.+s) +
			     mesh->GY[k][3]*(1.+t) 
			    );

      mesh->z[k][n] = 0.5*( -mesh->GZ[k][0]*(r+s+t+1) + 
			     mesh->GZ[k][1]*(1.+r) + 
			     mesh->GZ[k][2]*(1.+s) +
			     mesh->GZ[k][3]*(1.+t) 
			    );
    }
  }
  
  /* build node-node connectivity maps */
  void BuildMaps3d(Mesh *mesh);
  BuildMaps3d(mesh);

}
