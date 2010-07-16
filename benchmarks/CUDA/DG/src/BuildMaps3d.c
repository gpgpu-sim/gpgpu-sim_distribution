#include "mpi.h"
#include "fem.h"

void BuildMaps3d(Mesh *mesh){

  printf("Hello %d\n", 1002);

  int nprocs = mesh->nprocs;
  int procid = mesh->procid;

  int K       = mesh->K;
  int Nfaces  = mesh->Nfaces;

  mesh->vmapM = BuildIntVector(p_Nfp*p_Nfaces*K);
  mesh->vmapP = BuildIntVector(p_Nfp*p_Nfaces*K);

  int m;
  int k1,f1,p1,n1,id1, k2,f2,p2,n2,id2;

  double x1, y1,z1, x2, y2, z2, d12;
  
  double *nxk = BuildVector(Nfaces);
  double *nyk = BuildVector(Nfaces);
  double *nzk = BuildVector(Nfaces);
  double *sJk = BuildVector(Nfaces);

  printf("Hello %d\n", 1001);

  /* first build local */
  for(k1=0;k1<K;++k1){

    /* get some information about the face geometries */
    Normals3d(mesh, k1, nxk, nyk, nzk, sJk);

    for(f1=0;f1<Nfaces;++f1){

      /* volume -> face nodes */
      for(n1=0;n1<p_Nfp;++n1){
	id1 = n1+f1*p_Nfp+k1*p_Nfp*p_Nfaces;
	mesh->vmapM[id1] = mesh->Fmask[f1][n1] + k1*p_Np;
      }


      /* find neighbor */
      k2 = mesh->EToE[k1][f1]; 
      f2 = mesh->EToF[k1][f1];
      p2 = mesh->EToP[k1][f1];

      if(k1==k2 || procid!=p2 ){
	for(n1=0;n1<p_Nfp;++n1){
	  id1 = n1+f1*p_Nfp+k1*p_Nfp*p_Nfaces;
	  mesh->vmapP[id1] = k1*p_Np + mesh->Fmask[f1][n1];
	}
      }else{
	/* treat as boundary for the moment  */
	
	for(n1=0;n1<p_Nfp;++n1){
	  id1 = n1+f1*p_Nfp+k1*p_Nfp*p_Nfaces;

	  x1 = mesh->x[k1][mesh->Fmask[f1][n1]];
	  y1 = mesh->y[k1][mesh->Fmask[f1][n1]];
	  z1 = mesh->z[k1][mesh->Fmask[f1][n1]];

	  for(n2=0;n2<p_Nfp;++n2){

	    id2 = n2+f2*p_Nfp+k2*p_Nfp*p_Nfaces;

	    x2 = mesh->x[k2][mesh->Fmask[f2][n2]];
	    y2 = mesh->y[k2][mesh->Fmask[f2][n2]];
	    z2 = mesh->z[k2][mesh->Fmask[f2][n2]];

	    /* find normalized distance between these nodes */
	    /* [ use sJk as a measure of edge length (ignore factor of 2) ] */
	    d12 = ((x1-x2)*(x1-x2) +
		   (y1-y2)*(y1-y2) +
		   (z1-z2)*(z1-z2)); /* /(sJk[f1]*sJk[f1]);  */
	    if(d12<NODETOL){
	      mesh->vmapP[id1] = k2*p_Np + mesh->Fmask[f2][n2];
	      break;
	    }
	  }
	  if(n2==p_Nfp){
	    printf("LOST NODE !!!\n");
	  }
	}
      }
    }
  }

#if 0
  int n;
  for(k1=0;k1<mesh->K;++k1){
    double drdx, dsdx, dtdx;
    double drdy, dsdy, dtdy;
    double drdz, dsdz, dtdz, J;

    Normals3d(mesh, k1, nxk, nyk, nzk, sJk);
    
    GeometricFactors3d(mesh, k1, 
		       &drdx, &dsdx, &dtdx,
		       &drdy, &dsdy, &dtdy,
		       &drdz, &dsdz, &dtdz, &J);
    
    for(f1=0;f1<mesh->Nfaces;++f1){
      for(m=0;m<p_Nfp;++m){
	n = k1*p_Nfp*p_Nfaces+f1*p_Nfp+m;
	x1 = mesh->x[0][mesh->vmapM[n]];
	y1 = mesh->y[0][mesh->vmapM[n]];
	z1 = mesh->z[0][mesh->vmapM[n]];
	x2 = mesh->x[0][mesh->vmapP[n]];
	y2 = mesh->y[0][mesh->vmapP[n]];
	z2 = mesh->z[0][mesh->vmapP[n]];
	d12 = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) );
	printf("n:%d  %d -> %d  d=%lg sJ=%lg J=%lg (%d,%d,%d,%d)\n", 
	       n, mesh->vmapM[n], mesh->vmapP[n], d12, sJk[f1], J, 
	       mesh->EToV[k1][0],mesh->EToV[k1][1],mesh->EToV[k1][2],mesh->EToV[k1][3]);
      }
    }
  }
#endif

  /* now build parallel maps */
  double **xsend = (double**) calloc(nprocs, sizeof(double*));
  double **ysend = (double**) calloc(nprocs, sizeof(double*));
  double **zsend = (double**) calloc(nprocs, sizeof(double*));
  double **xrecv = (double**) calloc(nprocs, sizeof(double*));
  double **yrecv = (double**) calloc(nprocs, sizeof(double*));
  double **zrecv = (double**) calloc(nprocs, sizeof(double*));

  int **Esend = (int**) calloc(nprocs, sizeof(int*));
  int **Fsend = (int**) calloc(nprocs, sizeof(int*));
  int **Erecv = (int**) calloc(nprocs, sizeof(int*));
  int **Frecv = (int**) calloc(nprocs, sizeof(int*));

  for(p2=0;p2<nprocs;++p2){
    if(mesh->Npar[p2]){
      xsend[p2] = BuildVector(mesh->Npar[p2]*p_Nfp);
      ysend[p2] = BuildVector(mesh->Npar[p2]*p_Nfp);
      zsend[p2] = BuildVector(mesh->Npar[p2]*p_Nfp);
      Esend[p2] = BuildIntVector(mesh->Npar[p2]*p_Nfp);
      Fsend[p2] = BuildIntVector(mesh->Npar[p2]*p_Nfp);
      
      xrecv[p2] = BuildVector(mesh->Npar[p2]*p_Nfp);
      yrecv[p2] = BuildVector(mesh->Npar[p2]*p_Nfp);
      zrecv[p2] = BuildVector(mesh->Npar[p2]*p_Nfp);
      Erecv[p2] = BuildIntVector(mesh->Npar[p2]*p_Nfp);
      Frecv[p2] = BuildIntVector(mesh->Npar[p2]*p_Nfp);
    }
  }

  int *skP = BuildIntVector(nprocs);
  
  /* send coordinates in local order */
  int cnt = 0;
  for(k1=0;k1<K;++k1){
    for(f1=0;f1<p_Nfaces;++f1){
      p2 = mesh->EToP[k1][f1];
      if(p2!=procid){
	for(n1=0;n1<p_Nfp;++n1){
	  xsend[p2][skP[p2]] = mesh->x[k1][mesh->Fmask[f1][n1]];
	  ysend[p2][skP[p2]] = mesh->y[k1][mesh->Fmask[f1][n1]];
	  zsend[p2][skP[p2]] = mesh->z[k1][mesh->Fmask[f1][n1]];
	  Esend[p2][skP[p2]] = mesh->EToE[k1][f1];
	  Fsend[p2][skP[p2]] = mesh->EToF[k1][f1];
	  ++(skP[p2]);
	}
      }
    }
  }
  
  MPI_Request *xsendrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *ysendrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *zsendrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *xrecvrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *yrecvrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *zrecvrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *Esendrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *Fsendrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *Erecvrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *Frecvrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));

  MPI_Status  *status = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));

  cnt = 0;
  for(p2=0;p2<nprocs;++p2){
    if(p2!=procid && mesh->Npar[p2]!=0){
      int Nout = mesh->Npar[p2]*p_Nfp;
      
      MPI_Isend(xsend[p2], Nout, MPI_DOUBLE, p2,  666+p2, MPI_COMM_WORLD, xsendrequests+cnt);
      MPI_Isend(ysend[p2], Nout, MPI_DOUBLE, p2, 1666+p2, MPI_COMM_WORLD, ysendrequests+cnt);
      MPI_Isend(zsend[p2], Nout, MPI_DOUBLE, p2, 4666+p2, MPI_COMM_WORLD, zsendrequests+cnt);
      MPI_Isend(Esend[p2], Nout, MPI_INT,    p2, 2666+p2, MPI_COMM_WORLD, Esendrequests+cnt);
      MPI_Isend(Fsend[p2], Nout, MPI_INT,    p2, 3666+p2, MPI_COMM_WORLD, Fsendrequests+cnt);

      MPI_Irecv(xrecv[p2], Nout, MPI_DOUBLE, p2,  666+procid, MPI_COMM_WORLD, xrecvrequests+cnt);
      MPI_Irecv(yrecv[p2], Nout, MPI_DOUBLE, p2, 1666+procid, MPI_COMM_WORLD, yrecvrequests+cnt);
      MPI_Irecv(zrecv[p2], Nout, MPI_DOUBLE, p2, 4666+procid, MPI_COMM_WORLD, zrecvrequests+cnt);
      MPI_Irecv(Erecv[p2], Nout, MPI_INT,    p2, 2666+procid, MPI_COMM_WORLD, Erecvrequests+cnt);
      MPI_Irecv(Frecv[p2], Nout, MPI_INT,    p2, 3666+procid, MPI_COMM_WORLD, Frecvrequests+cnt);
      ++cnt;
    }
  }

  MPI_Waitall(cnt, xsendrequests, status);
  MPI_Waitall(cnt, ysendrequests, status);
  MPI_Waitall(cnt, zsendrequests, status);
  MPI_Waitall(cnt, Esendrequests, status); 
  MPI_Waitall(cnt, Fsendrequests, status);

  MPI_Waitall(cnt, xrecvrequests, status);
  MPI_Waitall(cnt, yrecvrequests, status);
  MPI_Waitall(cnt, zrecvrequests, status);
  MPI_Waitall(cnt, Erecvrequests, status); 
  MPI_Waitall(cnt, Frecvrequests, status);
  
  /* add up the total number of outgoing/ingoing nodes */
  mesh->parNtotalout = 0;
  for(p2=0;p2<nprocs;++p2)
    mesh->parNtotalout += skP[p2]*p_Nfields;

  mesh->parmapOUT = BuildIntVector(mesh->parNtotalout);

  /* now match up local nodes with the requested (recv'ed nodes) */
  int idout = -1;
  int sk = 0;
  for(p2=0;p2<nprocs;++p2){
    /* for each received face */
    for(m=0;m<skP[p2];++m){
      k1 = Erecv[p2][m];
      f1 = Frecv[p2][m];
      x2 = xrecv[p2][m];
      y2 = yrecv[p2][m];
      z2 = zrecv[p2][m];

      Normals3d(mesh, k1, nxk, nyk, nzk, sJk);
      
      for(n1=0;n1<p_Nfp;++n1){
	
	x1 = mesh->x[k1][mesh->Fmask[f1][n1]];
	y1 = mesh->y[k1][mesh->Fmask[f1][n1]];
	z1 = mesh->z[k1][mesh->Fmask[f1][n1]];
	
	d12 = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)); /* /(sJk[f1]*sJk[f1]); */
	
	if(d12<NODETOL){
	  int fld;
	  for(fld=0;fld<p_Nfields;++fld){
#ifdef CUDA
	    mesh->parmapOUT[sk++] = k1*BSIZE*p_Nfields+mesh->Fmask[f1][n1] + BSIZE*fld; 
#else
	    mesh->parmapOUT[sk++] = p_Nfields*(k1*p_Np+mesh->Fmask[f1][n1]) + fld;
#endif
	  }
	}
      }
    }
  }

  /* create incoming node map */
  int parcnt = -1;
  for(p2=0;p2<nprocs;++p2){
    for(k1=0;k1<K;++k1){
      for(f1=0;f1<p_Nfaces;++f1){
	if(mesh->EToP[k1][f1]==p2 && p2!=procid){
	  for(n1=0;n1<p_Nfp;++n1){
	    id1 = n1+f1*p_Nfp+k1*p_Nfp*p_Nfaces;
	    mesh->vmapP[id1] = parcnt;
	    --parcnt;
	  }
	}
      }
    }
  }

  /* buffers for communication */
  mesh->f_outQ = (float*) calloc(mesh->parNtotalout+1, sizeof(float));
  mesh->f_inQ  = (float*) calloc(mesh->parNtotalout+1, sizeof(float));
  
}
