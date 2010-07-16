#include "mpi.h"
#include <parmetisbin.h>

#include "fem.h"

void LoadBalance3d(Mesh *mesh){

  int n,p,k,v,f;

  int nprocs = mesh->nprocs;
  int procid = mesh->procid;
  int **EToV = mesh->EToV;
  double **VX = mesh->GX;
  double **VY = mesh->GY;
  double **VZ = mesh->GZ;

  if(!procid) printf("Root: Entering LoadBalance\n");

  int Nverts = mesh->Nverts;
  
  int *Kprocs = BuildIntVector(nprocs);

  /* local number of elements */
  int Klocal = mesh->K;

  /* find number of elements on all processors */
  MPI_Allgather(&Klocal, 1, MPI_INT, Kprocs, 1, MPI_INT, MPI_COMM_WORLD);

  /* element distribution -- cumulative element count on processes */
  idxtype *elmdist = idxmalloc(nprocs+1, "elmdist");

  elmdist[0] = 0;
  for(p=0;p<nprocs;++p)
    elmdist[p+1] = elmdist[p] + Kprocs[p];

  /* list of element starts */
  idxtype *eptr = idxmalloc(Klocal+1, "eptr");

  eptr[0] = 0;
  for(k=0;k<Klocal;++k)
    eptr[k+1] = eptr[k] + Nverts;

  /* local element to vertex */
  idxtype *eind = idxmalloc(Nverts*Klocal, "eind");

  for(k=0;k<Klocal;++k)
    for(n=0;n<Nverts;++n)
      eind[k*Nverts+n] = EToV[k][n];

  /* weight per element */
  idxtype *elmwgt = idxmalloc(Klocal, "elmwgt");

  for(k=0;k<Klocal;++k)
    elmwgt[k] = 1.;
  
  /* weight flag */
  int wgtflag = 0;
  
  /* number flag (1=fortran, 0=c) */
  int numflag = 0;

  /* ncon = 1 */
  int ncon = 1;

  /* nodes on element face */
  int ncommonnodes = 3;
  
  /* number of partitions */
  int nparts = nprocs;

  /* tpwgts */
  float *tpwgts = (float*) calloc(Klocal, sizeof(float));
 
  for(k=0;k<Klocal;++k)
    tpwgts[k] = 1./(float)nprocs;

  float ubvec[MAXNCON];

  for (n=0; n<ncon; ++n)
    ubvec[n] = UNBALANCE_FRACTION;
  
  int options[10];
  
  options[0] = 1;
  options[PMV3_OPTION_DBGLVL] = 7;
  options[PMV3_OPTION_SEED] = 0;

  int edgecut;

  idxtype *part = idxmalloc(Klocal, "part");

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  ParMETIS_V3_PartMeshKway
    (elmdist, 
     eptr, 
     eind, 
     elmwgt, 
     &wgtflag, 
     &numflag, 
     &ncon, 
     &ncommonnodes,
     &nparts, 
     tpwgts, 
     ubvec, 
     options, 
     &edgecut,
     part, 
     &comm);

  int **outlist = (int**) calloc(nprocs, sizeof(int*));
  double **xoutlist = (double**) calloc(nprocs, sizeof(double*));
  double **youtlist = (double**) calloc(nprocs, sizeof(double*));
  double **zoutlist = (double**) calloc(nprocs, sizeof(double*));

  int *outK = (int*) calloc(nprocs, sizeof(int));
  
  int *inK = (int*) calloc(nprocs, sizeof(int));

  MPI_Request *inrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *outrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *xinrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *xoutrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *yinrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *youtrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *zinrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  MPI_Request *zoutrequests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));

  for(k=0;k<Klocal;++k)
    ++outK[part[k]];
  
  /* get count of incoming elements from each process */
  MPI_Alltoall(outK, 1, MPI_INT, 
	       inK,  1, MPI_INT, 
	       MPI_COMM_WORLD);

  /* count totals on each process */
  int *  newKprocs = BuildIntVector(nprocs);
  MPI_Allreduce(outK, newKprocs, nprocs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int totalinK = 0;
  for(p=0;p<nprocs;++p){
    totalinK += inK[p];
  }

  int **newEToV = BuildIntMatrix(totalinK, Nverts);
  double **newVX   = BuildMatrix(totalinK, Nverts);
  double **newVY   = BuildMatrix(totalinK, Nverts);
  double **newVZ   = BuildMatrix(totalinK, Nverts);
  
  int cnt = 0;
  for(p=0;p<nprocs;++p){
    MPI_Irecv(newEToV[cnt], Nverts*inK[p], MPI_INT, p, 666+p, MPI_COMM_WORLD,
	      inrequests+p);
    MPI_Irecv(newVX[cnt], Nverts*inK[p], MPI_DOUBLE, p, 1666+p, MPI_COMM_WORLD,
	      xinrequests+p);
    MPI_Irecv(newVY[cnt], Nverts*inK[p], MPI_DOUBLE, p, 2666+p, MPI_COMM_WORLD,
	      yinrequests+p);
    MPI_Irecv(newVZ[cnt], Nverts*inK[p], MPI_DOUBLE, p, 3666+p, MPI_COMM_WORLD,
	      zinrequests+p);
    cnt = cnt + inK[p];
  }

  for(p=0;p<nprocs;++p){
    int cnt = 0;
    outlist[p]  = BuildIntVector(Nverts*outK[p]);
    xoutlist[p]  = BuildVector(Nverts*outK[p]);
    youtlist[p]  = BuildVector(Nverts*outK[p]);
    zoutlist[p]  = BuildVector(Nverts*outK[p]);

    for(k=0;k<Klocal;++k)
      if(part[k]==p){
	for(v=0;v<Nverts;++v){
	  outlist[p][cnt] = EToV[k][v]; 
	  xoutlist[p][cnt] = VX[k][v];
	  youtlist[p][cnt] = VY[k][v];
	  zoutlist[p][cnt] = VZ[k][v];
	  ++cnt;
	}
      }
    
    MPI_Isend(outlist[p], Nverts*outK[p], MPI_INT, p, 666+procid, MPI_COMM_WORLD, 
	      outrequests+p);
    MPI_Isend(xoutlist[p], Nverts*outK[p], MPI_DOUBLE, p, 1666+procid, MPI_COMM_WORLD, 
	      xoutrequests+p);
    MPI_Isend(youtlist[p], Nverts*outK[p], MPI_DOUBLE, p, 2666+procid, MPI_COMM_WORLD, 
	      youtrequests+p);
    MPI_Isend(zoutlist[p], Nverts*outK[p], MPI_DOUBLE, p, 3666+procid, MPI_COMM_WORLD, 
	      zoutrequests+p);
  }

  MPI_Status *instatus = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));
  MPI_Status *outstatus = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));

  MPI_Waitall(nprocs,  inrequests, instatus);
  MPI_Waitall(nprocs, xinrequests, instatus);
  MPI_Waitall(nprocs, yinrequests, instatus);
  MPI_Waitall(nprocs, zinrequests, instatus);

  MPI_Waitall(nprocs,  outrequests, outstatus);
  MPI_Waitall(nprocs, xoutrequests, outstatus);
  MPI_Waitall(nprocs, youtrequests, outstatus);
  MPI_Waitall(nprocs, zoutrequests, outstatus);

  if(mesh->GX!=NULL){
    DestroyMatrix(mesh->GX);
    DestroyMatrix(mesh->GY);
    DestroyMatrix(mesh->GZ);
    DestroyIntMatrix(mesh->EToV);
  }

  mesh->GX = newVX;
  mesh->GY = newVY;
  mesh->GZ = newVZ;
  mesh->EToV = newEToV;
  mesh->K =  totalinK;

  for(p=0;p<nprocs;++p){
    if(outlist[p]){
      free(outlist[p]);
      free(xoutlist[p]);
      free(youtlist[p]);
      free(zoutlist[p]);
    }
  }

  free(outK);
  free(inK);
  
  free(inrequests);
  free(outrequests);

  free(xinrequests);
  free(xoutrequests);
  free(yinrequests);
  free(youtrequests);
  free(zinrequests);
  free(zoutrequests);
  free(instatus);
  free(outstatus);

}
