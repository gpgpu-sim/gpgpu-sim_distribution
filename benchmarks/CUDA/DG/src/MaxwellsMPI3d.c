
#include "mpi.h"
#include <parmetisbin.h>

#include "fem.h"

static MPI_Request *mpi_out_requests = NULL;
static MPI_Request *mpi_in_requests  = NULL;

static int Nmess = 0;

void MaxwellsMPISend3d(Mesh *mesh){

  int p;

  int procid = mesh->procid;
  int nprocs = mesh->nprocs;

  MPI_Status status;

  if(mpi_out_requests==NULL){
    mpi_out_requests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
    mpi_in_requests  = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
  }

#ifdef CUDA
  get_partial_gpu_data3d(mesh->parNtotalout, mesh->c_parmapOUT, mesh->f_outQ);
#endif

  /* non-blocked send/recv partition surface data */
  Nmess = 0;

  /* now send piece to each proc */
  int sk = 0;
  for(p=0;p<nprocs;++p){

    if(p!=procid){
      int Nout = mesh->Npar[p]*p_Nfields*p_Nfp;
      if(Nout){
	/* symmetric communications (different ordering) */
	MPI_Isend(mesh->f_outQ+sk, Nout, MPI_FLOAT, p, 6666+p,      MPI_COMM_WORLD, mpi_out_requests +Nmess);
	MPI_Irecv(mesh->f_inQ+sk,  Nout, MPI_FLOAT, p, 6666+procid, MPI_COMM_WORLD,  mpi_in_requests +Nmess);
	sk+=Nout;
	++Nmess;
      }
    }
  }

}


void MaxwellsMPIRecv3d(Mesh *mesh, float *c_partQ){
  int p, n;
  int nprocs = mesh->nprocs;

  MPI_Status *instatus  = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));
  MPI_Status *outstatus = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));

  MPI_Waitall(Nmess, mpi_in_requests, instatus);
  
#ifdef CUDA
  cudaMemcpy(c_partQ, mesh->f_inQ, mesh->parNtotalout*sizeof(float), cudaMemcpyHostToDevice);
#endif

  MPI_Waitall(Nmess, mpi_out_requests, outstatus);

  free(outstatus);
  free(instatus);

}

