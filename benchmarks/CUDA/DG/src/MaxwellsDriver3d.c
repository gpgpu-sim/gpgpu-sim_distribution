#include "mpi.h"
#include "fem.h"

main(int argc, char **argv){

  Mesh *mesh;
  int procid, nprocs, maxNv;
  int k,n, sk=0;
  double minEz, maxEz, gminEz, gmaxEz;

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* assign gpu */
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

#ifdef CUDA  
  cudaSetDevice((procid+2)%4);
#endif

  /* (parallel) read part of fem mesh from file */
  mesh = ReadMesh3d(argv[1]);

  /* perform load balancing */
  LoadBalance3d(mesh);

  /* find element-element connectivity */
  FacePair3d(mesh, &maxNv);

  /* perform start up */
  StartUp3d(mesh);

  /* field storage (double) */
  double *Hx = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hy = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hz = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ex = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ey = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ez = (double*) calloc(mesh->K*p_Np, sizeof(double));

  /* initial conditions */
  for(k=0;k<mesh->K;++k){
    for(n=0;n<p_Np;++n) {
      Hx[sk] = 0;
      Hy[sk] = 0;
      Hz[sk] = 0;
      Ex[sk] = 0;
      Ey[sk] = 0;
      Ez[sk] = cos(M_PI*mesh->x[k][n])*cos(M_PI*mesh->y[k][n])*cos(M_PI*mesh->z[k][n]);
      ++sk;
    }
  }

  double dt, gdt;

#ifdef CUDA  
  /* initialize GPU info */
  dt = InitGPU3d(mesh, p_Nfields);

  /* load data onto GPU */
  gpu_set_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);
#else
  /* initialize GPU info */
  dt = InitCPU3d(mesh, p_Nfields);

  /* load data onto CPU float storage */
  cpu_set_data3d(mesh, Hx, Hy, Hz, Ex, Ey, Ez);
#endif

  MPI_Allreduce(&dt, &gdt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  dt = .5*gdt/((p_N+1)*(p_N+1));

  if(mesh->procid==0)
    printf("dt = %f\n", dt);

  double FinalTime = .00050;

  /* solve */
  MaxwellsRun3d(mesh, FinalTime, dt); 

#ifdef CUDA
  /* unload data from GPU */
  gpu_get_data3d(mesh->K, Hx, Hy, Hz, Ex, Ey, Ez);
#else
  cpu_get_data3d(mesh, Hx, Hy, Hz, Ex, Ey, Ez);
#endif

  /* find maximum & minimum values for Ez */
  minEz=Ez[0], maxEz=Ez[0];

  for(n=0;n<mesh->K*p_Np;++n) {
    minEz = (minEz>Ez[n])?Ez[n]:minEz;
    maxEz = (maxEz<Ez[n])?Ez[n]:maxEz;
    printf("%e\n", Ez[n]);
  }

  MPI_Reduce(&minEz, &gminEz, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxEz, &gmaxEz, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(procid==0)
    printf("t=%f Ez in [ %f, %f ] \n", FinalTime, gminEz, gmaxEz );

  /* nicely stop MPI */
  MPI_Finalize();

  /* end game */
  exit(1);
}
