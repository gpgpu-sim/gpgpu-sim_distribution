#include "mpi.h"
#include "fem.h"

void MaxwellsRun3d(Mesh *mesh, double FinalTime, double dt){

  double time = 0;
  int    INTRK, tstep=0;

  double mpitime0 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  dt = 0.001;

  /* outer time step loop  */
  while (time<FinalTime){

    /* adjust final step to end exactly at FinalTime */
    if (time+dt > FinalTime) { dt = FinalTime-time; }
    
    for (INTRK=1; INTRK<=5; ++INTRK) {
      
      /* compute rhs of TM-mode MaxwellsGPU's equations */
      const float fdt = dt;
      const float fa = (float)mesh->rk4a[INTRK-1];
      const float fb = (float)mesh->rk4b[INTRK-1];

#ifdef CUDA
      MaxwellsKernel3d(mesh, fa, fb, fdt);
#else
      MaxwellsRHS3d(mesh, fa, fb, fdt);
#endif
    }
    
    time += dt;     /* increment current time */
    tstep++;        /* increment timestep */
  }    

#ifdef CUDA
  cudaThreadSynchronize();
#endif

  double flopsV = p_Np*p_Np*36 + p_Np*66; /* V3 */
  double flopsS = p_Nfp*p_Nfaces*(15 + 10 + 36) + p_Np*(p_Nfaces*p_Nfp*12 + 6);
  double flopsR = p_Np*p_Nfields*4;

  int Kloc = mesh->K;
  
  double mpitime1 = MPI_Wtime();
  
  double time_total = mpitime1-mpitime0;
  
  MPI_Barrier(MPI_COMM_WORLD);
  printf("%d %d %lg %lg proc, N, time taken, GFLOPS/s (GPU)\n",
	 mesh->procid,
	 p_N, 
	 time_total,
	 5*(tstep)*( (flopsV+flopsS+flopsR)*((double)Kloc/(1.e9*time_total))));
	 

}
