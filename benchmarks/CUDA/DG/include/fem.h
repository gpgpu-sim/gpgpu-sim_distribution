#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Mesh.h"

/* prototypes for storage functions (Utils.c) */
double **BuildMatrix(int Nrows, int Ncols);
double  *BuildVector(int Nrows);
int    **BuildIntMatrix(int Nrows, int Ncols);
int     *BuildIntVector(int Nrows);

double **DestroyMatrix(double **);
double  *DestroyVector(double *);
int    **DestroyIntMatrix(int **);
int     *DestroyIntVector(int *);

void PrintMatrix(char *message, double **A, int Nrows, int Ncols);
void SaveMatrix(char *filename, double **A, int Nrows, int Ncols);

/* geometric/mesh functions */
Mesh *ReadMesh2d(char *filename);
Mesh *ReadMesh3d(char *filename);

void  PrintMesh ( Mesh *mesh );

void Normals2d(Mesh *mesh, int k, double *nx, double *ny, double *sJ);
void Normals3d(Mesh *mesh, int k, 
	       double *nx, double *ny, double *nz, double *sJ);

void GeometricFactors2d(Mesh *mesh, int k,
		      double *drdx, double *dsdx, double *drdy, double *dsdy, 
		      double *J);

void GeometricFactors3d(Mesh *mesh, int k,
		      double *drdx, double *dsdx, double *dtdx, 
		      double *drdy, double *dsdy, double *dtdy, 
		      double *drdz, double *dsdz, double *dtdz, 
			double *J);

/* start up */
void StartUp2d(Mesh *mesh);
void StartUp3d(Mesh *mesh);

void BuildMaps2d(Mesh *mesh);
void BuildMaps3d(Mesh *mesh);

/* Parallel */
void LoadBalance2d(Mesh *mesh);
void LoadBalance3d(Mesh *mesh);

void FacePair2d(Mesh *mesh, int *maxNv);
void FacePair3d(Mesh *mesh, int *maxNv);

void ParallelPairs(void *objs, int Nmyobjs, int sizeobj,
		   int  (*numget)(const void *),
		   void (*numset)(const void *, int ),
		   int  (*procget)(const void *),
		   void (*marry)(const void *, const void *),
		   int (*compare_objs)(const void *, const void *));

void MaxwellsMPISend2d(Mesh *mesh);
void MaxwellsMPIRecv2d(Mesh *mesh, float *c_partQ);

void MaxwellsMPISend3d(Mesh *mesh);
void MaxwellsMPIRecv3d(Mesh *mesh, float *c_partQ);


/* GPU functions */
void gpu_set_data2d(int K,
		    double *d_Hx, double *d_Hy, double *d_Ez);
void gpu_set_data3d(int K,
		    double *d_Hx, double *d_Hy, double *d_Hz,
		    double *d_Ex, double *d_Ey, double *d_Ez);

void gpu_get_data2d(int K,
		    double *d_Hx, double *d_Hy, double *d_Ez);
void gpu_get_data3d(int K,
		    double *d_Hx, double *d_Hy, double *d_Hz,
		    double *d_Ex, double *d_Ey, double *d_Ez);

double InitGPU2d(Mesh *mesh, int Nfields);
double InitGPU3d(Mesh *mesh, int Nfields);

/* CPU functions */
void cpu_set_data2d(Mesh *mesh, double *Hx, double *Hy, double *Ez);
void cpu_set_data3d(Mesh *mesh, double *Hx, double *Hy, double *Hz,
		                double *Ex, double *Ey, double *Ez);

double InitCPU2d(Mesh *mesh, int Nfields);
double InitCPU3d(Mesh *mesh, int Nfields);

/* Maxwells functions */
void MaxwellsKernel2d(Mesh *mesh, float frka, float frkb, float fdt);
void MaxwellsKernel3d(Mesh *mesh, float frka, float frkb, float fdt);

void MaxwellsRHS2d(Mesh *mesh, float frka, float frkb, float fdt);
void MaxwellsRHS3d(Mesh *mesh, float frka, float frkb, float fdt);

void MaxwellsRun2d(Mesh *mesh, double FinalTime, double dt);
void MaxwellsRun3d(Mesh *mesh, double FinalTime, double dt);

/* CUDA  headers */
#ifdef CUDA
#include "cuda.h"
#include "cuda_runtime_api.h"
#endif

