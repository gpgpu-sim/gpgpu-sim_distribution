#include "fem.h"

/* some very basic memory allocation routines */

/* row major storage for a 2D matrix array */
double **BuildMatrix(int Nrows, int Ncols){
  int n;
  double **A = (double**) calloc(Nrows, sizeof(double*));

  A[0] = (double*) calloc(Nrows*Ncols, sizeof(double));

  for(n=1;n<Nrows;++n){
    A[n] = A[n-1]+ Ncols;
  }

  return A;
}
    
double *BuildVector(int Nrows){

  double *A = (double*) calloc(Nrows, sizeof(double));

  return A;
}

/* row major storage for a 2D matrix array */
int **BuildIntMatrix(int Nrows, int Ncols){
  int n;
  int **A = (int**) calloc(Nrows, sizeof(int*));

  A[0] = (int*) calloc(Nrows*Ncols, sizeof(int));

  for(n=1;n<Nrows;++n){
    A[n] = A[n-1]+ Ncols;
  }

  return A;
}

int *BuildIntVector(int Nrows){

  int *A = (int*) calloc(Nrows, sizeof(int));

  return A;
}

double *DestroyVector(double *v){
  free(v);
  return NULL;
}

double **DestroyMatrix(double **A){
  free(A[0]);
  free(A);

  return NULL;
}

int *DestroyIntVector(int *v){
  free(v);
  return NULL;
}

int **DestroyIntMatrix(int **A){
  free(A[0]);
  free(A);

  return NULL;
}

void PrintMatrix(char *message, double **A, int Nrows, int Ncols){
  int n,m;

  printf("%s\n", message);
  for(n=0;n<Nrows;++n){
    for(m=0;m<Ncols;++m){
      printf(" %g ", A[n][m]);
    }
    printf(" \n");
  }
}


void SaveMatrix(char *filename, double **A, int Nrows, int Ncols){
  int n,m;

  FILE *fp = fopen(filename, "w");

  for(n=0;n<Nrows;++n){
    for(m=0;m<Ncols;++m){
      fprintf(fp, " %g ", A[n][m]);
    }
    fprintf(fp, " \n");
  }
  
  fclose(fp);
}


int trianglebase(Mesh *mesh, int k){

  double x1 = mesh->GX[k][0];
  double x2 = mesh->GX[k][1];
  double x3 = mesh->GX[k][2];

  double y1 = mesh->GY[k][0];
  double y2 = mesh->GY[k][1];
  double y3 = mesh->GY[k][2];

  double d1 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
  double d2 = (x2-x3)*(x2-x3) + (y2-y3)*(y2-y3);
  double d3 = (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1);

  /* find maximum length face */
  if(d1>=d2 && d1>=d3)
    return 0;
  else if(d2>=d3)
    return 1;
  
  return 2;
}


int tetbase(Mesh *mesh, int k){

  double x1 = mesh->GX[k][0];
  double x2 = mesh->GX[k][1];
  double x3 = mesh->GX[k][2];
  double x4 = mesh->GX[k][3];

  double y1 = mesh->GY[k][0];
  double y2 = mesh->GY[k][1];
  double y3 = mesh->GY[k][2];
  double y4 = mesh->GY[k][3];

  double z1 = mesh->GZ[k][0];
  double z2 = mesh->GZ[k][1];
  double z3 = mesh->GZ[k][2];
  double z4 = mesh->GZ[k][3];

  double d1 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
  double d2 = (x2-x3)*(x2-x3) + (y2-y3)*(y2-y3) + (z2-z3)*(z2-z3);
  double d3 = (x3-x4)*(x3-x4) + (y3-y4)*(y3-y4) + (z3-z4)*(z3-z4);
  double d4 = (x4-x1)*(x4-x1) + (y4-y1)*(y4-y1) + (z4-z1)*(z4-z1);

  /* find maximum length face */
  if(d1>=d2 && d1>=d3 && d1>=d4)
    return 0;
  else if(d2>=d3 && d2>=d4)
    return 1;
  else if(d3>=d4)
    return 2;
  
  return 3;
}
