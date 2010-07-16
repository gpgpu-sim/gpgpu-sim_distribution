/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>

#include "cuenergy.h"

#if UNROLLX != 1
# error "UNROLLX must be 1"
#endif

__constant__ float4 atominfo[MAXATOMS];

/* This is a reference version of the kernel.  It is simpler and slower
 * than the optimzed version. */

__global__ void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = __umul24(gridDim.x, blockDim.x) * yindex + xindex;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;

  int atomid;
  float energyval=0.0f;

  /* For each atom, compute and accumulate its contribution to energyval
   * for this thread's grid point */
  for (atomid=0; atomid<numatoms; atomid++) {
    float dx = coorx - atominfo[atomid].x;
    float dy = coory - atominfo[atomid].y;
    float r_1 = 1.0f / sqrtf(dx*dx + dy*dy + atominfo[atomid].z);
    energyval += atominfo[atomid].w * r_1;
  }

  energygrid[outaddr] += energyval;
}

// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.

int copyatomstoconstbuf(float *atoms, int count, float zplane) {
  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
  CUERR // check and clear any existing errors

  return 0;
}

