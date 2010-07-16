/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>

#include "cuenergy.h"

//#if UNROLLX != 8
//# error "UNROLLX must be 8"
//#endif

#if BLOCKSIZEX != 16
# error "BLOCKSIZEX must be 16"
#endif

// Max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about.
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.
__constant__ float4 atominfo[MAXATOMS];

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.

__global__ void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
                         + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
                         + xindex;

  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;

  float gridspacing_u = gridspacing * BLOCKSIZEX;

  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;

    float dx1 = coorx - atominfo[atomid].x;
    float dx2 = dx1 + gridspacing_u;

    energyvalx1 += atominfo[atomid].w * (1.0f / sqrtf(dx1*dx1 + dyz2));
    energyvalx2 += atominfo[atomid].w * (1.0f / sqrtf(dx2*dx2 + dyz2));
  }

  energygrid[outaddr]   += energyvalx1;
  energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
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

