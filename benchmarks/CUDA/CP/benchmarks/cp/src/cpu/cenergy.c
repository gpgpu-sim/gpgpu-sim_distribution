/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "cenergy.h"

/* Less readable kernel, but it vectorizes with SSE3 and Intel C  */
/* This version precalculates (dy^2 + dz^2) for each atom, before */
/* it proceeds with the innermost loop over X.                    */
/* The loop structure and data are arranged to allow the use of   */
/* SSE SIMD instructions for better performance.                  */
void cpuenergy(voldim3i grid,
	       int numatoms,
	       float gridspacing,
	       int k,
	       const float *atoms,
	       float *energygrid)
{
  float x,y,z;                     /* Coordinates of current grid point */
  int i,j,n;                       /* Loop counters */
  int atomarrdim = numatoms * 4;

  /* Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q... */
  float *xrq = (float *) malloc(3*numatoms * sizeof(float));
  int maxn = numatoms * 3;

  //printf("\tWorking on plane %i of %ld\n", k, grid.z);
  z = gridspacing * (float) k;

  for (j=0; j<grid.y; j++) {
    y = gridspacing * (float) j;
    long int voxaddr = grid.x*grid.y*k + grid.x*j;

    /* Prebuild a table of dy and dz values on a per atom basis */
    for (n=0; n<numatoms; n++) {
      int addr3 = n*3;
      int addr4 = n*4;
      float dy = y - atoms[addr4 + 1];
      float dz = z - atoms[addr4 + 2];
      xrq[addr3    ] = atoms[addr4];
      xrq[addr3 + 1] = dz*dz + dy*dy;
      xrq[addr3 + 2] = atoms[addr4 + 3];
    }

    x=0.0f;
/* help the vectorizer make the right decision */
#if defined(__INTEL_COMPILER)
#pragma vector always
#endif
    /* walk through more than one grid point at a time */
    for (i=0; i<grid.x; i+=8) {
      float energy1 = 0.0f;           /* Energy of first grid point */
      float energy2 = 0.0f;           /* Energy of second grid point */
      float energy3 = 0.0f;           /* Energy of third grid point */
      float energy4 = 0.0f;           /* Energy of fourth grid point */
      float energy5 = 0.0f;           /* Energy of fourth grid point */
      float energy6 = 0.0f;           /* Energy of fourth grid point */
      float energy7 = 0.0f;           /* Energy of fourth grid point */
      float energy8 = 0.0f;           /* Energy of fourth grid point */
      x = gridspacing * (float) i;

/* help the vectorizer make the right decision */
#if defined(__INTEL_COMPILER)
#pragma vector always
#endif
      /* Calculate the interaction with each atom */
      /* SSE allows simultaneous calculations of  */
      /* multiple iterations                      */
      /* 6 flops per grid point */
      for (n=0; n<maxn; n+=3) {
        float dy2pdz2 = xrq[n + 1];
        float q = xrq[n + 2];

        float dx1 = x - xrq[n];
        energy1 += q / sqrtf(dx1*dx1 + dy2pdz2);

        float dx2 = dx1 + gridspacing;
        energy2 += q / sqrtf(dx2*dx2 + dy2pdz2);

        float dx3 = dx2 + gridspacing;
        energy3 += q / sqrtf(dx3*dx3 + dy2pdz2);

        float dx4 = dx3 + gridspacing;
        energy4 += q / sqrtf(dx4*dx4 + dy2pdz2);

        float dx5 = dx4 + gridspacing;
        energy5 += q / sqrtf(dx5*dx5 + dy2pdz2);

        float dx6 = dx5 + gridspacing;
        energy6 += q / sqrtf(dx6*dx6 + dy2pdz2);

        float dx7 = dx6 + gridspacing;
        energy7 += q / sqrtf(dx7*dx7 + dy2pdz2);

        float dx8 = dx7 + gridspacing;
        energy8 += q / sqrtf(dx8*dx8 + dy2pdz2);
      }

      energygrid[voxaddr + i] = energy1;
      if (i+1 < grid.x)
        energygrid[voxaddr + i + 1] = energy2;
      if (i+2 < grid.x)
        energygrid[voxaddr + i + 2] = energy3;
      if (i+3 < grid.x)
        energygrid[voxaddr + i + 3] = energy4;
      if (i+4 < grid.x)
        energygrid[voxaddr + i + 4] = energy5;
      if (i+5 < grid.x)
        energygrid[voxaddr + i + 5] = energy6;
      if (i+6 < grid.x)
        energygrid[voxaddr + i + 6] = energy7;
      if (i+7 < grid.x)
        energygrid[voxaddr + i + 7] = energy8;
    }
  }

  free(xrq);
}

