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

void cpuenergy(voldim3i grid,
	       int numatoms,
	       float gridspacing,
	       int k,
	       const float *atoms,
	       float *energygrid)
{
  float energy;                   /* Energy of current grid point */
  float x,y,z;                    /* Coordinates of current grid point */
  int i,j,n;                      /* Loop counters */
  int atomarrdim = numatoms * 4;

  // printf("\tWorking on plane %i of %ld\n", k, grid.z);
  z = gridspacing * (float) k;

  /* For each x, y grid point in this plane */
  for (j=0; j<grid.y; j++) {
    y = gridspacing * (float) j;

    for (i=0; i<grid.x; i++) {
      x = gridspacing * (float) i;
      energy = 0.0f;

/* help the vectorizer make the right decision */
#if defined(__INTEL_COMPILER)
#pragma vector always
#endif
      /* Calculate the interaction with each atom */
      for (n=0; n<atomarrdim; n+=4) {
        float dx = x - atoms[n  ];
        float dy = y - atoms[n+1];
        float dz = z - atoms[n+2];
        energy += atoms[n+3] / sqrtf(dx*dx + dy*dy + dz*dz);
      }

      energygrid[grid.x*grid.y*k + grid.x*j + i] = energy;
    }
  }
}
