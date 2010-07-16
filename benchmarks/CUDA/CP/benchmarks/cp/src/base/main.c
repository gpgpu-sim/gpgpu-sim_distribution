/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * CUDA accelerated coulombic potential grid test code
 *   John E. Stone <johns@ks.uiuc.edu>
 *   http://www.ks.uiuc.edu/~johns/
 *
 * Coulombic potential grid calculation microbenchmark based on the time
 * consuming portions of the 'cionize' ion placement tool.
 */

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "cenergy.h"

/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int
initatoms(float **atombuf, int count, voldim3i volsize, float gridspacing) {
  voldim3i size;
  int i;
  float *atoms;

  srand(54321);			// Ensure that atom placement is repeatable

  atoms = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;

  // compute grid dimensions in angstroms
  size.x = gridspacing * volsize.x;
  size.y = gridspacing * volsize.y;
  size.z = gridspacing * volsize.z;

  for (i=0; i<count; i++) {
    int addr = i * 4;
    atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
    atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
    atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
    atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
  }  

  return 0;
}

/* writeenergy()
 * Write part of the energy array to an output file for verification.
 */
static int
writeenergy(char *filename, float *energy, voldim3i volsize)
{
  FILE *outfile;
  int x, y;

  outfile = fopen(filename, "w");
  if (outfile == NULL) {
    fputs("Cannot open output file\n", stderr);
    return -1;
    }

  /* Print the execution parameters */
  fprintf(outfile, "%d %d %d %d\n", volsize.x, volsize.y, volsize.z, ATOMCOUNT);

  /* Print a checksum */
  {
    double sum = 0.0;

    for (y = 0; y < volsize.y; y++) {
      for (x = 0; x < volsize.x; x++) {
        double t = energy[y*volsize.x+x];
        t = fmax(-20.0, fmin(20.0, t));
    	sum += t;
      }
    }
    fprintf(outfile, "%.4g\n", sum);
  }
  
  /* Print several rows of the computed data */
  for (y = 0; y < 17; y++) {
    for (x = 0; x < volsize.x; x++) {
      int addr = y * volsize.x + x;
      fprintf(outfile, "%.4g ", energy[addr]);
    }
    fprintf(outfile, "\n");
  }

  fclose(outfile);

  return 0;
}

int main(int argc, char** argv) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  float *energy = NULL;		// Output of calculation
  float *atoms = NULL;
  voldim3i volsize;

  // number of atoms to simulate
  int atomcount = ATOMCOUNT;

  // voxel spacing
  const float gridspacing = 0.1;

  // Size of buffer on GPU
  int volmemsz;

  printf("CUDA accelerated coulombic potential microbenchmark\n");
  printf("Original version by John E. Stone <johns@ks.uiuc.edu>\n");
  printf("This version maintained by Chris Rodrigues\n");

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if (parameters->inpFiles[0]) {
    fputs("No input files expected\n", stderr);
    return -1;
  }

  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  // setup energy grid size
  volsize.x = VOLSIZEX;
  volsize.y = VOLSIZEY;
  volsize.z = 1;

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, atomcount, volsize, gridspacing))
    return -1;

  // allocate the output array
  volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;
  energy = (float *) malloc(volmemsz);

  // Main computation
  cpuenergy(volsize, atomcount, gridspacing, 0, atoms, energy);

#if 0
    printf("Done\n");
#endif

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  /* Print a subset of the results to a file */
  if (parameters->outFile) {
    if (writeenergy(parameters->outFile, energy, volsize) == -1)
      return -1;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(atoms);
  free(energy);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}



