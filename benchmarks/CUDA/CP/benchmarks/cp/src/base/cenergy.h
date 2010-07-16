/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* Size of the benchmark problem.  The GPU can run larger problems in a
 * reasonable time.
 *
 * For VOLSIZEX, VOLSIZEY, size 1024 is suitable for a few seconds of
 * GPU computation and size 128 is suitable for a few seconds of
 * CPU computation.
 *
 * For ATOMCOUNT, 100000 is suitable for GPU computation and 10000 is
 * suitable for CPU computation.
 */
#define VOLSIZEX 512
#define VOLSIZEY 512
#define ATOMCOUNT 40000

/* The main compute kernel. */
void cpuenergy(voldim3i grid,
	       int numatoms,
	       float gridspacing,
	       int k,
	       const float *atoms,
	       float *energygrid);
