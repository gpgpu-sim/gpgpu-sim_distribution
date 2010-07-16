//
// Notes:
//
// 1) strategy: one thread per node in the 2D block;
//    after initialisation it marches in the k-direction
//    working with 3 planes of data at a time
//
// 2) each thread also loads in data for at most one halo node;
//    assumes the number of halo nodes is not more than the
//    number of interior nodes
//
// 3) corner halo nodes are included because they are needed
//    for more general applications with cross-derivatives
//
// 4) could try double-buffering in the future fairly easily
//


// definition to use efficient __mul24 intrinsic

#define INDEX(i,j,j_off)  (i +__mul24(j,j_off))


// device code

__global__ void GPU_laplace3d(int NX, int NY, int NZ, int pitch, 
                              float *d_u1, float *d_u2)
{
  int   indg, indg_h, indg0;
  int   i, j, k, ind, ind_h, halo, active;
  float u2, sixth=1.0f/6.0f;

  int NXM1 = NX-1;
  int NYM1 = NY-1;
  int NZM1 = NZ-1;

  //
  // define local array offsets
  //

#define IOFF  1
#define JOFF (BLOCK_X+2)
#define KOFF (BLOCK_X+2)*(BLOCK_Y+2)
  __shared__ float u1[3*KOFF];


  //
  // first set up indices for halos
  //

  k    = threadIdx.x + threadIdx.y*BLOCK_X;
  halo = k < 2*(BLOCK_X+BLOCK_Y+2);

  if (halo) {
    if (threadIdx.y<2) {               // y-halos (coalesced)
      i = threadIdx.x;
      j = threadIdx.y*(BLOCK_Y+1) - 1;
    }
    else {                             // x-halos (not coalesced)
      i = (k%2)*(BLOCK_X+1) - 1;
      j =  k/2 - BLOCK_X - 1;
    }

    ind_h  = INDEX(i+1,j+1,JOFF) + KOFF;

    i      = INDEX(i,blockIdx.x,BLOCK_X);   // global indices
    j      = INDEX(j,blockIdx.y,BLOCK_Y);
    indg_h = INDEX(i,j,pitch);

    halo   =  (i>=0) && (i<NX) && (j>=0) && (j<NY);
  }

  //
  // then set up indices for main block
  //

  i    = threadIdx.x;
  j    = threadIdx.y;
  ind  = INDEX(i+1,j+1,JOFF) + KOFF;

  i    = INDEX(i,blockIdx.x,BLOCK_X);     // global indices
  j    = INDEX(j,blockIdx.y,BLOCK_Y);
  indg = INDEX(i,j,pitch);

  active = (i<NX) && (j<NY);

  //
  // read initial plane of u1 array
  //

  if (active) u1[ind+KOFF] = d_u1[indg];
  if (halo) u1[ind_h+KOFF] = d_u1[indg_h];

  //
  // loop over k-planes
  //

  for (k=0; k<NZ; k++) {

    // move two planes down and read in new plane k+1

    if (active) {
      indg0 = indg;
      indg  = INDEX(indg,NY,pitch);
      u1[ind-KOFF] = u1[ind];
      u1[ind]      = u1[ind+KOFF];
      if (k<NZM1)
        u1[ind+KOFF] = d_u1[indg];
    }

    if (halo) {
      indg_h = INDEX(indg_h,NY,pitch);
      u1[ind_h-KOFF] = u1[ind_h];
      u1[ind_h]      = u1[ind_h+KOFF];
      if (k<NZM1)
        u1[ind_h+KOFF] = d_u1[indg_h];
    }

    __syncthreads();

  //
  // perform Jacobi iteration to set values in u2
  //

    if (active) {
      if (i==0 || i==NXM1 || j==0 || j==NYM1 || k==0 || k==NZM1) {
        u2 = u1[ind];          // Dirichlet b.c.'s
      }
      else {
        u2 = ( u1[ind-IOFF] + u1[ind+IOFF]
             + u1[ind-JOFF] + u1[ind+JOFF]
             + u1[ind-KOFF] + u1[ind+KOFF] ) * sixth;
      }
      d_u2[indg0] = u2;
    }

    __syncthreads();

  }
}
