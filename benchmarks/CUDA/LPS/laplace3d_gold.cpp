
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void Gold_laplace3d(int NX, int NY, int NZ, float* u1, float* u2) 
{
  int   i, j, k, ind;
  float sixth=1.0f/6.0f;  // predefining this improves performance more than 10%

  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
                    + u1[ind-NX   ] + u1[ind+NX   ]
                    + u1[ind-NX*NY] + u1[ind+NX*NY] ) * sixth;
        }
      }
    }
  }
}

