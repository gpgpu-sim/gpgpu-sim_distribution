
/* default order */
#ifndef p_N
#define p_N 6
#endif

#define NODETOL   1e-4
#ifdef NDG2d
#define p_Nfp     (p_N+1)
#define p_Np      ((p_N+1)*(p_N+2)/2)
#define p_Nfields 3
#define p_Nfaces  3
#endif

#ifdef NDG3d
#define p_Nfp     ((p_N+1)*(p_N+2)/2)
#define p_Np      ((p_N+1)*(p_N+2)*(p_N+3)/6)
#define p_Nfields 6
#define p_Nfaces  4
#endif

#define BSIZE   (16*((p_Np+15)/16))
//#define BSIZE p_Np

#define max(a,b)  ( (a>b)?a:b )
#define min(a,b)  ( (a<b)?a:b )

/* list of vertices on each edge */

typedef struct foo {

  int nprocs; /* number of processes */
  int procid; /* number of this process */
  
  int Nverts; /* number of vertices per element */
  int Nfaces; /* number of faces per element */
  int Nedges; /* number of faces per element (3d only) */

  int Nv;     /* number of mesh vertices */
  int K;      /* number of mesh elements */
  int **EToV; /* element to global vertex list  */
  int **EToG; /* element to global face number list */
  int **EToS; /* element to global edge number list */
  
  int **EToE; /* element to neighbor element (elements numbered by their proc) */
  int **EToF; /* element to neighbor face    (element local number 0,1,2) */
  int **EToP; /* element to neighbor proc    (element prc number 0,1,.., nprocs-1) */

  int **localEToV; /* element to process local vertex list */
  int localNunique; /* number of unique nodes on this process */

  int *bcflag; /* vector. entry n is 1 if vertex n is on a boundary */

  double **GX; /* x-coordinates of element vertices */
  double **GY; /* y-coordinates of element vertices */
  double **GZ; /* z-coordinates of element vertices (3d) */

  /* high order node info */
  int   **Fmask; /* face node numbers in element volume data */

  double  *r,   *s,   *t;  /* (r,s) coordinates of reference nodes */
  double **Dr, **Ds, **Dt; /* local nodal derivative matrices */
  double **LIFT;     /* local lift matrix */

  double **x; /* x-coordinates of element nodes */
  double **y; /* y-coordinates of element nodes */
  double **z; /* z-coordinates of element nodes (3d) */

  /* DG STUFF (EXPERIMENTAL) */
  int     *vmapM; /* volume id of -ve trace of face node */
  int     *vmapP; /* volume id of +ve trace of face node */
  int     *Npar;  /* # of nodes to send recv to each proc */
  int    **parK;  /* element of parallel nodes */
  int    **parF;  /* face of parallel nodes */

  /* MPI stuff */
  int    parNtotalout; /* total number of nodes to send */
  int   *parmapOUT;    /* list of nodes to send out */
  int   *c_parmapOUT;  /* device list */

  float *f_outQ, *f_inQ; /* MPI data buffers */

  /* float version of operators */
  float *f_Dr, *f_Ds, *f_Dt; 
  float *f_LIFT;   

  /* float geometric info */
  float   *vgeo;    /* geometric factors */
  float   *surfinfo; 

  /* float field storage (CPU) */
  float  *f_Q, *f_rhsQ, *f_resQ;

  /* time stepping constants */ 
  double *rk4a, *rk4b, *rk4c;

}Mesh;

