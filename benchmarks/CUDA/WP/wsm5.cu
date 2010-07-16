#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas.h"

#define IDEBUG 12
#define JDEBUG 0

#ifndef CRAY
# ifdef NOUNDERSCORE
#      define WSM5_HOST wsm5_host
#      define WSM5_HOST_2 wsm5_host_2
#      define WSM5_GPU_INIT wsm5_gpu_init
#      define GET_WSM5_GPU_LEVELS get_wsm5_gpu_levels
# else
#   ifdef F2CSTYLE
#      define WSM5_HOST wsm5_host__
#      define WSM5_HOST_2 wsm5_host_2__
#      define WSM5_GPU_INIT wsm5_gpu_init__
#      define GET_WSM5_GPU_LEVELS get_wsm5_gpu_levels__
#   else
#      define WSM5_HOST wsm5_host_
#      define WSM5_HOST_2 wsm5_host_2_
#      define WSM5_GPU_INIT wsm5_gpu_init_
#      define GET_WSM5_GPU_LEVELS get_wsm5_gpu_levels_
#   endif
# endif
#endif

#define I2(i,j,m) ((i)+((j)*(m)))
#define I3(i,j,m,k,n) (I2(i,j,m)+((k)*(m)*(n)))

#if 1
# define TODEV(A,s) float *A##_d;cudaMalloc((void**)&A##_d,((s))*sizeof(float));cudaMemcpy(A##_d,A,(s)*sizeof(float),cudaMemcpyHostToDevice);
# define FROMDEV(A,s) cudaMemcpy(A,A##_d,(s)*sizeof(float),cudaMemcpyDeviceToHost);
# define CLNUP(A) cudaFree(A##_d)
#else
# define TODEV(A,s) s1=rsl_internal_microclock_() ; float *A##_d;cudaMalloc((void**)&A##_d,((s))*sizeof(float));cudaMemcpy(A##_d,A,(s)*sizeof(float),cudaMemcpyHostToDevice); e1=rsl_internal_microclock_() ; fprintf(stderr,"TODEV %d\n",e1-s1) 
# define FROMDEV(A,s) s1=rsl_internal_microclock_() ; cudaMemcpy(A,A##_d,(s)*sizeof(float),cudaMemcpyDeviceToHost); e1=rsl_internal_microclock_() ; fprintf(stderr,"FROMDEV %d\n",e1-s1) 
# define CLNUP(A) s1=rsl_internal_microclock_() ; cudaFree(A##_d) ; e1=rsl_internal_microclock_() ; fprintf(stderr,"Free %d\n",e1-s1)
#endif

#if FLOAT_4==4
#define TODEV2(A)s1=rsl_internal_microclock_();\
float*A##_d;\
cudaMalloc((void**)&A##_d,(dipe*djpe*sizeof(float)));\
for(j=*jps-1;j<=*jpe-1;j++){\
  for(i=*ips-1;i<=*ipe-1;i++){\
    bigbuf[I2(i-*ips+1,j-*jps+1,dipe)]=\
      A[I2(i-*ims+1,j-*jms+1,(*ime-*ims+1))];\
}}\
cudaMemcpy(A##_d,bigbuf,(dipe*djpe)*sizeof(float),cudaMemcpyHostToDevice);\
e1=rsl_internal_microclock_();fprintf(stderr,"TODEV2 %d\n",e1-s1);

#define TODEV3(A)s1=rsl_internal_microclock_();\
float*A##_d;\
cudaMalloc((void**)&A##_d,(dipe*djpe*dkpe*sizeof(float)));\
for(j=*jps-1;j<=*jpe-1;j++){\
 for(k=*kps-1;k<=*kpe-1;k++){\
  for(i=*ips-1;i<=*ipe-1;i++){\
   bigbuf[I3(i-*ips+1,k-*kps+1,dipe,j-*jps+1,dkpe)]=\
     A[I3(i-*ims+1,k-*kms+1,*ime-*ims+1,j-*jms+1,*kme-*kms+1)];\
}}}\
cudaMemcpy(A##_d,bigbuf,(dipe*djpe*dkpe)*sizeof(float),cudaMemcpyHostToDevice);\
e1=rsl_internal_microclock_();fprintf(stderr,"TODEV3 %d\n",e1-s1);

// for debugging only
#define TODEV3a(A)s1=rsl_internal_microclock_();\
float*A##_d;\
cudaMalloc((void**)&A##_d,(dipe*djpe*dkpe*sizeof(float)));\
for(j=*jps-1;j<=*jpe-1;j++){\
 for(k=*kps-1;k<=*kpe-1;k++){\
  for(i=*ips-1;i<=*ipe-1;i++){\
   bigbuf[I3(i-*ips+1,k-*kps+1,dipe,j-*jps+1,dkpe)]=\
     A[I3(i-*ims+1,k-*kms+1,*ime-*ims+1,j-*jms+1,*kme-*kms+1)];\
if (i==*ips-1){\
   fprintf(stderr,"There %d %d %d (%d)| %d %d %d (%d)| %f\n",\
                   i-*ips+1,k-*kps+1,j-*jps+1,I3(i-*ips+1,k-*kps+1,dipe,j-*jps+1,dkpe),\
                   i-*ims+1,k-*kms+1,j-*jms+1,I3(i-*ims+1,k-*kms+1,*ime-*ims+1,j-*jms+1,*kme-*kms+1),\
                   bigbuf[I3(i-*ips+1,k-*kps+1,dipe,j-*jps+1,dkpe)]);\
   A[I3(i-*ims+1,k-*kms+1,*ime-*ims+1,j-*jms+1,*kme-*kms+1)]=199.;\
}\
}}}\
cudaMemcpy(A##_d,bigbuf,(dipe*djpe*dkpe)*sizeof(float),cudaMemcpyHostToDevice);\
e1=rsl_internal_microclock_();fprintf(stderr,"TODEV3 %d\n",e1-s1);


#define FROMDEV2(A) s1=rsl_internal_microclock_();\
cudaMemcpy(bigbuf,A##_d,dipe*djpe*sizeof(float),cudaMemcpyDeviceToHost);\
for(j=*jps-1;j<=*jpe-1;j++){\
  for(i=*ips-1;i<=*ipe-1;i++){\
    A[I2(i-*ims+1,j-*jms+1,(*ime-*ims+1))]=\
      bigbuf[I2(i-*ips+1,j-*jps+1,dipe)];\
}}\
e1=rsl_internal_microclock_() ; fprintf(stderr,"FROMDEV2 %d\n",e1-s1);

#define FROMDEV3(A) s1=rsl_internal_microclock_();\
cudaMemcpy(bigbuf,A##_d,dipe*djpe*dkpe*sizeof(float),cudaMemcpyDeviceToHost);\
for(j=*jps-1;j<=*jpe-1;j++){\
 for(k=*kps-1;k<=*kpe-1;k++){\
  for(i=*ips-1;i<=*ipe-1;i++){\
   A[I3(i-*ims+1,k-*kms+1,*ime-*ims+1,j-*jms+1,*kme-*kms+1)]=\
     bigbuf[I3(i-*ips+1,k-*kps+1,dipe,j-*jps+1,dkpe)];\
}}}\
e1=rsl_internal_microclock_();fprintf(stderr,"FROMDEV3 %d\n",e1-s1);
#else
# define TODEV3(A) TODEV(A,d3)
# define TODEV2(A) TODEV(A,d2)
# define FROMDEV3(A) FROMDEV(A,d3)
# define FROMDEV2(A) FROMDEV(A,d2)
#endif

extern "C" int rsl_internal_microclock_() ;

extern __global__ void wsm5_gpu (
                    float *th, float *pii                   //_def_ arg ikj:th,pii
                   ,float *q                                //_def_ arg ikj:q
                   ,float *qc,float *qi,float *qr,float *qs //_def_ arg ikj:qc,qi,qr,qs
                   ,float *den, float *p, float *delz       //_def_ arg ikj:den,p,delz
#ifdef DEBUGGAL_ARRAY
,float *debuggal //_def_ arg ikj:debuggal
#endif
                   ,float *rain,float *rainncv              //_def_ arg ij:rain,rainncv
                   ,float *sr                               //_def_ arg ij:sr
                   ,float *snow,float *snowncv              //_def_ arg ij:snow,snowncv
                   ,float delt
,float* retvals
                   ,int ids, int ide,  int jds, int jde,  int kds, int kde
                   ,int ims, int ime,  int jms, int jme,  int kms, int kme
                   ,int ips, int ipe,  int jps, int jpe,  int kps, int kpe
                         ) ;

extern "C" {

int gethostname(char *name, size_t len);
void bzero(void *s, size_t n);
char *strcpy(char *dest, const char *src);

#define MAXDEVICES 4
#define MAXNODES 16
int
WSM5_GPU_INIT ( int * myproc , int * nproc, int * mydevice )
{
   float x, *x_d ;
   int s, e ;
   int i, dc, m ;
   cudaError_t cerr ;
   char hostname[64] ;
   struct cudaDeviceProp dp ;
//  manage devices if multiheaded
   cudaGetDeviceCount( &dc ) ;
   if ( dc > MAXDEVICES ) 
     { fprintf(stderr, "warning: more than %d devices on node (%d)\n", MAXDEVICES, dc ) ; dc = MAXDEVICES ; }
   fprintf(stderr,"Number of devices on this node: %d\n", dc) ;

   // i = *myproc % dc ;

   i = *mydevice ;
   if ( dc > 0 ) 
   {
      if ( cerr = cudaSetDevice( i ) ) {
         fprintf(stderr,"    non-zero cerr %d\n",cerr) ;
      }
   }
   gethostname( hostname, 64 ) ;
   fprintf(stderr,"Setting device %02d for task %03d on host %s\n",i,*myproc,hostname) ;

   if ( cerr = cudaGetDeviceProperties( &dp, i ) ) {
         fprintf(stderr,"Device %02d: cerr = %d\n", cerr) ;
   } else {
         fprintf(stderr,"Device %02d: name %s\n",i,dp.name) ;
         fprintf(stderr,"Device %02d: mem       %d\n",i,dp.totalGlobalMem) ;
         fprintf(stderr,"Device %02d: smem      %d\n",i,dp.sharedMemPerBlock) ;
         fprintf(stderr,"Device %02d: nreg      %d\n",i,dp.regsPerBlock) ;
         fprintf(stderr,"Device %02d: warp      %d\n",i,dp.warpSize) ;
         fprintf(stderr,"Device %02d: pitch     %d\n",i,dp.memPitch) ;
         fprintf(stderr,"Device %02d: maxthrds  %d\n",i,dp.maxThreadsPerBlock) ;
         fprintf(stderr,"Device %02d: maxtdim   %d %d %d\n",i,dp.maxThreadsDim[0]
                                                             ,dp.maxThreadsDim[1]
                                                             ,dp.maxThreadsDim[2]) ;
         fprintf(stderr,"Device %02d: maxgdim   %d %d %d\n",i,dp.maxGridSize[0]
                                                             ,dp.maxGridSize[1]
                                                             ,dp.maxGridSize[2]) ;
         fprintf(stderr,"Device %02d: clock     %d\n",i,dp.clockRate) ;
         fprintf(stderr,"Device %02d: talign    %d\n",i,dp.textureAlignment) ;
   }

//  do a dummy init to get things going
   s=rsl_internal_microclock_() ;
   cudaMalloc((void **)&x_d,sizeof(float)) ;
   cudaMemcpy(x_d,&x,sizeof(float),cudaMemcpyHostToDevice) ;
   cudaFree(x_d) ;
   e=rsl_internal_microclock_() ;
   fprintf(stderr,"wsm5_init: %d\n",e-s) ;
   return(0) ;
}

int
WSM5_HOST (
                    float *th, float *pii
                   ,float *q
                   ,float *qc, float *qi, float *qr, float *qs
                   ,float *den, float *p, float *delz
#ifdef DEBUGGAL_ARRAY
,float *debuggal
#endif
                   ,float *delt
                   ,float *rain,float *rainncv
                   ,float *sr
                   ,float *snow,float *snowncv
                   ,int *ids, int *ide,  int *jds, int *jde,  int *kds, int *kde
                   ,int *ims, int *ime,  int *jms, int *jme,  int *kms, int *kme
                   ,int *ips, int *ipe,  int *jps, int *jpe,  int *kps, int *kpe          
          )
{
      int i, j, k ;
      float *bigbuf ;
      int s, e, s1, e1, s2, e2 ;
      int d3 = (*ime-*ims+1) * (*jme-*jms+1) * (*kme-*kms+1) ;
      int d2 = (*ime-*ims+1) * (*jme-*jms+1) ;


//fprintf(stderr,"d3 = %d\n",d3) ;
//fprintf(stderr,"d2 = %d\n",d2) ;

#if FLOAT_4 == 4
      int dips = 0 ; int dipe = (((*ipe-*ips+1+3)/4)*4) ;  // round up four
#else
      int dips = 0 ; int dipe = (*ipe-*ips+1) ;
#endif
      int djps = 0 ; int djpe = (*jpe-*jps+1) ;
      int dkps = 0 ; int dkpe = (*kpe-*kps+1) ;

      bigbuf = (float *)malloc( dipe * djpe * dkpe * sizeof(float) ) ;

//fprintf(stderr,"ids %d ide %d jds %d jde %d kds %d kde %d\n",*ids,*ide,*jds,*jde,*kds,*kde) ;
//fprintf(stderr,"ims %d ime %d jms %d jme %d kms %d kme %d\n",*ims,*ime,*jms,*jme,*kms,*kme) ;
//fprintf(stderr,"ips %d ipe %d jps %d jpe %d kps %d kpe %d\n",*ips,*ipe,*jps,*jpe,*kps,*kpe) ;
//fprintf(stderr,"dipe %d djpe %d dkpe %d\n",dipe,djpe,dkpe) ;

      s = rsl_internal_microclock_() ;
      TODEV3(th) ;
      TODEV3(pii) ;
      TODEV3(q) ;
      TODEV3(qc) ;
      TODEV3(qi) ;
      TODEV3(qr) ;
      TODEV3(qs) ;
      TODEV3(den) ;
      TODEV3(p) ;
      TODEV3(delz) ;
#ifdef DEBUGGAL_ARRAY
//TODEV3(debuggal) ;
#endif
      TODEV2(rain) ;
      TODEV2(rainncv) ;
      TODEV2(sr) ;
      TODEV2(snow) ;
      TODEV2(snowncv) ;
float retvals[100] ;
{ int k ;
for (k=0 ;k<*kme-*kms+1;k++) {retvals[k] = 0.; }
}
TODEV(retvals,(*kme-*kms+1)) ;

      int remx, remy ;  // remainder?

      remx = (*ipe-*ips+1) % XXX != 0 ? 1 : 0 ;
      remy = (*jpe-*jps+1) % YYY != 0 ? 1 : 0 ;

      dim3 dimBlock( XXX , YYY ) ;
//      fprintf(stderr,"ipe ips remx jpe jps remy %d %d %d %d %d %d\n",*ipe,*ips,remx,*jpe,*jps,remy) ;
      dim3 dimGrid ( (*ipe-*ips+1) / XXX + remx , (*jpe-*jps+1) / YYY + remy ) ;

      fprintf(stderr,"Call to wsm5_gpu: block dims %d %d\n",dimBlock.x,dimBlock.y) ;
      fprintf(stderr,"Call to wsm5_gpu: grid  dims %d %d\n",dimGrid.x,dimGrid.y) ;

#if 1
//fprintf(stderr,"calling wsm5_gpu \n") ;
//fprintf(stderr,"d %d %d %d %d %d %d\n",dips+1 , (*ipe-*ips+1) , djps+1 , (*jpe-*jps+1) , dkps+1 , (*kpe-*kps+1)) ;
//fprintf(stderr,"m %d %d %d %d %d %d\n",dips+1 , dipe , djps+1 , djpe , dkps+1 , dkpe ) ;
//fprintf(stderr,"p %d %d %d %d %d %d\n",dips+1 , dipe , djps+1 , djpe , dkps+1 , dkpe ) ;

      s2 = rsl_internal_microclock_() ;
      wsm5_gpu <<< dimGrid, dimBlock >>> (
                    th_d, pii_d, q_d, qc_d, qi_d, qr_d, qs_d, den_d, p_d, delz_d
#ifdef DEBUGGAL_ARRAY
,debuggal_d
#endif
                   ,rain_d,rainncv_d
                   ,sr_d
                   ,snow_d,snowncv_d
                   ,*delt
,retvals_d
                   ,dips+1 , (*ipe-*ips+1) , djps+1 , (*jpe-*jps+1) , dkps+1 , (*kpe-*kps+1)
                   ,dips+1 , dipe , djps+1 , djpe , dkps+1 , dkpe
                   ,dips+1 , dipe , djps+1 , djpe , dkps+1 , dkpe
                         ) ;
      cudaThreadSynchronize() ;
      e2 = rsl_internal_microclock_() ;
      fprintf(stderr,"Call to wsm5_gpu (not including data xfer): %d microseconds\n",e2-s2) ;
#endif

      FROMDEV3(th) ;
      FROMDEV3(pii) ;
      FROMDEV3(q) ;
      FROMDEV3(qc) ;
      FROMDEV3(qi) ;
      FROMDEV3(qr) ;
      FROMDEV3(qs) ;
#ifdef DEBUGGAL_ARRAY
FROMDEV3(debuggal) ;
#endif
      FROMDEV2(rain) ;
      FROMDEV2(rainncv) ;
      FROMDEV2(sr) ;
      FROMDEV2(snow) ;
      FROMDEV2(snowncv) ;
      e = rsl_internal_microclock_() ;
//fprintf(stderr,"retrieving retvals %d\n",*kme-*kms+1) ;
FROMDEV(retvals,(*kme-*kms+1)) ;
      fprintf(stderr,"Call to wsm5_gpu (including data xfer): %d microseconds\n",e-s) ;

{ int k ;
//for (k=0 ;k<*kme-*kms+1;k++) {fprintf(stderr,"retvals %d %f\n",k,retvals[k]) ;}
//for (k=0 ;k<5;k++) {fprintf(stderr,"retvals %d %f\n",k,retvals[k]) ;}
}

      CLNUP(th) ;
      CLNUP(pii) ;
      CLNUP(q) ;
      CLNUP(qc) ;
      CLNUP(qi) ;
      CLNUP(qr) ;
      CLNUP(qs) ;
      CLNUP(den) ;
      CLNUP(p) ;
      CLNUP(delz) ;
#ifdef DEBUGGAL_ARRAY
CLNUP(debuggal) ;
#endif
      CLNUP(rain) ;
      CLNUP(rainncv) ;
      CLNUP(sr) ;
      CLNUP(snow) ;
      CLNUP(snowncv) ;
CLNUP(retvals) ;

      return(0) ;
}

#if 0
static int first_wsm5_host_2=1 ;
// 3d
static float * th_h ;
static float * pii_h ;
static float * q_h ;
static float * qc_h ;
static float * qi_h ;
static float * qr_h ;
static float * qs_h ;
static float * den_h ;
static float * p_h ;
static float * delz_h ;
// 2d
static float * rain_h ;
static float * rainncv_h ;
static float * sr_h ;
static float * snow_h ;
static float * snowncv_h ;

// idea here is to copy the data into pinned (paged-locked) mem for faster xfer
int
WSM5_HOST_2 (
                    float *th, float *pii
                   ,float *q
                   ,float *qc, float *qi, float *qr, float *qs
                   ,float *den, float *p, float *delz
                   ,float *delt
                   ,float *rain,float *rainncv
                   ,float *sr
                   ,float *snow,float *snowncv
                   ,int *ids, int *ide,  int *jds, int *jde,  int *kds, int *kde
                   ,int *ims, int *ime,  int *jms, int *jme,  int *kms, int *kme
                   ,int *ips, int *ipe,  int *jps, int *jpe,  int *kps, int *kpe
          )
{
   int i,j,k  ;
   float *ptr ;
   int d3 = (*ipe-*ips+1) * (*jpe-*jps+1) * (*kpe-*kps+1) ;
   int d2 = (*ipe-*ips+1) * (*jpe-*jps+1) ;
  
   if ( first_wsm5_host_2 == 1 ) {
      cudaMallocHost( (void **)&th_h   , d3*sizeof(float) ) ; //3d
      cudaMallocHost( (void **)&pii_h  , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&q_h    , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&qc_h   , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&qi_h   , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&qr_h   , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&qs_h   , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&den_h  , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&p_h    , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&delz_h , d3*sizeof(float) ) ;
      cudaMallocHost( (void **)&rain_h    , d2*sizeof(float) ) ; //2d
      cudaMallocHost( (void **)&rainncv_h , d2*sizeof(float) ) ;
      cudaMallocHost( (void **)&sr_h      , d2*sizeof(float) ) ;
      cudaMallocHost( (void **)&snow_h    , d2*sizeof(float) ) ;
      cudaMallocHost( (void **)&snowncv_h , d2*sizeof(float) ) ;
      first_wsm5_host_2 = 0 ;
   }

#define PIN3(A) ptr=A##_h;for(j=*jps;j<=*jpe;j++){for(k=*kps;k<=*kpe;k++){for(i=*ips;i<=*ipe;i++){*ptr++=A [I3(i-*ims,k-*kms,*ime-*ims+1,j-*jms,*kme-*kms+1)];}}};
#define PIN2(A) ptr=A##_h;for(j=*jps;j<=*jpe;j++);for(i=*ips;i<=*ipe;i++){*ptr++=A [I2(i-*ims,j-*jms,*ime-*ims+1)];};
#define UNPIN3(A) ptr=A##_h;for(j=*jps;j<=*jpe;j++);for(k=*kps;k<=*kpe;k++);for(i=*ips;i<=*ipe;i++){A [I3(i-*ims,k-*kms,*ime-*ims+1,j-*jms,*kme-*kms+1)]=*ptr++;};
#define UNPIN2(A) ptr=A##_h;for(j=*jps;j<=*jpe;j++);for(i=*ips;i<=*ipe;i++){A [I2(i-*ims,j-*jms,*ime-*ims+1)]=*ptr++;};

   PIN3(th) ;
   PIN3(th) ;
   PIN3(pii) ;
   PIN3(q) ;
   PIN3(qc) ;
   PIN3(qi) ;
   PIN3(qr) ;
   PIN3(qs) ;
   PIN3(den) ;
   PIN3(p) ;
   PIN3(delz) ;
   PIN2(rain) ;
   PIN2(rainncv) ;
   PIN2(sr) ;
   PIN2(snow) ;
   PIN2(snowncv) ;

   WSM5_HOST (
                    th_h, pii_h
                   ,q_h
                   ,qc_h, qi_h, qr_h, qs_h
                   ,den_h, p_h, delz_h
                   ,delt
                   ,rain_h,rainncv_h
                   ,sr_h
                   ,snow_h,snowncv_h
                   ,ids, ide, jds, jde, kds, kde
                   ,ips, ipe, jps, jpe, kps, kpe
                   ,ips, ipe, jps, jpe, kps, kpe
          ) ;


   UNPIN3(th) ;
   UNPIN3(th) ;
   UNPIN3(pii) ;
   UNPIN3(q) ;
   UNPIN3(qc) ;
   UNPIN3(qi) ;
   UNPIN3(qr) ;
   UNPIN3(qs) ;
   UNPIN3(den) ;
   UNPIN3(p) ;
   UNPIN3(delz) ;
   UNPIN2(rain) ;
   UNPIN2(rainncv) ;
   UNPIN2(sr) ;
   UNPIN2(snow) ;
   UNPIN2(snowncv) ;

}
#endif

int
GET_WSM5_GPU_LEVELS ( int * retval )
{
    *retval = MKX ;  /* MKX is hard coded value set in the makefile */
}
}

#if 0
main( int argc, char **argv ) 
{
                   float *th ; float *pii ; float *q ;
                   float *qc; float *qi; float *qr; float *qs ;
                   float *den; float *p; float *delz ;
                   float *delt ;
                   float *rain;float *rainncv ;
                   float *sr ;
                   float *snow;float *snowncv ;
                   int *ids; int *ide;  int *jds; int *jde;  int *kds; int *kde ;
                   int *ims; int *ime;  int *jms; int *jme;  int *kms; int *kme ;
                   int *ips; int *ipe;  int *jps; int *jpe;  int *kps; int *kpe     ;
     WSM5_HOST (
                    th, pii, q, qc, qi, qr, qs, den, p, delz
                   ,rain,rainncv
                   ,sr
                   ,snow,snowncv
                   ,delt
                   ,ids, ide,  jds, jde,  kds, kde
                   ,ims, ime,  jms, jme,  kms, kme
                   ,ips, ipe,  jps, jpe,  kps, kpe
                         )  ;
}
#endif
