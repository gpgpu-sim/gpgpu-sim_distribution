#define REWORK_FALL
#define REWORK_PART2
// wsm5_gpu.cu gets preprocessed by spt.pl, which handles the _def_ directives before it is compiled

#ifndef PREPASS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas.h"
#endif

#define IDEBUG ((DEBUG_I)-2)
#define JDEBUG ((DEBUG_J)-2)
#define KDEBUG (DEBUG_K)

// this is an M4 include
include(debug.m4)

//SPTSTART

#include "spt.h"

#include "util.h"

# define float float



__global__ void wsm5_gpu ( 
                    float *th, float *pii                   //_def_ arg ikj:th,pii
                   ,float *q                                //_def_ arg ikj:q
                   ,float *qc,float *qi,float *qr,float *qs //_def_ arg ikj:qc,qi,qr,qs
                   ,float *den, float *p, float *delz       //_def_ arg ikj:den,p,delz
#ifdef DEBUGAL_ARRAY
,float *debuggal                           //_def_ arg ikj:debuggal
#endif
                   ,float *rain,float *rainncv              //_def_ arg ij:rain,rainncv
                   ,float *sr                               //_def_ arg ij:sr
                   ,float *snow,float *snowncv              //_def_ arg ij:snow,snowncv
                   ,float delt
,float* retvals
                   ,int ids, int ide,  int jds, int jde,  int kds, int kde          
                   ,int ims, int ime,  int jms, int jme,  int kms, int kme          
                   ,int ips, int ipe,  int jps, int jpe,  int kps, int kpe          
                         )
{

   float xlf, xmi, acrfac, vt2i, vt2s, supice, diameter ;
   float roqi0, xni0, qimax, value, source, factor, xlwork2 ;
   float t_k, q_k, qr_k, qc_k, qs_k, qi_k, qs1_k, qs2_k, cpm_k, xl_k, xni_k, w1_k, w2_k, w3_k  ;

#define hsub   xls
#define hvap   xlv0
#define cvap   cpv
     float ttp ;
     float dldt ;
     float xa ;
     float xb ;
     float dldti ;
     float xai ;
     float xbi ;

     //_def_ local k:qs1,qs2,rh1,rh2

#ifdef DEBUGAL_ARRAY
  debuggal[0] = 999.00 ;
#endif

if ( ig < ide-ids+1 && jg < jde-jds+1 ) {


   int k ;

#include "wsm5_constants.h"

   //_def_ local k:t
   //_def_ local k:cpm,xl

   for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
     t[k] = th[k] * pii[k] ;
   }

   for( k=kps-1 ;k<=kpe-1;k++) { 
                                if ( qc[k] < 0. ) { qc[k] = 0. ; } 
                                if ( qi[k] < 0. ) { qi[k] = 0. ; } 
                                if ( qr[k] < 0. ) { qr[k] = 0. ; } 
                                if ( qs[k] < 0. ) { qs[k] = 0. ; } 
                               }

// 564 !----------------------------------------------------------------
// 565 !     latent heat for phase changes and heat capacity. neglect the
// 566 !     changes during microphysical process calculation
// 567 !     emanuel(1994)

#define CPMCAL(x) (cpd*(1.-max(x,qmin))+max(x,qmin)*cpv)
#define XLCAL(x)  (xlv0-xlv1*((x)-t0c))

   for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
     cpm[k] = CPMCAL(q[k]) ;
     xl[k] = XLCAL(t[k]) ;
   }

// 576 !----------------------------------------------------------------
// 577 !     compute the minor time steps.

   float dtcldcr = 120. ;
   int loops = delt/dtcldcr+.5 ;

   loops = MAX(loops,1) ;
   float dtcld = delt/loops ;
   if ( delt <= dtcldcr) dtcld = delt ;

   int loop ;


   for ( loop = 1 ; loop <= loops ; loop++ ) {
// 585 !----------------------------------------------------------------
// 586 !     initialize the large scale variables
     int mstep = 1 ;

     ttp=t0c+0.01 ;
     dldt=cvap-cliq ;
     xa=-dldt/rv ;
     xb=xa+hvap/(rv*ttp) ;
     dldti=cvap-cice ;
     xai=-dldti/rv ;
     xbi=xai+hsub/(rv*ttp) ;


     float tr, ltr, tt, pp, qq ;

     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {

       pp = p[k] ;
       tt = t[k] ;
       tr = ttp/tt ;
       ltr = log(tr) ;

       qq=psat*exp(ltr*(xa)+xb*(1.-tr)) ;
       qq=ep2*qq/(pp-qq) ;
       qs1[k] = MAX(qq,qmin) ;
       rh1[k] = MAX( q[k]/qs1[k],qmin) ;

       if( tt < ttp ) {
         qq=psat*exp(ltr*(xai)+xbi*(1.-tr)) ;
       } else {
         qq=psat*exp(ltr*(xa)+xb*(1.-tr)) ;
       }
       qq = ep2 * qq / (pp - qq) ;
       qs2[k] = MAX(qq,qmin) ;
       rh2[k] = MAX(q[k]/qs2[k],qmin) ;

     }

     //_def_ register 0:prevp,psdep,praut,psaut,pracw,psaci,psacw,pigen,pidep,pcond,psmlt,psevp
     //_def_ local k:xni

     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          xni[k] = 1.e3 ;
      }

//     diffus(x,y) = 8.794e-5 * exp(log(x)*(1.81)) / y        ! 8.794e-5*x**1.81/y
//     viscos(x,y) = 1.496e-6 * (x*sqrt(x)) /(x+120.)/y  ! 1.496e-6*x**1.5/(x+120.)/y
//     xka(x,y) = 1.414e3*viscos(x,y)*y
//     diffac(a,b,c,d,e) = d*a*a/(xka(c,d)*rv*c*c)+1./(e*diffus(c,b))
//     venfac(a,b,c) = exp(log((viscos(b,c)/diffus(b,a)))*((.3333333)))    &
//                    /sqrt(viscos(b,c))*sqrt(sqrt(den0/c))

#define DIFFUS(x,y) (8.794e-5 * exp(log(x)*(1.81)) / (y))
#define VISCOS(x,y) (1.496e-6 * ((x)*sqrt(x)) /((x)+120.)/(y))
#define XKA(x,y) (1.414e3*VISCOS((x),(y))*(y))
#define DIFFAC(a,b,c,d,e) ((d)*(a)*(a)/(XKA((c),(d))*rv*(c)*(c))+1./((e)*DIFFUS((c),(b))))
#define VENFAC(a,b,c) (exp(log((VISCOS((b),(c))/DIFFUS((b),(a))))*((.3333333)))*rsqrt(VISCOS((b),(c)))*sqrt(sqrt(den0/(c))))
#define CONDEN(a,b,c,d,e) ((MAX((b),qmin)-(c))/(1.+(d)*(d)/(rv*(e))*(c)/((a)*(a))))

#define LAMDAR(x,y) sqrt(sqrt(pidn0r/((x)*(y))))
#define LAMDAS(x,y,z) sqrt(sqrt(pidn0s*(z)/((x)*(y))))

// calculate mstep for this colum

     //_def_ local k:rsloper,rslopebr,rslope2r,rslope3r
     //_def_ local k:rslopes,rslopebs,rslope2s,rslope3s
     //_def_ local k:denfac
     //_def_ local k:n0sfac
     //_def_ local k:w1,w2,w3


     float w ; 
     float rmstep ;
     int numdt ;
     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
       float supcol = t0c - t[k] ;
       n0sfac[k] = MAX(MIN(exp(alpha*supcol),n0smax/n0s),1.) ;
       if ( qr[k] <= qcrmin ) {
         rsloper[k]  = rslopermax ;
         rslopebr[k] = rsloperbmax ;
         rslope2r[k] = rsloper2max ;
         rslope3r[k] = rsloper3max ;
       } else {
         rsloper[k]  = 1./LAMDAR(qr[k],den[k]) ;
         rslopebr[k] = exp(log(rsloper[k])*bvtr) ;
         rslope2r[k] = rsloper[k] * rsloper[k] ; 
         rslope3r[k] = rslope2r[k] * rsloper[k] ; 
       }
       if ( qs[k] <= qcrmin ) {
         rslopes[k]  = rslopesmax ;
         rslopebs[k] = rslopesbmax ;
         rslope2s[k] = rslopes2max ;
         rslope3s[k] = rslopes3max ;
       } else {
         rslopes[k] = 1./LAMDAS(qs[k],den[k],n0sfac[k]) ;
         rslopebs[k] = exp(log(rslopes[k])*bvts) ;
         rslope2s[k] = rslopes[k] * rslopes[k] ; 
         rslope3s[k] = rslope2s[k] * rslopes[k] ; 
       }
       denfac[k] = sqrt(den0/den[k]) ;
       w1[k] = pvtr*rslopebr[k]*denfac[k]/delz[k] ;
       w2[k] = pvts*rslopebs[k]*denfac[k]/delz[k] ;

       w = MAX(w1[k],w2[k]) ;
       numdt = MAX((int)trunc(w*dtcld+.5+.5),1) ;
       if ( numdt >= mstep ) mstep = numdt ;
//-------------------------------------------------------------
// Ni: ice crystal number concentration   [HDC 5c]
//-------------------------------------------------------------
       float temp = (den[k]*MAX(qi[k],qmin)) ;
       temp = sqrt(sqrt(temp*temp*temp)) ;
#ifdef DEBUGDEBUG
       xni[k] = 1.e3 ;
#else
       xni[k] = MIN(MAX(5.38e7*temp,1.e3),1.e6) ;
#endif
     }
     rmstep = 1./mstep ;
   
     int n ;
     float dtcldden, coeres, rdelz ;


     float den_k, falk1_k, falk1_kp1, fall1_k, fall1_kp1, delz_k, delz_kp1 ;
     float        falk2_k, falk2_kp1, fall2_k, fall2_kp1                   ;

     for ( n = 1 ; n <= mstep ; n++ ) {
       k = kpe - 1 ;
       den_k = den[k] ;
       falk1_kp1 = den_k*qr[k]*w1[k]*rmstep ;
       fall1_kp1 = falk1_kp1 ;
       falk2_kp1 = den_k*qs[k]*w2[k]*rmstep ;
       fall2_kp1 = falk2_kp1 ;
       dtcldden = dtcld/den_k ;
       qr[k] = MAX(qr[k]-falk1_kp1*dtcldden,0.0) ;
       qs[k] = MAX(qs[k]-falk2_kp1*dtcldden,0.0) ;
       delz_kp1 = delz[k] ;
       for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
         den_k = den[k] ;
         falk1_k = den_k*qr[k]*w1[k]*rmstep ;
         fall1_k = falk1_k ;
         falk2_k = den_k*qs[k]*w2[k]*rmstep ;
         fall2_k = falk2_k ;
         dtcldden = dtcld/den_k ;
         delz_k = delz[k] ;
         rdelz = 1./delz_k ;
         qr[k] = MAX(qr[k]- (falk1_k-falk1_kp1*delz_kp1*rdelz)* dtcldden,0.) ;
         qs[k] = MAX(qs[k]- (falk2_k-falk2_kp1*delz_kp1*rdelz)* dtcldden,0.) ;
         delz_kp1 = delz_k ;
         falk1_kp1 = falk1_k ;
         fall1_kp1 = fall1_k ;
         falk2_kp1 = falk2_k ;
         fall2_kp1 = fall2_k ;
       }

       for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
         if ( t[k] > t0c && qs[k] > 0.) {
           xlf = xlf0 ;
           w3[k] = VENFAC(p[k],t[k],den[k]) ;
           coeres = rslope2s[k]*sqrt(rslopes[k]*rslopebs[2]) ;
           psmlt[k] = XKA(t[k],den[k])/xlf*(t0c-t[k])*pi/2.
                     *n0sfac[k]*(precs1*rslope2s[k]+precs2
                     *w3[k]*coeres) ;
           psmlt[k] = MIN(MAX(psmlt[k]*dtcld*rmstep,-qs[k]*rmstep),0.) ;
           qs[k] += psmlt[k] ;
           qr[k] -= psmlt[k] ;
           t[k] += xlf/CPMCAL(q[k])*psmlt[k] ;
         }
       }
     }

//---------------------------------------------------------------
// Vice [ms-1] : fallout of ice crystal [HDC 5a]
//---------------------------------------------------------------
     mstep = 1 ;
     numdt = 1 ;
     for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
       if (qi[k] <= 0.) {
         w2[k] = 0. ;
       } else {
         xmi = den[k]*qi[k]/xni[k] ;
         diameter  = MAX(MIN(dicon * sqrt(xmi),dimax), 1.e-25) ;
         w1[k] = 1.49e4*exp(log(diameter)*(1.31)) ;
         w2[k] = w1[k]/delz[k] ;
       }
       numdt = MAX( (int) trunc(w2[k]*dtcld+.5+.5),1) ;
       if(numdt > mstep) mstep = numdt ;
     }
     rmstep = 1./mstep ;

     float falkc_k, falkc_kp1, fallc_k, fallc_kp1 ;
     for ( n = 1 ; n <= mstep ; n++ ) {
       k = kpe - 1 ;
       den_k = den[k] ;
       falkc_kp1 = den_k*qi[k]*w2[k]*rmstep ;
       fallc_kp1 = fallc_kp1+falkc_kp1 ;
       qi[k] = MAX(qi[k]-falkc_kp1*dtcld/den_k,0.) ;
       delz_kp1 = delz[k] ;
       for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
         den_k = den[k] ;
         falkc_k = den_k*qi[k]*w2[k]*rmstep ;
         fallc_k = fallc_k+falkc_k ;
         delz_k = delz[k] ;
         qi[k] = MAX(qi[k]-(falkc_k-falkc_kp1
                 *delz_kp1/delz_k)*dtcld/den_k,0.) ;
         delz_kp1 = delz_k ;
         falkc_kp1 = falkc_k ;
         fallc_kp1 = fallc_k ;
       }
     }
     float fallsum = fall1_k+fall2_k+fallc_k ;
     float fallsum_qsi = fall2_k+fallc_k ;

     rainncv = 0. ;
     if(fallsum > 0.) {
       rainncv = fallsum*delz[1]/denr*dtcld*1000. ;
       rain = fallsum*delz[1]/denr*dtcld*1000. + rain ;
     }
     snowncv = 0. ;
     if(fallsum_qsi > 0.) {
       snowncv = fallsum_qsi*delz[0]/denr*dtcld*1000. ;
       snow = fallsum_qsi*delz[0]/denr*dtcld*1000. + snow ;
     }
     sr = 0. ;
     if ( fallsum > 0. ) sr = fallsum_qsi*delz[0]/denr*dtcld*1000./(rainncv+1.e-12) ;

//---------------------------------------------------------------
// pimlt: instantaneous melting of cloud ice [HL A47] [RH83 A28]
//       (T>T0: I->C)
//---------------------------------------------------------------


     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {

       //  note -- many of these are turned into scalars of form name_reg by _def_ above
       //  so that they will be stored in registers
       prevp[k] = 0. ;
       psdep[k] = 0. ;
       praut[k] = 0. ;
       psaut[k] = 0. ;
       pracw[k] = 0. ;
       psaci[k] = 0. ;
       psacw[k] = 0. ;
       pigen[k] = 0. ;
       pidep[k] = 0. ;
       pcond[k] = 0. ;
       psevp[k] = 0. ;

       q_k =  q[k] ;
       t_k = t[k] ;
       qr_k =  qr[k] ;
       qc_k = qc[k] ;
       qs_k = qs[k] ;
       qi_k = qi[k] ;
       qs1_k = qs1[k] ;
       qs2_k = qs2[k] ;
       cpm_k = cpm[k] ;
       xl_k = xl[k] ;
      
       float supcol = t0c-t_k ;
       xlf = xls-xl_k ;
       if( supcol < 0. ) xlf = xlf0 ;
       if( supcol < 0 && qi_k > 0. ) {
         qc_k = qc_k + qi_k ;
         t_k = t_k - xlf/cpm_k*qi_k ;
         qi_k = 0. ;
       }
//---------------------------------------------------------------
// pihmf: homogeneous freezing of cloud water below -40c [HL A45]
//        (T<-40C: C->I)
//---------------------------------------------------------------
       if( supcol > 40. && qc_k > 0. ) {
         qi_k = qi_k + qc_k ;
         t_k = t_k + xlf/cpm_k*qc_k ;
         qc_k = 0. ;
       }
//---------------------------------------------------------------
// pihtf: heterogeneous freezing of cloud water [HL A44]
//        (T0>T>-40C: C->I)
//---------------------------------------------------------------
       if ( supcol > 0. && qc_k > 0.) {
         float pfrzdtc = MIN(pfrz1*(exp(pfrz2*supcol)-1.)
           *den[k]/denr/xncr*qc_k*qc_k*dtcld,qc_k) ;
         qi_k = qi_k + pfrzdtc ;
         t_k = t_k + xlf/cpm_k*pfrzdtc ;
         qc_k = qc_k-pfrzdtc ;
       }
//---------------------------------------------------------------
// psfrz: freezing of rain water [HL A20] [LFO 45]
//        (T<T0, R->S)
//---------------------------------------------------------------
       if( supcol > 0. && qr_k > 0. ) {
         float temp = rsloper[k] ;
         temp = temp*temp*temp*temp*temp*temp*temp ;
         float pfrzdtr = MIN(20.*(pi*pi)*pfrz1*n0r*denr/den[k]
               *(exp(pfrz2*supcol)-1.)*temp*dtcld,
               qr_k) ;
         qs_k = qs_k + pfrzdtr ;
         t_k = t_k + xlf/cpm_k*pfrzdtr ;
         qr_k = qr_k-pfrzdtr ;
       }

//----------------------------------------------------------------
//     rsloper: reverse of the slope parameter of the rain(m)
//     xka:    thermal conductivity of air(jm-1s-1k-1)
//     work1:  the thermodynamic term in the denominator associated with
//             heat conduction and vapor diffusion
//             (ry88, y93, h85)
//     work2: parameter associated with the ventilation effects(y93)

       n0sfac[k] = MAX(MIN(exp(alpha*supcol),n0smax/n0s),1.) ;
       if ( qr_k <= qcrmin ) {
         rsloper[k]  = rslopermax ;
         rslopebr[k] = rsloperbmax ;
         rslope2r[k] = rsloper2max ;
         rslope3r[k] = rsloper3max ;
       } else {
         rsloper[k] = 1./(sqrt(sqrt(pidn0r/((qr_k)*(den[k]))))) ;
         rslopebr[k] = exp(log(rsloper[k])*bvtr) ;
         rslope2r[k] = rsloper[k] * rsloper[k] ;
         rslope3r[k] = rslope2r[k] * rsloper[k] ;
       }
       if ( qs_k <= qcrmin ) {
         rslopes[k]  = rslopesmax ;
         rslopebs[k] = rslopesbmax ;
         rslope2s[k] = rslopes2max ;
         rslope3s[k] = rslopes3max ;
       } else {
         rslopes[k] = 1./(sqrt(sqrt(pidn0s*(n0sfac[k])/((qs_k)*(den[k]))))) ;
         rslopebs[k] = exp(log(rslopes[k])*bvts) ;
         rslope2s[k] = rslopes[k] * rslopes[k] ;
         rslope3s[k] = rslope2s[k] * rslopes[k] ;
       }

       w1_k = DIFFAC(xl_k,p[k],t_k,den[k],qs1_k) ;
       w2_k = DIFFAC(xls,p[k],t_k,den[k],qs2_k) ;
       w3_k = VENFAC(p[k],t_k,den[k]) ;

//
//===============================================================
//
// warm rain processes
//
// - follows the processes in RH83 and LFO except for autoconcersion
//
//===============================================================
//
      float supsat = MAX(q_k,qmin)-qs1_k ;
      float satdt = supsat/dtcld ;
//---------------------------------------------------------------
// praut: auto conversion rate from cloud to rain [HDC 16]
//        (C->R)
//---------------------------------------------------------------
      if(qc_k > qc0) {
        praut[k] = qck1*exp(log(qc_k)*((7./3.))) ;
        praut[k] = MIN(praut[k],qc_k/dtcld) ;
      }
//---------------------------------------------------------------
// pracw: accretion of cloud water by rain [HL A40] [LFO 51]
//        (C->R)
//---------------------------------------------------------------
      if(qr_k > qcrmin && qc_k > qmin) {
        pracw[k] = MIN(pacrr*rslope3r[k]*rslopebr[k]
                   *qc_k*denfac[k],qc_k/dtcld) ;
      }
//---------------------------------------------------------------
// prevp: evaporation/condensation rate of rain [HDC 14]
//        (V->R or R->V)
//---------------------------------------------------------------
      if(qr_k > 0.) {
        coeres = rslope2r[k]*sqrt(rsloper[k]*rslopebr[k]) ;
        prevp[k] = (rh1[k]-1.)*(precr1*rslope2r[k]
                     +precr2*w3_k*coeres)/w1_k ;
        if(prevp[k] < 0.) {
          prevp[k] = MAX(prevp[k],-qr_k/dtcld) ;
          prevp[k] = MAX(prevp[k],satdt/2) ;
        } else {
          prevp[k] = MIN(prevp[k],satdt/2) ;
        }
      }

//
//===============================================================
//
// cold rain processes
//
// - follows the revised ice microphysics processes in HDC
// - the processes same as in RH83 and RH84  and LFO behave
//   following ice crystal hapits defined in HDC, inclduing
//   intercept parameter for snow (n0s), ice crystal number
//   concentration (ni), ice nuclei number concentration
//   (n0i), ice diameter (d)
//
//===============================================================
//
          float rdtcld = 1./dtcld ;
          supsat = MAX(q_k,qmin)-qs2_k ;
          satdt = supsat/dtcld ;
          int ifsat = 0 ;
//-------------------------------------------------------------
// Ni: ice crystal number concentraiton   [HDC 5c]
//-------------------------------------------------------------
          float temp = (den[k]*MAX(qi_k,qmin)) ;
          temp = sqrt(sqrt(temp*temp*temp)) ;
          xni[k] = MIN(MAX(5.38e7*temp,1.e3),1.e6) ;
          float eacrs = exp(0.07*(-supcol)) ;
//-------------------------------------------------------------
// psacw: Accretion of cloud water by snow  [HL A7] [LFO 24]
//        (T<T0: C->S, and T>=T0: C->R)
//-------------------------------------------------------------
          if(qs_k > qcrmin && qc_k > qmin) {
            psacw[k] = MIN(pacrc*n0sfac[k]*rslope3s[k] 
                         *rslopebs[k]*qc_k*denfac[k]
                         ,qc_k*rdtcld) ;
          }
//
          if(supcol > 0) {
            if(qs_k > qcrmin && qi_k > qmin) {
              xmi = den[k]*qi_k/xni[k] ;
              diameter  = MIN(dicon * sqrt(xmi),dimax) ;
              vt2i = 1.49e4*pow(diameter,(float)1.31) ;
              vt2s = pvts*rslopebs[k]*denfac[k] ;
//-------------------------------------------------------------
// psaci: Accretion of cloud ice by rain [HDC 10]
//        (T<T0: I->S)
//-------------------------------------------------------------
              acrfac = 2.*rslope3s[k]+2.*diameter*rslope2s[k]
                      +diameter*diameter*rslopes[k] ;
              psaci[k] = pi*qi_k*eacrs*n0s*n0sfac[k]
                           *abs(vt2s-vt2i)*acrfac*.25 ;
            }
//-------------------------------------------------------------
// pidep: Deposition/Sublimation rate of ice [HDC 9]
//       (T<T0: V->I or I->V)
//-------------------------------------------------------------
            if(qi_k > 0 && ifsat != 1) {
              xmi = den[k]*qi_k/xni[k] ;
              diameter = dicon * sqrt(xmi) ;
              pidep[k] = 4.*diameter*xni[k]*(rh2[k]-1.)/w2_k ;
              supice = satdt-prevp[k] ;
              if(pidep[k] < 0.) {
                pidep[k] = MAX(MAX(pidep[k],satdt*.5),supice) ;
                pidep[k] = MAX(pidep[k],-qi_k*rdtcld) ;
              } else {
                pidep[k] = MIN(MIN(pidep[k],satdt*.5),supice) ;
              }
              if(abs(prevp[k]+pidep[k]) >= abs(satdt)) ifsat = 1 ;
            }
//-------------------------------------------------------------
// psdep: deposition/sublimation rate of snow [HDC 14]
//        (V->S or S->V)
//-------------------------------------------------------------
            if( qs_k > 0. && ifsat != 1) {
              coeres = rslope2s[k]*sqrt(rslopes[k]*rslopebs[k]) ;
              psdep[k] = (rh2[k]-1.)*n0sfac[k]
                           *(precs1*rslope2s[k]+precs2
                           *w3_k*coeres)/w2_k ;
              supice = satdt-prevp[k]-pidep[k] ;
              if(psdep[k] < 0.) {
                psdep[k] = MAX(psdep[k],-qs_k*rdtcld) ;
                psdep[k] = MAX(MAX(psdep[k],satdt*.5),supice) ;
              } else {
                psdep[k] = MIN(MIN(psdep[k],satdt*.5),supice) ;
              }
              if(abs(prevp[k]+pidep[k]+psdep[k]) >= abs(satdt))
                ifsat = 1 ;
            }
//-------------------------------------------------------------
// pigen: generation(nucleation) of ice from vapor [HL A50] [HDC 7-8]
//       (T<T0: V->I)
//-------------------------------------------------------------
            if(supsat > 0 && ifsat != 1) {
              supice = satdt-prevp[k]-pidep[k]-psdep[k] ; 
              xni0 = 1.e3*exp(0.1*supcol) ;
              roqi0 = 4.92e-11*exp(log(xni0)*(1.33));
              pigen[k] = MAX(0.,(roqi0/den[k]-MAX(qi_k,0.))
                         *rdtcld) ;
              pigen[k] = MIN(MIN(pigen[k],satdt),supice) ;
            }
//
//-------------------------------------------------------------
// psaut: conversion(aggregation) of ice to snow [HDC 12]
//       (T<T0: I->S)
//-------------------------------------------------------------
            if(qi_k > 0.) {
              qimax = roqimax/den[k] ;
              psaut[k] = MAX(0.,(qi_k-qimax)*rdtcld) ;
            }
          }
//-------------------------------------------------------------
// psevp: Evaporation of melting snow [HL A35] [RH83 A27]
//       (T>T0: S->V)
//-------------------------------------------------------------
          if(supcol < 0.) {
            if(qs_k > 0. && rh1[k] < 1.) {
              psevp[k] = psdep[k]*w2_k/w1_k ;
            }  // asked Jimy about this, 11.6.07, JM
            psevp[k] = MIN(MAX(psevp[k],-qs_k*rdtcld),0.) ;
          }


//
//
//----------------------------------------------------------------
//     check mass conservation of generation terms and feedback to the
//     large scale
//
          if(t_k<=t0c) {
//
//     cloud water
//
            value = MAX(qmin,qc_k) ;
            source = (praut[k]+pracw[k]+psacw[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              praut[k] = praut[k]*factor ;
              pracw[k] = pracw[k]*factor ;
              psacw[k] = psacw[k]*factor ;
            }
//
//     cloud ice
//
            value = MAX(qmin,qi_k) ;
            source = (psaut[k]+psaci[k]-pigen[k]-pidep[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              psaut[k] = psaut[k]*factor ;
              psaci[k] = psaci[k]*factor ;
              pigen[k] = pigen[k]*factor ;
              pidep[k] = pidep[k]*factor ;
            }

//
//     rain (added for WRFV3.0.1)
//
            value = MAX(qmin,qr_k) ;
            source = (-praut[k]+pracw[k]-prevp[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              praut[k] = praut[k]*factor ;
              pracw[k] = pracw[k]*factor ;
              prevp[k] = prevp[k]*factor ;
            }
//
//     snow (added for WRFV3.0.1)
//
            value = MAX(qmin,qs_k) ;
            source = (-psdep[k]+psaut[k]-psaci[k]-psacw[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              psdep[k] = psdep[k]*factor ;
              psaut[k] = psaut[k]*factor ;
              psaci[k] = psaci[k]*factor ;
              psacw[k] = psacw[k]*factor ;
            }
//     (end added for WRFV3.0.1)

//
            w3_k=-(prevp[k]+psdep[k]+pigen[k]+pidep[k]) ;
//     update
            q_k = q_k+w3_k*dtcld ;
            qc_k = MAX(qc_k-(praut[k]+pracw[k]+psacw[k])*dtcld,0.) ;
            qr_k = MAX(qr_k+(praut[k]+pracw[k]+prevp[k])*dtcld,0.) ;
            qi_k = MAX(qi_k-(psaut[k]+psaci[k]-pigen[k]-pidep[k])*dtcld,0.) ;
            qs_k = MAX(qs_k+(psdep[k]+psaut[k]+psaci[k]+psacw[k])*dtcld,0.) ;
            xlf = xls-xl_k ;
            xlwork2 = -xls*(psdep[k]+pidep[k]+pigen[k])-xl_k*prevp[k]-xlf*psacw[k] ;
            t_k = t_k-xlwork2/cpm_k*dtcld ;
          } else {
//
//     cloud water
//
            value = MAX(qmin,qc_k) ;
            source=(praut[k]+pracw[k]+psacw[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              praut[k] = praut[k]*factor ;
              pracw[k] = pracw[k]*factor ;
              psacw[k] = psacw[k]*factor ;
            }
//
//     rain (added for WRFV3.0.1)
//
            value = MAX(qmin,qr_k) ;
            source = (-praut[k]-pracw[k]-prevp[k]-psacw[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              praut[k] = praut[k]*factor ;
              pracw[k] = pracw[k]*factor ;
              prevp[k] = prevp[k]*factor ;
              psacw[k] = psacw[k]*factor ;
            }
//     (end added for WRFV3.0.1)
//
//     snow
//
            value = MAX(qcrmin,qs_k) ;
            source=(-psevp[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              psevp[k] = psevp[k]*factor ;
            }
            w3_k=-(prevp[k]+psevp[k]) ;
//     update
            q_k = q_k+w3_k*dtcld ;
            qc_k = MAX(qc_k-(praut[k]+pracw[k]+psacw[k])*dtcld,0.) ;
            qr_k = MAX(qr_k+(praut[k]+pracw[k]+prevp[k] +psacw[k])*dtcld,0.) ;
            qs_k = MAX(qs_k+psevp[k]*dtcld,0.) ;
            xlf = xls-xl_k ;
            xlwork2 = -xl_k*(prevp[k]+psevp[k]) ;
            t_k = t_k-xlwork2/cpm_k*dtcld ;
          }
//
// Inline expansion for fpvs
          cvap = cpv ;
          ttp=t0c+0.01 ;
          dldt=cvap-cliq ;
          xa=-dldt/rv ;
          xb=xa+hvap/(rv*ttp) ;
          dldti=cvap-cice ;
          xai=-dldti/rv ;
          xbi=xai+hsub/(rv*ttp) ;
          tr=ttp/t_k ;
          qs1_k=psat*exp(log(tr)*(xa))*exp(xb*(1.-tr)) ;
          qs1_k = ep2 * qs1_k / (p[k] - qs1_k) ;
          qs1_k = MAX(qs1_k,qmin) ;
//
//----------------------------------------------------------------
//  pcond: condensational/evaporational rate of cloud water [HL A46] [RH83 A6]
//     if there exists additional water vapor condensated/if
//     evaporation of cloud water is not enough to remove subsaturation
//
          w1_k = ((MAX(q_k,qmin)-(qs1_k)) /
            (1.+(xl_k)*(xl_k)/(rv*(cpm_k))*(qs1_k)/((t_k)*(t_k)))) ;
          // w3_k = qc_k+w1_k ;   NOT USED
          pcond[k] = MIN(MAX(w1_k/dtcld,0.),MAX(q_k,0.)/dtcld) ;
          if(qc_k > 0. && w1_k < 0.) {
            pcond[k] = MAX(w1_k,-qc_k)/dtcld ;
          }
          q_k = q_k-pcond[k]*dtcld ;
          qc_k = MAX(qc_k+pcond[k]*dtcld,0.) ;
          t_k = t_k+pcond[k]*xl_k/cpm_k*dtcld ;
//
//
//----------------------------------------------------------------
//     padding for small values
//
          if(qc_k <= qmin) qc_k = 0.0 ;
          if(qi_k <= qmin) qi_k = 0.0 ;

          q[k] = q_k ;
          t[k] = t_k ;
          qr[k] = qr_k ;
          qc[k] = qc_k ;
          qs[k] = qs_k ;
          qi[k] = qi_k ;
          qs1[k] = qs1_k ;

     }
   }
   for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          th[k] = t[k] / pii[k] ;
   }
 } // guard 
}


