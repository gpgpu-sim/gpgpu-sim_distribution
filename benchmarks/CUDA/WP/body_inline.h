#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(t)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(q)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qc)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qi)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qr)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qs)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(den)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(p)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(delz)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(cpm)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(xl)
}
#endif

// 585 !----------------------------------------------------------------
// 586 !     initialize the large scale variables
     mstep = 1 ;

     ttp=t0c+0.01 ;
     dldt=cvap-cliq ;
     xa=-dldt/rv ;
     xb=xa+hvap/(rv*ttp) ;
     dldti=cvap-cice ;
     xai=-dldti/rv ;
     xbi=xai+hsub/(rv*ttp) ;

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

     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
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
          psmlt[k] = 0. ;
          psevp[k] = 0. ;
          falk1[k] = 0. ;
          falk2[k] = 0. ;
          fall1[k] = 0. ;
          fall2[k] = 0. ;
          fallc[k] = 0. ;
          falkc[k] = 0. ;
          xni[k] = 1.e3 ;
      }

#define LAMDAR(x,y) sqrt(sqrt(pidn0r/((x)*(y))))
#define LAMDAS(x,y,z) sqrt(sqrt(pidn0s*(z)/((x)*(y))))
// calculate mstep for this colum


     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
       float supcol = t0c - t[k] ;
#ifdef DEVICEEMU
if ( ig == IDEBUG && jg == JDEBUG && k+1 == KDEBUG ) fprintf(stderr,"ZAP t0c %25.17e\n",t0c) ;
if ( ig == IDEBUG && jg == JDEBUG && k+1 == KDEBUG ) fprintf(stderr,"ZAP supcol %25.17e\n",supcol) ;
#endif
DIAGOUTPUT1(t)
       n0sfac[k] = MAX(MIN(exp(alpha*supcol),n0smax/n0s),1.) ;
       if ( qr[k] <= qcrmin ) {
         rsloper[k]  = rslopermax ;
         rslopebr[k] = rsloperbmax ;
         rslope2r[k] = rsloper2max ;
         rslope3r[k] = rsloper3max ;
       } else {
DIAGOUTPUT1(qr)
DIAGOUTPUT1(den)
         rsloper[k]  = 1./LAMDAR(qr[k],den[k]) ;
DIAGOUTPUT1(rsloper)
         rslopebr[k] = exp(log(rsloper[k])*bvtr) ;
         rslope2r[k] = rsloper[k] * rsloper[k] ; 
         rslope3r[k] = rslope2r[k] * rsloper[k] ; 
       }
       if ( qs[k] <= qcrmin ) {
         rslopes[k]  = rslopesmax ;
DIAGOUTPUT1(rslopes) ;
         rslopebs[k] = rslopesbmax ;
DIAGOUTPUT1(rslopebs) ;
         rslope2s[k] = rslopes2max ;
         rslope3s[k] = rslopes3max ;
       } else {
DIAGOUTPUT1(qs) ;
DIAGOUTPUT1(den) ;
DIAGOUTPUT1(n0sfac) ;
         rslopes[k] = 1./LAMDAS(qs[k],den[k],n0sfac[k]) ;
DIAGOUTPUT1(rslopes) ;
         rslopebs[k] = exp(log(rslopes[k])*bvts) ;
DIAGOUTPUT1(rslopebs) ;
         rslope2s[k] = rslopes[k] * rslopes[k] ; 
         rslope3s[k] = rslope2s[k] * rslopes[k] ; 
       }
       denfac[k] = sqrt(den0/den[k]) ;
       w1[k] = pvtr*rslopebr[k]*denfac[k]/delz[k] ;
       w2[k] = pvts*rslopebs[k]*denfac[k]/delz[k] ;

DIAGOUTPUT1(w1)
DIAGOUTPUT1(rslopebr)
DIAGOUTPUT1(w2)
DIAGOUTPUT1(rslopebs)
DIAGOUTPUT1(denfac)
DIAGOUTPUT1(delz)

       w = MAX(w1[k],w2[k]) ;
       numdt = MAX(trunc(w*dtcld+.5+.5),1) ;
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
   
     for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
       fall1[k] = 0. ;
       fall2[k] = 0. ;
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
#define VENFAC(a,b,c) (exp(log((VISCOS((b),(c))/DIFFUS((b),(a))))*((.3333333)))/sqrt(VISCOS((b),(c)))*sqrt(sqrt(den0/(c))))
#define CONDEN(a,b,c,d,e) ((MAX((b),qmin)-(c))/(1.+(d)*(d)/(rv*(e))*(c)/((a)*(a))))

     for ( n = 1 ; n <= mstep ; n++ ) {
       k = kpe - 1 ;
       falk1[k] = den[k]*qr[k]*w1[k]*rmstep ;
       fall1[k] += falk1[k] ;
       falk2[k] = den[k]*qs[k]*w2[k]*rmstep ;
       fall2[k] += falk2[k] ;
       qr[k] = MAX(qr[k]-falk1[k]*dtcldden,0.) ;
       qs[k] = MAX(qs[k]-falk2[k]*dtcldden,0.) ;
       for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
         falk1[k] = den[k]*qr[k]*w1[k]*rmstep ;
         fall1[k] += falk1[k] ;
         falk2[k] = den[k]*qs[k]*w2[k]*rmstep ;
         fall2[k] += falk2[k] ;
         dtcldden = dtcld/den[k] ;
         rdelz = 1./delz[k] ;
DIAGOUTPUT1i(loop) ;
DIAGOUTPUT1i(mstep) ;
DIAGOUTPUT1i(n) ;
DIAGOUTPUT1(qr) ;
DIAGOUTPUT1(falk1) ;
DIAGOUTPUT11(falk1) ;
DIAGOUTPUT1(delz) ;
DIAGOUTPUT11(delz) ;
         qr[k] = MAX(qr[k]-
                            (falk1[k]-falk1[k+1]*delz[k+1]*rdelz)*
                            dtcldden,0.) ;
DIAGOUTPUT1(qr) ;
DIAGOUTPUT1(qs) ;
DIAGOUTPUT1(falk2) ;
DIAGOUTPUT1(w2) ;
DIAGOUTPUT11(falk2) ;
         qs[k] = MAX(qs[k]-
                            (falk2[k]-falk2[k+1]*delz[k+1]*rdelz)*
                            dtcldden,0.) ;
DIAGOUTPUT1(qs) ;
       }

       for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
DIAGOUTPUT1(t) ;
DIAGOUTPUT1(qs) ;
         if ( t[k] > t0c && qs[k] > 0.) {
           xlf = xlf0 ;
           w3[k] = VENFAC(p[k],t[k],den[k]) ;
           coeres = rslope2s[k]*sqrt(rslopes[k]*rslopebs[2]) ;
           psmlt[k] = XKA(t[k],den[k])/xlf*(t0c-t[k])*pi/2.
                     *n0sfac[k]*(precs1*rslope2s[k]+precs2
                     *w3[k]*coeres) ;
           psmlt[k] = MIN(MAX(psmlt[k]*dtcld*rmstep,-qs[k]*rmstep),0.) ;
           qs[k] += psmlt[k] ;
DIAGOUTPUT1i(mstep) ;
DIAGOUTPUT1i(n) ;
DIAGOUTPUT1(qr) ;
DIAGOUTPUT1(psmlt) ;
           qr[k] -= psmlt[k] ;
DIAGOUTPUT1(qr) ;

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
       numdt = MAX( trunc(w2[k]*dtcld+.5+.5),1) ;
       if(numdt > mstep) mstep = numdt ;
     }
     rmstep = 1./mstep ;

     for ( n = 1 ; n <= mstep ; n++ ) {
       k = kpe - 1 ;
       falkc[k] = den[k]*qi[k]*w2[k]*rmstep ;
       fallc[k] = fallc[k]+falkc[k] ;
       qi[k] = MAX(qi[k]-falkc[k]*dtcld/den[k],0.) ;
       for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
         falkc[k] = den[k]*qi[k]*w2[k]*rmstep ;
         fallc[k] = fallc[k]+falkc[k] ;
         qi[k] = MAX(qi[k]-(falkc[k]-falkc[k+1]
                 *delz[k+1]/delz[k])*dtcld/den[k],0.) ;
       }
     }
     fallsum = fall1[1]+fall2[1]+fallc[1] ;
     fallsum_qsi = fall2[1]+fallc[1] ;
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
       float supcol = t0c-t[k] ;
       xlf = xls-xl[k] ;
       if( supcol < 0. ) xlf = xlf0 ;
       if( supcol < 0 && qi[k] > 0. ) {
         qc[k] = qc[k] + qi[k] ;
         t[k] = t[k] - xlf/cpm[k]*qi[k] ;
         qi[k] = 0. ;
       }
//---------------------------------------------------------------
// pihmf: homogeneous freezing of cloud water below -40c [HL A45]
//        (T<-40C: C->I)
//---------------------------------------------------------------
       if( supcol > 40. && qc[k] > 0. ) {
         qi[k] = qi[k] + qc[k] ;
         t[k] = t[k] + xlf/cpm[k]*qc[k] ;
         qc[k] = 0. ;
       }
//---------------------------------------------------------------
// pihtf: heterogeneous freezing of cloud water [HL A44]
//        (T0>T>-40C: C->I)
//---------------------------------------------------------------
       if ( supcol > 0. && qc[k] > 0.) {
         float pfrzdtc = MIN(pfrz1*(exp(pfrz2*supcol)-1.)
           *den[k]/denr/xncr*qc[k]*qc[k]*dtcld,qc[k]) ;
         qi[k] = qi[k] + pfrzdtc ;
         t[k] = t[k] + xlf/cpm[k]*pfrzdtc ;
         qc[k] = qc[k]-pfrzdtc ;
       }
//---------------------------------------------------------------
// psfrz: freezing of rain water [HL A20] [LFO 45]
//        (T<T0, R->S)
//---------------------------------------------------------------
       if( supcol > 0. && qr[k] > 0. ) {
         float temp = rsloper[k] ;
         temp = temp*temp*temp*temp*temp*temp*temp ;
         float pfrzdtr = MIN(20.*(pi*pi)*pfrz1*n0r*denr/den[k]
               *(exp(pfrz2*supcol)-1.)*temp*dtcld,
               qr[k]) ;
         qs[k] = qs[k] + pfrzdtr ;
         t[k] = t[k] + xlf/cpm[k]*pfrzdtr ;
         qr[k] = qr[k]-pfrzdtr ;
DIAGOUTPUT1(qr) ;
       }
     }

//----------------------------------------------------------------
//     rsloper: reverse of the slope parameter of the rain(m)
//     xka:    thermal conductivity of air(jm-1s-1k-1)
//     work1:  the thermodynamic term in the denominator associated with
//             heat conduction and vapor diffusion
//             (ry88, y93, h85)
//     work2: parameter associated with the ventilation effects(y93)

     for ( k = kps-1 ; k <= kpe - 1 ; k++ ) {
       float supcol = t0c - t[k] ;
       n0sfac[k] = MAX(MIN(exp(alpha*supcol),n0smax/n0s),1.) ;
       if ( qr[k] <= qcrmin ) {
         rsloper[k]  = rslopermax ;
         rslopebr[k] = rsloperbmax ;
         rslope2r[k] = rsloper2max ;
         rslope3r[k] = rsloper3max ;
       } else {
         rsloper[k] = 1./(sqrt(sqrt(pidn0r/((qr[k])*(den[k]))))) ;
DIAGOUTPUT1(rsloper) ;
DIAGOUTPUT1(qr) ;
DIAGOUTPUT1(den) ;
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
         rslopes[k] = 1./(sqrt(sqrt(pidn0s*(n0sfac[k])/((qs[k])*(den[k]))))) ;
         rslopebs[k] = exp(log(rslopes[k])*bvts) ;
         rslope2s[k] = rslopes[k] * rslopes[k] ;
         rslope3s[k] = rslope2s[k] * rslopes[k] ;
       }
    }

    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
       w1[k] = DIFFAC(xl[k],p[k],t[k],den[k],qs1[k]) ;
       w2[k] = DIFFAC(xls,p[k],t[k],den[k],qs2[k]) ;
       w3[k] = VENFAC(p[k],t[k],den[k]) ;
    }

//
//===============================================================
//
// warm rain processes
//
// - follows the processes in RH83 and LFO except for autoconcersion
//
//===============================================================
//
    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
      float supsat = MAX(q[k],qmin)-qs1[k] ;
      float satdt = supsat/dtcld ;
//---------------------------------------------------------------
// praut: auto conversion rate from cloud to rain [HDC 16]
//        (C->R)
//---------------------------------------------------------------
      if(qc[k] > qc0) {
        praut[k] = qck1*exp(log(qc[k])*((7./3.))) ;
        praut[k] = MIN(praut[k],qc[k]/dtcld) ;
      }
//---------------------------------------------------------------
// pracw: accretion of cloud water by rain [HL A40] [LFO 51]
//        (C->R)
//---------------------------------------------------------------
      if(qr[k] > qcrmin && qc[k] > qmin) {
        pracw[k] = MIN(pacrr*rslope3r[k]*rslopebr[k]
                   *qc[k]*denfac[k],qc[k]/dtcld) ;
      }
//---------------------------------------------------------------
// prevp: evaporation/condensation rate of rain [HDC 14]
//        (V->R or R->V)
//---------------------------------------------------------------
      if(qr[k] > 0.) {
        coeres = rslope2r[k]*sqrt(rsloper[k]*rslopebr[k]) ;
        prevp[k] = (rh1[k]-1.)*(precr1*rslope2r[k]
                     +precr2*w3[k]*coeres)/w1[k] ;
DIAGOUTPUT1(prevp) ;
DIAGOUTPUT1(qr) ;
DIAGOUTPUT1(rsloper) ;
DIAGOUTPUT1(rslope2r) ;
DIAGOUTPUT1(rslopebr) ;
DIAGOUTPUT1(w1) ;
DIAGOUTPUT1(rh1) ;
        if(prevp[k] < 0.) {
          prevp[k] = MAX(prevp[k],-qr[k]/dtcld) ;
          prevp[k] = MAX(prevp[k],satdt/2) ;
        } else {
          prevp[k] = MIN(prevp[k],satdt/2) ;
        }
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
    rdtcld = 1./dtcld ;
    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          float supcol = t0c-t[k] ;
          float supsat = MAX(q[k],qmin)-qs2[k] ;
          float satdt = supsat/dtcld ;
          int ifsat = 0 ;
//-------------------------------------------------------------
// Ni: ice crystal number concentraiton   [HDC 5c]
//-------------------------------------------------------------
          float temp = (den[k]*MAX(qi[k],qmin)) ;
          temp = sqrt(sqrt(temp*temp*temp)) ;
          xni[k] = MIN(MAX(5.38e7*temp,1.e3),1.e6) ;
          float eacrs = exp(0.07*(-supcol)) ;
//
          if(supcol > 0) {
            if(qs[k] > qcrmin && qi[k] > qmin) {
              xmi = den[k]*qi[k]/xni[k] ;
              diameter  = MIN(dicon * sqrt(xmi),dimax) ;
              vt2i = 1.49e4*pow(diameter,(float)1.31) ;
              vt2s = pvts*rslopebs[k]*denfac[k] ;
//-------------------------------------------------------------
// psaci: Accretion of cloud ice by rain [HDC 10]
//        (T<T0: I->S)
//-------------------------------------------------------------
              acrfac = 2.*rslope3s[k]+2.*diameter*rslope2s[k]
                      +diameter*diameter*rslopes[k] ;
              psaci[k] = pi*qi[k]*eacrs*n0s*n0sfac[k]
                           *abs(vt2s-vt2i)*acrfac*.25 ;
            }
//-------------------------------------------------------------
// psacw: Accretion of cloud water by snow  [HL A7] [LFO 24]
//        (T<T0: C->S, and T>=T0: C->R)
//-------------------------------------------------------------
            if(qs[k] > qcrmin && qc[k] > qmin) {
              psacw[k] = MIN(pacrc*n0sfac[k]*rslope3s[k] 
                           *rslopebs[k]*qc[k]*denfac[k]
                           ,qc[k]*rdtcld) ;
            }
//-------------------------------------------------------------
// pidep: Deposition/Sublimation rate of ice [HDC 9]
//       (T<T0: V->I or I->V)
//-------------------------------------------------------------
            if(qi[k] > 0 && ifsat != 1) {
              xmi = den[k]*qi[k]/xni[k] ;
              diameter = dicon * sqrt(xmi) ;
              pidep[k] = 4.*diameter*xni[k]*(rh2[k]-1.)/w2[k] ;
              supice = satdt-prevp[k] ;
              if(pidep[k] < 0.) {
                pidep[k] = MAX(MAX(pidep[k],satdt*.5),supice) ;
                pidep[k] = MAX(pidep[k],-qi[k]*rdtcld) ;
              } else {
                pidep[k] = MIN(MIN(pidep[k],satdt*.5),supice) ;
              }
              if(abs(prevp[k]+pidep[k]) >= abs(satdt)) ifsat = 1 ;
            }
          }
//-------------------------------------------------------------
// psdep: deposition/sublimation rate of snow [HDC 14]
//        (V->S or S->V)
//-------------------------------------------------------------
          if( qs[k] > 0. && ifsat != 1) {
            coeres = rslope2s[k]*sqrt(rslopes[k]*rslopebs[k]) ;
            psdep[k] = (rh2[k]-1.)*n0sfac[k]
                         *(precs1*rslope2s[k]+precs2
                         *w3[k]*coeres)/w2[k] ;
            supice = satdt-prevp[k]-pidep[k] ;
            if(psdep[k] < 0.) {
              psdep[k] = MAX(psdep[k],-qs[k]*rdtcld) ;
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
          if(supcol > 0) {
            if(supsat > 0 && ifsat != 1) {
              supice = satdt-prevp[k]-pidep[k]-psdep[k] ; 
              xni0 = 1.e3*exp(0.1*supcol) ;
              roqi0 = 4.92e-11*exp(log(xni0)*(1.33));
              pigen[k] = MAX(0.,(roqi0/den[k]-MAX(qi[k],0.))
                         *rdtcld) ;
              pigen[k] = MIN(MIN(pigen[k],satdt),supice) ;
            }
//
//-------------------------------------------------------------
// psaut: conversion(aggregation) of ice to snow [HDC 12]
//       (T<T0: I->S)
//-------------------------------------------------------------
            if(qi[k] > 0.) {
              qimax = roqimax/den[k] ;
              psaut[k] = MAX(0.,(qi[k]-qimax)*rdtcld) ;
            }
          }
//-------------------------------------------------------------
// psevp: Evaporation of melting snow [HL A35] [RH83 A27]
//       (T>T0: S->V)
//-------------------------------------------------------------
          if(supcol < 0.) {
            if(qs[k] > 0. && rh1[k] < 1.) {
              psevp[k] = psdep[k]*w2[k]/w1[k] ;
            }  // asked Jimy about this, 11.6.07, JM
            psevp[k] = MIN(MAX(psevp[k],-qs[k]*rdtcld),0.) ;
          }
      }


//
//
//----------------------------------------------------------------
//     check mass conservation of generation terms and feedback to the
//     large scale
//
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          if(t[k]<=t0c) {
//
//     cloud water
//
            value = MAX(qmin,qc[k]) ;
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
            value = MAX(qmin,qi[k]) ;
            source = (psaut[k]+psaci[k]-pigen[k]-pidep[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              psaut[k] = psaut[k]*factor ;
              psaci[k] = psaci[k]*factor ;
              pigen[k] = pigen[k]*factor ;
              pidep[k] = pidep[k]*factor ;
            }
//
            w3[k]=-(prevp[k]+psdep[k]+pigen[k]+pidep[k]) ;
//     update
DIAGOUTPUT1(q) ;
DIAGOUTPUT1(prevp) ;
DIAGOUTPUT1(psdep) ;
DIAGOUTPUT1(pigen) ;
DIAGOUTPUT1(pidep) ;
            q[k] = q[k]+w3[k]*dtcld ;
DIAGOUTPUT1(q) ;
            qc[k] = MAX(qc[k]-(praut[k]+pracw[k]+psacw[k])*dtcld,0.) ;
            qr[k] = MAX(qr[k]+(praut[k]+pracw[k]+prevp[k])*dtcld,0.) ;
            qi[k] = MAX(qi[k]-(psaut[k]+psaci[k]-pigen[k]-pidep[k])*dtcld,0.) ;
DIAGOUTPUT1(qs)
            qs[k] = MAX(qs[k]+(psdep[k]+psaut[k]+psaci[k]+psacw[k])*dtcld,0.) ;
DIAGOUTPUT1(qs)
            xlf = xls-xl[k] ;
            xlwork2 = -xls*(psdep[k]+pidep[k]+pigen[k])-xl[k]*prevp[k]-xlf*psacw[k] ;
            t[k] = t[k]-xlwork2/cpm[k]*dtcld ;
          } else {
//
//     cloud water
//
            value = MAX(qmin,qc[k]) ;
            source=(praut[k]+pracw[k]+psacw[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              praut[k] = praut[k]*factor ;
              pracw[k] = pracw[k]*factor ;
              psacw[k] = psacw[k]*factor ;
            }
//
//     snow
//
            value = MAX(qcrmin,qs[k]) ;
            source=(-psevp[k])*dtcld ;
            if (source > value) {
              factor = value/source ;
              psevp[k] = psevp[k]*factor ;
            }
            w3[k]=-(prevp[k]+psevp[k]) ;
//     update
DIAGOUTPUT1(q) ;
DIAGOUTPUT1(prevp) ;
DIAGOUTPUT1(psdep) ;
DIAGOUTPUT1(pigen) ;
DIAGOUTPUT1(pidep) ;
            q[k] = q[k]+w3[k]*dtcld ;
DIAGOUTPUT1(q) ;
            qc[k] = MAX(qc[k]-(praut[k]+pracw[k]+psacw[k])*dtcld,0.) ;
            qr[k] = MAX(qr[k]+(praut[k]+pracw[k]+prevp[k] +psacw[k])*dtcld,0.) ;
DIAGOUTPUT1(qs)
DIAGOUTPUT1(psevp)

#ifdef DEVICEEMU
if (ig == IDEBUG && jg == JDEBUG && k+1 == KDEBUG ) fprintf(stderr,"%8s %25.17e\n","ZAP p*dt",psevp[k]*dtcld) ;
if (ig == IDEBUG && jg == JDEBUG && k+1 == KDEBUG ) fprintf(stderr,"%8s %25.17e\n","ZAP q+p*dt",qs[k]+psevp[k]*dtcld) ;
#endif
            qs[k] = MAX(qs[k]+psevp[k]*dtcld,0.) ;
DIAGOUTPUT1(qs)
            xlf = xls-xl[k] ;
            xlwork2 = -xl[k]*(prevp[k]+psevp[k]) ;
            t[k] = t[k]-xlwork2/cpm[k]*dtcld ;
          }
      }
DIAGOUTPUT2(qs)
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
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          tr=ttp/t[k] ;
          qs1[k]=psat*exp(log(tr)*(xa))*exp(xb*(1.-tr)) ;
          qs1[k] = ep2 * qs1[k] / (p[k] - qs1[k]) ;
          qs1[k] = MAX(qs1[k],qmin) ;
      }
//
//----------------------------------------------------------------
//  pcond: condensational/evaporational rate of cloud water [HL A46] [RH83 A6]
//     if there exists additional water vapor condensated/if
//     evaporation of cloud water is not enough to remove subsaturation
//
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          w1[k] = ((MAX(q[k],qmin)-(qs1[k])) /
            (1.+(xl[k])*(xl[k])/(rv*(cpm[k]))*(qs1[k])/((t[k])*(t[k])))) ;
          // w3[k] = qc[k]+w1[k] ;   NOT USED
          pcond[k] = MIN(MAX(w1[k]/dtcld,0.),MAX(q[k],0.)/dtcld) ;
          if(qc[k] > 0. && w1[k] < 0.) {
            pcond[k] = MAX(w1[k],-qc[k])/dtcld ;
          }
DIAGOUTPUT1(q) ;
DIAGOUTPUT1(pcond) ;
DIAGOUTPUT1(qs1) ;
          q[k] = q[k]-pcond[k]*dtcld ;
DIAGOUTPUT1(q) ;
          qc[k] = MAX(qc[k]+pcond[k]*dtcld,0.) ;
          t[k] = t[k]+pcond[k]*xl[k]/cpm[k]*dtcld ;
      }
//
//
//----------------------------------------------------------------
//     padding for small values
//
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
          if(qc[k] <= qmin) qc[k] = 0.0 ;
          if(qi[k] <= qmin) qi[k] = 0.0 ;
      }

//////////// end of loop ////////////////
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(t)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(q)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qc)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qi)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qr)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(qs)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(den)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(p)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(delz)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(cpm)
}
for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
kDIAGOUTPUT1(xl)
}
