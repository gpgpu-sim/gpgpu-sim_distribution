// WSM5 Constants

#if 1
# define epsilon        1.e-15
# define r_d            287.
# define rhoair0        1.28
# define rhosnow        100.
# define dens   rhosnow
# define rhowater       1000.
# define svpt0          .27314999389648438e+03
# define xlv            2.5e6
#endif

#define g       0.981000041961670E+01
#define r_v      0.461600006103516E+03
#define rv      r_v
#define cice    0.210600000000000E+04
#define cliq    0.419000000000000E+04
#define denr    0.100000000000000E+04
#define den0    0.127999997138977E+01
#define xlf0    0.350000000000000E+06
#define xlv0    0.250000000000000E+07
#define xls     0.285000000000000E+07
#define t0c     0.273149993896484E+03
#define qmin    0.100000000362749E-14
#define ep1     0.608362436294556E+00
#define ep2     0.621750414371490E+00
#define psat    0.610780029296875E+03
#define alpha   0.120000000000000E+00
#define n0smax  0.100000000000000E+12
#define n0s     0.200000000000000E+07
#define n0r     0.800000000000000E+07
#define qcrmin  0.100000000000000E-08
#define avtr    0.841900000000000E+03
#define bvtr    0.800000000000000E+00
#define g1pbr   0.931232915622909E+00
#define g3pbr   0.469078683336385E+01
#define g4pbr   0.178173289058329E+02
#define g5pbro2 0.182658695197891E+01
#define avts    0.117200000000000E+02
#define bvts    0.410000000000000E+00
#define g1pbs   0.886676521690526E+00
#define g3pbs   0.301156382231086E+01
#define g4pbs   0.102654190601850E+02
#define g5pbso2   1.550308
#define r0      0.800000000000000E-05
#define peaut   0.550000000000000E+00
#define xncr    0.300000000000000E+09
#define xmyu    0.171800000000000E-04
#define lamdarmax       0.800000000000000E+05
#define lamdasmax       0.100000000000000E+06
#define lamdagmax       0.600000000000000E+05
#define pi      0.314159265358979E+01
#define dicon   0.119000000000000E+02
#define dimax   0.500000000000000E-03
#define pfrz1   0.100000000000000E+03
#define pfrz2   0.660000000000000E+00
#define eacrr   0.100000000000000E+01
#define eacrc   0.100000000000000E+01

   double cpv = 4.*r_v     ;
   double cp  = 7.*r_d/2.  ;
   double cv  = cp-r_d     ;
   double cpd = cp         ;

   //double ep_1 = r_v/r_d-1.  ;
   //double ep_2 = r_d/r_v    ;
   double pvtr = avtr*g4pbr/6.  ;
   double pvts = avts*g4pbs/6.  ;
   double xlv1 = cliq - cv ;

   double rslopermax = 1./lamdarmax ;
   double rslopesmax = .10000000000000001e-04 ; // 1./lamdasmax ;
   double rsloperbmax = 0.11954406247375457E-03 ; // exp(log(rslopermax) * bvtr) ;
   double rslopesbmax = .89125093813374589e-02 ; // exp(log(rslopesmax) * bvts) ;
   double rsloper2max = rslopermax * rslopermax ;
   double rslopes2max = rslopesmax * rslopesmax ;
   double rsloper3max = rsloper2max * rslopermax ;
   double rslopes3max = rslopes2max * rslopesmax ;

   double pidn0r =  pi*denr*n0r ;
   double pidn0s =  pi*dens*n0s ;

   double precs1 = 4.*n0s*.65 ;
   double precs2 = 4.*n0s*.44*sqrt(avts)*g5pbso2 ;
   double qc0 = 4./3.*pi*denr*(r0*r0*r0)*xncr/den0 ;
   double qck1 = .104*9.8*peaut/pow((xncr*denr),(1./3.))/xmyu*pow(den0,(4./3.)) ;
   double precr1 = 2.*pi*n0r*.78 ;
   double precr2 = 2.*pi*n0r*.31*sqrt(avtr)*g5pbro2 ;
   double pacrr = pi*n0r*avtr*g3pbr*.25*eacrr ;
   double pacrc = pi*n0s*avts*g3pbs*.25*eacrc ;
   double roqimax = 2.08e22*pow(dimax,8) ;
