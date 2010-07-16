! IBM libmassv compatibility library
! 

#ifndef NATIVE_MASSV
      subroutine vdiv(z,x,y,n)
      double precision x(*),y(*),z(*)
      do 10 j=1,n
      z(j)=x(j)/y(j)
   10 continue
      return
      end

      subroutine vsdiv(z,x,y,n)
      real x(*),y(*),z(*)
      do 10 j=1,n
      z(j)=x(j)/y(j)
   10 continue
      return
      end

      subroutine vexp(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=exp(x(j))
   10 continue
      return
      end

      subroutine vsexp(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=exp(x(j))
   10 continue
      return
      end

      subroutine vlog(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=log(x(j))
   10 continue
      return
      end

      subroutine vslog(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=log(x(j))
   10 continue
      return
      end

      subroutine vrec(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=1.d0/x(j)
   10 continue
      return
      end

      subroutine vsrec(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=1.d0/x(j)
   10 continue
      return
      end

      subroutine vrsqrt(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=1.d0/sqrt(x(j))
   10 continue
      return
      end

      subroutine vsrsqrt(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=1.d0/sqrt(x(j))
   10 continue
      return
      end

      subroutine vsincos(x,y,z,n)
      double precision x(*),y(*),z(*)
      do 10 j=1,n
      x(j)=sin(z(j))
      y(j)=cos(z(j))
   10 continue
      return
      end

      subroutine vssincos(x,y,z,n)
      real x(*),y(*),z(*)
      do 10 j=1,n
      x(j)=sin(z(j))
      y(j)=cos(z(j))
   10 continue
      return
      end

      subroutine vsqrt(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=sqrt(x(j))
   10 continue
      return
      end

      subroutine vssqrt(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=sqrt(x(j))
   10 continue
      return
      end

      subroutine vtan(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=tan(x(j))
   10 continue
      return
      end

      subroutine vstan(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=tan(x(j))
   10 continue
      return
      end

      subroutine vatan2(z,y,x,n)
      double precision x(*),y(*),z(*)
      do 10 j=1,n
      z(j)=atan2(y(j),x(j))
   10 continue
      return
      end

      subroutine vsatan2(z,y,x,n)
      real x(*),y(*),z(*)
      do 10 j=1,n
      z(j)=atan2(y(j),x(j))
   10 continue
      return
      end

      subroutine vasin(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=asin(x(j))
   10 continue
      return
      end

      subroutine vsin(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=sin(x(j))
   10 continue
      return
      end

      subroutine vssin(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=sin(x(j))
   10 continue
      return
      end

      subroutine vacos(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=acos(x(j))
   10 continue
      return
      end

      subroutine vcos(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=cos(x(j))
   10 continue
      return
      end

      subroutine vscos(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=cos(x(j))
   10 continue
      return
      end

      subroutine vcosisin(y,x,n)
      complex*16 y(*)
      double precision x(*)
      do 10 j=1,n
      y(j)=dcmplx(cos(x(j)),sin(x(j)))
   10 continue
      return
      end

      subroutine vscosisin(y,x,n)
      complex*8 y(*)
      real x(*)
      do 10 j=1,n
      y(j)= cmplx(cos(x(j)),sin(x(j)))
   10 continue
      return
      end

      subroutine vdint(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
!     y(j)=dint(x(j))
      y(j)=int(x(j))
   10 continue
      return
      end

      subroutine vdnint(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
!     y(j)=dnint(x(j))
      y(j)=nint(x(j))
   10 continue
      return
      end

      subroutine vlog10(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=log10(x(j))
   10 continue
      return
      end

!      subroutine vlog1p(y,x,n)
!      double precision x(*),y(*)
!      interface
!        double precision function log1p(%val(x))
!          double precision x
!        end function log1p
!      end interface
!      do 10 j=1,n
!      y(j)=log1p(x(j))
!   10 continue
!      return
!      end

      subroutine vcosh(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=cosh(x(j))
   10 continue
      return
      end

      subroutine vsinh(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=sinh(x(j))
   10 continue
      return
      end

      subroutine vtanh(y,x,n)
      double precision x(*),y(*)
      do 10 j=1,n
      y(j)=tanh(x(j))
   10 continue
      return
      end

!      subroutine vexpm1(y,x,n)
!      double precision x(*),y(*)
!      interface
!        double precision function expm1(%val(x))
!          double precision x
!        end function expm1
!      end interface 
!      do 10 j=1,n
!      y(j)=expm1(x(j))
!   10 continue
!      return
!      end


      subroutine vsasin(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=asin(x(j))
   10 continue
      return
      end

      subroutine vsacos(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=acos(x(j))
   10 continue
      return
      end

      subroutine vscosh(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=cosh(x(j))
   10 continue
      return
      end

!      subroutine vsexpm1(y,x,n)
!      real x(*),y(*)
!      interface
!        double precision function expm1(%val(x))
!          double precision x
!        end function expm1
!      end interface
!      do 10 j=1,n
!      y(j)=expm1(real(x(j),8))
!   10 continue
!      return
!      end

      subroutine vslog10(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=log10(x(j))
   10 continue
      return
      end

!      subroutine vslog1p(y,x,n)
!      real x(*),y(*)
!      interface
!        double precision function log1p(%val(x))
!          double precision x
!        end function log1p
!      end interface
!      do 10 j=1,n
!      y(j)=log1p(real(x(j),8))
!   10 continue
!      return
!      end


      subroutine vssinh(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=sinh(x(j))
   10 continue
      return
      end

      subroutine vstanh(y,x,n)
      real x(*),y(*)
      do 10 j=1,n
      y(j)=tanh(x(j))
   10 continue
      return
      end
#endif

      subroutine vspow(z,y,x,n)
      real x(*),y(*),z(*)
      do 10 j=1,n
      z(j)=y(j)**x(j)
   10 continue
      return
      end

      subroutine vpow(z,y,x,n)
      double precision x(*),y(*),z(*)
      do 10 j=1,n
      z(j)=y(j)**x(j)
   10 continue
      return
      end

