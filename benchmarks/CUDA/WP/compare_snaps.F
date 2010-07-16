! file format
!write(46) ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
!write(46) th(its:ite,kts:kte,jts:jte)
!write(46) q(its:ite,kts:kte,jts:jte)
!write(46) qc(its:ite,kts:kte,jts:jte)
!write(46) qr(its:ite,kts:kte,jts:jte)
!write(46) qi(its:ite,kts:kte,jts:jte)
!write(46) qs(its:ite,kts:kte,jts:jte)
!!INOUT 2D
!write(46) rain(its:ite,jts:jte)
!write(46) rainncv(its:ite,jts:jte)
!write(46) sr(its:ite,jts:jte)
!write(46) snow(its:ite,jts:jte)
!write(46) snowncv(its:ite,jts:jte)

      program compare_snaps
      implicit none
      integer, parameter :: unit1 = 31
      integer, parameter :: unit2 = 32
      integer ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte

      open(unit=unit1,file="snap_out_010",status="old",form="unformatted",err=22)
      open(unit=unit2,file="snap_gpu_010",status="old",form="unformatted",err=23)

      read(unit=unit1) ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
      write(*,*)ids,ide,jds,jde,kds,kde
      write(*,*)ims,ime,jms,jme,kms,kme
      write(*,*)its,ite,jts,jte,kts,kte
      read(unit=unit2) ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
      write(*,*)ids,ide,jds,jde,kds,kde
      write(*,*)ims,ime,jms,jme,kms,kme
      write(*,*)its,ite,jts,jte,kts,kte

      call read_and_output3( 'th', unit1, 88 )
      call read_and_output3( 'th', unit2, 98 )
      call read_and_output3( 'q', unit1, 88 )
      call read_and_output3( 'q', unit2, 98 )
      call read_and_output3( 'qc', unit1, 88 )
      call read_and_output3( 'qc', unit2, 98 )
      call read_and_output3( 'qr', unit1, 88 )
      call read_and_output3( 'qr', unit2, 98 )
      call read_and_output3( 'qi', unit1, 88 )
      call read_and_output3( 'qi', unit2, 98 )
      call read_and_output3( 'qs', unit1, 88 )
      call read_and_output3( 'qs', unit2, 98 )

      close(unit1)
      close(unit2)
      stop
22    write(*,*)'error opening ',unit1
      stop
23    write(*,*)'error opening ',unit2
      stop
      end

      subroutine read_and_output3 ( varname,iu, ou )
      implicit none
      integer iu,ou
      character*(*) varname
!           2          72           2          59           1          27
      integer, parameter :: ix = 71
      integer, parameter :: jx = 58
      integer, parameter :: kx = 27
      integer i,j,k
      real s3(ix,kx,jx) 

      read(iu)s3
      write(ou,*)ix,jx,' ',trim(varname)
      do j = 1,jx
      do i = 1,ix
        write(ou,*)s3(i,1,j)
      enddo
      enddo
      return
      end

      


      

      

