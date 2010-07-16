#!/usr/bin/perl

# 0 line number
# 1 keyword
# 2 action

open DIRECTIVES, "| cat -n | grep '//_def_' | sed 's/	.*\\\/\\\/_def_/    / ' > directives__" or die "writing directives" ;
open TEMP1, "> tmp1__" or die "making copy of code" ;

while (<STDIN>) {
   print DIRECTIVES ;
   print TEMP1 ;
}
close DIRECTIVES ;
close TEMP1 ;

open DEBUG, "> debug__" or die ;
## first pass, preprocess the directives

open DIRECTIVES, "< directives__" or die ;
while (<DIRECTIVES>) {
  print TEMP1 ;
  $line = $_ ;
  @t = split( ' ',$line ) ;
  $keyword = $t[1] ;
  $action = $t[2] ;
  @actionlist = split( ';', $action ) ;
  foreach $act ( @actionlist ) {
    @dim_vlist = split ( ':', $act ) ;
    $dim = $dim_vlist[0] ; $vlist = $dim_vlist[1] ;
    foreach $v ( split( ',', $vlist ) ) {
      $vars{$v} = $v ;
      $dimensionality{$v} = $dim ;
      if ( $keyword eq "arg" ) {
        $key{$v} = $keyword ;
        if ( $dim eq "ikj" ) { $ikj_args{$v} = $v ; }
        if ( $dim eq "ij" )  { $ij_args{$v} = $v ; }
      }
      if ( $keyword eq "local" ) {
        $key{$v} = $keyword ;
        if ( $dim eq "k" ) { $k_local{$v} = $v ; }
      }
      if ( $keyword eq "register" ) {
        $key{$v} = $keyword ;
        if ( $dim eq "0" ) { $register{$v} = $v ; }
      }
      if ( $keyword eq "copy_up_mem" ) {
        if ( $key{$v} ne "arg" ) {
          print "//warning: copy_up_mem of $v when $v is not arg.\n" ;
        } else {
          $copy_up_mem{$v} = $dim ;
          if ( $dim eq "ikj" ) { $ikj_shared{$v} = $v ; }
          if ( $dim eq "ij" )  { $ij_shared{$v} = $v ; }
        }
      }
      if ( $keyword eq "shared_mem_local" ) {
        $key{$v} = $keyword ;
        $shared_mem_local{$v} = $dim ;
        if ( $dim eq "ikj" ) { $ikj_shared{$v} = $v ; }
        if ( $dim eq "ij" )  { $ij_shared{$v} = $v ; }
      }
      if ( $keyword eq "copy_down_mem" ) {
        if ( $key{$v} ne "arg" ) {
          print "//warning: copy_down_mem of $v when $v is not arg.\n" ;
        } else {
          $copy_down_mem{$v} = $dim ;
          if ( $dim eq "ikj" ) { $ikj_shared{$v} = $v ; }
          if ( $dim eq "ij" )  { $ij_shared{$v} = $v ; }
        }
      }
    }
  }
}
close DIRECTIVES ;

## seond pass, modify the code
## and preprocess deferred directives

$spton=0 ;

open TEMP1, "< tmp1__" or die ;
while (<TEMP1>) {
  $line = $_ ;
  # toggle on and off between SPTSTART and SPTSTOP
  if    ( $line =~ "SPTSTART" ) { $spton = 1 ; } 
  elsif ( $line =~ "SPTSTOP" ) { $spton = 0 ; } 
  if ( $spton == 1 ) {

  # handle copy_up_mem and copy_down_mem directives in line
  if    ( $line =~ m/\/\/\s*_def_\s+copy_up_mem\s/ ) {
    @t = split( ' ',$line ) ;
    $action = $t[2] ;
    @dim_vlist = split ( ':', $action ) ;
    $vlist = $dim_vlist[1] ;
    foreach $v ( split( ',', $vlist ) ) {
      print "LOCSM(${v}_s,bx*by*kx) ;\n" ;
    }
    print "{ int k ; \n" ;
    foreach $v ( split( ',', $vlist ) ) {
      print "for(k=kps-1;k<kpe;k++){${v}_s[S3(ti,k,tj)]=${v}[P3(ti,k,tj)];}\n" ;
    }
    print "}\n" ;
  }
  elsif ( $line =~ m/\/\/\s*_def_\s+register\s/ ) {
    @t = split( ' ',$line ) ;
    $action = $t[2] ;
    @dim_vlist = split ( ':', $action ) ;
    $vlist = $dim_vlist[1] ;
    foreach $v ( split( ',', $vlist ) ) {
      print "float ${v}_reg ;\n" ;
    }
  }
  elsif ( $line =~ m/\/\/\s*_def_\s+shared_mem_local\s/ ) {
    @t = split( ' ',$line ) ;
    $action = $t[2] ;
    @dim_vlist = split ( ':', $action ) ;
    $vlist = $dim_vlist[1] ;
    foreach $v ( split( ',', $vlist ) ) {
      print "LOCSM(${v}_s,bx*by*kx) ;\n" ;
    }
  }
  elsif ( $line =~ m/\/\/\s*_def_\s+copy_down_mem\s/ ) {
    @t = split( ' ',$line ) ;
    $action = $t[2] ;
    @dim_vlist = split ( ':', $action ) ;
    $vlist = $dim_vlist[1] ;
    print "{ int k ; \n" ;
    foreach $v ( split( ',', $vlist ) ) {
      print "for(k=kps-1;k<kpe;k++){${v}[P3(ti,k,tj)]=${v}_s[S3(ti,k,tj)];}\n" ;
    }
    print "}\n" ;
  }
  elsif ( $line =~ m/\/\/\s*_def_\s+local\s/ ) {
    $line = $_ ;
    @t = split( ' ',$line ) ;
    $keyword = $t[1] ;
    $action = $t[2] ;
    @actionlist = split( ';', $action ) ;
    foreach $act ( @actionlist ) {
      @dim_vlist = split ( ':', $act ) ;
      $dim = $dim_vlist[0] ; $vlist = $dim_vlist[1] ;
      foreach $v ( split( ',', $vlist ) ) {
        if ( $dim eq "k" ) {
          print "#if (FLOAT_4 == 4)\n" ;
          print "   Float4 ${v}[MKX] ; \n" ;
          print "#else\n" ;
          print "   float ${v}[MKX] ; \n" ;
          print "#endif\n" ;
        }
      }
    }
  }
  # otherwise do not touch lines with // in them
  elsif ( ! ($line =~ m/\/\//) ) {
    @t = split( /\W+/,$line ) ;
    %seen = "" ;
    foreach $token ( @t ) {
      if ( ! $seen{$token} ) {
        $seen{$token} = $token ;
        foreach $v ( keys %vars ) {
          if ( "$v" eq "$token" ) {
            $dim = $dimensionality{$v} ;
            $keyw = $key{$v} ;
            $nodex=0 ;
            if      ( $keyw eq "arg" || $keyw eq "shared_mem_local" ) {
              if    ( $copy_up_mem{$v} || $shared_mem_local{$v} ) {
                if    ( $dim eq "ikj" ) { $orig = $v."\\[\(.*\?\)\\]" ; $repl1 = $v."_s"."A|S3(ti," ; $repl2 = ",tj)B|" ; }
                elsif ( $dim eq "ij" )  { $orig = $v       ; $repl1 = $v."_s"."A|S2(ti,tj)B|" ; $repl2 = "" ; }
              } else {
                if    ( $dim eq "ikj" ) { $orig = $v."\\[\(.*\?\)\\]" ; $repl1 = $v."A|P3(ti," ; $repl2 = ",tj)B|" ; }
                elsif ( $dim eq "ij" )  { $orig = $v       ; $repl1 = $v."A|P2(ti,tj)B|" ; $repl2 = ""  ; }
              }
            } elsif ( $keyw eq "register" ) {
              if    ( $dim eq "0" ) { $orig = $v."\\[\(.*\?\)\\]" ; $repl1 = $v."_reg" ; $repl2 = "" ; $nodex=1}
            } elsif ( $keyw eq "local" ) {
              if    ( $dim eq "k" ) { $orig = $v."\\[\(.*\?\)\\]" ; $repl1 = $v."A|" ; $repl2 = "B|" ; }
            }
# these repetitions are to handle multiple instances of the
# variable being indexed differently on the same line.
            if ( $nodex == 0 ) {
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$2$repl2$3/g ;
            } else {
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
              $line =~ s/(\W)$orig(\W)/$1$repl1$repl2$3/g ;
            }

            $line =~ s/A\|/[/g ;
            $line =~ s/B\|/]/g ;
          }
        }
      }
    }
  }
  }
  print $line ;
}

close TEMP1 ;
close DEBUG ;

unlink "directives__" ;
unlink "tmp1__" ;
unlink "debug__" ;


