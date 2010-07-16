#ifndef CRAY
# ifdef NOUNDERSCORE
#      define RSL_INTERNAL_MICROCLOCK rsl_internal_microclock
# else
#   ifdef F2CSTYLE
#      define RSL_INTERNAL_MICROCLOCK rsl_internal_microclock__
#   else
#      define RSL_INTERNAL_MICROCLOCK rsl_internal_microclock_
#   endif
# endif
#endif
#include <sys/time.h>

RSL_INTERNAL_MICROCLOCK ()
{
    struct timeval tb ;
    struct timezone tzp ;
    int isec ;  /* seconds */
    int usec ;  /* microseconds */
    int msecs ;
    gettimeofday( &tb, &tzp ) ;
    isec = tb.tv_sec ;
    usec = tb.tv_usec ;
    msecs = 1000000 * isec + usec ;
    return(msecs) ;
}

c_pow_ ( float * a, float * b )
{
    *a = pow( *a , *b ) ;
}


