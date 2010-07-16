define( DIAGOUTPUT1, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG&&k+1==KDEBUG) fprintf(stderr,"ZAP %8s %25.17e \n", "$1", $1[k] );
#endif
')
define( DIAGOUTPUT1i, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG&&k+1==KDEBUG) fprintf(stderr,"ZAP %8s %20d \n", "$1", $1 );
#endif
')
define( DIAGOUTPUT11, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG&&k+1==KDEBUG) fprintf(stderr,"ZAP %8s %25.17e \n", "$1", $1[k+1] );
#endif
')
define( DIAGOUTPUT2, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG) fprintf(stderr,"ZAP %8s %25.17e \n", "$1", $1[KDEBUG-1] );
#endif
')
define( kDIAGOUTPUT1, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG) fprintf(stderr,"ZAP %8s %25.17e \n", "$1", $1[k] );
#endif
')
define( kDIAGOUTPUT1i, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG) fprintf(stderr,"ZAP %8s %20d \n", "$1", $1 );
#endif
')
define( kDIAGOUTPUT11, `
#if defined(DEVICEEMU) && defined(DEBUGOUTPUT)
 if (ig==IDEBUG&&jg==JDEBUG) fprintf(stderr,"ZAP %8s %25.17e \n", "$1", $1[k+1] );
#endif
')
