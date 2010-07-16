/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * metis.h
 *
 * This file includes all necessary header files
 *
 * Started 8/27/94
 * George
 *
 * $Id: metis.h,v 1.3 2003/07/25 13:52:00 karypis Exp $
 */

/*
#define	DEBUG		1
#define	DMALLOC		1
*/

#include <stdheaders.h>

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#include "../parmetis.h"  /* Get the idxtype definition */
#include <defs.h>
#include <struct.h>
#include <macros.h>
#include <rename.h>
#include <proto.h>

