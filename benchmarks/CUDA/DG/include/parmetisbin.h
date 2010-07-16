/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * par_metis.h
 *
 * This file includes all necessary header files
 *
 * Started 8/27/94
 * George
 *
 * $Id: parmetisbin.h,v 1.1 2003/07/21 17:50:23 karypis Exp $
 */

/*
#define DEBUG			1
#define DMALLOC			1
*/

#include <stdheaders.h>
#include "parmetis.h"

#ifdef DMALLOC
#include <dmalloc.h>
#endif

#include <rename.h>
#include <defs.h>
#include <struct.h>
#include <macros.h>
#include <proto.h>

