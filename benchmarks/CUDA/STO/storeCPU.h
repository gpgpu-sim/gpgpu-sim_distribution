#ifndef STORECPU_H
#define STORECPU_H
/*==========================================================================
                                S T O R E  C P U

* Copyright (c) 2008, NetSysLab at the University of British Columbia
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the University nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY NetSysLab ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NetSysLab BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DESCRIPTION
  CPU version of the storeGPU library.


==========================================================================*/

/*==========================================================================

                                  INCLUDES

==========================================================================*/



/*==========================================================================

                             DATA DECLARATIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                              TYPE DEFINITIONS
--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------
                             FUNCTION PROTOTYPES
--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
                                  CONSTANTS
--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
                              GLOBAL VARIABLES
--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------
                                    MACROS
--------------------------------------------------------------------------*/

/*==========================================================================

                                  FUNCTIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                                    LOCAL FUNCTIONS
--------------------------------------------------------------------------*/



/*--------------------------------------------------------------------------
                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/
/*===========================================================================

FUNCTION SC_MD5

DESCRIPTION
  CPU version of the MD5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
//extern "C"
void sc_md5( unsigned char* buffer, int size, 
	     unsigned char** output, int* output_size);


/*===========================================================================

FUNCTION SC_MD5_OVERLAP

DESCRIPTION
  CPU version of the MD5 overlap algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
//extern "C"
void sc_md5_overlap(unsigned char* buffer, int size, int block_size, int offset,
		    unsigned char** output, int* output_size);

/*===========================================================================

FUNCTION SC_SHA1

DESCRIPTION
  CPU version of the SHA1 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
//extern "C"
void sc_sha1( unsigned char* buffer, int size, 
	     unsigned char** output, int* output_size);

/*===========================================================================

FUNCTION SC_SHA1_OVERLAP

DESCRIPTION
  CPU version of the SHA1 overlap algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
//extern "C"
void sc_sha1_overlap(unsigned char* buffer, int size, int block_size, 
		     int offset, unsigned char** output, int* output_size);

/*===========================================================================

FUNCTION SC_MD5_STANDARD

DESCRIPTION
  The standard MD5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
//extern "C"
void sc_md5_standard( unsigned char* buffer, int size, unsigned char** output);

/*===========================================================================

FUNCTION SC_SHA1_STANDARD

DESCRIPTION
  The standard SHA1 algorithm

DEPENDENCIES
  None

RETURN VALUE
  Hash

===========================================================================*/
//extern "C"
void sc_sha1_standard( unsigned char* buffer, int size, unsigned char** output);

#endif /* STORECPU_H */
