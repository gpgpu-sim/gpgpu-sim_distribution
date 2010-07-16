#ifndef STOREGPU_H
#define STOREGPU_H
/*==========================================================================
                              S T O R E  G P U

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
  StoreGPU library.

==========================================================================*/

/*==========================================================================

                                  INCLUDES

==========================================================================*/
#include <cutil.h>

/*==========================================================================

                             DATA DECLARATIONS

==========================================================================*/


/*--------------------------------------------------------------------------
                              TYPE DEFINITIONS
--------------------------------------------------------------------------*/
typedef enum {
  
  SG_OK = 0,
  SG_ERR_DEV_MEM_OVERFLOW = -1,
  
  
} sg_status_type;

// defines a set of elapsed time measurements taken at specified states while 
// running the GPU version of the hashing algorithm
typedef struct sg_time_breakdown {

  float exec_time;              /* kernel execution time */
  float device_mem_alloc_time;  /* time elapsed to allocate device buffers */
  float host_output_buffer_alloc_time; /* time elapsed to allocate 
				       host output buffer */
  float copy_in_time;           /* time elapsed to push data into the GPU */
  float copy_out_time;          /* time elapsed to get data from the GPU */
  float last_stage_time;        /* time elapsed doing the last stage of hasing 
				   on the CPU*/
} sg_time_breakdown_type;

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

/*===========================================================================

FUNCTION SG_INIT

DESCRIPTION
  Library initialization

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void sg_init();

/*===========================================================================

FUNCTION SG_MALLOC

DESCRIPTION
  Allocate the required memory size.

DEPENDENCIES
  None

RETURN VALUE
  pointer to the reseved buffer

===========================================================================*/
void* sg_malloc(unsigned int size);

/*===========================================================================

FUNCTION SG_FREE

DESCRIPTION
  Free the allocated buffer.

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
void sg_free(void* buffer);

/*===========================================================================

FUNCTION SG_MD5

DESCRIPTION
  Returns the MD5 hash of a the supplied buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_md5(unsigned char* buffer, int size, 
		      unsigned char** output, int* output_size,
		      sg_time_breakdown_type* time_break_down );


/*===========================================================================

FUNCTION SG_MD5_OVERLAP

DESCRIPTION
  Returns the MD5 hash of each block for the provided buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_md5_overlap(unsigned char* buffer, int size,
			      int block_size, int offset,
			      unsigned char** output, int* output_size, 
			      sg_time_breakdown_type* time_breakdown);


/*===========================================================================

FUNCTION SG_SHA1

DESCRIPTION
  Returns the SHA1 hash of a the supplied buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_sha1(unsigned char* buffer, int size, 
		       unsigned char** output, int* output_size,
		       sg_time_breakdown_type* time_breakdown);


/*===========================================================================

FUNCTION SG_SHA1_OVERLAP

DESCRIPTION
  Returns the SHA1 hash of each block for the provided buffer

DEPENDENCIES
  None

RETURN VALUE
  Hash value

===========================================================================*/
sg_status_type sg_sha1_overlap(unsigned char* buffer, int size,
			       int block_size, int offset,
			       unsigned char** output, int* output_size,
			       sg_time_breakdown_type* time_breakdown);
#endif /* STOREGPU_H */
