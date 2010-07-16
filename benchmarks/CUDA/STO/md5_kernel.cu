/*==========================================================================
                                MD5 KERNEL

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
#include <string.h>
#include <stdio.h>
#include "cust.h"

/*==========================================================================

                             DATA DECLARATIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                              TYPE DEFINITIONS
--------------------------------------------------------------------------*/
typedef struct {
  unsigned long total[2];     /*!< number of bytes processed  */
  unsigned long state[4];     /*!< intermediate digest state  */
  unsigned char buffer[64];   /*!< data block being processed */
} md5_context;

/*--------------------------------------------------------------------------
                             FUNCTION PROTOTYPES
--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
                                  CONSTANTS
--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
                              GLOBAL VARIABLES
--------------------------------------------------------------------------*/

__device__
const unsigned char md5_padding[64] =
{
  0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/*--------------------------------------------------------------------------
                                    MACROS
--------------------------------------------------------------------------*/
// 32-bit integer manipulation macros (little endian)
#ifndef GET_UINT32_LE
#define GET_UINT32_LE(n,b,i)                            \
{                                                       \
    (n) = ( (unsigned long) (b)[(i)    ]       )        \
        | ( (unsigned long) (b)[(i) + 1] <<  8 )        \
        | ( (unsigned long) (b)[(i) + 2] << 16 )        \
        | ( (unsigned long) (b)[(i) + 3] << 24 );       \
}
#endif

#ifndef PUT_UINT32_LE
#define PUT_UINT32_LE(n,b,i)                            \
{                                                       \
    (b)[(i)    ] = (unsigned char) ( (n)       );       \
    (b)[(i) + 1] = (unsigned char) ( (n) >>  8 );       \
    (b)[(i) + 2] = (unsigned char) ( (n) >> 16 );       \
    (b)[(i) + 3] = (unsigned char) ( (n) >> 24 );       \
}
#endif

#ifdef FEATURE_SHARED_MEMORY
// current thread stride.
#define SHARED_MEMORY_INDEX(index) (32 * (index) + (threadIdx.x & 0x1F))
#endif /* FEATURE_SHARED_MEMORY */



/*==========================================================================

                                  FUNCTIONS

==========================================================================*/

/*--------------------------------------------------------------------------
                                    LOCAL FUNCTIONS
--------------------------------------------------------------------------*/


#ifndef FEATURE_SHARED_MEMORY
/*===========================================================================

FUNCTION <Name>

DESCRIPTION
  MD5 context setup

DEPENDENCIES
  <dep.>

RETURN VALUE
  <return>

===========================================================================*/
__device__
static void md5_starts( md5_context *ctx ) {
  ctx->total[0] = 0;
  ctx->total[1] = 0;
  
  ctx->state[0] = 0x67452301;
  ctx->state[1] = 0xEFCDAB89;
  ctx->state[2] = 0x98BADCFE;
  ctx->state[3] = 0x10325476;
}

/*===========================================================================

FUNCTION MD5_PROCESS

DESCRIPTION
  <Desc.>

DEPENDENCIES
  <dep.>

RETURN VALUE
  <return>

===========================================================================*/
__device__
static void md5_process( md5_context *ctx, unsigned char data[64] ) {

  unsigned long A, B, C, D;
  unsigned long *X = (unsigned long *)data;


  GET_UINT32_LE( X[ 0], data,  0 );
  GET_UINT32_LE( X[ 1], data,  4 );
  GET_UINT32_LE( X[ 2], data,  8 );
  GET_UINT32_LE( X[ 3], data, 12 );
  GET_UINT32_LE( X[ 4], data, 16 );
  GET_UINT32_LE( X[ 5], data, 20 );
  GET_UINT32_LE( X[ 6], data, 24 );
  GET_UINT32_LE( X[ 7], data, 28 );
  GET_UINT32_LE( X[ 8], data, 32 );
  GET_UINT32_LE( X[ 9], data, 36 );
  GET_UINT32_LE( X[10], data, 40 );
  GET_UINT32_LE( X[11], data, 44 );
  GET_UINT32_LE( X[12], data, 48 );
  GET_UINT32_LE( X[13], data, 52 );
  GET_UINT32_LE( X[14], data, 56 );
  GET_UINT32_LE( X[15], data, 60 );
  
#undef S
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))
  
#undef P
#define P(a,b,c,d,k,s,t) {                            \
    a += F(b,c,d) + X[k] + t; a = S(a,s) + b;         \
  }						      \
  
  A = ctx->state[0];
  B = ctx->state[1];
  C = ctx->state[2];
  D = ctx->state[3];
  
#define F(x,y,z) (z ^ (x & (y ^ z)))
  
  P( A, B, C, D,  0,  7, 0xD76AA478 );
  P( D, A, B, C,  1, 12, 0xE8C7B756 );
  P( C, D, A, B,  2, 17, 0x242070DB );
  P( B, C, D, A,  3, 22, 0xC1BDCEEE );
  P( A, B, C, D,  4,  7, 0xF57C0FAF );
  P( D, A, B, C,  5, 12, 0x4787C62A );
  P( C, D, A, B,  6, 17, 0xA8304613 );
  P( B, C, D, A,  7, 22, 0xFD469501 );
  P( A, B, C, D,  8,  7, 0x698098D8 );
  P( D, A, B, C,  9, 12, 0x8B44F7AF );
  P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
  P( B, C, D, A, 11, 22, 0x895CD7BE );
  P( A, B, C, D, 12,  7, 0x6B901122 );
  P( D, A, B, C, 13, 12, 0xFD987193 );
  P( C, D, A, B, 14, 17, 0xA679438E );
  P( B, C, D, A, 15, 22, 0x49B40821 );
  
#undef F
  
#define F(x,y,z) (y ^ (z & (x ^ y)))
  
  P( A, B, C, D,  1,  5, 0xF61E2562 );
  P( D, A, B, C,  6,  9, 0xC040B340 );
  P( C, D, A, B, 11, 14, 0x265E5A51 );
  P( B, C, D, A,  0, 20, 0xE9B6C7AA );
  P( A, B, C, D,  5,  5, 0xD62F105D );
  P( D, A, B, C, 10,  9, 0x02441453 );
  P( C, D, A, B, 15, 14, 0xD8A1E681 );
  P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
  P( A, B, C, D,  9,  5, 0x21E1CDE6 );
  P( D, A, B, C, 14,  9, 0xC33707D6 );
  P( C, D, A, B,  3, 14, 0xF4D50D87 );
  P( B, C, D, A,  8, 20, 0x455A14ED );
  P( A, B, C, D, 13,  5, 0xA9E3E905 );
  P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
  P( C, D, A, B,  7, 14, 0x676F02D9 );
  P( B, C, D, A, 12, 20, 0x8D2A4C8A );
  
#undef F
  
#define F(x,y,z) (x ^ y ^ z)
  
  P( A, B, C, D,  5,  4, 0xFFFA3942 );
  P( D, A, B, C,  8, 11, 0x8771F681 );
  P( C, D, A, B, 11, 16, 0x6D9D6122 );
  P( B, C, D, A, 14, 23, 0xFDE5380C );
  P( A, B, C, D,  1,  4, 0xA4BEEA44 );
  P( D, A, B, C,  4, 11, 0x4BDECFA9 );
  P( C, D, A, B,  7, 16, 0xF6BB4B60 );
  P( B, C, D, A, 10, 23, 0xBEBFBC70 );
  P( A, B, C, D, 13,  4, 0x289B7EC6 );
  P( D, A, B, C,  0, 11, 0xEAA127FA );
  P( C, D, A, B,  3, 16, 0xD4EF3085 );
  P( B, C, D, A,  6, 23, 0x04881D05 );
  P( A, B, C, D,  9,  4, 0xD9D4D039 );
  P( D, A, B, C, 12, 11, 0xE6DB99E5 );
  P( C, D, A, B, 15, 16, 0x1FA27CF8 );
  P( B, C, D, A,  2, 23, 0xC4AC5665 );
  
#undef F
  
#define F(x,y,z) (y ^ (x | ~z))
  
  P( A, B, C, D,  0,  6, 0xF4292244 );
  P( D, A, B, C,  7, 10, 0x432AFF97 );
  P( C, D, A, B, 14, 15, 0xAB9423A7 );
  P( B, C, D, A,  5, 21, 0xFC93A039 );
  P( A, B, C, D, 12,  6, 0x655B59C3 );
  P( D, A, B, C,  3, 10, 0x8F0CCC92 );
  P( C, D, A, B, 10, 15, 0xFFEFF47D );
  P( B, C, D, A,  1, 21, 0x85845DD1 );
  P( A, B, C, D,  8,  6, 0x6FA87E4F );
  P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
  P( C, D, A, B,  6, 15, 0xA3014314 );
  P( B, C, D, A, 13, 21, 0x4E0811A1 );
  P( A, B, C, D,  4,  6, 0xF7537E82 );
  P( D, A, B, C, 11, 10, 0xBD3AF235 );
  P( C, D, A, B,  2, 15, 0x2AD7D2BB );
  P( B, C, D, A,  9, 21, 0xEB86D391 );
  
#undef F
  
  ctx->state[0] += A;
  ctx->state[1] += B;
  ctx->state[2] += C;
  ctx->state[3] += D;
}

/*===========================================================================

FUNCTION MD5_UPDATE

DESCRIPTION
  MD5 process buffer

DEPENDENCIES
  <dep.>

RETURN VALUE
  <return>

===========================================================================*/
__device__
static void md5_update( md5_context *ctx, unsigned char *input, int ilen ) {
  int fill;
  unsigned long left;
  
  if( ilen <= 0 )
    return;
  
  left = ctx->total[0] & 0x3F;
  fill = 64 - left;
  
  ctx->total[0] += ilen;
  ctx->total[0] &= 0xFFFFFFFF;
  
  if( ctx->total[0] < (unsigned long) ilen )
    ctx->total[1]++;
  
  if( left && ilen >= fill ) {
    
    //<ELSN>
    /*memcpy( (void *) (ctx->buffer + left),
      (void *) input, fill );*/
    for (int i = 0; i < fill; i++) {
      ctx->buffer[i+left] = input[i];
    }
    //</ELSN>
    
    md5_process( ctx, ctx->buffer );
    input += fill;
    ilen  -= fill;
    left = 0;
  }
  
  while( ilen >= 64 ) {
    md5_process( ctx, input );
    input += 64;
    ilen  -= 64;
  }
  
  if( ilen > 0 ) {	
    
    //<ELSN>
    /*	memcpy( (void *) (ctx->buffer + left),
	(void *) input, ilen );*/
    for (int i = 0; i < ilen; i++) {
      ctx->buffer[i+left] = input[i];
    }
    //</ELSN>
    
  }
}

/*===========================================================================

FUNCTION MD5_FINISH

DESCRIPTION
  MD5 final digest

DEPENDENCIES
  None.

RETURN VALUE
  <return>

===========================================================================*/
__device__
void md5_finish( md5_context *ctx, unsigned char *output ) {

  unsigned long last, padn;
  unsigned long high, low;
  unsigned char msglen[8];
  
  high = ( ctx->total[0] >> 29 ) | ( ctx->total[1] <<  3 );
  low  = ( ctx->total[0] <<  3 );
  
  PUT_UINT32_LE( low,  msglen, 0 );
  PUT_UINT32_LE( high, msglen, 4 );
  
  last = ctx->total[0] & 0x3F;
  padn = ( last < 56 ) ? ( 56 - last ) : ( 120 - last );
  
  md5_update( ctx, (unsigned char *) md5_padding, padn );
  md5_update( ctx, msglen, 8 );
  

  PUT_UINT32_LE( ctx->state[0], output,  0 );
#ifndef FEATURE_REDUCED_HASH_SIZE
  PUT_UINT32_LE( ctx->state[1], output,  4 );
  PUT_UINT32_LE( ctx->state[2], output,  8 );
  PUT_UINT32_LE( ctx->state[3], output, 12 );
#endif
}

/*===========================================================================

FUNCTION MD5_INTERNAL

DESCRIPTION
  Does the real md5 algorithm

DEPENDENCIES
  None

RETURN VALUE
  output is the hash result

===========================================================================*/
__device__
static void md5_internal( unsigned char *input, int ilen, 
			  unsigned char *output ) {
  md5_context ctx;
  
  md5_starts( &ctx );
  md5_update( &ctx, input, ilen );
  md5_finish( &ctx, output );
  
}
#endif /* #ifndef FEATURE_SHARED_MEMORY */

#ifdef FEATURE_SHARED_MEMORY
/*===========================================================================

FUNCTION MD5_INTERNAL

DESCRIPTION
  Does the real md5 algorithm.

DEPENDENCIES
  None

RETURN VALUE
  output is the hash result

===========================================================================*/

__device__
static void md5_internal( unsigned int *input, unsigned int *sharedMemory, 
			  int chunkSize, unsigned char *output ) {

  /* Number of passes (512 bit blocks) we have to do */
  int numberOfPasses = chunkSize / 64 + 1;
  /* Used during the hashing process */
  unsigned long A, B, C, D;
  /* Needed to do the little endian stuff */
  unsigned char *data = (unsigned char *)sharedMemory;

  /* Will hold the hash value through the 
     intermediate stages of MD5 algorithm */
  unsigned int state0 = 0x67452301;
  unsigned int state1 = 0xEFCDAB89;
  unsigned int state2 = 0x98BADCFE;
  unsigned int state3 = 0x10325476;


  /* Used to cache the shared memory index calculations, but testing showed 
     that it has no performance effect. */
  int x0 = SHARED_MEMORY_INDEX(0);
  int x1 = SHARED_MEMORY_INDEX(1);
  int x2 = SHARED_MEMORY_INDEX(2);
  int x3 = SHARED_MEMORY_INDEX(3);
  int x4 = SHARED_MEMORY_INDEX(4);
  int x5 = SHARED_MEMORY_INDEX(5);
  int x6 = SHARED_MEMORY_INDEX(6);
  int x7 = SHARED_MEMORY_INDEX(7);
  int x8 = SHARED_MEMORY_INDEX(8);
  int x9 = SHARED_MEMORY_INDEX(9);
  int x10 = SHARED_MEMORY_INDEX(10);
  int x11 = SHARED_MEMORY_INDEX(11);
  int x12 = SHARED_MEMORY_INDEX(12);
  int x13 = SHARED_MEMORY_INDEX(13);
  int x14 = SHARED_MEMORY_INDEX(14);
  int x15 = SHARED_MEMORY_INDEX(15);

#undef GET_CACHED_INDEX
#define GET_CACHED_INDEX(index) (x##index)


  for( int index = 0 ; index < (numberOfPasses) ; index++ ) {

    /* Move data to the thread's shared memory space */
    sharedMemory[GET_CACHED_INDEX(0)] = input[0 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(1)] = input[1 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(2)] = input[2 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(3)] = input[3 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(4)] = input[4 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(5)] = input[5 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(6)] = input[6 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(7)] = input[7 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(8)] = input[8 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(9)] = input[9 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(10)] = input[10 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(11)] = input[11 + 16 * index];
    sharedMemory[GET_CACHED_INDEX(12)] = input[12 + 16 * index];

    /* Testing the code with and without this if statement shows that
       it has no effect on performance. */
    if(index == numberOfPasses -1 ) {
      /* The last pass will contain the size of the chunk size (according to
	 official MD5 algorithm). */
      sharedMemory[GET_CACHED_INDEX(13)] = 0x00000080;
      sharedMemory[GET_CACHED_INDEX(14)] = chunkSize << 3;
      sharedMemory[GET_CACHED_INDEX(15)] = chunkSize >> 29;
    } else {
      sharedMemory[GET_CACHED_INDEX(13)] = input[13 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(14)] = input[14 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(15)] = input[15 + 16 * index];
    }

	   /* Get the little endian stuff done. */
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(0)], 
		   data, GET_CACHED_INDEX(0) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(1)], 
		   data, GET_CACHED_INDEX(1) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(2)], 
		   data, GET_CACHED_INDEX(2) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(3)], 
		   data, GET_CACHED_INDEX(3) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(4)], 
		   data, GET_CACHED_INDEX(4) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(5)], 
		   data, GET_CACHED_INDEX(5) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(6)], 
		   data, GET_CACHED_INDEX(6) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(7)], 
		   data, GET_CACHED_INDEX(7) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(8)], 
		   data, GET_CACHED_INDEX(8) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(9)], 
		   data, GET_CACHED_INDEX(9) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(10)], 
		   data, GET_CACHED_INDEX(10) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(11)], 
		   data, GET_CACHED_INDEX(11) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(12)], 
		   data, GET_CACHED_INDEX(12) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(13)], 
		   data, GET_CACHED_INDEX(13) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(14)], 
		   data, GET_CACHED_INDEX(14) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(15)], 
		   data, GET_CACHED_INDEX(15) * 4 );


    /* Start the MD5 permutations */
#undef S
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))
#undef P  
#define P(a,b,c,d,k,s,t) {						\
      a += F(b,c,d) + sharedMemory[GET_CACHED_INDEX(k)] + t; a = S(a,s) + b; \
    }									\
    
    A = state0;
    B = state1;
    C = state2;
    D = state3;
    
#undef F
    
#define F(x,y,z) (z ^ (x & (y ^ z)))
    
    P( A, B, C, D,  0,  7, 0xD76AA478 );
    P( D, A, B, C,  1, 12, 0xE8C7B756 );
    P( C, D, A, B,  2, 17, 0x242070DB );
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );
    P( A, B, C, D,  4,  7, 0xF57C0FAF );
    P( D, A, B, C,  5, 12, 0x4787C62A );
    P( C, D, A, B,  6, 17, 0xA8304613 );
    P( B, C, D, A,  7, 22, 0xFD469501 );
    P( A, B, C, D,  8,  7, 0x698098D8 );
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
    P( B, C, D, A, 11, 22, 0x895CD7BE );
    P( A, B, C, D, 12,  7, 0x6B901122 );
    P( D, A, B, C, 13, 12, 0xFD987193 );
    P( C, D, A, B, 14, 17, 0xA679438E );
    P( B, C, D, A, 15, 22, 0x49B40821 );
    
#undef F
    
#define F(x,y,z) (y ^ (z & (x ^ y)))
    
    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
    P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );
    
#undef F
    
#define F(x,y,z) (x ^ y ^ z)
    
    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );
    
#undef F
    
#define F(x,y,z) (y ^ (x | ~z))
    
    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );
    
#undef F
    
    state0 += A;
    state1 += B;
    state2 += C;
    state3 += D;
  }

  /* Got the hash, store it in the output buffer. */
  PUT_UINT32_LE( state0, output,  0 );
#ifndef FEATURE_REDUCED_HASH_SIZE
  PUT_UINT32_LE( state1, output,  4 );
  PUT_UINT32_LE( state2, output,  8 );
  PUT_UINT32_LE( state3, output, 12 );
#endif
  
}

__device__
static void md5_internal_overlap( unsigned int *input, unsigned int *sharedMemory, 
			  int chunkSize, unsigned char *output ) {

  /* Number of passes (512 bit blocks) we have to do */
  int numberOfPasses = chunkSize / 64 + 1;
  /* Used during the hashing process */
  unsigned long A, B, C, D;
  /* Needed to do the little endian stuff */
  unsigned char *data = (unsigned char *)sharedMemory;
  // number of padding bytes.
  int numPadBytes = 0;
  int numPadInt = 0;
  //int numPadRemain = 0;

  /* Will hold the hash value through the 
     intermediate stages of MD5 algorithm */
  unsigned int state0 = 0x67452301;
  unsigned int state1 = 0xEFCDAB89;
  unsigned int state2 = 0x98BADCFE;
  unsigned int state3 = 0x10325476;


  /* Used to cache the shared memory index calculations, but testing showed 
     that it has no performance effect. */
  int x0 = SHARED_MEMORY_INDEX(0);
  int x1 = SHARED_MEMORY_INDEX(1);
  int x2 = SHARED_MEMORY_INDEX(2);
  int x3 = SHARED_MEMORY_INDEX(3);
  int x4 = SHARED_MEMORY_INDEX(4);
  int x5 = SHARED_MEMORY_INDEX(5);
  int x6 = SHARED_MEMORY_INDEX(6);
  int x7 = SHARED_MEMORY_INDEX(7);
  int x8 = SHARED_MEMORY_INDEX(8);
  int x9 = SHARED_MEMORY_INDEX(9);
  int x10 = SHARED_MEMORY_INDEX(10);
  int x11 = SHARED_MEMORY_INDEX(11);
  int x12 = SHARED_MEMORY_INDEX(12);
  int x13 = SHARED_MEMORY_INDEX(13);
  int x14 = SHARED_MEMORY_INDEX(14);
  int x15 = SHARED_MEMORY_INDEX(15);

#undef GET_CACHED_INDEX
#define GET_CACHED_INDEX(index) (x##index)


  for( int index = 0 ; index < (numberOfPasses) ; index++ ) {
    
    if(index == numberOfPasses - 1 ) {
      
      numPadBytes = (64-12) - (chunkSize - (numberOfPasses-1)*64);
      numPadInt = numPadBytes/sizeof(int);
      /*numPadRemain = numPadBytes-numPadInt*sizeof(int);
      printf("\nLast loop chunkSize = %d, numberOfPasses= %d and \nnumPadBytes = %d, numPadInt =%d, numPadRemain = %d\n",
	     chunkSize,numberOfPasses,numPadBytes,numPadInt,numPadRemain);*/
      
      int i=0;
      for(i = 0 ; i < numPadInt ; i++){
	sharedMemory[SHARED_MEMORY_INDEX(13-i)] = 0;
      }

      int j=0;
      for(j=0;j<(16-3-numPadInt);j++){
	//printf("j= %d\n",j);
	sharedMemory[SHARED_MEMORY_INDEX(j)] = input[j + 16 * index];
      }
      
      
      /* The last pass will contain the size of the chunk size (according to
	 official MD5 algorithm). */
      sharedMemory[SHARED_MEMORY_INDEX(13-i)] = 0x00000080;
      //printf("the last one at %d\n",13-i);
      
      sharedMemory[GET_CACHED_INDEX(14)] = chunkSize << 3;
      sharedMemory[GET_CACHED_INDEX(15)] = chunkSize >> 29;
    } else {
      /* Move data to the thread's shared memory space */
      //printf("Not last loop\n");
      sharedMemory[GET_CACHED_INDEX(0)] = input[0 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(1)] = input[1 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(2)] = input[2 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(3)] = input[3 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(4)] = input[4 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(5)] = input[5 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(6)] = input[6 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(7)] = input[7 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(8)] = input[8 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(9)] = input[9 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(10)] = input[10 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(11)] = input[11 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(12)] = input[12 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(13)] = input[13 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(14)] = input[14 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(15)] = input[15 + 16 * index];
    }
    
    /* Get the little endian stuff done. */
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(0)], 
		   data, GET_CACHED_INDEX(0) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(1)], 
		   data, GET_CACHED_INDEX(1) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(2)], 
		   data, GET_CACHED_INDEX(2) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(3)], 
		   data, GET_CACHED_INDEX(3) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(4)], 
		   data, GET_CACHED_INDEX(4) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(5)], 
		   data, GET_CACHED_INDEX(5) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(6)], 
		   data, GET_CACHED_INDEX(6) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(7)], 
		   data, GET_CACHED_INDEX(7) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(8)], 
		   data, GET_CACHED_INDEX(8) * 4 );
    GET_UINT32_LE( sharedMemory[ GET_CACHED_INDEX(9)], 
		   data, GET_CACHED_INDEX(9) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(10)], 
		   data, GET_CACHED_INDEX(10) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(11)], 
		   data, GET_CACHED_INDEX(11) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(12)], 
		   data, GET_CACHED_INDEX(12) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(13)], 
		   data, GET_CACHED_INDEX(13) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(14)], 
		   data, GET_CACHED_INDEX(14) * 4 );
    GET_UINT32_LE( sharedMemory[GET_CACHED_INDEX(15)], 
		   data, GET_CACHED_INDEX(15) * 4 );


    /* Start the MD5 permutations */
#undef S
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))
#undef P  
#define P(a,b,c,d,k,s,t) {						\
      a += F(b,c,d) + sharedMemory[GET_CACHED_INDEX(k)] + t; a = S(a,s) + b; \
    }									\
    
    A = state0;
    B = state1;
    C = state2;
    D = state3;
    
#undef F
    
#define F(x,y,z) (z ^ (x & (y ^ z)))
    
    P( A, B, C, D,  0,  7, 0xD76AA478 );
    P( D, A, B, C,  1, 12, 0xE8C7B756 );
    P( C, D, A, B,  2, 17, 0x242070DB );
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );
    P( A, B, C, D,  4,  7, 0xF57C0FAF );
    P( D, A, B, C,  5, 12, 0x4787C62A );
    P( C, D, A, B,  6, 17, 0xA8304613 );
    P( B, C, D, A,  7, 22, 0xFD469501 );
    P( A, B, C, D,  8,  7, 0x698098D8 );
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
    P( B, C, D, A, 11, 22, 0x895CD7BE );
    P( A, B, C, D, 12,  7, 0x6B901122 );
    P( D, A, B, C, 13, 12, 0xFD987193 );
    P( C, D, A, B, 14, 17, 0xA679438E );
    P( B, C, D, A, 15, 22, 0x49B40821 );
    
#undef F
    
#define F(x,y,z) (y ^ (z & (x ^ y)))
    
    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
    P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );
    
#undef F
    
#define F(x,y,z) (x ^ y ^ z)
    
    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );
    
#undef F
    
#define F(x,y,z) (y ^ (x | ~z))
    
    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );
    
#undef F
    
    state0 += A;
    state1 += B;
    state2 += C;
    state3 += D;
  }

  /* Got the hash, store it in the output buffer. */
  PUT_UINT32_LE( state0, output,  0 );
#ifndef FEATURE_REDUCED_HASH_SIZE
  PUT_UINT32_LE( state1, output,  4 );
  PUT_UINT32_LE( state2, output,  8 );
  PUT_UINT32_LE( state3, output, 12 );
#endif
  
}
#endif

/*--------------------------------------------------------------------------
                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/

/*===========================================================================

FUNCTION MD5

DESCRIPTION
  Main md5 hash function

DEPENDENCIES
  GPU must be initialized

RETURN VALUE
  output: the hash result

===========================================================================*/
__global__
void md5( unsigned char *input, int chunkSize, int totalThreads,
          int padSize, unsigned char *scratch) {
  
  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * chunkSize;
  int hashIndex  = threadIndex * MD5_HASH_SIZE;

  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1)) && (padSize > 0)) {
    for(int i = 0 ; i < padSize ; i++)
      input[chunkIndex + chunkSize - padSize + i] = 0;
  }


#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
  // 512 words are allocated for every warp of 32 threads
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned int *inputIndex = (unsigned int *)(input + chunkIndex);
  
  md5_internal(inputIndex, sharedMemoryIndex, chunkSize, 
	       scratch + hashIndex );

#else
  md5_internal(input + chunkIndex, chunkSize, scratch + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */

}


__global__
void md5_overlap( unsigned char *input, int chunkSize, int offset,
		  int totalThreads, int padSize, unsigned char *output ) {

  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * offset;
  int hashIndex  = threadIndex * MD5_HASH_SIZE;


  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1))) {
    chunkSize-= padSize;
  }


#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned int *inputIndex = (unsigned int *)(input + chunkIndex);
  
  md5_internal_overlap(inputIndex, sharedMemoryIndex, chunkSize, 
	       output + hashIndex );

#else
  md5_internal(input + chunkIndex, chunkSize, output + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */


}

