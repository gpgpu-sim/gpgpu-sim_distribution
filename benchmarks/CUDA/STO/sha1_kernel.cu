/*==========================================================================
                                SHA1 KERNEL

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
    unsigned long state[5];     /*!< intermediate digest state  */	
    unsigned char buffer[64];   /*!< data block being processed */	
} sha1_context;

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
static const unsigned char sha1_padding[64] =
{
 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/*--------------------------------------------------------------------------
                                    MACROS
--------------------------------------------------------------------------*/

#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE 1
#endif


/*
 * 32-bit integer manipulation macros (big endian)
 */
#ifndef GET_UINT32_BE
#define GET_UINT32_BE(n,b,i)                            \
{                                                       \
    (n) = ( (unsigned long) (b)[(i)    ] << 24 )        \
        | ( (unsigned long) (b)[(i) + 1] << 16 )        \
        | ( (unsigned long) (b)[(i) + 2] <<  8 )        \
        | ( (unsigned long) (b)[(i) + 3]       );       \
}
#endif

#ifndef PUT_UINT32_BE
#define PUT_UINT32_BE(n,b,i)                            \
{                                                       \
    (b)[(i)    ] = (unsigned char) ( (n) >> 24 );       \
    (b)[(i) + 1] = (unsigned char) ( (n) >> 16 );       \
    (b)[(i) + 2] = (unsigned char) ( (n) >>  8 );       \
    (b)[(i) + 3] = (unsigned char) ( (n)       );       \
}
#endif

#ifdef FEATURE_SHARED_MEMORY
// current thread stride.
#undef SHARED_MEMORY_INDEX
#define SHARED_MEMORY_INDEX(index) (32 * (index) + (threadIdx.x & 0x1F))

#endif /* FEATURE_SHARED_MEMORY */




/*--------------------------------------------------------------------------
                                    LOCAL FUNCTIONS
--------------------------------------------------------------------------*/
#ifndef FEATURE_SHARED_MEMORY
/*
 * SHA-1 context setup
 */

/*===========================================================================

FUNCTION SHA1_GPU_STARTS

DESCRIPTION
  SHA-1 context setup

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
__device__
void sha1_starts( sha1_context *ctx ) {
  ctx->total[0] = 0;
  ctx->total[1] = 0;
  
  ctx->state[0] = 0x67452301;
  ctx->state[1] = 0xEFCDAB89;
  ctx->state[2] = 0x98BADCFE;
  ctx->state[3] = 0x10325476;
  ctx->state[4] = 0xC3D2E1F0;
}

/*===========================================================================

FUNCTION SHA1_GPU_PROCESS

DESCRIPTION
  SHA1 process buffer

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
__device__
void sha1_process( sha1_context *ctx, unsigned char data[64] ) {

  unsigned long temp, W[16], A, B, C, D, E;
  
  GET_UINT32_BE( W[ 0], data,  0 );
  GET_UINT32_BE( W[ 1], data,  4 );
  GET_UINT32_BE( W[ 2], data,  8 );
  GET_UINT32_BE( W[ 3], data, 12 );
  GET_UINT32_BE( W[ 4], data, 16 );
  GET_UINT32_BE( W[ 5], data, 20 );
  GET_UINT32_BE( W[ 6], data, 24 );
  GET_UINT32_BE( W[ 7], data, 28 );
  GET_UINT32_BE( W[ 8], data, 32 );
  GET_UINT32_BE( W[ 9], data, 36 );
  GET_UINT32_BE( W[10], data, 40 );
  GET_UINT32_BE( W[11], data, 44 );
  GET_UINT32_BE( W[12], data, 48 );
  GET_UINT32_BE( W[13], data, 52 );
  GET_UINT32_BE( W[14], data, 56 );
  GET_UINT32_BE( W[15], data, 60 );
  
#undef S
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))
  
#undef R
#define R(t)                                            \
(                                                       \
    temp = W[(t -  3) & 0x0F] ^ W[(t - 8) & 0x0F] ^     \
           W[(t - 14) & 0x0F] ^ W[ t      & 0x0F],      \
    ( W[t & 0x0F] = S(temp,1) )                         \
)

#undef P
#define P(a,b,c,d,e,x)                                  \
{                                                       \
    e += S(a,5) + F(b,c,d) + K + x; b = S(b,30);        \
}

  A = ctx->state[0];
  B = ctx->state[1];
  C = ctx->state[2];
  D = ctx->state[3];
  E = ctx->state[4];
  
#define F(x,y,z) (z ^ (x & (y ^ z)))
#define K 0x5A827999
  
  P( A, B, C, D, E, W[0]  );
  P( E, A, B, C, D, W[1]  );
  P( D, E, A, B, C, W[2]  );
  P( C, D, E, A, B, W[3]  );
  P( B, C, D, E, A, W[4]  );
  P( A, B, C, D, E, W[5]  );
  P( E, A, B, C, D, W[6]  );
  P( D, E, A, B, C, W[7]  );
  P( C, D, E, A, B, W[8]  );
  P( B, C, D, E, A, W[9]  );
  P( A, B, C, D, E, W[10] );
  P( E, A, B, C, D, W[11] );
  P( D, E, A, B, C, W[12] );
  P( C, D, E, A, B, W[13] );
  P( B, C, D, E, A, W[14] );
  P( A, B, C, D, E, W[15] );
  P( E, A, B, C, D, R(16) );
  P( D, E, A, B, C, R(17) );
  P( C, D, E, A, B, R(18) );
  P( B, C, D, E, A, R(19) );
  
#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0x6ED9EBA1
  
  P( A, B, C, D, E, R(20) );
  P( E, A, B, C, D, R(21) );
  P( D, E, A, B, C, R(22) );
  P( C, D, E, A, B, R(23) );
  P( B, C, D, E, A, R(24) );
  P( A, B, C, D, E, R(25) );
  P( E, A, B, C, D, R(26) );
  P( D, E, A, B, C, R(27) );
  P( C, D, E, A, B, R(28) );
  P( B, C, D, E, A, R(29) );
  P( A, B, C, D, E, R(30) );
  P( E, A, B, C, D, R(31) );
  P( D, E, A, B, C, R(32) );
  P( C, D, E, A, B, R(33) );
  P( B, C, D, E, A, R(34) );
  P( A, B, C, D, E, R(35) );
  P( E, A, B, C, D, R(36) );
  P( D, E, A, B, C, R(37) );
  P( C, D, E, A, B, R(38) );
  P( B, C, D, E, A, R(39) );
  
#undef K
#undef F
  
#define F(x,y,z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC
  
  P( A, B, C, D, E, R(40) );
  P( E, A, B, C, D, R(41) );
  P( D, E, A, B, C, R(42) );
  P( C, D, E, A, B, R(43) );
  P( B, C, D, E, A, R(44) );
  P( A, B, C, D, E, R(45) );
  P( E, A, B, C, D, R(46) );
  P( D, E, A, B, C, R(47) );
  P( C, D, E, A, B, R(48) );
  P( B, C, D, E, A, R(49) );
  P( A, B, C, D, E, R(50) );
  P( E, A, B, C, D, R(51) );
  P( D, E, A, B, C, R(52) );
  P( C, D, E, A, B, R(53) );
  P( B, C, D, E, A, R(54) );
  P( A, B, C, D, E, R(55) );
  P( E, A, B, C, D, R(56) );
  P( D, E, A, B, C, R(57) );
  P( C, D, E, A, B, R(58) );
  P( B, C, D, E, A, R(59) );
  
#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0xCA62C1D6
  
  P( A, B, C, D, E, R(60) );
  P( E, A, B, C, D, R(61) );
  P( D, E, A, B, C, R(62) );
  P( C, D, E, A, B, R(63) );
  P( B, C, D, E, A, R(64) );
  P( A, B, C, D, E, R(65) );
  P( E, A, B, C, D, R(66) );
  P( D, E, A, B, C, R(67) );
  P( C, D, E, A, B, R(68) );
  P( B, C, D, E, A, R(69) );
  P( A, B, C, D, E, R(70) );
  P( E, A, B, C, D, R(71) );
  P( D, E, A, B, C, R(72) );
  P( C, D, E, A, B, R(73) );
  P( B, C, D, E, A, R(74) );
  P( A, B, C, D, E, R(75) );
  P( E, A, B, C, D, R(76) );
  P( D, E, A, B, C, R(77) );
  P( C, D, E, A, B, R(78) );
  P( B, C, D, E, A, R(79) );
  
#undef K
#undef F
  
  ctx->state[0] += A;
  ctx->state[1] += B;
  ctx->state[2] += C;
  ctx->state[3] += D;
  ctx->state[4] += E;
}

/*===========================================================================

FUNCTION SHA1_CPU_UPDATE

DESCRIPTION
  SHA1 update buffer

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
__device__
void sha1_update( sha1_context *ctx, unsigned char *input, int ilen ) {
  int fill;
  unsigned long left;
  
  if( ilen <= 0 )
    return;
  
  left = ctx->total[0] & 0x3F;
  fill = 64 - left;
  
  ctx->total[0] += ilen;
  ctx->total[0] &= 0xFFFFFFFF;
  
  if ( ctx->total[0] < (unsigned long) ilen )
    ctx->total[1]++;
  
  if ( left && ilen >= fill ) {
    /*memcpy( (void *) (ctx->buffer + left),
      (void *) input, fill );*/
    for (int i = 0; i < fill; i++) {
      ctx->buffer[i+left] = input[i];
    }
    
    
    sha1_process( ctx, ctx->buffer );
    input += fill;
    ilen  -= fill;
    left = 0;
  }

  while ( ilen >= 64 ) {
    sha1_process( ctx, input );
    input += 64;
    ilen  -= 64;
  }
  
  if ( ilen > 0 ) {
    /*memcpy( (void *) (ctx->buffer + left),
      (void *) input, ilen );*/
    for (int i = 0; i < ilen; i++) {
      ctx->buffer[i+left] = input[i];
    }
    
  }
}


/*===========================================================================

FUNCTION SHA1_CPU_FINISH

DESCRIPTION
  SHA1 final digest

DEPENDENCIES
  None

RETURN VALUE
  None

===========================================================================*/
__device__
void sha1_finish( sha1_context *ctx, unsigned char *output ) {
  unsigned long last, padn;
  unsigned long high, low;
  unsigned char msglen[8];
  
  high = ( ctx->total[0] >> 29 )
    | ( ctx->total[1] <<  3 );
  low  = ( ctx->total[0] <<  3 );
  
  PUT_UINT32_BE( high, msglen, 0 );
  PUT_UINT32_BE( low,  msglen, 4 );
  
  last = ctx->total[0] & 0x3F;
  padn = ( last < 56 ) ? ( 56 - last ) : ( 120 - last );
  
  sha1_update( ctx, (unsigned char *) sha1_padding, padn );
  sha1_update( ctx, msglen, 8 );
  
  PUT_UINT32_BE( ctx->state[0], output,  0 );
#ifndef FEATURE_REDUCED_HASH_SIZE
  PUT_UINT32_BE( ctx->state[1], output,  4 );
  PUT_UINT32_BE( ctx->state[2], output,  8 );
  PUT_UINT32_BE( ctx->state[3], output, 12 );
  PUT_UINT32_BE( ctx->state[4], output, 16 );
#endif
}

/*===========================================================================

FUNCTION SHA1_INTERNAL

DESCRIPTION
  Does the real sha1 algorithm

DEPENDENCIES
  None

RETURN VALUE
  output is the hash result

===========================================================================*/
__device__
void sha1_internal( unsigned char *input, int ilen,
		    unsigned char *output ) {
  sha1_context ctx;
  
  sha1_starts( &ctx );
  sha1_update( &ctx, input, ilen );
  sha1_finish( &ctx, output );
  
  memset( &ctx, 0, sizeof( sha1_context ) );
}

#endif

#ifdef FEATURE_SHARED_MEMORY
/*===========================================================================

FUNCTION SHA1_INTERNAL

DESCRIPTION
  Does the real sha1 algorithm.

DEPENDENCIES
  None

RETURN VALUE
  output is the hash result

===========================================================================*/

__device__
unsigned long macroRFunction(int t, unsigned int *sharedMemory) {
	return sharedMemory[SHARED_MEMORY_INDEX((t -  3) & 0x0F)] ^ sharedMemory[SHARED_MEMORY_INDEX((t - 8) & 0x0F)] ^
           sharedMemory[SHARED_MEMORY_INDEX((t - 14) & 0x0F)] ^ sharedMemory[SHARED_MEMORY_INDEX( t      & 0x0F)];
}


__device__
static void sha1_internal( unsigned int *input, unsigned int *sharedMemory, 
			  unsigned int chunkSize, unsigned char *output ) {

  /* Number of passes (512 bit blocks) we have to do */
  int numberOfPasses = chunkSize / 64 + 1;
  /* Used during the hashing process */
  unsigned long temp, A, B, C, D ,E;
  //unsigned long shared14, shared15;
  /* Needed to do the little endian stuff */
  unsigned char *data = (unsigned char *)sharedMemory;

  /* Will hold the hash value through the 
     intermediate stages of SHA1 algorithm */
  unsigned int state0 = 0x67452301;
  unsigned int state1 = 0xEFCDAB89;
  unsigned int state2 = 0x98BADCFE;
  unsigned int state3 = 0x10325476;
  unsigned int state4 = 0xC3D2E1F0;

  
/*  int x0 = SHARED_MEMORY_INDEX(0);
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
*/
#undef GET_CACHED_INDEX
#define GET_CACHED_INDEX(index) SHARED_MEMORY_INDEX(index)//(x##index)


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
	 official SHA1 algorithm). */
      sharedMemory[GET_CACHED_INDEX(13)] = 0x00000080;

	  PUT_UINT32_BE( chunkSize >> 29, 
		   data, GET_CACHED_INDEX(14) * 4 );
      PUT_UINT32_BE( chunkSize << 3, 
		   data, GET_CACHED_INDEX(15) * 4 );

    }
    else {
      sharedMemory[GET_CACHED_INDEX(13)] = input[13 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(14)] = input[14 + 16 * index];
      sharedMemory[GET_CACHED_INDEX(15)] = input[15 + 16 * index];
    }

	/* Get the little endian stuff done. */
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(0)], 
		   data, GET_CACHED_INDEX(0) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(1)], 
		   data, GET_CACHED_INDEX(1) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(2)], 
		   data, GET_CACHED_INDEX(2) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(3)], 
		   data, GET_CACHED_INDEX(3) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(4)], 
		   data, GET_CACHED_INDEX(4) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(5)], 
		   data, GET_CACHED_INDEX(5) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(6)], 
		   data, GET_CACHED_INDEX(6) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(7)], 
		   data, GET_CACHED_INDEX(7) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(8)], 
		   data, GET_CACHED_INDEX(8) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(9)], 
		   data, GET_CACHED_INDEX(9) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(10)], 
		   data, GET_CACHED_INDEX(10) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(11)], 
		   data, GET_CACHED_INDEX(11) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(12)], 
		   data, GET_CACHED_INDEX(12) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(13)], 
		   data, GET_CACHED_INDEX(13) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(14)], 
		   data, GET_CACHED_INDEX(14) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(15)], 
		   data, GET_CACHED_INDEX(15) * 4 );


#undef S
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))


#undef R
#define R(t)                                            \
(                                                       \
    temp = macroRFunction(t, sharedMemory) ,      \
    ( sharedMemory[SHARED_MEMORY_INDEX(t & 0x0F)] = S(temp,1) )       \
)

/*
#define R(t)                                            \
(                                                       \
    temp = sharedMemory[SHARED_MEMORY_INDEX((t -  3) & 0x0F)] ^ sharedMemory[SHARED_MEMORY_INDEX((t - 8) & 0x0F)] ^     \
           sharedMemory[SHARED_MEMORY_INDEX((t - 14) & 0x0F)] ^ sharedMemory[SHARED_MEMORY_INDEX( t      & 0x0F)],      \
    ( sharedMemory[SHARED_MEMORY_INDEX(t & 0x0F)] = S(temp,1) )                         \
)
*/

#undef P
#define P(a,b,c,d,e,x)                                  \
{                                                       \
    e += S(a,5) + F(b,c,d) + K + x; b = S(b,30);        \
}

    A = state0;
    B = state1;
    C = state2;
    D = state3;
    E = state4;


#define F(x,y,z) (z ^ (x & (y ^ z)))
#define K 0x5A827999

    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(0)]  );
    P( E, A, B, C, D, sharedMemory[ GET_CACHED_INDEX(1)]  );
    P( D, E, A, B, C, sharedMemory[ GET_CACHED_INDEX(2)]  );
    P( C, D, E, A, B, sharedMemory[ GET_CACHED_INDEX(3)]  );
    P( B, C, D, E, A, sharedMemory[ GET_CACHED_INDEX(4)]  );
    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(5)]  );
    P( E, A, B, C, D, sharedMemory[ GET_CACHED_INDEX(6)]  );
    P( D, E, A, B, C, sharedMemory[ GET_CACHED_INDEX(7)]  );
    P( C, D, E, A, B, sharedMemory[ GET_CACHED_INDEX(8)]  );
    P( B, C, D, E, A, sharedMemory[ GET_CACHED_INDEX(9)]  );
    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(10)] );
    P( E, A, B, C, D, sharedMemory[ GET_CACHED_INDEX(11)] );
    P( D, E, A, B, C, sharedMemory[ GET_CACHED_INDEX(12)] );
    P( C, D, E, A, B, sharedMemory[ GET_CACHED_INDEX(13)] );
    P( B, C, D, E, A, sharedMemory[ GET_CACHED_INDEX(14)] );
    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(15)] );
    P( E, A, B, C, D, R(16) );
    P( D, E, A, B, C, R(17) );
    P( C, D, E, A, B, R(18) );
    P( B, C, D, E, A, R(19) );


#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0x6ED9EBA1

    P( A, B, C, D, E, R(20) );
    P( E, A, B, C, D, R(21) );
    P( D, E, A, B, C, R(22) );
    P( C, D, E, A, B, R(23) );
    P( B, C, D, E, A, R(24) );
    P( A, B, C, D, E, R(25) );
    P( E, A, B, C, D, R(26) );
    P( D, E, A, B, C, R(27) );
    P( C, D, E, A, B, R(28) );
    P( B, C, D, E, A, R(29) );
    P( A, B, C, D, E, R(30) );
    P( E, A, B, C, D, R(31) );
    P( D, E, A, B, C, R(32) );
    P( C, D, E, A, B, R(33) );
    P( B, C, D, E, A, R(34) );
    P( A, B, C, D, E, R(35) );
    P( E, A, B, C, D, R(36) );
    P( D, E, A, B, C, R(37) );
    P( C, D, E, A, B, R(38) );
    P( B, C, D, E, A, R(39) );

#undef K
#undef F

#define F(x,y,z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC

    P( A, B, C, D, E, R(40) );
    P( E, A, B, C, D, R(41) );
    P( D, E, A, B, C, R(42) );
    P( C, D, E, A, B, R(43) );
    P( B, C, D, E, A, R(44) );
    P( A, B, C, D, E, R(45) );
    P( E, A, B, C, D, R(46) );
    P( D, E, A, B, C, R(47) );
    P( C, D, E, A, B, R(48) );
    P( B, C, D, E, A, R(49) );
    P( A, B, C, D, E, R(50) );
    P( E, A, B, C, D, R(51) );
    P( D, E, A, B, C, R(52) );
    P( C, D, E, A, B, R(53) );
    P( B, C, D, E, A, R(54) );
    P( A, B, C, D, E, R(55) );
    P( E, A, B, C, D, R(56) );
    P( D, E, A, B, C, R(57) );
    P( C, D, E, A, B, R(58) );
    P( B, C, D, E, A, R(59) );

#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0xCA62C1D6

    P( A, B, C, D, E, R(60) );
    P( E, A, B, C, D, R(61) );
    P( D, E, A, B, C, R(62) );
    P( C, D, E, A, B, R(63) );
    P( B, C, D, E, A, R(64) );
    P( A, B, C, D, E, R(65) );
    P( E, A, B, C, D, R(66) );
    P( D, E, A, B, C, R(67) );
    P( C, D, E, A, B, R(68) );
    P( B, C, D, E, A, R(69) );
    P( A, B, C, D, E, R(70) );
    P( E, A, B, C, D, R(71) );
    P( D, E, A, B, C, R(72) );
    P( C, D, E, A, B, R(73) );
    P( B, C, D, E, A, R(74) );
    P( A, B, C, D, E, R(75) );
    P( E, A, B, C, D, R(76) );
    P( D, E, A, B, C, R(77) );
    P( C, D, E, A, B, R(78) );
    P( B, C, D, E, A, R(79) );

#undef K
#undef F

    state0 += A;
    state1 += B;
    state2 += C;
    state3 += D;
    state4 += E;
	}

	/* Got the hash, store it in the output buffer. */
	PUT_UINT32_BE( state0, output,  0 );
#ifndef FEATURE_REDUCED_HASH_SIZE
	PUT_UINT32_BE( state1, output,  4 );
	PUT_UINT32_BE( state2, output,  8 );
	PUT_UINT32_BE( state3, output, 12 );
	PUT_UINT32_BE( state4, output, 16 );
#endif

}

__device__
static void sha1_internal_overlap( unsigned int *input, unsigned int *sharedMemory, 
			  unsigned int chunkSize, unsigned char *output ) {

  /* Number of passes (512 bit blocks) we have to do */
  int numberOfPasses = chunkSize / 64 + 1;
  /* Used during the hashing process */
  unsigned long temp, A, B, C, D ,E;
  //unsigned long shared14, shared15;
  /* Needed to do the big endian stuff */
  unsigned char *data = (unsigned char *)sharedMemory;
  // number of padding bytes.
  int numPadBytes = 0;
  int numPadInt = 0;
  //int numPadRemain = 0;

  /* Will hold the hash value through the 
     intermediate stages of SHA1 algorithm */
  unsigned int state0 = 0x67452301;
  unsigned int state1 = 0xEFCDAB89;
  unsigned int state2 = 0x98BADCFE;
  unsigned int state3 = 0x10325476;
  unsigned int state4 = 0xC3D2E1F0;

  
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
    
    if(index == numberOfPasses -1 ){
      
      numPadBytes = (64-12) - (chunkSize - (numberOfPasses-1)*64);
      numPadInt = numPadBytes/sizeof(int);
      /*numPadRemain = numPadBytes-numPadInt*sizeof(int);
      printf("\nLast loop chunkSize = %d, numberOfPasses= %d and \nnumPadBytes = %d, numPadInt =%d, numPadRemain = %d\n",
	chunkSize,numberOfPasses,numPadBytes,numPadInt,numPadRemain);*/
      
      int i=0;
      for(i=0;i<numPadInt;i++){
	sharedMemory[SHARED_MEMORY_INDEX(13-i)] = 0;
      }
      int j=0;
      for(j=0;j<(16-3-numPadInt);j++){
	//printf("j= %d\n",j);
	sharedMemory[SHARED_MEMORY_INDEX(j)] = input[j + 16 * index];
      }
      
      
      /* The last pass will contain the size of the chunk size (according to
	 official SHA1 algorithm). */
      sharedMemory[SHARED_MEMORY_INDEX(13-i)] = 0x00000080;
      //printf("the last one at %d\n",13-i);
      
      PUT_UINT32_BE( chunkSize >> 29, 
		     data, GET_CACHED_INDEX(14) * 4 );
      PUT_UINT32_BE( chunkSize << 3, 
			   data, GET_CACHED_INDEX(15) * 4 );
    }
    else{
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
    
    /*	  int k=0;
	  printf("\nGPU DATA\n");
	  for(k=0;k<16;k++){
	  printf("%d\t",sharedMemory[SHARED_MEMORY_INDEX(k)]);
	  }
	  printf("\n\n");*/
    
    /* Get the little endian stuff done. */
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(0)], 
		   data, GET_CACHED_INDEX(0) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(1)], 
		   data, GET_CACHED_INDEX(1) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(2)], 
		   data, GET_CACHED_INDEX(2) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(3)], 
		   data, GET_CACHED_INDEX(3) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(4)], 
		   data, GET_CACHED_INDEX(4) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(5)], 
		   data, GET_CACHED_INDEX(5) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(6)], 
		   data, GET_CACHED_INDEX(6) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(7)], 
		   data, GET_CACHED_INDEX(7) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(8)], 
		   data, GET_CACHED_INDEX(8) * 4 );
    GET_UINT32_BE( sharedMemory[ GET_CACHED_INDEX(9)], 
		   data, GET_CACHED_INDEX(9) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(10)], 
		   data, GET_CACHED_INDEX(10) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(11)], 
		   data, GET_CACHED_INDEX(11) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(12)], 
		   data, GET_CACHED_INDEX(12) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(13)], 
		   data, GET_CACHED_INDEX(13) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(14)], 
		   data, GET_CACHED_INDEX(14) * 4 );
    GET_UINT32_BE( sharedMemory[GET_CACHED_INDEX(15)], 
		   data, GET_CACHED_INDEX(15) * 4 );

#undef S
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))


#undef R
#define R(t)                                            \
(                                                       \
    temp = macroRFunction(t, sharedMemory) ,      \
    ( sharedMemory[SHARED_MEMORY_INDEX(t & 0x0F)] = S(temp,1) )       \
)

/*
#define R(t)                                            \
(                                                       \
    temp = sharedMemory[SHARED_MEMORY_INDEX((t -  3) & 0x0F)] ^ sharedMemory[SHARED_MEMORY_INDEX((t - 8) & 0x0F)] ^     \
           sharedMemory[SHARED_MEMORY_INDEX((t - 14) & 0x0F)] ^ sharedMemory[SHARED_MEMORY_INDEX( t      & 0x0F)],      \
    ( sharedMemory[SHARED_MEMORY_INDEX(t & 0x0F)] = S(temp,1) )                         \
)
*/

#undef P
#define P(a,b,c,d,e,x)                                  \
{                                                       \
    e += S(a,5) + F(b,c,d) + K + x; b = S(b,30);        \
}

    A = state0;
    B = state1;
    C = state2;
    D = state3;
    E = state4;


#define F(x,y,z) (z ^ (x & (y ^ z)))
#define K 0x5A827999

    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(0)]  );
    P( E, A, B, C, D, sharedMemory[ GET_CACHED_INDEX(1)]  );
    P( D, E, A, B, C, sharedMemory[ GET_CACHED_INDEX(2)]  );
    P( C, D, E, A, B, sharedMemory[ GET_CACHED_INDEX(3)]  );
    P( B, C, D, E, A, sharedMemory[ GET_CACHED_INDEX(4)]  );
    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(5)]  );
    P( E, A, B, C, D, sharedMemory[ GET_CACHED_INDEX(6)]  );
    P( D, E, A, B, C, sharedMemory[ GET_CACHED_INDEX(7)]  );
    P( C, D, E, A, B, sharedMemory[ GET_CACHED_INDEX(8)]  );
    P( B, C, D, E, A, sharedMemory[ GET_CACHED_INDEX(9)]  );
    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(10)] );
    P( E, A, B, C, D, sharedMemory[ GET_CACHED_INDEX(11)] );
    P( D, E, A, B, C, sharedMemory[ GET_CACHED_INDEX(12)] );
    P( C, D, E, A, B, sharedMemory[ GET_CACHED_INDEX(13)] );
    P( B, C, D, E, A, sharedMemory[ GET_CACHED_INDEX(14)] );
    P( A, B, C, D, E, sharedMemory[ GET_CACHED_INDEX(15)] );
    P( E, A, B, C, D, R(16) );
    P( D, E, A, B, C, R(17) );
    P( C, D, E, A, B, R(18) );
    P( B, C, D, E, A, R(19) );


#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0x6ED9EBA1

    P( A, B, C, D, E, R(20) );
    P( E, A, B, C, D, R(21) );
    P( D, E, A, B, C, R(22) );
    P( C, D, E, A, B, R(23) );
    P( B, C, D, E, A, R(24) );
    P( A, B, C, D, E, R(25) );
    P( E, A, B, C, D, R(26) );
    P( D, E, A, B, C, R(27) );
    P( C, D, E, A, B, R(28) );
    P( B, C, D, E, A, R(29) );
    P( A, B, C, D, E, R(30) );
    P( E, A, B, C, D, R(31) );
    P( D, E, A, B, C, R(32) );
    P( C, D, E, A, B, R(33) );
    P( B, C, D, E, A, R(34) );
    P( A, B, C, D, E, R(35) );
    P( E, A, B, C, D, R(36) );
    P( D, E, A, B, C, R(37) );
    P( C, D, E, A, B, R(38) );
    P( B, C, D, E, A, R(39) );

#undef K
#undef F

#define F(x,y,z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC

    P( A, B, C, D, E, R(40) );
    P( E, A, B, C, D, R(41) );
    P( D, E, A, B, C, R(42) );
    P( C, D, E, A, B, R(43) );
    P( B, C, D, E, A, R(44) );
    P( A, B, C, D, E, R(45) );
    P( E, A, B, C, D, R(46) );
    P( D, E, A, B, C, R(47) );
    P( C, D, E, A, B, R(48) );
    P( B, C, D, E, A, R(49) );
    P( A, B, C, D, E, R(50) );
    P( E, A, B, C, D, R(51) );
    P( D, E, A, B, C, R(52) );
    P( C, D, E, A, B, R(53) );
    P( B, C, D, E, A, R(54) );
    P( A, B, C, D, E, R(55) );
    P( E, A, B, C, D, R(56) );
    P( D, E, A, B, C, R(57) );
    P( C, D, E, A, B, R(58) );
    P( B, C, D, E, A, R(59) );

#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0xCA62C1D6

    P( A, B, C, D, E, R(60) );
    P( E, A, B, C, D, R(61) );
    P( D, E, A, B, C, R(62) );
    P( C, D, E, A, B, R(63) );
    P( B, C, D, E, A, R(64) );
    P( A, B, C, D, E, R(65) );
    P( E, A, B, C, D, R(66) );
    P( D, E, A, B, C, R(67) );
    P( C, D, E, A, B, R(68) );
    P( B, C, D, E, A, R(69) );
    P( A, B, C, D, E, R(70) );
    P( E, A, B, C, D, R(71) );
    P( D, E, A, B, C, R(72) );
    P( C, D, E, A, B, R(73) );
    P( B, C, D, E, A, R(74) );
    P( A, B, C, D, E, R(75) );
    P( E, A, B, C, D, R(76) );
    P( D, E, A, B, C, R(77) );
    P( C, D, E, A, B, R(78) );
    P( B, C, D, E, A, R(79) );

#undef K
#undef F

    state0 += A;
    state1 += B;
    state2 += C;
    state3 += D;
    state4 += E;
	}

	/* Got the hash, store it in the output buffer. */
	PUT_UINT32_BE( state0, output,  0 );
#ifndef FEATURE_REDUCED_HASH_SIZE
	PUT_UINT32_BE( state1, output,  4 );
	PUT_UINT32_BE( state2, output,  8 );
	PUT_UINT32_BE( state3, output, 12 );
	PUT_UINT32_BE( state4, output, 16 );
#endif

}
#endif

/*--------------------------------------------------------------------------

                                    GLOBAL FUNCTIONS
--------------------------------------------------------------------------*/
/*===========================================================================

FUNCTION SHA1

DESCRIPTION
  Main sha1 hash function

DEPENDENCIES
  GPU must be initialized

RETURN VALUE
  output: the hash result

===========================================================================*/
__global__
void sha1( unsigned char *input, int chunkSize, int totalThreads,
	   int padSize, unsigned char *scratch ) {
  
  // get the current thread index
  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * chunkSize;
  int hashIndex  = threadIndex * SHA1_HASH_SIZE;

  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1)) && (padSize > 0)) {
    for(int i = 0 ; i < padSize ; i++)
      input[chunkIndex + chunkSize - padSize + i] = 0;	
  }
  
#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned char *tempInput = input + chunkIndex;
  unsigned int *inputIndex = (unsigned int *)(tempInput);
  
  sha1_internal(inputIndex, sharedMemoryIndex, chunkSize, 
	       scratch + hashIndex );

#else
  sha1_internal(input + chunkIndex, chunkSize, scratch + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */

}

__global__
void sha1_overlap( unsigned char *input, int chunkSize, int offset,
		   int totalThreads, int padSize, unsigned char *output ) {

  int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int chunkIndex = threadIndex * offset;
  int hashIndex  = threadIndex * SHA1_HASH_SIZE;

  if(threadIndex >= totalThreads)
    return;
  
  if ((threadIndex == (totalThreads - 1))) {
    chunkSize-= padSize;
  }

#ifdef FEATURE_SHARED_MEMORY
  
  __shared__ unsigned int sharedMemory[4 * 1024 - 32];
  
    //NOTE : SAMER : this can exceed the size of the shared memory 
  unsigned int *sharedMemoryIndex = sharedMemory + ((threadIdx.x >> 5) * 512);
  unsigned int *inputIndex = (unsigned int *)(input + chunkIndex);
  
  sha1_internal_overlap(inputIndex, sharedMemoryIndex, chunkSize, 
	       output + hashIndex );

#else
  sha1_internal(input + chunkIndex, chunkSize, output + hashIndex );
#endif /* FEATURE_SHARED_MEMORY */


}
