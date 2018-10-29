// This file created from vector_types.h distributed with CUDA 1.1
// (see original copyright notice below)
// 
// Changes Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef CUDA_MATH
#define CUDA_MATH

// cuda math implementations
#undef max
#undef min
namespace cuda_math {
#define __attribute__(a) // to remove warnings inside math_functions.h
#undef INT_MAX

#if CUDART_VERSION < 3000
// DEVICE_BUILTIN
   struct int4 {
      int x, y, z, w;
   };
   struct uint4 {
      unsigned int x, y, z, w;
   };
   struct float4 {
      float x, y, z, w;
   };
   struct float2 {
      float x, y;
   };
    

// DEVICE_BUILTIN
   typedef struct int4 int4;
   typedef struct uint4 uint4;
   typedef struct float4 float4;
   typedef struct float2 float2;

extern float rsqrtf(float); // CUDA 2.3 beta

#define CUDA_FLOAT_MATH_FUNCTIONS
#include <device_types.h>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__
#undef __attribute__

// float to integer conversion 
int float2int(float a, enum cudaRoundMode mode)
{
   return __internal_float2uint(a, mode); 
}

// float to unsigned integer conversion 
unsigned int float2uint(float a, enum cudaRoundMode mode)
{
   return __internal_float2uint(a, mode); 
}

float __ll2float_rz(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TOWARDZERO); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ll2float_ru(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_UPWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ll2float_rd(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_DOWNWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}

#else

#define CUDA_FLOAT_MATH_FUNCTIONS
#define __CUDACC__

// implementing int to float intrinsics with different rounding modes 
#include <device_types.h>
#include <fenv.h>

// 32-bit integer to float
float __int2float_rn(int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TONEAREST); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __int2float_rz(int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TOWARDZERO); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __int2float_ru(int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_UPWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __int2float_rd(int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_DOWNWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}

// 32-bit unsigned integer to float
float __uint2float_rn(unsigned int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TONEAREST); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __uint2float_rz(unsigned int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TOWARDZERO); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __uint2float_ru(unsigned int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_UPWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __uint2float_rd(unsigned int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_DOWNWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}

// 64-bit integer to float
float __ll2float_rn(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TONEAREST); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ll2float_rz(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TOWARDZERO); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ll2float_ru(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_UPWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ll2float_rd(long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_DOWNWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}

// 64-bit unsigned integer to float 
float __ull2float_rn(unsigned long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TONEAREST); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ull2float_rz(unsigned long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_TOWARDZERO); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ull2float_ru(unsigned long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_UPWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}
float __ull2float_rd(unsigned long long int a) {
   int orig_rnd_mode = fegetround();
   fesetround(FE_DOWNWARD); 
   float b = a;
   fesetround(orig_rnd_mode); 
   return b;
}

// float to integer conversion 
int float2int(float a, enum cudaRoundMode mode)
{
   int tmp;
   switch (mode) {
   case cuda_math::cudaRoundZero: tmp = truncf(a);     break;
   case cuda_math::cudaRoundNearest: tmp = nearbyintf(a); break;
   case cuda_math::cudaRoundMinInf: tmp = floorf(a);     break;
   case cuda_math::cudaRoundPosInf: tmp = ceilf(a);      break;
   default: abort();
   }
   return tmp; 
}

int __internal_float2int(float a, enum cudaRoundMode mode) 
{
   return float2int(a, mode); 
}

// float to unsigned integer conversion 
unsigned int float2uint(float a, enum cudaRoundMode mode)
{
   unsigned int tmp;
   switch (mode) {
   case cuda_math::cudaRoundZero: tmp = truncf(a);     break;
   case cuda_math::cudaRoundNearest: tmp = nearbyintf(a); break;
   case cuda_math::cudaRoundMinInf: tmp = floorf(a);     break;
   case cuda_math::cudaRoundPosInf: tmp = ceilf(a);      break;
   default: abort();
   }
   return tmp; 
}

unsigned int __internal_float2uint(float a, enum cudaRoundMode mode) 
{
   return float2uint(a, mode); 
}

// intrinsic for division 
float fdividef(float a, float b)
{
   return (a / b); 
}

float __internal_accurate_fdividef(float a, float b)
{
   return fdividef(a, b); 
}

// intrinsic for saturate  (clamp values beyond 0 and 1)
float __saturatef(float a)
{
   float b; 
   if (isnan(a)) b = 0.0f; 
   else if (a >= 1.0f) b = 1.0f;
   else if (a <= 0.0f) b = 0.0f; 
   else b = a; 
   return b; 
}

// intrinsic for power 
float __powf(float a, float b)
{
   return powf(a, b);
}

// math functions missing in Mac OSX GCC
#ifdef __APPLE__
int __signbitd(double d)
{
   unsigned long long int u = *((unsigned long long int*)&d); 
   return ((u & 0x8000000000000000ULL) != 0);
}
#endif 

#undef __CUDACC__
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__
#undef __attribute__

#endif

}

// math functions missing in Mac OSX GCC
#ifdef __APPLE__
int isnanf(float a) 
{
   return (isnan(a)); 
}
#endif 

#endif
