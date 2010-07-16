#ifndef PREPASS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas.h"
#endif

__device__ float4 max4 ( const float4 a , const float4 b )
{
    float4 c  ;
    c.x = (a.x>b.x)?a.x:b.x;
    c.y = (a.y>b.y)?a.y:b.y;
    c.z = (a.z>b.z)?a.z:b.z;
    c.w = (a.w>b.w)?a.w:b.w;
    return(c) ;
}
__device__ float4 min4 ( const float4 a , const float4 b )
{
    float4 c  ;
    c.x = (a.x<b.x)?a.x:b.x;
    c.y = (a.y<b.y)?a.y:b.y;
    c.z = (a.z<b.z)?a.z:b.z;
    c.w = (a.w<b.w)?a.w:b.w;
    return(c) ;
}

__device__ float4 log4 ( const float4 a )
{
    float4 c  ;
    c.x = log(a.x) ; c.y = log(a.y) ; c.z = log(a.z) ; c.w = log(a.w) ;
    return(c) ;
}

__device__ float4 exp4 ( const float4 a )
{
    float4 c  ;
    c.x = exp(a.x) ; c.y = exp(a.y) ; c.z = exp(a.z) ; c.w = exp(a.w) ;
    return(c) ;
}

__device__ float4 sqrt4 ( const float4 a )
{
    float4 c  ;
    c.x = sqrt(a.x) ; c.y = sqrt(a.y) ; c.z = sqrt(a.z) ; c.w = sqrt(a.w) ;
    return(c) ;
}
