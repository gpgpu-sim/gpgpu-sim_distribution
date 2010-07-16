/*
 * Copyright 2008 BOROUJERDI Maxime. Tous droits reserves.
 */

//#define FIXED_CONST_PARSE
#ifndef __RAYTRACING_KERNEL_H__
#define __RAYTRACING_KERNEL_H__

#include "cutil_math.h"

typedef struct
{
    float4 m[3];
} matrice3x4;

typedef struct {
    float4 m[4];
} matrice4x4;

typedef struct{
	float3 A;	// origine
	float3 u;	// direction
} Rayon;

typedef struct Sphere{
	float3 C;	    // centre
	float  r;	    // rayon
	float  R,V,B,A;
	/*Sphere() : C(make_float3(0.0f,0.0f,0.0f)), r(0.5f), rvba(make_float4(1.0f,0.0f,0.0f,1.0f)) { }
	Sphere(const float3 _C, float _r, const float4 _rvba) : C(_C), r(_r), rvba(_rvba) { }
	Sphere(const float3 _C, float _r) : C(_C), r(_r), rvba(make_float4(1.0f,0.0f,0.0f,1.0f)) { }*/
} Sphere;

typedef struct Node {
	Sphere s;
	uint   fg, fd;
} Node;

/*__host__ __device__ void createNode(Node * n, Node * fg, Node * fd, const Sphere & s)
{
  n->fg = fg;
  n->fd = fd;
  n->C  = s.C;
  n->r  = s.r;
}

__host__ __device__ Node * filsGauche(Node * n) { return n->fg; }

__host__ __device__ Node * filsDroite(Node * n) { return n->fd; }*/

//__host__ __device__ Sphere sphere(Node * n) { return n->s; }

__constant__ matrice3x4 MView;  // matrice inverse de la matrice de vue

__constant__ Node cnode[numObj];

template <class T>
__device__ void swap(T & v1, T & v2)
{
	T tmp(v1);
	v1 = v2;
	v2 = tmp;
}

__device__ float intersectionSphere(Rayon R, float3 C, float r)
{
	float3 L(C-R.A);
	float d(dot(L,R.u)), l2(dot(L,L)), r2(r*r), m2, q, res;
  
	if( d < 0.0f && l2 > r2 ) {
		res = 0.0f;
	}
	else
	{
		m2 = l2 - d*d;
		if( m2 > r2 ) {
			res = 0.0f;
		}
		else
		{
			q = sqrt(r2-m2);
			if( l2 > r2 ) res = d - q;
			else res = d + q;
		}
	}
  
	return res;
}

__device__ float intersectionPlan( Rayon R, float3 C, float3 N2 )
{
  float res;
  float3 N = normalize(make_float3(0.0f,1.0f,0.0f));
  float m(dot(N,R.u)), d, t;
  float3 L;
  
  if( fabs(m) < 0.0001f ) {
    res = 0.0f;
  }
  else {
    L = R.A - C;
    d = dot(N,L);
    t = -d/m;
    if( t > 0 ) {
      res = t;
    }
    else {
      res = 0.0f;
    }
  }
  
  return res;
}

__device__ float3 getNormale(float3 P, float3 C)
{
  return normalize(P-C);
}

__device__ float3 getNormaleP(float3 P)
{
  return normalize(make_float3(0.0f,1.0f,0.0f));
}

// multiplication d'un vecteur par une matrice (sans translation)
__device__ float3 mul(matrice3x4 M, float3 v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// multiplication d'un vecteur par une matrice avec translation
__device__ float4 mul(matrice3x4 M, float4 v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
   #ifdef DEVICE_EMU
    printf("%d: rgba = %f %f %f %f\n", threadIdx.x, rgba.x, rgba.y, rgba.z, rgba.w);
    #endif
    rgba.x = __saturatef(rgba.x);   // clamp entre [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
#ifdef DEVICE_EMU
    printf("%d: rgba = %x %x %x %x\n", threadIdx.x, uint(rgba.x*255), uint(rgba.y*255), uint(rgba.z*255), uint(rgba.w*255));
#endif
    return (uint(rgba.w*255)<<24)
		 | (uint(rgba.z*255)<<16)
		 | (uint(rgba.y*255)<<8 )
		 | (uint(rgba.x*255)    );
}

/*__device__ void myswap(Sphere &x, Sphere &y)
{
Sphere t = x;
x = y;
y = t;
}*/
/*__global__ void d_render(uint * d_output, uint imageW, uint imageH, float pas, float df, float tPixel)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		//float tPixel = 2.0f/(float)min(imageW,imageH);
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		Sphere s(cnode[1].s), s2(cnode[2].s), st(cnode[2].s);
		float t, t2, tt;
		s.C.x += pas, s2.C.x += pas;
		t = intersectionSphere(R,s.C,s.r);
		t2 = intersectionSphere(R,s2.C,s2.r);
		if( !t ) {
			//myswap(s,s2);
			//swap(t,t2);
         tt = t;
			t = t2;
			t2 = tt;
			st = s;
			s = s2;
			s2 = st;
		}
		else if( t2 && t2 < t ) {
			//myswap(s,s2);
			//swap(t,t2);
         tt = t;
			t = t2;
			t2 = tt;
         st = s;
         s = s2;
         s2 = st;
		}
		float4 f = make_float4(0,1,0,1)*(dot(getNormale(R.A+R.u*t,s.C),(-1.0f)*R.u));
		uint n = rgbaFloatToInt(f);
		//printf("%f\n",d_node[0].s.r);
		if( t > 0.0f )
			d_output[id] = n;
		//else d_output[id] = 0;
	}
	__syncthreads();
}
*/
/*__global__ void rayCast (uint * d_output, uint * d_temp, uint imageW, uint imageH, float pas, float df)
//(uint * result, uint * temp, uint imageW, uint imageH, float pas, float df)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y);
	uint id = x + y * gridDim.x;
	//float tmp= float(imageW)/float(gridDim.x);
	float t;

	//if( x < gridDim.x && y < gridDim.y )
	if( d_temp[id] == 0 )
	{
		float tPixel = 2.0f/float(imageW);
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		Sphere s(cnode[1].s);
		s.C.x += pas;
		t = intersectionSphere(R,s.C,s.r/(imageW/gridDim.x));

		if( t > 0.0f ) {		
			//float4 f = make_float4(0,1,0,1)*(dot(getNormale(R.A+R.u*t,s.C),(-1.0f)*R.u));
			d_output[id] = rgbaFloatToInt(make_float4(0,1,0,1));
			//printf("%d %d\n",int(x*tmp),int((y*tmp)/2));
		}
		else {
//       float tmp= float(imageW)/gridDim.x;
//       d_temp[int(x*tmp+(y*tmp)*imageW)] = 1;
//       d_temp[int(x*tmp+(tmp*(float(y)+0.5f)*imageW))] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(y*tmp)*imageW)] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(tmp*(float(y)+0.5f)*imageW))] = 1;
			//if(gridDim.x==16) printf("hep %d %f\n",gridDim.x,t);
		}
	}
	else {
//       float tmp= float(imageW)/gridDim.x;
//       d_temp[int(x*tmp+(y*tmp)*imageW)] = 1;
//       d_temp[int(x*tmp+(tmp*(float(y)+0.5f)*imageW))] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(y*tmp)*imageW)] = 1;
//       d_temp[int(tmp*(float(x)+0.5f)+(tmp*(float(y)+0.5f)*imageW))] = 1;
			//if(gridDim.x==16) printf("hep %d %f\n",gridDim.x,t);
	}
	//__syncthreads();
}*/

/*__global__  __device__ void rayCalc(float3 * A, float3 * u, float * prof, uint imageW, uint imageH, float df, float tPixel)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		A[id] = R.A;
		u[id] = R.u;
		prof[id] = 1000.0f;
	}
}*/


/*__global__  __device__ void rayTrace(uint * Obj, float * prof, float3 * A, float3 * u, uint imageW, uint imageH, float pas, float df, uint nObj)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		Sphere s(cnode[nObj].s);
		float t;
		s.C.x += pas;
		Rayon R;
		R.A = A[id];
		R.u = u[id];
		t = intersectionSphere(R,s.C,s.r);

		if( t > 0.0f && t < prof[id] ) {
			prof[id] = t;
			Obj[id] = nObj;
		}
	}
}*/
/*
__global__  __device__ void color(uint * result, uint * Obj, float * prof, float3 * A, float3 * u, uint imageW, uint imageH, float pas)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint id = x + y * imageW;

	if( x < imageW && y < imageH )
	{
		float t(prof[id]);
		if( t > 0.0f  && t < 1000.0f ) {
			Rayon R;
			R.A = A[id];
			R.u = u[id];
			Sphere s(cnode[Obj[id]].s);
			s.C.x += pas;
			float4 f = make_float4(s.R,s.V,s.B,s.A)*(dot(getNormale(R.A+R.u*t,s.C),(-1.0f)*R.u));
			result[id] = rgbaFloatToInt(f);
		}
		else {
			result[id] = 0;
		}
		prof[id] = 100000.0f;
	}
}*/
#ifdef DEBUG_RT_CUDA
__device__ bool notShadowRay( float4* d_debug_float4, uint* d_debug_uint, int i, Node * node, float3 A, float3 u, float pas ) {

#else
__device__ bool notShadowRay( Node * node, float3 A, float3 u, float pas ) {
#endif
   float t(0.0f);
	Node  n;
	Rayon ray;
	float3 L(make_float3(10.0f,10.0f,10.0f)), tmp;
	float dst(dot(tmp=(L-A),tmp));
	ray.A = A+u*0.0001f;
	ray.u = u;
	for( int j(0); j < numObj && !t; j++ ) {
		n = cnode[j];
		n.s.C.x += pas;
		if( n.fg ){
         t = intersectionPlan(ray,n.s.C,n.s.C);
         #ifdef DEVICE_EMU
//       printf("%d: j=%d, intersectionPlan t=%e\n", threadIdx.x, j, t);
         #endif
         #ifdef DEBUG_RT_CUDA
         //d_debug_uint4[threadIdx.x*16+4*j+0]=10;
//       d_debug_float4[threadIdx.x*32+16*i+3*j+0].x = t;
//       d_debug_float4[threadIdx.x*32+16*i+3*j+0].y = 99999.9f;
         #endif
      }
		else{
#ifdef DEVICE_EMU
    printf("%d: i=%d, n.s.C = %e %e %e\n", threadIdx.x, i, n.s.C.x, n.s.C.y, n.s.C.z);
    printf("%d: i=%d, n.s.r = %e\n", threadIdx.x, i, n.s.r);
#endif
         #ifdef DEBUG_RT_CUDA
         d_debug_float4[threadIdx.x*32+16*i+3*j+0].x = n.s.C.x;
         d_debug_float4[threadIdx.x*32+16*i+3*j+0].y = n.s.C.y;
         d_debug_float4[threadIdx.x*32+16*i+3*j+0].z = n.s.C.z;
         d_debug_float4[threadIdx.x*32+16*i+3*j+0].w = n.s.r;
         #endif
         t = intersectionSphere(ray,n.s.C,n.s.r);
         #ifdef DEVICE_EMU
         printf("%d: j=%d, intersectionSphere t=%e\n", threadIdx.x, j, t);
         #endif
         #ifdef DEBUG_RT_CUDA
         d_debug_float4[threadIdx.x*32+16*i+3*j+1].x = t;
         d_debug_float4[threadIdx.x*32+16*i+3*j+1].y = 99999.9f;
         #endif
      }
		if( t > 0.0f && dot(tmp=(A+u*t),tmp) > dst ){
         t = 0.0f;
         #ifdef DEVICE_EMU
//       printf("%d: j=%d, && dot t=%e\n", threadIdx.x, j, t);
         #endif
         #ifdef DEBUG_RT_CUDA
//       d_debug_float4[threadIdx.x*32+16*i+3*j+2].x = t;
//       d_debug_float4[threadIdx.x*32+16*i+3*j+2].y = 99999.9f;
         #endif
      }
	}
   #ifdef DEVICE_EMU
// printf("%d: t=%e\n", threadIdx.x, t);
   #endif
   #ifdef DEBUG_RT_CUDA
// d_debug_float4[threadIdx.x*32+16*i+13].x = t;
// d_debug_float4[threadIdx.x*32+16*i+13].y = 99999.9f;
   d_debug_float4[threadIdx.x*32+16*i+15].x = 88888.8f;
   #endif
	return t == 0.0f;
}

__device__ float float2int_pow20(float a)
{
   return a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a;
}

__device__ float float2int_pow50(float a)
{
   return a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a;

}
#ifdef DEBUG_RT_CUDA
__global__  __device__ void render(float4* d_debug_float4, uint* d_debug_uint, uint * result, Node * dnode, uint imageW, uint imageH, float pas, float df)
#else
__global__  __device__ void render(uint * result, Node * dnode, uint imageW, uint imageH, float pas, float df)
#endif
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint tid(__umul24(threadIdx.y, blockDim.x) + threadIdx.x);

	uint id(x + y * imageW);
	float4 pile[5];
	uint Obj, nRec(5), n(0);
	//__shared__ Node node[numObj];
	float prof, tmp;

	//if( tid < numObj ) node[tid] = cnode[tid];

	for( int i(0); i < nRec; ++i )
		pile[i] = make_float4(0.0f,0.0f,0.0f,1.0f);

	if( x < imageW && y < imageH )
	{
		prof = 10000.0f;
		result[id] = 0;
		float tPixel(2.0f/float(min(imageW,imageH)));
		float4 f(make_float4(0.0f,0.0f,0.0f,1.0f));
		matrice3x4 M(MView);
		Rayon R;
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
#ifdef DEVICE_EMU
//    printf("%d: R.A = %e %e %e\n", threadIdx.x, R.A.x, R.A.y, R.A.z);
//    printf("%d: R.u = %e %e %e\n", threadIdx.x, R.u.x, R.u.y, R.u.z);
#endif
#ifdef DEBUG_RT_CUDA
//    d_debug_float4[threadIdx.x*2+0].x= R.A.x;
//    d_debug_float4[threadIdx.x*2+0].y= R.A.y;
//    d_debug_float4[threadIdx.x*2+0].z= R.A.z;
//    d_debug_float4[threadIdx.x*2+1].x= R.u.x;
//    d_debug_float4[threadIdx.x*2+1].y= R.u.y;
//    d_debug_float4[threadIdx.x*2+1].z= R.u.z;
#endif
		__syncthreads();

		for( int i(0); i < nRec && n == i; i++ ) {

			for( int j(0); j < numObj; j++ ) {
				Node nod(cnode[j]);
				Sphere s(nod.s);
				float t;
				s.C.x += pas;
				if( nod.fg )
					t = intersectionPlan(R,s.C,s.C);
				else
					t = intersectionSphere(R,s.C,s.r);

				if( t > 0.0f && t < prof ) {
					prof = t;
					Obj = j;
				}
			}
#ifdef DEBUG_RT_CUDA
         //d_debug_float4[threadIdx.x*5+i].x= prof;
#endif
#ifdef DEVICE_EMU
//       printf("%d: i=%d, t=%e\n", threadIdx.x, i, prof);
#endif
			float t = prof;
			if( t > 0.0f && t < 10000.0f ) {
				n++;
				Node nod(cnode[Obj]);
				Sphere s(nod.s);
				s.C.x += pas;
				float4 color(make_float4(s.R,s.V,s.B,s.A));
				float3 P(R.A+R.u*t), L(normalize(make_float3(10.0f,10.0f,10.0f)-P)), V(normalize(R.A-P));
				float3 N(nod.fg?getNormaleP(P):getNormale(P,s.C));
				float3 Np(dot(V,N)<0.0f?(-1*N):N);
				pile[i] = 0.05f * color;
            #ifdef DEVICE_EMU
//          printf("%d: i=%d, pile[i] = %e %e %e %e\n", threadIdx.x, i, pile[i].x, pile[i].y, pile[i].z, pile[i].w);
//          printf("%d: i=%d, color = %e %e %e %e\n", threadIdx.x, i, color.x, color.y, color.z, color.w);
//          printf("%d: i=%d, P = %e %e %e\n", threadIdx.x, i, P.x, P.y, P.z);
//          printf("%d: i=%d, L = %e %e %e\n", threadIdx.x, i, L.x, L.y, L.z);
//          printf("%d: i=%d, V = %e %e %e\n", threadIdx.x, i, V.x, V.y, V.z);
//          printf("%d: i=%d, N = %e %e %e\n", threadIdx.x, i, N.x, N.y, N.z);
//          printf("%d: i=%d, Np = %e %e %e\n", threadIdx.x, i, Np.x, Np.y, Np.z);
//          printf("%d: i=%d, dot(Np,L) = %e\n", threadIdx.x, i, dot(Np,L));
            //printf("%d: i=%d, notShadowRay(cnode,P,L,pas) = %d\n", threadIdx.x, i, (int) notShadowRay(cnode,P,L,pas));

            #endif
            #ifdef DEBUG_RT_CUDA
            //d_debug_float4[threadIdx.x*16+i*3+0]= pile[i];
//          d_debug_float4[threadIdx.x*16+i*8+0]= color;
//          d_debug_float4[threadIdx.x*16+i*8+1].x= P.x;d_debug_float4[threadIdx.x*16+i*8+1].y= P.y;d_debug_float4[threadIdx.x*16+i*8+1].z= P.z;
//          d_debug_float4[threadIdx.x*16+i*8+2].x= L.x;d_debug_float4[threadIdx.x*16+i*8+2].y= L.y;d_debug_float4[threadIdx.x*16+i*8+2].z= L.z;
//          d_debug_float4[threadIdx.x*16+i*8+3].x= V.x;d_debug_float4[threadIdx.x*16+i*8+3].y= V.y;d_debug_float4[threadIdx.x*16+i*8+3].z= V.z;
//          d_debug_float4[threadIdx.x*16+i*8+4].x= N.x;d_debug_float4[threadIdx.x*16+i*8+4].y= N.y;d_debug_float4[threadIdx.x*16+i*8+4].z= N.z;
//          d_debug_float4[threadIdx.x*16+i*8+5].x= Np.x;d_debug_float4[threadIdx.x*16+i*8+5].y= Np.y;d_debug_float4[threadIdx.x*16+i*8+5].z= Np.z;
//          d_debug_float4[threadIdx.x*16+i*8+6].x= dot(Np,L);
            //d_debug_float4[threadIdx.x*16+i*8+7].x= (float) notShadowRay(cnode,P,L,pas);
            #endif
            #ifdef DEBUG_RT_CUDA
            if( dot(Np,L) > 0.0f && notShadowRay(d_debug_float4, d_debug_uint, i, cnode,P,L,pas) ) {
            #else
            if( dot(Np,L) > 0.0f && notShadowRay(cnode,P,L,pas) ) {
            #endif
               //float3 Ri(2.0f*Np*dot(Np,L) - L);
					float3 Ri(normalize(L+V));
					//Ri = (L+V)/normalize(L+V);
					pile[i] += 0.3f * color* (min(1.0f,dot(Np,L)));
               #ifdef DEVICE_EMU
//             printf("%d: i=%d, pile[i] = %e %e %e %e\n", threadIdx.x, i, pile[i].x, pile[i].y, pile[i].z, pile[i].w);
               #endif
               #ifdef DEBUG_RT_CUDA
               //d_debug_float4[threadIdx.x*16+i*3+1]= pile[i];
               #endif
               #ifdef FIXED_CONST_PARSE
					tmp = 0.8f * pow(max(0.0f,min(1.0f,dot(Np,Ri))),50.0f);
               #else
               tmp = 0.8f * float2int_pow50(max(0.0f,min(1.0f,dot(Np,Ri))));
               #endif
					pile[i].x += tmp;
					pile[i].y += tmp;
					pile[i].z += tmp;
               #ifdef DEVICE_EMU
//             printf("%d: i=%d, pile[i] = %e %e %e %e\n", threadIdx.x, i, pile[i].x, pile[i].y, pile[i].z, pile[i].w);
               #endif
               #ifdef DEBUG_RT_CUDA
               //d_debug_float4[threadIdx.x*16+i*3+2]= pile[i];
               #endif
				}

				R.u = 2.0f*N*dot(N,V) - V;
				R.u = normalize(R.u);
				R.A = P+R.u*0.0001f;
			}
			prof = 10000.0f;
		}
      #ifdef DEBUG_RT_CUDA
      /*d_debug_float4[threadIdx.x*5+0]= pile[0];
      d_debug_float4[threadIdx.x*5+1]= pile[1];
      d_debug_float4[threadIdx.x*5+2]= pile[2];
      d_debug_float4[threadIdx.x*5+3]= pile[3];
      d_debug_float4[threadIdx.x*5+4]= pile[4];*/
      #endif
#ifdef DEVICE_EMU
//    printf("%d: pile[0] = %e %e %e %e\n", threadIdx.x, pile[0].x, pile[0].y, pile[0].z, pile[0].w);
//    printf("%d: pile[1] = %e %e %e %e\n", threadIdx.x, pile[1].x, pile[1].y, pile[1].z, pile[1].w);
//    printf("%d: pile[2] = %e %e %e %e\n", threadIdx.x, pile[2].x, pile[2].y, pile[2].z, pile[2].w);
//    printf("%d: pile[3] = %e %e %e %e\n", threadIdx.x, pile[3].x, pile[3].y, pile[3].z, pile[3].w);
//    printf("%d: pile[4] = %e %e %e %e\n", threadIdx.x, pile[4].x, pile[4].y, pile[4].z, pile[4].w);
#endif
      for( int i(n-1); i > 0; i-- )
				pile[i-1] = pile[i-1] + 0.8f*pile[i];
#ifdef DEVICE_EMU
//    printf("%d: pile[0] = %e %e %e %e\n", threadIdx.x, pile[0].x, pile[0].y, pile[0].z, pile[0].w);
#endif
      result[id] += rgbaFloatToInt(pile[0]);
	}
}

/*__global__  __device__ void renderPixel(uint * result, Node * dnode, uint imageW, uint imageH, float pas, float df)
{
	uint id(blockIdx.x + __umul24(blockIdx.y, imageW));
	uint tid(threadIdx.x), x(blockIdx.x), y(blockIdx.y);
	Node node;
	float t(0.0f), tPixel;
	float4 Color(make_float4(0.0f,0.0f,0.0f,1.0f));
	matrice3x4 M(MView);
	Rayon R;
	Sphere s;
	__shared__ float T[numObj];
	__shared__ uint Obj;

	T[tid] = 10000.0f;
	
	if( x < imageW && y < imageH && tid < numObj ) {
		node = dnode[tid];
		if( tid == 0 ) result[id] = 0;
		tPixel = 2.0f/float(min(imageW,imageH));
		R.A = make_float3(M.m[0].w,M.m[1].w,M.m[2].w);
		R.u = make_float3(M.m[0])*df
			+ make_float3(M.m[2])*(float(x)-float(imageW)*0.5f)*tPixel
			+ make_float3(M.m[1])*(float(y)-float(imageH)*0.5f)*tPixel;
		R.u = normalize(R.u);
		
		s = node.s;
		s.C.x += pas;

		if( node.fg )
			t = intersectionPlan(R,s.C,s.C);
		else
			t = intersectionSphere(R,s.C,s.r);

		T[tid] = t;

		__syncthreads();

		if( tid == 0 ) {
			float tmp(t);
			Obj = 0;
			for( int i(1); i < numObj; i++ ) {
				if( T[i] > 0.0f && ( tmp == 0.0f || T[i] < tmp ) ) {
					tmp = T[i];
					Obj = i;
				}
			}
		}

		__syncthreads();

		if( tid == Obj && t > 0.0f ) {
			s = node.s;
			s.C.x += pas;
			float3 P(R.A+R.u*t), L(normalize(make_float3(0,1,2)-P)), V(-1*R.u);
			float3 N(node.fg?getNormaleP(P):getNormale(P,s.C));
			if( dot(N,L) > 0.0f ) {
				Color = 0.5f*make_float4(s.R,s.V,s.B,s.A)*(max(0.0f,dot(N,L)));
            #ifdef FIXED_CONST_PARSE
				Color += 0.8f*make_float4(1.0f,1.0f,1.0f,1.0f)*pow(max(0.0f,min(1.0f,dot(2.0f*N*dot(N,L)-L,V))),20.0f);
            #else
            Color += 0.8f*make_float4(1.0f,1.0f,1.0f,1.0f)*float2int_pow20(max(0.0f,min(1.0f,dot(2.0f*N*dot(N,L)-L,V))));
            #endif
			}
			result[id] = rgbaFloatToInt(Color);
		}
	}

}
*/

#endif // __RAYTRACING_KERNEL_H__
