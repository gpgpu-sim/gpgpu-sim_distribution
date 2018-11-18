/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _PTXJIT_H_
#define _PTXJIT_H_

struct param{
    bool isPointer;
    size_t size;
    unsigned char *data;
    unsigned offset;
};

/*
 * PTX is equivalent to the following kernel:
 *
 * __global__ void myKernel(int *data)
 * {
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *     data[tid] = tid;
 * }
 *
 */

char myPtx64[] = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 64\n\
.visible .entry _Z8myKernelPi(\n\
	.param .u64 _Z8myKernelPi_param_0\n\
)\n\
{\n\
	.reg .s32 	%r<5>;\n\
	.reg .s64 	%rd<5>;\n\
	ld.param.u64 	%rd1, [_Z8myKernelPi_param_0];\n\
	cvta.to.global.u64 	%rd2, %rd1;\n\
	.loc 1 3 1\n\
	mov.u32 	%r1, %ntid.x;\n\
	mov.u32 	%r2, %ctaid.x;\n\
	mov.u32 	%r3, %tid.x;\n\
	mad.lo.s32 	%r4, %r1, %r2, %r3;\n\
	mul.wide.s32 	%rd3, %r4, 4;\n\
	add.s64 	%rd4, %rd2, %rd3;\n\
	.loc 1 4 1\n\
	st.global.u32 	[%rd4], %r4;\n\
	.loc 1 5 2\n\
	ret;\n\
}\n\
";

char myPtx32[] = "\n\
.version 3.2\n\
.target sm_20\n\
.address_size 32\n\
.visible .entry _Z8myKernelPi(\n\
	.param .u32 _Z8myKernelPi_param_0\n\
)\n\
{\n\
	.reg .s32 	%r<9>;\n\
	ld.param.u32 	%r1, [_Z8myKernelPi_param_0];\n\
	cvta.to.global.u32 	%r2, %r1;\n\
	.loc 1 3 1\n\
	mov.u32 	%r3, %ntid.x;\n\
	mov.u32 	%r4, %ctaid.x;\n\
	mov.u32 	%r5, %tid.x;\n\
	mad.lo.s32 	%r6, %r3, %r4, %r5;\n\
	.loc 1 4 1\n\
	shl.b32 	%r7, %r6, 2;\n\
	add.s32 	%r8, %r2, %r7;\n\
	st.global.u32 	[%r8], %r6;\n\
	.loc 1 5 2\n\
	ret;\n\
}\n\
";

#endif
