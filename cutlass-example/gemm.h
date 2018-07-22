/***************************************************************************************************
* Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted
* provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright notice, this list of
*       conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice, this list of
*       conditions and the following disclaimer in the documentation and/or other materials
*       provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
*       to endorse or promote products derived from this software without specific prior written
*       permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
* FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
**************************************************************************************************/

#include <cutlass/cutlass.h>
#include <gemm_testbed.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  typename Gemm::Params params;

  printf("run_gemm-2:m=%d\n",m);
  printf("run_gemm-2:n=%d\n",n);
  printf("run_gemm-2:k=%d\n",k);
  printf("run_gemm-2:lda=%d\n",lda);
  printf("run_gemm-2:ldb=%d\n",ldb);
  printf("run_gemm-2:ldc=%d\n",ldc);
  printf("run_gemm-2:alpha=%.2f\n",alpha);
  printf("run_gemm-2:beta=%.2f\n",beta);
  
  test::GemmTestbed<
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
      typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      >
      testbed(m,
              n,
              k,
              lda,
              ldb,
              ldc,
              cutlass::convert(GemmTraits_::kLayoutA),
              cutlass::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  testbed.initialize();

 // if (testbed.has_cublas_support()) {
 //   EXPECT_TRUE(testbed.verify_host_with_cublas());
 // }

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
                    testbed.alpha,
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    testbed.beta,
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.ptr_computed(),
                    testbed.ldc());

  printf("SIZE_OF_PARAM=%lu\n",sizeof(params));
  void *ptr =&params;
  for(int kk=0;kk<108;kk++){
  	printf("KERNELPARAM:%d:%08x\n",kk,*((((int *) ptr)+kk)));	
  }  
  printf("m=%lu\n",sizeof(params.m));
  printf("n=%lu\n",sizeof(params.n));
  printf("k=%lu\n",sizeof(params.k));
//  printf("alpha=%d\n",sizeof(params.alpha));
//  printf("beta=%d\n",sizeof(params.beta));
//  printf("d_a=%d\n",sizeof(params.d_a));
//  printf("lda=%d\n",sizeof(params.lda));
//  printf("d_b=%d\n",sizeof(params.d_b));
//  printf("ldb=%d\n",sizeof(params.ldb));
//  printf("d_c=%d\n",sizeof(params.d_c));
//  printf("ldc=%d\n",sizeof(params.ldc));
//  printf("d_d=%d\n",sizeof(params.d_d));
//  printf("ldd=%d\n",sizeof(params.ldd));
  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  if(result==cudaSuccess){
	printf("Successfully Launched\n");	
  }
  int save=1;
  int myval=testbed.verify_with_host(save,save);
  if(myval==1){
	printf("Result Verified\n");	
  }
 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
  int lda = GemmTraits_::kLayoutA == cutlass::MatrixLayout::kColumnMajor ? m : k;
  int ldb = GemmTraits_::kLayoutB == cutlass::MatrixLayout::kColumnMajor ? k : n;
  printf("run_gemm-1:m=%d\n",m);
  printf("run_gemm-1:n=%d\n",n);
  printf("run_gemm-1:k=%d\n",k);
  printf("run_gemm-1:alpha=%.2f\n",alpha);
  printf("run_gemm-1:beta=%.2f\n",beta);
  printf("run_gemm-1:lda=%d\n",lda);
  printf("run_gemm-1:ldb=%d\n",ldb);
  run_gemm<GemmTraits_>(m, n, k, lda, ldb, m, alpha, beta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
