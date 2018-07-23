//added by me
#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include <gemm_testbed.h>
#include <gemm.h>

int main(int argc, char* argv[]) {

#ifdef WMMA_GEMM_16x16x16_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
#endif

#ifdef WMMA_GEMM_16x16x32_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
#endif

#ifdef WMMA_GEMM_16x16x16_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
#endif


#ifdef WMMA_GEMM_16x16x32_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
#endif

#ifdef WMMA_GEMM_16x16x16_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
#endif

#ifdef WMMA_GEMM_16x16x32_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
#endif

#ifdef WMMA_GEMM_16x16x16_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
#endif

#ifdef WMMA_GEMM_16x16x32_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
#endif

#ifdef WMMA_16x16x16_GEMM_256x256x128_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
#endif

#ifdef WMMA_16x16x16_GEMM_256x256x128_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
#endif

#ifdef WMMA_16x16x16_GEMM_256x256x128_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
#endif

#ifdef WMMA_16x16x16_GEMM_256x256x128_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
#endif

}


