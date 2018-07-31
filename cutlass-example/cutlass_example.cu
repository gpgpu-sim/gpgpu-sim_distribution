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

#ifdef WMMA_GEMM_32x32x32_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(32, 32, 32);
#endif
#ifdef WMMA_GEMM_32x32x32_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(32, 32, 32);
#endif
#ifdef WMMA_GEMM_32x32x32_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(32, 32, 32);
#endif
#ifdef WMMA_GEMM_32x32x32_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(32, 32, 32);
#endif

#ifdef WMMA_GEMM_128x128x128_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(128, 128, 128);
#endif
#ifdef WMMA_GEMM_128x128x128_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(128, 128, 128);
#endif
#ifdef WMMA_GEMM_128x128x128_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(128, 128, 128);
#endif
#ifdef WMMA_GEMM_128x128x128_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(128, 128, 128);
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

#ifdef WMMA_GEMM_256x256x256_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 256);
#endif

#ifdef WMMA_GEMM_256x256x256_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 256);
#endif

#ifdef WMMA_GEMM_256x256x256_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 256);
#endif

#ifdef WMMA_GEMM_256x256x256_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 256);
#endif
#ifdef WMMA_GEMM_512x512x512_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(512, 512, 512);
#endif

#ifdef WMMA_GEMM_512x512x512_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(512, 512, 512);
#endif

#ifdef WMMA_GEMM_512x512x512_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(512, 512, 512);
#endif

#ifdef WMMA_GEMM_512x512x512_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(512, 512, 512);
#endif

#ifdef WMMA_GEMM_768x768x768_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(768, 768, 768);
#endif
#ifdef WMMA_GEMM_768x768x768_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(768, 768, 768);
#endif

#ifdef WMMA_GEMM_768x768x768_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(768, 768, 768);
#endif

#ifdef WMMA_GEMM_768x768x768_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(768, 768, 768);
#endif


#ifdef WMMA_GEMM_1024x1024x1024_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(1024, 1024, 1024);
#endif

#ifdef WMMA_GEMM_1024x1024x1024_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(1024, 1024, 1024);
#endif

#ifdef WMMA_GEMM_1024x1024x1024_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(1024, 1024, 1024);
#endif

#ifdef WMMA_GEMM_1024x1024x1024_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(1024, 1024, 1024);
#endif
#ifdef WMMA_GEMM_2048x2048x2048_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(2048, 2048, 2048);
#endif

#ifdef WMMA_GEMM_2048x2048x2048_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(2048, 2048, 2048);
#endif

#ifdef WMMA_GEMM_2048x2048x2048_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(2048, 2048, 2048);
#endif

#ifdef WMMA_GEMM_2048x2048x2048_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(2048, 2048, 2048);
#endif
#ifdef WMMA_GEMM_4096x4096x4096_NT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(4096, 4096, 4096);
#endif

#ifdef WMMA_GEMM_4096x4096x4096_NN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(4096, 4096, 4096);
#endif

#ifdef WMMA_GEMM_4096x4096x4096_TT
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(4096, 4096, 4096);
#endif

#ifdef WMMA_GEMM_4096x4096x4096_TN
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(4096, 4096, 4096);
#endif
}


