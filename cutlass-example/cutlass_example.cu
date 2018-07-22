//added by me
#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include <gemm_testbed.h>
#include <gemm.h>

int main(int argc, char* argv[]) {
    typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor, 
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);

}


