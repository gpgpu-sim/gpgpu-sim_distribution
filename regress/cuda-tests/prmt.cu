// prmt (byte permute, __byte_perm) in its default form, including selectors
// that replicate a byte's sign, and in its f4e, b4e, rc8, ecl, ecr and rc16
// modes, with operands whose top bit is set. Results must match the host.
//
//   prmt   prints PASSED or FAILED, exits 1 on failure
#include <cuda_runtime.h>
#include <cstdio>

const int N = 256;
const int OPS = 8;

__global__ void permute(const unsigned *in, unsigned *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2], r;
  unsigned *o = out + OPS * i;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[0] = r;
  o[1] = __byte_perm(a, b, c);
  asm("prmt.b32.f4e %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[2] = r;
  asm("prmt.b32.b4e %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[3] = r;
  asm("prmt.b32.rc8 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[4] = r;
  asm("prmt.b32.ecl %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[5] = r;
  asm("prmt.b32.ecr %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[6] = r;
  asm("prmt.b32.rc16 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  o[7] = r;
}

// The PTX ISA's definition: bytes 0-3 come from a, 4-7 from b.
static unsigned byte_of(unsigned a, unsigned b, int k) {
  unsigned long long v = a | (unsigned long long)b << 32;
  return (v >> (8 * k)) & 0xff;
}

static unsigned reference(unsigned a, unsigned b, unsigned c, int op) {
  unsigned r = 0, m = c & 3;
  for (int i = 0; i < 4; i++) {
    unsigned byte;
    if (op <= 1) {
      unsigned s = (c >> (4 * i)) & 0xf;
      byte = byte_of(a, b, s & 7);
      // __byte_perm uses only the low 3 bits of each selector
      if (op == 0 && (s & 8)) byte = byte & 0x80 ? 0xff : 0;
    } else {
      int k;
      switch (op) {
        case 2: k = m + i; break;                       // f4e
        case 3: k = (m - i) & 7; break;                 // b4e
        case 4: k = m; break;                           // rc8
        case 5: k = (int)m > i ? m : i; break;          // ecl
        case 6: k = (int)m < i ? m : i; break;          // ecr
        default: k = (m & 1) * 2 + (i & 1); break;      // rc16
      }
      byte = byte_of(a, b, k);
    }
    r |= byte << (8 * i);
  }
  return r;
}

int main() {
  static unsigned in[3 * N], out[OPS * N];
  unsigned x = 0x2545f491;
  for (int i = 0; i < 3 * N; i++) {
    x ^= x << 13, x ^= x >> 17, x ^= x << 5;  // xorshift32
    in[i] = x;
  }
  in[0] = 0x80ff7f01, in[1] = 0xfe800010, in[2] = 0x8888;  // sign bytes
  unsigned *din, *dout;
  cudaMalloc(&din, sizeof(in));
  cudaMalloc(&dout, sizeof(out));
  cudaMemcpy(din, in, sizeof(in), cudaMemcpyHostToDevice);
  permute<<<N / 64, 64>>>(din, dout);
  cudaError_t e = cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("CUDA error: %s\nFAILED\n", cudaGetErrorString(e));
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < N; i++)
    for (int k = 0; k < OPS; k++) {
      unsigned a = in[3 * i], b = in[3 * i + 1], c = in[3 * i + 2];
      unsigned want = reference(a, b, c, k), got = out[OPS * i + k];
      if (got != want) {
        if (bad < 10)
          printf("op %d on %08x %08x %08x: got %08x, expected %08x\n", k, a, b,
                 c, got, want);
        bad++;
      }
    }
  printf(bad ? "FAILED\n" : "PASSED\n");
  return bad != 0;
}
