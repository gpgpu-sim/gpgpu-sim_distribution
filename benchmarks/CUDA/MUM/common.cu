#ifndef COMMON_CU__
#define COMMON_CU__ 1
// Children are labeled as ACGT$
const int basecount = 5;

// Note: max pixel size is 16 bytes

const unsigned char DNA_A = 'A';
const unsigned char DNA_C = 'B';
const unsigned char DNA_G = 'C';
const unsigned char DNA_T = 'D';
const unsigned char DNA_S = 'E';

// 4 bytes
struct TextureAddress
{
  union
  {
    unsigned int data;

    struct
    {
	  unsigned short x;
      unsigned short y;
    };
  };
};

// Store the start, end coordinate of node, and $link in 1 pixel
struct PixelOfNode
{
  union
  {
    ulong4 data;
    struct
    {
      int start;
      int end;
      TextureAddress childD;
      TextureAddress suffix;
    };
  };
};

// Store the ACGT links in 1 pixel
struct PixelOfChildren
{
  union
  {
    ulong4 data;
	TextureAddress children[4];
  };
};

#define FORWARD   0x0000
#define REVERSE   0x8000
#define FRMASK    0x8000
#define FRUMASK   0x7FFF


#endif
