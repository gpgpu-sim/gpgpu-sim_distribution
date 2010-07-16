#include "EasyBMP.h"

BMP out_bmp;

void initialize_bmp(unsigned width, unsigned height, unsigned depth)
{
    SetEasyBMPwarningsOff();
    out_bmp.SetSize(width, height);
    out_bmp.SetBitDepth(depth);
}

void create_bmp(unsigned *data)
{
   unsigned height = out_bmp.TellHeight();
   unsigned width = out_bmp.TellWidth();
    for (unsigned y=0; y< height; y++){
       for  (unsigned x=0; x< width; x++) {
          //printf("%8x ", c_output[x+y*width]);
          out_bmp(x,(height-y-1))->Red   = 0x000000FF & data[x+y*width];
          out_bmp(x,(height-y-1))->Green = (0x0000FF00 & data[x+y*width]) >> 8;
          out_bmp(x,(height-y-1))->Blue  = (0x00FF0000 & data[x+y*width]) >> 16;
          out_bmp(x,(height-y-1))->Alpha = (0xFF000000 & data[x+y*width]) >> 24;
       }
    }
    out_bmp.WriteToFile("output.bmp");
}


