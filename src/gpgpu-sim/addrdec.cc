/* 
 * addrdec.c 
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include <string.h>
#include "addrdec.h"
//#include "gpu-sim.h"
#include "../option_parser.h"

int ADDR_CHIP_S = 10;
extern int gpgpu_mem_address_mask;

long int powli( long int x, long int y ) // compute x to the y
{
   long int r = 1;
   int i; 
   for (i = 0; i < y; ++i ) {
      r *= x;
   }
   return r;
}

void addrdec_display(addrdec_t *a) {
   //printf("DRAM:  unused:%x chip:%x row:%x col:%x bk:%x\n",
   //       a.dram.unused, a.dram.chip, GET_ROW(a), GET_COL(a), a.dram.bk);

   if (a->chip)   printf("\tchip:%x ",  a->chip);
   if (a->row)    printf("\trow:%x ",   a->row);
   if (a->col)    printf("\tcol:%x ",   a->col);
   if (a->bk)     printf("\tbk:%x ",    a->bk);
   if (a->burst)  printf("\tburst:%x ", a->burst);
}  

unsigned long long int addrdec_packbits(unsigned long long int mask, 
                                        unsigned long long int val,
                                        unsigned char high, unsigned char low) 
{
   int i, pos;
   unsigned long long int out;
   out = 0;
   pos = 0;
   for (i=low;i<high;i++) {
      if ((mask & ((unsigned long long int)1<<i)) != 0) {
         out |= ((val & ((unsigned long long int)1<<i)) >> i) << pos;
         pos++;
      }
      // printf("%02d: %016llx %d\n",i,out,pos);
   }

   return out;
}

unsigned long long int addrdec_mask[5] = {
   0x0000000000001C00,
   0x0000000000000300,
   0x000000000FFF0000,
   0x000000000000E0FF,
   0x000000000000000F
};

void addrdec_getmasklimit(unsigned long long int mask, unsigned char *high, unsigned char *low) 
{
   *high = 64;
   *low = 0;
   int i;
   int low_found = 0;

   for (i=0;i<64;i++) {
      if ((mask & ((unsigned long long int)1<<i)) != 0) {
         if (low_found) {
            *high = i + 1;
         } else {
            *high = i + 1;
            *low = i;
            low_found = 1;
         }
      }
      // printf("%02d: %016llx %d\n",i,out,pos);
   }
}

unsigned char addrdec_mklow[5] = { 0, 0, 0, 0, 0};
unsigned char addrdec_mkhigh[5] = { 64, 64, 64, 64, 64};

static unsigned int gap;
static int Nchips;

void addrdec_tlx(unsigned long long int addr, addrdec_t *tlx) 
{  
   unsigned long long int addr_for_chip,rest_of_addr;
   if (!gap) {
      tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr, addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
      tlx->bk   = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK], addrdec_mklow[BK]);
      tlx->row  = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
      tlx->col  = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL], addrdec_mklow[COL]);
      tlx->burst= addrdec_packbits(addrdec_mask[BURST], addr, addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
   } else {
      addr_for_chip= ( (addr>>ADDR_CHIP_S) % Nchips) << ADDR_CHIP_S;
      rest_of_addr= ( (addr>>ADDR_CHIP_S) / Nchips) << ADDR_CHIP_S;                

      tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr_for_chip, addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
      if (addrdec_mask[BK] > addrdec_mask[CHIP]) {
         tlx->bk   = addrdec_packbits(addrdec_mask[BK], rest_of_addr, addrdec_mkhigh[BK], addrdec_mklow[BK]);
      } else {
         tlx->bk   = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK], addrdec_mklow[BK]);
      }
      if (addrdec_mask[ROW] > addrdec_mask[CHIP]) {
         tlx->row  = addrdec_packbits(addrdec_mask[ROW], rest_of_addr, addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
      } else {
         tlx->row  = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
      }
      tlx->col  = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL], addrdec_mklow[COL]);
      tlx->burst= addrdec_packbits(addrdec_mask[BURST], addr, addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
   }
}

unsigned int LOGB2_32( unsigned int v ) {
   unsigned int shift;
   unsigned int r;

   r = 0;

   shift = (( v & 0xFFFF0000) != 0 ) << 4; v >>= shift; r |= shift;
   shift = (( v & 0xFF00    ) != 0 ) << 3; v >>= shift; r |= shift;
   shift = (( v & 0xF0      ) != 0 ) << 2; v >>= shift; r |= shift;
   shift = (( v & 0xC       ) != 0 ) << 1; v >>= shift; r |= shift;
   shift = (( v & 0x2       ) != 0 ) << 0; v >>= shift; r |= shift;

   return r;
}


static char *addrdec_option = NULL;
void addrdec_setoption(option_parser_t opp)
{
   option_parser_register(opp, "-gpgpu_mem_addr_mapping", OPT_CSTR, &addrdec_option,
      "mapping memory address to dram model {dramid@<start bit>;<memory address map>}",
      NULL);
}

void addrdec_parseoption(const char *option)
{
   unsigned int dramid_start = 0;
   int dramid_parsed = sscanf(option, "dramid@%d", &dramid_start);
   if (dramid_parsed == 1) {
      ADDR_CHIP_S = dramid_start;
   } else {
      ADDR_CHIP_S = -1;
   }
   
   const char *cmapping = strchr(option, ';');
   if (cmapping == NULL) {
      cmapping = option;
   } else {
      cmapping += 1;
   }

   addrdec_mask[CHIP] = 0x0;
   addrdec_mask[BK]   = 0x0;
   addrdec_mask[ROW]  = 0x0;
   addrdec_mask[COL]  = 0x0;
   addrdec_mask[BURST]= 0x0;
   
   int ofs = 63;
   while ((*cmapping) != '\0') {
      switch (*cmapping) {
         case 'D': case 'd':  
            assert(dramid_parsed != 1); addrdec_mask[CHIP]  |= (1ULL << ofs); ofs--; break;
         case 'B': case 'b':   addrdec_mask[BK]    |= (1ULL << ofs); ofs--; break;
         case 'R': case 'r':   addrdec_mask[ROW]   |= (1ULL << ofs); ofs--; break;
         case 'C': case 'c':   addrdec_mask[COL]   |= (1ULL << ofs); ofs--; break;
         case 'S': case 's':   addrdec_mask[BURST] |= (1ULL << ofs); addrdec_mask[COL]   |= (1ULL << ofs); ofs--; break;
         // ignore bit
         case '0': ofs--; break;
         // ignore character
         case '|':
         case ' ':
         case '.': break;
         default:
            fprintf(stderr, "ERROR: Invalid address mapping character '%c' in option '%s'\n", *cmapping, option);
      }
      cmapping += 1;
   }

   if (ofs != -1) {
      fprintf(stderr, "ERROR: Invalid address mapping length (%d) in option '%s'\n", 63 - ofs, option);
      assert(ofs == -1);
   }
}


void addrdec_setnchip(unsigned int nchips) 
{
   unsigned i;
   unsigned long long int mask;
   unsigned int nchipbits = LOGB2_32(nchips);
   Nchips = nchips;

   gap = (nchips - powli(2,nchipbits));
   if (gap) {
      nchipbits++;
   }
   switch (gpgpu_mem_address_mask) {
   case 0: 
      //old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 10;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000000300;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x0000000000001CFF;
      break;
   case 1:
      ADDR_CHIP_S = 13;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 2:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 3:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x000000000FFFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;

   case 14:
      ADDR_CHIP_S = 14;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 15:
      ADDR_CHIP_S = 15;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 16:
      ADDR_CHIP_S = 16;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 6:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 5:
      ADDR_CHIP_S = 5;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;                         
   case 100:
      ADDR_CHIP_S = 1;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000000003;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x0000000000001FFC;
      break;
   case 103:
      ADDR_CHIP_S = 3;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000000003;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x0000000000001FFC;
      break;
   case 106:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000001800;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x00000000000007FF;
      break;
   case 160: 
      //old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK]   = 0x0000000000000300;
      addrdec_mask[ROW]  = 0x0000000007FFE000;
      addrdec_mask[COL]  = 0x0000000000001CFF;

   default:
      break;
   }

   if (addrdec_option != NULL) {
      addrdec_parseoption(addrdec_option);
   }

   if (ADDR_CHIP_S != -1) {
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      addrdec_mask[BK]   = ((addrdec_mask[BK] & ~mask) << nchipbits) | (addrdec_mask[BK] & mask);
      addrdec_mask[ROW]  = ((addrdec_mask[ROW] & ~mask) << nchipbits) | (addrdec_mask[ROW] & mask);
      addrdec_mask[COL]  = ((addrdec_mask[COL] & ~mask) << nchipbits) | (addrdec_mask[COL] & mask);

      for (i=ADDR_CHIP_S;i<(ADDR_CHIP_S+nchipbits);i++) {
         mask = (unsigned long long int)1 << i;
         addrdec_mask[CHIP] |= mask;
      }
   } else {
      // make sure nchips is power of two when explicit dram id mask is used
      assert((nchips & (nchips - 1)) == 0); 
   }

   addrdec_getmasklimit(addrdec_mask[CHIP],  &addrdec_mkhigh[CHIP],  &addrdec_mklow[CHIP] );
   addrdec_getmasklimit(addrdec_mask[BK],    &addrdec_mkhigh[BK],    &addrdec_mklow[BK]   );
   addrdec_getmasklimit(addrdec_mask[ROW],   &addrdec_mkhigh[ROW],   &addrdec_mklow[ROW]  );
   addrdec_getmasklimit(addrdec_mask[COL],   &addrdec_mkhigh[COL],   &addrdec_mklow[COL]  );
   addrdec_getmasklimit(addrdec_mask[BURST], &addrdec_mkhigh[BURST], &addrdec_mklow[BURST]);

   printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n", addrdec_mask[CHIP],  addrdec_mkhigh[CHIP],  addrdec_mklow[CHIP] );
   printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask[BK],    addrdec_mkhigh[BK],    addrdec_mklow[BK]   );
   printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask[ROW],   addrdec_mkhigh[ROW],   addrdec_mklow[ROW]  );
   printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask[COL],   addrdec_mkhigh[COL],   addrdec_mklow[COL]  );
   printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n", addrdec_mask[BURST], addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
}

#ifdef UNIT_TEST

int main () {
   unsigned int tb = 1;
   unsigned pos;
   addrdec_t_o tlx;

   printf("DRAM: %d %d %d %d %d %d %d\n", 
          D_COLL, D_BK, D_COLU, D_ROWL, D_CHIP, D_ROWU, D_UNUSED);
   for (tb=1, pos=0; tb!=0; tb <<= 1, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      tlx.plain = tb;
      addrdec_fetch(tlx);
      addrdec_dram(tlx);
      printf("\n");
   }

   unsigned long long int packed;
   packed = addrdec_packbits(0xFFFF0000FFFF0000, 0x2244113322441133);
   assert (packed == 0x22442244);
   printf("%016llx\n", packed);

   packed = addrdec_packbits(0x5555555555555555, 0x3333333333333333);
   assert (packed == 0x55555555);
   printf("%016llx\n", packed);

   packed = addrdec_packbits(0x5555555555555555, 0x6363636363636363);
   assert (packed == 0x99999999);
   printf("%016llx\n", packed);

   addrdec_t tls;
   for (tb=1, pos=0; tb!=0; tb <<= 1, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      addrdec_tlx(tb, &tls);
      addrdec_display(&tls);
      printf("\n");
   }

   addrdec_setnchip(32);
   for (tb=1, pos=0; tb!=0; tb <<= 1, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      addrdec_tlx(tb, &tls);
      addrdec_display(&tls);
      printf("\n");
   }
   addrdec_setnchip(16);
   for (tb=1, pos=0; tb!=0; tb <<= 1, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      addrdec_tlx(tb, &tls);
      addrdec_display(&tls);
      printf("\n");
   }
   addrdec_setnchip(8);
   for (tb=1, pos=0; tb!=0; tb <<= 1, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      addrdec_tlx(tb, &tls);
      addrdec_display(&tls);
      printf("\n");
   }
   addrdec_setnchip(7);
   for (tb=1, pos=0; tb!=0; tb <<= 1, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      addrdec_tlx(tb, &tls);
      addrdec_display(&tls);
      printf("\n");
   }
/*   addrdec_setnchip(6);
   for(tb=1, pos=0; tb!=0; tb ++, pos++) {
      printf("%08lx|%02d =>", tb,pos);
      addrdec_tlx(tb, &tls);
      addrdec_display(&tls);
      printf("\n");
   }*/
   return 0;
}

#endif
