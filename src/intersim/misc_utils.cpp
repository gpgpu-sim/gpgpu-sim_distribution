#include "booksim.hpp"
#include "misc_utils.hpp"

int powi( int x, int y ) // compute x to the y
{
   int r = 1;

   for ( int i = 0; i < y; ++i ) {
      r *= x;
   }

   return r;
}

int log_two( int x )
{
   int r = 0;

   x >>= 1;
   while ( x ) {
      r++; x >>= 1;
   }

   return r;
}
