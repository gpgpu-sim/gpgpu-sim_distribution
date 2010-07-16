#include "rng.hpp"

#define main rng_double_main
#include "rng_double.cpp"

double ranf_next( )
{
   return ranf_arr_next( );
}
