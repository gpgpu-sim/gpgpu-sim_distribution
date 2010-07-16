#include "booksim.hpp"
#include "random_utils.hpp"

void RandomSeed( long seed )
{
   ran_start( seed );
   ranf_start( seed );
}

int RandomInt( int max ) 
// Returns a random integer in the range [0,max]
{
   return( ran_next( ) % (max+1) );
}

unsigned long RandomIntLong( )
{  
   return ran_next( );
}

float RandomFloat( float max )
// Returns a random floating-point value in the rage [0,max]
{
   return( ranf_next( ) * max );
}
