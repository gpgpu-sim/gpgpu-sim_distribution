#include "booksim.hpp"
#include <map>
#include <stdlib.h>

#include "traffic.hpp"
#include "network.hpp"
#include "random_utils.hpp"
#include "misc_utils.hpp"

map<string, tTrafficFunction> gTrafficFunctionMap;

int gResetTraffic = 0;
int gStepTraffic  = 0;

void src_dest_bin( int source, int dest, int lg )
{
   int b, t;

   cout << "from: ";
   t = source;
   for ( b = 0; b < lg; ++b ) {
      cout << ( ( t >> ( lg - b - 1 ) ) & 0x1 );
   }

   cout << " to ";
   t = dest;
   for ( b = 0; b < lg; ++b ) {
      cout << ( ( t >> ( lg - b - 1 ) ) & 0x1 );
   }
   cout << endl;
}

//=============================================================

int uniform( int source, int total_nodes )
{
   return RandomInt( total_nodes - 1 );
}

//=============================================================

int bitcomp( int source, int total_nodes )
{
   int lg   = log_two( total_nodes );
   int mask = total_nodes - 1;
   int dest;

   if ( ( 1 << lg ) != total_nodes ) {
      cout << "Error: The 'bitcomp' traffic pattern requires the number of"
      << " nodes to be a power of two!" << endl;
      exit(-1);
   }

   dest = ( ~source ) & mask;

   return dest;
}

//=============================================================

int transpose( int source, int total_nodes )
{
   int lg      = log_two( total_nodes );
   int mask_lo = (1 << (lg/2)) - 1;
   int mask_hi = mask_lo << (lg/2);
   int dest;

   if ( ( ( 1 << lg ) != total_nodes ) || ( lg & 0x1 ) ) {
      cout << "Error: The 'transpose' traffic pattern requires the number of"
      << " nodes to be an even power of two!" << endl;
      exit(-1);
   }

   dest = ( ( source >> (lg/2) ) & mask_lo ) |
          ( ( source << (lg/2) ) & mask_hi );

   return dest;
}

//=============================================================

int bitrev( int source, int total_nodes )
{
   int lg = log_two( total_nodes );
   int dest;

   if ( ( 1 << lg ) != total_nodes  ) {
      cout << "Error: The 'bitrev' traffic pattern requires the number of"
      << " nodes to be a power of two!" << endl;
      exit(-1);
   }

   // If you were fancy you could do this in O(log log total_nodes)
   // instructions, but I'm not

   dest = 0;
   for ( int b = 0; b < lg; ++b  ) {
      dest |= ( ( source >> b ) & 0x1 ) << ( lg - b - 1 );
   }

   return dest;
}

//=============================================================

int shuffle( int source, int total_nodes )
{
   int lg = log_two( total_nodes );
   int dest;

   if ( ( 1 << lg ) != total_nodes  ) {
      cout << "Error: The 'shuffle' traffic pattern requires the number of"
      << " nodes to be a power of two!" << endl;
      exit(-1);
   }

   dest = ( ( source << 1 ) & ( total_nodes - 1 ) ) | 
          ( ( source >> ( lg - 1 ) ) & 0x1 );

   return dest;
}

//=============================================================

int tornado( int source, int total_nodes )
{
   int offset = 1;
   int dest = 0;

   for ( int n = 0; n < gN; ++n ) {
      dest += offset *
              ( ( ( source / offset ) % gK + ( gK/2 - 1 ) ) % gK );
      offset *= gK;
   }

   return dest;
}

//=============================================================

int neighbor( int source, int total_nodes )
{
   int offset = 1;
   int dest = 0;

   for ( int n = 0; n < gN; ++n ) {
      dest += offset *
              ( ( ( source / offset ) % gK + 1 ) % gK );
      offset *= gK;
   }

   return dest;
}

//=============================================================

int *gPerm = 0;
int gPermSeed;

void GenerateRandomPerm( int total_nodes )
{
   int ind;
   int i,j;
   int cnt;
   unsigned long prev_rand;

   prev_rand = RandomIntLong( );
   RandomSeed( gPermSeed );

   if ( !gPerm ) {
      gPerm = new int [total_nodes];
   }

   for ( i = 0; i < total_nodes; ++i ) {
      gPerm[i] = -1;
   }

   for ( i = 0; i < total_nodes; ++i ) {
      ind = RandomInt( total_nodes - 1 - i );

      j   = 0;
      cnt = 0;
      while ( ( cnt < ind ) ||
              ( gPerm[j] != -1 ) ) {
         if ( gPerm[j] == -1 ) {
            ++cnt;
         }
         ++j;

         if ( j >= total_nodes ) {
            cout << "ERROR: GenerateRandomPerm( ) internal error" << endl;
            exit(-1);
         }
      }

      gPerm[j] = i;
   }

   RandomSeed( prev_rand );
}

int randperm( int source, int total_nodes )
{
   if ( gResetTraffic || !gPerm ) {
      GenerateRandomPerm( total_nodes );
      gResetTraffic = 0;
   }

   return gPerm[source];
}

//=============================================================

int diagonal( int source, int total_nodes )
{
   int t = RandomInt( 2 );
   int d;

   // 2/3 of traffic goes from source->source
   // 1/3 of traffic goes from source->(source+1)%total_nodes

   if ( t == 0 ) {
      d = ( source + 1 ) % total_nodes;
   } else {
      d = source;
   }

   return d;
}

//=============================================================

int asymmetric( int source, int total_nodes )
{
   int d;
   int half = total_nodes / 2;

   d = ( source % half ) + RandomInt( 1 ) * half;

   return d;
}

//=============================================================

void InitializeTrafficMap( )
{
   /* Register Traffic functions here */

   gTrafficFunctionMap["uniform"]  = &uniform;

   // "Bit" patterns

   gTrafficFunctionMap["bitcomp"]   = &bitcomp;
   gTrafficFunctionMap["bitrev"]    = &bitrev;
   gTrafficFunctionMap["transpose"] = &transpose;
   gTrafficFunctionMap["shuffle"]   = &shuffle;

   // "Digit" patterns

   gTrafficFunctionMap["tornado"]   = &tornado;
   gTrafficFunctionMap["neighbor"]  = &neighbor;

   // Other patterns

   gTrafficFunctionMap["randperm"]   = &randperm;

   gTrafficFunctionMap["diagonal"]   = &diagonal;
   gTrafficFunctionMap["asymmetric"] = &asymmetric;
}

void ResetTrafficFunction( )
{
   gResetTraffic++;
}

void StepTrafficFunction( )
{
   gStepTraffic++;
}

tTrafficFunction GetTrafficFunction( const Configuration& config )
{
   map<string, tTrafficFunction>::const_iterator match;
   tTrafficFunction tf;

   string fn;

   config.GetStr( "traffic", fn, "none" );
   match = gTrafficFunctionMap.find( fn );

   if ( match != gTrafficFunctionMap.end( ) ) {
      tf = match->second;
   } else {
      cout << "Error: Undefined traffic pattern '" << fn << "'." << endl;
      exit(-1);
   }

   gPermSeed = config.GetInt( "perm_seed" );

   return tf;
}


