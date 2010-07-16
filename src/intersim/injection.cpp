#include "booksim.hpp"
#include <map>
#include <assert.h>
#include <cstdlib>
#include "injection.hpp"
#include "network.hpp"
#include "random_utils.hpp"
#include "misc_utils.hpp"

map<string, tInjectionProcess> gInjectionProcessMap;

double gBurstAlpha;
double gBurstBeta;

int    gConstPacketSize;

int *gNodeStates = 0;
//=============================================================

int bernoulli( int /*source*/, double rate )
{
   return( RandomFloat( ) < ( rate / (double)gConstPacketSize ) ) ? 
   gConstPacketSize : 0;
}

//=============================================================

int on_off( int source, double rate )
{
   double r1;
   bool issue;

   assert( ( source >= 0 ) && ( source < gNodes ) );

   if ( !gNodeStates ) {
      gNodeStates = new int [gNodes];

      for ( int n = 0; n < gNodes; ++n ) {
         gNodeStates[n] = 0;
      }
   }

   // advance state

   if ( gNodeStates[source] == 0 ) {
      if ( RandomFloat( ) < gBurstAlpha ) { // from off to on
         gNodeStates[source] = 1;
      }
   } else if ( RandomFloat( ) < gBurstBeta ) { // from on to off
      gNodeStates[source] = 0;
   }

   // generate packet

   issue = false;
   if ( gNodeStates[source] ) { // on?
      r1 = rate * ( 1.0 + gBurstBeta / gBurstAlpha ) / 
           (double)gConstPacketSize;

      if ( RandomFloat( ) < r1 ) {
         issue = true;
      }
   }

   return issue ? gConstPacketSize : 0;
}

//=============================================================

void InitializeInjectionMap( )
{
   /* Register injection processes functions here */

   gInjectionProcessMap["bernoulli"] = &bernoulli;
   gInjectionProcessMap["on_off"]    = &on_off;
}

tInjectionProcess GetInjectionProcess( const Configuration& config )
{
   map<string, tInjectionProcess>::const_iterator match;
   tInjectionProcess ip;

   string fn;

   config.GetStr( "injection_process", fn );
   match = gInjectionProcessMap.find( fn );

   if ( match != gInjectionProcessMap.end( ) ) {
      ip = match->second;
   } else {
      cout << "Error: Undefined injection process '" << fn << "'." << endl;
      exit(-1);
   }

   gConstPacketSize = config.GetInt( "const_flits_per_packet" );
   gBurstAlpha      = config.GetFloat( "burst_alpha" );
   gBurstBeta       = config.GetFloat( "burst_beta" );

   return ip;
}
