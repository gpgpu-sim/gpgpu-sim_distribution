#include "booksim.hpp"

#include <map>
#include <stdlib.h>
#include <assert.h>

#include "routefunc.hpp"
#include "kncube.hpp"
#include "random_utils.hpp"

map<string, tRoutingFunction> gRoutingFunctionMap;

/* Global information used by routing functions */

int gNumVCS;

/* Add routing functions here */

//=============================================================

void singlerf( const Router *, const Flit *f, int, OutputSet *outputs, bool inject )
{
   outputs->Clear( );
   outputs->Add( f->dest, f->dest % gNumVCS ); // VOQing
}

//=============================================================

int dor_next_mesh( int cur, int dest )
{
   int dim_left;
   int out_port;

   for ( dim_left = 0; dim_left < gN; ++dim_left ) {
      if ( ( cur % gK ) != ( dest % gK ) ) {
         break;
      }
      cur /= gK; dest /= gK;
   }

   if ( dim_left < gN ) {
      cur %= gK; dest %= gK;

      if ( cur < dest ) {
         out_port = 2*dim_left;     // Right
      } else {
         out_port = 2*dim_left + 1; // Left
      }
   } else {
      out_port = 2*gN;  // Eject
   }

   return out_port;
}

//=============================================================

void dor_next_torus( int cur, int dest, int in_port,
                     int *out_port, int *partition,
                     bool balance = false )
{
   int dim_left;
   int dir;
   int dist2;

   for ( dim_left = 0; dim_left < gN; ++dim_left ) {
      if ( ( cur % gK ) != ( dest % gK ) ) {
         break;
      }
      cur /= gK; dest /= gK;
   }

   if ( dim_left < gN ) {

      if ( (in_port/2) != dim_left ) {
         // Turning into a new dimension

         cur %= gK; dest %= gK;
         dist2 = gK - 2 * ( ( dest - cur + gK ) % gK );

         if ( ( dist2 > 0 ) || 
              ( ( dist2 == 0 ) && ( RandomInt( 1 ) ) ) ) {
            *out_port = 2*dim_left;     // Right
            dir = 0;
         } else {
            *out_port = 2*dim_left + 1; // Left
            dir = 1;
         }

         if ( balance ) {
            // Cray's "Partition" allocation
            // Two datelines: one between k-1 and 0 which forces VC 1
            //                another between ((k-1)/2) and ((k-1)/2 + 1) which forces VC 0
            //                otherwise any VC can be used

            if ( ( ( dir == 0 ) && ( cur > dest ) ) ||
                 ( ( dir == 1 ) && ( cur < dest ) ) ) {
               *partition = 1;
            } else if ( ( ( dir == 0 ) && ( cur <= (gK-1)/2 ) && ( dest >  (gK-1)/2 ) ) ||
                        ( ( dir == 1 ) && ( cur >  (gK-1)/2 ) && ( dest <= (gK-1)/2 ) ) ) {
               *partition = 0;
            } else {
               *partition = RandomInt( 1 ); // use either VC set
            }
         } else {
            // Deterministic, fixed dateline between nodes k-1 and 0

            if ( ( ( dir == 0 ) && ( cur > dest ) ) ||
                 ( ( dir == 1 ) && ( dest < cur ) ) ) {
               *partition = 1;
            } else {
               *partition = 0;
            }
         }
      } else {
         // Inverting the least significant bit keeps
         // the packet moving in the same direction
         *out_port = in_port ^ 0x1;
      }    

   } else {
      *out_port = 2*gN;  // Eject
   }
}

//=============================================================

void dim_order_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;

   outputs->Clear( );

   if ( inject ) { // use any VC for injection
      outputs->AddRange( 0, 0, gNumVCS - 1 );
   } else {
      out_port = dor_next_mesh( r->GetID( ), f->dest );

      if ( f->watch ) {
         cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
         << f->dest << " using channel " << out_port << ", vc range = [" 
         << 0 << "," << gNumVCS - 1 << "] (in_channel is " << in_channel << ")" << endl;
      }

      outputs->AddRange( out_port, 0, gNumVCS - 1 );
   }
}

//=============================================================

void dim_order_ni_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;
   int vcs_per_dest = gNumVCS / gNodes;

   outputs->Clear( );
   out_port = dor_next_mesh( r->GetID( ), f->dest );

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << f->dest*vcs_per_dest << "," << (f->dest+1)*vcs_per_dest - 1  
      << "] (in_channel is " << in_channel << ")" << endl;
   }

   outputs->AddRange( out_port, f->dest*vcs_per_dest, (f->dest+1)*vcs_per_dest - 1 );
}

//=============================================================

// Random intermediate in the minimal quadrant defined
// by the source and destination
int rand_min_intr_mesh( int src, int dest )
{
   int dist;

   int intm = 0;
   int offset = 1;

   for ( int n = 0; n < gN; ++n ) {
      dist = ( dest % gK ) - ( src % gK );

      if ( dist > 0 ) {
         intm += offset * ( ( src % gK ) + RandomInt( dist ) );
      } else {
         intm += offset * ( ( dest % gK ) + RandomInt( -dist ) );
      }

      offset *= gK;
      dest /= gK; src /= gK;
   }

   return intm;
}

//=============================================================

void romm_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;
   int vc_min, vc_max;

   outputs->Clear( );

   if ( in_channel == 2*gN ) {
      f->ph   = 1;  // Phase 1
      f->intm = rand_min_intr_mesh( f->src, f->dest );
   }

   if ( ( f->ph == 1 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 2; // Go to phase 2
   }

   if ( f->ph == 1 ) { // In phase 1
      out_port = dor_next_mesh( r->GetID( ), f->intm );
      vc_min = 0;
      vc_max = gNumVCS/2 - 1; 
   } else { // In phase 2
      out_port = dor_next_mesh( r->GetID( ), f->dest );
      vc_min = gNumVCS/2;
      vc_max = gNumVCS - 1;
   }

   outputs->AddRange( out_port, vc_min, vc_max );
}

//=============================================================

void romm_ni_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;
   int vcs_per_dest = gNumVCS / gNodes;

   outputs->Clear( );

   if ( in_channel == 2*gN ) {
      f->ph   = 1;  // Phase 1
      f->intm = rand_min_intr_mesh( f->src, f->dest );
   }

   if ( ( f->ph == 1 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 2; // Go to phase 2
   }

   if ( f->ph == 1 ) { // In phase 1
      out_port = dor_next_mesh( r->GetID( ), f->intm );
   } else { // In phase 2
      out_port = dor_next_mesh( r->GetID( ), f->dest );
   }

   outputs->AddRange( out_port, f->dest*vcs_per_dest, (f->dest+1)*vcs_per_dest - 1 );
}

//=============================================================

void min_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;
   int cur, dest;
   int in_vc;

   outputs->Clear( );

   if ( in_channel == 2*gN ) {
      in_vc = gNumVCS - 1; // ignore the injection VC
   } else {
      in_vc = f->vc;
   }

   // DOR for the escape channel (VC 0), low priority 
   out_port = dor_next_mesh( r->GetID( ), f->dest );    
   outputs->AddRange( out_port, 0, 0, 0 );

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << 0 << "," << gNumVCS - 1 << "] (in_channel is " << in_channel << ")" << endl;
   }

   if ( in_vc != 0 ) { // If not in the escape VC
      // Minimal adaptive for all other channels
      cur = r->GetID( ); dest = f->dest;

      for ( int n = 0; n < gN; ++n ) {
         if ( ( cur % gK ) != ( dest % gK ) ) {
            // Add minimal direction in dimension 'n'
            if ( ( cur % gK ) < ( dest % gK ) ) { // Right
               outputs->AddRange( 2*n, 1, gNumVCS - 1, 1 ); 
            } else { // Left
               outputs->AddRange( 2*n + 1, 1, gNumVCS - 1, 1 ); 
            }
         }
         cur  /= gK;
         dest /= gK;
      }
   }
}

//=============================================================

void planar_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int cur, dest;
   int vc_mult;
   int vc_min, vc_max;
   int d1_min_c;
   int in_vc;
   int n;

   bool increase;
   bool fault;
   bool atedge;

   outputs->Clear( );

   cur     = r->GetID( ); 
   dest    = f->dest;
   in_vc   = f->vc;
   vc_mult = gNumVCS / 3;

   if ( cur != dest ) {

      // Find the first unmatched dimension -- except
      // for when we're in the first dimension because
      // of misrouting in the last adaptive plane.
      // In this case, go to the last dimension instead.

      for ( n = 0; n < gN; ++n ) {
         if ( ( ( cur % gK ) != ( dest % gK ) ) &&
              !( ( in_channel/2 == 0 ) &&
                 ( n == 0 ) &&
                 ( in_vc < 2*vc_mult ) ) ) {
            break;
         }

         cur  /= gK;
         dest /= gK;
      }

      assert( n < gN );

      if ( f->watch ) {
         cout << "PLANAR ADAPTIVE: flit " << f->id 
         << " in adaptive plane " << n << " at " << r->GetID( ) << endl;
      }

      // We're in adaptive plane n

      // Can route productively in d_{i,2}
      if ( ( cur % gK ) < ( dest % gK ) ) { // Increasing
         increase = true;
         if ( !r->IsFaultyOutput( 2*n ) ) {
            outputs->AddRange( 2*n, 2*vc_mult, gNumVCS - 1 );
            fault = false;

            if ( f->watch ) {
               cout << "PLANAR ADAPTIVE: increasing in dimension " << n << endl;
            }
         } else {
            fault = true;
         }
      } else { // Decreasing
         increase = false;
         if ( !r->IsFaultyOutput( 2*n + 1 ) ) {
            outputs->AddRange( 2*n + 1, 2*vc_mult, gNumVCS - 1 ); 
            fault = false;

            if ( f->watch ) {
               cout << "PLANAR ADAPTIVE: decreasing in dimension " << n << endl;
            }
         } else {
            fault = true;
         }
      }

      n = ( n + 1 ) % gN;
      cur  /= gK;
      dest /= gK;

      if ( increase ) {
         vc_min = 0;
         vc_max = vc_mult - 1;
      } else {
         vc_min = vc_mult;
         vc_max = 2*vc_mult - 1;
      }

      if ( ( cur % gK ) < ( dest % gK ) ) { // Increasing in d_{i+1}
         d1_min_c = 2*n;
      } else if ( ( cur % gK ) != ( dest % gK ) ) {  // Decreasing in d_{i+1}
         d1_min_c = 2*n + 1;
      } else {
         d1_min_c = -1;
      }

      // do we want to 180?  if so, the last
      // route was a misroute in this dimension,
      // if there is no fault in d_i, just ignore
      // this dimension, otherwise continue to misroute
      if ( d1_min_c == in_channel ) {
         if ( fault ) {
            d1_min_c = in_channel ^ 1;
         } else {
            d1_min_c = -1;
         }

         if ( f->watch ) {
            cout << "PLANAR ADAPTIVE: avoiding 180 in dimension " << n << endl;
         }
      }

      if ( d1_min_c != -1 ) {
         if ( !r->IsFaultyOutput( d1_min_c ) ) {
            outputs->AddRange( d1_min_c, vc_min, vc_max );
         } else if ( fault ) {
            // major problem ... fault in d_i and d_{i+1}
            r->Error( "There seem to be faults in d_i and d_{i+1}" );
         }
      } else if ( fault ) { // need to misroute!
         if ( cur % gK == 0 ) {
            d1_min_c = 2*n;
            atedge = true;
         } else if ( cur % gK == gK - 1 ) {
            d1_min_c = 2*n + 1;
            atedge = true;
         } else {
            d1_min_c = 2*n + RandomInt( 1 ); // random misroute

            if ( d1_min_c  == in_channel ) { // don't 180
               d1_min_c = in_channel ^ 1;
            }
            atedge = false;
         }

         if ( !r->IsFaultyOutput( d1_min_c ) ) {
            outputs->AddRange( d1_min_c, vc_min, vc_max );
         } else if ( !atedge && !r->IsFaultyOutput( d1_min_c ^ 1 ) ) {
            outputs->AddRange( d1_min_c ^ 1, vc_min, vc_max );
         } else {
            // major problem ... fault in d_i and d_{i+1}
            r->Error( "There seem to be faults in d_i and d_{i+1}" );
         }
      }
   } else {
      outputs->AddRange( 2*gN, 0, gNumVCS - 1 ); 
   }
}

//=============================================================

void limited_adapt_mesh_old( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int in_vc;
   int in_dim;

   int min_port;

   bool dor_dim;
   bool equal;

   int cur, dest;

   outputs->Clear( );

   if ( inject ) {
      outputs->AddRange( 0, 0, gNumVCS - 1 );
      f->ph = 0; // zero dimension reversals
   } else {

      cur = r->GetID( ); dest = f->dest;
      if ( cur != dest ) {

         if ( f->ph == 0 ) {
            f->ph = 1;

            in_vc  = 0;
            in_dim = 0;
         } else {
            in_vc  = f->vc;
            in_dim = in_channel/2;
         }

         // The first remaining is the DOR escape path
         dor_dim = true;

         for ( int n = 0; n < gN; ++n ) {
            if ( ( cur % gK ) != ( dest % gK ) ) {
               if ( ( cur % gK ) < ( dest % gK ) ) {
                  min_port = 2*n; // Right
               } else {
                  min_port = 2*n + 1; // Left
               }

               if ( dor_dim ) {
                  // Low priority escape path
                  outputs->AddRange( min_port, gNumVCS - 1, gNumVCS - 1, 0 ); 
                  dor_dim = false;
               }

               equal = false;
            } else {
               equal = true;
               min_port = 2*n;
            }

            if ( in_vc < gNumVCS - 1 ) {  // adaptive VC's left?
               if ( n < in_dim ) {
                  // Productive (minimal) direction, with reversal
                  if ( in_vc == gNumVCS - 2 ) {
                     outputs->AddRange( min_port, in_vc + 1, in_vc + 1, equal ? 1 : 2 ); 
                  } else {
                     outputs->AddRange( min_port, in_vc + 1, gNumVCS - 2, equal ? 1 : 2 ); 
                  }

                  // Unproductive (non-minimal) direction, with reversal
                  if ( in_vc <  gNumVCS - 2 ) {
                     if ( in_vc == gNumVCS - 3 ) {
                        outputs->AddRange( min_port ^ 0x1, in_vc + 1, in_vc + 1, 1 );
                     } else {
                        outputs->AddRange( min_port ^ 0x1, in_vc + 1, gNumVCS - 3, 1 );
                     }
                  }
               } else if ( n == in_dim ) {
                  if ( !equal ) {
                     // Productive (minimal) direction, no reversal
                     outputs->AddRange( min_port, in_vc, gNumVCS - 2, 4 ); 
                  }
               } else {
                  // Productive (minimal) direction, no reversal
                  outputs->AddRange( min_port, in_vc, gNumVCS - 2, equal ? 1 : 3 ); 
                  // Unproductive (non-minimal) direction, no reversal
                  if ( in_vc <  gNumVCS - 2 ) {
                     outputs->AddRange( min_port ^ 0x1, in_vc, gNumVCS - 2, 1 );
                  }
               }
            }

            cur  /= gK;
            dest /= gK;
         }
      } else { // at destination
         outputs->AddRange( 2*gN, 0, gNumVCS - 1 ); 
      }
   } 
}

void limited_adapt_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int min_port;

   int cur, dest;

   outputs->Clear( );

   if ( inject ) {
      outputs->AddRange( 0, 0, gNumVCS - 2 );
      f->dr = 0; // zero dimension reversals
   } else {
      cur = r->GetID( ); dest = f->dest;

      if ( cur != dest ) {
         if ( ( f->vc != gNumVCS - 1 ) && 
              ( f->dr != gNumVCS - 2 ) ) {

            for ( int n = 0; n < gN; ++n ) {
               if ( ( cur % gK ) != ( dest % gK ) ) {
                  if ( ( cur % gK ) < ( dest % gK ) ) {
                     min_port = 2*n; // Right
                  } else {
                     min_port = 2*n + 1; // Left
                  }

                  // Go in a productive direction with high priority
                  outputs->AddRange( min_port, 0, gNumVCS - 2, 2 );

                  // Go in the non-productive direction with low priority
                  outputs->AddRange( min_port ^ 0x1, 0, gNumVCS - 2, 1 );
               } else {
                  // Both directions are non-productive
                  outputs->AddRange( 2*n, 0, gNumVCS - 2, 1 );
                  outputs->AddRange( 2*n+1, 0, gNumVCS - 2, 1 );
               }

               cur  /= gK;
               dest /= gK;
            }

         } else {
            outputs->AddRange( dor_next_mesh( cur, dest ),
                               gNumVCS - 1, gNumVCS - 1, 0 );
         }

      } else { // at destination
         outputs->AddRange( 2*gN, 0, gNumVCS - 1 ); 
      }
   } 
}

//=============================================================

void valiant_mesh( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;
   int vc_min, vc_max;

   outputs->Clear( );


   if ( in_channel == 2*gN ) {
      f->ph   = 1;  // Phase 1
      f->intm = RandomInt( gNodes - 1 );
   }

   if ( ( f->ph == 1 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 2; // Go to phase 2
   }

   if ( f->ph == 1 ) { // In phase 1
      out_port = dor_next_mesh( r->GetID( ), f->intm );
      vc_min = 0;
      vc_max = gNumVCS/2 - 1; 
   } else { // In phase 2
      out_port = dor_next_mesh( r->GetID( ), f->dest );
      vc_min = gNumVCS/2;
      vc_max = gNumVCS - 1;
   }

   outputs->AddRange( out_port, vc_min, vc_max );
}

//=============================================================

void valiant_torus( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int out_port;
   int vc_min, vc_max;

   outputs->Clear( );

   if ( in_channel == 2*gN ) {
      f->ph   = 1;  // Phase 1
      f->intm = RandomInt( gNodes - 1 );
   }

   if ( ( f->ph == 1 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 2; // Go to phase 2
      in_channel = 2*gN; // ensures correct vc selection at the beginning of phase 2
   }

   if ( f->ph == 1 ) { // In phase 1
      dor_next_torus( r->GetID( ), f->intm, in_channel,
                      &out_port, &f->ring_par, false );

      if ( f->ring_par == 0 ) {
         vc_min = 0;
         vc_max = gNumVCS/4 - 1;
      } else {
         vc_min = gNumVCS/4;
         vc_max = gNumVCS/2 - 1;
      }
   } else { // In phase 2
      dor_next_torus( r->GetID( ), f->dest, in_channel,
                      &out_port, &f->ring_par, false );

      if ( f->ring_par == 0 ) {
         vc_min = gNumVCS/2;
         vc_max = (3*gNumVCS)/4 - 1;
      } else {
         vc_min = (3*gNumVCS)/4;
         vc_max = gNumVCS - 1;
      }
   }

   outputs->AddRange( out_port, vc_min, vc_max );
}

//=============================================================

void valiant_ni_torus( const Router *r, const Flit *f, int in_channel, 
                       OutputSet *outputs, bool inject )
{
   int out_port;
   int vc_min, vc_max;

   outputs->Clear( );

   if ( in_channel == 2*gN ) {
      f->ph   = 1;  // Phase 1
      f->intm = RandomInt( gNodes - 1 );
   }

   if ( ( f->ph == 1 ) && ( r->GetID( ) == f->intm ) ) {
      f->ph = 2; // Go to phase 2
      in_channel = 2*gN; // ensures correct vc selection at the beginning of phase 2
   }

   if ( f->ph == 1 ) { // In phase 1
      dor_next_torus( r->GetID( ), f->intm, in_channel,
                      &out_port, &f->ring_par, false );

      if ( f->ring_par == 0 ) {
         vc_min = f->dest;
         vc_max = f->dest;
      } else {
         vc_min = f->dest + gNodes;
         vc_max = f->dest + gNodes;
      }

   } else { // In phase 2
      dor_next_torus( r->GetID( ), f->dest, in_channel,
                      &out_port, &f->ring_par, false );

      if ( f->ring_par == 0 ) {
         vc_min = f->dest + 2*gNodes;
         vc_max = f->dest + 2*gNodes;
      } else {
         vc_min = f->dest + 3*gNodes;
         vc_max = f->dest + 3*gNodes;
      }
   }

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << vc_min << "," << vc_max 
      << "] (in_channel is " << in_channel << ")" << endl;
   }

   outputs->AddRange( out_port, vc_min, vc_max );
}

//=============================================================

void dim_order_torus( const Router *r, const Flit *f, int in_channel, 
                      OutputSet *outputs, bool inject )
{
   int cur;
   int dest;

   int out_port;
   int vc_min, vc_max;

   outputs->Clear( );

   cur  = r->GetID( );
   dest = f->dest;

   dor_next_torus( cur, dest, in_channel,
                   &out_port, &f->ring_par, false );

   if ( f->ring_par == 0 ) {
      vc_min = 0;
      vc_max = gNumVCS/2 - 1;
   } else {
      vc_min = gNumVCS/2;
      vc_max = gNumVCS - 1;
   }

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << vc_min << "," << vc_max << "] (in_channel is " << in_channel << ")" << endl;
   }

   outputs->AddRange( out_port, vc_min, vc_max );
}

//=============================================================

void dim_order_ni_torus( const Router *r, const Flit *f, int in_channel, 
                         OutputSet *outputs, bool inject )
{
   int cur;
   int dest;

   int out_port;
   int vcs_per_dest = gNumVCS / gNodes;

   outputs->Clear( );

   cur  = r->GetID( );
   dest = f->dest;

   outputs->Clear( );
   dor_next_torus( cur, dest, in_channel,
                   &out_port, &f->ring_par, false );

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << f->dest*vcs_per_dest << "," << (f->dest+1)*vcs_per_dest - 1  
      << "] (in_channel is " << in_channel << ")" << endl;
   }

   outputs->AddRange( out_port, f->dest*vcs_per_dest, (f->dest+1)*vcs_per_dest - 1 );
}

//=============================================================

void dim_order_bal_torus( const Router *r, const Flit *f, int in_channel, 
                          OutputSet *outputs, bool inject )
{
   int cur;
   int dest;

   int out_port;
   int vc_min, vc_max;

   outputs->Clear( );

   cur  = r->GetID( );
   dest = f->dest;

   dor_next_torus( cur, dest, in_channel,
                   &out_port, &f->ring_par, true );

   if ( f->ring_par == 0 ) {
      vc_min = 0;
      vc_max = gNumVCS/2 - 1;
   } else {
      vc_min = gNumVCS/2;
      vc_max = gNumVCS - 1;
   } 

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << vc_min << "," << vc_max << "] (in_channel is " << in_channel << ")" << endl;
   }

   outputs->AddRange( out_port, vc_min, vc_max );
}

//=============================================================

void min_adapt_torus( const Router *r, const Flit *f, int in_channel, OutputSet *outputs, bool inject )
{
   int cur, dest, dist2;
   int in_vc;
   int out_port;

   outputs->Clear( );

   if ( in_channel == 2*gN ) {
      in_vc = gNumVCS - 1; // ignore the injection VC
   } else {
      in_vc = f->vc;
   }

   if ( in_vc > 1 ) { // If not in the escape VCs
      // Minimal adaptive for all other channels
      cur = r->GetID( ); dest = f->dest;

      for ( int n = 0; n < gN; ++n ) {
         if ( ( cur % gK ) != ( dest % gK ) ) {
            dist2 = gK - 2 * ( ( ( dest % gK ) - ( cur % gK ) + gK ) % gK );

            if ( dist2 > 0 ) { /*) || 
                       ( ( dist2 == 0 ) && ( RandomInt( 1 ) ) ) ) {*/
               outputs->AddRange( 2*n, 3, 3, 1 ); // Right
            } else {
               outputs->AddRange( 2*n + 1, 3, 3, 1 ); // Left
            }
         }

         cur  /= gK;
         dest /= gK;
      }

      // DOR for the escape channel (VCs 0-1), low priority --- 
      // trick the algorithm with the in channel.  want VC assignment
      // as if we had injected at this node
      dor_next_torus( r->GetID( ), f->dest, 2*gN,
                      &out_port, &f->ring_par, false );
   } else {
      // DOR for the escape channel (VCs 0-1), low priority 
      dor_next_torus( r->GetID( ), f->dest, in_channel,
                      &out_port, &f->ring_par, false );
   }

   if ( f->ring_par == 0 ) {
      outputs->AddRange( out_port, 0, 0, 0 );
   } else {
      outputs->AddRange( out_port, 1, 1, 0 );
   } 

   if ( f->watch ) {
      cout << "flit " << f->id << " (" << f << ") at " << r->GetID( ) << " destined to " 
      << f->dest << " using channel " << out_port << ", vc range = [" 
      << 0 << "," << gNumVCS - 1 << "] (in_channel is " << in_channel << ")" << endl;
   }


}

//=============================================================

void dest_tag( const Router *r, const Flit *f, int in_channel, 
               OutputSet *outputs, bool inject )
{
   outputs->Clear( );

   int stage = ( r->GetID( ) * gK ) / gNodes;
   int dest  = f->dest;

   while ( stage < ( gN - 1 ) ) {
      dest /= gK;
      ++stage;
   }

   int out_port = dest % gK;

   outputs->AddRange( out_port, 0, gNumVCS - 1 );
}

//=============================================================

void chaos_torus( const Router *r, const Flit *f, 
                  int in_channel, OutputSet *outputs, bool inject )
{
   int cur, dest;
   int dist2;

   outputs->Clear( );

   cur = r->GetID( ); dest = f->dest;

   if ( cur != dest ) {
      for ( int n = 0; n < gN; ++n ) {

         if ( ( cur % gK ) != ( dest % gK ) ) {
            dist2 = gK - 2 * ( ( ( dest % gK ) - ( cur % gK ) + gK ) % gK );

            if ( dist2 >= 0 ) {
               outputs->AddRange( 2*n, 0, 0 ); // Right
            }

            if ( dist2 <= 0 ) {
               outputs->AddRange( 2*n + 1, 0, 0 ); // Left
            }
         }

         cur  /= gK;
         dest /= gK;
      }
   } else {
      outputs->AddRange( 2*gN, 0, 0 ); 
   }
}


//=============================================================

void chaos_mesh( const Router *r, const Flit *f, 
                 int in_channel, OutputSet *outputs, bool inject )
{
   int cur, dest;

   outputs->Clear( );

   cur = r->GetID( ); dest = f->dest;

   if ( cur != dest ) {
      for ( int n = 0; n < gN; ++n ) {
         if ( ( cur % gK ) != ( dest % gK ) ) {
            // Add minimal direction in dimension 'n'
            if ( ( cur % gK ) < ( dest % gK ) ) { // Right
               outputs->AddRange( 2*n, 0, 0 ); 
            } else { // Left
               outputs->AddRange( 2*n + 1, 0, 0 ); 
            }
         }
         cur  /= gK;
         dest /= gK;
      }
   } else {
      outputs->AddRange( 2*gN, 0, 0 ); 
   }
}

//=============================================================

void InitializeRoutingMap( )
{
   /* Register routing functions here */

   gRoutingFunctionMap["single_single"]   = &singlerf;

   gRoutingFunctionMap["dim_order_mesh"]  = &dim_order_mesh;
   gRoutingFunctionMap["dim_order_ni_mesh"]  = &dim_order_ni_mesh;
   gRoutingFunctionMap["dim_order_torus"] = &dim_order_torus;
   gRoutingFunctionMap["dim_order_ni_torus"] = &dim_order_ni_torus;
   gRoutingFunctionMap["dim_order_bal_torus"] = &dim_order_bal_torus;

   gRoutingFunctionMap["romm_mesh"]       = &romm_mesh; 
   gRoutingFunctionMap["romm_ni_mesh"]    = &romm_ni_mesh;

   gRoutingFunctionMap["min_adapt_mesh"]   = &min_adapt_mesh;
   gRoutingFunctionMap["min_adapt_torus"]  = &min_adapt_torus;

   gRoutingFunctionMap["planar_adapt_mesh"] = &planar_adapt_mesh;

   gRoutingFunctionMap["limited_adapt_mesh"] = &limited_adapt_mesh;

   gRoutingFunctionMap["valiant_mesh"]  = &valiant_mesh;
   gRoutingFunctionMap["valiant_torus"] = &valiant_torus;
   gRoutingFunctionMap["valiant_ni_torus"] = &valiant_ni_torus;

   gRoutingFunctionMap["dest_tag_fly"] = &dest_tag;

   gRoutingFunctionMap["chaos_mesh"]  = &chaos_mesh;
   gRoutingFunctionMap["chaos_torus"] = &chaos_torus;
}

tRoutingFunction GetRoutingFunction( const Configuration& config )
{
   map<string, tRoutingFunction>::const_iterator match;
   tRoutingFunction rf;

   string fn, topo, fn_topo;

   gNumVCS = config.GetInt( "num_vcs" );

   config.GetStr( "topology", topo );

   config.GetStr( "routing_function", fn, "none" );
   fn_topo = fn + "_" + topo;
   match = gRoutingFunctionMap.find( fn_topo );

   if ( match != gRoutingFunctionMap.end( ) ) {
      rf = match->second;
   } else {
      if ( fn == "none" ) {
         cout << "Error: No routing function specified in configuration." << endl;
      } else {
         cout << "Error: Undefined routing function '" << fn << "' for the topology '" 
         << topo << "'." << endl;
      }
      exit(-1);
   }

   return rf;
}


