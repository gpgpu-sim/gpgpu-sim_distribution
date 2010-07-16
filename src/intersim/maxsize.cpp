#include "booksim.hpp"
#include <iostream>

#include "maxsize.hpp"

// shortest augmenting path:
//
// for all unmatched left nodes,
//    push node onto work stack
// end
//
// for all j,
//   from[j] = undefined
// end
//
// do,
//
//   while( !stack.empty ),
//     
//     nl = stack.pop
//     for each edge (nl,j),
//       if ( ( lmatch[nl] != j ) && ( from[j] == undefined ) ),
//         if ( rmatch[j] == undefined ),
//           stop // augmenting path found
//         else
//           from[j] = nl
//           newstack.push( rmatch[j] ) 
//         end
//       end
//     end
//   end
//
//   stack = newstack
// end
//

//#define DEBUG_MAXSIZE
//#define PRINT_MATCHING

MaxSizeMatch::MaxSizeMatch( const Configuration &config,
                            Module *parent, const string& name,
                            int inputs, int outputs ) :
DenseAllocator( config, parent, name, inputs, outputs )
{
   _from = new int [_outputs];
   _s    = new int [_inputs];
   _ns   = new int [_inputs];
}

MaxSizeMatch::~MaxSizeMatch( )
{
   delete [] _from;
   delete [] _s;
   delete [] _ns;
}

void MaxSizeMatch::Allocate( )
{
   // clear matching
   for ( int i = 0; i < _inputs; ++i ) {
      _inmatch[i] = -1;
   }
   for ( int j = 0; j < _outputs; ++j ) {
      _outmatch[j] = -1;
   }

   // augment as many times as possible 
   // (this is an O(N^3) maximum-size matching algorithm)
   while ( _ShortestAugmenting( ) );
}


bool MaxSizeMatch::_ShortestAugmenting( )
{
   int i, j, jn;
   int slen, nslen;
   int *t;

   slen = 0;
   for ( i = 0; i < _inputs; ++i ) {
      if ( _inmatch[i] == -1 ) { // start with unmatched left nodes
         _s[slen] = i;
         slen++;
      }
      _from[i] = -1;
   }

   for ( int iter = 0; iter < _inputs; iter++ ) {
      nslen = 0;

      for ( int e = 0; e < slen; ++e ) {
         i = _s[e];

         for ( j = 0; j < _outputs; ++j ) {
            if ( ( _request[i][j].label != -1 ) && // edge (i,j) exists
                 ( _inmatch[i] != j ) &&     // (i,j) is not contained in the current matching
                 ( _from[j] == -1 ) ) {      // no shorter path to j exists

               _from[j] = i;                  // how did we get to j?

#ifdef DEBUG_MAXSIZE
               cout << "  got to " << j << " from " << i << endl;
#endif
               if ( _outmatch[j] == -1 ) {   // j is unmatched -- augmenting path found
                  goto found_augmenting;
               } else {                      // j is matched
                  _ns[nslen] = _outmatch[j];  // add the destination of this edge to the leaf nodes
                  nslen++;

#ifdef DEBUG_MAXSIZE
                  cout << "  adding " << _outmatch[j] << endl;
#endif
               }
            }
         }
      }

      // no augmenting path found yet, swap stacks
      t = _s;
      _s = _ns;
      _ns = t;
      slen = nslen;
   }

   return false; // no augmenting paths

   found_augmenting:

   // the augmenting path ends at node j on the right

#ifdef DEBUG_MAXSIZE
   cout << "Found path: " << j << "c <- ";
#endif

   i = _from[j];
   _outmatch[j] = i;

#ifdef DEBUG_MAXSIZE
   cout << i;
#endif

   while ( _inmatch[i] != -1 ) {  // loop until the end of the path
      jn = _inmatch[i];            // remove previous edge (i,jn) and add (i,j)
      _inmatch[i] = j;

#ifdef DEBUG_MAXSIZE
      cout << " <- " << j << "c <- ";
#endif

      j = jn;                    // add edge from (jn,in)
      i = _from[j];
      _outmatch[j] = i; 

#ifdef DEBUG_MAXSIZE
      cout << i;
#endif
   }

#ifdef DEBUG_MAXSIZE
   cout << endl;
#endif

   _inmatch[i] = j;

#ifdef PRINT_MATCHING
   cout << "left  matching: ";

   for ( i = 0; i < _inputs; i++ ) {
      cout << _inmatch[i] << " ";
   }
   cout << endl;

   cout << "right matching: ";
   for ( i = 0; i < _outputs; i++ ) {
      cout << _outmatch[i] << " ";
   }
   cout << endl;
#endif

   return true;
}
