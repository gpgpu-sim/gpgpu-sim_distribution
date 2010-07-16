#include "booksim.hpp"
#include <iostream>

#include "selalloc.hpp"
#include "random_utils.hpp"

//#define DEBUG_SELALLOC

SelAlloc::SelAlloc( const Configuration &config,
                    Module *parent, const string& name,
                    int inputs, int outputs ) :
SparseAllocator( config, parent, name, inputs, outputs )
{
   _iter = config.GetInt( "alloc_iters" );

   _grants = new int [_outputs];
   _gptrs  = new int [_outputs];
   _aptrs  = new int [_inputs];

   for ( int i = 0; i < _inputs; ++i ) {
      _aptrs[i] = 0;
   }
   for ( int j = 0; j < _outputs; ++j ) {
      _gptrs[j] = 0;
   }
}

SelAlloc::~SelAlloc( )
{
   delete [] _grants;
   delete [] _aptrs;
   delete [] _gptrs;
}

void SelAlloc::Allocate( )
{
   int input;
   int output;

   int input_offset;
   int output_offset;

   list<sRequest>::iterator p;
   list<int>::iterator outer_iter;
   bool wrapped;

   int max_index;
   int max_pri;

   _ClearMatching( );

   for ( int i = 0; i < _outputs; ++i ) {
      _grants[i] = -1;
   }

   for ( int iter = 0; iter < _iter; ++iter ) {
      // Grant phase

      for ( outer_iter = _out_occ.begin( ); 
          outer_iter != _out_occ.end( ); ++outer_iter ) {
         output = *outer_iter;

         // Skip loop if there are no requests
         // or the output is already matched or
         // the output is masked
         if ( ( _out_req[output].empty( ) ) ||
              ( _outmatch[output] != -1 ) ||
              ( _outmask[output] != 0 ) ) {
            continue;
         }

         // A round-robin arbiter between input requests
         input_offset = _gptrs[output];

         p = _out_req[output].begin( );
         while ( ( p != _out_req[output].end( ) ) &&
                 ( p->port < input_offset ) ) {
            p++;
         }

         max_index = -1;
         max_pri   = 0;

         wrapped = false;
         while ( (!wrapped) || ( p->port < input_offset ) ) {
            if ( p == _out_req[output].end( ) ) {
               if ( wrapped ) {
                  break;
               }
               // p is valid here because empty lists
               // are skipped (above)
               p = _out_req[output].begin( );
               wrapped = true;
            }

            input = p->port;

            // we know the output is free (above) and
            // if the input is free, check if request is the
            // highest priority so far
            if ( ( _inmatch[input] == -1 ) &&
                 ( ( p->out_pri > max_pri ) || ( max_index == -1 ) ) ) {
               max_pri   = p->out_pri;
               max_index = input;
            }

            p++;
         }   

         if ( max_index != -1 ) { // grant
            _grants[output] = max_index;
         }
      }

#ifdef DEBUG_SELALLOC
      cout << "grants: ";
      for ( int i = 0; i < _outputs; ++i ) {
         cout << _grants[i] << " ";
      }
      cout << endl;

      cout << "aptrs: ";
      for ( int i = 0; i < _inputs; ++i ) {
         cout << _aptrs[i] << " ";
      }
      cout << endl;
#endif 

      // Accept phase

      for ( outer_iter = _in_occ.begin( ); 
          outer_iter != _in_occ.end( ); ++outer_iter ) {
         input = *outer_iter;

         if ( _in_req[input].empty( ) ) {
            continue;
         }

         // A round-robin arbiter between output grants
         output_offset = _aptrs[input];

         p = _in_req[input].begin( );
         while ( ( p != _in_req[input].end( ) ) &&
                 ( p->port < output_offset ) ) {
            p++;
         }

         max_index = -1;
         max_pri   = 0;

         wrapped = false;
         while ( (!wrapped) || ( p->port < output_offset ) ) {
            if ( p == _in_req[input].end( ) ) {
               if ( wrapped ) {
                  break;
               }
               // p is valid here because empty lists
               // are skipped (above)
               p = _in_req[input].begin( );
               wrapped = true;
            }

            output = p->port;

            // we know the output is free (above) and
            // if the input is free, check if the highest
            // priroity
            if ( ( _grants[output] == input ) && 
                 ( !_out_req[output].empty( ) ) &&
                 ( ( p->in_pri > max_pri ) || ( max_index == -1 ) ) ) {
               max_pri   = p->in_pri;
               max_index = output;
            }

            p++;
         } 

         if ( max_index != -1 ) {
            // Accept
            output = max_index;

            _inmatch[input]   = output;
            _outmatch[output] = input;

            // Only update pointers if accepted during the 1st iteration
            if ( iter == 0 ) {
               _gptrs[output] = ( input + 1 ) % _inputs;
               _aptrs[input]  = ( output + 1 ) % _outputs;
            }
         }
      }
   }

#ifdef DEBUG_SELALLOC
   cout << "input match: ";
   for ( int i = 0; i < _inputs; ++i ) {
      cout << _inmatch[i] << " ";
   }
   cout << endl;

   cout << "output match: ";
   for ( int j = 0; j < _outputs; ++j ) {
      cout << _outmatch[j] << " ";
   }
   cout << endl;
#endif 
}
