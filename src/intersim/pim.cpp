#include "booksim.hpp"
#include <iostream>

#include "pim.hpp"
#include "random_utils.hpp"

//#define DEBUG_PIM

PIM::PIM( const Configuration &config,
          Module *parent, const string& name,
          int inputs, int outputs ) :
DenseAllocator( config, parent, name, inputs, outputs )
{
   _PIM_iter = config.GetInt( "alloc_iters" );

   _grants = new int [_outputs];
}

PIM::~PIM( )
{
   delete [] _grants;
}

void PIM::Allocate( )
{
   int input;
   int output;

   int input_offset;
   int output_offset;

   _ClearMatching( );

   for ( int iter = 0; iter < _PIM_iter; ++iter ) {
      // Grant phase --- outputs randomly choose
      // between one of their requests

      for ( output = 0; output < _outputs; ++output ) {
         _grants[output] = -1;

         // A random arbiter between input requests
         input_offset  = RandomInt( _inputs - 1 );

         for ( int i = 0; i < _inputs; ++i ) {
            input = ( i + input_offset ) % _inputs;  

            if ( ( _request[input][output].label != -1 ) && 
                 ( _inmatch[input] == -1 ) &&
                 ( _outmatch[output] == -1 ) ) {

               // Grant
               _grants[output] = input;
               break;
            }
         }
      }

      // Accept phase -- inputs randomly choose
      // between input_speedup of their grants

      for ( input = 0; input < _inputs; ++input ) {

         // A random arbiter between output grants
         output_offset  = RandomInt( _outputs - 1 );

         for ( int o = 0; o < _outputs; ++o ) {
            output = ( o + output_offset ) % _outputs;

            if ( _grants[output] == input ) {

               // Accept
               _inmatch[input]   = output;
               _outmatch[output] = input;

               break;
            }
         }
      }
   }

#ifdef DEBUG_PIM
   if ( _outputs == 8 ) {
      cout << "input match: " << endl;
      for ( int i = 0; i < _inputs; ++i ) {
         cout << "  from " << i << " to " << _inmatch[i] << endl;
      }
      cout << endl;
   }

   cout << "output match: ";
   for ( int j = 0; j < _outputs; ++j ) {
      cout << _outmatch[j] << " ";
   }
   cout << endl;
#endif
}


