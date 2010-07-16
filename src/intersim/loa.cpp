#include "booksim.hpp"
#include <iostream>

#include "loa.hpp"
#include "random_utils.hpp"

LOA::LOA( const Configuration &config,
          Module *parent, const string& name,
          int inputs, int input_speedup,
          int outputs, int output_speedup ) :
DenseAllocator( config, parent, name, inputs, outputs )
{
   _req    = new int [_inputs];
   _counts = new int [_outputs];

   _rptr   = new int [_inputs];
   _gptr   = new int [_outputs];
}

LOA::~LOA( )
{
   delete [] _req;
   delete [] _counts;
   delete [] _rptr;
   delete [] _gptr;
}

void LOA::Allocate( )
{
   int input;
   int output;

   int input_offset;
   int output_offset;

   int lonely;
   int lonely_cnt;

   // Clear matching

   // Count phase --- the number of requests
   // per output is counted

   for ( int i = 0; i < _inputs; ++i ) {
      _inmatch[i] = -1;
   }
   for ( int j = 0; j < _outputs; ++j ) {
      _outmatch[j] = -1;
      _counts[j]   = 0;

      for ( int i = 0; i < _inputs; ++i ) {
         _counts[j] += ( _request[i][j].label != -1 ) ? 1 : 0;
      }
   }

   // Request phase
   for ( input = 0; input < _inputs; ++input ) {

      // Find the lonely output
      output_offset = _rptr[input];
      lonely        = -1;
      lonely_cnt    = _inputs + 1;

      for ( int o = 0; o < _outputs; ++o ) {
         output = ( o + output_offset ) % _outputs;

         if ( ( _request[input][output].label != -1 ) && 
              ( _counts[output] < lonely_cnt ) ) {
            lonely = output;
            lonely_cnt = _counts[output];
         }
      }

      // Request the lonely output (-1 for no request)
      _req[input] = lonely;
   }

   // Grant phase
   for ( output = 0; output < _outputs; ++output ) {
      input_offset = _gptr[output];

      for ( int i = 0; i < _inputs; ++i ) {
         input = ( i + input_offset ) % _inputs;  

         if ( _req[input] == output ) {
            // Grant!

            _inmatch[input]   = output;
            _outmatch[output] = input;

            _rptr[input] = ( _rptr[input] + 1 ) % _outputs;
            _gptr[output] = ( _gptr[output] + 1 ) % _inputs;

            break;
         }
      }
   }


}


