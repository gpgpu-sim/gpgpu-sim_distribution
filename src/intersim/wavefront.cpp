#include "booksim.hpp"
#include <iostream>

#include "wavefront.hpp"
#include "random_utils.hpp"

Wavefront::Wavefront( const Configuration &config,
                      Module *parent, const string& name,
                      int inputs, int outputs ) :
DenseAllocator( config, parent, name, inputs, outputs )
{
   // We need a square wavefront allocator, so take the max dimension
   _square = ( _inputs > _outputs ) ? _inputs : _outputs;

   // The diagonal with priority
   _pri = 0;
}

Wavefront::~Wavefront( )
{
}

void Wavefront::Allocate( )
{
   int input;
   int output;

   // Clear matching

   for ( int i = 0; i < _inputs; ++i ) {
      _inmatch[i] = -1;
   }
   for ( int j = 0; j < _outputs; ++j ) {
      _outmatch[j] = -1;
   }

   // Loop through diagonals of request matrix

   for ( int p = 0; p < _square; ++p ) {
      output = ( _pri + p ) % _square;

      // Step through the current diagonal
      for ( input = 0; input < _inputs; ++input ) {
         if ( ( output < _outputs ) && 
              ( _inmatch[input] == -1 ) && 
              ( _outmatch[output] == -1 ) &&
              ( _request[input][output].label != -1 ) ) {
            // Grant!
            _inmatch[input] = output;
            _outmatch[output] = input;
         }

         output = ( output + 1 ) % _square;
      }
   }

   // Round-robin the priority diagonal
   _pri = ( _pri + 1 ) % _square;
}


