#include "booksim.hpp"
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "stats.hpp"

Stats::Stats( Module *parent, const string &name,
              double bin_size, int num_bins ) :
Module( parent, name ),
_num_bins( num_bins ), _bin_size( bin_size )
{
   _hist = new int [_num_bins];

   Clear( );
}

Stats::~Stats( )
{
   delete [] _hist;
}

void Stats::Clear( )
{
   _num_samples = 0;
   _sample_sum  = 0.0;

   for ( int b = 0; b < _num_bins; ++b ) {
      _hist[b] = 0;
   }

   _reset = true;
}

double Stats::Average( ) const
{
   return _sample_sum / (double)_num_samples;
}

double Stats::Min( ) const
{
   return _min;
}

double Stats::Max( ) const
{
   return _max;
}

int Stats::NumSamples( ) const
{
   return _num_samples;
}

void Stats::AddSample( double val )
{
   int b;

   _num_samples++;
   _sample_sum += val;

   if ( _reset ) {
      _reset = false;
      _max = val;
      _min = val;
   } else {
      if ( val > _max ) {
         _max = val;
      }
      if ( val < _min ) {
         _min = val;
      }
   }

   b = (int)floor( val / _bin_size );

   if ( b < 0 ) {
      b = 0;
   } else if ( b >= _num_bins ) {
      b = _num_bins - 1;
   }

   _hist[b]++;
}

void Stats::AddSample( int val )
{
   AddSample( (double)val );
}

void Stats::Display( ) const
{
   int b;

   if (_bin_size != 1.0 ) {
      cout<<_fullname<<"_";
      printf("bins = [ ");
      for ( b = 0; b < _num_bins; ++b ) {
         printf("%d ",  b* (unsigned)_bin_size);
      }
      printf("];\n");
   }

   cout<<_fullname<<"_";
   printf("freq = [ ");
   for ( b = 0; b < _num_bins; ++b ) {
      printf("%d ", (unsigned) _hist[b]);
   }
   printf("];\n");
}

bool Stats::NeverUsed() const
{
   if ( _reset ) {
      return true;
   } else {
      return false;
   }
}
