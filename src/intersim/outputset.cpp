#include "booksim.hpp"
#include <assert.h>

#include "outputset.hpp"

OutputSet::OutputSet( int num_outputs )
{
   _num_outputs = num_outputs;

   _outputs = new list<sSetElement> [_num_outputs];
}

OutputSet::~OutputSet( )
{
   delete [] _outputs;
}

void OutputSet::Clear( )
{
   for ( int i = 0; i < _num_outputs; ++i ) {
      _outputs[i].clear( );
   }
}

void OutputSet::Add( int output_port, int vc, int pri  )
{
   AddRange( output_port, vc, vc, pri );
}

void OutputSet::AddRange( int output_port, int vc_start, int vc_end, int pri )
{
   assert( ( output_port >= 0 ) && 
           ( output_port < _num_outputs ) &&
           ( vc_start <= vc_end ) );

   sSetElement s;

   s.vc_start = vc_start;
   s.vc_end   = vc_end;
   s.pri      = pri;

   _outputs[output_port].push_back( s );
}

int OutputSet::Size( ) const
{
   return _num_outputs;
}

bool OutputSet::OutputEmpty( int output_port ) const
{
   assert( ( output_port >= 0 ) && 
           ( output_port < _num_outputs ) );

   return _outputs[output_port].empty( );
}

int OutputSet::NumVCs( int output_port ) const
{
   assert( ( output_port >= 0 ) && 
           ( output_port < _num_outputs ) );

   int total = 0;

   for ( list<sSetElement>::const_iterator i = _outputs[output_port].begin( );
       i != _outputs[output_port].end( ); i++ ) {
      total += i->vc_end - i->vc_start + 1;
   }

   return total;
}

int OutputSet::GetVC( int output_port, int vc_index, int *pri ) const
{
   assert( ( output_port >= 0 ) && 
           ( output_port < _num_outputs ) );

   int range;
   int remaining = vc_index;
   int vc = -1;

   if ( pri ) {
      *pri = -1;
   }

   for ( list<sSetElement>::const_iterator i = _outputs[output_port].begin( );
       i != _outputs[output_port].end( ); i++ ) {

      range = i->vc_end - i->vc_start + 1;
      if ( remaining >= range ) {
         remaining -= range;
      } else {
         vc = i->vc_start + remaining;
         if ( pri ) {
            *pri = i->pri;
         }
         break;
      }
   }

   return vc;
}

bool OutputSet::GetPortVC( int *out_port, int *out_vc ) const
{
   bool single_output = false;
   int  used_outputs  = 0;

   for ( int output = 0; output < _num_outputs; ++output ) {

      list<sSetElement>::const_iterator i = _outputs[output].begin( );

      if ( i != _outputs[output].end( ) ) {
         ++used_outputs;

         if ( i->vc_start == i->vc_end ) {
            *out_vc   = i->vc_start;
            *out_port = output;
            single_output = true;
         } else {
            // multiple vc's selected
            break;
         }
      }

      if ( used_outputs > 1 ) {
         // multiple outputs selected
         single_output = false;
         break;
      }
   }

   return single_output;
}
