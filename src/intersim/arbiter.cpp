#include "booksim.hpp"
#include <assert.h>

#include "arbiter.hpp"


Arbiter::Arbiter( const Configuration &,
                  Module *parent, const string& name,
                  int inputs )
: Module( parent, name ), _inputs( inputs )
{
}

Arbiter::~Arbiter( )
{
}

void Arbiter::Clear( )
{
   _requests.clear( );
}

void Arbiter::AddRequest( int in, int label, int pri )
{
   sRequest r;
   list<sRequest>::iterator insert_point;

   r.in = in; r.label = label; r.pri = pri;

   insert_point = _requests.begin( );
   while ( ( insert_point != _requests.end( ) ) &&
           ( insert_point->in < in ) ) {
      insert_point++;
   }

   bool del = false;
   bool add = true;

   // For consistant behavior, delete the existing request
   // if it is for the same input and has a higher
   // priority

   if ( ( insert_point != _requests.end( ) ) &&
        ( insert_point->in == in ) ) {
      if ( insert_point->pri < pri ) {
         del = true;
      } else {
         add = false;
      }
   }

   if ( add ) {
      _requests.insert( insert_point, r );
   }

   if ( del ) {
      _requests.erase( insert_point );
   }
}

void Arbiter::RemoveRequest( int in, int label )
{
   list<sRequest>::iterator erase_point;

   erase_point = _requests.begin( );
   while ( ( erase_point != _requests.end( ) ) &&
           ( erase_point->in < in ) ) {
      erase_point++;
   }

   assert( erase_point != _requests.end( ) );
   _requests.erase( erase_point );
}

int Arbiter::Match( ) const
{
   return _match;
}

//==================================================
// PriorityArbiter
//==================================================

PriorityArbiter::PriorityArbiter( const Configuration &config,
                                  Module *parent, const string& name,
                                  int inputs ) 
: Arbiter( config, parent, name, inputs )
{
   _rr_ptr = 0;
}

PriorityArbiter::~PriorityArbiter( )
{
}

void PriorityArbiter::Arbitrate( )
{
   list<sRequest>::iterator p;

   int max_index, max_pri;
   bool wrapped;

   if ( _requests.begin( ) != _requests.end( ) ) {
      // A round-robin arbiter between input requests
      p = _requests.begin( );
      while ( ( p != _requests.end( ) ) &&
              ( p->in < _rr_ptr ) ) {
         p++;
      }

      max_index = -1;
      max_pri   = 0;

      wrapped = false;
      while ( (!wrapped) || ( p->in < _rr_ptr ) ) {
         if ( p == _requests.end( ) ) {
            if ( wrapped ) {
               break;
            }
            // p is valid here because empty lists
            // are skipped (above)
            p = _requests.begin( );
            wrapped = true;
         }

         // check if request is the highest priority so far
         if ( ( p->pri > max_pri ) || ( max_index == -1 ) ) {
            max_pri   = p->pri;
            max_index = p->in;
         }

         p++;
      }   

      _match = max_index; // -1 for no match
      if ( _match != -1 ) {
         _rr_ptr = ( _match + 1 ) % _inputs;
      }

   } else {
      _match = -1;
   }
}
