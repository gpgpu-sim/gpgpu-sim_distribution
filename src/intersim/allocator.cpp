#include "booksim.hpp"
#include <iostream>
#include <assert.h>

#include "allocator.hpp"
#include "maxsize.hpp"
#include "pim.hpp"
#include "islip.hpp"
#include "loa.hpp"
#include "wavefront.hpp"
#include "selalloc.hpp"

//==================================================
// Allocator base class
//==================================================

Allocator::Allocator( const Configuration &config,
                      Module *parent, const string& name,
                      int inputs, int outputs ) :
Module( parent, name ), _inputs( inputs ), _outputs( outputs )
{
   _inmatch  = new int [_inputs];   
   _outmatch = new int [_outputs];
   _outmask  = new int [_outputs];

   for ( int out = 0; out < _outputs; ++out ) {
      _outmask[out] = 0; // active
   }
}

Allocator::~Allocator( )
{
   delete [] _inmatch;
   delete [] _outmatch;
   delete [] _outmask;
}

void Allocator::_ClearMatching( )
{
   for ( int i = 0; i < _inputs; ++i ) {
      _inmatch[i] = -1;
   }

   for ( int j = 0; j < _outputs; ++j ) {
      _outmatch[j] = -1;
   }
}

int Allocator::OutputAssigned( int in ) const
{
   assert( ( in >= 0 ) && ( in < _inputs ) );

   return _inmatch[in];
}

int Allocator::InputAssigned( int out ) const
{
   assert( ( out >= 0 ) && ( out < _outputs ) );

   return _outmatch[out];
}

void Allocator::MaskOutput( int out, int mask )
{
   assert( ( out >= 0 ) && ( out < _outputs ) );
   _outmask[out] = mask;
}

//==================================================
// DenseAllocator
//==================================================

DenseAllocator::DenseAllocator( const Configuration &config,
                                Module *parent, const string& name,
                                int inputs, int outputs ) :
Allocator( config, parent, name, inputs, outputs )
{
   _request  = new sRequest * [_inputs];

   for ( int i = 0; i < _inputs; ++i ) {
      _request[i]  = new sRequest [_outputs];  
   }

   Clear( );
}

DenseAllocator::~DenseAllocator( )
{  
   for ( int i = 0; i < _inputs; ++i ) {
      delete [] _request[i];
   }

   delete [] _request;
}

void DenseAllocator::Clear( )
{
   for ( int i = 0; i < _inputs; ++i ) {
      for ( int j = 0; j < _outputs; ++j ) {
         _request[i][j].label = -1;
      }
   }
}

int DenseAllocator::ReadRequest( int in, int out ) const
{
   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) );

   return _request[in][out].label;
}

bool DenseAllocator::ReadRequest( sRequest &req, int in, int out ) const
{
   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) );

   req = _request[in][out];

   return( req.label != -1 );
}

void DenseAllocator::AddRequest( int in, int out, int label, 
                                 int in_pri, int out_pri )
{
   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) );

   _request[in][out].label   = label;
   _request[in][out].in_pri  = in_pri;
   _request[in][out].out_pri = out_pri;
}

void DenseAllocator::RemoveRequest( int in, int out, int label )
{
   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) ); 

   _request[in][out].label = -1;
}

void DenseAllocator::PrintRequests( ) const
{
   cout << "requests for " << _fullname << endl;
   for ( int i = 0; i < _inputs; ++i ) {
      for ( int j = 0; j < _outputs; ++j ) {
         cout << ( _request[i][j].label != -1 ) << " ";
      }
      cout << endl;
   }
   cout << endl;
}

//==================================================
// SparseAllocator
//==================================================

SparseAllocator::SparseAllocator( const Configuration &config,
                                  Module *parent, const string& name,
                                  int inputs, int outputs ) :
Allocator( config, parent, name, inputs, outputs )
{
   _in_req =  new list<sRequest> [_inputs];
   _out_req = new list<sRequest> [_outputs];
}


SparseAllocator::~SparseAllocator( )
{
   delete [] _in_req;
   delete [] _out_req;
}

void SparseAllocator::Clear( )
{
   for ( int i = 0; i < _inputs; ++i ) {
      _in_req[i].clear( );
   }

   for ( int j = 0; j < _outputs; ++j ) {
      _out_req[j].clear( );
   }

   _in_occ.clear( );
   _out_occ.clear( );
}

int SparseAllocator::ReadRequest( int in, int out ) const
{
   sRequest r;

   if ( ! ReadRequest( r, in, out ) ) {
      r.label = -1;
   }

   return r.label;
}

bool SparseAllocator::ReadRequest( sRequest &req, int in, int out ) const
{
   bool found;

   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) );

   list<sRequest>::const_iterator match;

   match = _in_req[in].begin( );
   while ( ( match != _in_req[in].end( ) ) &&
           ( match->port != out ) ) {
      match++;
   }

   if ( match != _in_req[in].end( ) ) {
      req = *match;
      found = true;
   } else {
      found = false;
   }

   return found;
}

void SparseAllocator::AddRequest( int in, int out, int label, 
                                  int in_pri, int out_pri )
{
   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) );

   list<sRequest>::iterator insert_point;
   list<int>::iterator occ_insert;
   sRequest req;

   // insert into occupied inputs list if
   // input is currently empty
   if ( _in_req[in].empty( ) ) {
      occ_insert = _in_occ.begin( );
      while ( ( occ_insert != _in_occ.end( ) ) &&
              ( *occ_insert < in ) ) {
         occ_insert++;
      }
      assert( ( occ_insert == _in_occ.end( ) ) || 
              ( *occ_insert != in ) );

      _in_occ.insert( occ_insert, in );
   }

   // similarly for the output
   if ( _out_req[out].empty( ) ) {
      occ_insert = _out_occ.begin( );
      while ( ( occ_insert != _out_occ.end( ) ) &&
              ( *occ_insert < out ) ) {
         occ_insert++;
      }
      assert( ( occ_insert == _out_occ.end( ) ) || 
              ( *occ_insert != out ) );

      _out_occ.insert( occ_insert, out );
   }

   // insert input request in order of it's output
   insert_point = _in_req[in].begin( );
   while ( ( insert_point != _in_req[in].end( ) ) &&
           ( insert_point->port < out ) ) {
      insert_point++;
   }

   req.port    = out;
   req.label   = label;
   req.in_pri  = in_pri;
   req.out_pri = out_pri;

   bool del = false;
   bool add = true;

   // For consistent behavior, delete the existing request
   // if it is for the same output and has a higher
   // priority

   if ( ( insert_point != _in_req[in].end( ) ) &&
        ( insert_point->port == out ) ) {
      if ( insert_point->in_pri < in_pri ) {
         del = true;
      } else {
         add = false;
      }
   }

   if ( add ) {
      _in_req[in].insert( insert_point, req );
   }

   if ( del ) {
      _in_req[in].erase( insert_point );
   }

   insert_point = _out_req[out].begin( );
   while ( ( insert_point != _out_req[out].end( ) ) &&
           ( insert_point->port < in ) ) {
      insert_point++;
   }

   req.port  = in;
   req.label = label;

   if ( add ) {
      _out_req[out].insert( insert_point, req );
   }

   if ( del ) {
      // This should be consistent, but check for sanity
      if ( ( insert_point == _out_req[out].end( ) ) ||
           ( insert_point->port != in ) ) {
         Error( "Internal allocator error --- input and output requests non consistent" );
      }
      _out_req[out].erase( insert_point );
   }
}

void SparseAllocator::RemoveRequest( int in, int out, int label )
{
   assert( ( in >= 0 ) && ( in < _inputs ) &&
           ( out >= 0 ) && ( out < _outputs ) ); 

   list<sRequest>::iterator erase_point;
   list<int>::iterator occ_remove;

   // insert input request in order of it's output
   erase_point = _in_req[in].begin( );
   while ( ( erase_point != _in_req[in].end( ) ) &&
           ( erase_point->port != out ) ) {
      erase_point++;
   }

   assert( erase_point != _in_req[in].end( ) );
   _in_req[in].erase( erase_point );

   // remove from occupied inputs list if
   // input is now empty
   if ( _in_req[in].empty( ) ) {
      occ_remove = _in_occ.begin( );
      while ( ( occ_remove != _in_occ.end( ) ) &&
              ( *occ_remove != in ) ) {
         occ_remove++;
      }

      assert( occ_remove != _in_occ.end( ) );
      _in_occ.erase( occ_remove );
   }

   // similarly for the output
   erase_point = _out_req[out].begin( );
   while ( ( erase_point != _out_req[out].end( ) ) &&
           ( erase_point->port != in ) ) {
      erase_point++;
   }

   assert( erase_point != _out_req[out].end( ) );
   _out_req[out].erase( erase_point );

   if ( _out_req[out].empty( ) ) {
      occ_remove = _out_occ.begin( );
      while ( ( occ_remove != _out_occ.end( ) ) &&
              ( *occ_remove != out ) ) {
         occ_remove++;
      }

      assert( occ_remove != _out_occ.end( ) );
      _out_occ.erase( occ_remove );
   }
}

void SparseAllocator::PrintRequests( ) const
{
   list<sRequest>::const_iterator iter;

   cout << "input requests:" << endl;
   for ( int input = 0; input < _inputs; ++input ) {
      cout << "  input " << input << " : ";
      for ( iter = _in_req[input].begin( ); 
          iter != _in_req[input].end( ); iter++ ) {
         cout << iter->port << "  ";
      }
      cout << endl;
   }

   cout << "output requests:" << endl;
   for ( int output = 0; output < _outputs; ++output ) {
      cout << "  output " << output << " : ";
      if ( _outmask[output] == 0 ) {
         for ( iter = _out_req[output].begin( ); 
             iter != _out_req[output].end( ); iter++ ) {
            cout << iter->port << "  ";
         }
         cout << endl;
      } else {
         cout << "masked" << endl;
      }
   }
}

//==================================================
// Global allocator allocation function
//==================================================

Allocator *Allocator::NewAllocator( const Configuration &config,
                                    Module *parent, const string& name,
                                    const string &alloc_type, 
                                    int inputs, int input_speedup,
                                    int outputs, int output_speedup )
{
   Allocator *a = 0;

   if ( alloc_type == "max_size" ) {
      a = new MaxSizeMatch( config, parent, name, inputs, outputs );
   } else if ( alloc_type == "pim" ) {
      a = new PIM( config, parent, name, inputs, outputs );
   } else if ( alloc_type == "islip" ) {
      a = new iSLIP_Sparse( config, parent, name, inputs, outputs );
   } else if ( alloc_type == "loa" ) {
      a = new LOA( config, parent, name, inputs, input_speedup, outputs, output_speedup );
   } else if ( alloc_type == "wavefront" ) {
      a = new Wavefront( config, parent, name, inputs, outputs );
   } else if ( alloc_type == "select" ) {
      a = new SelAlloc( config, parent, name, inputs, outputs );
   }

   return a;
}
