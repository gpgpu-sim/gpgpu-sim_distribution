#ifndef _ALLOCATOR_HPP_
#define _ALLOCATOR_HPP_

#include <string>
#include <list>

#include "module.hpp"
#include "config_utils.hpp"

class Allocator : public Module {
protected:
   const int _inputs;
   const int _outputs;

   int *_inmatch;
   int *_outmatch;

   int *_outmask;

   void _ClearMatching( );
public:

   struct sRequest {
      int port;
      int label;
      int in_pri;
      int out_pri;
   };

   Allocator( const Configuration &config,
              Module *parent, const string& name,
              int inputs, int outputs );
   virtual ~Allocator( );

   virtual void Clear( ) = 0;

   virtual int  ReadRequest( int in, int out ) const = 0;
   virtual bool ReadRequest( sRequest &req, int in, int out ) const = 0;

   virtual void AddRequest( int in, int out, int label = 1, 
                            int in_pri = 0, int out_pri = 0 ) = 0;
   virtual void RemoveRequest( int in, int out, int label = 1 ) = 0;

   virtual void Allocate( ) = 0;

   void MaskOutput( int out, int mask = 1 );

   int OutputAssigned( int in ) const;
   int InputAssigned( int out ) const;

   virtual void PrintRequests( ) const = 0;

   static Allocator *NewAllocator( const Configuration &config,
                                   Module *parent, const string& name,
                                   const string &alloc_type, 
                                   int inputs, int input_speedup,
                                   int outputs, int output_speedup );
};

//==================================================
// A dense allocator stores the entire request
// matrix.
//==================================================

class DenseAllocator : public Allocator {
protected:
   sRequest **_request;

public:
   DenseAllocator( const Configuration &config,
                   Module *parent, const string& name,
                   int inputs, int outputs );
   virtual ~DenseAllocator( );

   void Clear( );

   int  ReadRequest( int in, int out ) const;
   bool ReadRequest( sRequest &req, int in, int out ) const;

   void AddRequest( int in, int out, int label = 1, 
                    int in_pri = 0, int out_pri = 0 );
   void RemoveRequest( int in, int out, int label = 1 );

   virtual void Allocate( ) = 0;

   void PrintRequests( ) const;
};

//==================================================
// A sparse allocator only stores the requests
// (allows for a more efficient implementation).
//==================================================

class SparseAllocator : public Allocator {
protected:
   list<int> *_in_occ;
   list<int> *_out_occ;

   list<sRequest> *_in_req;
   list<sRequest> *_out_req;

public:
   SparseAllocator( const Configuration &config,
                    Module *parent, const string& name,
                    int inputs, int outputs );
   virtual ~SparseAllocator( );

   void Clear( );

   int  ReadRequest( int in, int out ) const;
   bool ReadRequest( sRequest &req, int in, int out ) const;

   void AddRequest( int in, int out, int label = 1, 
                    int in_pri = 0, int out_pri = 0 );
   void RemoveRequest( int in, int out, int label = 1 );

   virtual void Allocate( ) = 0;

   void PrintRequests( ) const;
};

#endif
