#ifndef _ISLIP_HPP_
#define _ISLIP_HPP_

#include "allocator.hpp"

class iSLIP_Sparse : public SparseAllocator {
   int _iSLIP_iter;

   int *_grants;
   int *_gptrs;
   int *_aptrs;

public:
   iSLIP_Sparse( const Configuration &config,
                 Module *parent, const string& name,
                 int inputs, int outputs );
   ~iSLIP_Sparse( );

   void Allocate( );
};

#endif 
