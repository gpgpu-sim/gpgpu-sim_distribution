#ifndef _PIM_HPP_
#define _PIM_HPP_

#include "allocator.hpp"

class PIM : public DenseAllocator {
   int _PIM_iter;

   int *_grants;
public:
   PIM( const Configuration &config,
        Module *parent, const string& name,
        int inputs, int outputs );

   ~PIM( );

   void Allocate( );
};

#endif
