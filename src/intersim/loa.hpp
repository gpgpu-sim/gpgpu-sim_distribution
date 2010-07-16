#ifndef _LOA_HPP_
#define _LOA_HPP_

#include "allocator.hpp"

class LOA : public DenseAllocator {
   int *_counts;
   int *_req;

   int *_rptr;
   int *_gptr;

public:
   LOA( const Configuration &config,
        Module *parent, const string& name,
        int inputs, int input_speedup,
        int outputs, int output_speedup );
   ~LOA( );

   void Allocate( );
};

#endif
