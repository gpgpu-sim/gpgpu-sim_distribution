#ifndef _WAVEFRONT_HPP_
#define _WAVEFRONT_HPP_

#include "allocator.hpp"

class Wavefront : public DenseAllocator {
   int _square;
   int _pri;

public:
   Wavefront( const Configuration &config,
              Module *parent, const string& name,
              int inputs, int outputs );
   ~Wavefront( );

   void Allocate( );
};

#endif
