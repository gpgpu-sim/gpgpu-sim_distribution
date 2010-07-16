#ifndef _MAXSIZE_HPP_
#define _MAXSIZE_HPP_

#include "allocator.hpp"

class MaxSizeMatch : public DenseAllocator {
   int *_from;   // array to hold breadth-first tree
   int *_s;      // stack of leaf nodes in tree
   int *_ns;     // next stack

   bool _ShortestAugmenting( );

public:
   MaxSizeMatch( const Configuration &config,
                 Module *parent, const string& name,
                 int inputs, int ouputs ); 
   ~MaxSizeMatch( );

   void Allocate( );
};

#endif 
