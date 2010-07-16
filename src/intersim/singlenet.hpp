#ifndef _SINGLENET_HPP_
#define _SINGLENET_HPP_

#include "network.hpp"

class SingleNet : public Network {

   void _ComputeSize( const Configuration &config );
   void _BuildNet( const Configuration &config );

public:
   SingleNet( const Configuration &config );

   void Display( ) const;
};

#endif
