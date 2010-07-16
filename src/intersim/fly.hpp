#ifndef _FLY_HPP_
#define _FLY_HPP_

#include "network.hpp"

class KNFly : public Network {

   int _k;
   int _n;

   void _ComputeSize( const Configuration &config );
   void _BuildNet( const Configuration &config );

   int _OutChannel( int stage, int addr, int port ) const;
   int _InChannel( int stage, int addr, int port ) const;

public:
   KNFly( const Configuration &config );

   int GetN( ) const;
   int GetK( ) const;

   double Capacity( ) const;
};

#endif 
