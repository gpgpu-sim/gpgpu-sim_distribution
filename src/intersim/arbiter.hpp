#ifndef _ARBITER_HPP_
#define _ARBITER_HPP_

#include <list>

#include "module.hpp"
#include "config_utils.hpp"

class Arbiter : public Module {
protected:
   const int _inputs;

   struct sRequest {
      int in;
      int label;
      int pri;
   };

   list<sRequest> _requests;

   int _match;

public:
   Arbiter( const Configuration &,
            Module *parent, const string& name,
            int inputs );
   virtual ~Arbiter( );

   void Clear( );

   void AddRequest( int in, int label = 0, int pri = 0 );
   void RemoveRequest( int in, int label = 0 );

   virtual void Arbitrate( ) = 0;

   int Match( ) const;
};

class PriorityArbiter : public Arbiter {
   int _rr_ptr;

public:
   PriorityArbiter( const Configuration &config,
                    Module *parent, const string& name,
                    int inputs );
   ~PriorityArbiter( );

   void Arbitrate( );
};

#endif 
