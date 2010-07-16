#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include <vector>

#include "module.hpp"
#include "flit.hpp"
#include "credit.hpp"
#include "router.hpp"
#include "module.hpp"

#include "config_utils.hpp"

extern int gN;
extern int gK;

extern int gNodes;

class Network : public Module {
protected:

   int _size;
   int _sources;
   int _dests;
   int _channels;

   Router **_routers;

   Flit   **_inject;
   Credit **_inject_cred;

   Flit   **_eject;
   Credit **_eject_cred;

   Flit   **_chan;
   Credit **_chan_cred;

   int *_chan_use;
   int _chan_use_cycles;

   virtual void _ComputeSize( const Configuration &config ) = 0;
   virtual void _BuildNet( const Configuration &config ) = 0;

   void _Alloc( );

public:
   Network( const Configuration &config );
   virtual ~Network( );

   void WriteFlit( Flit *f, int source );
   Flit *ReadFlit( int dest );

   void    WriteCredit( Credit *c, int dest );
   Credit *ReadCredit( int source );

   int  NumSources( ) const;
   int  NumDests( ) const;

   virtual void InsertRandomFaults( const Configuration &config );
   void OutChannelFault( int r, int c, bool fault = true );

   virtual double Capacity( ) const;

   void ReadInputs( );
   void InternalStep( );
   void WriteOutputs( );

   void Display( ) const;
};

#endif 

