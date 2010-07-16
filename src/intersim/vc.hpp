#ifndef _VC_HPP_
#define _VC_HPP_

#include <queue>

#include "flit.hpp"
#include "outputset.hpp"
#include "routefunc.hpp"
#include "config_utils.hpp"

class VCRouter;

class VC : public Module {
public:
   enum eVCState {
      idle, routing, vc_alloc, active
   };

private:
   int _size;

   queue<Flit *> _buffer;

   eVCState _state;
   int      _state_time;

   OutputSet *_route_set;
   int _out_port, _out_vc;

   int _occupied_cnt;
   int _total_cycles;
   int _vc_alloc_cycles;
   int _active_cycles;
   int _idle_cycles;

   int _pri;

   void _Init( const Configuration& config, int outputs );

   bool _watched;

public:
    VC() : Module() {}
   void init( const Configuration& config, int outputs );
   VC( const Configuration& config, int outputs,
       Module *parent, const string& name );
   ~VC( );

   bool AddFlit( Flit *f );
   Flit *FrontFlit( );
   Flit *RemoveFlit( );

   bool Empty( ) const;

   eVCState GetState( ) const;
   int      GetStateTime( ) const;
   void     SetState( eVCState s );

   const OutputSet *GetRouteSet( ) const;

   void SetOutput( int port, int vc );
   int  GetOutputPort( ) const;
   int  GetOutputVC( ) const;

   int  GetPriority( ) const;

   void Route( tRoutingFunction rf, const Router* router, const Flit* f, int in_channel );

   void AdvanceTime( );

   // ==== Debug functions ====

   void SetWatch( bool watch = true );
   bool IsWatched( ) const;

   void Display( ) const;
};

#endif 
