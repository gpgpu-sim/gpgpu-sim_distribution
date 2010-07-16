#ifndef CORE_T_INCLUDED
#define CORE_T_INCLUDED

class core_t {
public:
   virtual ~core_t() {}
	virtual void set_at_barrier( unsigned cta_id, unsigned warp_id ) = 0;
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) = 0;
};

#endif
