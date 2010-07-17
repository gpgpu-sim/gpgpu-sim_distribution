#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

class core_t {
public:
   virtual ~core_t() {}
	virtual void set_at_barrier( unsigned cta_id, unsigned warp_id ) = 0;
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) = 0;
};

typedef unsigned address_type;
typedef unsigned addr_t;

#define NO_OP -1
#define ALU_OP 1000
#define LOAD_OP 2000
#define STORE_OP 3000
#define BRANCH_OP 4000
#define BARRIER_OP 5000

#endif
