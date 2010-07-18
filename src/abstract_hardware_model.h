#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

#ifdef __cplusplus

class core_t {
public:
   virtual ~core_t() {}
	virtual void set_at_barrier( unsigned cta_id, unsigned warp_id ) = 0;
   virtual void warp_exit( unsigned warp_id ) = 0;
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) = 0;
};

#endif

typedef unsigned address_type;
typedef unsigned addr_t;

// these are operations the timing model can see
#define NO_OP -1
#define ALU_OP 1000
#define LOAD_OP 2000
#define STORE_OP 3000
#define BRANCH_OP 4000
#define BARRIER_OP 5000

typedef enum _memory_space_t {
   undefined_space=0,
   reg_space,
   local_space,
   shared_space,
   param_space_unclassified,
   param_space_kernel,  /* global to all threads in a kernel : read-only */
   param_space_local,   /* local to a thread : read-writable */
   const_space,
   tex_space,
   surf_space,
   global_space,
   generic_space
} memory_space_t;

#endif
