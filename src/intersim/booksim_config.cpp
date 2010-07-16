#include "booksim.hpp"
#include "booksim_config.hpp"

BookSimConfig::BookSimConfig( )
{
   _int_map["perfect_icnt"] = 0; // if set overrides fixed_lat_per_hop setting 
   _int_map["fixed_lat_per_hop"] = 0; // if set icnt is NOT simulated instead packets are sent into destination based on a fixed_lat_per_hop   
   _int_map["network_count"] = 2; // number of independent interconnection networks (if it is set to 2 then 2 identical networks are created: sh2mem and mem2shd )

   _int_map["output_extra_latency"] = 0;

   _int_map["use_map"] = 1;

   _int_map["flit_size"] = 32;
   //stats
   _int_map["enable_link_stats"]    = 0;     // show output link and VC utilization stats

   _int_map["MATLAB_OUTPUT"]        = 0;     // output data in MATLAB friendly format
   _int_map["DISPLAY_LAT_DIST"]     = 0; // distribution of packet latencies
   _int_map["DISPLAY_HOP_DIST"]     = 0;     // distribution of hop counts
   _int_map["DISPLAY_PAIR_LATENCY"] = 0;     // avg. latency for each s-d pair

   _int_map["input_buf_size"] = 0;     
   _int_map["ejection_buf_size"] = 0; // if left zero the simulator will use the vc_buf_size instead
   _int_map["boundary_buf_size"] = 16;       

   //========================================================
   // Network options
   //========================================================

   //==== Multi-node topology options =======================

   AddStrField( "topology", "torus" );

   _int_map["k"] = 8;
   _int_map["n"] = 2;

   AddStrField( "routing_function", "none" );
   AddStrField( "selection_function", "random" );

   _int_map["link_failures"] = 0;
   _int_map["fail_seed"]     = 0;

   _int_map["wire_delay"] = 0;

   //==== Single-node options ===============================

   _int_map["in_ports"]  = 5;
   _int_map["out_ports"] = 5;

   _int_map["voq"] = 0;

   //========================================================
   // Router options
   //========================================================

   //==== General options ===================================

   AddStrField( "router", "iq" ); 

   _int_map["output_delay"] = 0;
   _int_map["credit_delay"] = 0;
   _float_map["internal_speedup"] = 1.0;

   //==== Input-queued ======================================

   _int_map["num_vcs"]         = 1;  
   _int_map["vc_buf_size"]     = 4;  
   _int_map["vc_buffer_pool"]  = 0;

   _int_map["wait_for_tail_credit"] = 1; // reallocate a VC before a tail credit?

   _int_map["hold_switch_for_packet"] = 0; // hold a switch config for the entire packet

   _int_map["input_speedup"]     = 1;  // expansion of input ports into crossbar
   _int_map["output_speedup"]    = 1;  // expansion of output ports into crossbar

   _int_map["routing_delay"]    = 0;  
   _int_map["vc_alloc_delay"]   = 0;  
   _int_map["sw_alloc_delay"]   = 0;  
   _int_map["st_prepare_delay"] = 0;
   _int_map["st_final_delay"]   = 0;

   //==== Event-driven =====================================

   _int_map["vct"] = 0; 

   //==== Allocators ========================================

   AddStrField( "vc_allocator", "max_size" ); 
   AddStrField( "sw_allocator", "max_size" ); 

   _int_map["alloc_iters"] = 1;

   //==== Traffic ========================================

   AddStrField( "traffic", "uniform" );

   _int_map["perm_seed"] = 0;         // seed value for random perms

   _float_map["injection_rate"]       = 0.2;
   _int_map["const_flits_per_packet"] = 1;

   AddStrField( "injection_process", "bernoulli" );

   _float_map["burst_alpha"] = 0.5; // burst interval
   _float_map["burst_beta"]  = 0.5; // burst length

   AddStrField( "priority", "age" );  // message priorities

   //==== Simulation parameters ==========================

   // types:
   //   latency    - average + latency distribution for a particular injection rate
   //   throughput - sustained throughput for a particular injection rate

   AddStrField( "sim_type", "latency" );

   _int_map["warmup_periods"] = 0; // number of samples periods to "warm-up" the simulation

   _int_map["sample_period"] = 1000; // how long between measurements
   _int_map["max_samples"]   = 10;   // maximum number of sample periods in a simulation

   _float_map["latency_thres"] = 1000.0; // if avg. latency exceeds the threshold, assume unstable

   _int_map["sim_count"]     = 1;   // number of simulations to perform

   _int_map["auto_periods"]  = 1;   // non-zero for the simulator to automatically
                                    //   control the length of warm-up and the
                                    //   total length of the simulation

   _int_map["include_queuing"] = 1; // non-zero includes source queuing latency

   _int_map["reorder"]         = 0;

   _int_map["flit_timing"]     = 0;  // know what you're doing
   _int_map["split_packets"]   = 0;

   _int_map["seed"]            = 0;
}
