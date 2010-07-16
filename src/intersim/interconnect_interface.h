#ifndef INTERCONNECT_INTERFACE_H
#define INTERCONNECT_INTERFACE_H

#include "flit.hpp"
#include "trafficmanager.hpp"

struct glue_buf {
   int flit_c; // flit count 
   void *data;
   int net_num; // which network is this flit in (we might have several icnt networks)
   int  src;
   int  dest;
};

//node side functions
int interconnect_has_buffer(unsigned int input, unsigned int *size); 
void interconnect_push ( unsigned int input, unsigned int output, 
		    void* data, unsigned int size); 
void* interconnect_pop(unsigned int output);
void init_interconnect (char* config_file, 
		   unsigned int n_shader, unsigned int n_mem);
void advance_interconnect();
unsigned interconnect_busy();
void interconnect_stats() ;

//interconnect side functions
void create_buf(int src_n,int warp_n, int vc_n);
int in_map (int input) ;
void write_out_buf(int output, Flit * data);
void transfer2boundary_buf(int output);
void time_vector_update_icnt_injected(void* mf, int input);

#endif


