#pragma once

#include "dram.h"

// L2 Cache Creation 
void L2c_create ( dram_t* dram_p, const char* cache_opt );

void L2c_qlen ( dram_t *dram_p );

// service memory request in icnt-to-L2 queue, writing to L2 as necessary
// (if L2 writeback miss, writeback to memory) 
void L2c_service_mem_req ( dram_t* dram_p, int dm_id );

// service memory request in L2todramqueue, pushing to dram 
void L2c_push_miss_to_dram ( dram_t* dram_p );

// pop completed memory request from dram and push it to dram-to-L2 queue 
void L2c_get_dram_output ( dram_t* dram_p );

// service memory request in dramtoL2queue, writing to L2 as necessary
// (may cause cache eviction and subsequent writeback) 
void L2c_process_dram_output ( dram_t* dram_p, int dm_id );

// Writeback from L2 to DRAM: 
// - Takes in memory address and their parameters and pushes to dram request queue
// - This is used only for L2 writeback 
unsigned char L2c_write_back(unsigned long long int addr, int bsize, int dram_id );

unsigned int L2c_cache_flush ( dram_t* dram_p);

unsigned L2c_get_linesize( dram_t *dram_p );

// probe L2 cache for fullness 
int L2c_full( dram_t *dram_p );
void L2c_push( dram_t *dram_p, struct mem_fetch *mf );
struct mem_fetch* L2c_pop( dram_t *dram_p );
struct mem_fetch* L2c_top( dram_t *dram_p );

void L2c_init_stat();
void L2c_update_stat( dram_t* dram_p);
void L2c_print_stat();
void L2c_print_cache_stat();
void L2c_print_debug();
void L2c_log(int task);
void L2c_latency_log_dump();

void L2c_options(class OptionParser *opp);

extern unsigned L2_write_miss;
extern unsigned L2_write_hit;
extern unsigned L2_read_hit;
extern unsigned L2_read_miss;
extern bool gpgpu_l2_readoverwrite;
