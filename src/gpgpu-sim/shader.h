/* 
 * shader.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Ivan Sham, Henry Wong, Dan O'Connor, Henry Tran and the 
 * University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <map>

#include "../cuda-sim/ptx.tab.h"
#include "../cuda-sim/dram_callback.h"

#include "gpu-cache.h"
#include "delayqueue.h"
#include "stack.h"
#include "dram.h"
#include "../abstract_hardware_model.h"

#ifndef SHADER_H
#define SHADER_H

#define NO_OP_FLAG            0xFF

//READ_PACKET_SIZE: bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel, so good for now), 2 bytes [shaderid + mshrid](14 bits) + req_size(0-2 bits (if req_size variable) - so up to 2^14 = 16384 mshr total
#define READ_PACKET_SIZE 8
//WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous. 
#define WRITE_PACKET_SIZE 8

#include <bitset>
const unsigned partial_write_mask_bits = 128; //must be at least size of largest memory access.
typedef std::bitset<partial_write_mask_bits> partial_write_mask_t;

#define WRITE_MASK_SIZE 8
#define NO_PARTIAL_WRITE (partial_write_mask_t())

//this is used a lot of places where it maybe should be more variable?
#define WORD_SIZE 4

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32

typedef unsigned op_type;

enum {
   NO_RECONVERGE = 0,
   POST_DOMINATOR = 1,
   MIMD = 2,
   DWF = 3,
   NUM_SIMD_MODEL
};

//Defines number of threads grouped together to be executed together


typedef struct {

   address_type pc;

   op_type op;
   int space;

   unsigned long long int memreqaddr;
   //Each instruction keeps track of which hardware thread it came from
   short hw_thread_id;
   short wlane;

   /* reg label of the instruction */
   unsigned out[4];
   unsigned in[4];
   unsigned char is_vectorin;
   unsigned char is_vectorout;
   int arch_reg[8]; // register number for bank conflict evaluation
   unsigned data_size; // what is the size of the word being operated on?

   int reg_bank_access_pending;
   int reg_bank_conflict_stall_checked; // flag to turn off register bank conflict checker to avoid double stalling

   unsigned char inst_type;

   unsigned priority;

   unsigned uid;

   void *ptx_thd_info; 
   dram_callback_t callback;
   unsigned warp_active_mask;
   unsigned long long  ts_cycle;
   unsigned long long  if_cycle;
   unsigned long long  id_cycle;
   unsigned long long  ex_cycle;
   unsigned long long  mm_cycle;

} inst_t;

typedef struct {

   void *ptx_thd_info; // pointer to the functional state of the thread in cuda-sim

   int avail4fetch; // 1 if its instrucion can be fetch into the pipeline, 0 otherwise 
   int warp_priority;

   int id;

   //unsigned n_completed;         // number of threads in warp completed -- set for first thread in each warp
   //unsigned n_avail4fetch;       // number of threads in warp available to fetch -- set for first thread in each warp
   //int n_waiting_at_barrier;  // number of threads in warp that have reached the barrier
   unsigned in_scheduler;     // used by dynamic warp formation for error check

   int         m_waiting_at_barrier;
   int         m_reached_barrier;

   unsigned n_insn;
   unsigned n_insn_ac;
   unsigned n_l1_mis_ac,
   n_l1_mrghit_ac,
   n_l1_access_ac; //used to collect "per thread" l1 miss statistics 
                   // ac stands for accumulative.
   unsigned cta_id; // which hardware CTA does this thread belong to?
} thread_ctx_t;

struct shd_warp_t
{
    shd_warp_t(unsigned warp_size){reset(warp_size); assert(warp_size <= bitset_size);}
    void reset(unsigned warp_size){n_completed = warp_size; n_avail4fetch = n_waiting_at_barrier = 0; threads_completed.reset(); threads_functionally_executed.reset();}

    unsigned wid;
    unsigned n_completed; // number of threads in warp completed
    unsigned n_avail4fetch; // number of threads in warp available to fetch 
    int n_waiting_at_barrier; // number of threads in warp that have reached the barrier

    const static unsigned bitset_size = 32;
    std::bitset<bitset_size> threads_completed;
    std::bitset<bitset_size> threads_functionally_executed;
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i){return wid * warp_size + i;};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size){return tid/warp_size;};

typedef struct {

   int m_stack_top;

   address_type *m_pc;
   unsigned int *m_active_mask;
   address_type *m_recvg_pc;
   unsigned int *m_calldepth;

   unsigned long long  *m_branch_div_cycle;

} pdom_warp_ctx_t; // bounded stack that implements pdom reconvergence (see MICRO'07 paper)


enum mshr_status {
   INITIALIZED = 0,
   IN_ICNT2MEM,
   IN_CBTOL2QUEUE,
   IN_L2TODRAMQUEUE,
   IN_DRAM_REQ_QUEUE,
   IN_DRAMRETURN_Q,
   IN_DRAMTOL2QUEUE,
   IN_L2TOCBQUEUE_HIT,
   IN_L2TOCBQUEUE_MISS,
   IN_ICNT2SHADER,
   FETCHED,
   NUM_MSHR_STATUS
};

//used to stages that time_vector will keep track of their timing 
enum mem_req_stat {
   MR_UNUSED,
   MR_FQPUSHED,
   MR_ICNT_PUSHED,
   MR_ICNT_INJECTED,
   MR_ICNT_AT_DEST,
   MR_DRAMQ, //icnt_pop at dram side and mem_ctrl_push
   MR_DRAM_PROCESSING_START,
   MR_DRAM_PROCESSING_END,
   MR_DRAM_OUTQ,
   MR_2SH_ICNT_PUSHED, // icnt_push and mem_ctl_pop //STORES END HERE!
   MR_2SH_ICNT_INJECTED,
   MR_2SH_ICNT_AT_DEST,
   MR_2SH_FQ_POP, //icnt_pop called inside fq_pop
   MR_RETURN_Q,
   MR_WRITEBACK, //done
   NUM_MEM_REQ_STAT
};
#include <vector>
typedef struct mshr_entry_t {
#ifdef _GLIBCXX_DEBUG
    //satisfy cxx debug conditions on iterators, needs to be nonsingular to copy, which messes completely with structures containing them.
    mshr_entry_t(){
        static std::vector<mshr_entry_t> dummy_vector;
        this_mshr = dummy_vector.begin(); //initialize it to something nonsingular so it can be copied. 
    }
#endif
private:
   friend class mshr_shader_unit;
   std::vector<mshr_entry_t>::iterator this_mshr; //to ease tracking and update.
public:
   unsigned request_uid;
    
   /* memory address of the data */
   unsigned long long int addr;

   // instructions are stored here.
   std::vector<inst_t> insts;

   /* Current stage of the load: fetched or not? */
   bool fetched(){return status == FETCHED;};

   bool iswrite;

   bool merged_on_other_reqest; //true if waiting for another mshr - this mshr doesn't send a memory request
   struct mshr_entry_t *merged_requests; //mshrs waiting on this mshr

   enum mshr_status status; 

   void *mf; // link to corresponding memory fetch structure

   //unsigned space; //does below.
   bool istexture; //if it's a request from the texture cache
   bool isconst; //if it's a request from the constant cache
   bool islocal; //if it's a request to the local memory of a thread

   bool wt_no_w2cache; //in write_through, sometimes need to prevent writing back returning data into cache, because its been written in the meantime. 
} mshr_entry;

enum mem_access_type { 
   GLOBAL_ACC_R = 0, 
   LOCAL_ACC_R = 1, 
   CONST_ACC_R = 2, 
   TEXTURE_ACC_R = 3, 
   GLOBAL_ACC_W = 4, 
   LOCAL_ACC_W = 5,
   L2_WRBK_ACC = 6, 
   NUM_MEM_ACCESS_TYPE = 7
};


/* A pointer to the function that glues the shader with the memory hiearchy */
typedef unsigned char (*fq_push_t)(unsigned long long int addr, int bsize, unsigned char readwrite,
                                   partial_write_mask_t, 
                                   int sid, int wid, mshr_entry* mshr, int cache_hits_waiting,  
                                   enum mem_access_type mem_acc, address_type pc);

typedef unsigned char (*fq_has_buffer_t)(unsigned long long int addr, int bsize, bool write, int sid);

const unsigned WARP_PER_CTA_MAX = 32;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

class barrier_set_t {
public:
   barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core );

   // during cta allocation
   void allocate_barrier( unsigned cta_id, warp_set_t warps );

   // during cta deallocation
   void deallocate_barrier( unsigned cta_id );

   typedef std::map<unsigned, warp_set_t >  cta_to_warp_t;

   // individual warp hits barrier
   void warp_reaches_barrier( unsigned cta_id, unsigned warp_id );

   // fetching a warp
   bool available_for_fetch( unsigned warp_id ) const;

   // warp reaches exit 
   void warp_exit( unsigned warp_id );

   // assertions
   bool warp_waiting_at_barrier( unsigned warp_id );

   // debug
   void dump() const;

private:
   unsigned m_max_cta_per_core;
   unsigned m_max_warps_per_core;

   cta_to_warp_t m_cta_to_warps; 
   warp_set_t m_warp_active;
   warp_set_t m_warp_at_barrier;
};

class mshr_shader_unit;
  
extern unsigned int warp_size; 

typedef struct shader_core_ctx : public core_t 
{
   shader_core_ctx( unsigned max_warps_per_cta, unsigned max_cta_per_core );

	virtual void set_at_barrier( unsigned cta_id, unsigned warp_id );
   virtual void warp_exit( unsigned warp_id );
   virtual bool warp_waiting_at_barrier( unsigned warp_id );
   void allocate_barrier( unsigned cta_id, warp_set_t warps );
   void deallocate_barrier( unsigned cta_id );

////
   
   const char *name;
   int sid;

   // array of the threads running on this shader core 
   thread_ctx_t *thread;
   unsigned int n_threads;
   unsigned int last_issued_thread;

   //per warp information array
   std::vector<shd_warp_t> warp;

   barrier_set_t m_barriers;

   //Keeps track of which warp of instructions to fetch/execute
   int next_warp; 

   // number of threads to be completed ( ==0 when all thread on this core completed) 
   int not_completed; 
   // number of Cuda Thread Arrays (blocks) currently running on this shader.
   int n_active_cta;
   //Keep track of multiple CTAs in shader 
   int cta_status[MAX_CTA_PER_SHADER]; 
   // registers holding the instruction between pipeline stages. 
   // see below for definition of pipeline stages
   inst_t** pipeline_reg;
   inst_t** pre_mem_pipeline;
   int warp_part2issue; // which part of warp to issue to pipeline 
   int new_warp_TS; // new warp at TS pipeline register

   shd_cache_t *L1cache;
   shd_cache_t *L1texcache;
   shd_cache_t *L1constcache;

   // pointer to memory access wrapping function 
   fq_push_t fq_push;
   fq_has_buffer_t fq_has_buffer;

   // simulation cycles happened to the shader, kept for cacheline replacement
   unsigned int gpu_cycle;
   // number of instructions committed by this shader core
   unsigned int num_sim_insn;

   // reconvergence
   unsigned int model;

   // Structure is used to keep track of the branching within the warp of instructions.
   // As a group of instructions is grouped together from different threads to be executed, when
   // a branch does occur, then the sub-set that does not get run will be given the value of warp_priority,
   // and warp_priority will increase. Each time a sub-set branches further, a similar scheme is used.
   // When a sub-set completes fully, then this table will determine which next sub-set to finish, which 
   // will be the next largest value in the table.
   int branch_priority;  
   int* max_branch_priority; //Keeps track of the maximum priority of the threads running within a warp. need n_threads number of these

   // pdom reconvergence context for each warp
   pdom_warp_ctx_t *pdom_warp;

   int waiting_at_barrier; // number of threads current waiting at a barrier in this shader.
   int RR_k; //counter for register read pipeline

   int using_dwf; //is the scheduler using dynamic warp formation
   int using_rrstage; //is the pipeline using an extra stage for register read
   int using_commit_queue; //is the scheduler using commit_queue?

   delay_queue *thd_commit_queue;

   int pending_shmem_bkacc; // 0 = check conflict for new insn
   int pending_cache_bkacc; // 0 = check conflict for new insn

   bool shader_memory_new_instruction_processed;

   int pending_mem_access; // number of memory access to be serviced (use for W0 classification)

   int pending_cmem_acc; //number of accesses to differrnt addresses in the constant memory cache

   unsigned int n_diverge; // number of divergence occurred in this shader

   //Shader core resources
   unsigned int shmem_size;
   unsigned int n_registers;   //registers available in the shader core 
   unsigned int n_cta;      //Limit on number of concurrent CTAs in shader core

   //void *req_hist; //not used anywhere

   mshr_shader_unit *mshr_unit;
} shader_core_ctx_t;


shader_core_ctx_t* shader_create( const char *name, int sid, unsigned int n_threads, 
                                  unsigned int n_mshr, fq_push_t fq_push, fq_has_buffer_t fq_has_buffer, unsigned int model);
unsigned shader_reinit(shader_core_ctx_t *sc, int start_thread, int end_thread);
void shader_init_CTA(shader_core_ctx_t *shader, int start_thread, int end_thread);

void shader_fetch( shader_core_ctx_t *shader, 
                   unsigned int shader_number,
                   int grid_num );
void shader_decode( shader_core_ctx_t *shader, 
                    unsigned int shader_number,
                    unsigned int grid_num );
void shader_preexecute( shader_core_ctx_t *shader, 
                        unsigned int shader_number );
void shader_execute( shader_core_ctx_t *shader, 
                     unsigned int shader_number );
void shader_pre_memory( shader_core_ctx_t *shader, 
                        unsigned int shader_number );
void shader_const_memory( shader_core_ctx_t *shader, 
                          unsigned int shader_number );
void shader_texture_memory( shader_core_ctx_t *shader, 
                            unsigned int shader_number );
void shader_memory( shader_core_ctx_t *shader, 
                    unsigned int shader_number );
void shader_writeback( shader_core_ctx_t *shader, 
                       unsigned int shader_number,
                       int grid_num );

void shader_display_pipeline(shader_core_ctx_t *shader, FILE *fout, int print_mem, int mask3bit );
void shader_dump_thread_state(shader_core_ctx_t *shader, FILE *fout );
void shader_cycle( shader_core_ctx_t *shader, 
                   unsigned int shader_number,
                   int grid_num );

void mshr_print(FILE *fp, shader_core_ctx_t *shader);

void mshr_update_status(mshr_entry* mshr, enum mshr_status new_status);

mshr_entry* fetchMSHR(delay_queue** mshr, shader_core_ctx_t* sc);
mshr_entry* shader_check_mshr4tag(shader_core_ctx_t* sc, unsigned long long int addr,int mem_type);
void shader_update_mshr(shader_core_ctx_t* sc, unsigned long long int fetched_addr, unsigned int mshr_idx, int mem_type );
void shader_visualizer_dump(FILE *fp, shader_core_ctx_t* sc);

void init_mshr_pool();
mshr_entry* alloc_mshr_entry();
void free_mshr_entry( mshr_entry * );

void shader_clean(shader_core_ctx_t *sc, unsigned int n_threads);
void shader_cache_flush(shader_core_ctx_t* sc);

// print out the accumulative statistics for shaders (those that are not local to one shader)
void shader_print_accstats( FILE* fout );
void shader_print_runtime_stat( FILE *fout );
void shader_print_l1_miss_stat( FILE *fout );

//return the maximum CTAs that can be running at the same on shader 
//based on on the current kernel's CTA size and is 1 if mutiple CTA per block is not supported
unsigned int max_cta_per_shader( shader_core_ctx_t *shader);

#define N_PIPELINE_STAGES 7
#define TS_IF 0
#define IF_ID 1
#define ID_RR 2
#define ID_EX 3
#define RR_EX 3
#define EX_MM 4
#define MM_WB 5
#define WB_RT 6


#endif /* SHADER_H */
