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
#include <assert.h>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <bitset>
#include <utility>
#include <algorithm>
#include <deque>

#include "../cuda-sim/ptx.tab.h"
#include "../cuda-sim/dram_callback.h"

#include "gpu-cache.h"
#include "delayqueue.h"
#include "stack.h"
#include "dram.h"
#include "../abstract_hardware_model.h"
#include "scoreboard.h"
#include "mem_fetch.h"
#include "stats.h"

#ifndef SHADER_H
#define SHADER_H

#define NO_OP_FLAG            0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel, so good for now),
          2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits if req_size variable) - so up to 2^14 = 16384 mshr total 
 */

#define READ_PACKET_SIZE 8

//WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous. 
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8
#define NO_PARTIAL_WRITE (partial_write_mask_t())

#define WORD_SIZE 4

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32

class thread_ctx_t {
public:
   class ptx_thread_info *m_functional_model_thread_state; 
   unsigned m_cta_id; // hardware CTA this thread belongs

   // used for controlling fetch
   bool m_avail4fetch;  // false if instruction from thread is in pipeline
   bool m_in_scheduler; // DWF error checking
   bool m_waiting_at_barrier; // DWF and MIMD models
   bool m_reached_barrier; // DWF only

   // per thread stats (ac stands for accumulative).
   unsigned n_insn;
   unsigned n_insn_ac;
   unsigned n_l1_mis_ac;
   unsigned n_l1_mrghit_ac;
   unsigned n_l1_access_ac; 
};

class shd_warp_t {
public:
    shd_warp_t( class shader_core_ctx *shader, unsigned warp_size) 
        : m_shader(shader), m_warp_size(warp_size)
    {
        m_stores_outstanding=0;
        m_inst_in_pipeline=0;
        reset(); 
    }
    void reset()
    {
        assert( m_stores_outstanding==0);
        assert( m_inst_in_pipeline==0);
        m_imiss_pending=false;
        m_warp_id=(unsigned)-1;
        n_completed = m_warp_size; 
        n_avail4fetch = n_waiting_at_barrier = 0; 
        m_n_atomic=0;
        m_membar=false;
        m_done_exit=false;
        m_last_fetch=0;
        m_next=0;
        for(unsigned i=0;i<IBUFFER_SIZE;i++) 
            m_ibuffer[i]=NULL; 
    }
    void init( address_type start_pc, unsigned wid, unsigned active )
    {
        m_warp_id=wid;
        m_next_pc=start_pc;
        assert( n_completed >= active );
        assert( n_completed <= m_warp_size);
        assert( n_avail4fetch < m_warp_size );
        n_completed   -= active; // active threads are not yet completed
        n_avail4fetch += active; // number of threads in warp available to be fetched
    }

    bool done();
    bool waiting();

    bool done_exit() const { return m_done_exit; }
    void set_done_exit() { m_done_exit=true; }

    void print( FILE *fout ) const;
    void print_ibuffer( FILE *fout ) const;

    unsigned get_avail4fetch() const { return n_avail4fetch; }
    void inc_avail4fetch() { n_avail4fetch++; }
    void dec_avail4fetch() { n_avail4fetch--; }

    unsigned get_n_completed() const { return n_completed; }
    void inc_n_completed() { n_completed++; }

    void set_last_fetch( unsigned long long sim_cycle ) { m_last_fetch=sim_cycle; }

    unsigned get_n_atomic() const { return m_n_atomic; }
    void inc_n_atomic() { m_n_atomic++; }
    void dec_n_atomic() { m_n_atomic--; }

    void inc_waiting_at_barrier() { n_waiting_at_barrier++; }
    void clear_waiting_at_barrier() { n_waiting_at_barrier=0; }

    void set_membar() { m_membar=true; }
    void clear_membar() { m_membar=false; }
    bool get_membar() const { return m_membar; }
    address_type get_pc() const { return m_next_pc; }
    void set_next_pc( address_type pc ) { m_next_pc = pc; }

    void ibuffer_fill( unsigned slot, const inst_t *pI )
    {
       assert(slot < IBUFFER_SIZE );
       m_ibuffer[slot]=pI;
       m_next=0; 
    }
    bool ibuffer_empty() const
    {
        for( unsigned i=0; i < IBUFFER_SIZE; i++) 
            if(m_ibuffer[i]) 
                return false;
        return true;
    }
    void ibuffer_flush()
    {
        for(unsigned i=0;i<IBUFFER_SIZE;i++) {
            if( m_ibuffer[i] )
                dec_inst_in_pipeline();
            m_ibuffer[i]=NULL; 
        }
    }
    const inst_t *ibuffer_next()
    {
        const inst_t *result = m_ibuffer[m_next];
        return result;
    }
    void ibuffer_free()
    {
        m_ibuffer[m_next] = NULL;
    }
    void ibuffer_step()
    {
        m_next = (m_next+1)%IBUFFER_SIZE;
    }
    bool imiss_pending() const { return m_imiss_pending!=NULL; }
    void set_imiss_pending( class mshr_entry *mshr )
    { 
        m_imiss_pending=mshr; 
    }
    void clear_imiss_pending() { m_imiss_pending=NULL; }

    bool stores_done() const { return m_stores_outstanding == 0; }
    void inc_store_req() { m_stores_outstanding++; }
    void dec_store_req() 
    {
        assert( m_stores_outstanding > 0 );
        m_stores_outstanding--;
    }

    bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }

    void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
    void dec_inst_in_pipeline() 
    {
        assert( m_inst_in_pipeline > 0 );
        m_inst_in_pipeline--;
    }

private:
    static const unsigned IBUFFER_SIZE=2;
    class shader_core_ctx *m_shader;
    unsigned m_warp_id;
    unsigned m_warp_size;

    address_type m_next_pc;
    unsigned n_completed;          // number of threads in warp completed
    unsigned n_avail4fetch;        // number of threads in warp available to fetch 

    class mshr_entry *m_imiss_pending;
                                  
    const inst_t *m_ibuffer[IBUFFER_SIZE]; 
    unsigned m_next;
                                   
    int      n_waiting_at_barrier; // number of threads in warp that have reached the barrier
    unsigned m_n_atomic;           // number of outstanding atomic operations 
    bool     m_membar;             // if true, warp is waiting at memory barrier

    bool m_done_exit; // true once thread exit has been registered for threads in this warp

    unsigned long long m_last_fetch;

    unsigned m_stores_outstanding; // number of store requests sent but not yet acknowledged
    unsigned m_inst_in_pipeline;
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i){return wid * warp_size + i;};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size){return tid/warp_size;};

// bounded stack that implements pdom reconvergence (see MICRO'07 paper)
class pdom_warp_ctx_t {
public:
    pdom_warp_ctx_t( unsigned wid, class shader_core_ctx *shdr );

    void reset();
    void launch( address_type start_pc, unsigned active_mask );
    void pdom_update_warp_mask();

    unsigned get_active_mask() const;
    void get_pdom_stack_top_info( unsigned *pc, unsigned *rpc );
    unsigned get_rp() const;
    void print(FILE*fp) const;

private:
    unsigned m_warp_id;
    class shader_core_ctx *m_shader;
    unsigned m_stack_top;
    unsigned m_warp_size;
    
    address_type *m_pc;
    unsigned int *m_active_mask;
    address_type *m_recvg_pc;
    unsigned int *m_calldepth;
    
    unsigned long long  *m_branch_div_cycle;
};


enum mshr_status {
   INITIALIZED = 0,
   INVALID,
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

class mshr_entry {
public:
   mshr_entry() 
   { 
       m_status = INVALID; 
       m_merged_requests=NULL;
       m_mf=NULL;
       m_id=0;
   }
   void set_id( unsigned n ) { m_id = n; }
   void init( new_addr_type address, bool wr, memory_space_t space, unsigned warp_id );
   void clear() { m_insts.clear(); }
   void set_mf( class mem_fetch *mf ) { m_mf=mf; }
   void add_inst( inst_t inst ) { m_insts.push_back(inst); }
   void set_status( enum mshr_status status );
   void merge( mshr_entry *mshr )
   {
       //merge this request;
       m_merged_requests = mshr;
       mshr->m_merged_on_other_reqest = true;
   }

   dram_callback_t &get_atomic_callback() 
   {
       assert(isatomic());
       return m_insts[0].callback;
   }
   mshr_entry *get_last_merged()
   {
       assert(m_status!=INVALID);
       mshr_entry *mshr_hit = this;
       while (mshr_hit->m_merged_requests) 
          mshr_hit = mshr_hit->m_merged_requests;
       return mshr_hit;
   }
   void get_insts( std::vector<inst_t> &done_insts )
   {
       done_insts.insert(done_insts.end(),m_insts.begin(),m_insts.end());
   }
   void add_to_queue( std::deque<mshr_entry*> &q )
   {
       // place all merged requests in return queue
       mshr_entry *req = this;
       while (req) {
           q.push_back(req);
           req = req->m_merged_requests;
       } 
   }

   unsigned get_warp_id() const { return m_warp_id; }
   bool ismerged() const { return m_merged_on_other_reqest; }
   bool fetched() const { return m_status == FETCHED;};
   bool iswrite() const { return m_iswrite; }
   bool isinst()  const { return m_isinst; }
   bool istexture() const { return m_istexture; }
   bool isconst() const { return m_isconst; }
   bool islocal() const { return m_islocal; }
   bool has_inst() const { return m_insts.size()>0; }
   unsigned num_inst() const { return m_insts.size(); }
   inst_t &get_inst(unsigned n) 
   { 
       assert(m_status!=INVALID&&m_insts.size()>0);
       return m_insts[n]; 
   }
   unsigned get_insts_uid() const 
   {
       assert(m_status!=INVALID&&m_insts.size()>0);
       return m_insts[0].uid;
   }
   bool isatomic() const
   {
       assert(m_status!=INVALID);
       if( isinst() ) 
           return false;
       assert(m_insts.size()>0);
       return (m_insts[0].callback.function != NULL);
   }
   new_addr_type get_addr() const { return m_addr; }
   void print(FILE *fp, unsigned mask) const;
    
private:
    unsigned m_id;
    unsigned m_request_uid;
    unsigned m_warp_id;
    new_addr_type m_addr; // address being fetched
    std::vector<inst_t> m_insts;
    bool m_iswrite;
    bool m_merged_on_other_reqest; //true if waiting for another mshr - this mshr doesn't send a memory request
    struct mshr_entry *m_merged_requests; //mshrs waiting on this mshr
    enum mshr_status m_status; 
    class mem_fetch *m_mf; // link to corresponding memory fetch structure
    bool m_isinst;    //if it's a request from the instruction cache
    bool m_istexture; //if it's a request from the texture cache
    bool m_isconst;   //if it's a request from the constant cache
    bool m_islocal;   //if it's a request to the local memory of a thread
    bool m_wt_no_w2cache; //in write_through, sometimes need to prevent writing back returning data into cache, because its been written in the meantime. 
};

const unsigned WARP_PER_CTA_MAX = 32;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

int register_bank(int regnum, int tid, unsigned num_banks, unsigned bank_warp_shift);

class shader_core_ctx;

class opndcoll_rfu_t { // operand collector based register file unit
public:
   // constructors
   opndcoll_rfu_t()
   {
      m_num_collectors=0;
      m_num_banks=0;
      m_cu = NULL;
      m_shader=NULL;
      m_sfu_port=NULL;
      m_alu_port=NULL;
   }
   void init( unsigned num_collectors_alu, 
              unsigned num_collectors_sfu, 
              unsigned num_banks, 
              shader_core_ctx *shader,
              inst_t **alu_port,
              inst_t **sfu_port );

   // modifiers
   bool writeback( const inst_t &fvt );
   bool writeback( inst_t *warp ); // might cause stall 

   void step( inst_t *&id_oc_reg ) 
   {
      dispatch_ready_cu();   
      allocate_reads();
      allocate_cu(id_oc_reg);
      process_banks();
   }

   void step( inst_t *&alu_issue_port, inst_t *&sfu_issue_port ) 
   {
      dispatch_ready_cu();   
      allocate_reads();
      allocate_cu(alu_issue_port);
      allocate_cu(sfu_issue_port);
      process_banks();
   }

   void dump( FILE *fp ) const
   {
      fprintf(fp,"\n");
      fprintf(fp,"Operand Collector State:\n");
      for( unsigned n=0; n < m_num_collectors; n++ ) {
         fprintf(fp,"   CU-%2u: ", n);
         m_cu[n].dump(fp,m_shader);
      }
      m_arbiter.dump(fp);
   }

   shader_core_ctx *shader_core() { return m_shader; }

private:

   void process_banks()
   {
      m_arbiter.reset_alloction();
   }

   void dispatch_ready_cu();
   void allocate_cu( inst_t *&id_oc_reg );
   void allocate_reads();

   // types

   class collector_unit_t;

   class op_t {
   public:

      op_t() { m_valid = false; }
      op_t( collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid = true;
         m_fvi=NULL;
         m_cu = cu;
         m_operand = op;
         m_register = reg;
         m_tid = cu->get_tid();
         m_bank = register_bank(reg,m_tid,num_banks,bank_warp_shift);
      }
      op_t( const inst_t *fvi, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid=true;
         m_fvi=fvi;
         m_register=reg;
         m_cu=NULL;
         m_operand = -1;
         m_tid = fvi->hw_thread_id;
         m_bank = register_bank(reg,m_tid,num_banks,bank_warp_shift);
      }

      // accessors
      bool valid() const { return m_valid; }
      unsigned get_reg() const
      {
         assert( m_valid );
         return m_register;
      }
      unsigned get_oc_id() const { return m_cu->get_id(); }
      unsigned get_tid() const { return m_tid; }
      unsigned get_bank() const { return m_bank; }
      unsigned get_operand() const { return m_operand; }
      void dump(FILE *fp) const 
      {
         if(m_cu) 
            fprintf(fp," <R%u, CU:%u, w:%02u> ", m_register,m_cu->get_id(),m_cu->get_warp_id());
         else if( m_fvi )
            fprintf(fp," <R%u, fvi tid:%02u> ", m_register,m_fvi->hw_thread_id );
      }
      std::string get_reg_string() const
      {
         char buffer[64];
         snprintf(buffer,64,"R%u", m_register);
         return std::string(buffer);
      }

      // modifiers
      void reset() { m_valid = false; }
   private:
      bool m_valid;
      collector_unit_t  *m_cu; 
      const inst_t      *m_fvi;
      unsigned  m_operand; // operand offset in instruction. e.g., add r1,r2,r3; r2 is oprd 0, r3 is 1 (r1 is dst)
      unsigned  m_register;
      unsigned  m_bank;
      unsigned  m_tid;
   };

   enum alloc_t {
      NO_ALLOC,
      READ_ALLOC,
      WRITE_ALLOC,
   };

   class allocation_t {
   public:
      allocation_t() { m_allocation = NO_ALLOC; }
      bool is_read() const { return m_allocation==READ_ALLOC; }
      bool is_write() const {return m_allocation==WRITE_ALLOC; }
      bool is_free() const {return m_allocation==NO_ALLOC; }
      void dump(FILE *fp) const {
         if( m_allocation == NO_ALLOC ) { fprintf(fp,"<free>"); }
         else if( m_allocation == READ_ALLOC ) { fprintf(fp,"rd: "); m_op.dump(fp); }
         else if( m_allocation == WRITE_ALLOC ) { fprintf(fp,"wr: "); m_op.dump(fp); }
         fprintf(fp,"\n");
      }
      void alloc_read( const op_t &op )  { assert(is_free()); m_allocation=READ_ALLOC; m_op=op; }
      void alloc_write( const op_t &op ) { assert(is_free()); m_allocation=WRITE_ALLOC; m_op=op; }
      void reset() { m_allocation = NO_ALLOC; }
   private:
      enum alloc_t m_allocation;
      op_t m_op;
   };

   class arbiter_t {
   public:
      // constructors
      arbiter_t()
      {
         m_queue=NULL;
         m_allocated_bank=NULL;
         m_allocator_rr_head=NULL;
      }
      void init( unsigned num_cu, unsigned num_banks ) 
      { 
         m_num_collectors = num_cu;
         m_num_banks = num_banks;
         m_queue = new std::list<op_t>[num_banks];
         m_allocated_bank = new allocation_t[num_banks];
         m_allocator_rr_head = new unsigned[num_cu];
         for( unsigned n=0; n<num_cu;n++ ) 
            m_allocator_rr_head[n] = n%num_banks;
         reset_alloction();
      }

      // accessors
      void dump(FILE *fp) const
      {
         fprintf(fp,"\n");
         fprintf(fp,"  Arbiter State:\n");
         fprintf(fp,"  requests:\n");
         for( unsigned b=0; b<m_num_banks; b++ ) {
            fprintf(fp,"    bank %u : ", b );
            std::list<op_t>::const_iterator o = m_queue[b].begin();
            for(; o != m_queue[b].end(); o++ ) {
               o->dump(fp);
            }
            fprintf(fp,"\n");
         }
         fprintf(fp,"  grants:\n");
         for(unsigned b=0;b<m_num_banks;b++) {
            fprintf(fp,"    bank %u : ", b );
            m_allocated_bank[b].dump(fp);
         }
         fprintf(fp,"\n");
      }

      // modifiers
      std::list<op_t> allocate_reads(); 

      void add_read_requests( collector_unit_t *cu ) 
      {
         const op_t *src = cu->get_operands();
         for( unsigned i=0; i<MAX_REG_OPERANDS; i++) {
            const op_t &op = src[i];
            if( op.valid() ) {
               unsigned bank = op.get_bank();
               m_queue[bank].push_back(op);
            }
         }
      }
      bool bank_idle( unsigned bank ) const
      {
          return m_allocated_bank[bank].is_free();
      }
      void allocate_bank_for_write( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_write(op);
      }
      void allocate_for_read( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_read(op);
      }
      void reset_alloction()
      {
         for( unsigned b=0; b < m_num_banks; b++ ) 
            m_allocated_bank[b].reset();
      }

   private:
      unsigned m_num_banks;
      unsigned m_num_collectors;

      allocation_t *m_allocated_bank; // bank # -> register that wins
      std::list<op_t> *m_queue;

      unsigned *m_allocator_rr_head; // cu # -> next bank to check for request (rr-arb)
      unsigned  m_last_cu; // first cu to check while arb-ing banks (rr)
   };

   class collector_unit_t {
   public:
      // constructors
      collector_unit_t()
      { 
         m_free = true;
         m_warp = NULL;
         m_src_op = new op_t[MAX_REG_OPERANDS];
         m_not_ready.reset();
         m_tid = -1;
         m_warp_id = -1;
         m_num_banks = 0;
         m_bank_warp_shift = 0;
      }
      // accessors
      bool ready() const;
      const op_t *get_operands() const { return m_src_op; }
      void dump(FILE *fp, const shader_core_ctx *shader ) const;

      unsigned get_tid() const { return m_tid; } // returns hw id of first valid instruction
      unsigned get_warp_id() const { return m_warp_id; }
      unsigned get_id() const { return m_cuid; } // returns CU hw id

      // modifiers
      void init(unsigned n, 
                inst_t **port, 
                unsigned num_banks, 
                unsigned log2_warp_size,
                unsigned warp_size,
                opndcoll_rfu_t *rfu ); 
      void allocate( inst_t *&pipeline_reg );

      void collect_operand( unsigned op )
      {
         m_not_ready.reset(op);
      }

      void dispatch();

   private:
      bool m_free;
      unsigned m_tid;
      unsigned m_cuid; // collector unit hw id
      inst_t **m_port; // pipeline register to issue to when ready
      unsigned m_warp_id;
      inst_t  *m_warp;
      op_t *m_src_op;
      std::bitset<MAX_REG_OPERANDS> m_not_ready;
      unsigned m_num_banks;
      unsigned m_bank_warp_shift;
      opndcoll_rfu_t *m_rfu;
   };

   class dispatch_unit_t {
   public:
      dispatch_unit_t() 
      { 
         m_last_cu=0;
         m_num_collectors=0;
         m_collector_units=NULL;
         m_next_cu=0;
      }

      void init( unsigned num_collectors )
      { 
         m_num_collectors = num_collectors;
         m_collector_units = new collector_unit_t * [num_collectors];
         m_next_cu=0;
      }

      void add_cu( collector_unit_t *cu )
      {
         assert(m_next_cu<m_num_collectors);
         m_collector_units[m_next_cu] = cu;
         m_next_cu++;
      }

      collector_unit_t *find_ready()
      {
         for( unsigned n=0; n < m_num_collectors; n++ ) {
            unsigned c=(m_last_cu+n+1)%m_num_collectors;
            if( m_collector_units[c]->ready() ) {
               m_last_cu=c;
               return m_collector_units[c];
            }
         }
         return NULL;
      }

   private:
      unsigned m_num_collectors;
      collector_unit_t **m_collector_units;
      unsigned m_last_cu; // dispatch ready cu's rr
      unsigned m_next_cu;  // for initialization
   };

   // opndcoll_rfu_t data members

   unsigned                         m_num_collectors;
   unsigned                         m_num_banks;
   unsigned                         m_bank_warp_shift;
   unsigned                         m_warp_size;
   collector_unit_t                *m_cu;
   arbiter_t                        m_arbiter;

   inst_t **m_alu_port;
   inst_t **m_sfu_port;

   typedef std::map<inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
   port_to_du_t                     m_dispatch_units;
   std::map<inst_t**,std::list<collector_unit_t*> > m_free_cu;
   shader_core_ctx                 *m_shader;
};

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
   bool warp_waiting_at_barrier( unsigned warp_id ) const;

   // debug
   void dump() const;

private:
   unsigned m_max_cta_per_core;
   unsigned m_max_warps_per_core;

   cta_to_warp_t m_cta_to_warps; 
   warp_set_t m_warp_active;
   warp_set_t m_warp_at_barrier;
};

class warp_tracker;
class warp_tracker_pool;

enum memory_pipe_t {
   NO_MEM_PATH = 0,
   SHARED_MEM_PATH,
   GLOBAL_MEM_PATH,
   TEXTURE_MEM_PATH,
   CONSTANT_MEM_PATH,
   NUM_MEM_PATHS //not a mem path
};

class mem_access_t {
public:
   mem_access_t() : space(undefined_space)
   { 
      init();
   }
   mem_access_t(address_type a, memory_space_t s, memory_pipe_t p, bool atomic, bool w, unsigned r, unsigned quarter, unsigned idx )
   {
      init();
      addr = a;
      space = s;
      mem_pipe = p;
      isatomic = atomic;
      iswrite = w;
      req_size = r;
      quarter_count[quarter]++;
      warp_indices.push_back(idx);
   }

   bool operator<(const mem_access_t &other) const {return (order > other.order);}//this is reverse

private:
   void init() 
   {
      uid=++next_access_uid;
      addr=0;
      req_size=0;
      order=0;
      _quarter_count_all=0;
      mem_pipe = NO_MEM_PATH;
      isatomic = false;
      cache_hit = false;
      cache_checked = false;
      recheck_cache = false;
      iswrite = false;
      need_wb = false;
      wb_addr = 0;
      reserved_mshr = NULL;
   }

public:

   unsigned uid;
   address_type addr; //address of the segment to load.
   unsigned req_size; //bytes
   unsigned order; // order of accesses, based on banks.
   union{
     unsigned _quarter_count_all;
     char quarter_count[4]; //access counts to each quarter of segment, for compaction;
   };
   std::vector<unsigned> warp_indices; // warp indicies for this request.
   memory_space_t space;
   memory_pipe_t  mem_pipe;
   bool isatomic;
   bool cache_hit;
   bool cache_checked;
   bool recheck_cache;
   bool iswrite;
   bool need_wb;
   address_type wb_addr; // writeback address (if necessary).
   mshr_entry* reserved_mshr;

private:
   static unsigned next_access_uid;
};

class mshr_lookup {
public:
   mshr_lookup( const struct shader_core_config *config ) { m_shader_config = config; }
   bool can_merge(mshr_entry * mshr);
   void mshr_fast_lookup_insert(mshr_entry* mshr);
   void mshr_fast_lookup_remove(mshr_entry* mshr);
   mshr_entry* shader_get_mergeable_mshr(mshr_entry* mshr);

private:
   void insert(mshr_entry* mshr);
   mshr_entry* lookup(new_addr_type addr) const;
   void remove(mshr_entry* mshr);

   typedef std::multimap<new_addr_type, mshr_entry*> mshr_lut_t; // multimap since multiple mshr entries can have the same tag

   const shader_core_config *m_shader_config;
   mshr_lut_t m_lut; 
};

class mshr_shader_unit {
public:
   mshr_shader_unit( const shader_core_config *config );

   bool has_mshr(unsigned num)
   {
       return (num <= m_free_list.size());
   }

   //return queue access; (includes texture pipeline return)
   mshr_entry* return_head();

   //return queue pop; (includes texture pipeline return)
   void pop_return_head();

   mshr_entry* add_mshr(mem_access_t &access, inst_t* warp);
   void mshr_return_from_mem(mshr_entry *mshr);
   unsigned get_max_mshr_used() const {return m_max_mshr_used;}  
   void print(FILE* fp, class shader_core_ctx* shader,unsigned mask);

private:
   mshr_entry *alloc_free_mshr(bool istexture);
   void free_mshr( mshr_entry *mshr );
   unsigned mshr_used() const;
   bool has_return() 
   { 
       return (not m_mshr_return_queue.empty()) or 
              ((not m_texture_mshr_pipeline.empty()) and m_texture_mshr_pipeline.front()->fetched());
   }
   std::deque<mshr_entry*> &choose_return_queue();

   const struct shader_core_config *m_shader_config;

   typedef std::vector<mshr_entry> mshr_storage_type;
   mshr_storage_type m_mshrs; 
   std::deque<mshr_entry*> m_free_list;
   std::deque<mshr_entry*> m_mshr_return_queue;
   std::deque<mshr_entry*> m_texture_mshr_pipeline;
   unsigned m_max_mshr_used;
   mshr_lookup m_mshr_lookup;
};

struct shader_queues_t {
   std::vector<mem_access_t> shared;
   std::vector<mem_access_t> constant;
   std::vector<mem_access_t> texture;
   std::vector<mem_access_t> local_global;
};

struct insn_latency_info {
   unsigned pc;
   unsigned long latency;
};

struct ifetch_buffer_t {
    ifetch_buffer_t() { m_valid=false; }

    ifetch_buffer_t( address_type pc, unsigned nbytes, unsigned warp_id ) 
    { 
        m_valid=true; 
        m_pc=pc; 
        m_nbytes=nbytes; 
        m_warp_id=warp_id;
    }

    bool m_valid;
    address_type m_pc;
    unsigned m_nbytes;
    unsigned m_warp_id;
};

// Struct for storing warp information in fixeddelay_queue
struct fixeddelay_queue_warp_t {
   unsigned long long ready_cycle;
   std::vector<int> tids; // list of tid's in this warp (to unlock)
     inst_t inst;
};

struct fixeddelay_queue_warp_comp {
   inline bool operator()(const fixeddelay_queue_warp_t& left,const fixeddelay_queue_warp_t& right) const
   {
       return left.ready_cycle < right.ready_cycle;
   }
};

typedef address_type (*tag_func_t)(address_type add, unsigned line_size);

class shader_core_ctx : public core_t 
{
public:
   shader_core_ctx( class gpgpu_sim *gpu,
                    const char *name, 
                    unsigned shader_id,
                    unsigned tpc_id,
                    const struct shader_core_config *config,
                    struct shader_core_stats *stats );

   void issue_block2core( class kernel_info_t &kernel );
   void get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc );
   void new_cache_window();
   bool ptx_thread_done( unsigned hw_thread_id ) const;
   class ptx_thread_info *get_thread_state( unsigned hw_thread_id );

   virtual void set_at_barrier( unsigned cta_id, unsigned warp_id );
   virtual void warp_exit( unsigned warp_id );
   virtual bool warp_waiting_at_barrier( unsigned warp_id ) const;
   virtual bool warp_waiting_for_atomics( unsigned warp_id ) const;
   virtual class gpgpu_sim *get_gpu();
   void set_at_memory_barrier( unsigned warp_id );
   bool warp_waiting_at_mem_barrier( unsigned warp_id );
   void allocate_barrier( unsigned cta_id, warp_set_t warps );
   void deallocate_barrier( unsigned cta_id );
   void decrement_atomic_count( unsigned wid );

   void cycle();
   void cycle_gt200();

   void reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed );
   void init_warps(unsigned start_thread, unsigned end_thread);

   unsigned max_cta( class function_info *kernel );
   void cache_flush();
   void display_pdom_state(FILE *fout, int mask );
   void display_pipeline( FILE *fout, int print_mem, int mask3bit );
   void register_cta_thread_exit(int cta_num );
   void fill_shd_L1_with_new_line( class mem_fetch * mf );
   void store_ack( class mem_fetch *mf );
   void dump_istream_state( FILE *fout );
   void mshr_print(FILE* fp, unsigned mask);
   inst_t *first_valid_thread( inst_t *warp );
   inst_t *first_valid_thread( unsigned stage );
   class ptx_thread_info* get_functional_thread( unsigned tid ) { return m_thread[tid].m_functional_model_thread_state; }
   void move_warp( inst_t *&dst, inst_t *&src );
   void print_warp( inst_t *warp, FILE *fout, int print_mem, int mask ) const;
   void clear_stage(inst_t *warp);
   std::list<unsigned> get_regs_written( const inst_t &fvt ) const;
   bool pipeline_regster_empty( inst_t *reg );
   const shader_core_config *get_config() const { return m_config; }
   unsigned get_num_sim_insn() const { return m_num_sim_insn; }
   int get_not_completed() const { return m_not_completed; }
   unsigned get_n_diverge() const { return m_n_diverge; }
   unsigned get_thread_n_insn( unsigned tid ) const { return m_thread[tid].n_insn; }
   unsigned get_thread_n_insn_ac( unsigned tid ) const { return m_thread[tid].n_insn_ac; }
   unsigned get_thread_n_l1_mis_ac( unsigned tid ) const { return m_thread[tid].n_l1_mis_ac; }
   unsigned get_thread_n_l1_mrghit_ac( unsigned tid ) const { return m_thread[tid].n_l1_mrghit_ac; }
   unsigned get_thread_n_l1_access_ac( unsigned tid ) const { return m_thread[tid].n_l1_access_ac; }
   unsigned get_max_mshr_used() const { return m_mshr_unit->get_max_mshr_used(); }
   void L1cache_print( FILE *fp, unsigned &total_accesses, unsigned &total_misses) const;
   void L1texcache_print( FILE *fp, unsigned &total_accesses, unsigned &total_misses) const;
   void L1constcache_print( FILE *fp, unsigned &total_accesses, unsigned &total_misses) const;
   unsigned get_n_active_cta() const { return m_n_active_cta; }
   float L1_windowed_cache_miss_rate( int x ) const { return shd_cache_windowed_cache_miss_rate(m_L1D,x); }
   float L1tex_windowed_cache_miss_rate( int x ) const { return shd_cache_windowed_cache_miss_rate(m_L1T,x); }
   float L1const_windowed_cache_miss_rate( int x ) const { return shd_cache_windowed_cache_miss_rate(m_L1C,x); }
  
private:

   void clear_stage_reg(int stage);

   address_type next_pc( int tid ) const;

   void ptx_exec_inst( inst_t &inst );
   void fetch_new();
   void issue_warp(const inst_t *pI, unsigned active_mask, inst_t *&warp, unsigned warp_id );
   void decode_new();

   void fetch();

   void fetch_mimd();
   void fetch_simd_dwf();
   void fetch_simd_postdominator();
   int  pdom_sched_find_next_warp (int ready_warp_count);
   bool fetch_stalled();
   void shader_issue_thread(int tid, int wlane, unsigned active_mask );
   int warp_reached_barrier(int *tid_in);

   void decode();

   void preexecute();

   void execute();
   void execute_pipe( unsigned pipeline, unsigned next_stage );

   void pre_memory();

   void memory(); // advance memory pipeline stage
   void memory_queue();
   void memory_shared_process_warp(); 
   void memory_const_process_warp();
   void memory_texture_process_warp();
   void memory_global_process_warp();
   bool memory_shared_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_constant_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_texture_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_cycle( mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   address_type translate_local_memaddr(address_type localaddr, int tid, unsigned num_shader );

   mem_stage_stall_type ccache_check(mem_access_t& access){ return NO_RC_FAIL;}
   mem_stage_stall_type tcache_check(mem_access_t& access){ return NO_RC_FAIL;}
   mem_stage_stall_type dcache_check(mem_access_t& access);

   typedef mem_stage_stall_type (shader_core_ctx::*cache_check_t)(mem_access_t&);

   mem_stage_stall_type process_memory_access_queue( shader_core_ctx::cache_check_t cache_check,
                                                     unsigned ports_per_bank, 
                                                     unsigned memory_send_max, 
                                                     std::vector<mem_access_t> &accessq );

   typedef int (shader_core_ctx::*bank_func_t)(address_type add, unsigned line_size);
   
   int null_bank_func(address_type add, unsigned line_size);
   int shmem_bank_func(address_type add, unsigned line_size);
   int dcache_bank_func(address_type add, unsigned line_size);

   void get_memory_access_list( bank_func_t bank_func,
                                tag_func_t tag_func,
                                memory_pipe_t mem_pipe, 
                                unsigned warp_parts, 
                                unsigned line_size, 
                                bool limit_broadcast, 
                                std::vector<mem_access_t> &accessq );
   mem_stage_stall_type send_mem_request(mem_access_t &access);

   void check_stage_pcs( unsigned stage );
   void check_pm_stage_pcs( unsigned stage );

   void writeback();
   int  split_warp_by_pc(int *tid_in, int **tid_split, address_type *pc);
   int  split_warp_by_cta(int *tid_in, int **tid_split, address_type *pc, int *cta);

   unsigned char fq_push( unsigned long long int addr, 
                          int bsize, 
                          unsigned char write, 
                          partial_write_mask_t partial_write_mask, 
                          int wid, mshr_entry* mshr, 
                          int cache_hits_waiting,
                          enum mem_access_type mem_acc, 
                          address_type pc );

   bool warp_scoreboard_hazard(int warp_id);
   mshr_entry* check_mshr4tag(unsigned long long int addr,int mem_type);
   void update_mshr(unsigned long long int fetched_addr, unsigned int mshr_idx, int mem_type );
   void visualizer_dump(FILE *fp);
   void clean(unsigned int n_threads);
   void call_thread_done(inst_t &done_inst );
   void queue_warp_unlocking(int *tids, const inst_t &inst );
   void process_delay_queue();
   void unlock_warp(std::vector<int> tids );

   void print_pre_mem_stages( FILE *fout, int print_mem, int mask );
   void print_stage(unsigned int stage, FILE *fout, int print_mem, int mask );
   void print_shader_cycle_distro( FILE *fout );

   // general information
   unsigned m_sid; // shader id
   unsigned m_tpc; // texture processor cluster id (aka, node id when using interconnect concentration)
   const char *m_name;
   const shader_core_config *m_config;
   class gpgpu_sim *m_gpu;

   // statistics 
   struct shader_core_stats *m_stats; // pointer to single object shared by all shader cores in GPU
   unsigned int m_num_sim_insn; // number of instructions committed by this shader core
   unsigned int m_n_diverge; // number of divergence occurred in this shader

   // CTA scheduling / hardware thread allocation
   int m_n_active_cta; // number of Cooperative Thread Arrays (blocks) currently running on this shader.
   int m_cta_status[MAX_CTA_PER_SHADER]; // CTAs status 
   int m_not_completed; // number of threads to be completed (==0 when all thread on this core completed) 

   // thread contexts 
   thread_ctx_t             *m_thread; // functional state, per thread fetch state
   std::vector<shd_warp_t>   m_warp;   // per warp information array
   barrier_set_t             m_barriers;
   ifetch_buffer_t           m_inst_fetch_buffer;
   pdom_warp_ctx_t         **m_pdom_warp; // pdom reconvergence context for each warp

   class warp_tracker_pool *m_warp_tracker;
   inst_t** m_pipeline_reg;
   inst_t** pre_mem_pipeline;
   Scoreboard *m_scoreboard;
   opndcoll_rfu_t m_operand_collector;
   mshr_shader_unit *m_mshr_unit;
   shader_queues_t m_memory_queue;
   fifo_pipeline<std::vector<int> > *m_thd_commit_queue;
   std::multiset<fixeddelay_queue_warp_t, fixeddelay_queue_warp_comp> m_fixeddelay_queue;

   // fetch
   int  m_last_warp_fetched;
   int  m_last_warp_issued;

   bool m_new_warp_TS; // new warp at TS pipeline register
   int  m_last_warp;   // SIMT: last warp issued
   int  m_next_warp;   // SIMT: Keeps track of which warp of instructions to fetch/execute
   unsigned m_last_issued_thread; // MIMD

   int *m_ready_warps;
   int *m_tmp_ready_warps;
   int *m_fetch_tid_out;

   // pre-execute stage
   int  m_dwf_RR_k;          // counter for register read pipeline
   int *m_dwf_rrstage_bank_access_counter;

   shd_cache_t *m_L1I; // instruction cache
   shd_cache_t *m_L1D; // data cache (global/local memory accesses)
   shd_cache_t *m_L1T; // texture cache
   shd_cache_t *m_L1C; // constant cache

   bool m_shader_memory_new_instruction_processed;
   int m_pending_mem_access; // number of memory access to be serviced (use for W0 classification)

   // used in writeback
   int *m_pl_tid;
   insn_latency_info *m_mshr_lat_info;
   insn_latency_info *m_pl_lat_info;

   class thread_pc_tracker *m_thread_pc_tracker;
};

void init_mshr_pool();
mshr_entry* alloc_mshr_entry();
void free_mshr_entry( mshr_entry * );

// print out the accumulative statistics for shaders (those that are not local to one shader)
void shader_print_runtime_stat( FILE *fout );
void shader_print_l1_miss_stat( FILE *fout );

#define TS_IF 0
#define IF_ID 1
#define ID_RR 2
#define ID_EX 3
#define RR_EX 3
#define EX_MM 4
#define MM_WB 5
#define WB_RT 6
#define ID_OC 7 
#define ID_OC_SFU 8 
#define OC_EX_SFU 9
#define N_PIPELINE_STAGES 10

extern unsigned int *shader_cycle_distro;
extern unsigned int n_regconflict_stall;

int is_store ( const inst_t &op );

#endif /* SHADER_H */
