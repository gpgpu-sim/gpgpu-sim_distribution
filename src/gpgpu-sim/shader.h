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

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32

class thread_ctx_t {
public:
   class ptx_thread_info *m_functional_model_thread_state; 
   unsigned m_cta_id; // hardware CTA this thread belongs

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
        m_n_atomic=0;
        m_membar=false;
        m_done_exit=false;
        m_last_fetch=0;
        m_next=0;
    }
    void init( address_type start_pc, unsigned cta_id, unsigned wid, unsigned active )
    {
        m_cta_id=cta_id;
        m_warp_id=wid;
        m_next_pc=start_pc;
        assert( n_completed >= active );
        assert( n_completed <= m_warp_size);
        n_completed   -= active; // active threads are not yet completed
    }

    bool done();
    bool waiting();

    bool done_exit() const { return m_done_exit; }
    void set_done_exit() { m_done_exit=true; }

    void print( FILE *fout ) const;
    void print_ibuffer( FILE *fout ) const;

    unsigned get_n_completed() const { return n_completed; }
    void inc_n_completed() { n_completed++; }

    void set_last_fetch( unsigned long long sim_cycle ) { m_last_fetch=sim_cycle; }

    unsigned get_n_atomic() const { return m_n_atomic; }
    void inc_n_atomic() { m_n_atomic++; }
    void dec_n_atomic() { m_n_atomic--; }

    void set_membar() { m_membar=true; }
    void clear_membar() { m_membar=false; }
    bool get_membar() const { return m_membar; }
    address_type get_pc() const { return m_next_pc; }
    void set_next_pc( address_type pc ) { m_next_pc = pc; }

    void ibuffer_fill( unsigned slot, const warp_inst_t *pI )
    {
       assert(slot < IBUFFER_SIZE );
       m_ibuffer[slot].m_inst=pI;
       m_ibuffer[slot].m_valid=true;
       m_next=0; 
    }
    bool ibuffer_empty() const
    {
        for( unsigned i=0; i < IBUFFER_SIZE; i++) 
            if(m_ibuffer[i].m_valid) 
                return false;
        return true;
    }
    void ibuffer_flush()
    {
        for(unsigned i=0;i<IBUFFER_SIZE;i++) {
            if( m_ibuffer[i].m_valid )
                dec_inst_in_pipeline();
            m_ibuffer[i].m_inst=NULL; 
            m_ibuffer[i].m_valid=false; 
        }
    }
    const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
    bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
    void ibuffer_free()
    {
        m_ibuffer[m_next].m_inst = NULL;
        m_ibuffer[m_next].m_valid = false;
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
    unsigned get_cta_id() const { return m_cta_id; }

private:
    static const unsigned IBUFFER_SIZE=2;
    class shader_core_ctx *m_shader;
    unsigned m_cta_id;
    unsigned m_warp_id;
    unsigned m_warp_size;

    address_type m_next_pc;
    unsigned n_completed;          // number of threads in warp completed

    class mshr_entry *m_imiss_pending;
    
    struct ibuffer_entry {
       ibuffer_entry() { m_valid = false; m_inst = NULL; }
       const warp_inst_t *m_inst;
       bool m_valid;
    };
    ibuffer_entry m_ibuffer[IBUFFER_SIZE]; 
    unsigned m_next;
                                   
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
   class mem_fetch *get_mf() { return m_mf; }
   void add_inst( warp_inst_t inst ) { m_insts.push_back(inst); }
   void set_status( enum mshr_status status );
   void merge( mshr_entry *mshr )
   {
       //merge this request;
       m_merged_requests = mshr;
       mshr->m_merged_on_other_reqest = true;
   }
   void do_atomic() 
   {
       for( std::vector<warp_inst_t>::iterator e=m_insts.begin(); e != m_insts.end(); ++e ) {
           warp_inst_t &inst = *e;
           inst.do_atomic();
       }
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
   void add_to_queue( std::list<mshr_entry*> &q )
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
   bool isatomic() const
   {
       assert(m_status!=INVALID);
       if( isinst() ) 
           return false;
       assert(m_insts.size()>0);
       return m_insts[0].isatomic();
   }
   new_addr_type get_addr() const { return m_addr; }
   void print(FILE *fp) const;
    
private:
    unsigned m_id;
    unsigned m_request_uid;
    unsigned m_warp_id;
    new_addr_type m_addr; // address being fetched
    std::vector<warp_inst_t> m_insts;
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

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift);

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
      m_num_ports=0;
   }
   void add_port( unsigned num_collector_units,
                  warp_inst_t **input_port,
                  warp_inst_t **output_port );
   void init( unsigned num_banks, shader_core_ctx *shader );

   // modifiers
   bool writeback( const warp_inst_t &warp ); // might cause stall 

   void step()
   {
      dispatch_ready_cu();   
      allocate_reads();
      for( unsigned p=0; p < m_num_ports; p++ ) 
          allocate_cu( p );
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
   void allocate_cu( unsigned port );
   void allocate_reads();

   // types

   class collector_unit_t;

   class op_t {
   public:

      op_t() { m_valid = false; }
      op_t( collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid = true;
         m_warp=NULL;
         m_cu = cu;
         m_operand = op;
         m_register = reg;
         m_bank = register_bank(reg,cu->get_warp_id(),num_banks,bank_warp_shift);
      }
      op_t( const warp_inst_t *warp, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid=true;
         m_warp=warp;
         m_register=reg;
         m_cu=NULL;
         m_operand = -1;
         m_bank = register_bank(reg,warp->warp_id(),num_banks,bank_warp_shift);
      }

      // accessors
      bool valid() const { return m_valid; }
      unsigned get_reg() const
      {
         assert( m_valid );
         return m_register;
      }
      unsigned get_wid() const
      {
          if( m_warp ) return m_warp->warp_id();
          else if( m_cu ) return m_cu->get_warp_id();
          else abort();
          return 0;
      }
      unsigned get_oc_id() const { return m_cu->get_id(); }
      unsigned get_bank() const { return m_bank; }
      unsigned get_operand() const { return m_operand; }
      void dump(FILE *fp) const 
      {
         if(m_cu) 
            fprintf(fp," <R%u, CU:%u, w:%02u> ", m_register,m_cu->get_id(),m_cu->get_warp_id());
         else if( !m_warp->empty() )
            fprintf(fp," <R%u, wid:%02u> ", m_register,m_warp->warp_id() );
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
      const warp_inst_t *m_warp;
      unsigned  m_operand; // operand offset in instruction. e.g., add r1,r2,r3; r2 is oprd 0, r3 is 1 (r1 is dst)
      unsigned  m_register;
      unsigned  m_bank;
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
         m_warp_id = -1;
         m_num_banks = 0;
         m_bank_warp_shift = 0;
      }
      // accessors
      bool ready() const;
      const op_t *get_operands() const { return m_src_op; }
      void dump(FILE *fp, const shader_core_ctx *shader ) const;

      unsigned get_warp_id() const { return m_warp_id; }
      unsigned get_id() const { return m_cuid; } // returns CU hw id

      // modifiers
      void init(unsigned n, 
                warp_inst_t **port, 
                unsigned num_banks, 
                unsigned log2_warp_size,
                const shader_core_config *config,
                opndcoll_rfu_t *rfu ); 
      void allocate( warp_inst_t *&pipeline_reg );

      void collect_operand( unsigned op )
      {
         m_not_ready.reset(op);
      }

      void dispatch();

   private:
      bool m_free;
      unsigned m_cuid; // collector unit hw id
      warp_inst_t **m_port; // pipeline register to issue to when ready
      unsigned m_warp_id;
      warp_inst_t  *m_warp;
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

   unsigned m_num_ports;
   std::vector<warp_inst_t**> m_input;
   std::vector<warp_inst_t**> m_output;
   std::vector<unsigned> m_num_collector_units;
   warp_inst_t **m_alu_port;

   typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
   port_to_du_t                     m_dispatch_units;
   std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
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

   bool has_mshr(unsigned num) const { return (num <= m_free_list.size()); }

   mshr_entry* return_head();

   //return queue pop; (includes texture pipeline return)
   void pop_return_head();

   mshr_entry* add_mshr(mem_access_t &access, warp_inst_t* inst);
   void mshr_return_from_mem(mshr_entry *mshr);
   unsigned get_max_mshr_used() const {return m_max_mshr_used;}  
   void print(FILE* fp);

private:
   mshr_entry *alloc_free_mshr(bool istexture);
   void free_mshr( mshr_entry *mshr );
   unsigned mshr_used() const;
   bool has_return() 
   { 
       return (not m_mshr_return_queue.empty()) or 
              ((not m_texture_mshr_pipeline.empty()) and m_texture_mshr_pipeline.front()->fetched());
   }
   std::list<mshr_entry*> &choose_return_queue();

   const struct shader_core_config *m_shader_config;

   typedef std::vector<mshr_entry> mshr_storage_type;
   mshr_storage_type m_mshrs; 
   std::list<mshr_entry*> m_free_list;
   std::list<mshr_entry*> m_mshr_return_queue;
   std::list<mshr_entry*> m_texture_mshr_pipeline;
   unsigned m_max_mshr_used;
   mshr_lookup m_mshr_lookup;
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

class simd_function_unit {
public:
    simd_function_unit( warp_inst_t **result_port, const shader_core_config *config ) 
    { 
        m_result_port = result_port;
        m_config=config;
        m_dispatch_reg = new warp_inst_t(config); 
    }
    ~simd_function_unit() { delete m_dispatch_reg; }

    // modifiers
    virtual void issue( warp_inst_t *&inst ) { move_warp(m_dispatch_reg,inst); }
    virtual void cycle() = 0;

    // accessors
    virtual bool can_issue( const warp_inst_t & ) const { return m_dispatch_reg->empty(); }
    virtual bool stallable() const = 0;
    virtual void print( FILE *fp ) const
    {
        fprintf(fp,"%s dispatch= ", m_name.c_str() );
        m_dispatch_reg->print(fp);
    }
protected:
    std::string m_name;
    const shader_core_config *m_config;
    warp_inst_t **m_result_port;
    warp_inst_t *m_dispatch_reg;
};

class alu : public simd_function_unit {
public:
    alu( warp_inst_t **result_port, const shader_core_config *config, unsigned max_latency ) 
        : simd_function_unit(result_port,config) 
    {
        m_result_port = result_port;
        m_pipeline_depth = max_latency;
        m_pipeline_reg = new warp_inst_t*[m_pipeline_depth];
        for( unsigned i=0; i < m_pipeline_depth; i++ ) 
            m_pipeline_reg[i] = new warp_inst_t( config );
    }

    //modifiers
    virtual void cycle() 
    {
        if( !m_pipeline_reg[0]->empty() )
            move_warp(*m_result_port,m_pipeline_reg[0]); // non-stallable pipeline
        for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);
        if( !m_dispatch_reg->empty() ) {
            if( !m_dispatch_reg->dispatch_delay() ) {
                int start_stage = m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
                move_warp(m_pipeline_reg[start_stage],m_dispatch_reg);
            }
        }
    }
    virtual void issue( warp_inst_t *&inst )
    {
        move_warp(m_dispatch_reg,inst);
    }

    // accessors
    virtual bool stallable() const { return false; }
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        return simd_function_unit::can_issue(inst);
    }
    virtual void print(FILE *fp) const
    {
        simd_function_unit::print(fp);
        for( int s=m_pipeline_depth-1; s>=0; s-- ) {
            if( !m_pipeline_reg[s]->empty() ) { 
                fprintf(fp,"      %s[%2d] ", m_name.c_str(), s );
                m_pipeline_reg[s]->print(fp);
            }
        }
    }
private:
    unsigned m_pipeline_depth;
    warp_inst_t **m_pipeline_reg;
};

class sfu : public alu
{
public:
    sfu( warp_inst_t **result_port, const shader_core_config *config ) 
        : alu(result_port,config,config->max_sfu_latency) { m_name = "SFU"; }
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: break;
        case ALU_SFU_OP: break;
        default: return false;
        }
        return alu::can_issue(inst);
    }
};

class sp_unit : public alu
{
public:
    sp_unit( warp_inst_t **result_port, const shader_core_config *config ) 
        : alu(result_port,config,config->max_sp_latency) { m_name = "SP "; }
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: return false; 
        case LOAD_OP: return false;
        case STORE_OP: return false;
        case MEMORY_BARRIER_OP: return false;
        default: break;
        }
        return alu::can_issue(inst);
    }
};

class ldst_unit : public simd_function_unit {
public:
    ldst_unit( gpgpu_sim *gpu, 
               shader_core_ctx *core, 
               warp_inst_t **result_port, 
               shader_core_config *config, 
               shader_core_stats *stats, 
               unsigned sid, unsigned tpc );

    // modifiers
    virtual void cycle(); 
    void fill( mem_fetch *mf );
    void flush();

    // accessors
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case LOAD_OP: break;
        case STORE_OP: break;
        case MEMORY_BARRIER_OP: break;
        default: return false;
        }
        return simd_function_unit::can_issue(inst);
    }
    virtual bool stallable() const { return true; }
    void print(FILE *fout) const;

private:
   void generate_mem_accesses(warp_inst_t &pipe_reg);
   void tex_cache_access(warp_inst_t &inst);
   void const_cache_access(warp_inst_t &inst);

   bool shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool memory_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);

   mem_stage_stall_type ccache_check(warp_inst_t &inst, mem_access_t& access){ return NO_RC_FAIL;}
   mem_stage_stall_type tcache_check(warp_inst_t &inst, mem_access_t& access){ return NO_RC_FAIL;}
   mem_stage_stall_type dcache_check(warp_inst_t &inst, mem_access_t& access);

   typedef mem_stage_stall_type (ldst_unit::*cache_check_t)(warp_inst_t &,mem_access_t&);

   mem_stage_stall_type process_memory_access_queue( ldst_unit::cache_check_t cache_check,
                                                     unsigned ports_per_bank, 
                                                     unsigned memory_send_max, 
                                                     warp_inst_t &inst );
   mem_stage_stall_type send_mem_request(warp_inst_t &inst, mem_access_t &access);

   gpgpu_sim *m_gpu;
   shader_core_ctx *m_core;
   unsigned m_sid;
   unsigned m_tpc;

   cache_t *m_L1D; // data cache (global/local memory accesses)
   cache_t *m_L1T; // texture cache
   cache_t *m_L1C; // constant cache
   mshr_shader_unit *m_mshr_unit;

   enum mem_stage_stall_type m_mem_rc;

   shader_core_stats *m_stats; 
};

enum pipeline_stage_name_t {
    ID_OC_SP=0,
    ID_OC_SFU,  
    ID_OC_MEM,  
    OC_EX_SP,
    OC_EX_SFU,
    OC_EX_MEM,
    EX_WB,
    N_PIPELINE_STAGES 
};

class shader_core_ctx : public core_t 
{
public:
   shader_core_ctx( class gpgpu_sim *gpu,
                    const char *name, 
                    unsigned shader_id,
                    unsigned tpc_id,
                    struct shader_core_config *config,
                    struct shader_core_stats *stats );

   void issue_block2core( class kernel_info_t &kernel );
   void get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc );
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

   void reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed );
   void init_warps(unsigned cta_id, unsigned start_thread, unsigned end_thread);

   unsigned max_cta( class function_info *kernel );
   void cache_flush();
   void display_pdom_state(FILE *fout, int mask );
   void display_pipeline( FILE *fout, int print_mem, int mask3bit );
   void register_cta_thread_exit(int cta_num );
   void fill_shd_L1_with_new_line( class mem_fetch * mf );
   void store_ack( class mem_fetch *mf );
   void dump_istream_state( FILE *fout );
   unsigned first_valid_thread( unsigned stage );
   class ptx_thread_info* get_functional_thread( unsigned tid ) { return m_thread[tid].m_functional_model_thread_state; }
   std::list<unsigned> get_regs_written( const inst_t &fvt ) const;
   const shader_core_config *get_config() const { return m_config; }
   unsigned get_num_sim_insn() const { return m_num_sim_insn; }
   int get_not_completed() const { return m_not_completed; }
   unsigned get_n_diverge() const { return m_n_diverge; }
   unsigned get_thread_n_insn( unsigned tid ) const { return m_thread[tid].n_insn; }
   unsigned get_thread_n_insn_ac( unsigned tid ) const { return m_thread[tid].n_insn_ac; }
   unsigned get_thread_n_l1_mis_ac( unsigned tid ) const { return m_thread[tid].n_l1_mis_ac; }
   unsigned get_thread_n_l1_mrghit_ac( unsigned tid ) const { return m_thread[tid].n_l1_mrghit_ac; }
   unsigned get_thread_n_l1_access_ac( unsigned tid ) const { return m_thread[tid].n_l1_access_ac; }
   unsigned get_n_active_cta() const { return m_n_active_cta; }
   void inc_store_req( unsigned warp_id) { m_warp[warp_id].inc_store_req(); }
  
private:

   void clear_stage_reg(int stage);

   address_type next_pc( int tid ) const;

   void fetch();

   void decode();
   void issue_warp( warp_inst_t *&warp, const warp_inst_t *pI, unsigned active_mask, unsigned warp_id );
   void func_exec_inst( warp_inst_t &inst );
   address_type translate_local_memaddr(address_type localaddr, unsigned tid, unsigned num_shader );

   void execute();

   void writeback();

   void call_thread_done(inst_t &done_inst );

   void print_stage(unsigned int stage, FILE *fout) const;

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

   // fetch
   cache_t *m_L1I; // instruction cache
   int  m_last_warp_fetched;

   // decode/dispatch
   int  m_last_warp_issued;
   std::vector<shd_warp_t>   m_warp;   // per warp information array
   barrier_set_t             m_barriers;
   ifetch_buffer_t           m_inst_fetch_buffer;
   pdom_warp_ctx_t         **m_pdom_warp; // pdom reconvergence context for each warp
   warp_inst_t** m_pipeline_reg;
   Scoreboard *m_scoreboard;
   opndcoll_rfu_t m_operand_collector;

   // execute
   unsigned m_num_function_units;
   enum pipeline_stage_name_t *m_dispatch_port;
   enum pipeline_stage_name_t *m_issue_port;
   simd_function_unit **m_fu; // stallable pipelines should be last in this array
   ldst_unit *m_ldst_unit;
   static const unsigned MAX_ALU_LATENCY = 64;
   std::bitset<MAX_ALU_LATENCY> m_result_bus;
};

#endif /* SHADER_H */
