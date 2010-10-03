/* 
 * mem_fetch.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the 
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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include "addrdec.h"
#include "../abstract_hardware_model.h"
#include <bitset>

enum mf_type {
   RD_REQ = 0,
   WT_REQ,
   REPLY_DATA, // send to shader
   L2_WTBK_DATA,
   N_MF_TYPE
};

enum mem_access_type { 
   GLOBAL_ACC_R = 0, 
   LOCAL_ACC_R = 1, 
   CONST_ACC_R = 2, 
   TEXTURE_ACC_R = 3, 
   GLOBAL_ACC_W = 4, 
   LOCAL_ACC_W = 5,
   L2_WRBK_ACC = 6, 
   INST_ACC_R = 7, 
   NUM_MEM_ACCESS_TYPE = 8
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

const unsigned partial_write_mask_bits = 128; //must be at least size of largest memory access.
typedef std::bitset<partial_write_mask_bits> partial_write_mask_t;

class mem_fetch {
public:
   mem_fetch( new_addr_type addr,
              unsigned data_size,
              unsigned ctrl_size,
              unsigned sid,
              unsigned tpc,
              unsigned wid,
              class mshr_entry   * mshr,
              bool                  write,
              partial_write_mask_t partial_write_mask,
              enum mem_access_type mem_acc,
              enum mf_type type,
              address_type pc );

   void set_status( enum mshr_status status, enum mem_req_stat stat, unsigned long long cycle );
   void set_type( enum mf_type t ) { type=t; }
   void do_atomic();

   void print( FILE *fp ) const;

   const addrdec_t &get_tlx_addr() const { return tlx; }
   unsigned get_data_size() const { return nbytes_L1; }
   unsigned get_ctrl_size() const { return ctrl_size; }
   unsigned size() const { return nbytes_L1+ctrl_size; }
   new_addr_type get_addr() const { return addr; }
   class mshr_entry *get_mshr() { return mshr; }
   bool get_is_write() const { return m_write; }
   unsigned get_request_uid() const { return request_uid; }
   unsigned get_sid() const { return sid; }
   unsigned get_tpc() const { return tpc; }
   unsigned get_wid() const { return wid; }
   bool isinst() const;
   bool istexture() const;
   bool isconst() const;
   enum mf_type get_type() const { return type; }
   bool isatomic() const;
   void set_return_timestamp( unsigned t ) { timestamp2=t; }
   void set_icnt_receive_time( unsigned t ) { icnt_receive_time=t; }
   unsigned get_timestamp() const { return timestamp; }
   unsigned get_return_timestamp() const { return timestamp2; }
   unsigned get_icnt_receive_time() const { return icnt_receive_time; }
   enum mem_access_type get_mem_acc() const { return mem_acc; }
   address_type get_pc() const { return pc; }

private:
   // request origination
   unsigned request_uid;
   address_type pc;
   unsigned sid;
   unsigned tpc;
   unsigned wid;
   class mshr_entry* mshr;

   // request type, address, size, mask
   bool m_write;
   enum mem_access_type mem_acc;
   enum mf_type type;
   new_addr_type addr;
   addrdec_t tlx;
   partial_write_mask_t write_mask;
   unsigned nbytes_L1;
   unsigned ctrl_size;

   // statistics
   unsigned timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
   unsigned timestamp2; // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed onto icnt to shader; only used for reads
   unsigned icnt_receive_time; // set to gpu_sim_cycle + interconnect_latency when fixed icnt latency mode is enabled

   static unsigned sm_next_mf_request_uid;
};

#endif
