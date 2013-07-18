// Copyright (c) 2009-2011, Tor M. Aamodt, Ivan Sham, Ali Bakhoda, 
// George L. Yuan, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef DRAM_H
#define DRAM_H

#include "delayqueue.h"
#include <set>
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>

#define READ 'R'  //define read and write states
#define WRITE 'W'
#define BANK_IDLE 'I'
#define BANK_ACTIVE 'A'

class dram_req_t {
public:
   dram_req_t( class mem_fetch *data );

   unsigned int row;
   unsigned int col;
   unsigned int bk;
   unsigned int nbytes;
   unsigned int txbytes;
   unsigned int dqbytes;
   unsigned int age;
   unsigned int timestamp;
   unsigned char rw;    //is the request a read or a write?
   unsigned long long int addr;
   unsigned int insertion_time;
   class mem_fetch * data;
};

struct bankgrp_t
{
	unsigned int CCDLc;
	unsigned int RTPLc;
};

struct bank_t
{
   unsigned int RCDc;
   unsigned int RCDWRc;
   unsigned int RASc;
   unsigned int RPc;
   unsigned int RCc;
   unsigned int WTPc; // write to precharge
   unsigned int RTPc; // read to precharge

   unsigned char rw;    //is the bank reading or writing?
   unsigned char state; //is the bank active or idle?
   unsigned int curr_row;

   dram_req_t *mrq;

   unsigned int n_access;
   unsigned int n_writes;
   unsigned int n_idle;

   unsigned int bkgrpindex;
};

struct mem_fetch;

class dram_t 
{
public:
   dram_t( unsigned int parition_id, const struct memory_config *config, class memory_stats_t *stats, 
           class memory_partition_unit *mp );

   bool full() const;
   void print( FILE* simFile ) const;
   void visualize() const;
   void print_stat( FILE* simFile );
   unsigned que_length() const; 
   bool returnq_full() const;
   unsigned int queue_limit() const;
   void visualizer_print( gzFile visualizer_file );

   class mem_fetch* return_queue_pop();
   class mem_fetch* return_queue_top();
   void push( class mem_fetch *data );
   void cycle();
   void dram_log (int task);

   class memory_partition_unit *m_memory_partition_unit;
   unsigned int id;

   // Power Model
   void set_dram_power_stats(unsigned &cmd,
								unsigned &activity,
								unsigned &nop,
								unsigned &act,
								unsigned &pre,
								unsigned &rd,
								unsigned &wr,
								unsigned &req) const;

private:
   void scheduler_fifo();
   void scheduler_frfcfs();

   const struct memory_config *m_config;

   bankgrp_t **bkgrp;

   bank_t **bk;
   unsigned int prio;

   unsigned int RRDc;
   unsigned int CCDc;
   unsigned int RTWc;   //read to write penalty applies across banks
   unsigned int WTRc;   //write to read penalty applies across banks

   unsigned char rw; //was last request a read or write? (important for RTW, WTR)

   unsigned int pending_writes;

   fifo_pipeline<dram_req_t> *rwq;
   fifo_pipeline<dram_req_t> *mrqq;
   //buffer to hold packets when DRAM processing is over
   //should be filled with dram clock and popped with l2or icnt clock 
   fifo_pipeline<mem_fetch> *returnq;

   unsigned int dram_util_bins[10];
   unsigned int dram_eff_bins[10];
   unsigned int last_n_cmd, last_n_activity, last_bwutil;

   unsigned int n_cmd;
   unsigned int n_activity;
   unsigned int n_nop;
   unsigned int n_act;
   unsigned int n_pre;
   unsigned int n_rd;
   unsigned int n_wr;
   unsigned int n_req;
   unsigned int max_mrqs_temp;

   unsigned int bwutil;
   unsigned int max_mrqs;
   unsigned int ave_mrqs;

   class frfcfs_scheduler* m_frfcfs_scheduler;

   unsigned int n_cmd_partial;
   unsigned int n_activity_partial;
   unsigned int n_nop_partial; 
   unsigned int n_act_partial; 
   unsigned int n_pre_partial; 
   unsigned int n_req_partial;
   unsigned int ave_mrqs_partial;
   unsigned int bwutil_partial;

   struct memory_stats_t *m_stats;
   class Stats* mrqq_Dist; //memory request queue inside DRAM  

   friend class frfcfs_scheduler;
};

#endif /*DRAM_H*/
