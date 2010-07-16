/* 
 * dram.c
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, George L. Yuan,
 * Ivan Sham, Justin Kwong, Dan O'Connor and the 
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
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files 
 * src/gpgpusim_entrypoint.c and src/simplesim-3.0/ are derived from the 
 * SimpleScalar Toolset available from http://www.simplescalar.com/ 
 * (property of SimpleScalar LLC) and the files src/intersim/ are derived 
 * from Booksim (Simulator provided with the textbook "Principles and 
 * Practices of Interconnection Networks" available from 
 * http://cva.stanford.edu/books/ppin/).  As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.
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

#include "delayqueue.h"
#include "../cuda-sim/dram_callback.h"

#ifndef DRAM_H
#define DRAM_H

#define FIFO_AGE_LIMIT 50     //used for both BANK_CONF and REALISTIC schedulers
#define FIFO_NUM_WRITE_LIMIT 3   //used for both BANK_CONF and REALISTIC schedulers
#define LOOKAHEAD_VALUE 10    //used for REALISTIC scheduler ONLY

enum {
   DRAM_FIFO,
   DRAM_IDEAL_FAST,
   DRAM_NUM_HANDLES
};


#define READ 'R'  //define read and write states
#define WRITE 'W'
typedef struct {
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
   void* data;

   int cache_hits_waiting; 
} dram_req_t;

#define BANK_IDLE 'I'
#define BANK_ACTIVE 'A'

typedef struct {
   unsigned int RCDc;
   unsigned int RCDWRc;
   unsigned int RASc;
   unsigned int RPc;
   unsigned int RCc;

   unsigned char rw;    //is the bank reading or writing?
   unsigned char state; //is the bank active or idle?
   unsigned int curr_row;

   dram_req_t *mrq;

   unsigned int n_access;
   unsigned int n_writes;
   unsigned int n_idle;
} bank_t;

typedef struct {
   unsigned int id;

   unsigned int tCCD;   //column to column delay
   unsigned int tRRD;   //minimal time required between activation of rows in different banks
   unsigned int tRCD;   //row to column delay - time required to activate a row before a read
   unsigned int tRCDWR;//row to column delay for a write command
   unsigned int tRAS;   //time needed to activate row
   unsigned int tRP; //row precharge ie. deactivate row
   unsigned int tRC; //row cycle time ie. precharge current, then activate different row

   unsigned int CL;  //CAS latency
   unsigned int WL;  //WRITE latency
   unsigned int BL;  //Burst Length in bytes (we're using 4? could be 8)
   unsigned int tRTW;   //time to switch from read to write
   unsigned int tWTR;   //time to switch from write to read 5? look in datasheet
   unsigned int busW;

   unsigned int nbk;
   bank_t **bk;
   unsigned int prio;

   unsigned int RRDc;
   unsigned int CCDc;
   unsigned int RTWc;   //read to write penalty applies across banks
   unsigned int WTRc;   //write to read penalty applies across banks

   unsigned char rw; //was last request a read or write? (important for RTW, WTR)

   unsigned int pending_writes;
   unsigned char realistic_scheduler_mode;

   delay_queue *rwq;
   delay_queue *mrqq;
   //buffer to hold packets when DRAM processing is over
   //should be filled with dram clock and popped with l2or icnt clock 
   delay_queue *returnq;      


   unsigned int dram_util_bins[10];
   unsigned int dram_eff_bins[10];
   unsigned int last_n_cmd, last_n_activity, last_bwutil;

   unsigned int queue_limit;

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
   unsigned char scheduler_type;

   void *m_fast_ideal_scheduler;

   void *m_L2cache;

   unsigned int n_cmd_partial;
   unsigned int n_activity_partial;
   unsigned int n_nop_partial; 
   unsigned int n_act_partial; 
   unsigned int n_pre_partial; 
   unsigned int n_req_partial;
   unsigned int ave_mrqs_partial;
   unsigned int bwutil_partial;

   void * req_hist;
} dram_t;


dram_t* dram_create( unsigned int id, unsigned int nbk, 
		unsigned int tCCD, unsigned int tRRD,
		unsigned int tRCD, unsigned int tRAS,
		unsigned int tRP, unsigned int tRC,
		unsigned int CL, unsigned int WL, 
		unsigned int BL, unsigned int tWTR,
		unsigned int busW, unsigned int queue_limit,
		unsigned char scheduler_type );
void dram_free( dram_t *dm );
int dram_full( dram_t *dm );
void dram_push( dram_t *dm, unsigned int bank,
	   unsigned int row, unsigned int col,
	   unsigned int nbytes, unsigned int write,
	   unsigned int wid, unsigned int sid, int cache_hits_waiting, unsigned long long addr,
	   void *data );
void scheduler_fifo(dram_t* dm);
void dram_issueCMD (dram_t* dm);
void* dram_pop( dram_t *dm );
void* dram_top( dram_t *dm );
unsigned dram_busy( dram_t *dm);
void dram_print( dram_t* dm, FILE* simFile );
void dram_visualize( dram_t* dm );
void dram_print_stat( dram_t* dm, FILE* simFile );
void fast_scheduler_ideal(dram_t* dm);
void* alloc_fast_ideal_scheduler(dram_t *dm);
void dump_fast_ideal_scheduler(dram_t *dm);
unsigned fast_scheduler_queue_length(dram_t *dm);

//supposed to return the current queue length for all memory scheduler types.
unsigned int dram_que_length( dram_t *dm ); 

#endif /*DRAM_H*/
