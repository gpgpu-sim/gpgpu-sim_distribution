/* 
 * delayqueue.h
 *
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * Ivan Sham, Henry Tran and the University of British Columbia
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
#include <assert.h>
#include <stdlib.h>

#ifndef DELAYQUEUE_H
#define DELAYQUEUE_H

#include "../util.h"

typedef struct delay_data_t delay_data;
struct delay_data_t {
   void *data;
   unsigned int time_elapsed;
   delay_data *next;
   unsigned long long  push_time; //for stat collection
};

typedef struct {
   const char* name;
   int uid;

   unsigned int latency;
   unsigned int min_len;
   unsigned int max_len;
   unsigned int length;
   unsigned int n_element;

   delay_data *head;
   delay_data *tail;

   void* lat_stat; //a pointer to latency stats distribution structure
   //occupancy stat
   unsigned int max_size_stat;
   unsigned int n_stat_samples;
   float  avg_size_stat;
} delay_queue;

unsigned char dq_full(delay_queue* dq );
unsigned char dq_empty(delay_queue* dq );
unsigned int dq_n_element(delay_queue* dq );
unsigned char dq_push(delay_queue* dq, void* data);
void* dq_pop(delay_queue* dq);
void dq_set_min_length(delay_queue* dq, unsigned int new_min_len);
void removeEntry(void* data, delay_queue** dq, int size_dq);
delay_queue* dq_create( const char* name, 
		   unsigned int latency, 
		   unsigned int min_len, 
		   unsigned int max_len);
void dq_remove(void* data, delay_queue* dq);
void dq_print(delay_queue* dq);
void dq_free(delay_queue* dq);
void* dq_top(delay_queue* dq);//return the data in the head without poping the queue

void dq_update_stat(delay_queue* dq);
void dq_print_stat(delay_queue* dq);

#endif
