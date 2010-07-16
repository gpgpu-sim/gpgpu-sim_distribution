/* 
 * delayqueue.c
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

#include "delayqueue.h"
#include "gpu-misc.h"
#include "../intersim/statwraper.h"

extern unsigned long long  gpu_sim_cycle; //for stat collection

unsigned char dq_full( delay_queue* dq ) 
{
   if (dq->max_len && dq->length >= dq->max_len)
      return 1;
   return 0;
}

unsigned char dq_empty(delay_queue* dq )
{
   return(dq->head == NULL)?1:0;
}

unsigned int dq_n_element(delay_queue* dq )
{
   return(dq->n_element);
}

unsigned char dq_push(delay_queue* dq, void* data) {
   if (dq->max_len) assert(dq->length < dq->max_len);
   if (dq->head) {
      if (dq->tail->data || dq->length < dq->min_len) {
         dq->tail->next = (delay_data*) malloc(sizeof(delay_data)); 
         dq->tail = dq->tail->next;
         dq->length++;
         dq->n_element++;
      }
   } else {
      dq->head = dq->tail = (delay_data*) malloc(sizeof(delay_data)); 
      dq->length++;
      dq->n_element++;
   }
   dq->tail->next = NULL;
   dq->tail->time_elapsed = dq->latency;
   dq->tail->data = (void*)data;
   dq->tail->push_time = gpu_sim_cycle;
   return 1;
}

void* dq_top(delay_queue* dq) {
   if (dq->head) {
      return dq->head->data;
   } else {
      return NULL;
   }
}

void* dq_pop(delay_queue* dq) {
   delay_data* next;
   void* data;
   if (dq->head) {
      if (dq->head->time_elapsed) {
         dq->head->time_elapsed--;
         data = NULL;
      } else {
         next = dq->head->next;
         data = dq->head->data;
         StatAddSample(dq->lat_stat, LOGB2 (gpu_sim_cycle - dq->head->push_time));
         if ( dq->head == dq->tail ) {
            assert( next == NULL );
            dq->tail = NULL;     
         }
         free(dq->head);
         dq->head = next;
         dq->length--;
         if (dq->length == 0) {
            assert( dq->head == NULL );
            dq->tail = dq->head;
         }
         dq->n_element--; 
      }
      if (dq->min_len && dq->length < dq->min_len) {
         dq_push(dq,NULL);
         dq->n_element--; // uncount NULL elements inserted to create delays
      }
   } else {
      data = NULL;
   }
   return data;
}

void dq_set_min_length(delay_queue* dq, unsigned int new_min_len) {
   if (new_min_len == dq->min_len) return;

   if (new_min_len > dq->min_len) {
      dq->min_len = new_min_len;
      while (dq->length < dq->min_len) {
         dq_push(dq,NULL);
         dq->n_element--; // uncount NULL elements inserted to create delays
      }
   } else {
      // in this branch imply that the original min_len is larger then 0
      // ie. dq->head != 0
      assert(dq->head);
      dq->min_len = new_min_len;
      while ((dq->length > dq->min_len) && (dq->tail->data == 0)) {
         delay_data *iter;
         iter = dq->head;
         while (iter && (iter->next != dq->tail))
            iter = iter->next;
         if (!iter) {
            // there is only one node, and that node is empty
            assert(dq->head->data == 0);
            dq_pop(dq);
         } else {
            // there are more than one node, and tail node is empty
            assert(iter->next == dq->tail);
            free(dq->tail);
            dq->tail = iter;
            dq->tail->next = 0;
            dq->length--;
         }
      }
   }
}

void dq_remove(void* data, delay_queue* dq)
{
   // removes an item from the queue without deallocating the memory
   delay_data* ptr = NULL;
   delay_data* temp = NULL;

   assert(dq);
   assert(data);

   ptr = dq->head;
   if (ptr) {
      if (ptr->data == data) {
         StatAddSample(dq->lat_stat, LOGB2 (gpu_sim_cycle - ptr->push_time));
         dq->head = ptr->next;
         if ( dq->head == NULL )
            dq->tail = NULL;
         dq->length--;
         return;      
      }
      while (ptr->next) {
         if (ptr->next->data == data) {
            temp = ptr->next;
            StatAddSample(dq->lat_stat, LOGB2 (gpu_sim_cycle - temp->push_time));
            if ( ptr->next == dq->tail ) {
               dq->tail = ptr;
            }
            ptr->next = ptr->next->next;
            dq->length--;
            return;
         }
         ptr = ptr->next;
      }
   }
}

void removeEntry(void* data, delay_queue** dqq, int size_dq)
{
   int i;
   delay_data* ptr = NULL;
   delay_queue* dq = NULL;
   delay_data* temp = NULL;

   assert(dqq);
   assert(data);


   for (i = 0; i<size_dq; i++) {
      dq = dqq[i];
      ptr = dq->head;
      if (ptr) {
         if (ptr->data == data) {
            dq->head = ptr->next;
            if ( dq->head == NULL )
               dq->tail = NULL;
            StatAddSample(dq->lat_stat, LOGB2 (gpu_sim_cycle - ptr->push_time));
            free(ptr);  
            dq->length--;
            return;     
         }
         while (ptr->next) {
            if (ptr->next->data == data) {
               temp = ptr->next;
               if ( ptr->next == dq->tail ) {
                  dq->tail = ptr;
               }
               ptr->next = ptr->next->next;
               StatAddSample(dq->lat_stat, LOGB2 (gpu_sim_cycle - temp->push_time));
               free(temp);
               dq->length--;
               return;
            }
            ptr = ptr->next;
         }
      }

   }
}

static int dq_uid_counter = 0;

delay_queue* dq_create(const char* name, unsigned int latency, unsigned int min_len, unsigned int max_len) {
   unsigned i;
   delay_queue* dq;
   dq = (delay_queue*) malloc(sizeof(delay_queue));
   dq->name = name;
   dq->latency = latency;
   dq->min_len = min_len;
   dq->max_len = max_len;
   dq->length = 0;
   dq->n_element = 0;
   dq->head = NULL;
   dq->tail = NULL;
   for (i=0;i<min_len;i++) dq_push(dq,NULL);
   dq->uid = dq_uid_counter;
   dq_uid_counter++;
   if (1) {
      dq->lat_stat = StatCreate(dq->name,1,32);  
   }
   dq->max_size_stat = 0;
   dq->avg_size_stat =0.0 ;
   return dq;
}

void dq_print(delay_queue* dq) {
   delay_data* ddp = dq->head;
   printf("%s(%d): ", dq->name, dq->length);
   while (ddp) {
      printf("%p ", ddp->data);
      ddp = ddp->next;
   }
   printf("\n");
}

void dq_free(delay_queue* dq) {
   while (dq->head) {
      dq->tail = dq->head;
      dq->head = dq->head->next;
      free(dq->tail);
   }
   free(dq);
   dq = NULL;
}

void dq_update_stat(delay_queue* dq) {
   if (dq->n_element > dq->max_size_stat) {
      dq->max_size_stat = dq->n_element;
   }
   dq->avg_size_stat = (dq->avg_size_stat*dq->n_stat_samples + dq->n_element)/(++dq->n_stat_samples); 
}
void dq_print_stat(delay_queue* dq) {
   printf("Max Length: %d, Average Length: %f\n",dq->max_size_stat,dq->avg_size_stat );
}


#ifdef TEST_DQ

void regresstion_test01() {
   delay_queue *dqa, *dqb;
   int i;
   int a[7];
   for (i=0;i<7;i++) a[i]=i;

   dqa = dq_create("dqa", 0, 7, 0);
   for (i=0;i<3;i++) dq_push(dqa, &a[i]);

   for (i=0;i<6;i++) {
      dq_print(dqa);
      assert(dq_pop(dqa) == 0);
   }
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[0]);

   // shortening queue 
   dq_print(dqa);
   dq_set_min_length(dqa, 4);
   // see if data in the queue still persist
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[1]);
   // see if the queue behave with min length = 4
   dq_push(dqa, &a[3]);
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[2]);
   for (i=0;i<2;i++) {
      dq_print(dqa);
      assert(dq_pop(dqa) == 0);
   }
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[3]);

   // lengthening queue 
   dq_set_min_length(dqa, 6);
   dq_push(dqa, &a[4]);
   dq_push(dqa, &a[5]);
   for (i=0;i<5;i++) {
      dq_print(dqa);
      assert(dq_pop(dqa) == 0);
   }
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[4]);

   // queue with no min length
   dq_set_min_length(dqa, 0);
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[5]);
   dq_print(dqa);
   dq_push(dqa, &a[6]);
   dq_print(dqa);
   assert(dq_pop(dqa) == &a[6]);

   // lengthening the queue, then shorten it again, 
   // but with some data exceeding the new min length
   // the data should retain.
   dq_print(dqa);
   dq_set_min_length(dqa, 7);
   dq_print(dqa);
   dq_push(dqa, &a[0]);
   assert(dq_pop(dqa) == 0);
   dq_print(dqa);
   dq_set_min_length(dqa, 4);
   dq_print(dqa);
   assert(dq_pop(dqa) == 0);
   assert(dq_pop(dqa) == 0);
   assert(dq_pop(dqa) == 0);
   assert(dq_pop(dqa) == 0);
   assert(dq_pop(dqa) == 0);
   // This is the 7th pop: min-length is obeyed
   assert(dq_pop(dqa) == &a[0]); 
   dq_print(dqa);

   // Shortening a queue with null entry only
   dq_set_min_length(dqa, 0);
   assert(dqa->length == 0);
   dq_print(dqa);

   // Lengthening
   dq_set_min_length(dqa, 6);
   assert(dqa->length == 6);
   dq_print(dqa);

   // Shortening a queue with null entry only
   dq_set_min_length(dqa, 3);
   assert(dqa->length == 3);
   dq_print(dqa);

   dq_free(dqa);
   printf("regression test 01 passed!\n");
}

int regresstion_test00() {
   delay_queue *dqa, *dqb, *dqc, *dqd;
   int i;
   int a[4];
   int *b;
   for (i=0;i<4;i++) a[i]=i;
   dqa = dq_create("dqa", 0, 4, 0);
   dqb = dq_create("dqb", 0, 10, 0);
   dq_print(dqa);
   dq_print(dqb);
   dq_push(dqa,a);
   dq_print(dqa);
   dq_pop(dqa);
   dq_print(dqa);
   dq_push(dqa,a);
   dq_print(dqa);
   dq_pop(dqa);
   dq_print(dqa);
   dq_pop(dqa);
   dq_print(dqa);
   b = dq_pop(dqa);
   dq_print(dqa);
   for (i=0;i<4;i++) printf("%d\n",b[i]);
   dqc = dq_create("dqc", 0, 0, 3);
   for (i=0;i<4;i++) {
      if (!dq_push(dqc,&a[i])) printf("cannot push.\n");
      dq_print(dqc);
   }
   dqd = dq_create("dqd", 0, 2, 3);
   if (!dq_push(dqd,&a[0])) printf("cannot push.\n");
   dq_print(dqd);
   if (!dq_push(dqd,&a[1])) printf("cannot push.\n");
   dq_print(dqd);
   if (!dq_push(dqd,&a[2])) printf("cannot push.\n");
   dq_print(dqd);
   dq_pop(dqd);
   if (!dq_push(dqd,&a[3])) printf("cannot push.\n");
   dq_print(dqd);

   dq_free(dqa);
   dq_free(dqb);
   dq_free(dqc);
   dq_free(dqd);

   return 0;
}

int main() {
   regresstion_test01();
   return 0;
}


#endif
