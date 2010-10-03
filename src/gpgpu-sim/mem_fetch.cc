/* 
 * mem_fetch.cc
 *
 * Copyright (c) 2009 by Tor M. Aamodt and 
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

#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include "gpu-sim.h"

unsigned mem_fetch::sm_next_mf_request_uid=1;

mem_fetch::mem_fetch( new_addr_type addr,
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
                      address_type pc )
{
   class mem_fetch *mf = this;
   mf->request_uid = sm_next_mf_request_uid++;

   mf->addr = addr;
   mf->nbytes_L1 = data_size;
   mf->ctrl_size = ctrl_size;
   mf->sid = sid;
   mf->wid = wid;
   mf->tpc = tpc;
   mf->mshr = mshr;
   mf->m_write = write;
   addrdec_tlx(addr,&mf->tlx);
   mf->mem_acc = mem_acc;
   mf->type = type;
   mf->pc = pc;
   mf->timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
   mf->timestamp2 = 0;
}

void mem_fetch::print( FILE *fp ) const
{
   fprintf(fp,"  mf: uid=%6u, addr=0x%08llx, sid=%u, wid=%u, pc=0x%04x, %s, bank=%u, ", 
           request_uid, addr, sid, wid, pc, (m_write?"write":"read "), tlx.bk);
   if( mshr ) mshr->print(fp,0x100);
   else fprintf(fp,"\n");
}

void mem_fetch::set_status( enum mshr_status status, enum mem_req_stat stat, unsigned long long cycle ) 
{
   if ( mshr ) {
      mshr->set_status(status);
      if( mshr->has_inst() ) 
         time_vector_update(mshr->get_insts_uid(),stat,cycle,type);
      else 
         time_vector_update(request_uid,stat,cycle,type);
   }
}

bool mem_fetch::isatomic() const
{
   if( !mshr ) return false;
   return mshr->isatomic();
}

void mem_fetch::do_atomic()
{
   dram_callback_t &cb = mshr->get_atomic_callback();
   cb.function(cb.instruction, cb.thread);
}

bool mem_fetch::isinst() const 
{ 
   return (mshr==NULL)?false:mshr->isinst(); 
}

bool mem_fetch::istexture() const
{ 
   return (mshr==NULL)?false:mshr->istexture(); 
}

bool mem_fetch::isconst() const
{ 
   return (mshr==NULL)?false:mshr->isconst(); 
}
