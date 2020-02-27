// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef STREAM_MANAGER_H_INCLUDED
#define STREAM_MANAGER_H_INCLUDED

#include "abstract_hardware_model.h"
#include <list>
#include <pthread.h>
#include <time.h>

//class stream_barrier {
//public:
//    stream_barrier() { m_pending_streams=0; }
//    void inc() { m_pending_streams++; }
//    void dec() { assert(m_pending_streams); m_pending_streams--; }
//    unsigned value() const { return m_pending_streams; }
//private:
//    unsigned m_pending_streams;
//};

struct CUevent_st {
public:
   CUevent_st( bool blocking )
   {
      m_uid = ++m_next_event_uid;
      m_blocking = blocking;
      m_updates = 0;
      m_wallclock = 0;
      m_gpu_tot_sim_cycle = 0;
      m_issued = 0;
      m_done = false;
   }
   void update( double cycle, time_t clk )
   {
      m_updates++;
      m_wallclock=clk;
      m_gpu_tot_sim_cycle=cycle;
      m_done = true;
   }
   //void set_done() { assert(!m_done); m_done=true; }
   int get_uid() const { return m_uid; }
   unsigned num_updates() const { return m_updates; }
   bool done() const { return m_updates==m_issued; }
   time_t clock() const { return m_wallclock; }
   void issue(){ m_issued++; }
   unsigned int num_issued() const{ return m_issued; }
private:
   int m_uid;
   bool m_blocking;
   bool m_done;
   int m_updates;
   unsigned int m_issued;
   time_t m_wallclock;
   double m_gpu_tot_sim_cycle;

   static int m_next_event_uid;
};


enum stream_operation_type {
    stream_no_op,
    stream_memcpy_host_to_device,
    stream_memcpy_device_to_host,
    stream_memcpy_device_to_device,
    stream_memcpy_to_symbol,
    stream_memcpy_from_symbol,
    stream_kernel_launch,
    stream_event,
    stream_wait_event
};

class stream_operation {
public:
    stream_operation()
    {
        m_kernel=NULL;
        m_type = stream_no_op;
        m_stream = NULL;
        m_done=true;
    }
    stream_operation( const void *src, const char *symbol, size_t count, size_t offset, struct CUstream_st *stream )
    {
        m_kernel=NULL;
        m_stream = stream;
        m_type=stream_memcpy_to_symbol;
        m_host_address_src=src;
        m_symbol=symbol;
        m_cnt=count;
        m_offset=offset;
        m_done=false;
    }
    stream_operation( const char *symbol, void *dst, size_t count, size_t offset, struct CUstream_st *stream )
    {
        m_kernel=NULL;
        m_stream = stream;
        m_type=stream_memcpy_from_symbol;
        m_host_address_dst=dst;
        m_symbol=symbol;
        m_cnt=count;
        m_offset=offset;
        m_done=false;
    }
    stream_operation( kernel_info_t *kernel, bool sim_mode, struct CUstream_st *stream )
    {
        m_type=stream_kernel_launch;
        m_kernel=kernel;
        m_sim_mode=sim_mode;
        m_stream=stream;
        m_done=false;
    }
    stream_operation( struct CUevent_st *e, struct CUstream_st *stream )
    {
        m_kernel=NULL;
        m_type=stream_event;
        m_event=e;
        m_stream=stream;
        m_done=false;
    }
    stream_operation( struct CUstream_st *stream, class CUevent_st *e, unsigned int flags )
    {
        m_kernel=NULL;
        m_type=stream_wait_event;
        m_event=e;
        m_cnt = m_event->num_issued();
        m_stream=stream;
        m_done=false;
    }
    stream_operation( const void *host_address_src, size_t device_address_dst, size_t cnt, struct CUstream_st *stream )
    {
        m_kernel=NULL;
        m_type=stream_memcpy_host_to_device;
        m_host_address_src  =host_address_src;
        m_device_address_dst=device_address_dst;
        m_host_address_dst=NULL;
        m_device_address_src=0;
        m_cnt=cnt;
        m_stream=stream;
        m_sim_mode=false;
        m_done=false;
    }
    stream_operation( size_t device_address_src, void *host_address_dst, size_t cnt, struct CUstream_st *stream  )
    {
        m_kernel=NULL;
        m_type=stream_memcpy_device_to_host;
        m_device_address_src=device_address_src;
        m_host_address_dst=host_address_dst;
        m_device_address_dst=0;
        m_host_address_src=NULL;
        m_cnt=cnt;
        m_stream=stream;
        m_sim_mode=false;
        m_done=false;
    }
    stream_operation( size_t device_address_src, size_t device_address_dst, size_t cnt, struct CUstream_st *stream  )
    {
        m_kernel=NULL;
        m_type=stream_memcpy_device_to_device;
        m_device_address_src=device_address_src;
        m_device_address_dst=device_address_dst;
        m_host_address_src=NULL;
        m_host_address_dst=NULL;
        m_cnt=cnt;
        m_stream=stream;
        m_sim_mode=false;
        m_done=false;
    }

    bool is_kernel() const { return m_type == stream_kernel_launch; }
    bool is_mem() const {
        return m_type == stream_memcpy_host_to_device ||
               m_type == stream_memcpy_device_to_host ||
               m_type == stream_memcpy_host_to_device;
    }
    bool is_noop() const { return m_type == stream_no_op; }
    bool is_done() const { return m_done; }
    kernel_info_t *get_kernel() { return m_kernel; }
    bool do_operation( gpgpu_sim *gpu );
    void print( FILE *fp ) const;
    struct CUstream_st *get_stream() { return m_stream; }
    void set_stream( CUstream_st *stream ) { m_stream = stream; }
private:
    struct CUstream_st *m_stream;

    bool m_done;

    stream_operation_type m_type;
    size_t      m_device_address_dst;
    size_t      m_device_address_src;
    void       *m_host_address_dst;
    const void *m_host_address_src;
    size_t      m_cnt;

    const char *m_symbol;
    size_t m_offset;

    bool m_sim_mode;
    kernel_info_t *m_kernel;
    struct CUevent_st *m_event;
};
struct CUstream_st {
public:
    CUstream_st(); 
    bool empty();
    bool busy();
    void synchronize();
    void push( const stream_operation &op );
    void record_next_done();
    stream_operation next();
    void cancel_front(); //front operation fails, cancle the pending status
    stream_operation &front() { return m_operations.front(); }
    void print( FILE *fp );
    unsigned get_uid() const { return m_uid; }

private:
    unsigned m_uid;
    static unsigned sm_next_stream_uid;

    std::list<stream_operation> m_operations;
    bool m_pending; // front operation has started but not yet completed

    pthread_mutex_t m_lock; // ensure only one host or gpu manipulates stream operation at one time
};

class stream_manager {
public:
    stream_manager( gpgpu_sim *gpu, bool cuda_launch_blocking );
    bool register_finished_kernel(unsigned grid_uid  );
    bool check_finished_kernel(  );
    stream_operation front();
    void add_stream( CUstream_st *stream );
    void destroy_stream( CUstream_st *stream );
    bool concurrent_streams_empty();
    bool empty_protected();
    bool empty();
    void print( FILE *fp);
    void push( stream_operation op );
    void pushCudaStreamWaitEventToAllStreams( CUevent_st *e, unsigned int flags );
    bool operation(bool * sim);
    void stop_all_running_kernels();
private:
    void print_impl( FILE *fp);

    bool m_cuda_launch_blocking;
    gpgpu_sim *m_gpu;
    std::list<CUstream_st *> m_streams;
    std::map<unsigned,CUstream_st *> m_grid_id_to_stream;
    CUstream_st m_stream_zero;
    bool m_service_stream_zero;
    pthread_mutex_t m_lock;
    std::list<struct CUstream_st*>::iterator m_last_stream;
};

#endif
