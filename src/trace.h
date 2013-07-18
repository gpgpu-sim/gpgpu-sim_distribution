// Copyright (c) 2009-2013, Tor M. Aamodt, Timothy Rogers,
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

// This file is inspired by the trace system in gem5.
// This is a highly simplified version adpated for gpgpusim

#ifndef __TRACE_H__
#define __TRACE_H__

extern unsigned long long  gpu_sim_cycle;
extern unsigned long long  gpu_tot_sim_cycle;

namespace Trace {

#define TS_TUP_BEGIN(X) enum X {
#define TS_TUP(X) X
#define TS_TUP_END(X) };
#include "trace_streams.tup"
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END

    extern bool enabled;
    extern int sampling_core;
    extern int sampling_memory_partition;
    extern const char* trace_streams_str[];
    extern bool trace_streams_enabled[NUM_TRACE_STREAMS];
    extern const char* config_str;

    void init();

} // namespace Trace


#if TRACING_ON

#define SIM_PRINT_STR "GPGPU-Sim Cycle %llu: %s - "
#define DTRACE(x) ((Trace::trace_streams_enabled[Trace::x]) && Trace::enabled)
#define DPRINTF(x, ...) do {\
    if (DTRACE(x)) {\
        printf( SIM_PRINT_STR,\
                gpu_sim_cycle + gpu_tot_sim_cycle,\
                Trace::trace_streams_str[Trace::x] );\
        printf(__VA_ARGS__);\
    }\
} while (0)


#else 

#define DTRACE(x) (false)
#define DPRINTF(x, ...) do {} while (0)

#endif  

#endif 
