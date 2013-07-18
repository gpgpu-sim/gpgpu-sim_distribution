// Copyright (c) 2009-2011, Tor M. Aamodt, Tim Rogers, Wilson W. L. Fung 
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

#pragma once 

#include "../trace.h"

#if TRACING_ON

#define MEMPART_PRINT_STR SIM_PRINT_STR " %d - "
#define MEMPART_DTRACE(x)  ( DTRACE(x) && (Trace::sampling_memory_partition == -1 || Trace::sampling_memory_partition == (int)get_mpid()) )

// Intended to be called from inside components of a memory partition
// Depends on a get_mpid() function
#define MEMPART_DPRINTF(...) do {\
    if (MEMPART_DTRACE(MEMORY_PARTITION_UNIT)) {\
        printf( MEMPART_PRINT_STR,\
                gpu_sim_cycle + gpu_tot_sim_cycle,\
                Trace::trace_streams_str[Trace::MEMORY_PARTITION_UNIT],\
                get_mpid() );\
        printf(__VA_ARGS__);\
    }\
} while (0)

#else

#define MEMPART_DTRACE(x)  (false)
#define MEMPART_DPRINTF(x, ...) do {} while (0)

#endif

