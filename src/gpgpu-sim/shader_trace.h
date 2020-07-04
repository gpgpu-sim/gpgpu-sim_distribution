// Copyright (c) 2009-2011, Tor M. Aamodt, Tim Rogers
// George L. Yuan, Andrew Turner, Inderpreet Singh
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __SHADER_TRACE_H__
#define __SHADER_TRACE_H__

#include "../trace.h"

#if TRACING_ON

#define SHADER_PRINT_STR SIM_PRINT_STR "Core %d - "
#define SCHED_PRINT_STR SHADER_PRINT_STR "Scheduler %d - "
#define SHADER_DTRACE(x) \
  (DTRACE(x) &&          \
   (Trace::sampling_core == get_sid() || Trace::sampling_core == -1))

// Intended to be called from inside components of a shader core.
// Depends on a get_sid() function
#define SHADER_DPRINTF(x, ...)                                \
  do {                                                        \
    if (SHADER_DTRACE(x)) {                                   \
      printf(SHADER_PRINT_STR,                                \
             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x], get_sid());  \
      printf(__VA_ARGS__);                                    \
    }                                                         \
  } while (0)

// Intended to be called from inside a scheduler_unit.
// Depends on a m_id member
#define SCHED_DPRINTF(...)                                               \
  do {                                                                   \
    if (SHADER_DTRACE(WARP_SCHEDULER)) {                                 \
      printf(SCHED_PRINT_STR,                                            \
             m_shader->get_gpu()->gpu_sim_cycle +                        \
                 m_shader->get_gpu()->gpu_tot_sim_cycle,                 \
             Trace::trace_streams_str[Trace::WARP_SCHEDULER], get_sid(), \
             m_id);                                                      \
      printf(__VA_ARGS__);                                               \
    }                                                                    \
  } while (0)

#else

#define SHADER_DTRACE(x) (false)
#define SHADER_DPRINTF(x, ...) \
  do {                         \
  } while (0)
#define SCHED_DPRINTF(x, ...) \
  do {                        \
  } while (0)

#endif

#endif
