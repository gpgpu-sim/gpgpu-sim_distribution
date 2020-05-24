// Copyright (c) 2009-2013, Tor M. Aamodt, Timothy Rogers,
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

#include "trace.h"
#include "string.h"

namespace Trace {

#define TS_TUP_BEGIN(X) const char* trace_streams_str[] = {
#define TS_TUP(X) #X
#define TS_TUP_END(X) \
  }                   \
  ;
#include "trace_streams.tup"
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END

bool enabled = false;
int sampling_core = 0;
int sampling_memory_partition = -1;
bool trace_streams_enabled[NUM_TRACE_STREAMS] = {false};
const char* config_str;

void init() {
  for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) {
    if (strstr(config_str, trace_streams_str[i]) != NULL) {
      trace_streams_enabled[i] = true;
    }
  }
}
}  // namespace Trace
