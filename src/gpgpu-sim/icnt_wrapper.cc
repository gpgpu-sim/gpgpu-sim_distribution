// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
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

#include "icnt_wrapper.h"
#include <assert.h>
#include "../intersim/interconnect_interface.h"

icnt_has_buffer_p icnt_has_buffer;
icnt_push_p       icnt_push;
icnt_pop_p        icnt_pop;
icnt_transfer_p   icnt_transfer;
icnt_busy_p       icnt_busy;

int   g_network_mode;
char* g_network_config_filename;

#include "../option_parser.h"

void icnt_reg_options( class OptionParser * opp )
{
   option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode, "Interconnection network mode", "1");
   option_parser_register(opp, "-inter_config_file", OPT_CSTR, &g_network_config_filename, "Interconnection network config file", "mesh");
}

void icnt_init( unsigned int n_shader, unsigned int n_mem )
{
   switch (g_network_mode) {
   case INTERSIM:
      init_interconnect(g_network_config_filename, n_shader, n_mem );
      icnt_has_buffer = interconnect_has_buffer;
      icnt_push       = interconnect_push;
      icnt_pop        = interconnect_pop;
      icnt_transfer   = advance_interconnect;
      icnt_busy       = interconnect_busy;
     break;

   default:
      assert(0);
      break;
   }
}
