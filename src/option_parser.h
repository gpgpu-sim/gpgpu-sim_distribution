// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#pragma once

#include <stdio.h>
#include <stdlib.h>

// pointer to C++ class
typedef class OptionParser *option_parser_t;

// data type of the option
enum option_dtype {
  OPT_INT32,
  OPT_UINT32,
  OPT_INT64,
  OPT_UINT64,
  OPT_BOOL,
  OPT_FLOAT,
  OPT_DOUBLE,
  OPT_CHAR,
  OPT_CSTR
};

// create and destroy option parser
option_parser_t option_parser_create();
void option_parser_destroy(option_parser_t opp);

// register new option
void option_parser_register(option_parser_t opp, const char *name,
                            enum option_dtype type, void *variable,
                            const char *desc, const char *defaultvalue);

// parse command line
void option_parser_cmdline(option_parser_t opp, int argc, const char *argv[]);

// parse config file
void option_parser_cfgfile(option_parser_t opp, const char *filename);

// parse a delimited string
void option_parser_delimited_string(option_parser_t opp,
                                    const char *inputstring,
                                    const char *delimiters);
// print options
void option_parser_print(option_parser_t opp, FILE *fout);
