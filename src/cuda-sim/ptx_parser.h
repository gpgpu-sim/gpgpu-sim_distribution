// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef ptx_parser_INCLUDED
#define ptx_parser_INCLUDED

#include "../abstract_hardware_model.h"
#include "ptx_ir.h"

class gpgpu_context;
typedef void *yyscan_t;
class ptx_recognizer {
 public:
  ptx_recognizer(gpgpu_context *ctx) : g_return_var(ctx) {
    scanner = NULL;
    g_size = -1;
    g_add_identifier_cached__identifier = NULL;
    g_alignment_spec = -1;
    g_var_type = NULL;
    g_opcode = -1;
    g_space_spec = undefined_space;
    g_ptr_spec = undefined_space;
    g_scalar_type_spec = -1;
    g_vector_spec = -1;
    g_extern_spec = 0;
    g_func_decl = 0;
    g_ident_add_uid = 0;
    g_const_alloc = 1;
    g_max_regs_per_thread = 0;
    g_global_symbol_table = NULL;
    g_current_symbol_table = NULL;
    g_last_symbol = NULL;
    g_error_detected = 0;
    g_entry_func_param_index = 0;
    g_func_info = NULL;
    g_debug_ir_generation = false;
    gpgpu_ctx = ctx;
  }
  // global list
  yyscan_t scanner;
#define PTX_LINEBUF_SIZE (4 * 1024)
  char linebuf[PTX_LINEBUF_SIZE];
  unsigned col;
  int g_size;
  char *g_add_identifier_cached__identifier;
  int g_add_identifier_cached__array_dim;
  int g_add_identifier_cached__array_ident;
  int g_alignment_spec;
  // variable declaration stuff:
  type_info *g_var_type;
  // instruction definition stuff:
  const symbol *g_pred;
  int g_neg_pred;
  int g_pred_mod;
  symbol *g_label;
  int g_opcode;
  std::list<operand_info> g_operands;
  std::list<int> g_options;
  std::list<int> g_wmma_options;
  std::list<int> g_scalar_type;
  // type specifier stuff:
  memory_space_t g_space_spec;
  memory_space_t g_ptr_spec;
  int g_scalar_type_spec;
  int g_vector_spec;
  int g_extern_spec;
  int g_func_decl;
  int g_ident_add_uid;
  unsigned g_const_alloc;
  unsigned g_max_regs_per_thread;
  symbol_table *g_global_symbol_table;
  symbol_table *g_current_symbol_table;
  symbol *g_last_symbol;
  std::list<ptx_instruction *> g_instructions;
  int g_error_detected;
  unsigned g_entry_func_param_index;
  function_info *g_func_info;
  operand_info g_return_var;
  bool g_debug_ir_generation;
  int g_entry_point;
  const struct core_config *g_shader_core_config;
  std::map<std::string, std::map<unsigned, const ptx_instruction *> >
      g_inst_lookup;
  // the program intermediate representation...
  std::map<std::string, symbol_table *> g_sym_name_to_symbol_table;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

  // member function list
  void init_directive_state();
  void init_instruction_state();
  void start_function(int entry_point);
  void add_function_name(const char *fname);
  void add_directive();
  void end_function();
  void add_identifier(const char *s, int array_dim, unsigned array_ident);
  void add_function_arg();
  void add_scalar_type_spec(int type_spec);
  void add_scalar_operand(const char *identifier);
  void add_neg_pred_operand(const char *identifier);
  void add_variables();
  void set_variable_type();
  void add_opcode(int opcode);
  void add_pred(const char *identifier, int negate, int predModifier);
  void add_1vector_operand(const char *d1);
  void add_2vector_operand(const char *d1, const char *d2);
  void add_3vector_operand(const char *d1, const char *d2, const char *d3);
  void add_4vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4);
  void add_8vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4, const char *d5, const char *d6,
                           const char *d7, const char *d8);
  void add_option(int option);
  void add_wmma_option(int option);
  void add_builtin_operand(int builtin, int dim_modifier);
  void add_memory_operand();
  void add_literal_int(int value);
  void add_literal_float(float value);
  void add_literal_double(double value);
  void add_address_operand(const char *identifier, int offset);
  void add_address_operand2(int offset);
  void add_label(const char *idenfiier);
  void add_vector_spec(int spec);
  void add_space_spec(enum _memory_space_t spec, int value);
  void add_ptr_spec(enum _memory_space_t spec);
  void add_extern_spec();
  void add_instruction();
  void set_return();
  void add_alignment_spec(int spec);
  void add_array_initializer();
  void add_file(unsigned num, const char *filename);
  void add_version_info(float ver, unsigned ext);
  void *reset_symtab();
  void set_symtab(void *);
  void add_pragma(const char *str);
  void func_header(const char *a);
  void func_header_info(const char *a);
  void func_header_info_int(const char *a, int b);
  void add_constptr(const char *identifier1, const char *identifier2,
                    int offset);
  void target_header(char *a);
  void target_header2(char *a, char *b);
  void target_header3(char *a, char *b, char *c);
  void add_double_operand(const char *d1, const char *d2);
  void change_memory_addr_space(const char *identifier);
  void change_operand_lohi(int lohi);
  void change_double_operand_type(int addr_type);
  void change_operand_neg();
  void set_immediate_operand_type();
  void version_header(double a);
  void maxnt_id(int x, int y, int z);
  void parse_error_impl(const char *file, unsigned line, const char *msg, ...);
  void parse_assert_impl(int test_value, const char *file, unsigned line,
                         const char *msg, ...);
  // Jin: handle instructino group for cdp
  void start_inst_group();
  void end_inst_group();
  bool check_for_duplicates(const char *identifier);
  void read_parser_environment_variables();
  void set_ptx_warp_size(const struct core_config *warp_size);
  const class ptx_instruction *ptx_instruction_lookup(const char *filename,
                                                      unsigned linenumber);
};

const char *decode_token(int type);
void read_parser_environment_variables();

#define NON_ARRAY_IDENTIFIER 1
#define ARRAY_IDENTIFIER_NO_DIM 2
#define ARRAY_IDENTIFIER 3

#endif
