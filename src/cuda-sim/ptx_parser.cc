// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung
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

#include "ptx_parser.h"
#include "../../libcuda/gpgpu_context.h"
#include "ptx_ir.h"

typedef void *yyscan_t;
#include <stdarg.h>
#include "ptx.tab.h"

extern int ptx_get_lineno(yyscan_t yyscanner);
extern YYSTYPE *ptx_get_lval(yyscan_t yyscanner);
extern int ptx_error(yyscan_t yyscanner, ptx_recognizer *recognizer,
                     const char *s);
extern int ptx_lex_init(yyscan_t *scanner);
extern void ptx_set_in(FILE *_in_str, yyscan_t yyscanner);
extern FILE *ptx_get_in(yyscan_t yyscanner);
extern int ptx_parse(yyscan_t scanner, ptx_recognizer *recognizer);
extern int ptx_lex_destroy(yyscan_t scanner);

void ptx_recognizer::set_ptx_warp_size(const struct core_config *warp_size) {
  g_shader_core_config = warp_size;
}

#define PTX_PARSE_DPRINTF(...)                                            \
  if (g_debug_ir_generation) {                                            \
    printf(" %s:%u => ", gpgpu_ctx->g_filename, ptx_get_lineno(scanner)); \
    printf("   (%s:%u) ", __FILE__, __LINE__);                            \
    printf(__VA_ARGS__);                                                  \
    printf("\n");                                                         \
    fflush(stdout);                                                       \
  }

static std::map<unsigned, std::string> g_ptx_token_decode;

const char *decode_token(int type) { return g_ptx_token_decode[type].c_str(); }

void ptx_recognizer::read_parser_environment_variables() {
  gpgpu_ctx->g_filename = getenv("PTX_SIM_KERNELFILE");
  char *dbg_level = getenv("PTX_SIM_DEBUG");
  if (dbg_level && strlen(dbg_level)) {
    int debug_execution = 0;
    sscanf(dbg_level, "%d", &debug_execution);
    if (debug_execution >= 30) g_debug_ir_generation = true;
  }
}

void ptx_recognizer::init_directive_state() {
  PTX_PARSE_DPRINTF("init_directive_state");
  g_space_spec = undefined_space;
  g_ptr_spec = undefined_space;
  g_scalar_type_spec = -1;
  g_vector_spec = -1;
  g_opcode = -1;
  g_alignment_spec = -1;
  g_size = -1;
  g_extern_spec = 0;
  g_scalar_type.clear();
  g_operands.clear();
  g_last_symbol = NULL;
}

void ptx_recognizer::init_instruction_state() {
  PTX_PARSE_DPRINTF("init_instruction_state");
  g_pred = NULL;
  g_neg_pred = 0;
  g_pred_mod = -1;
  g_label = NULL;
  g_opcode = -1;
  g_options.clear();
  g_wmma_options.clear();
  g_return_var = operand_info(gpgpu_ctx);
  init_directive_state();
}

symbol_table *gpgpu_context::init_parser(const char *ptx_filename) {
  g_filename = strdup(ptx_filename);
  if (g_global_allfiles_symbol_table == NULL) {
    g_global_allfiles_symbol_table =
        new symbol_table("global_allfiles", 0, NULL, this);
    ptx_parser->g_global_symbol_table = ptx_parser->g_current_symbol_table =
        g_global_allfiles_symbol_table;
  }
  /*else {
      g_global_symbol_table = g_current_symbol_table = new
  symbol_table("global",0,g_global_allfiles_symbol_table);
  }*/

#define DEF(X, Y) g_ptx_token_decode[X] = Y;
#include "ptx_parser_decode.def"
#undef DEF
  g_ptx_token_decode[undefined_space] = "undefined_space";
  g_ptx_token_decode[undefined_space] = "undefined_space=0";
  g_ptx_token_decode[reg_space] = "reg_space";
  g_ptx_token_decode[local_space] = "local_space";
  g_ptx_token_decode[shared_space] = "shared_space";
  g_ptx_token_decode[param_space_unclassified] = "param_space_unclassified";
  g_ptx_token_decode[param_space_kernel] = "param_space_kernel";
  g_ptx_token_decode[param_space_local] = "param_space_local";
  g_ptx_token_decode[const_space] = "const_space";
  g_ptx_token_decode[tex_space] = "tex_space";
  g_ptx_token_decode[surf_space] = "surf_space";
  g_ptx_token_decode[global_space] = "global_space";
  g_ptx_token_decode[generic_space] = "generic_space";
  g_ptx_token_decode[instruction_space] = "instruction_space";

  ptx_lex_init(&(ptx_parser->scanner));
  ptx_parser->init_directive_state();
  ptx_parser->init_instruction_state();

  FILE *ptx_in;
  ptx_in = fopen(ptx_filename, "r");
  ptx_set_in(ptx_in, ptx_parser->scanner);
  ptx_parse(ptx_parser->scanner, ptx_parser);
  ptx_in = ptx_get_in(ptx_parser->scanner);
  ptx_lex_destroy(ptx_parser->scanner);
  fclose(ptx_in);
  return ptx_parser->g_global_symbol_table;
}

void ptx_recognizer::start_function(int entry_point) {
  PTX_PARSE_DPRINTF("start_function");
  init_directive_state();
  init_instruction_state();
  g_entry_point = entry_point;
  g_func_info = NULL;
  g_entry_func_param_index = 0;
}

void ptx_recognizer::add_function_name(const char *name) {
  PTX_PARSE_DPRINTF(
      "add_function_name %s %s", name,
      ((g_entry_point == 1) ? "(entrypoint)"
                            : ((g_entry_point == 2) ? "(extern)" : "")));
  bool prior_decl = g_global_symbol_table->add_function_decl(
      name, g_entry_point, &g_func_info, &g_current_symbol_table);
  if (g_add_identifier_cached__identifier) {
    add_identifier(g_add_identifier_cached__identifier,
                   g_add_identifier_cached__array_dim,
                   g_add_identifier_cached__array_ident);
    free(g_add_identifier_cached__identifier);
    g_add_identifier_cached__identifier = NULL;
    g_func_info->add_return_var(g_last_symbol);
    init_directive_state();
  }
  if (prior_decl) {
    g_func_info->remove_args();
  }
  g_global_symbol_table->add_function(g_func_info, gpgpu_ctx->g_filename,
                                      ptx_get_lineno(scanner));
}

// Jin: handle instruction group for cdp
void ptx_recognizer::start_inst_group() {
  PTX_PARSE_DPRINTF("start_instruction_group");
  g_current_symbol_table = g_current_symbol_table->start_inst_group();
}

void ptx_recognizer::end_inst_group() {
  PTX_PARSE_DPRINTF("end_instruction_group");
  g_current_symbol_table = g_current_symbol_table->end_inst_group();
}

void ptx_recognizer::add_directive() {
  PTX_PARSE_DPRINTF("add_directive");
  init_directive_state();
}

#define mymax(a, b) ((a) > (b) ? (a) : (b))

void ptx_recognizer::end_function() {
  PTX_PARSE_DPRINTF("end_function");

  init_directive_state();
  init_instruction_state();
  g_max_regs_per_thread = mymax(g_max_regs_per_thread,
                                (g_current_symbol_table->next_reg_num() - 1));
  g_func_info->add_inst(g_instructions);
  g_instructions.clear();
  gpgpu_ptx_assemble(g_func_info->get_name(), g_func_info);
  g_current_symbol_table = g_global_symbol_table;

  PTX_PARSE_DPRINTF("function %s, PC = %d\n", g_func_info->get_name().c_str(),
                    g_func_info->get_start_PC());
}

#define parse_error(msg, ...) \
  parse_error_impl(__FILE__, __LINE__, msg, ##__VA_ARGS__)
#define parse_assert(cond, msg, ...) \
  parse_assert_impl((cond), __FILE__, __LINE__, msg, ##__VA_ARGS__)

void ptx_recognizer::parse_error_impl(const char *file, unsigned line,
                                      const char *msg, ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  g_error_detected = 1;
  printf("%s:%u: Parse error: %s (%s:%u)\n\n", gpgpu_ctx->g_filename,
         ptx_get_lineno(scanner), buf, file, line);
  ptx_error(scanner, this, NULL);
  abort();
  exit(1);
}

void ptx_recognizer::parse_assert_impl(int test_value, const char *file,
                                       unsigned line, const char *msg, ...) {
  va_list ap;
  char buf[1024];
  va_start(ap, msg);
  vsnprintf(buf, 1024, msg, ap);
  va_end(ap);

  if (test_value == 0) parse_error_impl(file, line, msg);
}

void ptx_recognizer::set_return() {
  parse_assert((g_opcode == CALL_OP || g_opcode == CALLP_OP),
               "only call can have return value");
  g_operands.front().set_return();
  g_return_var = g_operands.front();
}

const ptx_instruction *ptx_recognizer::ptx_instruction_lookup(
    const char *filename, unsigned linenumber) {
  std::map<std::string, std::map<unsigned, const ptx_instruction *> >::iterator
      f = g_inst_lookup.find(filename);
  if (f == g_inst_lookup.end()) return NULL;
  std::map<unsigned, const ptx_instruction *>::iterator l =
      f->second.find(linenumber);
  if (l == f->second.end()) return NULL;
  return l->second;
}

void ptx_recognizer::add_instruction() {
  PTX_PARSE_DPRINTF("add_instruction: %s",
                    ((g_opcode > 0) ? g_opcode_string[g_opcode] : "<label>"));
  assert(g_shader_core_config != 0);
  ptx_instruction *i = new ptx_instruction(
      g_opcode, g_pred, g_neg_pred, g_pred_mod, g_label, g_operands,
      g_return_var, g_options, g_wmma_options, g_scalar_type, g_space_spec,
      gpgpu_ctx->g_filename, ptx_get_lineno(scanner), linebuf,
      g_shader_core_config, gpgpu_ctx);
  g_instructions.push_back(i);
  g_inst_lookup[gpgpu_ctx->g_filename][ptx_get_lineno(scanner)] = i;
  init_instruction_state();
}

void ptx_recognizer::add_variables() {
  PTX_PARSE_DPRINTF("add_variables");
  if (!g_operands.empty()) {
    assert(g_last_symbol != NULL);
    g_last_symbol->add_initializer(g_operands);
  }
  init_directive_state();
}

void ptx_recognizer::set_variable_type() {
  PTX_PARSE_DPRINTF("set_variable_type space_spec=%s scalar_type_spec=%s",
                    g_ptx_token_decode[g_space_spec.get_type()].c_str(),
                    g_ptx_token_decode[g_scalar_type_spec].c_str());
  parse_assert(g_space_spec != undefined_space,
               "variable has no space specification");
  parse_assert(
      g_scalar_type_spec != -1,
      "variable has no type information");  // need to extend for structs?
  g_var_type = g_current_symbol_table->add_type(
      g_space_spec, g_scalar_type_spec, g_vector_spec, g_alignment_spec,
      g_extern_spec);
}

bool ptx_recognizer::check_for_duplicates(const char *identifier) {
  const symbol *s = g_current_symbol_table->lookup(identifier);
  return (s != NULL);
}

// Returns padding that needs to be inserted ahead of address to make it aligned
// to min(size, maxalign)
/*
 * @param address the address in bytes
 * @param size the size of the memory to be allocated in bytes
 * @param maximum alignment in bytes. i.e. if size is too big then align to this
 * instead
 */
int pad_address(new_addr_type address, unsigned size, unsigned maxalign) {
  assert(size >= 0);
  assert(maxalign > 0);
  int alignto = maxalign;
  if (size < maxalign && (size & (size - 1)) == 0) {  // size is a power of 2
    alignto = size;
  }
  return alignto ? ((alignto - (address % alignto)) % alignto) : 0;
}

void ptx_recognizer::add_identifier(const char *identifier, int array_dim,
                                    unsigned array_ident) {
  if (array_ident == ARRAY_IDENTIFIER) {
    g_size *= array_dim;
  }
  if (g_func_decl && (g_func_info == NULL)) {
    // return variable decl...
    assert(g_add_identifier_cached__identifier == NULL);
    g_add_identifier_cached__identifier = strdup(identifier);
    g_add_identifier_cached__array_dim = array_dim;
    g_add_identifier_cached__array_ident = array_ident;
    return;
  }
  PTX_PARSE_DPRINTF("add_identifier \"%s\" (%u)", identifier, g_ident_add_uid);
  g_ident_add_uid++;
  type_info *type = g_var_type;
  type_info_key ti = type->get_key();
  int basic_type;
  int regnum;
  size_t num_bits;
  unsigned addr_pad;
  new_addr_type addr;
  ti.type_decode(num_bits, basic_type);

  bool duplicates = check_for_duplicates(identifier);
  if (duplicates) {
    symbol *s = g_current_symbol_table->lookup(identifier);
    g_last_symbol = s;
    if (g_func_decl) return;
    std::string msg = std::string(identifier) + " was declared previous at " +
                      s->decl_location() + " skipping new declaration";
    printf("GPGPU-Sim PTX: Warning %s\n", msg.c_str());
    return;
  }

  assert(g_var_type != NULL);
  switch (array_ident) {
    case ARRAY_IDENTIFIER:
      type = g_current_symbol_table->get_array_type(type, array_dim);
      num_bits = array_dim * num_bits;
      break;
    case ARRAY_IDENTIFIER_NO_DIM:
      type = g_current_symbol_table->get_array_type(type, (unsigned)-1);
      num_bits = 0;
      break;
    default:
      break;
  }
  g_last_symbol = g_current_symbol_table->add_variable(
      identifier, type, num_bits / 8, gpgpu_ctx->g_filename,
      ptx_get_lineno(scanner));
  switch (ti.get_memory_space().get_type()) {
    case reg_space: {
      regnum = g_current_symbol_table->next_reg_num();
      int arch_regnum = -1;
      for (int d = 0; d < strlen(identifier); d++) {
        if (isdigit(identifier[d])) {
          sscanf(identifier + d, "%d", &arch_regnum);
          break;
        }
      }
      if (strcmp(identifier, "%sp") == 0) {
        arch_regnum = 0;
      }
      g_last_symbol->set_regno(regnum, arch_regnum);
    } break;
    case shared_space:
      printf("GPGPU-Sim PTX: allocating shared region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_shared_next();
      addr_pad = pad_address(addr, num_bits / 8, 128);
      printf("from 0x%llx to 0x%llx (shared memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_shared(num_bits / 8 + addr_pad);
      break;
    case sstarr_space:
      printf("GPGPU-Sim PTX: allocating sstarr region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_sstarr_next();
      addr_pad = pad_address(addr, num_bits / 8, 128);
      printf("from 0x%llx to 0x%llx (sstarr memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_sstarr(num_bits / 8 + addr_pad);
      break;
    case const_space:
      if (array_ident == ARRAY_IDENTIFIER_NO_DIM) {
        printf(
            "GPGPU-Sim PTX: deferring allocation of constant region for \"%s\" "
            "(need size information)\n",
            identifier);
      } else {
        printf("GPGPU-Sim PTX: allocating constant region for \"%s\" ",
               identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_global_next();
        addr_pad = pad_address(addr, num_bits / 8, 128);
        printf("from 0x%llx to 0x%llx (global memory space) %u\n",
               addr + addr_pad, addr + addr_pad + num_bits / 8,
               g_const_alloc++);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_global(num_bits / 8 + addr_pad);
      }
      if (g_current_symbol_table == g_global_symbol_table) {
        gpgpu_ctx->func_sim->g_constants.insert(identifier);
      }
      assert(g_current_symbol_table != NULL);
      g_sym_name_to_symbol_table[identifier] = g_current_symbol_table;
      break;
    case global_space:
      printf("GPGPU-Sim PTX: allocating global region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_global_next();
      addr_pad = pad_address(addr, num_bits / 8, 128);
      printf("from 0x%llx to 0x%llx (global memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_global(num_bits / 8 + addr_pad);
      gpgpu_ctx->func_sim->g_globals.insert(identifier);
      assert(g_current_symbol_table != NULL);
      g_sym_name_to_symbol_table[identifier] = g_current_symbol_table;
      break;
    case local_space:
      if (g_func_info == NULL) {
        printf("GPGPU-Sim PTX: allocating local region for \"%s\" ",
               identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_local_next();
        addr_pad = pad_address(addr, num_bits / 8, 128);
        printf("from 0x%llx to 0x%llx (local memory space)\n", addr + addr_pad,
               addr + addr_pad + num_bits / 8);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_local(num_bits / 8 + addr_pad);
      } else {
        printf(
            "GPGPU-Sim PTX: allocating stack frame region for .local \"%s\" ",
            identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_local_next();
        addr_pad = pad_address(addr, num_bits / 8, 128);
        printf("from 0x%llx to 0x%llx\n", addr + addr_pad,
               addr + addr_pad + num_bits / 8);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_local(num_bits / 8 + addr_pad);
        g_func_info->set_framesize(g_current_symbol_table->get_local_next());
      }
      break;
    case tex_space:
      printf("GPGPU-Sim PTX: encountered texture directive %s.\n", identifier);
      break;
    case param_space_local:
      printf(
          "GPGPU-Sim PTX: allocating stack frame region for .param \"%s\" from "
          "0x%x to 0x%lx\n",
          identifier, g_current_symbol_table->get_local_next(),
          g_current_symbol_table->get_local_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      g_last_symbol->set_address(g_current_symbol_table->get_local_next());
      g_current_symbol_table->alloc_local(num_bits / 8);
      g_func_info->set_framesize(g_current_symbol_table->get_local_next());
      break;
    case param_space_kernel:
      break;
    default:
      abort();
      break;
  }

  assert(!ti.is_param_unclassified());
  if (ti.is_param_kernel()) {
    bool is_ptr = (g_ptr_spec != undefined_space);
    g_func_info->add_param_name_type_size(g_entry_func_param_index, identifier,
                                          ti.scalar_type(), num_bits, is_ptr,
                                          g_ptr_spec);
    g_entry_func_param_index++;
  }
}

void ptx_recognizer::add_constptr(const char *identifier1,
                                  const char *identifier2, int offset) {
  symbol *s1 = g_current_symbol_table->lookup(identifier1);
  const symbol *s2 = g_current_symbol_table->lookup(identifier2);
  parse_assert(s1 != NULL, "'from' constant identifier does not exist.");
  parse_assert(s1 != NULL, "'to' constant identifier does not exist.");

  unsigned addr = s2->get_address();

  printf("GPGPU-Sim PTX: moving \"%s\" from 0x%x to 0x%x (%s+%x)\n",
         identifier1, s1->get_address(), addr + offset, identifier2, offset);

  s1->set_address(addr + offset);
}

void ptx_recognizer::add_function_arg() {
  assert(g_size > 0);
  if (g_func_info) {
    PTX_PARSE_DPRINTF("add_function_arg \"%s\"", g_last_symbol->name().c_str());
    g_func_info->add_arg(g_last_symbol);
    unsigned alignment = (g_alignment_spec == -1) ? g_size : g_alignment_spec;
    assert(alignment == 1 || alignment == 2 || alignment == 4 ||
           alignment == 8 || alignment == 16);  // known valid alignment values
    g_func_info->add_config_param(g_size, alignment);
  }
}

void ptx_recognizer::add_extern_spec() {
  PTX_PARSE_DPRINTF("add_extern_spec");
  g_extern_spec = 1;
}

void ptx_recognizer::add_alignment_spec(int spec) {
  PTX_PARSE_DPRINTF("add_alignment_spec");
  parse_assert(
      g_alignment_spec == -1,
      "multiple .align specifiers per variable declaration not allowed.");
  g_alignment_spec = spec;
}

void ptx_recognizer::add_ptr_spec(enum _memory_space_t spec) {
  PTX_PARSE_DPRINTF("add_ptr_spec \"%s\"", g_ptx_token_decode[spec].c_str());
  parse_assert(g_ptr_spec == undefined_space,
               "multiple ptr space specifiers not allowed.");
  parse_assert(
      spec == global_space or spec == local_space or spec == shared_space,
      "invalid space for ptr directive.");
  g_ptr_spec = spec;
}

void ptx_recognizer::add_space_spec(enum _memory_space_t spec, int value) {
  PTX_PARSE_DPRINTF("add_space_spec \"%s\"", g_ptx_token_decode[spec].c_str());
  parse_assert(g_space_spec == undefined_space,
               "multiple space specifiers not allowed.");
  if (spec == param_space_unclassified) {
    if (g_func_decl) {
      if (g_entry_point == 1)
        g_space_spec = param_space_kernel;
      else
        g_space_spec = param_space_local;
    } else
      g_space_spec = param_space_unclassified;
  } else {
    g_space_spec = spec;
    if (g_space_spec == const_space) g_space_spec.set_bank((unsigned)value);
  }
}

void ptx_recognizer::add_vector_spec(int spec) {
  PTX_PARSE_DPRINTF("add_vector_spec");
  parse_assert(g_vector_spec == -1, "multiple vector specifiers not allowed.");
  g_vector_spec = spec;
}

void ptx_recognizer::add_scalar_type_spec(int type_spec) {
  // save size of parameter
  switch (type_spec) {
    case B8_TYPE:
    case S8_TYPE:
    case U8_TYPE:
      g_size = 1;
      break;
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
    case F16_TYPE:
      g_size = 2;
      break;
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
    case F32_TYPE:
      g_size = 4;
      break;
    case B64_TYPE:
    case BB64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      g_size = 8;
      break;
    case BB128_TYPE:
      g_size = 16;
      break;
  }
  PTX_PARSE_DPRINTF("add_scalar_type_spec \"%s\"",
                    g_ptx_token_decode[type_spec].c_str());
  g_scalar_type.push_back(type_spec);
  if (g_scalar_type.size() > 1) {
    parse_assert((g_opcode == -1) || (g_opcode == CVT_OP) ||
                     (g_opcode == SET_OP) || (g_opcode == SLCT_OP) ||
                     (g_opcode == TEX_OP) || (g_opcode == MMA_OP) ||
                     (g_opcode == DP4A_OP) || (g_opcode == VMIN_OP) || 
                     (g_opcode == VMAX_OP),
                 "only cvt, set, slct, tex, vmin, vmax and dp4a can have more than one "
                 "type specifier.");
  }
  g_scalar_type_spec = type_spec;
}

void ptx_recognizer::add_label(const char *identifier) {
  PTX_PARSE_DPRINTF("add_label");
  symbol *s = g_current_symbol_table->lookup(identifier);
  if (s != NULL) {
    g_label = s;
  } else {
    g_label = g_current_symbol_table->add_variable(
        identifier, NULL, 0, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
  }
}

void ptx_recognizer::add_opcode(int opcode) { g_opcode = opcode; }

void ptx_recognizer::add_pred(const char *identifier, int neg,
                              int predModifier) {
  PTX_PARSE_DPRINTF("add_pred");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    std::string msg =
        std::string("predicate \"") + identifier + "\" has no declaration.";
    parse_error(msg.c_str());
  }
  g_pred = s;
  g_neg_pred = neg;
  g_pred_mod = predModifier;
}

void ptx_recognizer::add_option(int option) {
  PTX_PARSE_DPRINTF("add_option");
  g_options.push_back(option);
}
void ptx_recognizer::add_wmma_option(int option) {
  PTX_PARSE_DPRINTF("add_option");
  g_wmma_options.push_back(option);
}
void ptx_recognizer::add_double_operand(const char *d1, const char *d2) {
  // operands that access two variables.
  // eg. s[$ofs1+$r0], g[$ofs1+=$r0]
  // TODO: Not sure if I'm going to use this for storing to two destinations or
  // not.

  PTX_PARSE_DPRINTF("add_double_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  parse_assert(s1 != NULL && s2 != NULL, "component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, s2, gpgpu_ctx));
}

void ptx_recognizer::add_1vector_operand(const char *d1) {
  // handles the single element vector operand ({%v1}) found in tex.1d
  // instructions
  PTX_PARSE_DPRINTF("add_1vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  parse_assert(s1 != NULL, "component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, NULL, NULL, NULL, gpgpu_ctx));
}

void ptx_recognizer::add_2vector_operand(const char *d1, const char *d2) {
  PTX_PARSE_DPRINTF("add_2vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  parse_assert(s1 != NULL && s2 != NULL,
               "v2 component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, s2, NULL, NULL, gpgpu_ctx));
}

void ptx_recognizer::add_3vector_operand(const char *d1, const char *d2,
                                         const char *d3) {
  PTX_PARSE_DPRINTF("add_3vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  const symbol *s3 = g_current_symbol_table->lookup(d3);
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL,
               "v3 component(s) missing declarations.");
  g_operands.push_back(operand_info(s1, s2, s3, NULL, gpgpu_ctx));
}

void ptx_recognizer::add_4vector_operand(const char *d1, const char *d2,
                                         const char *d3, const char *d4) {
  PTX_PARSE_DPRINTF("add_4vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  const symbol *s3 = g_current_symbol_table->lookup(d3);
  const symbol *s4 = g_current_symbol_table->lookup(d4);
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL,
               "v4 component(s) missing declarations.");
  const symbol *null_op = g_current_symbol_table->lookup("_");
  if (s2 == null_op) s2 = NULL;
  if (s3 == null_op) s3 = NULL;
  if (s4 == null_op) s4 = NULL;
  g_operands.push_back(operand_info(s1, s2, s3, s4, gpgpu_ctx));
}
void ptx_recognizer::add_8vector_operand(const char *d1, const char *d2,
                                         const char *d3, const char *d4,
                                         const char *d5, const char *d6,
                                         const char *d7, const char *d8) {
  PTX_PARSE_DPRINTF("add_8vector_operand");
  const symbol *s1 = g_current_symbol_table->lookup(d1);
  const symbol *s2 = g_current_symbol_table->lookup(d2);
  const symbol *s3 = g_current_symbol_table->lookup(d3);
  const symbol *s4 = g_current_symbol_table->lookup(d4);
  const symbol *s5 = g_current_symbol_table->lookup(d5);
  const symbol *s6 = g_current_symbol_table->lookup(d6);
  const symbol *s7 = g_current_symbol_table->lookup(d7);
  const symbol *s8 = g_current_symbol_table->lookup(d8);
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL &&
                   s5 != NULL && s6 != NULL && s7 != NULL && s8 != NULL,
               "v4 component(s) missing declarations.");
  const symbol *null_op = g_current_symbol_table->lookup("_");
  if (s2 == null_op) s2 = NULL;
  if (s3 == null_op) s3 = NULL;
  if (s4 == null_op) s4 = NULL;
  if (s5 == null_op) s5 = NULL;
  if (s6 == null_op) s6 = NULL;
  if (s7 == null_op) s7 = NULL;
  if (s8 == null_op) s8 = NULL;
  g_operands.push_back(operand_info(s1, s2, s3, s4, s5, s6, s7, s8, gpgpu_ctx));
}

void ptx_recognizer::add_builtin_operand(int builtin, int dim_modifier) {
  PTX_PARSE_DPRINTF("add_builtin_operand");
  g_operands.push_back(operand_info(builtin, dim_modifier, gpgpu_ctx));
}

void ptx_recognizer::add_memory_operand() {
  PTX_PARSE_DPRINTF("add_memory_operand");
  assert(!g_operands.empty());
  g_operands.back().make_memory_operand();
}

/*TODO: add other memory locations*/
void ptx_recognizer::change_memory_addr_space(const char *identifier) {
  /*0 = N/A, not reading from memory
   *1 = global memory
   *2 = shared memory
   *3 = const memory segment
   *4 = local memory segment
   */

  bool recognizedType = false;

  PTX_PARSE_DPRINTF("change_memory_addr_space");
  assert(!g_operands.empty());
  if (!strcmp(identifier, "g")) {
    g_operands.back().set_addr_space(global_space);
    recognizedType = true;
  }
  if (!strcmp(identifier, "s")) {
    g_operands.back().set_addr_space(shared_space);
    recognizedType = true;
  }
  // For constants, check if the first character is 'c'
  char c[2];
  strncpy(c, identifier, 1);
  c[1] = '\0';
  if (!strcmp(c, "c")) {
    g_operands.back().set_addr_space(const_space);
    parse_assert(g_current_symbol_table->lookup(identifier) != NULL,
                 "Constant was not defined.");
    g_operands.back().set_const_mem_offset(
        g_current_symbol_table->lookup(identifier)->get_address());
    recognizedType = true;
  }
  // For local memory, check if the first character is 'l'
  char l[2];
  strncpy(l, identifier, 1);
  l[1] = '\0';
  if (!strcmp(l, "l")) {
    g_operands.back().set_addr_space(local_space);
    // parse_assert(g_current_symbol_table->lookup(identifier) != NULL, "Local
    // memory segment was not defined.");
    // g_operands.back().set_const_mem_offset(g_current_symbol_table->lookup(identifier)->get_address());
    recognizedType = true;
  }

  parse_assert(recognizedType, "Error: unrecognized memory type.");
}

void ptx_recognizer::change_operand_lohi(int lohi) {
  /*0 = N/A, read entire operand
   *1 = lo, reading from lowest bits
   *2 = hi, reading from highest bits
   */

  PTX_PARSE_DPRINTF("change_operand_lohi");
  assert(!g_operands.empty());

  g_operands.back().set_operand_lohi(lohi);
}

void ptx_recognizer::set_immediate_operand_type() {
  PTX_PARSE_DPRINTF("set_immediate_operand_type");
  assert(!g_operands.empty());
  g_operands.back().set_immediate_addr();
}

void ptx_recognizer::change_double_operand_type(int operand_type) {
  /*
   *-3 = reg / reg (set instruction, but both get same value)
   *-2 = reg | reg (cvt instruction)
   *-1 = reg | reg (set instruction)
   *0 = N/A, default
   *1 = reg + reg
   *2 = reg += reg
   *3 = reg += immediate
   */

  PTX_PARSE_DPRINTF("change_double_operand_type");
  assert(!g_operands.empty());

  // For double destination operands, ensure valid instruction
  if (operand_type == -1 || operand_type == -2) {
    if ((g_opcode == SET_OP) || (g_opcode == SETP_OP))
      g_operands.back().set_double_operand_type(-1);
    else
      g_operands.back().set_double_operand_type(-2);
  } else if (operand_type == -3) {
    if (g_opcode == SET_OP || g_opcode == MAD_OP)
      g_operands.back().set_double_operand_type(operand_type);
    else
      parse_assert(0, "Error: Unsupported use of double destination operand.");
  } else {
    g_operands.back().set_double_operand_type(operand_type);
  }
}

void ptx_recognizer::change_operand_neg() {
  PTX_PARSE_DPRINTF("change_operand_neg");
  assert(!g_operands.empty());

  g_operands.back().set_operand_neg();
}

void ptx_recognizer::add_literal_int(int value) {
  PTX_PARSE_DPRINTF("add_literal_int");
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_literal_float(float value) {
  PTX_PARSE_DPRINTF("add_literal_float");
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_literal_double(double value) {
  PTX_PARSE_DPRINTF("add_literal_double");
  g_operands.push_back(operand_info(value, gpgpu_ctx));
}

void ptx_recognizer::add_scalar_operand(const char *identifier) {
  PTX_PARSE_DPRINTF("add_scalar_operand");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    if (g_opcode == BRA_OP || g_opcode == CALLP_OP) {
      // forward branch target...
      s = g_current_symbol_table->add_variable(
          identifier, NULL, 0, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
    } else {
      std::string msg =
          std::string("operand \"") + identifier + "\" has no declaration.";
      parse_error(msg.c_str());
    }
  }
  g_operands.push_back(operand_info(s, gpgpu_ctx));
}

void ptx_recognizer::add_neg_pred_operand(const char *identifier) {
  PTX_PARSE_DPRINTF("add_neg_pred_operand");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    s = g_current_symbol_table->add_variable(
        identifier, NULL, 1, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
  }
  operand_info op(s, gpgpu_ctx);
  op.set_neg_pred();
  g_operands.push_back(op);
}

void ptx_recognizer::add_address_operand(const char *identifier, int offset) {
  PTX_PARSE_DPRINTF("add_address_operand");
  const symbol *s = g_current_symbol_table->lookup(identifier);
  if (s == NULL) {
    std::string msg =
        std::string("operand \"") + identifier + "\" has no declaration.";
    parse_error(msg.c_str());
  }
  g_operands.push_back(operand_info(s, offset, gpgpu_ctx));
}

void ptx_recognizer::add_address_operand2(int offset) {
  PTX_PARSE_DPRINTF("add_address_operand");
  g_operands.push_back(operand_info((unsigned)offset, gpgpu_ctx));
}

void ptx_recognizer::add_array_initializer() {
  g_last_symbol->add_initializer(g_operands);
}

void ptx_recognizer::add_version_info(float ver, unsigned ext) {
  g_global_symbol_table->set_ptx_version(ver, ext);
}

void ptx_recognizer::add_file(unsigned num, const char *filename) {
  if (gpgpu_ctx->g_filename == NULL) {
    char *b = strdup(filename);
    char *l = b;
    char *n = b;
    while (*n != '\0') {
      if (*n == '/') l = n + 1;
      n++;
    }

    char *p = strtok(l, ".");
    char buf[1024];
    snprintf(buf, 1024, "%s.ptx", p);

    char *q = strtok(NULL, ".");
    if (q && !strcmp(q, "cu")) {
      gpgpu_ctx->g_filename = strdup(buf);
    }

    free(b);
  }

  g_current_symbol_table = g_global_symbol_table;
}

void *ptx_recognizer::reset_symtab() {
  void *result = g_current_symbol_table;
  g_current_symbol_table = g_global_symbol_table;
  return result;
}

void ptx_recognizer::set_symtab(void *symtab) {
  g_current_symbol_table = (symbol_table *)symtab;
}

void ptx_recognizer::add_pragma(const char *str) {
  printf("GPGPU-Sim PTX: Warning -- ignoring pragma '%s'\n", str);
}

void ptx_recognizer::version_header(double a) {}  // intentional dummy function

void ptx_recognizer::target_header(char *a) {
  g_global_symbol_table->set_sm_target(a, NULL, NULL);
}

void ptx_recognizer::target_header2(char *a, char *b) {
  g_global_symbol_table->set_sm_target(a, b, NULL);
}

void ptx_recognizer::target_header3(char *a, char *b, char *c) {
  g_global_symbol_table->set_sm_target(a, b, c);
}

void ptx_recognizer::maxnt_id(int x, int y, int z) {
  g_func_info->set_maxnt_id(x * y * z);
}

void ptx_recognizer::func_header(const char *a) {}  // intentional dummy
                                                    // function
void ptx_recognizer::func_header_info(const char *a) {
}  // intentional dummy function
void ptx_recognizer::func_header_info_int(const char *a, int b) {
}  // intentional dummy function
