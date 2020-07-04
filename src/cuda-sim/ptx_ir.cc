// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan
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

#include "ptx_ir.h"
#include "ptx_parser.h"
typedef void *yyscan_t;
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <list>
#include "assert.h"
#include "opcodes.h"
#include "ptx.tab.h"

#include "../../libcuda/gpgpu_context.h"
#include "cuda-sim.h"

#define STR_SIZE 1024

const ptx_instruction *gpgpu_context::pc_to_instruction(unsigned pc) {
  if (pc < s_g_pc_to_insn.size())
    return s_g_pc_to_insn[pc];
  else
    return NULL;
}

unsigned symbol::get_uid() {
  unsigned result = (gpgpu_ctx->symbol_sm_next_uid)++;
  return result;
}

void symbol::add_initializer(const std::list<operand_info> &init) {
  m_initializer = init;
}

void symbol::print_info(FILE *fp) const {
  fprintf(fp, "uid:%u, decl:%s, type:%p, ", m_uid, m_decl_location.c_str(),
          m_type);
  if (m_address_valid) fprintf(fp, "<address valid>, ");
  if (m_is_label) fprintf(fp, " is_label ");
  if (m_is_shared) fprintf(fp, " is_shared ");
  if (m_is_const) fprintf(fp, " is_const ");
  if (m_is_global) fprintf(fp, " is_global ");
  if (m_is_local) fprintf(fp, " is_local ");
  if (m_is_tex) fprintf(fp, " is_tex ");
  if (m_is_func_addr) fprintf(fp, " is_func_addr ");
  if (m_function) fprintf(fp, " %p ", m_function);
}

symbol_table::symbol_table() { assert(0); }

symbol_table::symbol_table(const char *scope_name, unsigned entry_point,
                           symbol_table *parent, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_scope_name = std::string(scope_name);
  m_reg_allocator = 0;
  m_shared_next = 0;
  m_const_next = 0;
  m_global_next = 0x100;
  m_local_next = 0;
  m_tex_next = 0;

  // Jin: handle instruction group for cdp
  m_inst_group_id = 0;

  m_parent = parent;
  if (m_parent) {
    m_shared_next = m_parent->m_shared_next;
    m_global_next = m_parent->m_global_next;
  }
}

void symbol_table::set_name(const char *name) {
  m_scope_name = std::string(name);
}

const ptx_version &symbol_table::get_ptx_version() const {
  if (m_parent == NULL)
    return m_ptx_version;
  else
    return m_parent->get_ptx_version();
}

unsigned symbol_table::get_sm_target() const {
  if (m_parent == NULL)
    return m_ptx_version.target();
  else
    return m_parent->get_sm_target();
}

void symbol_table::set_ptx_version(float ver, unsigned ext) {
  m_ptx_version = ptx_version(ver, ext);
}

void symbol_table::set_sm_target(const char *target, const char *ext,
                                 const char *ext2) {
  m_ptx_version.set_target(target, ext, ext2);
}

symbol *symbol_table::lookup(const char *identifier) {
  std::string key(identifier);
  std::map<std::string, symbol *>::iterator i = m_symbols.find(key);
  if (i != m_symbols.end()) {
    return i->second;
  }
  if (m_parent) {
    return m_parent->lookup(identifier);
  }
  return NULL;
}

symbol *symbol_table::add_variable(const char *identifier,
                                   const type_info *type, unsigned size,
                                   const char *filename, unsigned line) {
  char buf[1024];
  std::string key(identifier);
  assert(m_symbols.find(key) == m_symbols.end());
  snprintf(buf, 1024, "%s:%u", filename, line);
  symbol *s = new symbol(identifier, type, buf, size, gpgpu_ctx);
  m_symbols[key] = s;

  if (type != NULL && type->get_key().is_global()) {
    m_globals.push_back(s);
  }
  if (type != NULL && type->get_key().is_const()) {
    m_consts.push_back(s);
  }

  return s;
}

void symbol_table::add_function(function_info *func, const char *filename,
                                unsigned linenumber) {
  std::map<std::string, symbol *>::iterator i =
      m_symbols.find(func->get_name());
  if (i != m_symbols.end()) return;
  char buf[1024];
  snprintf(buf, 1024, "%s:%u", filename, linenumber);
  type_info *type = add_type(func);
  symbol *s = new symbol(func->get_name().c_str(), type, buf, 0, gpgpu_ctx);
  s->set_function(func);
  m_symbols[func->get_name()] = s;
}

// Jin: handle instruction group for cdp
symbol_table *symbol_table::start_inst_group() {
  char inst_group_name[4096];
  snprintf(inst_group_name, 4096, "%s_inst_group_%u", m_scope_name.c_str(),
           m_inst_group_id);

  // previous added
  assert(m_inst_group_symtab.find(std::string(inst_group_name)) ==
         m_inst_group_symtab.end());
  symbol_table *sym_table =
      new symbol_table(inst_group_name, 3 /*inst group*/, this, gpgpu_ctx);

  sym_table->m_global_next = m_global_next;
  sym_table->m_shared_next = m_shared_next;
  sym_table->m_local_next = m_local_next;
  sym_table->m_reg_allocator = m_reg_allocator;
  sym_table->m_tex_next = m_tex_next;
  sym_table->m_const_next = m_const_next;

  m_inst_group_symtab[std::string(inst_group_name)] = sym_table;

  return sym_table;
}

symbol_table *symbol_table::end_inst_group() {
  symbol_table *sym_table = m_parent;

  sym_table->m_global_next = m_global_next;
  sym_table->m_shared_next = m_shared_next;
  sym_table->m_local_next = m_local_next;
  sym_table->m_reg_allocator = m_reg_allocator;
  sym_table->m_tex_next = m_tex_next;
  sym_table->m_const_next = m_const_next;
  sym_table->m_inst_group_id++;

  return sym_table;
}

void register_ptx_function(const char *name,
                           function_info *impl);  // either libcuda or libopencl

bool symbol_table::add_function_decl(const char *name, int entry_point,
                                     function_info **func_info,
                                     symbol_table **sym_table) {
  std::string key = std::string(name);
  bool prior_decl = false;
  if (m_function_info_lookup.find(key) != m_function_info_lookup.end()) {
    *func_info = m_function_info_lookup[key];
    prior_decl = true;
  } else {
    *func_info = new function_info(entry_point, gpgpu_ctx);
    (*func_info)->set_name(name);
    (*func_info)->set_maxnt_id(0);
    m_function_info_lookup[key] = *func_info;
  }

  if (m_function_symtab_lookup.find(key) != m_function_symtab_lookup.end()) {
    assert(prior_decl);
    *sym_table = m_function_symtab_lookup[key];
  } else {
    assert(!prior_decl);
    *sym_table = new symbol_table("", entry_point, this, gpgpu_ctx);

    // Initial setup code to support a register represented as "_".
    // This register is used when an instruction operand is
    // not read or written.  However, the parser must recognize it
    // as a legitimate register but we do not want to pass
    // it to the micro-architectural register to the performance simulator.
    // For this purpose we add a symbol to the symbol table but
    // mark it as a non_arch_reg so it does not effect the performance sim.
    type_info_key null_key(reg_space, 0, 0, 0, 0, 0);
    null_key.set_is_non_arch_reg();
    // First param is null - which is bad.
    // However, the first parameter is actually unread in the constructor...
    // TODO - remove the symbol_table* from type_info
    type_info *null_type_info = new type_info(NULL, null_key);
    symbol *null_reg =
        (*sym_table)->add_variable("_", null_type_info, 0, "", 0);
    null_reg->set_regno(0, 0);

    (*sym_table)->set_name(name);
    (*func_info)->set_symtab(*sym_table);
    m_function_symtab_lookup[key] = *sym_table;
    assert((*func_info)->get_symtab() == *sym_table);
    register_ptx_function(name, *func_info);
  }
  return prior_decl;
}

function_info *symbol_table::lookup_function(std::string name) {
  std::string key = std::string(name);
  std::map<std::string, function_info *>::iterator it =
      m_function_info_lookup.find(key);
  assert(it != m_function_info_lookup.end());
  return it->second;
}

type_info *symbol_table::add_type(memory_space_t space_spec,
                                  int scalar_type_spec, int vector_spec,
                                  int alignment_spec, int extern_spec) {
  if (space_spec == param_space_unclassified) space_spec = param_space_local;
  type_info_key t(space_spec, scalar_type_spec, vector_spec, alignment_spec,
                  extern_spec, 0);
  type_info *pt;
  pt = new type_info(this, t);
  return pt;
}

type_info *symbol_table::add_type(function_info *func) {
  type_info_key t;
  type_info *pt;
  t.set_is_func();
  pt = new type_info(this, t);
  return pt;
}

type_info *symbol_table::get_array_type(type_info *base_type,
                                        unsigned array_dim) {
  type_info_key t = base_type->get_key();
  t.set_array_dim(array_dim);
  type_info *pt = new type_info(this, t);
  // Where else is m_types being used? As of now, I dont find any use of it and
  // causing seg fault. So disabling m_types.
  // TODO: find where m_types can be used in future and solve the seg fault.
  // pt = m_types[t] = new type_info(this,t);
  return pt;
}

void symbol_table::set_label_address(const symbol *label, unsigned addr) {
  std::map<std::string, symbol *>::iterator i = m_symbols.find(label->name());
  assert(i != m_symbols.end());
  symbol *s = i->second;
  s->set_label_address(addr);
}

void symbol_table::dump() {
  printf("\n\n");
  printf("Symbol table for \"%s\":\n", m_scope_name.c_str());
  std::map<std::string, symbol *>::iterator i;
  for (i = m_symbols.begin(); i != m_symbols.end(); i++) {
    printf("%30s : ", i->first.c_str());
    if (i->second)
      i->second->print_info(stdout);
    else
      printf(" <no symbol object> ");
    printf("\n");
  }
  printf("\n");
}

unsigned operand_info::get_uid() {
  unsigned result = (gpgpu_ctx->operand_info_sm_next_uid)++;
  return result;
}

std::list<ptx_instruction *>::iterator
function_info::find_next_real_instruction(
    std::list<ptx_instruction *>::iterator i) {
  while ((i != m_instructions.end()) && (*i)->is_label()) i++;
  return i;
}

void function_info::create_basic_blocks() {
  std::list<ptx_instruction *> leaders;
  std::list<ptx_instruction *>::iterator i, l;

  // first instruction is a leader
  i = m_instructions.begin();
  leaders.push_back(*i);
  i++;
  while (i != m_instructions.end()) {
    ptx_instruction *pI = *i;
    if (pI->is_label()) {
      leaders.push_back(pI);
      i = find_next_real_instruction(++i);
    } else {
      switch (pI->get_opcode()) {
        case BRA_OP:
        case RET_OP:
        case EXIT_OP:
        case RETP_OP:
        case BREAK_OP:
          i++;
          if (i != m_instructions.end()) leaders.push_back(*i);
          i = find_next_real_instruction(i);
          break;
        case CALL_OP:
        case CALLP_OP:
          if (pI->has_pred()) {
            printf("GPGPU-Sim PTX: Warning found predicated call\n");
            i++;
            if (i != m_instructions.end()) leaders.push_back(*i);
            i = find_next_real_instruction(i);
          } else
            i++;
          break;
        default:
          i++;
      }
    }
  }

  if (leaders.empty()) {
    printf("GPGPU-Sim PTX: Function \'%s\' has no basic blocks\n",
           m_name.c_str());
    return;
  }

  unsigned bb_id = 0;
  l = leaders.begin();
  i = m_instructions.begin();
  m_basic_blocks.push_back(
      new basic_block_t(bb_id++, *find_next_real_instruction(i), NULL, 1, 0));
  ptx_instruction *last_real_inst = *(l++);

  for (; i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;
    if (l != leaders.end() && *i == *l) {
      // found start of next basic block
      m_basic_blocks.back()->ptx_end = last_real_inst;
      if (find_next_real_instruction(i) !=
          m_instructions.end()) {  // if not bogus trailing label
        m_basic_blocks.push_back(new basic_block_t(
            bb_id++, *find_next_real_instruction(i), NULL, 0, 0));
        last_real_inst = *find_next_real_instruction(i);
      }
      // start search for next leader
      l++;
    }
    pI->assign_bb(m_basic_blocks.back());
    if (!pI->is_label()) last_real_inst = pI;
  }
  m_basic_blocks.back()->ptx_end = last_real_inst;
  m_basic_blocks.push_back(
      /*exit basic block*/ new basic_block_t(bb_id, NULL, NULL, 0, 1));
}

void function_info::print_basic_blocks() {
  printf("Printing basic blocks for function \'%s\':\n", m_name.c_str());
  std::list<ptx_instruction *>::iterator ptx_itr;
  unsigned last_bb = 0;
  for (ptx_itr = m_instructions.begin(); ptx_itr != m_instructions.end();
       ptx_itr++) {
    if ((*ptx_itr)->get_bb()) {
      if ((*ptx_itr)->get_bb()->bb_id != last_bb) {
        printf("\n");
        last_bb = (*ptx_itr)->get_bb()->bb_id;
      }
      printf("bb_%02u\t: ", (*ptx_itr)->get_bb()->bb_id);
      (*ptx_itr)->print_insn();
      printf("\n");
    }
  }
  printf("\nSummary of basic blocks for \'%s\':\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("bb_%02u\t:", (*bb_itr)->bb_id);
    if ((*bb_itr)->ptx_begin)
      printf(" first: %s\t", ((*bb_itr)->ptx_begin)->get_opcode_cstr());
    else
      printf(" first: NULL\t");
    if ((*bb_itr)->ptx_end) {
      printf(" last: %s\t", ((*bb_itr)->ptx_end)->get_opcode_cstr());
    } else
      printf(" last: NULL\t");
    printf("\n");
  }
  printf("\n");
}

void function_info::print_basic_block_links() {
  printf("Printing basic blocks links for function \'%s\':\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("ID: %d\t:", (*bb_itr)->bb_id);
    if (!(*bb_itr)->predecessor_ids.empty()) {
      printf("Predecessors:");
      std::set<int>::iterator p;
      for (p = (*bb_itr)->predecessor_ids.begin();
           p != (*bb_itr)->predecessor_ids.end(); p++) {
        printf(" %d", *p);
      }
      printf("\t");
    }
    if (!(*bb_itr)->successor_ids.empty()) {
      printf("Successors:");
      std::set<int>::iterator s;
      for (s = (*bb_itr)->successor_ids.begin();
           s != (*bb_itr)->successor_ids.end(); s++) {
        printf(" %d", *s);
      }
    }
    printf("\n");
  }
}
operand_info *function_info::find_break_target(
    ptx_instruction *p_break_insn)  // find the target of a break instruction
{
  const basic_block_t *break_bb = p_break_insn->get_bb();
  // go through the dominator tree
  for (const basic_block_t *p_bb = break_bb; p_bb->immediatedominator_id != -1;
       p_bb = m_basic_blocks[p_bb->immediatedominator_id]) {
    // reverse search through instructions in basic block for breakaddr
    // instruction
    unsigned insn_addr = p_bb->ptx_end->get_m_instr_mem_index();
    while (insn_addr >= p_bb->ptx_begin->get_m_instr_mem_index()) {
      ptx_instruction *pI = m_instr_mem[insn_addr];
      insn_addr -= 1;
      if (pI == NULL)
        continue;  // temporary solution for variable size instructions
      if (pI->get_opcode() == BREAKADDR_OP) {
        return &(pI->dst());
      }
    }
  }

  assert(0);

  // lazy fallback: just traverse backwards?
  for (int insn_addr = p_break_insn->get_m_instr_mem_index(); insn_addr >= 0;
       insn_addr--) {
    ptx_instruction *pI = m_instr_mem[insn_addr];
    if (pI->get_opcode() == BREAKADDR_OP) {
      return &(pI->dst());
    }
  }

  return NULL;
}
void function_info::connect_basic_blocks()  // iterate across m_basic_blocks of
                                            // function, connecting basic blocks
                                            // together
{
  std::vector<basic_block_t *>::iterator bb_itr;
  std::vector<basic_block_t *>::iterator bb_target_itr;
  basic_block_t *exit_bb = m_basic_blocks.back();

  // start from first basic block, which we know is the entry point
  bb_itr = m_basic_blocks.begin();
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    ptx_instruction *pI = (*bb_itr)->ptx_end;
    if ((*bb_itr)->is_exit)  // reached last basic block, no successors to link
      continue;
    if (pI->get_opcode() == RETP_OP || pI->get_opcode() == RET_OP ||
        pI->get_opcode() == EXIT_OP) {
      (*bb_itr)->successor_ids.insert(exit_bb->bb_id);
      exit_bb->predecessor_ids.insert((*bb_itr)->bb_id);
      if (pI->has_pred()) {
        printf("GPGPU-Sim PTX: Warning detected predicated return/exit.\n");
        // if predicated, add link to next block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
        if (next_addr < m_instr_mem_size && m_instr_mem[next_addr]) {
          basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
          (*bb_itr)->successor_ids.insert(next_bb->bb_id);
          next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
        }
      }
      continue;
    } else if (pI->get_opcode() == BRA_OP) {
      // find successor and link that basic_block to this one
      operand_info &target = pI->dst();  // get operand, e.g. target name
      unsigned addr = labels[target.name()];
      ptx_instruction *target_pI = m_instr_mem[addr];
      basic_block_t *target_bb = target_pI->get_bb();
      (*bb_itr)->successor_ids.insert(target_bb->bb_id);
      target_bb->predecessor_ids.insert((*bb_itr)->bb_id);
    }

    if (!(pI->get_opcode() == BRA_OP && (!pI->has_pred()))) {
      // if basic block does not end in an unpredicated branch,
      // then next basic block is also successor
      // (this is better than testing for .uni)
      unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
      basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
      (*bb_itr)->successor_ids.insert(next_bb->bb_id);
      next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
    } else
      assert(pI->get_opcode() == BRA_OP);
  }
}
bool function_info::connect_break_targets()  // connecting break instructions
                                             // with proper targets
{
  std::vector<basic_block_t *>::iterator bb_itr;
  std::vector<basic_block_t *>::iterator bb_target_itr;
  bool modified = false;

  // start from first basic block, which we know is the entry point
  bb_itr = m_basic_blocks.begin();
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    basic_block_t *p_bb = *bb_itr;
    ptx_instruction *pI = p_bb->ptx_end;
    if (p_bb->is_exit)  // reached last basic block, no successors to link
      continue;
    if (pI->get_opcode() == BREAK_OP) {
      // backup existing successor_ids for stability check
      std::set<int> orig_successor_ids = p_bb->successor_ids;

      // erase the previous linkage with old successors
      for (std::set<int>::iterator succ_ids = p_bb->successor_ids.begin();
           succ_ids != p_bb->successor_ids.end(); ++succ_ids) {
        basic_block_t *successor_bb = m_basic_blocks[*succ_ids];
        successor_bb->predecessor_ids.erase(p_bb->bb_id);
      }
      p_bb->successor_ids.clear();

      // find successor and link that basic_block to this one
      // successor of a break is set by an preceeding breakaddr instruction
      operand_info *target = find_break_target(pI);
      unsigned addr = labels[target->name()];
      ptx_instruction *target_pI = m_instr_mem[addr];
      basic_block_t *target_bb = target_pI->get_bb();
      p_bb->successor_ids.insert(target_bb->bb_id);
      target_bb->predecessor_ids.insert(p_bb->bb_id);

      if (pI->has_pred()) {
        // predicated break - add link to next basic block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
        basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
        p_bb->successor_ids.insert(next_bb->bb_id);
        next_bb->predecessor_ids.insert(p_bb->bb_id);
      }

      modified = modified || (orig_successor_ids != p_bb->successor_ids);
    }
  }

  return modified;
}
void function_info::do_pdom() {
  create_basic_blocks();
  connect_basic_blocks();
  bool modified = false;
  do {
    find_dominators();
    find_idominators();
    modified = connect_break_targets();
  } while (modified == true);

  if (g_debug_execution >= 50) {
    print_basic_blocks();
    print_basic_block_links();
    print_basic_block_dot();
  }
  if (g_debug_execution >= 2) {
    print_dominators();
  }
  find_postdominators();
  find_ipostdominators();
  if (g_debug_execution >= 50) {
    print_postdominators();
    print_ipostdominators();
  }
  printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n",
         m_name.c_str());
  for (unsigned ii = 0; ii < m_n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];
    pI->pre_decode();
  }
  printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n",
         m_name.c_str());
  fflush(stdout);
  m_assembled = true;
}
void intersect(std::set<int> &A, const std::set<int> &B) {
  // return intersection of A and B in A
  for (std::set<int>::iterator a = A.begin(); a != A.end();) {
    std::set<int>::iterator a_next = a;
    a_next++;
    if (B.find(*a) == B.end()) {
      A.erase(*a);
      a = a_next;
    } else
      a++;
  }
}

bool is_equal(const std::set<int> &A, const std::set<int> &B) {
  if (A.size() != B.size()) return false;
  for (std::set<int>::iterator b = B.begin(); b != B.end(); b++)
    if (A.find(*b) == A.end()) return false;
  return true;
}

void print_set(const std::set<int> &A) {
  std::set<int>::iterator a;
  for (a = A.begin(); a != A.end(); a++) {
    printf("%d ", (*a));
  }
  printf("\n");
}

void function_info::find_dominators() {
  // find dominators using algorithm of Muchnick's Adv. Compiler Design &
  // Implemmntation Fig 7.14
  printf("GPGPU-Sim PTX: Finding dominators for \'%s\'...\n", m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished entry block
  std::vector<basic_block_t *>::iterator bb_itr = m_basic_blocks.begin();
  (*bb_itr)->dominator_ids.insert(
      (*bb_itr)->bb_id);  // the only dominator of the entry block is the entry
  // copy all basic blocks to all dominator lists EXCEPT for the entry block
  for (++bb_itr; bb_itr != m_basic_blocks.end(); bb_itr++) {
    for (unsigned i = 0; i < m_basic_blocks.size(); i++)
      (*bb_itr)->dominator_ids.insert(i);
  }
  bool change = true;
  while (change) {
    change = false;
    for (int h = 1 /*skip entry*/; h < m_basic_blocks.size(); ++h) {
      assert(m_basic_blocks[h]->bb_id == (unsigned)h);
      std::set<int> T;
      for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
      for (std::set<int>::iterator s =
               m_basic_blocks[h]->predecessor_ids.begin();
           s != m_basic_blocks[h]->predecessor_ids.end(); s++)
        intersect(T, m_basic_blocks[*s]->dominator_ids);
      T.insert(h);
      if (!is_equal(T, m_basic_blocks[h]->dominator_ids)) {
        change = true;
        m_basic_blocks[h]->dominator_ids = T;
      }
    }
  }
  // clean the basic block of dominators of it has no predecessors -- except for
  // entry block
  bb_itr = m_basic_blocks.begin();
  for (++bb_itr; bb_itr != m_basic_blocks.end(); bb_itr++) {
    if ((*bb_itr)->predecessor_ids.empty()) (*bb_itr)->dominator_ids.clear();
  }
}

void function_info::find_postdominators() {
  // find postdominators using algorithm of Muchnick's Adv. Compiler Design &
  // Implemmntation Fig 7.14
  printf("GPGPU-Sim PTX: Finding postdominators for \'%s\'...\n",
         m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished exit block
  std::vector<basic_block_t *>::reverse_iterator bb_itr =
      m_basic_blocks.rbegin();
  (*bb_itr)->postdominator_ids.insert(
      (*bb_itr)
          ->bb_id);  // the only postdominator of the exit block is the exit
  for (++bb_itr; bb_itr != m_basic_blocks.rend();
       bb_itr++) {  // copy all basic blocks to all postdominator lists EXCEPT
                    // for the exit block
    for (unsigned i = 0; i < m_basic_blocks.size(); i++)
      (*bb_itr)->postdominator_ids.insert(i);
  }
  bool change = true;
  while (change) {
    change = false;
    for (int h = m_basic_blocks.size() - 2 /*skip exit*/; h >= 0; --h) {
      assert(m_basic_blocks[h]->bb_id == (unsigned)h);
      std::set<int> T;
      for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
      for (std::set<int>::iterator s = m_basic_blocks[h]->successor_ids.begin();
           s != m_basic_blocks[h]->successor_ids.end(); s++)
        intersect(T, m_basic_blocks[*s]->postdominator_ids);
      T.insert(h);
      if (!is_equal(T, m_basic_blocks[h]->postdominator_ids)) {
        change = true;
        m_basic_blocks[h]->postdominator_ids = T;
      }
    }
  }
}

void function_info::find_ipostdominators() {
  // find immediate postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  printf("GPGPU-Sim PTX: Finding immediate postdominators for \'%s\'...\n",
         m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished exit block
  for (unsigned i = 0; i < m_basic_blocks.size();
       i++) {  // initialize Tmp(n) to all pdoms of n except for n
    m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->postdominator_ids;
    assert(m_basic_blocks[i]->bb_id == i);
    m_basic_blocks[i]->Tmp_ids.erase(i);
  }
  for (int n = m_basic_blocks.size() - 2; n >= 0; --n) {
    // point iterator to basic block before the exit
    for (std::set<int>::iterator s = m_basic_blocks[n]->Tmp_ids.begin();
         s != m_basic_blocks[n]->Tmp_ids.end(); s++) {
      int bb_s = *s;
      for (std::set<int>::iterator t = m_basic_blocks[n]->Tmp_ids.begin();
           t != m_basic_blocks[n]->Tmp_ids.end();) {
        std::set<int>::iterator t_next = t;
        t_next++;  // might erase thing pointed to be t, invalidating iterator t
        if (*s == *t) {
          t = t_next;
          continue;
        }
        int bb_t = *t;
        if (m_basic_blocks[bb_s]->postdominator_ids.find(bb_t) !=
            m_basic_blocks[bb_s]->postdominator_ids.end())
          m_basic_blocks[n]->Tmp_ids.erase(bb_t);
        t = t_next;
      }
    }
  }
  unsigned num_ipdoms = 0;
  for (int n = m_basic_blocks.size() - 1; n >= 0; --n) {
    assert(m_basic_blocks[n]->Tmp_ids.size() <= 1);
    // if the above assert fails we have an error in either postdominator
    // computation, the flow graph does not have a unique exit, or some other
    // error
    if (!m_basic_blocks[n]->Tmp_ids.empty()) {
      m_basic_blocks[n]->immediatepostdominator_id =
          *m_basic_blocks[n]->Tmp_ids.begin();
      num_ipdoms++;
    }
  }
  assert(num_ipdoms == m_basic_blocks.size() - 1);
  // the exit node does not have an immediate post dominator, but everyone else
  // should
}

void function_info::find_idominators() {
  // find immediate dominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  printf("GPGPU-Sim PTX: Finding immediate dominators for \'%s\'...\n",
         m_name.c_str());
  fflush(stdout);
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished entry block
  for (unsigned i = 0; i < m_basic_blocks.size();
       i++) {  // initialize Tmp(n) to all doms of n except for n
    m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->dominator_ids;
    assert(m_basic_blocks[i]->bb_id == i);
    m_basic_blocks[i]->Tmp_ids.erase(i);
  }
  for (int n = 0; n < m_basic_blocks.size(); ++n) {
    // point iterator to basic block before the exit
    for (std::set<int>::iterator s = m_basic_blocks[n]->Tmp_ids.begin();
         s != m_basic_blocks[n]->Tmp_ids.end(); s++) {
      int bb_s = *s;
      for (std::set<int>::iterator t = m_basic_blocks[n]->Tmp_ids.begin();
           t != m_basic_blocks[n]->Tmp_ids.end();) {
        std::set<int>::iterator t_next = t;
        t_next++;  // might erase thing pointed to be t, invalidating iterator t
        if (*s == *t) {
          t = t_next;
          continue;
        }
        int bb_t = *t;
        if (m_basic_blocks[bb_s]->dominator_ids.find(bb_t) !=
            m_basic_blocks[bb_s]->dominator_ids.end())
          m_basic_blocks[n]->Tmp_ids.erase(bb_t);
        t = t_next;
      }
    }
  }
  unsigned num_idoms = 0;
  unsigned num_nopred = 0;
  for (int n = 0; n < m_basic_blocks.size(); ++n) {
    // assert( m_basic_blocks[n]->Tmp_ids.size() <= 1 );
    // if the above assert fails we have an error in either dominator
    // computation, the flow graph does not have a unique entry, or some other
    // error
    if (!m_basic_blocks[n]->Tmp_ids.empty()) {
      m_basic_blocks[n]->immediatedominator_id =
          *m_basic_blocks[n]->Tmp_ids.begin();
      num_idoms++;
    } else if (m_basic_blocks[n]->predecessor_ids.empty()) {
      num_nopred += 1;
    }
  }
  assert(num_idoms == m_basic_blocks.size() - num_nopred);
  // the entry node does not have an immediate dominator, but everyone else
  // should
}

void function_info::print_dominators() {
  printf("Printing dominators for function \'%s\':\n", m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    for (std::set<int>::iterator j = m_basic_blocks[i]->dominator_ids.begin();
         j != m_basic_blocks[i]->dominator_ids.end(); j++)
      printf(" %d", *j);
    printf("\n");
  }
}

void function_info::print_postdominators() {
  printf("Printing postdominators for function \'%s\':\n", m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    for (std::set<int>::iterator j =
             m_basic_blocks[i]->postdominator_ids.begin();
         j != m_basic_blocks[i]->postdominator_ids.end(); j++)
      printf(" %d", *j);
    printf("\n");
  }
}

void function_info::print_ipostdominators() {
  printf("Printing immediate postdominators for function \'%s\':\n",
         m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    printf("%d\n", m_basic_blocks[i]->immediatepostdominator_id);
  }
}

void function_info::print_idominators() {
  printf("Printing immediate dominators for function \'%s\':\n",
         m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    printf("%d\n", m_basic_blocks[i]->immediatedominator_id);
  }
}

unsigned function_info::get_num_reconvergence_pairs() {
  if (!num_reconvergence_pairs) {
    if (m_basic_blocks.size() == 0) return 0;
    for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
         i++) {  // last basic block containing exit obviously won't have a pair
      if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
        num_reconvergence_pairs++;
      }
    }
  }
  return num_reconvergence_pairs;
}

void function_info::get_reconvergence_pairs(gpgpu_recon_t *recon_points) {
  unsigned idx = 0;  // array index
  if (m_basic_blocks.size() == 0) return;
  for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
       i++) {  // last basic block containing exit obviously won't have a pair
#ifdef DEBUG_GET_RECONVERG_PAIRS
    printf("i=%d\n", i);
    fflush(stdout);
#endif
    if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\tbranch!\n");
      printf("\tbb_id=%d; ipdom=%d\n", m_basic_blocks[i]->bb_id,
             m_basic_blocks[i]->immediatepostdominator_id);
      printf("\tm_instr_mem index=%d\n",
             m_basic_blocks[i]->ptx_end->get_m_instr_mem_index());
      fflush(stdout);
#endif
      recon_points[idx].source_pc = m_basic_blocks[i]->ptx_end->get_PC();
      recon_points[idx].source_inst = m_basic_blocks[i]->ptx_end;
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\trecon_points[idx].source_pc=%d\n", recon_points[idx].source_pc);
#endif
      if (m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
              ->ptx_begin) {
        recon_points[idx].target_pc =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin->get_PC();
        recon_points[idx].target_inst =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin;
      } else {
        // reconverge after function return
        recon_points[idx].target_pc = -2;
        recon_points[idx].target_inst = NULL;
      }
#ifdef DEBUG_GET_RECONVERG_PAIRS
      m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
          ->ptx_begin->print_insn();
      printf("\trecon_points[idx].target_pc=%d\n", recon_points[idx].target_pc);
      fflush(stdout);
#endif
      idx++;
    }
  }
}

// interface with graphviz (print the graph in DOT language) for plotting
void function_info::print_basic_block_dot() {
  printf("Basic Block in DOT\n");
  printf("digraph %s {\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("\t");
    std::set<int>::iterator s;
    for (s = (*bb_itr)->successor_ids.begin();
         s != (*bb_itr)->successor_ids.end(); s++) {
      unsigned succ_bb = *s;
      printf("%d -> %d; ", (*bb_itr)->bb_id, succ_bb);
    }
    printf("\n");
  }
  printf("}\n");
}

unsigned ptx_kernel_shmem_size(void *kernel_impl) {
  function_info *f = (function_info *)kernel_impl;
  const struct gpgpu_ptx_sim_info *kernel_info = f->get_kernel_info();
  return kernel_info->smem;
}

unsigned ptx_kernel_nregs(void *kernel_impl) {
  function_info *f = (function_info *)kernel_impl;
  const struct gpgpu_ptx_sim_info *kernel_info = f->get_kernel_info();
  return kernel_info->regs;
}

unsigned type_info_key::type_decode(size_t &size, int &basic_type) const {
  int type = scalar_type();
  return type_decode(type, size, basic_type);
}

unsigned type_info_key::type_decode(int type, size_t &size, int &basic_type) {
  switch (type) {
    case S8_TYPE:
      size = 8;
      basic_type = 1;
      return 0;
    case S16_TYPE:
      size = 16;
      basic_type = 1;
      return 1;
    case S32_TYPE:
      size = 32;
      basic_type = 1;
      return 2;
    case S64_TYPE:
      size = 64;
      basic_type = 1;
      return 3;
    case U8_TYPE:
      size = 8;
      basic_type = 0;
      return 4;
    case U16_TYPE:
      size = 16;
      basic_type = 0;
      return 5;
    case U32_TYPE:
      size = 32;
      basic_type = 0;
      return 6;
    case U64_TYPE:
      size = 64;
      basic_type = 0;
      return 7;
    case F16_TYPE:
      size = 16;
      basic_type = -1;
      return 8;
    case F32_TYPE:
      size = 32;
      basic_type = -1;
      return 9;
    case F64_TYPE:
      size = 64;
      basic_type = -1;
      return 10;
    case FF64_TYPE:
      size = 64;
      basic_type = -1;
      return 10;
    case PRED_TYPE:
      size = 1;
      basic_type = 2;
      return 11;
    case B8_TYPE:
      size = 8;
      basic_type = 0;
      return 12;
    case B16_TYPE:
      size = 16;
      basic_type = 0;
      return 13;
    case B32_TYPE:
      size = 32;
      basic_type = 0;
      return 14;
    case B64_TYPE:
      size = 64;
      basic_type = 0;
      return 15;
    case BB64_TYPE:
      size = 64;
      basic_type = 0;
      return 15;
    case BB128_TYPE:
      size = 128;
      basic_type = 0;
      return 16;
    case TEXREF_TYPE:
    case SAMPLERREF_TYPE:
    case SURFREF_TYPE:
      size = 32;
      basic_type = 3;
      return 16;
    default:
      printf("ERROR ** type_decode() does not know about \"%s\"\n",
             decode_token(type));
      assert(0);
      return 0xDEADBEEF;
  }
}

arg_buffer_t copy_arg_to_buffer(ptx_thread_info *thread,
                                operand_info actual_param_op,
                                const symbol *formal_param) {
  if (actual_param_op.is_reg()) {
    ptx_reg_t value = thread->get_reg(actual_param_op.get_symbol());
    return arg_buffer_t(formal_param, actual_param_op, value);
  } else if (actual_param_op.is_param_local()) {
    unsigned size = formal_param->get_size_in_bytes();
    addr_t frame_offset = actual_param_op.get_symbol()->get_address();
    addr_t from_addr = thread->get_local_mem_stack_pointer() + frame_offset;
    char buffer[1024];
    assert(size < 1024);
    thread->m_local_mem->read(from_addr, size, buffer);
    return arg_buffer_t(formal_param, actual_param_op, buffer, size);
  } else {
    printf(
        "GPGPU-Sim PTX: ERROR ** need to add support for this operand type in "
        "call/return\n");
    abort();
  }
}

void copy_args_into_buffer_list(const ptx_instruction *pI,
                                ptx_thread_info *thread,
                                const function_info *target_func,
                                arg_buffer_list_t &arg_values) {
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);
    const symbol *formal_param = target_func->get_arg(arg);
    arg_values.push_back(
        copy_arg_to_buffer(thread, actual_param_op, formal_param));
  }
}

void copy_buffer_to_frame(ptx_thread_info *thread, const arg_buffer_t &a) {
  if (a.is_reg()) {
    ptx_reg_t value = a.get_reg();
    operand_info dst_reg =
        operand_info(a.get_dst(), thread->get_gpu()->gpgpu_ctx);
    thread->set_reg(dst_reg.get_symbol(), value);
  } else {
    const void *buffer = a.get_param_buffer();
    size_t size = a.get_param_buffer_size();
    const symbol *dst = a.get_dst();
    addr_t frame_offset = dst->get_address();
    addr_t to_addr = thread->get_local_mem_stack_pointer() + frame_offset;
    thread->m_local_mem->write(to_addr, size, buffer, NULL, NULL);
  }
}

void copy_buffer_list_into_frame(ptx_thread_info *thread,
                                 arg_buffer_list_t &arg_values) {
  arg_buffer_list_t::iterator a;
  for (a = arg_values.begin(); a != arg_values.end(); a++) {
    copy_buffer_to_frame(thread, *a);
  }
}

static std::list<operand_info> check_operands(
    int opcode, const std::list<int> &scalar_type,
    const std::list<operand_info> &operands, gpgpu_context *ctx) {
  static int g_warn_literal_operands_two_type_inst;
  if ((opcode == CVT_OP) || (opcode == SET_OP) || (opcode == SLCT_OP) ||
      (opcode == TEX_OP) || (opcode == MMA_OP) || (opcode == DP4A_OP) || 
      (opcode == VMIN_OP) || (opcode == VMAX_OP) ) {
    // just make sure these do not have have const operands...
    if (!g_warn_literal_operands_two_type_inst) {
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {
        const operand_info &op = *o;
        if (op.is_literal()) {
          printf(
              "GPGPU-Sim PTX: PTX uses two scalar type intruction with literal "
              "operand.\n");
          g_warn_literal_operands_two_type_inst = 1;
        }
      }
    }
  } else {
    assert(scalar_type.size() < 2);
    if (scalar_type.size() == 1) {
      std::list<operand_info> result;
      int inst_type = scalar_type.front();
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {
        const operand_info &op = *o;
        if (op.is_literal()) {
          if ((op.get_type() == double_op_t) && (inst_type == F32_TYPE)) {
            ptx_reg_t v = op.get_literal_value();
            float u = (float)v.f64;
            operand_info n(u, ctx);
            result.push_back(n);
          } else {
            result.push_back(op);
          }
        } else {
          result.push_back(op);
        }
      }
      return result;
    }
  }
  return operands;
}

ptx_instruction::ptx_instruction(
    int opcode, const symbol *pred, int neg_pred, int pred_mod, symbol *label,
    const std::list<operand_info> &operands, const operand_info &return_var,
    const std::list<int> &options, const std::list<int> &wmma_options,
    const std::list<int> &scalar_type, memory_space_t space_spec,
    const char *file, unsigned line, const char *source,
    const core_config *config, gpgpu_context *ctx)
    : warp_inst_t(config), m_return_var(ctx) {
  gpgpu_ctx = ctx;
  m_uid = ++(ctx->g_num_ptx_inst_uid);
  m_PC = 0;
  m_opcode = opcode;
  m_pred = pred;
  m_neg_pred = neg_pred;
  m_pred_mod = pred_mod;
  m_label = label;
  const std::list<operand_info> checked_operands =
      check_operands(opcode, scalar_type, operands, ctx);
  m_operands.insert(m_operands.begin(), checked_operands.begin(),
                    checked_operands.end());
  m_return_var = return_var;
  m_options = options;
  m_wmma_options = wmma_options;
  m_wide = false;
  m_hi = false;
  m_lo = false;
  m_uni = false;
  m_exit = false;
  m_abs = false;
  m_neg = false;
  m_to_option = false;
  m_cache_option = 0;
  m_rounding_mode = RN_OPTION;
  m_compare_op = -1;
  m_saturation_mode = 0;
  m_geom_spec = 0;
  m_vector_spec = 0;
  m_atomic_spec = 0;
  m_membar_level = 0;
  m_inst_size = 8;  // bytes
  int rr = 0;
  std::list<int>::const_iterator i;
  unsigned n = 1;
  for (i = wmma_options.begin(); i != wmma_options.end(); i++, n++) {
    int last_ptx_inst_option = *i;
    switch (last_ptx_inst_option) {
      case SYNC_OPTION:
      case LOAD_A:
      case LOAD_B:
      case LOAD_C:
      case STORE_D:
      case MMA:
        m_wmma_type = last_ptx_inst_option;
        break;
      case ROW:
      case COL:
        m_wmma_layout[rr++] = last_ptx_inst_option;
        break;
      case M16N16K16:
      case M32N8K16:
      case M8N32K16:
        break;
      default:
        assert(0);
        break;
    }
  }
  rr = 0;
  n = 1;
  for (i = options.begin(); i != options.end(); i++, n++) {
    int last_ptx_inst_option = *i;
    switch (last_ptx_inst_option) {
      case SYNC_OPTION:
      case ARRIVE_OPTION:
      case RED_OPTION:
        m_barrier_op = last_ptx_inst_option;
        break;
      case EQU_OPTION:
      case NEU_OPTION:
      case LTU_OPTION:
      case LEU_OPTION:
      case GTU_OPTION:
      case GEU_OPTION:
      case EQ_OPTION:
      case NE_OPTION:
      case LT_OPTION:
      case LE_OPTION:
      case GT_OPTION:
      case GE_OPTION:
      case LS_OPTION:
      case HS_OPTION:
        m_compare_op = last_ptx_inst_option;
        break;
      case NUM_OPTION:
      case NAN_OPTION:
        m_compare_op = last_ptx_inst_option;
        // assert(0); // finish this
        break;
      case SAT_OPTION:
        m_saturation_mode = 1;
        break;
      case RNI_OPTION:
      case RZI_OPTION:
      case RMI_OPTION:
      case RPI_OPTION:
      case RN_OPTION:
      case RZ_OPTION:
      case RM_OPTION:
      case RP_OPTION:
        m_rounding_mode = last_ptx_inst_option;
        break;
      case HI_OPTION:
        m_compare_op = last_ptx_inst_option;
        m_hi = true;
        assert(!m_lo);
        assert(!m_wide);
        break;
      case LO_OPTION:
        m_compare_op = last_ptx_inst_option;
        m_lo = true;
        assert(!m_hi);
        assert(!m_wide);
        break;
      case WIDE_OPTION:
        m_wide = true;
        assert(!m_lo);
        assert(!m_hi);
        break;
      case UNI_OPTION:
        m_uni = true;  // don't care... < now we DO care when constructing
                       // flowgraph>
        break;
      case GEOM_MODIFIER_1D:
      case GEOM_MODIFIER_2D:
      case GEOM_MODIFIER_3D:
        m_geom_spec = last_ptx_inst_option;
        break;
      case V2_TYPE:
      case V3_TYPE:
      case V4_TYPE:
        m_vector_spec = last_ptx_inst_option;
        break;
      case ATOMIC_AND:
      case ATOMIC_OR:
      case ATOMIC_XOR:
      case ATOMIC_CAS:
      case ATOMIC_EXCH:
      case ATOMIC_ADD:
      case ATOMIC_INC:
      case ATOMIC_DEC:
      case ATOMIC_MIN:
      case ATOMIC_MAX:
        m_atomic_spec = last_ptx_inst_option;
        break;
      case APPROX_OPTION:
        break;
      case FULL_OPTION:
        break;
      case ANY_OPTION:
        m_vote_mode = vote_any;
        break;
      case ALL_OPTION:
        m_vote_mode = vote_all;
        break;
      case BALLOT_OPTION:
        m_vote_mode = vote_ballot;
        break;
      case GLOBAL_OPTION:
        m_membar_level = GLOBAL_OPTION;
        break;
      case CTA_OPTION:
        m_membar_level = CTA_OPTION;
        break;
      case SYS_OPTION:
        m_membar_level = SYS_OPTION;
        break;
      case FTZ_OPTION:
        break;
      case EXIT_OPTION:
        m_exit = true;
        break;
      case ABS_OPTION:
        m_abs = true;
        break;
      case NEG_OPTION:
        m_neg = true;
        break;
      case TO_OPTION:
        m_to_option = true;
        break;
      case CA_OPTION:
      case CG_OPTION:
      case CS_OPTION:
      case LU_OPTION:
      case CV_OPTION:
        m_cache_option = last_ptx_inst_option;
        break;
      case HALF_OPTION:
        m_inst_size = 4;  // bytes
        break;
      case EXTP_OPTION:
        break;
      case NC_OPTION:
        m_cache_option = last_ptx_inst_option;
        break;
      case UP_OPTION:
      case DOWN_OPTION:
      case BFLY_OPTION:
      case IDX_OPTION:
        m_shfl_op = last_ptx_inst_option;
        break;
      case PRMT_F4E_MODE:
      case PRMT_B4E_MODE:
      case PRMT_RC8_MODE:
      case PRMT_ECL_MODE:
      case PRMT_ECR_MODE:
      case PRMT_RC16_MODE:
        m_prmt_op = last_ptx_inst_option;
        break;
      default:
        assert(0);
        break;
    }
  }
  m_scalar_type = scalar_type;
  m_space_spec = space_spec;
  if ((opcode == ST_OP || opcode == LD_OP || opcode == LDU_OP) &&
      (space_spec == undefined_space)) {
    m_space_spec = generic_space;
  }
  for (std::vector<operand_info>::const_iterator i = m_operands.begin();
       i != m_operands.end(); ++i) {
    const operand_info &op = *i;
    if (op.get_addr_space() != undefined_space)
      m_space_spec =
          op.get_addr_space();  // TODO: can have more than one memory space for
                                // ptxplus (g8x) inst
  }
  if (opcode == TEX_OP) m_space_spec = tex_space;

  m_source_file = file ? file : "<unknown>";
  m_source_line = line;
  m_source = source;
  // Trim tabs
  m_source.erase(std::remove(m_source.begin(), m_source.end(), '\t'),
                 m_source.end());

  if (opcode == CALL_OP) {
    const operand_info &target = func_addr();
    assert(target.is_function_address());
    const symbol *func_addr = target.get_symbol();
    const function_info *target_func = func_addr->get_pc();
    std::string fname = target_func->get_name();

    if (fname == "vprintf") {
      m_is_printf = true;
    }
    if (fname == "cudaStreamCreateWithFlags") m_is_cdp = 1;
    if (fname == "cudaGetParameterBufferV2") m_is_cdp = 2;
    if (fname == "cudaLaunchDeviceV2") m_is_cdp = 4;
  }
}

void ptx_instruction::print_insn() const {
  print_insn(stdout);
  fflush(stdout);
}

void ptx_instruction::print_insn(FILE *fp) const {
  fprintf(fp, "%s", to_string().c_str());
}

std::string ptx_instruction::to_string() const {
  char buf[STR_SIZE];
  unsigned used_bytes = 0;
  if (!is_label()) {
    used_bytes +=
        snprintf(buf + used_bytes, STR_SIZE - used_bytes, " PC=0x%03x ", m_PC);
  } else {
    used_bytes +=
        snprintf(buf + used_bytes, STR_SIZE - used_bytes, "                ");
  }
  used_bytes +=
      snprintf(buf + used_bytes, STR_SIZE - used_bytes, "(%s:%d) %s",
               m_source_file.c_str(), m_source_line, m_source.c_str());
  return std::string(buf);
}
operand_info ptx_instruction::get_pred() const {
  return operand_info(m_pred, gpgpu_ctx);
}

function_info::function_info(int entry_point, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_uid = (gpgpu_ctx->function_info_sm_next_uid)++;
  m_entry_point = (entry_point == 1) ? true : false;
  m_extern = (entry_point == 2) ? true : false;
  num_reconvergence_pairs = 0;
  m_symtab = NULL;
  m_assembled = false;
  m_return_var_sym = NULL;
  m_kernel_info.cmem = 0;
  m_kernel_info.lmem = 0;
  m_kernel_info.regs = 0;
  m_kernel_info.smem = 0;
  m_local_mem_framesize = 0;
  m_args_aligned_size = -1;
  pdom_done = false;  // initialize it to false
}

unsigned function_info::print_insn(unsigned pc, FILE *fp) const {
  unsigned inst_size = 1;  // return offset to next instruction or 1 if unknown
  unsigned index = pc - m_start_PC;
  char command[1024];
  char buffer[1024];
  memset(command, 0, 1024);
  memset(buffer, 0, 1024);
  snprintf(command, 1024, "c++filt -p %s", m_name.c_str());
  FILE *p = popen(command, "r");
  buffer[0] = 0;
  assert(fgets(buffer, 1023, p) != NULL);
  // Remove trailing "\n" in buffer
  char *c;
  if ((c = strchr(buffer, '\n')) != NULL) *c = '\0';
  fprintf(fp, "%s", buffer);
  if (index >= m_instr_mem_size) {
    fprintf(fp, "<past last instruction (max pc=%u)>",
            m_start_PC + m_instr_mem_size - 1);
  } else {
    if (m_instr_mem[index] != NULL) {
      m_instr_mem[index]->print_insn(fp);
      inst_size = m_instr_mem[index]->isize;
    } else
      fprintf(fp, "<no instruction at pc = %u>", pc);
  }
  pclose(p);
  return inst_size;
}

std::string function_info::get_insn_str(unsigned pc) const {
  unsigned index = pc - m_start_PC;
  if (index >= m_instr_mem_size) {
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';
    snprintf(buff, STR_SIZE, "<past last instruction (max pc=%u)>",
             m_start_PC + m_instr_mem_size - 1);
    return std::string(buff);
  } else {
    if (m_instr_mem[index] != NULL) {
      return m_instr_mem[index]->to_string();
    } else {
      char buff[STR_SIZE];
      buff[STR_SIZE - 1] = '\0';
      snprintf(buff, STR_SIZE, "<no instruction at pc = %u>", pc);
      return std::string(buff);
    }
  }
}

void gpgpu_ptx_assemble(std::string kname, void *kinfo) {
  function_info *func_info = (function_info *)kinfo;
  if ((function_info *)kinfo == NULL) {
    printf("GPGPU-Sim PTX: Warning - missing function definition \'%s\'\n",
           kname.c_str());
    return;
  }
  if (func_info->is_extern()) {
    printf(
        "GPGPU-Sim PTX: skipping assembly for extern declared function "
        "\'%s\'\n",
        func_info->get_name().c_str());
    return;
  }
  func_info->ptx_assemble();
}
