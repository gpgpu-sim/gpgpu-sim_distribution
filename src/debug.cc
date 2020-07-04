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

#include "debug.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/shader.h"

#include <stdio.h>
#include <string.h>
#include <map>

void gpgpu_sim::hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                               const ptx_instruction *pI) {
  g_watchpoint_hits[watchpoint_num] = watchpoint_event(thd, pI);
}

/// interactive debugger

void gpgpu_sim::gpgpu_debug() {
  bool done = true;

  static bool single_step = true;
  static unsigned next_brkpt = 1;
  static std::map<unsigned, brk_pt> breakpoints;

  /// if single stepping, go to interactive debugger

  if (single_step) done = false;

  /// check if we've reached a breakpoint
  const ptx_thread_info *brk_thd = NULL;
  const ptx_instruction *brk_inst = NULL;

  for (std::map<unsigned, brk_pt>::iterator i = breakpoints.begin();
       i != breakpoints.end(); i++) {
    unsigned num = i->first;
    brk_pt &b = i->second;
    if (b.is_watchpoint()) {
      unsigned addr = b.get_addr();
      unsigned new_value;
      m_global_mem->read(addr, 4, &new_value);
      if (new_value != b.get_value() ||
          g_watchpoint_hits.find(num) != g_watchpoint_hits.end()) {
        printf(
            "GPGPU-Sim PTX DBG: watch point %u triggered (old value=%x, new "
            "value=%x)\n",
            num, b.get_value(), new_value);
        std::map<unsigned, watchpoint_event>::iterator w =
            g_watchpoint_hits.find(num);
        if (w == g_watchpoint_hits.end())
          printf("GPGPU-Sim PTX DBG: memory transfer modified value\n");
        else {
          watchpoint_event wa = w->second;
          brk_thd = wa.thread();
          brk_inst = wa.inst();
          printf(
              "GPGPU-Sim PTX DBG: modified by thread uid=%u, sid=%u, "
              "hwtid=%u\n",
              brk_thd->get_uid(), brk_thd->get_hw_sid(), brk_thd->get_hw_tid());
          printf("GPGPU-Sim PTX DBG: ");
          brk_inst->print_insn(stdout);
          printf("\n");
          g_watchpoint_hits.erase(w);
        }
        b.set_value(new_value);
        done = false;
      }
    } else {
      /*
     for( unsigned sid=0; sid < m_n_shader; sid++ ) {
        unsigned hw_thread_id = -1;
        abort();
        ptx_thread_info *thread =
     m_sc[sid]->get_functional_thread(hw_thread_id); if( thread_at_brkpt(thread,
     b) ) { done = false; printf("GPGPU-Sim PTX DBG: reached breakpoint %u at %s
     (sm=%u, hwtid=%u)\n", num, b.location().c_str(), sid, hw_thread_id );
           brk_thd = thread;
           brk_inst = brk_thd->get_inst();
           printf( "GPGPU-Sim PTX DBG: reached by thread uid=%u, sid=%u,
     hwtid=%u\n", brk_thd->get_uid(),brk_thd->get_hw_sid(),
     brk_thd->get_hw_tid() ); printf( "GPGPU-Sim PTX DBG: ");
           brk_inst->print_insn(stdout);
           printf( "\n" );
        }
     }
     */
    }
  }

  if (done) assert(g_watchpoint_hits.empty());

  /// enter interactive debugger loop

  while (!done) {
    printf("(ptx debugger) ");
    fflush(stdout);

    char line[1024];
    fgets(line, 1024, stdin);

    char *tok = strtok(line, " \t\n");
    if (!strcmp(tok, "dp")) {
      int shader_num = 0;
      tok = strtok(NULL, " \t\n");
      sscanf(tok, "%d", &shader_num);
      dump_pipeline((0x40 | 0x4 | 0x1), shader_num, 0);
      printf("\n");
      fflush(stdout);
    } else if (!strcmp(tok, "q") || !strcmp(tok, "quit")) {
      printf("\nreally quit GPGPU-Sim (y/n)?\n");
      fgets(line, 1024, stdin);
      tok = strtok(line, " \t\n");
      if (!strcmp(tok, "y")) {
        exit(0);
      } else {
        printf("not quiting.\n");
      }
    } else if (!strcmp(tok, "b")) {
      tok = strtok(NULL, " \t\n");
      char brkpt[1024];
      sscanf(tok, "%s", brkpt);
      tok = strtok(NULL, " \t\n");
      unsigned uid;
      sscanf(tok, "%u", &uid);
      breakpoints[next_brkpt++] = brk_pt(brkpt, uid);
    } else if (!strcmp(tok, "d")) {
      tok = strtok(NULL, " \t\n");
      unsigned uid;
      sscanf(tok, "%u", &uid);
      breakpoints.erase(uid);
    } else if (!strcmp(tok, "s")) {
      done = true;
    } else if (!strcmp(tok, "c")) {
      single_step = false;
      done = true;
    } else if (!strcmp(tok, "w")) {
      tok = strtok(NULL, " \t\n");
      unsigned addr;
      sscanf(tok, "%x", &addr);
      unsigned value;
      m_global_mem->read(addr, 4, &value);
      m_global_mem->set_watch(addr, next_brkpt);
      breakpoints[next_brkpt++] = brk_pt(addr, value);
    } else if (!strcmp(tok, "l")) {
      if (brk_thd == NULL) {
        printf("no thread selected\n");
      } else {
        addr_t pc = brk_thd->get_pc();
        addr_t start_pc = (pc < 5) ? 0 : (pc - 5);
        for (addr_t p = start_pc; p <= pc + 5; p++) {
          const ptx_instruction *i = brk_thd->get_inst(p);
          if (i) {
            if (p != pc)
              printf("    ");
            else
              printf("==> ");
            i->print_insn(stdout);
            printf("\n");
          }
        }
      }
    } else if (!strcmp(tok, "h")) {
      printf("commands:\n");
      printf("  q                           - quit GPGPU-Sim\n");
      printf("  b <file>:<line> <thead uid> - set breakpoint\n");
      printf("  w <global address>          - set watchpoint\n");
      printf("  del <n>                     - delete breakpoint\n");
      printf(
          "  s                           - single step one shader cycle (all "
          "cores)\n");
      printf(
          "  c                           - continue simulation without single "
          "stepping\n");
      printf(
          "  l                           - list PTX around current "
          "breakpoint\n");
      printf(
          "  dp <n>                      - display pipeline contents on SM "
          "<n>\n");
      printf("  h                           - print this message\n");
    } else {
      printf("\ncommand not understood.\n");
    }
    fflush(stdout);
  }
}

bool thread_at_brkpt(ptx_thread_info *thread, const class brk_pt &b) {
  return b.is_equal(thread->get_location(), thread->get_uid());
}
