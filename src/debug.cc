#include "debug.h"
#include "gpgpu-sim/shader.h"
#include "gpgpu-sim/gpu-sim.h"
#include "cuda-sim/ptx_sim.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"

#include <map>
#include <stdio.h>
#include <string.h>

class watchpoint_event {
public:
   watchpoint_event()
   {
      m_thread=NULL;
      m_inst=NULL;
   }
   watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) 
   {
      m_thread=thd;
      m_inst = pI;
   }
   const ptx_thread_info *thread() const { return m_thread; }
   const ptx_instruction *inst() const { return m_inst; }
private:
   const ptx_thread_info *m_thread;
   const ptx_instruction *m_inst;
};

std::map<unsigned,watchpoint_event> g_watchpoint_hits;

void hit_watchpoint( unsigned watchpoint_num, ptx_thread_info *thd, const ptx_instruction *pI )
{
   g_watchpoint_hits[watchpoint_num]=watchpoint_event(thd,pI);
}

/// interactive debugger 

void gpgpu_debug()
{
   bool done=true;

   static bool single_step=true;
   static unsigned next_brkpt=1;
   static std::map<unsigned,brk_pt> breakpoints;

   /// if single stepping, go to interactive debugger

   if( single_step ) 
      done=false;

   /// check if we've reached a breakpoint

   for( std::map<unsigned,brk_pt>::iterator i=breakpoints.begin(); i!=breakpoints.end(); i++) {
      unsigned num=i->first;
      brk_pt &b=i->second;
      if( b.is_watchpoint() ) {
         unsigned addr = b.get_addr();
         unsigned new_value = read_location(addr);
         if( new_value != b.get_value() ) {
            printf( "GPGPU-Sim DBG: watch point %u triggered (old value=%x, new value=%x)\n",
                     num,b.get_value(),new_value );
            std::map<unsigned,watchpoint_event>::iterator w=g_watchpoint_hits.find(num);
            if( w==g_watchpoint_hits.end() ) 
               printf( "GPGPU-Sim DBG: memory transfer modified value\n");
            else {
               watchpoint_event wa = w->second;
               const ptx_thread_info *thd = wa.thread();
               const ptx_instruction *pI = wa.inst();
               printf( "GPGPU-Sim DBG: modified by thread uid=%u, sid=%u, hwtid=%u\n",
                       thd->get_uid(),thd->get_hw_sid(), thd->get_hw_tid() );
               printf( "GPGPU-Sim DBG: ");
               pI->print_insn(stdout);
               printf( "\n" );
               g_watchpoint_hits.erase(w);
            }
            b.set_value(new_value);
            done = false; 
         }
      } else {
         for( unsigned sid=0; sid < gpu_n_shader; sid++ ) { 
            inst_t *fvi = first_valid_thread(sc[sid]->pipeline_reg[IF_ID]);
            if( !fvi ) continue;
            if( thread_at_brkpt(fvi->ptx_thd_info, b) ) {
               done = false;
               printf("GPGPU-Sim DBG: reached breakpoint %u at %s (sm=%u, hwtid=%u)\n", 
                      num, b.location().c_str(), sid, fvi->hw_thread_id );
            }
         }
      }
   }

   if( done ) 
      assert( g_watchpoint_hits.empty() );

   /// enter interactive debugger loop

   while (!done) {
      printf("(gpgpu-sim dbg) ");
      fflush(stdout);
      
      char line[1024];
      fgets(line,1024,stdin);

      char *tok = strtok(line," \t\n");
      if( !strcmp(tok,"dp") ) {
         int shader_num = 0;
         tok = strtok(NULL," \t\n");
         sscanf(tok,"%d",&shader_num);
         dump_pipeline_impl((0x40|0x4|0x1),shader_num,0);
         printf("\n");
         fflush(stdout);
      } else if( !strcmp(tok,"q") || !strcmp(tok,"quit") ) {
         printf("\nreally quit GPGPU-Sim (y/n)?\n");
         fgets(line,1024,stdin);
         tok = strtok(NULL," \t\n");
         if( !strcmp(tok,"y") ) {
            exit(0);
         } else {
            printf("not quiting.\n");
         }
      } else if( !strcmp(tok,"b") ) {
         tok = strtok(NULL," \t\n");
         char brkpt[1024];
         sscanf(tok,"%s",brkpt);
         tok = strtok(NULL," \t\n");
         unsigned uid;
         sscanf(tok,"%u",&uid);
         breakpoints[next_brkpt++] = brk_pt(brkpt,uid);
      } else if( !strcmp(tok,"d") ) {
         tok = strtok(NULL," \t\n");
         unsigned uid;
         sscanf(tok,"%u",&uid);
         breakpoints.erase(uid);
      } else if( !strcmp(tok,"s") ) {
         done = true;
      } else if( !strcmp(tok,"c") ) {
         single_step=false;
         done = true;
      } else if( !strcmp(tok,"w") ) {
         tok = strtok(NULL," \t\n");
         unsigned addr;
         sscanf(tok,"%x",&addr);
         unsigned value = read_location(addr);
         g_global_mem->set_watch(addr,next_brkpt); 
         breakpoints[next_brkpt++] = brk_pt(addr,value);
      } else if( !strcmp(tok,"h") ) {
         printf("commands:\n");
         printf("  q                           - quit GPGPU-Sim\n");
         printf("  b <file>:<line> <thead uid> - set breakpoint\n");
         printf("  w <global address>          - set watchpoint\n");
         printf("  del <n>                     - delete breakpoint\n");
         printf("  s                           - single step one shader cycle (all cores)\n");
         printf("  c                           - continue simulation without single stepping\n");
         printf("  dp <n>                      - display pipeline contents on SM <n>\n");
         printf("  h                           - print this message\n");
      } else {
         printf("\ncommand not understood.\n");
      }
      fflush(stdout);
   }
}

bool thread_at_brkpt( void *thd, const struct brk_pt &b )
{
   ptx_thread_info *thread = (ptx_thread_info *) thd;
   return b.is_equal(thread->get_location(),thread->get_uid());
}

unsigned read_location( addr_t addr )
{
   unsigned result=0;
   g_global_mem->read(addr,4,&result);
   return result;
}
