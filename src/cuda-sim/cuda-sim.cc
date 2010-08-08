/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Ali Bakhoda, Wilson W. L. Fung, 
 * George L. Yuan, Henry Wong, Dan O'Connor, Zev Weiss and the 
 * University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include "cuda-sim.h"

#include "instructions.h"
#include "ptx_ir.h"
#include "ptx_sim.h"
#include <stdio.h>

#include "opcodes.h"
#include "../intersim/statwraper.h"
#include "dram_callback.h"
#include <set>
#include <map>
#include "../abstract_hardware_model.h"
#include "memory.h"
#include "ptx-stats.h"
#include "ptx_loader.h"

extern bool g_interactive_debugger_enabled;

int gpgpu_ptx_instruction_classification=0;
void ** g_inst_classification_stat = NULL;
void ** g_inst_op_classification_stat= NULL;
int g_ptx_kernel_count = -1; // used for classification stat collection purposes 

int g_debug_execution = 0;
int g_debug_thread_uid = 0;
addr_t g_debug_pc = 0xBEEF1518;

const char *g_filename;
bool g_debug_ir_generation = false;
unsigned g_ptx_sim_num_insn = 0;

std::map<const struct textureReference*,const struct cudaArray*> TextureToArrayMap; // texture bindings
std::map<const struct textureReference*, const struct textureInfo*> TextureToInfoMap;
std::map<std::string, const struct textureReference*> NameToTextureMap;
unsigned int g_texcache_linesize;
int gpgpu_option_spread_blocks_across_cores = 0;
unsigned gpgpu_param_num_shaders = 0;

void gpgpu_ptx_sim_bindNameToTexture(const char* name, const struct textureReference* texref)
{
   std::string texname(name);
   NameToTextureMap[texname] = texref;
}

const struct textureReference* gpgpu_ptx_sim_accessTextureofName(const char* name) {
   std::string texname(name);
   return NameToTextureMap[texname];
}

const char* gpgpu_ptx_sim_findNamefromTexture(const struct textureReference* texref)
{
   std::map<std::string, const struct textureReference*>::iterator itr = NameToTextureMap.begin();
   while (itr != NameToTextureMap.end()) {
      if ((*itr).second == texref) {
         const char *p = ((*itr).first).c_str();
         return p;
      }
      itr++;
   }
   return NULL;
}

unsigned int intLOGB2( unsigned int v ) {
   unsigned int shift;
   unsigned int r;

   r = 0;

   shift = (( v & 0xFFFF0000) != 0 ) << 4; v >>= shift; r |= shift;
   shift = (( v & 0xFF00    ) != 0 ) << 3; v >>= shift; r |= shift;
   shift = (( v & 0xF0      ) != 0 ) << 2; v >>= shift; r |= shift;
   shift = (( v & 0xC       ) != 0 ) << 1; v >>= shift; r |= shift;
   shift = (( v & 0x2       ) != 0 ) << 0; v >>= shift; r |= shift;

   return r;
}

void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference* texref, const struct cudaArray* array)
{
   TextureToArrayMap[texref] = array;
   unsigned int texel_size_bits = array->desc.w + array->desc.x + array->desc.y + array->desc.z;
   unsigned int texel_size = texel_size_bits/8;
   unsigned int Tx, Ty;
   int r;

   printf("GPGPU-Sim PTX:   texel size = %d\n", texel_size);
   printf("GPGPU-Sim PTX:   texture cache linesize = %d\n", g_texcache_linesize);
   //first determine base Tx size for given linesize
   switch (g_texcache_linesize) {
   case 16:
      Tx = 4;
      break;
   case 32:
      Tx = 8;
      break;
   case 64:
      Tx = 8;
      break;
   case 128:
      Tx = 16;
      break;
   case 256:
      Tx = 16;
      break;
   default:
      printf("GPGPU-Sim PTX:   Line size of %d bytes currently not supported.\n", g_texcache_linesize);
      assert(0);
      break;
   }
   r = texel_size >> 2;
   //modify base Tx size to take into account size of each texel in bytes
   while (r != 0) {
      Tx = Tx >> 1;
      r = r >> 2;
   }
   //by now, got the correct Tx size, calculate correct Ty size
   Ty = g_texcache_linesize/(Tx*texel_size);

   printf("GPGPU-Sim PTX:   Tx = %d; Ty = %d, Tx_numbits = %d, Ty_numbits = %d\n", Tx, Ty, intLOGB2(Tx), intLOGB2(Ty));
   printf("GPGPU-Sim PTX:   Texel size = %d bytes; texel_size_numbits = %d\n", texel_size, intLOGB2(texel_size));
   printf("GPGPU-Sim PTX:   Binding texture to array starting at devPtr32 = 0x%x\n", array->devPtr32);
   printf("GPGPU-Sim PTX:   Texel size = %d bytes\n", texel_size);
   struct textureInfo* texInfo = (struct textureInfo*) malloc(sizeof(struct textureInfo)); 
   texInfo->Tx = Tx;
   texInfo->Ty = Ty;
   texInfo->Tx_numbits = intLOGB2(Tx);
   texInfo->Ty_numbits = intLOGB2(Ty);
   texInfo->texel_size = texel_size;
   texInfo->texel_size_numbits = intLOGB2(texel_size);
   TextureToInfoMap[texref] = texInfo;
}

const struct cudaArray* gpgpu_ptx_sim_accessArrayofTexture(struct textureReference* texref) {
   return TextureToArrayMap[texref];
}

int gpgpu_ptx_sim_sizeofTexture(const char* name)
{
   std::string texname(name);
   const struct textureReference* texref = NameToTextureMap[texname];
   const struct cudaArray* array = TextureToArrayMap[texref];
   return array->size;
}

unsigned g_assemble_code_next_pc=1; 
std::map<unsigned,function_info*> g_pc_to_finfo;
std::vector<ptx_instruction*> function_info::s_g_pc_to_insn;

void function_info::ptx_assemble()
{
   if( m_assembled ) {
      return;
   }

   // get the instructions into instruction memory...
   unsigned num_inst = m_instructions.size();
   m_instr_mem = new ptx_instruction*[ num_inst ];
   m_instr_mem_size = num_inst;

   printf("GPGPU-Sim PTX: instruction assembly for function \'%s\'... ", m_name.c_str() );
   fflush(stdout);
   std::list<ptx_instruction*>::iterator i;
   addr_t n=0; // offset in m_instr_mem
   addr_t PC = g_assemble_code_next_pc; // globally unique address (across functions)
   m_start_PC = PC;
   s_g_pc_to_insn.reserve(s_g_pc_to_insn.size() + m_instructions.size());
   for ( i=m_instructions.begin(); i != m_instructions.end(); i++ ) {
      ptx_instruction *pI = *i;
      if ( pI->is_label() ) {
         const symbol *l = pI->get_label();
         labels[l->name()] = n;
      } else {
         g_pc_to_finfo[PC] = this;
         m_instr_mem[n] = pI;
         s_g_pc_to_insn.push_back(pI);
         assert(pI == s_g_pc_to_insn[PC - 1]);
         pI->set_m_instr_mem_index(n);
         pI->set_PC(PC);
         n++;
         PC++;
      }
   }
   g_assemble_code_next_pc=PC;
   for ( unsigned ii=0; ii < n; ii++ ) { // handle branch instructions
      ptx_instruction *pI = m_instr_mem[ii];
      if ( pI->get_opcode() == BRA_OP ) {
         operand_info &target = pI->dst(); //get operand, e.g. target name
         if ( labels.find(target.name()) == labels.end() ) {
            printf("GPGPU-Sim PTX: Loader error (%s:%u): Branch label \"%s\" does not appear in assembly code.",
                   pI->source_file(),pI->source_line(), target.name().c_str() );
            abort();
         }
         unsigned index = labels[ target.name() ]; //determine address from name
         unsigned PC = m_instr_mem[index]->get_PC();
         m_symtab->set_label_address( target.get_symbol(), PC );
         target.set_type(label_t);
      }
   }
   printf("  done.\n");
   fflush(stdout);

   create_basic_blocks();
   connect_basic_blocks();
   if ( g_debug_execution>=50 ) {
      print_basic_blocks();
      print_basic_block_links();
      print_basic_block_dot();
   }
   find_postdominators();
   find_ipostdominators();
   if ( g_debug_execution>=50 ) {
      print_postdominators();
      print_ipostdominators();
   }
   m_assembled = true;
}



void gpgpu_ptx_sim_init_memory()
{
   static bool initialized = false;
   if ( !initialized ) {
      g_global_mem = new memory_space_impl<8192>("global",64*1024);
      g_param_mem = new memory_space_impl<8192>("param",64*1024);
      g_tex_mem = new memory_space_impl<8192>("tex",64*1024);
      g_surf_mem = new memory_space_impl<8192>("surf",64*1024);
      initialized = true;
   }
}

addr_t shared_to_generic( unsigned smid, addr_t addr )
{
   assert( addr < SHARED_MEM_SIZE_MAX );
   return SHARED_GENERIC_START + smid*SHARED_MEM_SIZE_MAX + addr;
}

addr_t global_to_generic( addr_t addr )
{
   return addr;
}

bool isspace_shared( unsigned smid, addr_t addr )
{
   addr_t start = SHARED_GENERIC_START + smid*SHARED_MEM_SIZE_MAX;
   addr_t end = SHARED_GENERIC_START + (smid+1)*SHARED_MEM_SIZE_MAX;
   if( (addr >= end) || (addr < start) ) 
      return false;
   return true;
}

bool isspace_global( addr_t addr )
{
   return (addr > GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT);
}

memory_space_t whichspace( addr_t addr )
{
   if( (addr > GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT) ) {
      return global_space;
   } else if( addr > SHARED_GENERIC_START ) {
      return shared_space;
   } else {
      return local_space;
   }
}

addr_t generic_to_shared( unsigned smid, addr_t addr )
{
   assert(isspace_shared(smid,addr));
   return addr - (SHARED_GENERIC_START + smid*SHARED_MEM_SIZE_MAX);
}

addr_t local_to_generic( unsigned smid, unsigned hwtid, addr_t addr )
{
   assert(addr < LOCAL_MEM_SIZE_MAX); 
   return LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * hwtid) + addr;
}

bool isspace_local( unsigned smid, unsigned hwtid, addr_t addr )
{
   addr_t start = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * hwtid);
   addr_t end   = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * (hwtid+1));
   if( (addr >= end) || (addr < start) ) 
      return false;
   return true;
}

addr_t generic_to_local( unsigned smid, unsigned hwtid, addr_t addr )
{
   assert(isspace_local(smid,hwtid,addr));
   return addr - (LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) + (LOCAL_MEM_SIZE_MAX * hwtid));
}

addr_t generic_to_global( addr_t addr )
{
   return addr;
}


unsigned long long g_dev_malloc=GLOBAL_HEAP_START; 

void* gpgpu_ptx_sim_malloc( size_t size )
{
   unsigned long long result = g_dev_malloc;
   printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, g_dev_malloc );
   fflush(stdout);
   g_dev_malloc += size;
   if (size%64) g_dev_malloc += (64 - size%64); //align to 64 byte boundaries
   return(void*) result;
}

void* gpgpu_ptx_sim_mallocarray( size_t size )
{
   unsigned long long result = g_dev_malloc;
   printf("GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address 0x%Lx\n", size, g_dev_malloc );
   fflush(stdout);
   g_dev_malloc += size;
   if (size%64) g_dev_malloc += (64 - size%64); //align to 64 byte boundaries
   return(void*) result;
}


void gpgpu_ptx_sim_memcpy_to_gpu( size_t dst_start_addr, const void *src, size_t count )
{
   printf("GPGPU-Sim PTX: copying %zu bytes from CPU[0x%Lx] to GPU[0x%Lx] ... ", count, (unsigned long long) src, (unsigned long long) dst_start_addr );
   fflush(stdout);
   char *src_data = (char*)src;
   for (unsigned n=0; n < count; n ++ ) 
      g_global_mem->write(dst_start_addr+n,1, src_data+n,NULL,NULL);
   printf( " done.\n");
   fflush(stdout);
}

void gpgpu_ptx_sim_memcpy_from_gpu( void *dst, size_t src_start_addr, size_t count )
{
   printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to CPU[0x%Lx] ...", count, (unsigned long long) src_start_addr, (unsigned long long) dst );
   fflush(stdout);
   unsigned char *dst_data = (unsigned char*)dst;
   for (unsigned n=0; n < count; n ++ ) 
      g_global_mem->read(src_start_addr+n,1,dst_data+n);
   printf( " done.\n");
   fflush(stdout);
}

void gpgpu_ptx_sim_memcpy_gpu_to_gpu( size_t dst, size_t src, size_t count )
{
   printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to GPU[0x%Lx] ...", count, 
          (unsigned long long) src, (unsigned long long) dst );
   fflush(stdout);
   for (unsigned n=0; n < count; n ++ ) {
      unsigned char tmp;
      g_global_mem->read(src+n,1,&tmp); 
      g_global_mem->write(dst+n,1, &tmp,NULL,NULL);
   }
   printf( " done.\n");
   fflush(stdout);
}

void gpgpu_ptx_sim_memset( size_t dst_start_addr, int c, size_t count )
{
   printf("GPGPU-Sim PTX: setting %zu bytes of memory to 0x%x starting at 0x%Lx... ", 
          count, (unsigned char) c, (unsigned long long) dst_start_addr );
   fflush(stdout);
   unsigned char c_value = (unsigned char)c;
   for (unsigned n=0; n < count; n ++ ) 
      g_global_mem->write(dst_start_addr+n,1,&c_value,NULL,NULL);
   printf( " done.\n");
   fflush(stdout);
}

int ptx_thread_done( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   int result = 0;
   result = (the_thread==NULL) || the_thread->is_done();
   return result;
}

const char * ptx_get_fname( unsigned PC )
{
    static const char *null_ptr = "<null finfo ptr>";
    std::map<unsigned,function_info*>::iterator f=g_pc_to_finfo.find(PC);
    if( f== g_pc_to_finfo.end() ) 
        return null_ptr;
    return f->second->get_name().c_str();
}

unsigned ptx_thread_donecycle( void *thr )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thr;
   if( the_thread == NULL ) 
      return 0;
   return the_thread->donecycle();
}

int ptx_thread_get_next_pc( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return -1;
   return the_thread->get_pc(); // PC should already be updatd to next PC at this point (was set in shader_decode() last time thread ran)
}

void* ptx_thread_get_next_finfo( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return NULL;
   return the_thread->get_finfo(); // finfo should already be updatd to next PC at this point (was set in shader_decode() last time thread ran)
}

int ptx_thread_at_barrier( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return 0;
   return the_thread->is_at_barrier();
}

int ptx_thread_all_at_barrier( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return 0;
   return the_thread->all_at_barrier()?1:0;
}

unsigned long long ptx_thread_get_cta_uid( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return 0;
   return the_thread->get_cta_uid();
}

void ptx_thread_reset_barrier( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return;
   the_thread->clear_barrier();
}

void ptx_thread_release_barrier( void *thd )
{
   ptx_thread_info *the_thread = (ptx_thread_info *) thd;
   if ( the_thread == NULL )
      return;
   the_thread->release_barrier();
}

void ptx_print_insn( address_type pc, FILE *fp )
{
   std::map<unsigned,function_info*>::iterator f = g_pc_to_finfo.find(pc);
   if( f == g_pc_to_finfo.end() ) {
       fprintf(fp,"<no instruction at address 0x%x (%u)>", pc, pc );
       return;
   }
   function_info *finfo = f->second;
   assert( finfo );
   finfo->print_insn(pc,fp);
}

void function_info::ptx_decode_inst( ptx_thread_info *thread, 
                                     unsigned *op_type, 
                                     int *i1, int *i2, int *i3, int *i4, 
                                     int *o1, int *o2, int *o3, int *o4, 
                                     int *vectorin, 
                                     int *vectorout,
                                     int *arch_reg )
{
   addr_t pc = thread->get_pc();
   unsigned index = pc - m_start_PC;
   assert( index < m_instr_mem_size );
   ptx_instruction *pI = m_instr_mem[index]; //get instruction from m_instr_mem[PC]

   bool has_dst = false ;
   int opcode = pI->get_opcode(); //determine the opcode

   switch ( pI->get_opcode() ) {
#define OP_DEF(OP,FUNC,STR,DST,CLASSIFICATION) case OP: has_dst = (DST!=0); break;
#include "opcodes.def"
#undef OP_DEF
   default:
      printf( "Execution error: Invalid opcode (0x%x)\n", pI->get_opcode() );
      break;
   }

   *op_type = ALU_OP;
   if ( opcode == LD_OP ) {
      *op_type = LOAD_OP;
   } else if ( opcode == ST_OP ) {
      *op_type = STORE_OP;
   } else if ( opcode == BRA_OP ) {
      *op_type = BRANCH_OP;
   } else if ( opcode == TEX_OP ) {
      *op_type = LOAD_OP;
   } else if ( opcode == ATOM_OP ) {
      *op_type = LOAD_OP; // make atomics behave more like a load.
   } else if ( opcode == BAR_OP ) {
      *op_type = BARRIER_OP;
   }

   int n=0,m=0;
   ptx_instruction::const_iterator op=pI->op_iter_begin();
   for ( ; op != pI->op_iter_end(); op++, n++ ) { //process operands

      const operand_info &o = *op;
      if ( has_dst && n==0 ) {
         if ( o.is_reg() ) { //but is destination an actual register? (seems like it fails if it's a vector)
            *o1 = o.reg_num();
            arch_reg[0] = o.arch_reg_num();
         } else if ( o.is_vector() ) { //but is destination an actual register? (seems like it fails if it's a vector)
            *vectorin = 1;
            *o1 = o.reg1_num();
            *o2 = o.reg2_num();
            *o3 = o.reg3_num();
            *o4 = o.reg4_num();
            for (int i = 0; i < 4; i++) 
               arch_reg[i] = o.arch_reg_num(i);
         }
      } else {
         if ( o.is_reg() ) {
            int reg_num = o.reg_num();
            arch_reg[m + 4] = o.arch_reg_num();
            switch ( m ) {
            case 0: *i1 = reg_num; break;
            case 1: *i2 = reg_num; break;
            case 2: *i3 = reg_num; break;
            default: 
               break; 
            }
            m++;
         } else if ( o.is_vector() ) {
            assert(m == 0); //only support 1 vector operand (for textures) right now
            *vectorout = 1;
            *i1 = o.reg1_num();
            *i2 = o.reg2_num();
            *i3 = o.reg3_num();
            *i4 = o.reg4_num();
            for (int i = 0; i < 4; i++) 
               arch_reg[i + 4] = o.arch_reg_num(i);
            m+=4;
         }
      }
   }
}

void function_info::add_param_name_type_size( unsigned index, std::string name, int type, size_t size )
{
   unsigned parsed_index;
   char buffer[2048];
   snprintf(buffer,2048,"%s_param_%%u", m_name.c_str() );
   int ntokens = sscanf(name.c_str(),buffer,&parsed_index);
   if( ntokens == 1 ) {
      assert( m_ptx_kernel_param_info.find(parsed_index) == m_ptx_kernel_param_info.end() );
      m_ptx_kernel_param_info[parsed_index] = param_info(name, type, size);
   } else {
      assert( m_ptx_kernel_param_info.find(index) == m_ptx_kernel_param_info.end() );
      m_ptx_kernel_param_info[index] = param_info(name, type, size);
   }
}

void function_info::add_param_data( unsigned argn, struct gpgpu_ptx_sim_arg *args )
{
   const void *data = args->m_start;

   if( data ) {
      param_t tmp;

      tmp.pdata = args->m_start;
      tmp.size = args->m_nbytes;
      tmp.offset = args->m_offset;
      tmp.type = 0;
      std::map<unsigned,param_info>::iterator i=m_ptx_kernel_param_info.find(argn);
      if( i != m_ptx_kernel_param_info.end()) {
         i->second.add_data(tmp);
      } else {
         // This should only happen for OpenCL:
         // 
         // The LLVM PTX compiler in NVIDIA's driver (version 190.29)
         // does not generate an argument in the function declaration 
         // for __constant arguments.
         //
         // The associated constant memory space can be allocated in two 
         // ways. It can be explicitly initialized in the .ptx file where
         // it is declared.  Or, it can be allocated using the clCreateBuffer
         // on the host. In this later case, the .ptx file will contain 
         // a global declaration of the parameter, but it will have an unknown
         // array size.  Thus, the symbol's address will not be set and we need
         // to set it here before executing the PTX.
         
         char buffer[2048];
         snprintf(buffer,2048,"%s_param_%u",m_name.c_str(),argn);
         
         symbol *p = m_symtab->lookup(buffer);
         if( p == NULL ) {
            printf("GPGPU-Sim PTX: ERROR ** could not locate symbol for \'%s\' : cannot bind buffer\n", buffer);
            abort();
         }
         p->set_address((addr_t)*(size_t*)data);
      } 
   } else {
      // This should only happen for OpenCL, but doesn't cause problems
   }
}

void function_info::finalize( memory_space *param_mem ) 
{
   unsigned param_address = 0;
   for( std::map<unsigned,param_info>::iterator i=m_ptx_kernel_param_info.begin(); i!=m_ptx_kernel_param_info.end(); i++ ) {
      param_info &p = i->second;
      std::string name = p.get_name();
      int type = p.get_type();
      param_t param_value = p.get_value();
      param_value.type = type;
      symbol *param = m_symtab->lookup(name.c_str());
      unsigned xtype = param->type()->get_key().scalar_type();
      assert(xtype==(unsigned)type);
      size_t size;
      size = param_value.size; // size of param in bytes
      //assert(param_value.offset == param_address);
      if( size != p.get_size() / 8) {
         printf("GPGPU-Sim PTX: WARNING actual kernel paramter size = %zu bytes vs. formal size = %zu (using smaller of two)\n",
                size, p.get_size()/8);
         size = (size<(p.get_size()/8))?size:(p.get_size()/8);
      } 
      // copy the parameter over word-by-word so that parameter that crosses a memory page can be copied over
      const size_t word_size = 4; 
      for (size_t idx = 0; idx < size; idx += word_size) {
         const char *pdata = reinterpret_cast<const char*>(param_value.pdata) + idx; // cast to char * for ptr arithmetic
         param_mem->write(param_address + idx, word_size, pdata,NULL,NULL); 
      }
      param->set_address(param_address);
      param_address += size; 
   }
}

void function_info::list_param( FILE *fout ) const
{
   for( std::map<unsigned,param_info>::const_iterator i=m_ptx_kernel_param_info.begin(); i!=m_ptx_kernel_param_info.end(); i++ ) {
      const param_info &p = i->second;
      std::string name = p.get_name();
      symbol *param = m_symtab->lookup(name.c_str());
      addr_t param_addr = param->get_address();
      fprintf(fout, "%s: %#08x\n", name.c_str(), param_addr);
   }
   fflush(fout);
}

template<int activate_level> 
bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc)
{
   if (g_debug_execution >= activate_level) {
      // check each type of debug dump constraint to filter out dumps
      if ( (g_debug_thread_uid != 0) && (thd_uid != (unsigned)g_debug_thread_uid) ) {
         return false;
      }
      if ( (g_debug_pc != 0xBEEF1518) && (pc != g_debug_pc) ) {
         return false;
      }

      return true;
   } 
   
   return false;
}

unsigned datatype2size( unsigned data_type )
{
   unsigned data_size;
   switch ( data_type ) {
      case B8_TYPE:
      case S8_TYPE:
      case U8_TYPE: 
         data_size = 1; break;
      case B16_TYPE:
      case S16_TYPE:
      case U16_TYPE:
      case F16_TYPE: 
         data_size = 2; break;
      case B32_TYPE:
      case S32_TYPE:
      case U32_TYPE:
      case F32_TYPE: 
         data_size = 4; break;
      case B64_TYPE:
      case S64_TYPE:
      case U64_TYPE:
      case F64_TYPE: 
         data_size = 8; break;
      default: assert(0); break;
   }
   return data_size; 
}

extern unsigned long long  gpu_sim_cycle;
unsigned g_warp_active_mask;

void function_info::ptx_exec_inst( ptx_thread_info *thread, 
                                   addr_t *addr, 
                                   memory_space_t *space, 
                                   unsigned *data_size, 
                                   dram_callback_t* callback, 
                                   unsigned warp_active_mask  )
{
   bool skip = false;
   int op_classification = 0;
   addr_t pc = thread->next_instr();
   unsigned index = pc - m_start_PC;
   assert( index < m_instr_mem_size );
   ptx_instruction *pI = m_instr_mem[index];
   try {

   thread->clearRPC();
   thread->m_last_set_operand_value.u64 = 0;

   if ( g_debug_execution >= 6 ) {
      if ( (g_debug_thread_uid==0) || (thread->get_uid() == (unsigned)g_debug_thread_uid) ) {
         thread->clear_modifiedregs();
         thread->enable_debug_trace();
      }
   }
   if( pI->has_pred() ) {
      const operand_info &pred = pI->get_pred();
      ptx_reg_t pred_value = thread->get_operand_value(pred);
      skip = !pred_value.pred ^ pI->get_pred_neg();
   }
   g_warp_active_mask = warp_active_mask;
   if( !skip ) {
      switch ( pI->get_opcode() ) {
#define OP_DEF(OP,FUNC,STR,DST,CLASSIFICATION) case OP: FUNC(pI,thread); op_classification = CLASSIFICATION; break;
#include "opcodes.def"
#undef OP_DEF
      default:
         printf( "Execution error: Invalid opcode (0x%x)\n", pI->get_opcode() );
         break;
      }
   }

   if ( ptx_debug_exec_dump_cond<5>(thread->get_uid(), pc) ) {
      dim3 ctaid = thread->get_ctaid();
      dim3 tid = thread->get_tid();
      printf("%u [cyc=%u][thd=%u][i=%u] : ctaid=(%u,%u,%u) tid=(%u,%u,%u) icount=%u [pc=%u] (%s:%u - %s)  [0x%llx]\n", 
             g_ptx_sim_num_insn, 
             (unsigned)gpu_sim_cycle,
             thread->get_uid(),
             pI->uid(), ctaid.x,ctaid.y,ctaid.z,tid.x,tid.y,tid.z,
             thread->get_icount(),
             pc, pI->source_file(), pI->source_line(), pI->get_source(),
             thread->m_last_set_operand_value.u64 );
      fflush(stdout);
   }

   addr_t insn_memaddr = 0xFEEBDAED;
   memory_space_t insn_space = undefined_space;
   unsigned insn_data_size = 0;
   if ( pI->get_opcode() == LD_OP || pI->get_opcode() == ST_OP || pI->get_opcode() == TEX_OP ) {
      insn_memaddr = thread->last_eaddr();
      insn_space = thread->last_space();

      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
   }

   if ( pI->get_opcode() == ATOM_OP ) {
      insn_memaddr = thread->last_eaddr();
      insn_space = thread->last_space();
      callback->function = thread->last_callback().function;
      callback->instruction = thread->last_callback().instruction;
      callback->thread = thread;

      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
   } else {
      // make sure that the callback isn't set
      callback->function = NULL;
      callback->instruction = NULL;
   }

   if ( g_debug_execution >= 6 ) {
      if ( ptx_debug_exec_dump_cond<6>(thread->get_uid(), pc) )
         thread->dump_modifiedregs();
   } else if ( g_debug_execution >= 10 ) {
      if ( ptx_debug_exec_dump_cond<10>(thread->get_uid(), pc) )
         thread->dump_regs();
   }
   thread->update_pc();
   g_ptx_sim_num_insn++;
   ptx_file_line_stats_add_exec_count(pI);
   if ( gpgpu_ptx_instruction_classification ) {
      unsigned space_type=0;
      switch ( pI->get_space().get_type() ) {
      case global_space: space_type = 10; break;
      case local_space:  space_type = 11; break; 
      case tex_space:    space_type = 12; break; 
      case surf_space:   space_type = 13; break; 
      case param_space_kernel:
      case param_space_local:
                         space_type = 14; break; 
      case shared_space: space_type = 15; break; 
      case const_space:  space_type = 16; break;
      default: 
         space_type = 0 ;
         break;
      }
      StatAddSample( g_inst_classification_stat[g_ptx_kernel_count],  op_classification);
      if (space_type) StatAddSample( g_inst_classification_stat[g_ptx_kernel_count], ( int )space_type);
      StatAddSample( g_inst_op_classification_stat[g_ptx_kernel_count], (int)  pI->get_opcode() );
   }
   if ( (g_ptx_sim_num_insn % 100000) == 0 ) {
      dim3 ctaid = thread->get_ctaid();
      dim3 tid = thread->get_tid();
      printf("GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) tid=(%u,%u,%u)\n",
             g_ptx_sim_num_insn, ctaid.x,ctaid.y,ctaid.z,tid.x,tid.y,tid.z );
      fflush(stdout);
   }

   // "Return values"
   *space = insn_space;
   *addr = insn_memaddr;
   *data_size = insn_data_size;

   } catch ( int x  ) {
      printf("GPGPU-Sim PTX: ERROR (%d) executing intruction (%s:%u)\n", x, pI->source_file(), pI->source_line() );
      printf("GPGPU-Sim PTX:       '%s'\n", pI->get_source() );
      abort();
   }
}

unsigned g_gx, g_gy, g_gz;

dim3 g_cudaGridDim, g_cudaBlockDim;

unsigned g_cta_launch_sid;
std::list<ptx_thread_info *> g_active_threads;
std::map<unsigned,unsigned> g_sm_idx_offset_next;
unsigned g_sm_next_index;
std::map<unsigned,memory_space*> g_shared_memory_lookup;
std::map<unsigned,ptx_cta_info*> g_ptx_cta_lookup;
std::map<unsigned,std::map<unsigned,memory_space*> > g_local_memory_lookup;

// return number of blocks in grid
unsigned ptx_sim_grid_size()
{
   return g_cudaGridDim.x * g_cudaGridDim.y * g_cudaGridDim.z;
}

void set_option_gpgpu_spread_blocks_across_cores(int option)
{
   gpgpu_option_spread_blocks_across_cores = option;
}

void set_param_gpgpu_num_shaders(int num_shaders)
{
   gpgpu_param_num_shaders = num_shaders;
}

unsigned ptx_sim_cta_size()
{
   return g_cudaBlockDim.x * g_cudaBlockDim.y * g_cudaBlockDim.z;
} 

const struct gpgpu_ptx_sim_kernel_info* ptx_sim_kernel_info() {
   return g_entrypoint_func_info->get_kernel_info();
}

void ptx_sim_free_sm( ptx_thread_info** thread_info )
{
}

unsigned ptx_sim_init_thread( ptx_thread_info** thread_info,int sid,unsigned tid,unsigned threads_left,unsigned num_threads, core_t *core, unsigned hw_cta_id, unsigned hw_warp_id )
{
   if ( *thread_info != NULL ) {
      ptx_thread_info *thd = *thread_info;
      assert( thd->is_done() );
      if ( g_debug_execution==-1 ) {
         dim3 ctaid = thd->get_ctaid();
         dim3 tid = thd->get_tid();
         printf("GPGPU-Sim PTX simulator:  thread exiting ctaid=(%u,%u,%u) tid=(%u,%u,%u) uid=%u\n",
                ctaid.x,ctaid.y,ctaid.z,tid.x,tid.y,tid.z, thd->get_uid() );
         fflush(stdout);
      }
      thd->m_cta_info->assert_barrier_empty();
      thd->m_cta_info->register_deleted_thread(thd);
      delete thd;
      *thread_info = NULL;
   }

   if ( !g_active_threads.empty() ) { //if g_active_threads not empty...
      assert( g_active_threads.size() <= threads_left );
      if ( g_cta_launch_sid == (unsigned)-1 )
         g_cta_launch_sid = sid;
      assert( g_cta_launch_sid == (unsigned)sid );
      ptx_thread_info *thd = g_active_threads.front(); 
      g_active_threads.pop_front();
      *thread_info = thd;
      thd->set_hw_tid(tid);
      thd->set_hw_wid(hw_warp_id);
      thd->set_hw_ctaid(hw_cta_id);
      thd->set_core(core);
      thd->set_hw_sid(sid);
      return 1;
   }

   if ( g_gx >= g_cudaGridDim.x  || g_gy >= g_cudaGridDim.y || g_gz >= g_cudaGridDim.z ) {
      return 0; //finished!
   }

   if ( threads_left < ptx_sim_cta_size() ) {
      return 0;
   }

   if ( g_debug_execution==-1 ) {
      printf("GPGPU-Sim PTX simulator:  STARTING THREAD ALLOCATION --> \n");
      fflush(stdout);
   }

   //initializing new CTA
   ptx_cta_info *cta_info = NULL;
   memory_space *shared_mem = NULL;

   unsigned cta_size = ptx_sim_cta_size(); //blocksize
   unsigned sm_offset = g_sm_idx_offset_next[sid];
   unsigned max_cta_per_sm = num_threads/cta_size; // e.g., 256 / 48 = 5 
   assert( max_cta_per_sm > 0 );

   unsigned sm_idx = sid*max_cta_per_sm + sm_offset;
   sm_idx = max_cta_per_sm*sid + tid/cta_size;

   if (!gpgpu_option_spread_blocks_across_cores) {
      // update offset...
      if ( (sm_offset + 1) >= max_cta_per_sm ) {
         sm_offset = 0;
      } else {
         sm_offset++;
      }
      g_sm_idx_offset_next[sid] = sm_offset;
   } else {
      sm_idx = (tid/cta_size)*gpgpu_param_num_shaders + sid;
   }

   if ( g_shared_memory_lookup.find(sm_idx) == g_shared_memory_lookup.end() ) {
      if ( g_debug_execution >= 1 ) {
         printf("  <CTA alloc> : sm_idx=%u sid=%u sm_offset=%u max_cta_per_sm=%u\n", 
                sm_idx, sid, sm_offset, max_cta_per_sm );
      }
      char buf[512];
      snprintf(buf,512,"shared_%u", sid);
      shared_mem = new memory_space_impl<16*1024>(buf,4);
      g_shared_memory_lookup[sm_idx] = shared_mem;
      cta_info = new ptx_cta_info(sm_idx);
      g_ptx_cta_lookup[sm_idx] = cta_info;
   } else {
      if ( g_debug_execution >= 1 ) {
         printf("  <CTA realloc> : sm_idx=%u sid=%u sm_offset=%u max_cta_per_sm=%u\n", 
                sm_idx, sid, sm_offset, max_cta_per_sm );
      }
      shared_mem = g_shared_memory_lookup[sm_idx];
      cta_info = g_ptx_cta_lookup[sm_idx];
      cta_info->check_cta_thread_status_and_reset();
   }

   std::map<unsigned,memory_space*> &local_mem_lookup = g_local_memory_lookup[sid];
   unsigned new_tid;
   for ( unsigned tz=0; tz < g_cudaBlockDim.z; tz++ ) {
      for ( unsigned ty=0; ty < g_cudaBlockDim.y; ty++ ) {
         for ( unsigned tx=0; tx < g_cudaBlockDim.x; tx++ ) {
            new_tid = tx + g_cudaBlockDim.x*ty + g_cudaBlockDim.x*g_cudaBlockDim.y*tz;
            new_tid += tid;
            ptx_thread_info *thd = new ptx_thread_info();

            memory_space *local_mem = NULL;
            std::map<unsigned,memory_space*>::iterator l = local_mem_lookup.find(new_tid);
            if ( l != local_mem_lookup.end() ) {
               local_mem = l->second;
            } else {
               char buf[512];
               snprintf(buf,512,"local_%u_%u", sid, new_tid);
               local_mem = new memory_space_impl<32>(buf,32);
               local_mem_lookup[new_tid] = local_mem;
            }
            thd->set_info(g_entrypoint_func_info);
            thd->set_nctaid(g_cudaGridDim.x,g_cudaGridDim.y,g_cudaGridDim.z);
            thd->set_ntid(g_cudaBlockDim.x,g_cudaBlockDim.y,g_cudaBlockDim.z);
            thd->set_ctaid(g_gx,g_gy,g_gz);
            thd->set_tid(tx,ty,tz);
            thd->set_hw_tid((unsigned)-1);
            thd->set_hw_wid((unsigned)-1);
            thd->set_hw_ctaid((unsigned)-1);
            thd->set_core(NULL);
            thd->set_hw_sid((unsigned)-1);
            thd->set_valid();
            thd->m_shared_mem = shared_mem;
            thd->m_cta_info = cta_info;
            cta_info->add_thread(thd);
            thd->m_local_mem = local_mem;
            if ( g_debug_execution==-1 ) {
               printf("GPGPU-Sim PTX simulator:  allocating thread ctaid=(%u,%u,%u) tid=(%u,%u,%u) @ 0x%Lx\n",
                      g_gx,g_gy,g_gz,tx,ty,tz, (unsigned long long)thd );
               fflush(stdout);
            }
            g_active_threads.push_back(thd);
         }
      }
   }
   if ( g_debug_execution==-1 ) {
      printf("GPGPU-Sim PTX simulator:  <-- FINISHING THREAD ALLOCATION\n");
      fflush(stdout);
   }

   g_gx++;
   if ( g_gx >= g_cudaGridDim.x ) {
      g_gx = 0;
      g_gy++;
      if ( g_gy >= g_cudaGridDim.y ) {
         g_gy = 0;
         g_gz++;
      }
   }

   g_cta_launch_sid = -1;

   assert( g_active_threads.size() <= threads_left );

   g_cta_launch_sid = sid;
   *thread_info = g_active_threads.front();
   (*thread_info)->set_hw_tid(tid);
   (*thread_info)->set_hw_wid(hw_warp_id);
   (*thread_info)->set_hw_ctaid(hw_cta_id);
   (*thread_info)->set_core(core);
   (*thread_info)->set_hw_sid(sid);
   g_active_threads.pop_front();

   return 1;
}

void init_inst_classification_stat() {
   char kernelname[256] ="";
#define MAX_CLASS_KER 256
   if (!g_inst_classification_stat) g_inst_classification_stat = (void**)calloc(MAX_CLASS_KER, sizeof(void*));
   snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",g_ptx_kernel_count  );         
   assert( g_ptx_kernel_count < MAX_CLASS_KER ) ; // a static limit on number of kernels increase it if it fails! 
   g_inst_classification_stat[g_ptx_kernel_count] = StatCreate(kernelname,1,20);
   if (!g_inst_op_classification_stat) g_inst_op_classification_stat = (void**)calloc(MAX_CLASS_KER, sizeof(void*));
   snprintf(kernelname, MAX_CLASS_KER, "Kernel %d OP Classification\n",g_ptx_kernel_count  );         
   g_inst_op_classification_stat[g_ptx_kernel_count] = StatCreate(kernelname,1,100);
}

unsigned g_max_regs_per_thread = 0;

std::map<std::string,function_info*> *g_kernel_name_to_function_lookup=NULL;
std::map<std::string,symbol_table*> g_kernel_name_to_symtab_lookup;
std::map<const void*,std::string> *g_host_to_kernel_entrypoint_name_lookup=NULL;
extern unsigned g_ptx_thread_info_uid_next;

void gpgpu_ptx_sim_init_grid( const char *kernel_key, struct gpgpu_ptx_sim_arg* args,
                                         struct dim3 gridDim, struct dim3 blockDim ) 
{
   g_gx=0;
   g_gy=0;
   g_gz=0;
   g_cudaGridDim = gridDim;
   g_cudaBlockDim = blockDim;
   g_sm_idx_offset_next.clear();
   g_sm_next_index = 0;  

   if ( g_host_to_kernel_entrypoint_name_lookup->find(kernel_key) ==
        g_host_to_kernel_entrypoint_name_lookup->end() ) {
      printf("GPGPU-Sim PTX: ERROR ** cannot locate PTX entry point\n" );
      printf("GPGPU-Sim PTX: existing entry points: \n");
      std::map<const void*,std::string>::iterator i_eptr = g_host_to_kernel_entrypoint_name_lookup->begin();
      for (; i_eptr != g_host_to_kernel_entrypoint_name_lookup->end(); ++i_eptr) {
         printf("GPGPU-Sim PTX: (%p,%s)\n", i_eptr->first, i_eptr->second.c_str());
      }
      printf("\n");
      abort();
   } 

   std::string kname = (*g_host_to_kernel_entrypoint_name_lookup)[kernel_key];
   printf("GPGPU-Sim PTX: Launching kernel \'%s\' gridDim= (%u,%u,%u) blockDim = (%u,%u,%u); ntuid=%u\n",
          kname.c_str(), g_cudaGridDim.x,g_cudaGridDim.y,g_cudaGridDim.z,g_cudaBlockDim.x,g_cudaBlockDim.y,g_cudaBlockDim.z, 
          g_ptx_thread_info_uid_next );

   if ( g_kernel_name_to_function_lookup->find(kname) ==
        g_kernel_name_to_function_lookup->end() ) {
      printf("GPGPU-Sim PTX: ERROR ** function \'%s\' not found in ptx file\n", kname.c_str() );
      abort();
   }
   g_entrypoint_func_info = g_func_info = (*g_kernel_name_to_function_lookup)[kname];
   g_entrypoint_symbol_table = g_kernel_name_to_symtab_lookup[kname];

   unsigned argcount=0;
   struct gpgpu_ptx_sim_arg *tmparg = args;
   while (tmparg) {
      tmparg = tmparg->m_next;
      argcount++;
   }

   unsigned argn=1;
   while (args) {
      g_func_info->add_param_data(argcount-argn,args);
      args = args->m_next;
      argn++;
   }
   g_func_info->finalize(g_param_mem);
   g_ptx_kernel_count++; 
   if ( gpgpu_ptx_instruction_classification ) {
      init_inst_classification_stat();
   }
   fflush(stdout);
}

const char *g_gpgpusim_version_string = "2.1.1b (beta)";

void print_splash()
{
   static int splash_printed=0;
   if ( !splash_printed ) {
      fprintf(stdout, "\n\n        *** GPGPU-Sim version %s ***\n\n\n", g_gpgpusim_version_string );
      splash_printed=1;
   }
}

void gpgpu_ptx_sim_register_kernel(const char *hostFun, const char *deviceFun)
{
   const void* key=hostFun;
   print_splash();
   if ( g_host_to_kernel_entrypoint_name_lookup == NULL )
        g_host_to_kernel_entrypoint_name_lookup = new std::map<const void*,std::string>;
   if( g_kernel_name_to_function_lookup == NULL )
        g_kernel_name_to_function_lookup = new std::map<std::string,function_info*>;
   if ( g_host_to_kernel_entrypoint_name_lookup->find(key) !=
        g_host_to_kernel_entrypoint_name_lookup->end() ) {
      printf("GPGPU-Sim Loader error: Don't know how to identify PTX kernels during cudaLaunch\n"
             "                        for this application.\n");
      abort();
   }
   (*g_host_to_kernel_entrypoint_name_lookup)[key] = deviceFun;
   if( g_kernel_name_to_function_lookup->find(deviceFun) ==
       g_kernel_name_to_function_lookup->end() ) {
      (*g_kernel_name_to_function_lookup)[deviceFun] = NULL; // we set this later, set keys now for error checking
   }

   printf("GPGPU-Sim PTX: __cudaRegisterFunction %s : 0x%Lx\n", deviceFun, (unsigned long long)hostFun);
}

extern int ptx_lineno;

void register_ptx_function( const char *name, function_info *impl, symbol_table *symtab )
{
   printf("GPGPU-Sim PTX: parsing function %s\n", name );
   if( g_kernel_name_to_function_lookup == NULL )
      g_kernel_name_to_function_lookup = new std::map<std::string,function_info*>;

   std::map<std::string,function_info*>::iterator i_kernel = g_kernel_name_to_function_lookup->find(name);
   if (i_kernel != g_kernel_name_to_function_lookup->end() && i_kernel->second != NULL) {
      printf("GPGPU-Sim PTX: WARNING: Function already parsed once. Overwriting.\n");
   }
   (*g_kernel_name_to_function_lookup)[name] = impl;
   g_kernel_name_to_symtab_lookup[name] = symtab;
}

std::map<const void*,std::string>   g_const_name_lookup; // indexed by hostVar
std::map<const void*,std::string>   g_global_name_lookup; // indexed by hostVar
std::set<std::string>   g_globals;
std::set<std::string>   g_constants;

void gpgpu_ptx_sim_register_const_variable(void *hostVar, const char *deviceName, size_t size )
{
   printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name mapping\n", deviceName, size );
   g_const_name_lookup[hostVar] = deviceName;
   //assert( g_current_symbol_table != NULL );
   //g_sym_name_to_symbol_table[deviceName] = g_current_symbol_table;
}

void gpgpu_ptx_sim_register_global_variable(void *hostVar, const char *deviceName, size_t size )
{
   printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n", deviceName );
   g_global_name_lookup[hostVar] = deviceName;
   //assert( g_current_symbol_table != NULL );
   //g_sym_name_to_symbol_table[deviceName] = g_current_symbol_table;
}

void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src, size_t count, size_t offset, int to )
{
   printf("GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar 0x%p\n", hostVar);
   bool found_sym = false;
   memory_space_t mem_region = undefined_space;
   std::string sym_name;

   std::map<const void*,std::string>::iterator c=g_const_name_lookup.find(hostVar);
   if ( c!=g_const_name_lookup.end() ) {
      found_sym = true;
      sym_name = c->second;
      mem_region = const_space;
   }
   std::map<const void*,std::string>::iterator g=g_global_name_lookup.find(hostVar);
   if ( g!=g_global_name_lookup.end() ) {
      if ( found_sym ) {
         printf("Execution error: PTX symbol \"%s\" w/ hostVar=0x%Lx is declared both const and global?\n", 
                sym_name.c_str(), (unsigned long long)hostVar );
         abort();
      }
      found_sym = true;
      sym_name = g->second;
      mem_region = global_space;
   }
   if( g_globals.find(hostVar) != g_globals.end() ) {
      found_sym = true;
      sym_name = hostVar;
      mem_region = global_space;
   }
   if( g_constants.find(hostVar) != g_constants.end() ) {
      found_sym = true;
      sym_name = hostVar;
      mem_region = const_space;
   }

   if ( !found_sym ) {
      printf("Execution error: No information for PTX symbol w/ hostVar=0x%Lx\n", (unsigned long long)hostVar );
      abort();
   } else printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ hostVar=0x%Lx\n", (unsigned long long)hostVar ); 
   const char *mem_name = NULL;
   memory_space *mem = NULL;

   std::map<std::string,symbol_table*>::iterator st = g_sym_name_to_symbol_table.find(sym_name.c_str());
   assert( st != g_sym_name_to_symbol_table.end() );
   symbol_table *symtab = st->second;

   symbol *sym = symtab->lookup(sym_name.c_str());
   assert(sym);
   unsigned dst = sym->get_address() + offset; 
   switch (mem_region.get_type()) {
   case const_space:
      mem = g_global_mem;
      mem_name = "global";
      break;
   case global_space:
      mem = g_global_mem;
      mem_name = "global";
      break;
   default:
      abort();
   }
   printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %zu bytes %s symbol %s+%zu @0x%x ...\n", 
          count, (to?" to ":"from"), sym_name.c_str(), offset, dst );
   for ( unsigned n=0; n < count; n++ ) {
      if( to ) mem->write(dst+n,1,((char*)src)+n,NULL,NULL); 
      else mem->read(dst+n,1,((char*)src)+n); 
   }
   fflush(stdout);
}

int g_ptx_sim_mode=0; 
// used by libcuda.a if non-zer cudaLaunch() will call gpgpu_ptx_sim_main_func()
// if zero it calls gpgpu_ptx_sim_main_perf()

extern "C" int ptx_debug;

void read_sim_environment_variables() 
{
   ptx_debug = 0;
   g_debug_execution = 0;
   g_debug_ir_generation = false;
   g_interactive_debugger_enabled = false;

   char *mode = getenv("PTX_SIM_MODE_FUNC");
   if ( mode )
      sscanf(mode,"%u", &g_ptx_sim_mode);
   printf("GPGPU-Sim PTX: simulation mode %d (can change with PTX_SIM_MODE_FUNC environment variable:\n", g_ptx_sim_mode);
   printf("               1=functional simulation only, 0=detailed performance simulator)\n");
   g_filename = getenv("PTX_SIM_KERNELFILE"); 
   char *dbg_inter = getenv("GPGPUSIM_DEBUG");
   if ( dbg_inter && strlen(dbg_inter) ) {
      printf("GPGPU-Sim PTX: enabling interactive debugger\n");
      fflush(stdout);
      g_interactive_debugger_enabled = true;
   }
   char *dbg_level = getenv("PTX_SIM_DEBUG");
   if ( dbg_level && strlen(dbg_level) ) {
      printf("GPGPU-Sim PTX: setting debug level to %s\n", dbg_level );
      fflush(stdout);
      sscanf(dbg_level,"%d", &g_debug_execution);
   }
   char *dbg_thread = getenv("PTX_SIM_DEBUG_THREAD_UID");
   if ( dbg_thread && strlen(dbg_thread) ) {
      printf("GPGPU-Sim PTX: printing debug information for thread uid %s\n", dbg_thread );
      fflush(stdout);
      sscanf(dbg_thread,"%d", &g_debug_thread_uid);
   }
   char *dbg_pc = getenv("PTX_SIM_DEBUG_PC");
   if ( dbg_pc && strlen(dbg_pc) ) {
      printf("GPGPU-Sim PTX: printing debug information for instruction with PC = %s\n", dbg_pc );
      fflush(stdout);
      sscanf(dbg_pc,"%d", &g_debug_pc);
   }

#if CUDART_VERSION > 1010
    g_override_embedded_ptx = false;
    char *usefile = getenv("PTX_SIM_USE_PTX_FILE");
    if (usefile && strlen(usefile)) {
        printf("GPGPU-Sim PTX: overriding embedded ptx with ptx file (PTX_SIM_USE_PTX_FILE is set)\n");
        fflush(stdout);
        g_override_embedded_ptx = true;
    }
#else
   g_override_embedded_ptx = true;
#endif

   if ( g_debug_execution >= 40 ) {
      ptx_debug = 1;
   }
   if ( g_debug_execution >= 30 ) {
      g_debug_ir_generation = true;
   }
}






extern time_t simulation_starttime;

ptx_cta_info *g_func_cta_info = NULL;

#define MAX(a,b) (((a)>(b))?(a):(b))

void gpgpu_ptx_sim_main_func( const char *kernel_key, dim3 gridDim, dim3 blockDim, struct gpgpu_ptx_sim_arg *args)
{
   printf("GPGPU-Sim: Performing Functional Simulation...\n");

   printf("ERROR: Need to derived core_t for functional simulation, functional simulation no longer operational\n"); 
      // also: need PDOM stack, etc... for functional simulation
   exit(1);
   
   time_t end_time, elapsed_time, days, hrs, minutes, sec;
   int i1, i2, i3, i4, o1, o2, o3, o4;
   int vectorin, vectorout;

   gpgpu_ptx_sim_init_grid(kernel_key, args,gridDim,blockDim);

   memory_space *shared_mem = new memory_space_impl<16*1024>("shared",4);

   std::map<unsigned,memory_space*> lm_lookup;

   if ( g_func_cta_info == NULL )
      g_func_cta_info = new ptx_cta_info(0);

   for ( unsigned gx=0; gx < gridDim.x; gx++ ) {
      for ( unsigned gy=0; gy < gridDim.y; gy++ ) {
         for ( unsigned gz=0; gz < gridDim.z; gz++ ) {
            std::list<ptx_thread_info *> active_threads;
            std::list<ptx_thread_info *> blocked_threads;

            g_func_cta_info->check_cta_thread_status_and_reset();

            for ( unsigned tx=0; tx < blockDim.x; tx++ ) {
               for ( unsigned ty=0; ty < blockDim.y; ty++ ) {
                  for ( unsigned tz=0; tz < blockDim.z; tz++ ) {
                     memory_space *local_mem = NULL;
                     ptx_thread_info *thd = new ptx_thread_info();

                     unsigned lm_idx = blockDim.x*blockDim.y*tz + blockDim.x * ty + tx;
                     std::map<unsigned,memory_space*>::iterator lm=lm_lookup.find(lm_idx);
                     if ( lm == lm_lookup.end() ) {
                        char buf[1024];
                        snprintf(buf,1024,"local_(%u,%u,%u)", tx, ty, tz );
                        local_mem = new memory_space_impl<32>(buf,32);
                        lm_lookup[lm_idx] = local_mem;
                     } else {
                        local_mem = lm->second;
                     }


                     thd->set_info(g_func_info);
                     thd->set_nctaid(gridDim.x,gridDim.y,gridDim.z);
                     thd->set_ntid(blockDim.x, blockDim.y, blockDim.z);
                     thd->set_ctaid(gx,gy,gz);
                     thd->set_tid(tx,ty,tz);
                     thd->set_valid();
                     thd->m_shared_mem = shared_mem;
                     thd->m_local_mem = local_mem;
                     thd->m_cta_info = g_func_cta_info;
                     g_func_cta_info->add_thread(thd);
                     active_threads.push_back(thd);
                  }
               }
            }

            while ( !(active_threads.empty() && blocked_threads.empty()) ) {
               // while there are still threads left to execute in this CTA
               ptx_thread_info *thread = NULL;

               if ( !active_threads.empty() ) {
                  thread = active_threads.front();
                  active_threads.pop_front();
               } else {
                  active_threads = blocked_threads;
                  blocked_threads.clear();
                  std::list<ptx_thread_info *>::iterator a=active_threads.begin();
                  for ( ; a != active_threads.end(); a++ ) {
                     ptx_thread_info *thd = *a;
                     thd->clear_barrier();
                  }
                  g_func_cta_info->release_barrier(); 
               }

               while ( thread != NULL ) {
                  if ( thread->is_at_barrier() ) {
                     blocked_threads.push_back(thread);
                     thread = NULL;
                     break;
                  }
                  if ( thread->is_done() ) {
                     thread->m_cta_info->register_deleted_thread(thread);
                     delete thread;
                     thread = NULL;
                     break;
                  }

                  unsigned op_type;
                  addr_t addr;
                  memory_space_t space;
                  int arch_reg[MAX_REG_OPERANDS] = { -1 };
                  unsigned data_size;
                  dram_callback_t callback;
                  unsigned warp_active_mask = (unsigned)-1; // vote instruction with diverged warps won't execute correctly
                                                            // in functional simulation mode

                  g_func_info->ptx_decode_inst( thread, &op_type, &i1, &i2, &i3, &i4, &o1, &o2, &o3, &o4, &vectorin, &vectorout, arch_reg );
                  g_func_info->ptx_exec_inst( thread, &addr, &space, &data_size, &callback, warp_active_mask );
               }
            }
         }
      }
   }
   printf( "GPGPU-Sim: Done functional simulation (%u instructions simulated).\n", g_ptx_sim_num_insn );
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp ( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }
   end_time = time((time_t *)NULL);
   elapsed_time = MAX(end_time - simulation_starttime, 1);

   days    = elapsed_time/(3600*24);
   hrs     = elapsed_time/3600 - 24*days;
   minutes = elapsed_time/60 - 60*(hrs + 24*days);
   sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));

   fflush(stderr);
   printf("\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
          (unsigned)days, (unsigned)hrs, (unsigned)minutes, (unsigned)sec, (unsigned)elapsed_time );
   printf("gpgpu_simulation_rate = %u (inst/sec)\n", (unsigned)(g_ptx_sim_num_insn / elapsed_time) );
   fflush(stdout); 
}

void ptx_decode_inst( void *thd, unsigned *op, int *i1, int *i2, int *i3, int *i4, int *o1, int *o2, int *o3, int *o4, int *vectorin, int *vectorout, int *arch_reg  )
{
   *op = NO_OP;
   *o1 = 0;
   *o2 = 0;
   *o3 = 0;
   *o4 = 0;
   *i1 = 0;
   *i2 = 0;
   *i3 = 0;
   *i4 = 0;
   *vectorin = 0;
   *vectorout = 0;
   std::fill_n(arch_reg, MAX_REG_OPERANDS, -1);

   if ( thd == NULL )
      return;

   ptx_thread_info *thread = (ptx_thread_info *) thd;
   g_func_info = thread->func_info();
   g_func_info->ptx_decode_inst(thread,op,i1,i2,i3,i4,o1,o2,o3,o4,vectorin,vectorout,arch_reg);
}

void ptx_exec_inst( void *thd, address_type *addr, memory_space_t *space, unsigned *data_size, dram_callback_t* callback, unsigned warp_active_mask )
{
   if ( thd == NULL )
      return;
   ptx_thread_info *thread = (ptx_thread_info *) thd;
   g_func_info = thread->func_info();
   g_func_info->ptx_exec_inst( thread, addr, space, data_size, callback, warp_active_mask );
}

void ptx_dump_regs( void *thd )
{
   if ( thd == NULL )
      return;
   ptx_thread_info *t = (ptx_thread_info *) thd;
   t->dump_regs();
}

unsigned ptx_set_tex_cache_linesize(unsigned linesize)
{
   g_texcache_linesize = linesize;
   return 0;
}

unsigned ptx_kernel_program_size()
{
   return g_func_info->get_function_size();
}

unsigned translate_pc_to_ptxlineno(unsigned pc)
{
   // this function assumes that the kernel fits inside a single PTX file
   // function_info *pFunc = g_func_info; // assume that the current kernel is the one in query
   const ptx_instruction *pInsn = function_info::pc_to_instruction(pc);
   unsigned ptx_line_number = pInsn->source_line();

   return ptx_line_number;
}

int g_ptxinfo_error_detected;


static char *g_ptxinfo_kname = NULL;
static struct gpgpu_ptx_sim_kernel_info g_ptxinfo_kinfo;

extern "C" void ptxinfo_function(const char *fname )
{
    g_ptxinfo_kinfo.regs=0;
    g_ptxinfo_kinfo.lmem=0;
    g_ptxinfo_kinfo.smem=0;
    g_ptxinfo_kinfo.cmem=0;
    g_ptxinfo_kname = strdup(fname);
}

extern "C" void ptxinfo_regs( unsigned nregs )
{
    g_ptxinfo_kinfo.regs=nregs;
}

extern "C" void ptxinfo_lmem( unsigned declared, unsigned system )
{
    g_ptxinfo_kinfo.lmem=declared+system;
}

extern "C" void ptxinfo_smem( unsigned declared, unsigned system )
{
    g_ptxinfo_kinfo.smem=declared+system;
}

extern "C" void ptxinfo_cmem( unsigned nbytes, unsigned bank )
{
    g_ptxinfo_kinfo.cmem+=nbytes;
}

extern "C" void ptxinfo_addinfo()
{
    if ( g_kernel_name_to_function_lookup ) {
        std::map<std::string,function_info*>::iterator i=g_kernel_name_to_function_lookup->find(g_ptxinfo_kname);
        if ( (g_kernel_name_to_function_lookup == NULL) || (i == g_kernel_name_to_function_lookup->end()) ) {
           printf ("GPGPU-Sim PTX: Kernel '%s' in %s not found. Ignoring.\n", g_ptxinfo_kname, g_filename);
        } else {
           printf ("GPGPU-Sim PTX: Kernel %s\n", g_ptxinfo_kname);
           function_info *fi = i->second;
           fi->set_kernel_info(&g_ptxinfo_kinfo);
        }
    } else {
        printf ("GPGPU-Sim PTX: Kernel '%s' in %s not found (no kernels registered).\n", g_ptxinfo_kname, g_filename);
    }

    free(g_ptxinfo_kname);
    g_ptxinfo_kname=NULL;
    g_ptxinfo_kinfo.regs=0;
    g_ptxinfo_kinfo.lmem=0;
    g_ptxinfo_kinfo.smem=0;
    g_ptxinfo_kinfo.cmem=0;
}

void dwf_insert_reconv_pt(address_type pc); 

struct rec_pts {
   gpgpu_recon_t *s_kernel_recon_points;
   int s_num_recon;
};

struct std::map<function_info*,rec_pts> g_rpts;

struct rec_pts find_reconvergence_points( function_info *finfo )
{
   rec_pts tmp;
   std::map<function_info*,rec_pts>::iterator r=g_rpts.find(finfo);
 
   if( r==g_rpts.end() ) {
      int num_recon = finfo->get_num_reconvergence_pairs();
      
      gpgpu_recon_t *kernel_recon_points = (struct gpgpu_recon_t*) calloc(num_recon, sizeof(struct gpgpu_recon_t));
      finfo->get_reconvergence_pairs(kernel_recon_points);
      printf("GPGPU-Sim PTX: Reconvergence Pairs for %s\n", finfo->get_name().c_str() );
      for (int i=0;i<num_recon;i++) 
         printf("GPGPU-Sim PTX:   branch pc = %d\ttarget pc = %d\n", kernel_recon_points[i].source_pc, kernel_recon_points[i].target_pc); 
      tmp.s_kernel_recon_points = kernel_recon_points;
      tmp.s_num_recon = num_recon;
      g_rpts[finfo] = tmp;
   } else {
      tmp = r->second;
   }
   return tmp;
}

unsigned int get_converge_point( unsigned int pc, void *thd ) 
{
   // the branch could encode the reconvergence point and/or a bit that indicates the 
   // reconvergence point is the return PC on the call stack in the case the branch has 
   // no immediate postdominator in the function (i.e., due to multiple return points). 

   std::map<unsigned,function_info*>::iterator f=g_pc_to_finfo.find(pc);
   assert( f != g_pc_to_finfo.end() );
   function_info *finfo = f->second;
   rec_pts tmp = find_reconvergence_points(finfo);

   int i=0;
   for (; i < tmp.s_num_recon; ++i) {
      if (tmp.s_kernel_recon_points[i].source_pc == pc) {
          if( tmp.s_kernel_recon_points[i].target_pc == (unsigned) -2 ) {
              // function call return
              ptx_thread_info *the_thread = (ptx_thread_info*)thd;
              assert( the_thread != NULL );
              return the_thread->get_return_PC();
          } else {
              return tmp.s_kernel_recon_points[i].target_pc;
          }
      }
   }
   assert(i < tmp.s_num_recon);
   abort(); // returning garbage!
}

void find_reconvergence_points()
{
    find_reconvergence_points(g_func_info);
}

void dwf_process_reconv_pts()
{
   rec_pts tmp = find_reconvergence_points(g_func_info);
   for (int i = 0; i < tmp.s_num_recon; ++i) {
      dwf_insert_reconv_pt(tmp.s_kernel_recon_points[i].target_pc);
   }
}

void *gpgpusim_opencl_getkernel_Object( const char *kernel_name )
{
   std::map<std::string,function_info*>::iterator i=g_kernel_name_to_function_lookup->find(kernel_name);

   if( i == g_kernel_name_to_function_lookup->end() ) {
      abort();
      return NULL;
   }
   return i->second;
}

