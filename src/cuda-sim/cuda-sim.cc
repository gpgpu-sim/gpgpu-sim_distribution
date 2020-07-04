// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan, Jimmy Kwa
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

#include "cuda-sim.h"

#include "instructions.h"
#include "ptx_ir.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include <stdio.h>
#include <map>
#include <set>
#include <sstream>
#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../stream_manager.h"
#include "cuda_device_runtime.h"
#include "decuda_pred_table/decuda_pred_table.h"
#include "memory.h"
#include "opcodes.h"
#include "ptx-stats.h"
#include "ptx.tab.h"
#include "ptx_loader.h"
#include "ptx_parser.h"
#include "ptx_sim.h"

int g_debug_execution = 0;
// Output debug information to file options

void cuda_sim::ptx_opcocde_latency_options(option_parser_t opp) {
  option_parser_register(
      opp, "-ptx_opcode_latency_int", OPT_CSTR, &opcode_latency_int,
      "Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,19,25,145,32",
      "1,1,19,25,145,32");
  option_parser_register(opp, "-ptx_opcode_latency_fp", OPT_CSTR,
                         &opcode_latency_fp,
                         "Opcode latencies for single precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,30",
                         "1,1,1,1,30");
  option_parser_register(opp, "-ptx_opcode_latency_dp", OPT_CSTR,
                         &opcode_latency_dp,
                         "Opcode latencies for double precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,335",
                         "8,8,8,8,335");
  option_parser_register(opp, "-ptx_opcode_latency_sfu", OPT_CSTR,
                         &opcode_latency_sfu,
                         "Opcode latencies for SFU instructions"
                         "Default 8",
                         "8");
  option_parser_register(opp, "-ptx_opcode_latency_tesnor", OPT_CSTR,
                         &opcode_latency_tensor,
                         "Opcode latencies for Tensor instructions"
                         "Default 64",
                         "64");
  option_parser_register(
      opp, "-ptx_opcode_initiation_int", OPT_CSTR, &opcode_initiation_int,
      "Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,4,4,32,4",
      "1,1,4,4,32,4");
  option_parser_register(opp, "-ptx_opcode_initiation_fp", OPT_CSTR,
                         &opcode_initiation_fp,
                         "Opcode initiation intervals for single precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,5",
                         "1,1,1,1,5");
  option_parser_register(opp, "-ptx_opcode_initiation_dp", OPT_CSTR,
                         &opcode_initiation_dp,
                         "Opcode initiation intervals for double precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,130",
                         "8,8,8,8,130");
  option_parser_register(opp, "-ptx_opcode_initiation_sfu", OPT_CSTR,
                         &opcode_initiation_sfu,
                         "Opcode initiation intervals for sfu instructions"
                         "Default 8",
                         "8");
  option_parser_register(opp, "-ptx_opcode_initiation_tensor", OPT_CSTR,
                         &opcode_initiation_tensor,
                         "Opcode initiation intervals for tensor instructions"
                         "Default 64",
                         "64");
  option_parser_register(opp, "-cdp_latency", OPT_CSTR, &cdp_latency_str,
                         "CDP API latency <cudaStreamCreateWithFlags, \
cudaGetParameterBufferV2_init_perWarp, cudaGetParameterBufferV2_perKernel, \
cudaLaunchDeviceV2_init_perWarp, cudaLaunchDevicV2_perKernel>"
                         "Default 7200,8000,100,12000,1600",
                         "7200,8000,100,12000,1600");
}

void gpgpu_t::gpgpu_ptx_sim_bindNameToTexture(
    const char *name, const struct textureReference *texref, int dim,
    int readmode, int ext) {
  std::string texname(name);
  if (m_NameToTextureRef.find(texname) == m_NameToTextureRef.end()) {
    m_NameToTextureRef[texname] = std::set<const struct textureReference *>();
  } else {
    const struct textureReference *tr = *m_NameToTextureRef[texname].begin();
    assert(tr != NULL);
    // asserts that all texrefs in set have same fields
    assert(tr->normalized == texref->normalized &&
           tr->filterMode == texref->filterMode &&
           tr->addressMode[0] == texref->addressMode[0] &&
           tr->addressMode[1] == texref->addressMode[1] &&
           tr->addressMode[2] == texref->addressMode[2] &&
           tr->channelDesc.x == texref->channelDesc.x &&
           tr->channelDesc.y == texref->channelDesc.y &&
           tr->channelDesc.z == texref->channelDesc.z &&
           tr->channelDesc.w == texref->channelDesc.w &&
           tr->channelDesc.f == texref->channelDesc.f);
  }
  m_NameToTextureRef[texname].insert(texref);
  m_TextureRefToName[texref] = texname;
  const textureReferenceAttr *texAttr = new textureReferenceAttr(
      texref, dim, (enum cudaTextureReadMode)readmode, ext);
  m_NameToAttribute[texname] = texAttr;
}

const char *gpgpu_t::gpgpu_ptx_sim_findNamefromTexture(
    const struct textureReference *texref) {
  std::map<const struct textureReference *, std::string>::const_iterator t =
      m_TextureRefToName.find(texref);
  assert(t != m_TextureRefToName.end());
  return t->second.c_str();
}

unsigned int intLOGB2(unsigned int v) {
  unsigned int shift;
  unsigned int r;

  r = 0;

  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift;
  r |= shift;

  return r;
}

void gpgpu_t::gpgpu_ptx_sim_bindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array) {
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);

  std::map<std::string, const struct cudaArray *>::const_iterator t =
      m_NameToCudaArray.find(texname);
  // check that there's nothing there first
  if (t != m_NameToCudaArray.end()) {
    printf(
        "GPGPU-Sim PTX:   Warning: binding to texref associated with %s, which "
        "was previously bound.\nImplicitly unbinding texref associated to %s "
        "first\n",
        texname.c_str(), texname.c_str());
  }
  m_NameToCudaArray[texname] = array;
  unsigned int texel_size_bits =
      array->desc.w + array->desc.x + array->desc.y + array->desc.z;
  unsigned int texel_size = texel_size_bits / 8;
  unsigned int Tx, Ty;
  int r;

  printf("GPGPU-Sim PTX:   texel size = %d\n", texel_size);
  printf("GPGPU-Sim PTX:   texture cache linesize = %d\n",
         m_function_model_config.get_texcache_linesize());
  // first determine base Tx size for given linesize
  switch (m_function_model_config.get_texcache_linesize()) {
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
      printf(
          "GPGPU-Sim PTX:   Line size of %d bytes currently not supported.\n",
          m_function_model_config.get_texcache_linesize());
      assert(0);
      break;
  }
  r = texel_size >> 2;
  // modify base Tx size to take into account size of each texel in bytes
  while (r != 0) {
    Tx = Tx >> 1;
    r = r >> 2;
  }
  // by now, got the correct Tx size, calculate correct Ty size
  Ty = m_function_model_config.get_texcache_linesize() / (Tx * texel_size);

  printf(
      "GPGPU-Sim PTX:   Tx = %d; Ty = %d, Tx_numbits = %d, Ty_numbits = %d\n",
      Tx, Ty, intLOGB2(Tx), intLOGB2(Ty));
  printf("GPGPU-Sim PTX:   Texel size = %d bytes; texel_size_numbits = %d\n",
         texel_size, intLOGB2(texel_size));
  printf(
      "GPGPU-Sim PTX:   Binding texture to array starting at devPtr32 = 0x%x\n",
      array->devPtr32);
  printf("GPGPU-Sim PTX:   Texel size = %d bytes\n", texel_size);
  struct textureInfo *texInfo =
      (struct textureInfo *)malloc(sizeof(struct textureInfo));
  texInfo->Tx = Tx;
  texInfo->Ty = Ty;
  texInfo->Tx_numbits = intLOGB2(Tx);
  texInfo->Ty_numbits = intLOGB2(Ty);
  texInfo->texel_size = texel_size;
  texInfo->texel_size_numbits = intLOGB2(texel_size);
  m_NameToTextureInfo[texname] = texInfo;
}

void gpgpu_t::gpgpu_ptx_sim_unbindTexture(
    const struct textureReference *texref) {
  // assumes bind-use-unbind-bind-use-unbind pattern
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);
  m_NameToCudaArray.erase(texname);
  m_NameToTextureInfo.erase(texname);
}

#define MAX_INST_SIZE 8 /*bytes*/

void function_info::ptx_assemble() {
  if (m_assembled) {
    return;
  }

  // get the instructions into instruction memory...
  unsigned num_inst = m_instructions.size();
  m_instr_mem_size = MAX_INST_SIZE * (num_inst + 1);
  m_instr_mem = new ptx_instruction *[m_instr_mem_size];

  printf("GPGPU-Sim PTX: instruction assembly for function \'%s\'... ",
         m_name.c_str());
  fflush(stdout);
  std::list<ptx_instruction *>::iterator i;

  addr_t PC =
      gpgpu_ctx->func_sim->g_assemble_code_next_pc;  // globally unique address
                                                     // (across functions)
  // start function on an aligned address
  for (unsigned i = 0; i < (PC % MAX_INST_SIZE); i++)
    gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);
  PC += PC % MAX_INST_SIZE;
  m_start_PC = PC;

  addr_t n = 0;  // offset in m_instr_mem
  // Why s_g_pc_to_insn.size() is needed to reserve additional memory for insts?
  // reserve is cumulative. s_g_pc_to_insn.reserve(s_g_pc_to_insn.size() +
  // MAX_INST_SIZE*m_instructions.size());
  gpgpu_ctx->s_g_pc_to_insn.reserve(MAX_INST_SIZE * m_instructions.size());
  for (i = m_instructions.begin(); i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;
    if (pI->is_label()) {
      const symbol *l = pI->get_label();
      labels[l->name()] = n;
    } else {
      gpgpu_ctx->func_sim->g_pc_to_finfo[PC] = this;
      m_instr_mem[n] = pI;
      gpgpu_ctx->s_g_pc_to_insn.push_back(pI);
      assert(pI == gpgpu_ctx->s_g_pc_to_insn[PC]);
      pI->set_m_instr_mem_index(n);
      pI->set_PC(PC);
      assert(pI->inst_size() <= MAX_INST_SIZE);
      for (unsigned i = 1; i < pI->inst_size(); i++) {
        gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);
        m_instr_mem[n + i] = NULL;
      }
      n += pI->inst_size();
      PC += pI->inst_size();
    }
  }
  gpgpu_ctx->func_sim->g_assemble_code_next_pc = PC;
  for (unsigned ii = 0; ii < n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];
    if (pI->get_opcode() == BRA_OP || pI->get_opcode() == BREAKADDR_OP ||
        pI->get_opcode() == CALLP_OP) {
      operand_info &target = pI->dst();  // get operand, e.g. target name
      if (labels.find(target.name()) == labels.end()) {
        printf(
            "GPGPU-Sim PTX: Loader error (%s:%u): Branch label \"%s\" does not "
            "appear in assembly code.",
            pI->source_file(), pI->source_line(), target.name().c_str());
        abort();
      }
      unsigned index = labels[target.name()];  // determine address from name
      unsigned PC = m_instr_mem[index]->get_PC();
      m_symtab->set_label_address(target.get_symbol(), PC);
      target.set_type(label_t);
    }
  }
  m_n = n;
  printf("  done.\n");
  fflush(stdout);

  // disable pdom analysis  here and do it at runtime
#if 0
   printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n", m_name.c_str() );
   create_basic_blocks();
   connect_basic_blocks();
   bool modified = false; 
   do {
      find_dominators();
      find_idominators();
      modified = connect_break_targets(); 
   } while (modified == true);

   if ( g_debug_execution>=50 ) {
      print_basic_blocks();
      print_basic_block_links();
      print_basic_block_dot();
   }
   if ( g_debug_execution>=2 ) {
      print_dominators();
   }
   find_postdominators();
   find_ipostdominators();
   if ( g_debug_execution>=50 ) {
      print_postdominators();
      print_ipostdominators();
   }

   printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n", m_name.c_str() );
   for ( unsigned ii=0; ii < n; ii += m_instr_mem[ii]->inst_size() ) { // handle branch instructions
      ptx_instruction *pI = m_instr_mem[ii];
      pI->pre_decode();
   }
   printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n", m_name.c_str() );
   fflush(stdout);

   m_assembled = true;
#endif
}

addr_t shared_to_generic(unsigned smid, addr_t addr) {
  assert(addr < SHARED_MEM_SIZE_MAX);
  return SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX + addr;
}

addr_t global_to_generic(addr_t addr) { return addr; }

bool isspace_shared(unsigned smid, addr_t addr) {
  addr_t start = SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX;
  addr_t end = SHARED_GENERIC_START + (smid + 1) * SHARED_MEM_SIZE_MAX;
  if ((addr >= end) || (addr < start)) return false;
  return true;
}

bool isspace_global(addr_t addr) {
  return (addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT);
}

memory_space_t whichspace(addr_t addr) {
  if ((addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT)) {
    return global_space;
  } else if (addr >= SHARED_GENERIC_START) {
    return shared_space;
  } else {
    return local_space;
  }
}

addr_t generic_to_shared(unsigned smid, addr_t addr) {
  assert(isspace_shared(smid, addr));
  return addr - (SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX);
}

addr_t local_to_generic(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(addr < LOCAL_MEM_SIZE_MAX);
  return LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
         (LOCAL_MEM_SIZE_MAX * hwtid) + addr;
}

bool isspace_local(unsigned smid, unsigned hwtid, addr_t addr) {
  addr_t start = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
                 (LOCAL_MEM_SIZE_MAX * hwtid);
  addr_t end = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
               (LOCAL_MEM_SIZE_MAX * (hwtid + 1));
  if ((addr >= end) || (addr < start)) return false;
  return true;
}

addr_t generic_to_local(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(isspace_local(smid, hwtid, addr));
  return addr - (LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
                 (LOCAL_MEM_SIZE_MAX * hwtid));
}

addr_t generic_to_global(addr_t addr) { return addr; }

void *gpgpu_t::gpu_malloc(size_t size) {
  unsigned long long result = m_dev_malloc;
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }
  m_dev_malloc += size;
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
  return (void *)result;
}

void *gpgpu_t::gpu_mallocarray(size_t size) {
  unsigned long long result = m_dev_malloc;
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }
  m_dev_malloc += size;
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
  return (void *)result;
}

void gpgpu_t::memcpy_to_gpu(size_t dst_start_addr, const void *src,
                            size_t count) {
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: copying %zu bytes from CPU[0x%Lx] to GPU[0x%Lx] ... ",
        count, (unsigned long long)src, (unsigned long long)dst_start_addr);
    fflush(stdout);
  }
  char *src_data = (char *)src;
  for (unsigned n = 0; n < count; n++)
    m_global_mem->write(dst_start_addr + n, 1, src_data + n, NULL, NULL);

  // Copy into the performance model.
  // extern gpgpu_sim* g_the_gpu;
  gpgpu_ctx->the_gpgpusim->g_the_gpu->perf_memcpy_to_gpu(dst_start_addr, count);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

void gpgpu_t::memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count) {
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to CPU[0x%Lx] ...",
           count, (unsigned long long)src_start_addr, (unsigned long long)dst);
    fflush(stdout);
  }
  unsigned char *dst_data = (unsigned char *)dst;
  for (unsigned n = 0; n < count; n++)
    m_global_mem->read(src_start_addr + n, 1, dst_data + n);

  // Copy into the performance model.
  // extern gpgpu_sim* g_the_gpu;
  gpgpu_ctx->the_gpgpusim->g_the_gpu->perf_memcpy_to_gpu(src_start_addr, count);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

void gpgpu_t::memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count) {
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to GPU[0x%Lx] ...",
           count, (unsigned long long)src, (unsigned long long)dst);
    fflush(stdout);
  }
  for (unsigned n = 0; n < count; n++) {
    unsigned char tmp;
    m_global_mem->read(src + n, 1, &tmp);
    m_global_mem->write(dst + n, 1, &tmp, NULL, NULL);
  }
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

void gpgpu_t::gpu_memset(size_t dst_start_addr, int c, size_t count) {
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: setting %zu bytes of memory to 0x%x starting at "
        "0x%Lx... ",
        count, (unsigned char)c, (unsigned long long)dst_start_addr);
    fflush(stdout);
  }
  unsigned char c_value = (unsigned char)c;
  for (unsigned n = 0; n < count; n++)
    m_global_mem->write(dst_start_addr + n, 1, &c_value, NULL, NULL);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

void cuda_sim::ptx_print_insn(address_type pc, FILE *fp) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  if (f == g_pc_to_finfo.end()) {
    fprintf(fp, "<no instruction at address 0x%x>", pc);
    return;
  }
  function_info *finfo = f->second;
  assert(finfo);
  finfo->print_insn(pc, fp);
}

std::string cuda_sim::ptx_get_insn_str(address_type pc) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  if (f == g_pc_to_finfo.end()) {
#define STR_SIZE 255
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';
    snprintf(buff, STR_SIZE, "<no instruction at address 0x%x>", pc);
    return std::string(buff);
  }
  function_info *finfo = f->second;
  assert(finfo);
  return finfo->get_insn_str(pc);
}

void ptx_instruction::set_fp_or_int_archop() {
  oprnd_type = UN_OP;
  if ((m_opcode == MEMBAR_OP) || (m_opcode == SSY_OP) || (m_opcode == BRA_OP) ||
      (m_opcode == BAR_OP) || (m_opcode == RET_OP) || (m_opcode == RETP_OP) ||
      (m_opcode == NOP_OP) || (m_opcode == EXIT_OP) || (m_opcode == CALLP_OP) ||
      (m_opcode == CALL_OP)) {
    // do nothing
  } else if ((m_opcode == CVT_OP || m_opcode == SET_OP ||
              m_opcode == SLCT_OP)) {
    if (get_type2() == F16_TYPE || get_type2() == F32_TYPE ||
        get_type2() == F64_TYPE || get_type2() == FF64_TYPE) {
      oprnd_type = FP_OP;
    } else
      oprnd_type = INT_OP;

  } else {
    if (get_type() == F16_TYPE || get_type() == F32_TYPE ||
        get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      oprnd_type = FP_OP;
    } else
      oprnd_type = INT_OP;
  }
}
void ptx_instruction::set_mul_div_or_other_archop() {
  sp_op = OTHER_OP;
  if ((m_opcode != MEMBAR_OP) && (m_opcode != SSY_OP) && (m_opcode != BRA_OP) &&
      (m_opcode != BAR_OP) && (m_opcode != EXIT_OP) && (m_opcode != NOP_OP) &&
      (m_opcode != RETP_OP) && (m_opcode != RET_OP) && (m_opcode != CALLP_OP) &&
      (m_opcode != CALL_OP)) {
    if (get_type() == F32_TYPE || get_type() == F64_TYPE ||
        get_type() == FF64_TYPE) {
      switch (get_opcode()) {
        case MUL_OP:
        case MAD_OP:
          sp_op = FP_MUL_OP;
          break;
        case DIV_OP:
          sp_op = FP_DIV_OP;
          break;
        case LG2_OP:
          sp_op = FP_LG_OP;
          break;
        case RSQRT_OP:
        case SQRT_OP:
          sp_op = FP_SQRT_OP;
          break;
        case RCP_OP:
          sp_op = FP_DIV_OP;
          break;
        case SIN_OP:
        case COS_OP:
          sp_op = FP_SIN_OP;
          break;
        case EX2_OP:
          sp_op = FP_EXP_OP;
          break;
        default:
          if ((op == ALU_OP) || (op == TENSOR_CORE_OP)) sp_op = FP__OP;
          break;
      }
    } else {
      switch (get_opcode()) {
        case MUL24_OP:
        case MAD24_OP:
          sp_op = INT_MUL24_OP;
          break;
        case MUL_OP:
        case MAD_OP:
          if (get_type() == U32_TYPE || get_type() == S32_TYPE ||
              get_type() == B32_TYPE)
            sp_op = INT_MUL32_OP;
          else
            sp_op = INT_MUL_OP;
          break;
        case DIV_OP:
          sp_op = INT_DIV_OP;
          break;
        default:
          if ((op == ALU_OP)) sp_op = INT__OP;
          break;
      }
    }
  }
}

void ptx_instruction::set_bar_type() {
  if (m_opcode == BAR_OP) {
    switch (m_barrier_op) {
      case SYNC_OPTION:
        bar_type = SYNC;
        break;
      case ARRIVE_OPTION:
        bar_type = ARRIVE;
        break;
      case RED_OPTION:
        bar_type = RED;
        switch (m_atomic_spec) {
          case ATOMIC_POPC:
            red_type = POPC_RED;
            break;
          case ATOMIC_AND:
            red_type = AND_RED;
            break;
          case ATOMIC_OR:
            red_type = OR_RED;
            break;
        }
        break;
      default:
        abort();
    }
  } else if (m_opcode == SST_OP) {
    bar_type = SYNC;
  }
}

void ptx_instruction::set_opcode_and_latency() {
  unsigned int_latency[6];
  unsigned fp_latency[5];
  unsigned dp_latency[5];
  unsigned sfu_latency;
  unsigned tensor_latency;
  unsigned int_init[6];
  unsigned fp_init[5];
  unsigned dp_init[5];
  unsigned sfu_init;
  unsigned tensor_init;
  /*
   * [0] ADD,SUB
   * [1] MAX,Min
   * [2] MUL
   * [3] MAD
   * [4] DIV
   * [5] SHFL
   */
  sscanf(gpgpu_ctx->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u,%u",
         &int_latency[0], &int_latency[1], &int_latency[2], &int_latency[3],
         &int_latency[4], &int_latency[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_int, "%u,%u,%u,%u,%u,%u",
         &int_init[0], &int_init[1], &int_init[2], &int_init[3], &int_init[4],
         &int_init[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_fp, "%u,%u,%u,%u,%u",
         &fp_init[0], &fp_init[1], &fp_init[2], &fp_init[3], &fp_init[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_dp, "%u,%u,%u,%u,%u",
         &dp_init[0], &dp_init[1], &dp_init[2], &dp_init[3], &dp_init[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_sfu, "%u", &sfu_init);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_tensor, "%u", &tensor_init);
  sscanf(gpgpu_ctx->func_sim->cdp_latency_str, "%u,%u,%u,%u,%u",
         &gpgpu_ctx->func_sim->cdp_latency[0],
         &gpgpu_ctx->func_sim->cdp_latency[1],
         &gpgpu_ctx->func_sim->cdp_latency[2],
         &gpgpu_ctx->func_sim->cdp_latency[3],
         &gpgpu_ctx->func_sim->cdp_latency[4]);

  if (!m_operands.empty()) {
    std::vector<operand_info>::iterator it;
    for (it = ++m_operands.begin(); it != m_operands.end(); it++) {
      num_operands++;
      if ((it->is_reg() || it->is_vector())) {
        num_regs++;
      }
    }
  }
  op = ALU_OP;
  mem_op = NOT_TEX;
  initiation_interval = latency = 1;
  switch (m_opcode) {
    case MOV_OP:
      assert(!(has_memory_read() && has_memory_write()));
      if (has_memory_read()) op = LOAD_OP;
      if (has_memory_write()) op = STORE_OP;
      break;
    case LD_OP:
      op = LOAD_OP;
      break;
    case MMA_LD_OP:
      op = TENSOR_CORE_LOAD_OP;
      break;
    case LDU_OP:
      op = LOAD_OP;
      break;
    case ST_OP:
      op = STORE_OP;
      break;
    case MMA_ST_OP:
      op = TENSOR_CORE_STORE_OP;
      break;
    case BRA_OP:
      op = BRANCH_OP;
      break;
    case BREAKADDR_OP:
      op = BRANCH_OP;
      break;
    case TEX_OP:
      op = LOAD_OP;
      mem_op = TEX;
      break;
    case ATOM_OP:
      op = LOAD_OP;
      break;
    case BAR_OP:
      op = BARRIER_OP;
      break;
    case SST_OP:
      op = BARRIER_OP;
      break;
    case MEMBAR_OP:
      op = MEMORY_BARRIER_OP;
      break;
    case CALL_OP: {
      if (m_is_printf || m_is_cdp) {
        op = ALU_OP;
      } else
        op = CALL_OPS;
      break;
    }
    case CALLP_OP: {
      if (m_is_printf || m_is_cdp) {
        op = ALU_OP;
      } else
        op = CALL_OPS;
      break;
    }
    case RET_OP:
    case RETP_OP:
      op = RET_OPS;
      break;
    case ADD_OP:
    case ADDP_OP:
    case ADDC_OP:
    case SUB_OP:
    case SUBC_OP:
      // ADD,SUB latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[0];
          initiation_interval = fp_init[0];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[0];
          initiation_interval = dp_init[0];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[0];
          initiation_interval = int_init[0];
          op = INTP_OP;
          break;
      }
      break;
    case MAX_OP:
    case MIN_OP:
      // MAX,MIN latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[1];
          initiation_interval = fp_init[1];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[1];
          initiation_interval = dp_init[1];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[1];
          initiation_interval = int_init[1];
          op = INTP_OP;
          break;
      }
      break;
    case MUL_OP:
      // MUL latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[2];
          initiation_interval = fp_init[2];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[2];
          initiation_interval = dp_init[2];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[2];
          initiation_interval = int_init[2];
          op = INTP_OP;
          break;
      }
      break;
    case MAD_OP:
    case MADC_OP:
    case MADP_OP:
      // MAD latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[3];
          initiation_interval = fp_init[3];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[3];
          initiation_interval = dp_init[3];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[3];
          initiation_interval = int_init[3];
          op = INTP_OP;
          break;
      }
      break;
    case DIV_OP:
      // Floating point only
      op = SFU_OP;
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[4];
          initiation_interval = fp_init[4];
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[4];
          initiation_interval = dp_init[4];
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[4];
          initiation_interval = int_init[4];
          break;
      }
      break;
    case SQRT_OP:
    case SIN_OP:
    case COS_OP:
    case EX2_OP:
    case LG2_OP:
    case RSQRT_OP:
    case RCP_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      op = SFU_OP;
      break;
    case MMA_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      op = TENSOR_CORE_OP;
      break;
    case SHFL_OP:
      latency = int_latency[5];
      initiation_interval = int_init[5];
      break;
    default:
      break;
  }
  set_fp_or_int_archop();
  set_mul_div_or_other_archop();
}

void ptx_thread_info::ptx_fetch_inst(inst_t &inst) const {
  addr_t pc = get_pc();
  const ptx_instruction *pI = m_func_info->get_instruction(pc);
  inst = (const inst_t &)*pI;
  assert(inst.valid());
}

static unsigned datatype2size(unsigned data_type) {
  unsigned data_size;
  switch (data_type) {
    case B8_TYPE:
    case S8_TYPE:
    case U8_TYPE:
      data_size = 1;
      break;
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
    case F16_TYPE:
      data_size = 2;
      break;
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
    case F32_TYPE:
      data_size = 4;
      break;
    case B64_TYPE:
    case BB64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      data_size = 8;
      break;
    case BB128_TYPE:
      data_size = 16;
      break;
    default:
      assert(0);
      break;
  }
  return data_size;
}

void ptx_instruction::pre_decode() {
  pc = m_PC;
  isize = m_inst_size;
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }
  incount = 0;
  outcount = 0;
  is_vectorin = 0;
  is_vectorout = 0;
  std::fill_n(arch_reg.src, MAX_REG_OPERANDS, -1);
  std::fill_n(arch_reg.dst, MAX_REG_OPERANDS, -1);
  pred = 0;
  ar1 = 0;
  ar2 = 0;
  space = m_space_spec;
  memory_op = no_memory_op;
  data_size = 0;
  if (has_memory_read() || has_memory_write()) {
    unsigned to_type = get_type();
    data_size = datatype2size(to_type);
    memory_op = has_memory_read() ? memory_load : memory_store;
  }

  bool has_dst = false;

  switch (get_opcode()) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                         \
    has_dst = (DST != 0);                          \
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                           \
    has_dst = (DST != 0);                            \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
    default:
      printf("Execution error: Invalid opcode (0x%x)\n", get_opcode());
      break;
  }

  switch (m_cache_option) {
    case CA_OPTION:
      cache_op = CACHE_ALL;
      break;
    case NC_OPTION:
      cache_op = CACHE_L1;
      break;
    case CG_OPTION:
      cache_op = CACHE_GLOBAL;
      break;
    case CS_OPTION:
      cache_op = CACHE_STREAMING;
      break;
    case LU_OPTION:
      cache_op = CACHE_LAST_USE;
      break;
    case CV_OPTION:
      cache_op = CACHE_VOLATILE;
      break;
    case WB_OPTION:
      cache_op = CACHE_WRITE_BACK;
      break;
    case WT_OPTION:
      cache_op = CACHE_WRITE_THROUGH;
      break;
    default:
      // if( m_opcode == LD_OP || m_opcode == LDU_OP )
      if (m_opcode == MMA_LD_OP || m_opcode == LD_OP || m_opcode == LDU_OP)
        cache_op = CACHE_ALL;
      // else if( m_opcode == ST_OP )
      else if (m_opcode == MMA_ST_OP || m_opcode == ST_OP)
        cache_op = CACHE_WRITE_BACK;
      else if (m_opcode == ATOM_OP)
        cache_op = CACHE_GLOBAL;
      break;
  }

  set_opcode_and_latency();
  set_bar_type();
  // Get register operands
  int n = 0, m = 0;
  ptx_instruction::const_iterator opr = op_iter_begin();
  for (; opr != op_iter_end(); opr++, n++) {  // process operands
    const operand_info &o = *opr;
    if (has_dst && n == 0) {
      // Do not set the null register "_" as an architectural register
      if (o.is_reg() && !o.is_non_arch_reg()) {
        out[0] = o.reg_num();
        arch_reg.dst[0] = o.arch_reg_num();
      } else if (o.is_vector()) {
        is_vectorin = 1;
        unsigned num_elem = o.get_vect_nelem();
        if (num_elem >= 1) out[0] = o.reg1_num();
        if (num_elem >= 2) out[1] = o.reg2_num();
        if (num_elem >= 3) out[2] = o.reg3_num();
        if (num_elem >= 4) out[3] = o.reg4_num();
        if (num_elem >= 5) out[4] = o.reg5_num();
        if (num_elem >= 6) out[5] = o.reg6_num();
        if (num_elem >= 7) out[6] = o.reg7_num();
        if (num_elem >= 8) out[7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++) arch_reg.dst[i] = o.arch_reg_num(i);
      }
    } else {
      if (o.is_reg() && !o.is_non_arch_reg()) {
        int reg_num = o.reg_num();
        arch_reg.src[m] = o.arch_reg_num();
        switch (m) {
          case 0:
            in[0] = reg_num;
            break;
          case 1:
            in[1] = reg_num;
            break;
          case 2:
            in[2] = reg_num;
            break;
          default:
            break;
        }
        m++;
      } else if (o.is_vector()) {
        // assert(m == 0); //only support 1 vector operand (for textures) right
        // now
        is_vectorout = 1;
        unsigned num_elem = o.get_vect_nelem();
        if (num_elem >= 1) in[m + 0] = o.reg1_num();
        if (num_elem >= 2) in[m + 1] = o.reg2_num();
        if (num_elem >= 3) in[m + 2] = o.reg3_num();
        if (num_elem >= 4) in[m + 3] = o.reg4_num();
        if (num_elem >= 5) in[m + 4] = o.reg5_num();
        if (num_elem >= 6) in[m + 5] = o.reg6_num();
        if (num_elem >= 7) in[m + 6] = o.reg7_num();
        if (num_elem >= 8) in[m + 7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++)
          arch_reg.src[m + i] = o.arch_reg_num(i);
        m += num_elem;
      }
    }
  }

  // Setting number of input and output operands which is required for
  // scoreboard check
  for (int i = 0; i < MAX_OUTPUT_VALUES; i++)
    if (out[i] > 0) outcount++;

  for (int i = 0; i < MAX_INPUT_VALUES; i++)
    if (in[i] > 0) incount++;

  // Get predicate
  if (has_pred()) {
    const operand_info &p = get_pred();
    pred = p.reg_num();
  }

  // Get address registers inside memory operands.
  // Assuming only one memory operand per instruction,
  //  and maximum of two address registers for one memory operand.
  if (has_memory_read() || has_memory_write()) {
    ptx_instruction::const_iterator op = op_iter_begin();
    for (; op != op_iter_end(); op++, n++) {  // process operands
      const operand_info &o = *op;

      if (o.is_memory_operand()) {
        // We do not support the null register as a memory operand
        assert(!o.is_non_arch_reg());

        // Check PTXPlus-type operand
        // memory operand with addressing (ex. s[0x4] or g[$r1])
        if (o.is_memory_operand2()) {
          // memory operand with one address register (ex. g[$r1+0x4] or
          // s[$r2+=0x4])
          if (o.get_double_operand_type() == 0 ||
              o.get_double_operand_type() == 3) {
            ar1 = o.reg_num();
            arch_reg.src[4] = o.arch_reg_num();
            // TODO: address register in $r2+=0x4 should be an output register
            // as well
          }
          // memory operand with two address register (ex. s[$r1+$r1] or
          // g[$r1+=$r2])
          else if (o.get_double_operand_type() == 1 ||
                   o.get_double_operand_type() == 2) {
            ar1 = o.reg1_num();
            arch_reg.src[4] = o.arch_reg_num();
            ar2 = o.reg2_num();
            arch_reg.src[5] = o.arch_reg_num();
            // TODO: first address register in $r1+=$r2 should be an output
            // register as well
          }
        } else if (o.is_immediate_address()) {
        }
        // Regular PTX operand
        else if (o.get_symbol()
                     ->type()
                     ->get_key()
                     .is_reg()) {  // Memory operand contains a register
          ar1 = o.reg_num();
          arch_reg.src[4] = o.arch_reg_num();
        }
      }
    }
  }

  // get reconvergence pc
  reconvergence_pc = gpgpu_ctx->func_sim->get_converge_point(pc);

  m_decoded = true;
}

void function_info::add_param_name_type_size(unsigned index, std::string name,
                                             int type, size_t size, bool ptr,
                                             memory_space_t space) {
  unsigned parsed_index;
  char buffer[2048];
  snprintf(buffer, 2048, "%s_param_%%u", m_name.c_str());
  int ntokens = sscanf(name.c_str(), buffer, &parsed_index);
  if (ntokens == 1) {
    assert(m_ptx_kernel_param_info.find(parsed_index) ==
           m_ptx_kernel_param_info.end());
    m_ptx_kernel_param_info[parsed_index] =
        param_info(name, type, size, ptr, space);
  } else {
    assert(m_ptx_kernel_param_info.find(index) ==
           m_ptx_kernel_param_info.end());
    m_ptx_kernel_param_info[index] = param_info(name, type, size, ptr, space);
  }
}

void function_info::add_param_data(unsigned argn,
                                   struct gpgpu_ptx_sim_arg *args) {
  const void *data = args->m_start;

  bool scratchpad_memory_param =
      false;  // Is this parameter in CUDA shared memory or OpenCL local memory

  std::map<unsigned, param_info>::iterator i =
      m_ptx_kernel_param_info.find(argn);
  if (i != m_ptx_kernel_param_info.end()) {
    if (i->second.is_ptr_shared()) {
      assert(
          args->m_start == NULL &&
          "OpenCL parameter pointer to local memory must have NULL as value");
      scratchpad_memory_param = true;
    } else {
      param_t tmp;
      tmp.pdata = args->m_start;
      tmp.size = args->m_nbytes;
      tmp.offset = args->m_offset;
      tmp.type = 0;
      i->second.add_data(tmp);
      i->second.add_offset((unsigned)args->m_offset);
    }
  } else {
    scratchpad_memory_param = true;
  }

  if (scratchpad_memory_param) {
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
    snprintf(buffer, 2048, "%s_param_%u", m_name.c_str(), argn);

    symbol *p = m_symtab->lookup(buffer);
    if (p == NULL) {
      printf(
          "GPGPU-Sim PTX: ERROR ** could not locate symbol for \'%s\' : cannot "
          "bind buffer\n",
          buffer);
      abort();
    }
    if (data)
      p->set_address((addr_t) * (size_t *)data);
    else {
      // clSetKernelArg was passed NULL pointer for data...
      // this is used for dynamically sized shared memory on NVIDIA platforms
      bool is_ptr_shared = false;
      if (i != m_ptx_kernel_param_info.end()) {
        is_ptr_shared = i->second.is_ptr_shared();
      }

      if (!is_ptr_shared and !p->is_shared()) {
        printf(
            "GPGPU-Sim PTX: ERROR ** clSetKernelArg passed NULL but arg not "
            "shared memory\n");
        abort();
      }
      unsigned num_bits = 8 * args->m_nbytes;
      printf(
          "GPGPU-Sim PTX: deferred allocation of shared region for \"%s\" from "
          "0x%x to 0x%x (shared memory space)\n",
          p->name().c_str(), m_symtab->get_shared_next(),
          m_symtab->get_shared_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr_t addr = m_symtab->get_shared_next();
      addr_t addr_pad =
          num_bits
              ? (((num_bits / 8) - (addr % (num_bits / 8))) % (num_bits / 8))
              : 0;
      p->set_address(addr + addr_pad);
      m_symtab->alloc_shared(num_bits / 8 + addr_pad);
    }
  }
}

unsigned function_info::get_args_aligned_size() {
  if (m_args_aligned_size >= 0) return m_args_aligned_size;

  unsigned param_address = 0;
  unsigned int total_size = 0;
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());

    size_t arg_size = p.get_size() / 8;  // size of param in bytes
    total_size = (total_size + arg_size - 1) / arg_size * arg_size;  // aligned
    p.add_offset(total_size);
    param->set_address(param_address + total_size);
    total_size += arg_size;
  }

  m_args_aligned_size = (total_size + 3) / 4 * 4;  // final size aligned to word

  return m_args_aligned_size;
}

void function_info::finalize(memory_space *param_mem) {
  unsigned param_address = 0;
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    std::string name = p.get_name();
    int type = p.get_type();
    param_t param_value = p.get_value();
    param_value.type = type;
    symbol *param = m_symtab->lookup(name.c_str());
    unsigned xtype = param->type()->get_key().scalar_type();
    assert(xtype == (unsigned)type);
    size_t size;
    size = param_value.size;  // size of param in bytes
    // assert(param_value.offset == param_address);
    if (size != p.get_size() / 8) {
      printf(
          "GPGPU-Sim PTX: WARNING actual kernel paramter size = %zu bytes vs. "
          "formal size = %zu (using smaller of two)\n",
          size, p.get_size() / 8);
      size = (size < (p.get_size() / 8)) ? size : (p.get_size() / 8);
    }
    // copy the parameter over word-by-word so that parameter that crosses a
    // memory page can be copied over
    // Jin: copy parameter using aligned rules
    const type_info *paramtype = param->type();
    int align_amount = paramtype->get_key().get_alignment_spec();
    align_amount = (align_amount == -1) ? size : align_amount;
    param_address = (param_address + align_amount - 1) / align_amount *
                    align_amount;  // aligned

    const size_t word_size = 4;
    // param_address = (param_address + size - 1) / size * size; //aligned with
    // size
    for (size_t idx = 0; idx < size; idx += word_size) {
      const char *pdata = reinterpret_cast<const char *>(param_value.pdata) +
                          idx;  // cast to char * for ptr arithmetic
      param_mem->write(param_address + idx, word_size, pdata, NULL, NULL);
    }
    unsigned offset = p.get_offset();
    assert(offset == param_address);
    param->set_address(param_address);
    param_address += size;
  }
}

void function_info::param_to_shared(memory_space *shared_mem,
                                    symbol_table *symtab) {
  // TODO: call this only for PTXPlus with GT200 models
  // extern gpgpu_sim* g_the_gpu;
  if (not gpgpu_ctx->the_gpgpusim->g_the_gpu->get_config().convert_to_ptxplus())
    return;

  // copies parameters into simulated shared memory
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    std::string name = p.get_name();
    int type = p.get_type();
    param_t value = p.get_value();
    value.type = type;
    symbol *param = symtab->lookup(name.c_str());
    unsigned xtype = param->type()->get_key().scalar_type();
    assert(xtype == (unsigned)type);

    int tmp;
    size_t size;
    unsigned offset = p.get_offset();
    type_info_key::type_decode(xtype, size, tmp);

    // Write to shared memory - offset + 0x10
    shared_mem->write(offset + 0x10, size / 8, value.pdata, NULL, NULL);
  }
}

void function_info::list_param(FILE *fout) const {
  for (std::map<unsigned, param_info>::const_iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    const param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    fprintf(fout, "%s: %#08x\n", name.c_str(), param_addr);
  }
  fflush(fout);
}

void function_info::ptx_jit_config(
    std::map<unsigned long long, size_t> mallocPtr_Size,
    memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim, dim3 blockDim) {
  static unsigned long long counter = 0;
  std::vector<std::pair<size_t, unsigned char *> > param_data;
  std::vector<unsigned> offsets;
  std::vector<bool> paramIsPointer;

  char *gpgpusim_path = getenv("GPGPUSIM_ROOT");
  assert(gpgpusim_path != NULL);
  char *wys_exec_path = getenv("WYS_EXEC_PATH");
  assert(wys_exec_path != NULL);
  std::string command =
      std::string("mkdir ") + gpgpusim_path + "/debug_tools/WatchYourStep/data";
  std::string filename(std::string(gpgpusim_path) +
                       "/debug_tools/WatchYourStep/data/params.config" +
                       std::to_string(counter));

  // initialize paramList
  char buff[1024];
  std::string filename_c(filename + "_c");
  snprintf(buff, 1024, "c++filt %s > %s", get_name().c_str(),
           filename_c.c_str());
  assert(system(buff) != NULL);
  FILE *fp = fopen(filename_c.c_str(), "r");
  fgets(buff, 1024, fp);
  fclose(fp);
  std::string fn(buff);
  size_t pos1, pos2;
  pos1 = fn.find_last_of("(");
  pos2 = fn.find(")", pos1);
  assert(pos2 > pos1 && pos1 > 0);
  strcpy(buff, fn.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  char *tok;
  tok = strtok(buff, ",");
  std::string tmp;
  while (tok != NULL) {
    std::string param(tok);
    if (param.find("<") != std::string::npos) {
      assert(param.find(">") == std::string::npos);
      assert(param.find("*") == std::string::npos);
      tmp = param;
    } else {
      if (tmp.length() > 0) {
        tmp = "";
        assert(param.find(">") != std::string::npos);
        assert(param.find("<") == std::string::npos);
        assert(param.find("*") == std::string::npos);
      }
      printf("%s\n", param.c_str());
      if (param.find("*") != std::string::npos) {
        paramIsPointer.push_back(true);
      } else {
        paramIsPointer.push_back(false);
      }
    }
    tok = strtok(NULL, ",");
  }

  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    param_t param_value = p.get_value();
    offsets.push_back((unsigned)p.get_offset());

    if (paramIsPointer[i->first] &&
        (*(unsigned long long *)param_value.pdata != 0)) {
      // is pointer
      assert(param_value.size == sizeof(void *) &&
             "MisID'd this param as pointer");
      size_t array_size = 0;
      unsigned long long param_pointer =
          *(unsigned long long *)param_value.pdata;
      if (mallocPtr_Size.find(param_pointer) != mallocPtr_Size.end()) {
        array_size = mallocPtr_Size[param_pointer];
      } else {
        for (std::map<unsigned long long, size_t>::iterator j =
                 mallocPtr_Size.begin();
             j != mallocPtr_Size.end(); j++) {
          if (param_pointer > j->first &&
              param_pointer < j->first + j->second) {
            array_size = j->first + j->second - param_pointer;
            break;
          }
        }
        assert(array_size > 0 && "pointer was not previously malloc'd");
      }

      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      unsigned char *array_val = (unsigned char *)malloc(array_size);
      gpu->get_global_memory()->read(*(unsigned *)((void *)val), array_size,
                                     (void *)array_val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(array_size, array_val));
      paramIsPointer.push_back(true);
    } else {
      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(param_value.size, val));
      paramIsPointer.push_back(false);
    }
  }

  FILE *fout = fopen(filename.c_str(), "w");
  printf("Writing data to %s ...\n", filename.c_str());
  fprintf(fout, "%s\n", get_name().c_str());
  fprintf(fout, "%u,%u,%u %u,%u,%u\n", gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z);
  size_t index = 0;
  for (std::vector<std::pair<size_t, unsigned char *> >::const_iterator i =
           param_data.begin();
       i != param_data.end(); i++) {
    if (paramIsPointer[index]) {
      fprintf(fout, "*");
    }
    fprintf(fout, "%lu :", i->first);
    for (size_t j = 0; j < i->first; j++) {
      fprintf(fout, " %u", i->second[j]);
    }
    fprintf(fout, " : %u", offsets[index]);
    free(i->second);
    fprintf(fout, "\n");
    index++;
  }
  fflush(fout);
  fclose(fout);

  // ptx config
  std::string ptx_config_fn(std::string(gpgpusim_path) +
                            "/debug_tools/WatchYourStep/data/ptx.config" +
                            std::to_string(counter));
  snprintf(buff, 1024,
           "grep -rn \".entry %s\" %s/*.ptx | cut -d \":\" -f 1-2 > %s",
           get_name().c_str(), wys_exec_path, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx source \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  FILE *fin = fopen(ptx_config_fn.c_str(), "r");
  char ptx_source[256];
  unsigned line_number;
  int numscanned = fscanf(fin, "%[^:]:%u", ptx_source, &line_number);
  assert(numscanned == 2);
  fclose(fin);
  snprintf(buff, 1024,
           "grep -rn \".version\" %s | cut -d \":\" -f 1 | xargs -I \"{}\" awk "
           "\"NR>={}&&NR<={}+2\" %s > %s",
           ptx_source, ptx_source, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx header \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  fin = fopen(ptx_source, "r");
  assert(fin != NULL);
  printf("Writing data to %s ...\n", ptx_config_fn.c_str());
  fout = fopen(ptx_config_fn.c_str(), "a");
  assert(fout != NULL);
  for (unsigned i = 0; i < line_number; i++) {
    assert(fgets(buff, 1024, fin) != NULL);
    assert(!feof(fin));
  }
  fprintf(fout, "\n\n");
  do {
    fprintf(fout, "%s", buff);
    assert(fgets(buff, 1024, fin) != NULL);
    if (feof(fin)) {
      break;
    }
  } while (strstr(buff, "entry") == NULL);

  fclose(fin);
  fflush(fout);
  fclose(fout);
  counter++;
}

template <int activate_level>
bool cuda_sim::ptx_debug_exec_dump_cond(int thd_uid, addr_t pc) {
  if (g_debug_execution >= activate_level) {
    // check each type of debug dump constraint to filter out dumps
    if ((g_debug_thread_uid != 0) &&
        (thd_uid != (unsigned)g_debug_thread_uid)) {
      return false;
    }
    if ((g_debug_pc != 0xBEEF1518) && (pc != g_debug_pc)) {
      return false;
    }

    return true;
  }

  return false;
}

void cuda_sim::init_inst_classification_stat() {
  static std::set<unsigned> init;
  if (init.find(g_ptx_kernel_count) != init.end()) return;
  init.insert(g_ptx_kernel_count);

#define MAX_CLASS_KER 1024
  char kernelname[MAX_CLASS_KER] = "";
  if (!g_inst_classification_stat)
    g_inst_classification_stat = (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",
           g_ptx_kernel_count);
  assert(g_ptx_kernel_count <
         MAX_CLASS_KER);  // a static limit on number of kernels increase it if
                          // it fails!
  g_inst_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 20);
  if (!g_inst_op_classification_stat)
    g_inst_op_classification_stat =
        (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d OP Classification\n",
           g_ptx_kernel_count);
  g_inst_op_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 100);
}

static unsigned get_tex_datasize(const ptx_instruction *pI,
                                 ptx_thread_info *thread) {
  const operand_info &src1 = pI->src1();  // the name of the texture
  std::string texname = src1.name();

  /*
    For programs with many streams, textures can be bound and unbound
    asynchronously.  This means we need to use the kernel's "snapshot" of
    the state of the texture mappings when it was launched (so that we
    don't try to access the incorrect texture mapping if it's been updated,
    or that we don't access a mapping that has been unbound).
   */
  kernel_info_t &k = thread->get_kernel();
  const struct textureInfo *texInfo = k.get_texinfo(texname);

  unsigned data_size = texInfo->texel_size;
  return data_size;
}

int tensorcore_op(int inst_opcode) {
  if ((inst_opcode == MMA_OP) || (inst_opcode == MMA_LD_OP) ||
      (inst_opcode == MMA_ST_OP))
    return 1;
  else
    return 0;
}
void ptx_thread_info::ptx_exec_inst(warp_inst_t &inst, unsigned lane_id) {
  bool skip = false;
  int op_classification = 0;
  addr_t pc = next_instr();
  assert(pc ==
         inst.pc);  // make sure timing model and functional model are in sync
  const ptx_instruction *pI = m_func_info->get_instruction(pc);

  set_npc(pc + pI->inst_size());

  try {
    clearRPC();
    m_last_set_operand_value.u64 = 0;

    if (is_done()) {
      printf(
          "attempted to execute instruction on a thread that is already "
          "done.\n");
      assert(0);
    }

    if (g_debug_execution >= 6 ||
        m_gpu->get_config().get_ptx_inst_debug_to_file()) {
      if ((m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid == 0) ||
          (get_uid() ==
           (unsigned)(m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid))) {
        clear_modifiedregs();
        enable_debug_trace();
      }
    }

    if (pI->has_pred()) {
      const operand_info &pred = pI->get_pred();
      ptx_reg_t pred_value = get_operand_value(pred, pred, PRED_TYPE, this, 0);
      if (pI->get_pred_mod() == -1) {
        skip = (pred_value.pred & 0x0001) ^
               pI->get_pred_neg();  // ptxplus inverts the zero flag
      } else {
        skip = !pred_lookup(pI->get_pred_mod(), pred_value.pred & 0x000F);
      }
    }
    int inst_opcode = pI->get_opcode();

    if (skip) {
      inst.set_not_active(lane_id);
    } else {
      const ptx_instruction *pI_saved = pI;
      ptx_instruction *pJ = NULL;
      if (pI->get_opcode() == VOTE_OP || pI->get_opcode() == ACTIVEMASK_OP) {
        pJ = new ptx_instruction(*pI);
        *((warp_inst_t *)pJ) = inst;  // copy active mask information
        pI = pJ;
      }

      if (((inst_opcode == MMA_OP || inst_opcode == MMA_LD_OP ||
            inst_opcode == MMA_ST_OP))) {
        if (inst.active_count() != MAX_WARP_SIZE) {
          printf(
              "Tensor Core operation are warp synchronous operation. All the "
              "threads needs to be active.");
          assert(0);
        }
      }

      // Tensorcore is warp synchronous operation. So these instructions needs
      // to be executed only once. To make the simulation faster removing the
      // redundant tensorcore operation
      if (!tensorcore_op(inst_opcode) ||
          ((tensorcore_op(inst_opcode)) && (lane_id == 0))) {
        switch (inst_opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                         \
    FUNC(pI, this);                                \
    op_classification = CLASSIFICATION;            \
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                           \
    FUNC(pI, get_core(), inst);                      \
    op_classification = CLASSIFICATION;              \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
          default:
            printf("Execution error: Invalid opcode (0x%x)\n",
                   pI->get_opcode());
            break;
        }
      }
      delete pJ;
      pI = pI_saved;

      // Run exit instruction if exit option included
      if (pI->is_exit()) exit_impl(pI, this);
    }

    const gpgpu_functional_sim_config &config = m_gpu->get_config();

    // Output instruction information to file and stdout
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      fprintf(m_gpu->get_ptx_inst_debug_file(), "[thd=%u] : (%s:%u - %s)\n",
              get_uid(), pI->source_file(), pI->source_line(),
              pI->get_source());
      // fprintf(ptx_inst_debug_file, "has memory read=%d, has memory
      // write=%d\n", pI->has_memory_read(), pI->has_memory_write());
      fflush(m_gpu->get_ptx_inst_debug_file());
    }

    if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<5>(get_uid(),
                                                                pc)) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      printf(
          "%u [thd=%u][i=%u] : ctaid=(%u,%u,%u) tid=(%u,%u,%u) icount=%u "
          "[pc=%u] (%s:%u - %s)  [0x%llx]\n",
          m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, get_uid(), pI->uid(),
          ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z, get_icount(), pc,
          pI->source_file(), pI->source_line(), pI->get_source(),
          m_last_set_operand_value.u64);
      fflush(stdout);
    }

    addr_t insn_memaddr = 0xFEEBDAED;
    memory_space_t insn_space = undefined_space;
    _memory_op_t insn_memory_op = no_memory_op;
    unsigned insn_data_size = 0;
    if ((pI->has_memory_read() || pI->has_memory_write())) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        insn_memaddr = last_eaddr();
        insn_space = last_space();
        unsigned to_type = pI->get_type();
        insn_data_size = datatype2size(to_type);
        insn_memory_op = pI->has_memory_read() ? memory_load : memory_store;
      }
    }

    if (pI->get_opcode() == BAR_OP && pI->barrier_op() == RED_OPTION) {
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this,
                        false /*not atomic*/);
    }

    if (pI->get_opcode() == ATOM_OP) {
      insn_memaddr = last_eaddr();
      insn_space = last_space();
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this, true /*atomic*/);
      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
    }

    if (pI->get_opcode() == TEX_OP) {
      inst.set_addr(lane_id, last_eaddr());
      assert(inst.space == last_space());
      insn_data_size = get_tex_datasize(
          pI,
          this);  // texture obtain its data granularity from the texture info
    }

    // Output register information to file and stdout
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      dump_modifiedregs(m_gpu->get_ptx_inst_debug_file());
      dump_regs(m_gpu->get_ptx_inst_debug_file());
    }

    if (g_debug_execution >= 6) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<6>(get_uid(),
                                                                  pc))
        dump_modifiedregs(stdout);
    }
    if (g_debug_execution >= 10) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<10>(get_uid(),
                                                                   pc))
        dump_regs(stdout);
    }
    update_pc();
    m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn++;

    // not using it with functional simulation mode
    if (!(this->m_functionalSimulationMode))
      ptx_file_line_stats_add_exec_count(pI);

    if (m_gpu->gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
      m_gpu->gpgpu_ctx->func_sim->init_inst_classification_stat();
      unsigned space_type = 0;
      switch (pI->get_space().get_type()) {
        case global_space:
          space_type = 10;
          break;
        case local_space:
          space_type = 11;
          break;
        case tex_space:
          space_type = 12;
          break;
        case surf_space:
          space_type = 13;
          break;
        case param_space_kernel:
        case param_space_local:
          space_type = 14;
          break;
        case shared_space:
          space_type = 15;
          break;
        case const_space:
          space_type = 16;
          break;
        default:
          space_type = 0;
          break;
      }
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    op_classification);
      if (space_type)
        StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                          [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                      (int)space_type);
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_op_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    (int)pI->get_opcode());
    }
    if ((m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn % 100000) == 0) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      DPRINTF(LIVENESS,
              "GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) "
              "tid=(%u,%u,%u)\n",
              m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, ctaid.x, ctaid.y,
              ctaid.z, tid.x, tid.y, tid.z);
      fflush(stdout);
    }

    // "Return values"
    if (!skip) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        inst.space = insn_space;
        inst.set_addr(lane_id, insn_memaddr);
        inst.data_size = insn_data_size;  // simpleAtomicIntrinsics
        assert(inst.memory_op == insn_memory_op);
      }
    }

  } catch (int x) {
    printf("GPGPU-Sim PTX: ERROR (%d) executing intruction (%s:%u)\n", x,
           pI->source_file(), pI->source_line());
    printf("GPGPU-Sim PTX:       '%s'\n", pI->get_source());
    abort();
  }
}

void cuda_sim::set_param_gpgpu_num_shaders(int num_shaders) {
  gpgpu_param_num_shaders = num_shaders;
}

const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const function_info *kernel) {
  return kernel->get_kernel_info();
}

const warp_inst_t *gpgpu_context::ptx_fetch_inst(address_type pc) {
  return pc_to_instruction(pc);
}

unsigned ptx_sim_init_thread(kernel_info_t &kernel,
                             ptx_thread_info **thread_info, int sid,
                             unsigned tid, unsigned threads_left,
                             unsigned num_threads, core_t *core,
                             unsigned hw_cta_id, unsigned hw_warp_id,
                             gpgpu_t *gpu, bool isInFunctionalSimulationMode) {
  std::list<ptx_thread_info *> &active_threads = kernel.active_threads();

  static std::map<unsigned, memory_space *> shared_memory_lookup;
  static std::map<unsigned, memory_space *> sstarr_memory_lookup;
  static std::map<unsigned, ptx_cta_info *> ptx_cta_lookup;
  static std::map<unsigned, ptx_warp_info *> ptx_warp_lookup;
  static std::map<unsigned, std::map<unsigned, memory_space *> >
      local_memory_lookup;

  if (*thread_info != NULL) {
    ptx_thread_info *thd = *thread_info;
    assert(thd->is_done());
    if (g_debug_execution == -1) {
      dim3 ctaid = thd->get_ctaid();
      dim3 t = thd->get_tid();
      printf(
          "GPGPU-Sim PTX simulator:  thread exiting ctaid=(%u,%u,%u) "
          "tid=(%u,%u,%u) uid=%u\n",
          ctaid.x, ctaid.y, ctaid.z, t.x, t.y, t.z, thd->get_uid());
      fflush(stdout);
    }
    thd->m_cta_info->register_deleted_thread(thd);
    delete thd;
    *thread_info = NULL;
  }

  if (!active_threads.empty()) {
    assert(active_threads.size() <= threads_left);
    ptx_thread_info *thd = active_threads.front();
    active_threads.pop_front();
    *thread_info = thd;
    thd->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,
              isInFunctionalSimulationMode);
    return 1;
  }

  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (threads_left < kernel.threads_per_cta()) {
    return 0;
  }

  if (g_debug_execution == -1) {
    printf("GPGPU-Sim PTX simulator:  STARTING THREAD ALLOCATION --> \n");
    fflush(stdout);
  }

  // initializing new CTA
  ptx_cta_info *cta_info = NULL;
  memory_space *shared_mem = NULL;
  memory_space *sstarr_mem = NULL;

  unsigned cta_size = kernel.threads_per_cta();
  unsigned max_cta_per_sm = num_threads / cta_size;  // e.g., 256 / 48 = 5
  assert(max_cta_per_sm > 0);

  // unsigned sm_idx = (tid/cta_size)*gpgpu_param_num_shaders + sid;
  unsigned sm_idx =
      hw_cta_id * gpu->gpgpu_ctx->func_sim->gpgpu_param_num_shaders + sid;

  if (shared_memory_lookup.find(sm_idx) == shared_memory_lookup.end()) {
    if (g_debug_execution >= 1) {
      printf("  <CTA alloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n", sm_idx,
             sid, max_cta_per_sm);
    }
    char buf[512];
    snprintf(buf, 512, "shared_%u", sid);
    shared_mem = new memory_space_impl<16 * 1024>(buf, 4);
    shared_memory_lookup[sm_idx] = shared_mem;
    snprintf(buf, 512, "sstarr_%u", sid);
    sstarr_mem = new memory_space_impl<16 * 1024>(buf, 4);
    sstarr_memory_lookup[sm_idx] = sstarr_mem;
    cta_info = new ptx_cta_info(sm_idx, gpu->gpgpu_ctx);
    ptx_cta_lookup[sm_idx] = cta_info;
  } else {
    if (g_debug_execution >= 1) {
      printf("  <CTA realloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n", sm_idx,
             sid, max_cta_per_sm);
    }
    shared_mem = shared_memory_lookup[sm_idx];
    sstarr_mem = sstarr_memory_lookup[sm_idx];
    cta_info = ptx_cta_lookup[sm_idx];
    cta_info->check_cta_thread_status_and_reset();
  }

  std::map<unsigned, memory_space *> &local_mem_lookup =
      local_memory_lookup[sid];
  while (kernel.more_threads_in_cta()) {
    dim3 ctaid3d = kernel.get_next_cta_id();
    unsigned new_tid = kernel.get_next_thread_id();
    dim3 tid3d = kernel.get_next_thread_id_3d();
    kernel.increment_thread_id();
    new_tid += tid;
    ptx_thread_info *thd = new ptx_thread_info(kernel);
    ptx_warp_info *warp_info = NULL;
    if (ptx_warp_lookup.find(hw_warp_id) == ptx_warp_lookup.end()) {
      warp_info = new ptx_warp_info();
      ptx_warp_lookup[hw_warp_id] = warp_info;
    } else {
      warp_info = ptx_warp_lookup[hw_warp_id];
    }
    thd->m_warp_info = warp_info;

    memory_space *local_mem = NULL;
    std::map<unsigned, memory_space *>::iterator l =
        local_mem_lookup.find(new_tid);
    if (l != local_mem_lookup.end()) {
      local_mem = l->second;
    } else {
      char buf[512];
      snprintf(buf, 512, "local_%u_%u", sid, new_tid);
      local_mem = new memory_space_impl<32>(buf, 32);
      local_mem_lookup[new_tid] = local_mem;
    }
    thd->set_info(kernel.entry());
    thd->set_nctaid(kernel.get_grid_dim());
    thd->set_ntid(kernel.get_cta_dim());
    thd->set_ctaid(ctaid3d);
    thd->set_tid(tid3d);
    if (kernel.entry()->get_ptx_version().extensions())
      thd->cpy_tid_to_reg(tid3d);
    thd->set_valid();
    thd->m_shared_mem = shared_mem;
    thd->m_sstarr_mem = sstarr_mem;
    function_info *finfo = thd->func_info();
    symbol_table *st = finfo->get_symtab();
    thd->func_info()->param_to_shared(thd->m_shared_mem, st);
    thd->func_info()->param_to_shared(thd->m_sstarr_mem, st);
    thd->m_cta_info = cta_info;
    cta_info->add_thread(thd);
    thd->m_local_mem = local_mem;
    if (g_debug_execution == -1) {
      printf(
          "GPGPU-Sim PTX simulator:  allocating thread ctaid=(%u,%u,%u) "
          "tid=(%u,%u,%u) @ 0x%Lx\n",
          ctaid3d.x, ctaid3d.y, ctaid3d.z, tid3d.x, tid3d.y, tid3d.z,
          (unsigned long long)thd);
      fflush(stdout);
    }
    active_threads.push_back(thd);
  }
  if (g_debug_execution == -1) {
    printf("GPGPU-Sim PTX simulator:  <-- FINISHING THREAD ALLOCATION\n");
    fflush(stdout);
  }

  kernel.increment_cta_id();

  assert(active_threads.size() <= threads_left);
  *thread_info = active_threads.front();
  (*thread_info)
      ->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,
             isInFunctionalSimulationMode);
  active_threads.pop_front();
  return 1;
}

size_t get_kernel_code_size(class function_info *entry) {
  return entry->get_function_size();
}

kernel_info_t *cuda_sim::gpgpu_opencl_ptx_sim_init_grid(
    class function_info *entry, gpgpu_ptx_sim_arg_list_t args,
    struct dim3 gridDim, struct dim3 blockDim, gpgpu_t *gpu) {
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }
  entry->finalize(result->get_param_memory());
  g_ptx_kernel_count++;
  fflush(stdout);

  return result;
}

#include "../../version"
#include "detailed_version"

void print_splash() {
  static int splash_printed = 0;
  if (!splash_printed) {
    fprintf(stdout, "\n\n        *** %s [build %s] ***\n\n\n",
            g_gpgpusim_version_string, g_gpgpusim_build_string);
    splash_printed = 1;
  }
}

void cuda_sim::gpgpu_ptx_sim_register_const_variable(void *hostVar,
                                                     const char *deviceName,
                                                     size_t size) {
  printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name mapping\n",
         deviceName, size);
  g_const_name_lookup[hostVar] = deviceName;
}

void cuda_sim::gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                                      const char *deviceName,
                                                      size_t size) {
  printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n",
         deviceName);
  g_global_name_lookup[hostVar] = deviceName;
}

void cuda_sim::gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                           size_t count, size_t offset, int to,
                                           gpgpu_t *gpu) {
  printf(
      "GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar 0x%p\n",
      hostVar);
  bool found_sym = false;
  memory_space_t mem_region = undefined_space;
  std::string sym_name;

  std::map<const void *, std::string>::iterator c =
      gpu->gpgpu_ctx->func_sim->g_const_name_lookup.find(hostVar);
  if (c != gpu->gpgpu_ctx->func_sim->g_const_name_lookup.end()) {
    found_sym = true;
    sym_name = c->second;
    mem_region = const_space;
  }
  std::map<const void *, std::string>::iterator g =
      gpu->gpgpu_ctx->func_sim->g_global_name_lookup.find(hostVar);
  if (g != gpu->gpgpu_ctx->func_sim->g_global_name_lookup.end()) {
    if (found_sym) {
      printf(
          "Execution error: PTX symbol \"%s\" w/ hostVar=0x%Lx is declared "
          "both const and global?\n",
          sym_name.c_str(), (unsigned long long)hostVar);
      abort();
    }
    found_sym = true;
    sym_name = g->second;
    mem_region = global_space;
  }
  if (g_globals.find(hostVar) != g_globals.end()) {
    found_sym = true;
    sym_name = hostVar;
    mem_region = global_space;
  }
  if (g_constants.find(hostVar) != g_constants.end()) {
    found_sym = true;
    sym_name = hostVar;
    mem_region = const_space;
  }

  if (!found_sym) {
    printf("Execution error: No information for PTX symbol w/ hostVar=0x%Lx\n",
           (unsigned long long)hostVar);
    abort();
  } else
    printf(
        "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ "
        "hostVar=0x%Lx\n",
        (unsigned long long)hostVar);
  const char *mem_name = NULL;
  memory_space *mem = NULL;

  std::map<std::string, symbol_table *>::iterator st =
      gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.find(sym_name.c_str());
  assert(st != gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.end());
  symbol_table *symtab = st->second;

  symbol *sym = symtab->lookup(sym_name.c_str());
  assert(sym);
  unsigned dst = sym->get_address() + offset;
  switch (mem_region.get_type()) {
    case const_space:
      mem = gpu->get_global_memory();
      mem_name = "const";
      break;
    case global_space:
      mem = gpu->get_global_memory();
      mem_name = "global";
      break;
    default:
      abort();
  }
  printf(
      "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %s memory %zu bytes "
      "%s symbol %s+%zu @0x%x ...\n",
      mem_name, count, (to ? " to " : "from"), sym_name.c_str(), offset, dst);
  for (unsigned n = 0; n < count; n++) {
    if (to)
      mem->write(dst + n, 1, ((char *)src) + n, NULL, NULL);
    else
      mem->read(dst + n, 1, ((char *)src) + n);
  }
  fflush(stdout);
}

extern int ptx_debug;

void cuda_sim::read_sim_environment_variables() {
  ptx_debug = 0;
  g_debug_execution = 0;
  g_interactive_debugger_enabled = false;

  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &g_ptx_sim_mode);
  printf(
      "GPGPU-Sim PTX: simulation mode %d (can change with PTX_SIM_MODE_FUNC "
      "environment variable:\n",
      g_ptx_sim_mode);
  printf(
      "               1=functional simulation only, 0=detailed performance "
      "simulator)\n");
  char *dbg_inter = getenv("GPGPUSIM_DEBUG");
  if (dbg_inter && strlen(dbg_inter)) {
    printf("GPGPU-Sim PTX: enabling interactive debugger\n");
    fflush(stdout);
    g_interactive_debugger_enabled = true;
  }
  char *dbg_level = getenv("PTX_SIM_DEBUG");
  if (dbg_level && strlen(dbg_level)) {
    printf("GPGPU-Sim PTX: setting debug level to %s\n", dbg_level);
    fflush(stdout);
    sscanf(dbg_level, "%d", &g_debug_execution);
  }
  char *dbg_thread = getenv("PTX_SIM_DEBUG_THREAD_UID");
  if (dbg_thread && strlen(dbg_thread)) {
    printf("GPGPU-Sim PTX: printing debug information for thread uid %s\n",
           dbg_thread);
    fflush(stdout);
    sscanf(dbg_thread, "%d", &g_debug_thread_uid);
  }
  char *dbg_pc = getenv("PTX_SIM_DEBUG_PC");
  if (dbg_pc && strlen(dbg_pc)) {
    printf(
        "GPGPU-Sim PTX: printing debug information for instruction with PC = "
        "%s\n",
        dbg_pc);
    fflush(stdout);
    sscanf(dbg_pc, "%d", &g_debug_pc);
  }

#if CUDART_VERSION > 1010
  g_override_embedded_ptx = false;
  char *usefile = getenv("PTX_SIM_USE_PTX_FILE");
  if (usefile && strlen(usefile)) {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with ptx file "
        "(PTX_SIM_USE_PTX_FILE is set)\n");
    fflush(stdout);
    g_override_embedded_ptx = true;
  }
  char *blocking = getenv("CUDA_LAUNCH_BLOCKING");
  if (blocking && !strcmp(blocking, "1")) {
    g_cuda_launch_blocking = true;
  }
#else
  g_cuda_launch_blocking = true;
  g_override_embedded_ptx = true;
#endif

  if (g_debug_execution >= 40) {
    ptx_debug = 1;
  }
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

unsigned max_cta(const struct gpgpu_ptx_sim_info *kernel_info,
                 unsigned threads_per_cta, unsigned int warp_size,
                 unsigned int n_thread_per_shader,
                 unsigned int gpgpu_shmem_size,
                 unsigned int gpgpu_shader_registers,
                 unsigned int max_cta_per_core) {
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
  unsigned int result_thread = n_thread_per_shader / padded_cta_size;

  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem;
  unsigned int result_regs = (unsigned)-1;
  if (kernel_info->regs > 0)
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));
  printf("padded cta size is %d and %d and %d", padded_cta_size,
         kernel_info->regs, ((kernel_info->regs + 3) & ~3));
  // Limit by CTA
  unsigned int result_cta = max_cta_per_core;

  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
  if (result == result_thread) printf(" threads");
  if (result == result_shmem) printf(" shmem");
  if (result == result_regs) printf(" regs");
  if (result == result_cta) printf(" cta_limit");
  printf("\n");

  return result;
}
/*!
This function simulates the CUDA code functionally, it takes a kernel_info_t
parameter which holds the data for the CUDA kernel to be executed
!*/
void cuda_sim::gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel,
                                            bool openCL) {
  printf(
      "GPGPU-Sim: Performing Functional Simulation, executing kernel %s...\n",
      kernel.name().c_str());

  // using a shader core object for book keeping, it is not needed but as most
  // function built for performance simulation need it we use it here
  // extern gpgpu_sim *g_the_gpu;
  // before we execute, we should do PDOM analysis for functional simulation
  // scenario.
  function_info *kernel_func_info = kernel.entry();
  const struct gpgpu_ptx_sim_info *kernel_info =
      ptx_sim_kernel_info(kernel_func_info);
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();

  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kernel.name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kernel.name().c_str());
    kernel_func_info->do_pdom();
    kernel_func_info->set_pdom();
  }

  unsigned max_cta_tot = max_cta(
      kernel_info, kernel.threads_per_cta(),
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->n_thread_per_shader,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shmem_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shader_registers,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->max_cta_per_core);
  printf("Max CTA : %d\n", max_cta_tot);

  int cp_op = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_option;
  int cp_kernel = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_kernel;
  cp_count = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_insn_Y;
  cp_cta_resume = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_CTA_t;
  int cta_launched = 0;

  // we excute the kernel one CTA (Block) at the time, as synchronization
  // functions work block wise
  while (!kernel.no_more_ctas_to_run()) {
    unsigned temp = kernel.get_next_cta_id_single();

    if (cp_op == 0 ||
        (cp_op == 1 && cta_launched < cp_cta_resume &&
         kernel.get_uid() == cp_kernel) ||
        kernel.get_uid() < cp_kernel)  // just fro testing
    {
      functionalCoreSim cta(
          &kernel, gpgpu_ctx->the_gpgpusim->g_the_gpu,
          gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size);
      cta.execute(cp_count, temp);

#if (CUDART_VERSION >= 5000)
      gpgpu_ctx->device_runtime->launch_all_device_kernels();
#endif
    } else {
      kernel.increment_cta_id();
    }
    cta_launched++;
  }

  if (cp_op == 1) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             kernel.get_uid());
    g_checkpoint->store_global_mem(
        gpgpu_ctx->the_gpgpusim->g_the_gpu->get_global_memory(), f1name,
        (char *)"%08x");
  }

  // registering this kernel as done

  // openCL kernel simulation calls don't register the kernel so we don't
  // register its exit
  if (!openCL) {
    // extern stream_manager *g_stream_manager;
    gpgpu_ctx->the_gpgpusim->g_stream_manager->register_finished_kernel(
        kernel.get_uid());
  }

  //******PRINTING*******
  printf("GPGPU-Sim: Done functional simulation (%u instructions simulated).\n",
         g_ptx_sim_num_insn);
  if (gpgpu_ptx_instruction_classification) {
    StatDisp(g_inst_classification_stat[g_ptx_kernel_count]);
    StatDisp(g_inst_op_classification_stat[g_ptx_kernel_count]);
  }

  // time_t variables used to calculate the total simulation time
  // the start time of simulation is hold by the global variable
  // g_simulation_starttime g_simulation_starttime is initilized by
  // gpgpu_ptx_sim_init_perf() in gpgpusim_entrypoint.cc upon starting gpgpu-sim
  time_t end_time, elapsed_time, days, hrs, minutes, sec;
  end_time = time((time_t *)NULL);
  elapsed_time =
      MAX(end_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);

  // calculating and printing simulation time in terms of days, hours, minutes
  // and seconds
  days = elapsed_time / (3600 * 24);
  hrs = elapsed_time / 3600 - 24 * days;
  minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
  sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)days, (unsigned)hrs, (unsigned)minutes, (unsigned)sec,
      (unsigned)elapsed_time);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(g_ptx_sim_num_insn / elapsed_time));
  fflush(stdout);
}

void functionalCoreSim::initializeCTA(unsigned ctaid_cp) {
  int ctaLiveThreads = 0;
  symbol_table *symtab = m_kernel->entry()->get_symtab();

  for (int i = 0; i < m_warp_count; i++) {
    m_warpAtBarrier[i] = false;
    m_liveThreadCount[i] = 0;
  }
  for (int i = 0; i < m_warp_count * m_warp_size; i++) m_thread[i] = NULL;

  // get threads for a cta
  for (unsigned i = 0; i < m_kernel->threads_per_cta(); i++) {
    ptx_sim_init_thread(*m_kernel, &m_thread[i], 0, i,
                        m_kernel->threads_per_cta() - i,
                        m_kernel->threads_per_cta(), this, 0, i / m_warp_size,
                        (gpgpu_t *)m_gpu, true);
    assert(m_thread[i] != NULL && !m_thread[i]->is_done());
    char fname[2048];
    snprintf(fname, 2048, "checkpoint_files/thread_%d_0_reg.txt", i);
    if (m_gpu->gpgpu_ctx->func_sim->cp_cta_resume == 1)
      m_thread[i]->resume_reg_thread(fname, symtab);
    ctaLiveThreads++;
  }

  for (int k = 0; k < m_warp_count; k++) createWarp(k);
}

void functionalCoreSim::createWarp(unsigned warpId) {
  simt_mask_t initialMask;
  unsigned liveThreadsCount = 0;
  initialMask.set();
  for (int i = warpId * m_warp_size; i < warpId * m_warp_size + m_warp_size;
       i++) {
    if (m_thread[i] == NULL)
      initialMask.reset(i - warpId * m_warp_size);
    else
      liveThreadsCount++;
  }

  assert(m_thread[warpId * m_warp_size] != NULL);
  m_simt_stack[warpId]->launch(m_thread[warpId * m_warp_size]->get_pc(),
                               initialMask);
  char fname[2048];
  snprintf(fname, 2048, "checkpoint_files/warp_%d_0_simt.txt", warpId);

  if (m_gpu->gpgpu_ctx->func_sim->cp_cta_resume == 1) {
    unsigned pc, rpc;
    m_simt_stack[warpId]->resume(fname);
    m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
    for (int i = warpId * m_warp_size; i < warpId * m_warp_size + m_warp_size;
         i++) {
      m_thread[i]->set_npc(pc);
      m_thread[i]->update_pc();
    }
  }
  m_liveThreadCount[warpId] = liveThreadsCount;
}

void functionalCoreSim::execute(int inst_count, unsigned ctaid_cp) {
  m_gpu->gpgpu_ctx->func_sim->cp_count = m_gpu->checkpoint_insn_Y;
  m_gpu->gpgpu_ctx->func_sim->cp_cta_resume = m_gpu->checkpoint_CTA_t;
  initializeCTA(ctaid_cp);

  int count = 0;
  while (true) {
    bool someOneLive = false;
    bool allAtBarrier = true;
    for (unsigned i = 0; i < m_warp_count; i++) {
      executeWarp(i, allAtBarrier, someOneLive);
      count++;
    }

    if (inst_count > 0 && count > inst_count &&
        (m_kernel->get_uid() == m_gpu->checkpoint_kernel) &&
        (ctaid_cp >= m_gpu->checkpoint_CTA) &&
        (ctaid_cp < m_gpu->checkpoint_CTA_t) && m_gpu->checkpoint_option == 1) {
      someOneLive = false;
      break;
    }
    if (!someOneLive) break;
    if (allAtBarrier) {
      for (unsigned i = 0; i < m_warp_count; i++) m_warpAtBarrier[i] = false;
    }
  }

  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();

  ptx_reg_t regval;
  regval.u64 = 123;

  unsigned ctaid = m_kernel->get_next_cta_id_single();
  if (m_gpu->checkpoint_option == 1 &&
      (m_kernel->get_uid() == m_gpu->checkpoint_kernel) &&
      (ctaid_cp >= m_gpu->checkpoint_CTA) &&
      (ctaid_cp < m_gpu->checkpoint_CTA_t)) {
    char fname[2048];
    snprintf(fname, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid - 1);
    g_checkpoint->store_global_mem(m_thread[0]->m_shared_mem, fname,
                                   (char *)"%08x");
    for (int i = 0; i < 32 * m_warp_count; i++) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt", i,
               ctaid - 1);
      m_thread[i]->print_reg_thread(fname);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i, ctaid - 1);
      g_checkpoint->store_global_mem(m_thread[i]->m_local_mem, f1name,
                                     (char *)"%08x");
      m_thread[i]->set_done();
      m_thread[i]->exitCore();
      m_thread[i]->registerExit();
    }

    for (int i = 0; i < m_warp_count; i++) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt", i,
               ctaid - 1);
      FILE *fp = fopen(fname, "w");
      assert(fp != NULL);
      m_simt_stack[i]->print_checkpoint(fp);
      fclose(fp);
    }
  }
}

void functionalCoreSim::executeWarp(unsigned i, bool &allAtBarrier,
                                    bool &someOneLive) {
  if (!m_warpAtBarrier[i] && m_liveThreadCount[i] != 0) {
    warp_inst_t inst = getExecuteWarp(i);
    execute_warp_inst_t(inst, i);
    if (inst.isatomic()) inst.do_atomic(true);
    if (inst.op == BARRIER_OP || inst.op == MEMORY_BARRIER_OP)
      m_warpAtBarrier[i] = true;
    updateSIMTStack(i, &inst);
  }
  if (m_liveThreadCount[i] > 0) someOneLive = true;
  if (!m_warpAtBarrier[i] && m_liveThreadCount[i] > 0) allAtBarrier = false;
}

unsigned gpgpu_context::translate_pc_to_ptxlineno(unsigned pc) {
  // this function assumes that the kernel fits inside a single PTX file
  // function_info *pFunc = g_func_info; // assume that the current kernel is
  // the one in query
  const ptx_instruction *pInsn = pc_to_instruction(pc);
  unsigned ptx_line_number = pInsn->source_line();

  return ptx_line_number;
}

// ptxinfo parser

extern std::map<unsigned, const char *> get_duplicate();

static char *g_ptxinfo_kname = NULL;
static struct gpgpu_ptx_sim_info g_ptxinfo;
static std::map<unsigned, const char *> g_duplicate;
static const char *g_last_dup_type;

const char *get_ptxinfo_kname() { return g_ptxinfo_kname; }

void print_ptxinfo() {
  if (!get_ptxinfo_kname()) {
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
  }
  if (get_ptxinfo_kname()) {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        get_ptxinfo_kname(), g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
  }
}

struct gpgpu_ptx_sim_info get_ptxinfo() {
  return g_ptxinfo;
}

std::map<unsigned, const char *> get_duplicate() { return g_duplicate; }

void ptxinfo_linenum(unsigned linenum) {
  g_duplicate[linenum] = g_last_dup_type;
}

void ptxinfo_dup_type(const char *dup_type) { g_last_dup_type = dup_type; }

void ptxinfo_function(const char *fname) {
  clear_ptxinfo();
  g_ptxinfo_kname = strdup(fname);
}

void ptxinfo_regs(unsigned nregs) { g_ptxinfo.regs = nregs; }

void ptxinfo_lmem(unsigned declared, unsigned system) {
  g_ptxinfo.lmem = declared + system;
}

void ptxinfo_gmem(unsigned declared, unsigned system) {
  g_ptxinfo.gmem = declared + system;
}

void ptxinfo_smem(unsigned declared, unsigned system) {
  g_ptxinfo.smem = declared + system;
}

void ptxinfo_cmem(unsigned nbytes, unsigned bank) { g_ptxinfo.cmem += nbytes; }

void clear_ptxinfo() {
  free(g_ptxinfo_kname);
  g_ptxinfo_kname = NULL;
  g_ptxinfo.regs = 0;
  g_ptxinfo.lmem = 0;
  g_ptxinfo.smem = 0;
  g_ptxinfo.cmem = 0;
  g_ptxinfo.gmem = 0;
  g_ptxinfo.ptx_version = 0;
  g_ptxinfo.sm_target = 0;
}

void ptxinfo_opencl_addinfo(std::map<std::string, function_info *> &kernels) {
  if (!g_ptxinfo_kname) {
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
    clear_ptxinfo();
    return;
  }

  if (!strcmp("__cuda_dummy_entry__", g_ptxinfo_kname)) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  std::map<std::string, function_info *>::iterator k =
      kernels.find(g_ptxinfo_kname);
  if (k == kernels.end()) {
    printf("GPGPU-Sim PTX: ERROR ** implementation for '%s' not found.\n",
           g_ptxinfo_kname);
    abort();
  } else {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        g_ptxinfo_kname, g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
    function_info *finfo = k->second;
    assert(finfo != NULL);
    finfo->set_kernel_info(g_ptxinfo);
  }
  clear_ptxinfo();
}

struct rec_pts cuda_sim::find_reconvergence_points(function_info *finfo) {
  rec_pts tmp;
  std::map<function_info *, rec_pts>::iterator r = g_rpts.find(finfo);

  if (r == g_rpts.end()) {
    int num_recon = finfo->get_num_reconvergence_pairs();

    gpgpu_recon_t *kernel_recon_points =
        (struct gpgpu_recon_t *)calloc(num_recon, sizeof(struct gpgpu_recon_t));
    finfo->get_reconvergence_pairs(kernel_recon_points);
    printf("GPGPU-Sim PTX: reconvergence points for %s...\n",
           finfo->get_name().c_str());
    for (int i = 0; i < num_recon; i++) {
      printf("GPGPU-Sim PTX: %2u (potential) branch divergence @ ", i + 1);
      kernel_recon_points[i].source_inst->print_insn();
      printf("\n");
      printf("GPGPU-Sim PTX:    immediate post dominator      @ ");
      if (kernel_recon_points[i].target_inst)
        kernel_recon_points[i].target_inst->print_insn();
      printf("\n");
    }
    printf("GPGPU-Sim PTX: ... end of reconvergence points for %s\n",
           finfo->get_name().c_str());

    tmp.s_kernel_recon_points = kernel_recon_points;
    tmp.s_num_recon = num_recon;
    g_rpts[finfo] = tmp;
  } else {
    tmp = r->second;
  }
  return tmp;
}

address_type get_return_pc(void *thd) {
  // function call return
  ptx_thread_info *the_thread = (ptx_thread_info *)thd;
  assert(the_thread != NULL);
  return the_thread->get_return_PC();
}

address_type cuda_sim::get_converge_point(address_type pc) {
  // the branch could encode the reconvergence point and/or a bit that indicates
  // the reconvergence point is the return PC on the call stack in the case the
  // branch has no immediate postdominator in the function (i.e., due to
  // multiple return points).

  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  assert(f != g_pc_to_finfo.end());
  function_info *finfo = f->second;
  rec_pts tmp = find_reconvergence_points(finfo);

  int i = 0;
  for (; i < tmp.s_num_recon; ++i) {
    if (tmp.s_kernel_recon_points[i].source_pc == pc) {
      if (tmp.s_kernel_recon_points[i].target_pc == (unsigned)-2) {
        return RECONVERGE_RETURN_PC;
      } else {
        return tmp.s_kernel_recon_points[i].target_pc;
      }
    }
  }
  return NO_BRANCH_DIVERGENCE;
}

void functionalCoreSim::warp_exit(unsigned warp_id) {
  for (int i = 0; i < m_warp_count * m_warp_size; i++) {
    if (m_thread[i] != NULL) {
      m_thread[i]->m_cta_info->register_deleted_thread(m_thread[i]);
      delete m_thread[i];
    }
  }
}
