// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpusim_entrypoint.h"
#include "../option_parser.h"
#include "ISA_Def/kepler_opcode.h"
#include "ISA_Def/pascal_opcode.h"
#include "ISA_Def/trace_opcode.h"
#include "ISA_Def/turing_opcode.h"
#include "ISA_Def/volta_opcode.h"
#include "trace_driven.h"

bool is_number(const std::string& s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

void split(const std::string& str, std::vector<std::string>& cont,
           char delimi = ' ') {
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimi)) {
    cont.push_back(token);
  }
}

trace_parser::trace_parser(const char* kernellist_filepath,
                           gpgpu_sim* m_gpgpu_sim,
                           gpgpu_context* m_gpgpu_context) {
  this->m_gpgpu_sim = m_gpgpu_sim;
  this->m_gpgpu_context = m_gpgpu_context;
  kernellist_filename = kernellist_filepath;
}

std::vector<std::string> trace_parser::parse_kernellist_file() {
  ifs.open(kernellist_filename);

  if (!ifs.is_open()) {
    std::cout << "Unable to open file: " << kernellist_filename << std::endl;
    exit(1);
  }

  std::string directory(kernellist_filename);
  const size_t last_slash_idx = directory.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = directory.substr(0, last_slash_idx);
  }

  std::string line, filepath;
  std::vector<std::string> kernellist;
  while (!ifs.eof()) {
    getline(ifs, line);
    if (line.empty())
      continue;
    else if (line.substr(0, 6) == "Memcpy") {
      kernellist.push_back(line);
    } else if (line.substr(0, 6) == "kernel") {
      filepath = directory + "/" + line;
      kernellist.push_back(filepath);
    }
  }

  ifs.close();
  return kernellist;
}

void trace_parser::parse_memcpy_info(const std::string& memcpy_command,
                                     size_t& address, size_t& count) {
  std::vector<std::string> params;
  split(memcpy_command, params, ',');
  assert(params.size() == 3);
  std::stringstream ss;
  ss.str(params[1]);
  ss >> std::hex >> address;
  ss.clear();
  ss.str(params[2]);
  ss >> std::dec >> count;
}

trace_kernel_info_t* trace_parser::parse_kernel_info(
    const std::string& kerneltraces_filepath, trace_config* config) {
  ifs.open(kerneltraces_filepath.c_str());

  if (!ifs.is_open()) {
    std::cout << "Unable to open file: " << kerneltraces_filepath << std::endl;
    exit(1);
  }

  std::cout << "Processing kernel " << kerneltraces_filepath << std::endl;

  unsigned grid_dim_x = 0, grid_dim_y = 0, grid_dim_z = 0, tb_dim_x = 0,
           tb_dim_y = 0, tb_dim_z = 0;
  unsigned shmem = 0, nregs = 0, cuda_stream_id = 0, kernel_id = 0,
           binary_verion = 0;
  std::string line;
  std::stringstream ss;
  std::string string1, string2;
  std::string kernel_name;

  while (!ifs.eof()) {
    getline(ifs, line);

    if (line.length() == 0) {
      continue;
    } else if (line[0] == '#') {
      // the trace format, ignore this and assume fixed format for now
      break;  // the begin of the instruction stream
    } else if (line[0] == '-') {
      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;
      if (string1 == "kernel" && string2 == "name") {
        const size_t equal_idx = line.find('=');
        kernel_name = line.substr(equal_idx + 1);
      } else if (string1 == "kernel" && string2 == "id") {
        sscanf(line.c_str(), "-kernel id = %d", &kernel_id);
      } else if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &grid_dim_x, &grid_dim_y,
               &grid_dim_z);
      } else if (string1 == "block" && string2 == "dim") {
        sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &tb_dim_x, &tb_dim_y,
               &tb_dim_z);
      } else if (string1 == "shmem") {
        sscanf(line.c_str(), "-shmem = %d", &shmem);
      } else if (string1 == "nregs") {
        sscanf(line.c_str(), "-nregs = %d", &nregs);
      } else if (string1 == "cuda" && string2 == "stream") {
        sscanf(line.c_str(), "-cuda stream id = %d", &cuda_stream_id);
      } else if (string1 == "binary" && string2 == "version") {
        sscanf(line.c_str(), "-binary version = %d", &binary_verion);
      }
      std::cout << line << std::endl;
      continue;
    }
  }

  gpgpu_ptx_sim_info info;
  info.smem = shmem;
  info.regs = nregs;
  dim3 gridDim(grid_dim_x, grid_dim_y, grid_dim_z);
  dim3 blockDim(tb_dim_x, tb_dim_y, tb_dim_z);
  trace_function_info* function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_name.c_str());
  trace_kernel_info_t* kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, binary_verion, function_info,
                              &ifs, m_gpgpu_sim, m_gpgpu_context, config);

  return kernel_info;
}

void trace_parser::kernel_finalizer(trace_kernel_info_t* kernel_info) {
  if (ifs.is_open()) ifs.close();

  delete kernel_info->entry();
  delete kernel_info;
}

const trace_warp_inst_t* trace_shd_warp_t::get_next_trace_inst() {
  if (trace_pc < warp_traces.size()) {
    return &warp_traces[trace_pc++];
  } else
    return NULL;
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  warp_traces.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() { return trace_pc == (warp_traces.size()); }

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].pc;
}

address_type trace_shd_warp_t::get_pc() {
  assert(warp_traces.size() > 0);
  assert(trace_pc < warp_traces.size());
  return warp_traces[trace_pc].pc;
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         unsigned m_binary_verion,
                                         trace_function_info* m_function_info,
                                         std::ifstream* inputstream,
                                         gpgpu_sim* gpgpu_sim,
                                         gpgpu_context* gpgpu_context,
                                         class trace_config* config)
    : kernel_info_t(gridDim, blockDim, m_function_info) {
  ifs = inputstream;
  m_gpgpu_sim = gpgpu_sim;
  m_gpgpu_context = gpgpu_context;
  binary_verion = m_binary_verion;
  m_tconfig = config;

  // resolve the binary version
  if (m_binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (m_binary_verion == PASCAL_TITANX_BINART_VERSION ||
           m_binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (m_binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (m_binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else
    assert(0 && "unsupported binary version");
}

bool trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::vector<trace_warp_inst_t>*> threadblock_traces) {
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  unsigned block_id_x = 0, block_id_y = 0, block_id_z = 0;
  unsigned warp_id = 0;
  unsigned insts_num = 0;

  bool start_of_tb_stream_found = false;

  while (!ifs->eof()) {
    std::string line;
    std::stringstream ss;
    std::string string1, string2;

    getline(*ifs, line);

    if (line.length() == 0) {
      continue;
    } else {
      ss.str(line);
      ss >> string1 >> string2;
      if (string1 == "#BEGIN_TB") {
        if (!start_of_tb_stream_found) {
          start_of_tb_stream_found = true;
        } else
          assert(0 &&
                 "Parsing error: thread block start before the previous one "
                 "finish");
      } else if (string1 == "#END_TB") {
        assert(start_of_tb_stream_found);
        break;  // end of TB stream
      } else if (string1 == "thread" && string2 == "block") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x,
               &block_id_y, &block_id_z);
        std::cout << line << std::endl;
      } else if (string1 == "warp") {
        // the start of new warp stream
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "warp = %d", &warp_id);
      } else if (string1 == "insts") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "insts = %d", &insts_num);
        threadblock_traces[warp_id]->reserve(insts_num);
      } else {
        assert(start_of_tb_stream_found);
        trace_warp_inst_t inst(m_gpgpu_sim->getShaderCoreConfig(),
                               m_gpgpu_context, m_tconfig);
        inst.parse_from_string(line, OpcodeMap);
        threadblock_traces[warp_id]->push_back(inst);
      }
    }
  }

  return true;
}

bool trace_warp_inst_t::check_opcode_contain(
    const std::vector<std::string>& opcode, std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param) return true;

  return false;
}

unsigned trace_warp_inst_t::get_datawidth_from_opcode(
    const std::vector<std::string>& opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4;  // default is 4 bytes
}

bool trace_warp_inst_t::parse_from_string(
    std::string trace,
    const std::unordered_map<std::string, OpcodeChar>* OpcodeMap) {
  std::stringstream ss;
  ss.str(trace);

  std::string temp;
  unsigned threadblock_x = 0, threadblock_y = 0, threadblock_z = 0,
           warpid_tb = 0, sm_id = 0, warpid_sm = 0;
  unsigned long long m_pc = 0;
  unsigned mask = 0;
  unsigned reg_dest[4];
  std::string opcode;
  unsigned reg_dsts_num = 0;
  unsigned reg_srcs_num = 0;
  unsigned reg_srcs[4];
  unsigned mem_width = 0;
  unsigned long long mem_addresses[warp_size()];
  unsigned address_mode = 0;
  unsigned long long base_address = 0;
  int stride = 0;

  // Start Parsing
  ss >> std::dec >> threadblock_x >> threadblock_y >> threadblock_z >>
      warpid_tb;

  // ignore core id
  // ss>>std::dec>>sm_id>>warpid_sm;

  ss >> std::hex >> m_pc;
  ss >> std::hex >> mask;

  std::bitset<MAX_WARP_SIZE> mask_bits(mask);

  ss >> std::dec >> reg_dsts_num;
  for (unsigned i = 0; i < reg_dsts_num; ++i) {
    ss >> std::dec >> temp;
    sscanf(temp.c_str(), "R%d", &reg_dest[i]);
  }

  ss >> opcode;

  ss >> reg_srcs_num;
  for (unsigned i = 0; i < reg_srcs_num; ++i) {
    ss >> temp;
    sscanf(temp.c_str(), "R%d", &reg_srcs[i]);
  }

  ss >> mem_width;

  if (mem_width > 0)  // then it is a memory inst
  {
    ss >> std::dec >> address_mode;
    if (address_mode == 0) {
      // read addresses one by one from the file
      for (int s = 0; s < warp_size(); s++) {
        if (mask_bits.test(s))
          ss >> std::hex >> mem_addresses[s];
        else
          mem_addresses[s] = 0;
      }
    } else if (address_mode == 1) {
      // read addresses as base address and stride
      ss >> std::hex >> base_address;
      ss >> std::dec >> stride;
      bool first_bit1_found = false;
      bool last_bit1_found = false;
      unsigned long long addra = base_address;
      for (int s = 0; s < warp_size(); s++) {
        if (mask_bits.test(s) && !first_bit1_found) {
          first_bit1_found = true;
          mem_addresses[s] = base_address;
        } else if (first_bit1_found && !last_bit1_found) {
          if (mask_bits.test(s)) {
            addra += stride;
            mem_addresses[s] = addra;
          } else
            last_bit1_found = true;
        } else
          mem_addresses[s] = 0;
      }
    }
  }
  // Finish Parsing
  // After parsing, fill the inst_t and warp_inst_t params

  // fill active mask
  active_mask_t active_mask = mask_bits;
  set_active(active_mask);

  // get the opcode
  std::istringstream iss(opcode);
  std::vector<std::string> opcode_tokens;
  std::string token;
  while (std::getline(iss, token, '.')) {
    if (!token.empty()) opcode_tokens.push_back(token);
  }

  std::string opcode1 = opcode_tokens[0];

  // fill and initialize common params
  m_decoded = true;
  pc = (address_type)m_pc;  // we will lose the high 32 bits from casting long
                            // to unsigned, it should be okay!

  isize =
      16;  // starting from MAXWELL isize=16 bytes (including the control bytes)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }

  is_vectorin = 0;
  is_vectorout = 0;
  ar1 = 0;
  ar2 = 0;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  mem_op = NOT_TEX;

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);
  if (it != OpcodeMap->end()) {
    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
  } else {
    std::cout << "ERROR:  undefined instruction : " << opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }

  // fill regs information
  num_regs = reg_srcs_num + reg_dsts_num;
  num_operands = num_regs;
  outcount = reg_dsts_num;
  for (unsigned m = 0; m < reg_dsts_num; ++m) {
    out[m] = reg_dest[m] + 1;  // Increment by one because GPGPU-sim starts from
                               // R1, while SASS starts from R0
    arch_reg.dst[m] = reg_dest[m] + 1;
  }

  incount = reg_srcs_num;
  for (unsigned m = 0; m < reg_srcs_num; ++m) {
    in[m] = reg_srcs[m] + 1;  // Increment by one because GPGPU-sim starts from
                              // R1, while SASS starts from R0
    arch_reg.src[m] = reg_srcs[m] + 1;
  }
  // TO DO: handle: vector, store insts have no output, double inst and hmma,
  // and 64 bit address remove redundant registers

  // fill latency and initl
  m_tconfig->set_latency(op, latency, initiation_interval);

  // fill addresses
  if (mem_width > 0) {
    for (unsigned i = 0; i < warp_size(); ++i) set_addr(i, mem_addresses[i]);
  }

  // handle special cases and fill memory space
  switch (m_opcode) {
    case OP_LDG:
    case OP_LDL:
      assert(mem_width > 0);
      // Nvbit reports incorrect data width, and we have to parse the opcode to
      // get the correct data width
      data_size = get_datawidth_from_opcode(opcode_tokens);
      memory_op = memory_load;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_LDL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      // check the cache scope, if its strong GPU, then bypass L1
      if (check_opcode_contain(opcode_tokens, "STRONG") &&
          check_opcode_contain(opcode_tokens, "GPU")) {
        cache_op = CACHE_GLOBAL;
      }
      break;
    case OP_STG:
    case OP_STL:
    case OP_ATOMG:
    case OP_RED:
    case OP_ATOM:
      assert(mem_width > 0);
      data_size = get_datawidth_from_opcode(opcode_tokens);
      memory_op = memory_store;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_STL)
        space.set_type(local_space);
      else
        space.set_type(global_space);

      if (m_opcode == OP_ATOMG || m_opcode == OP_ATOM || m_opcode == OP_RED) {
        m_isatomic = true;
        memory_op = memory_load;
        op = LOAD_OP;
        cache_op = CACHE_GLOBAL;

        // ATOMIC writes to the first operand, we missed that in the trace so we
        // fixed it here. TO be fixed in tracer
        outcount = reg_dsts_num + 1;
        out[0] = in[0];  // Increment by one because GPGPU-sim starts from R1,
                         // while SASS starts from R0
        arch_reg.dst[0] = reg_srcs[0];
        num_regs = reg_srcs_num + reg_dsts_num + 1;
        num_operands = num_regs;
      }

      break;
    case OP_LDS:
    case OP_STS:
    case OP_ATOMS:
      assert(mem_width > 0);
      data_size = mem_width;
      space.set_type(shared_space);
      if (m_opcode == OP_ATOMS || m_opcode == OP_LDS) {
        // m_isatomic = true;
        op = LOAD_OP;
        memory_op = memory_load;
      }
      break;
    case OP_ST:
    case OP_LD:
      // TO DO: set generic load based on the address
      // right now, we consider all loads are shared.
      assert(mem_width > 0);
      data_size = get_datawidth_from_opcode(opcode_tokens);
      space.set_type(shared_space);
      if (m_opcode == OP_LD)
        memory_op = memory_load;
      else
        memory_op = memory_store;
      break;
    case OP_BAR:
      // TO DO: fill this correctly
      bar_id = 0;
      bar_count = (unsigned)-1;
      bar_type = SYNC;
      // TO DO
      // if bar_type = RED;
      // set bar_type
      // barrier_type bar_type;
      // reduction_type red_type;
      break;
    case OP_HADD2:
    case OP_HADD2_32I:
    case OP_HFMA2:
    case OP_HFMA2_32I:
    case OP_HMUL2_32I:
    case OP_HSET2:
    case OP_HSETP2:
      initiation_interval =
          initiation_interval / 2;  // FP16 has 2X throughput than FP32
      break;
    default:
      break;
  }

  return true;
}

trace_config::trace_config() {}

void trace_config::reg_options(option_parser_t opp) {
  option_parser_register(opp, "-trace", OPT_CSTR, &g_traces_filename,
                         "traces kernel file"
                         "traces kernel file directory",
                         "./traces/kernelslist.g");

  option_parser_register(opp, "-trace_opcode_latency_initiation_int", OPT_CSTR,
                         &trace_opcode_latency_initiation_int,
                         "Opcode latencies and initiation for integers in "
                         "trace driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sp", OPT_CSTR,
                         &trace_opcode_latency_initiation_sp,
                         "Opcode latencies and initiation for sp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_dp", OPT_CSTR,
                         &trace_opcode_latency_initiation_dp,
                         "Opcode latencies and initiation for dp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sfu", OPT_CSTR,
                         &trace_opcode_latency_initiation_sfu,
                         "Opcode latencies and initiation for sfu in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_tensor",
                         OPT_CSTR, &trace_opcode_latency_initiation_tensor,
                         "Opcode latencies and initiation for tensor in trace "
                         "driven mode <latency,initiation>",
                         "4,1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-trace_opcode_latency_initiation_spec_op_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &trace_opcode_latency_initiation_specialized_op[j],
                           "specialized unit config"
                           " <latency,initiation>",
                           "4,4");
  }
}

void trace_config::parse_config() {
  sscanf(trace_opcode_latency_initiation_int, "%u,%u", &int_latency, &int_init);
  sscanf(trace_opcode_latency_initiation_sp, "%u,%u", &fp_latency, &fp_init);
  sscanf(trace_opcode_latency_initiation_dp, "%u,%u", &dp_latency, &dp_init);
  sscanf(trace_opcode_latency_initiation_sfu, "%u,%u", &sfu_latency, &sfu_init);
  sscanf(trace_opcode_latency_initiation_tensor, "%u,%u", &tensor_latency,
         &tensor_init);

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    sscanf(trace_opcode_latency_initiation_specialized_op[j], "%u,%u",
           &specialized_unit_latency[j], &specialized_unit_initiation[j]);
  }
}
void trace_config::set_latency(unsigned category, unsigned& latency,
                               unsigned& initiation_interval) {
  initiation_interval = latency = 1;

  switch (category) {
    case ALU_OP:
    case INTP_OP:
    case BRANCH_OP:
    case CALL_OPS:
    case RET_OPS:
      latency = int_latency;
      initiation_interval = int_init;
      break;
    case SP_OP:
      latency = fp_latency;
      initiation_interval = fp_init;
      break;
    case DP_OP:
      latency = dp_latency;
      initiation_interval = dp_init;
      break;
    case SFU_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      break;
    case TENSOR_CORE_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      break;
    default:
      break;
  }
  // for specialized units
  if (category >= SPEC_UNIT_START_ID) {
    unsigned spec_id = category - SPEC_UNIT_START_ID;
    assert(spec_id >= 0 && spec_id < SPECIALIZED_UNIT_NUM);
    latency = specialized_unit_latency[spec_id];
    initiation_interval = specialized_unit_initiation[spec_id];
  }
}

void trace_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new trace_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                    m_shader_stats, m_memory_stats);
}

void trace_simt_core_cluster::create_shader_core_ctx() {
  m_core = new shader_core_ctx*[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}

void trace_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new trace_shd_warp_t(this, m_config->warp_size);
  }
}

void trace_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                    const warp_inst_t* pI,
                                                    unsigned* pc,
                                                    unsigned* rpc) {
  // In trace-driven mode, we assume no control hazard
  *pc = pI->pc;
  *rpc = pI->pc;
}

const active_mask_t& trace_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t* pI) {
  // For Trace-driven, the active mask already set in traces, so
  // just read it from the inst
  return pI->get_active_mask();
}

unsigned trace_shader_core_ctx::sim_init_thread(
    kernel_info_t& kernel, ptx_thread_info** thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t* core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t* gpu) {
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (kernel.more_threads_in_cta()) {
    kernel.increment_thread_id();
  }

  if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

  return 1;
}

void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size, kernel_info_t& kernel) {
  // call base class
  shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size,
                              kernel);

  // then init traces
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  init_traces(start_warp, end_warp, kernel);
}

const warp_inst_t* trace_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                        address_type pc) {
  // read the inst from the traces
  trace_shd_warp_t* m_trace_warp =
      static_cast<trace_shd_warp_t*>(m_warp[warp_id]);
  return m_trace_warp->get_next_trace_inst();
}

void trace_shader_core_ctx::updateSIMTStack(unsigned warpId,
                                            warp_inst_t* inst) {
  // No SIMT-stack in trace-driven  mode
}

void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        kernel_info_t& kernel) {
  std::vector<std::vector<trace_warp_inst_t>*> threadblock_traces;
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t* m_trace_warp = static_cast<trace_shd_warp_t*>(m_warp[i]);
    m_trace_warp->clear();
    threadblock_traces.push_back(&(m_trace_warp->warp_traces));
  }
  trace_kernel_info_t& trace_kernel = static_cast<trace_kernel_info_t&>(kernel);
  trace_kernel.get_next_threadblock_traces(threadblock_traces);

  // set the pc from the traces and ignore the functional model
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t* m_trace_warp = static_cast<trace_shd_warp_t*>(m_warp[i]);
    m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
  }
}

void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t& inst,
                                                          unsigned t,
                                                          unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();

  if (inst.op == EXIT_OPS) {
    m_warp[inst.warp_id()]->set_completed(t);
  }
}

void trace_shader_core_ctx::func_exec_inst(warp_inst_t& inst) {
  // here, we generate memory acessess and set the status if thread (done?)
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
  }
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (inst.active(t)) {
      unsigned warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }
  trace_shd_warp_t* m_trace_warp =
      static_cast<trace_shd_warp_t*>(m_warp[inst.warp_id()]);
  if (m_trace_warp->trace_done() && m_trace_warp->functional_done()) {
    m_trace_warp->ibuffer_flush();
    m_barriers.warp_exit(inst.warp_id());
  }
}
