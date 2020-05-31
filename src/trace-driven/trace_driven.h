// developed by Mahmoud Khairy, Purdue Univ

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef TRACE_DRIVEN_H
#define TRACE_DRIVEN_H

#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/shader.h"
#include "ISA_Def/trace_opcode.h"

class trace_function_info : public function_info {
 public:
  trace_function_info(const struct gpgpu_ptx_sim_info& info,
                      gpgpu_context* m_gpgpu_context)
      : function_info(0, m_gpgpu_context) {
    m_kernel_info = info;
  }

  virtual const struct gpgpu_ptx_sim_info* get_kernel_info() const {
    return &m_kernel_info;
  }

  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info& info) {
    m_kernel_info = info;
  }

  virtual ~trace_function_info() {}
};

class trace_warp_inst_t : public warp_inst_t {
 public:
  trace_warp_inst_t() {
    m_gpgpu_context = NULL;
    m_opcode = 0;
    m_tconfig = NULL;
    should_do_atomic = false;
  }

  trace_warp_inst_t(const class core_config* config,
                    gpgpu_context* gpgpu_context, class trace_config* tconfig)
      : warp_inst_t(config) {
    m_gpgpu_context = gpgpu_context;
    m_opcode = 0;
    m_tconfig = tconfig;
    should_do_atomic = false;
  }

  bool parse_from_string(
      std::string trace,
      const std::unordered_map<std::string, OpcodeChar>* OpcodeMap);

 private:
  gpgpu_context* m_gpgpu_context;
  class trace_config* m_tconfig;
  unsigned m_opcode;
  bool check_opcode_contain(const std::vector<std::string>& opcode,
                            std::string param);
  unsigned get_datawidth_from_opcode(const std::vector<std::string>& opcode);
};

class trace_kernel_info_t : public kernel_info_t {
 public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim, unsigned m_binary_verion,
                      trace_function_info* m_function_info,
                      std::ifstream* inputstream, gpgpu_sim* gpgpu_sim,
                      gpgpu_context* gpgpu_context, class trace_config* config);

  bool get_next_threadblock_traces(
      std::vector<std::vector<trace_warp_inst_t>*> threadblock_traces);

 private:
  std::ifstream* ifs;
  gpgpu_sim* m_gpgpu_sim;
  gpgpu_context* m_gpgpu_context;
  trace_config* m_tconfig;
  unsigned binary_verion;
  const std::unordered_map<std::string, OpcodeChar>* OpcodeMap;
};

class trace_config {
 public:
  trace_config();

  void set_latency(unsigned category, unsigned& latency,
                   unsigned& initiation_interval);
  void parse_config();
  void reg_options(option_parser_t opp);
  char* get_traces_filename() { return g_traces_filename; }

 private:
  unsigned int_latency, fp_latency, dp_latency, sfu_latency, tensor_latency;
  unsigned int_init, fp_init, dp_init, sfu_init, tensor_init;
  unsigned specialized_unit_latency[SPECIALIZED_UNIT_NUM];
  unsigned specialized_unit_initiation[SPECIALIZED_UNIT_NUM];

  char* g_traces_filename;
  char* trace_opcode_latency_initiation_int;
  char* trace_opcode_latency_initiation_sp;
  char* trace_opcode_latency_initiation_dp;
  char* trace_opcode_latency_initiation_sfu;
  char* trace_opcode_latency_initiation_tensor;
  char* trace_opcode_latency_initiation_specialized_op[SPECIALIZED_UNIT_NUM];
};

class trace_parser {
 public:
  trace_parser(const char* kernellist_filepath, gpgpu_sim* m_gpgpu_sim,
               gpgpu_context* m_gpgpu_context);

  std::vector<std::string> parse_kernellist_file();
  trace_kernel_info_t* parse_kernel_info(
      const std::string& kerneltraces_filepath, trace_config* config);
  void parse_memcpy_info(const std::string& memcpy_command, size_t& add,
                         size_t& count);

  void kernel_finalizer(trace_kernel_info_t* kernel_info);

 private:
  std::string kernellist_filename;
  std::ifstream ifs;
  gpgpu_sim* m_gpgpu_sim;
  gpgpu_context* m_gpgpu_context;
};

class trace_shd_warp_t : public shd_warp_t {
 public:
  trace_shd_warp_t(class shader_core_ctx* shader, unsigned warp_size)
      : shd_warp_t(shader, warp_size) {
    trace_pc = 0;
  }

  std::vector<trace_warp_inst_t> warp_traces;
  const trace_warp_inst_t* get_next_trace_inst();
  void clear();
  bool trace_done();
  address_type get_start_trace_pc();
  virtual address_type get_pc();

 private:
  unsigned trace_pc;
};

class trace_gpgpu_sim : public gpgpu_sim {
 public:
  trace_gpgpu_sim(const gpgpu_sim_config& config, gpgpu_context* ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

class trace_simt_core_cluster : public simt_core_cluster {
 public:
  trace_simt_core_cluster(class gpgpu_sim* gpu, unsigned cluster_id,
                          const shader_core_config* config,
                          const memory_config* mem_config,
                          class shader_core_stats* stats,
                          class memory_stats_t* mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

class trace_shader_core_ctx : public shader_core_ctx {
 public:
  trace_shader_core_ctx(class gpgpu_sim* gpu, class simt_core_cluster* cluster,
                        unsigned shader_id, unsigned tpc_id,
                        const shader_core_config* config,
                        const memory_config* mem_config,
                        shader_core_stats* stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t& inst, unsigned t,
                                             unsigned tid);
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t& kernel);
  virtual void func_exec_inst(warp_inst_t& inst);
  virtual unsigned sim_init_thread(kernel_info_t& kernel,
                                   ptx_thread_info** thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t* core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t* gpu);
  virtual void create_shd_warp();
  virtual const warp_inst_t* get_next_inst(unsigned warp_id, address_type pc);
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t* inst);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t* pI,
                                       unsigned* pc, unsigned* rpc);
  virtual const active_mask_t& get_active_mask(unsigned warp_id,
                                               const warp_inst_t* pI);

 private:
  void init_traces(unsigned start_warp, unsigned end_warp,
                   kernel_info_t& kernel);
};

#endif
