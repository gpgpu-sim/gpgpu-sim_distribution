//developed by Mahmoud Khairy, Purdue Univ

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifndef TRACE_DRIVEN_H
#define TRACE_DRIVEN_H

#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/shader.h"

class trace_function_info: public function_info {
public:
	trace_function_info(const struct gpgpu_ptx_sim_info &info, gpgpu_context* m_gpgpu_context):function_info(0, m_gpgpu_context ) {
		m_kernel_info = info;
	}

	virtual const struct gpgpu_ptx_sim_info* get_kernel_info () const {
		return &m_kernel_info;
	}

	virtual const void set_kernel_info (const struct gpgpu_ptx_sim_info &info) {
		m_kernel_info = info;
	}

private:


};

class trace_warp_inst_t: public warp_inst_t {
public:

	trace_warp_inst_t() {

	}
	trace_warp_inst_t(const class core_config *config):warp_inst_t(config) {
	}

	bool parse_from_string(std::string trace);


private:


};

class trace_kernel_info_t: public kernel_info_t {
public:
	trace_kernel_info_t(dim3 gridDim, dim3 blockDim, trace_function_info* m_function_info, std::ifstream* inputstream, gpgpu_sim * gpgpu_sim):kernel_info_t(gridDim, blockDim, m_function_info) {
		ifs = inputstream;
		m_gpgpu_sim = gpgpu_sim;
	}

	bool get_next_threadblock_traces(std::vector<std::vector<trace_warp_inst_t>>& threadblock_traces);

private:
	std::ifstream* ifs;
	gpgpu_sim * m_gpgpu_sim;

};



class trace_parser {
public:
	trace_parser(const char* kernellist_filepath, gpgpu_sim * m_gpgpu_sim, gpgpu_context* m_gpgpu_context);

	std::vector<std::string> parse_kernellist_file();
	trace_kernel_info_t* parse_kernel_info(const std::string& kerneltraces_filepath);
	void kernel_finalizer(trace_kernel_info_t* kernel_info);

private:

	std::string kernellist_filename;
	std::ifstream ifs;
	gpgpu_sim * m_gpgpu_sim;
	gpgpu_context* m_gpgpu_context;

};

class trace_shd_warp_t: public shd_warp_t {
public:
	trace_shd_warp_t(class shader_core_ctx *shader, unsigned warp_size):shd_warp_t(shader, warp_size) {
	}

	bool get_next_threadblock_traces(std::vector<std::vector<trace_warp_inst_t>>& threadblock_traces);

private:

};

#endif
