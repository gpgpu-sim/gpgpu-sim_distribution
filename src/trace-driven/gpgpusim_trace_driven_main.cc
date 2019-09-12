//developed by Mahmoud Khairy, Purdue Univ
//abdallm@purdue.edu

//#include "../abstract_hardware_model.h"
#include <time.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <math.h>


//#include "../option_parser.h"
//#include "../cuda-sim/cuda-sim.h"
//#include "../cuda-sim/ptx_ir.h"
//#include "../cuda-sim/ptx_parser.h"
#include "../gpgpu-sim/gpu-sim.h"
//#include "../gpgpu-sim/icnt_wrapper.h"
//#include "../gpgpu-sim/icnt_wrapper.h"
#include "../../libcuda/gpgpu_context.h"
#include "trace_driven.h"

//#include "../stream_manager.h"


void arguments_check();

int main ( int argc, const char **argv )
{

	gpgpu_context* m_gpgpu_context = GPGPU_Context();
	gpgpu_sim * m_gpgpu_sim = m_gpgpu_context->gpgpu_trace_sim_init_perf(argc,argv);
	m_gpgpu_sim->init();

	//for each kernel
	//load file
	//parse and create kernel info
	//launch
	//while loop till the end of the end kernel execution
	//prints stats

	trace_parser tracer(m_gpgpu_sim->get_config().get_traces_filename(), m_gpgpu_sim, m_gpgpu_context);

	std::vector<std::string> kernellist = tracer.parse_kernellist_file();

	for(unsigned i=0; i<kernellist.size(); ++i) {

		trace_kernel_info_t* kernel_info  = tracer.parse_kernel_info(kernellist[i]);
		m_gpgpu_sim->launch(kernel_info);

		bool active = false;
		bool sim_cycles = false;
		bool break_limit = false;

		do {
			if(!m_gpgpu_sim->active())
				break;

			//performance simulation
			if( m_gpgpu_sim->active() ) {
				m_gpgpu_sim->cycle();
				sim_cycles = true;
				m_gpgpu_sim->deadlock_check();
			}else {
				if(m_gpgpu_sim->cycle_insn_cta_max_hit()){
					m_gpgpu_context->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
					break_limit = true;
				}
			}

			active=m_gpgpu_sim->active() ;

		} while( active );

		tracer.kernel_finalizer(kernel_info);

		m_gpgpu_sim->print_stats();

		if(sim_cycles) {
			m_gpgpu_sim->update_stats();
			m_gpgpu_context->print_simulation_time();
		}

		if(break_limit) {
			printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **\n");
			fflush(stdout);
			exit(1);
		}
	}

	return 1;
}

trace_parser::trace_parser(const char* kernellist_filepath, gpgpu_sim * m_gpgpu_sim, gpgpu_context* m_gpgpu_context)
{

	this->m_gpgpu_sim = m_gpgpu_sim;
	this->m_gpgpu_context = m_gpgpu_context;
	kernellist_filename = kernellist_filepath;
}

std::vector<std::string> trace_parser::parse_kernellist_file() {

	ifs.open(kernellist_filename);

	if (!ifs.is_open()) {
		std::cout << "Unable to open file: " <<kernellist_filename<<std::endl;
		exit(1);
	}

	std::string directory(kernellist_filename);
	const size_t last_slash_idx = directory.rfind('/');
	if (std::string::npos != last_slash_idx)
	{
		directory = directory.substr(0, last_slash_idx);
	}

	std::string line, filepath;
	std::vector<std::string> kernellist;
	while(!ifs.eof()) {
		getline(ifs, line);
		if(line.empty())
			continue;
		filepath = directory+"/"+line;
		kernellist.push_back(filepath);
	}

	ifs.close();
	return kernellist;
}


trace_kernel_info_t* trace_parser::parse_kernel_info(const std::string& kerneltraces_filepath) {

	ifs.open(kerneltraces_filepath.c_str());

	if (!ifs.is_open()) {
		std::cout << "Unable to open file: " <<kerneltraces_filepath<<std::endl;
		exit(1);
	}

	std::cout << "Processing kernel " <<kerneltraces_filepath<<std::endl;

	unsigned grid_dim_x=0, grid_dim_y=0, grid_dim_z=0, tb_dim_x=0, tb_dim_y=0, tb_dim_z=0;
	unsigned shmem=0, nregs=0, cuda_stream_id=0, kernel_id=0;
	std::string line;
	std::stringstream ss;
	std::string string1, string2;
	std::string  kernel_name;

	while(!ifs.eof()) {
		getline(ifs, line);

		if (line.length() == 0) {
			continue;
		}
		else if(line[0] == '#'){
			//the trace format, ignore this and assume fixed format for now
			break;  //the begin of the instruction stream
		}
		else if(line[0] == '-') {
			ss.str(line);
			ss.ignore();
			ss>>string1>>string2;
			if(string1 == "kernel" && string2 == "name") {
				const size_t equal_idx = line.find('=');
				kernel_name = line.substr(equal_idx+1);
			}
			else if(string1 == "kernel" && string2 == "id") {
				sscanf(line.c_str(), "-kernel id = %d", &kernel_id);
			}
			else if(string1 == "grid" && string2 == "dim") {
				sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &grid_dim_x, &grid_dim_y, &grid_dim_z);
			}
			else if (string1 == "block" && string2 == "dim") {
				sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &tb_dim_x, &tb_dim_y, &tb_dim_z);
			}
			else if (string1 == "shmem") {
				sscanf(line.c_str(), "-shmem = %d", &shmem);
			}
			else if (string1 == "nregs") {
				sscanf(line.c_str(), "-nregs = %d", &nregs);
			}
			else if (string1 == "cuda" && string2 == "stream") {
				sscanf(line.c_str(), "-cuda stream id = %d", &cuda_stream_id);
			}
			continue;
		}
	}

	gpgpu_ptx_sim_info info;
	info.smem = shmem;
	info.regs = nregs;
	dim3 gridDim(grid_dim_x, grid_dim_y, grid_dim_z);
	dim3 blockDim(tb_dim_x, tb_dim_y, tb_dim_z);
	trace_function_info* function_info = new trace_function_info(info, m_gpgpu_context);
	function_info->set_name(kernel_name.c_str());
	trace_kernel_info_t* kernel_info =  new trace_kernel_info_t(gridDim, blockDim, function_info, &ifs, m_gpgpu_sim);

	return kernel_info;
}


void trace_parser::kernel_finalizer(trace_kernel_info_t* kernel_info){
	if (ifs.is_open())
		ifs.close();

	delete kernel_info->entry();
	delete kernel_info;
}

bool trace_kernel_info_t::get_next_threadblock_traces(std::vector<std::vector<trace_warp_inst_t>>& threadblock_traces) {

	threadblock_traces.clear();
	unsigned warps_per_tb = ceil(float(threads_per_cta()/32));
	threadblock_traces.resize(warps_per_tb);

	unsigned block_id_x=0, block_id_y=0, block_id_z=0;
	unsigned warp_id=0;
	unsigned insts_num=0;
	std::string line;
	std::stringstream ss;
	std::string string1, string2;

	bool start_of_tb_stream_found = false;

	while(!ifs->eof()) {
		getline(*ifs, line);

		if (line.length() == 0) {
			continue;
		}
		else {
			ss.str(line);
			ss>>string1>>string2;
			if (string1 == "#BEGIN_TB") {
				if(!start_of_tb_stream_found)
					start_of_tb_stream_found=true;
				else assert(0 && "Parsing error: thread block start before the previous one finish");
			}
			else if (string1 == "#END_TB") {
				assert(start_of_tb_stream_found);
				break; //end of TB stream
			}
			else if(string1 == "thread" && string2 == "block") {
				assert(start_of_tb_stream_found);
				sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x, &block_id_y, &block_id_z);
			}
			else if (string1 == "warp") {
				//the start of new warp stream
				assert(start_of_tb_stream_found);
				sscanf(line.c_str(), "warp = %d", &warp_id);
			}
			else if (string1 == "insts") {
				assert(start_of_tb_stream_found);
				sscanf(line.c_str(), "insts = %d", &insts_num);
				threadblock_traces[warp_id].resize(insts_num);
			}
			else {
				assert(start_of_tb_stream_found);
				trace_warp_inst_t inst(m_gpgpu_sim->getShaderCoreConfig());
				inst.parse_from_string(line);
				threadblock_traces[warp_id].push_back(inst);
			}
		}
	}

	return true;
}


bool trace_warp_inst_t::parse_from_string(std::string trace){

	std::stringstream ss;
	ss.str(trace);


	std::string temp;
	unsigned threadblock_x=0, threadblock_y=0, threadblock_z=0, warpid_tb=0, sm_id=0, warpid_sm=0;
	unsigned long long m_pc=0;
	unsigned mask=0;
	unsigned reg_dest=0;
	std::string opcode;
	unsigned reg_srcs_num=0;
	unsigned reg_srcs[4];
	unsigned mem_width=0;
	unsigned long long mem_addresses[warp_size()];

	ss>>std::dec>>threadblock_x>>threadblock_y>>threadblock_z>>warpid_tb>>sm_id>>warpid_sm;

	ss>>std::hex>>m_pc>>mask;
	std::bitset<MAX_WARP_SIZE> mask_bits(mask);

	ss>>std::dec>>temp;
	sscanf(temp.c_str(), "R%d", &reg_dest);

	ss>>opcode;
	ss>>reg_srcs_num;

	for(unsigned i=0; i<reg_srcs_num; ++i) {
		ss>>temp;
		sscanf(temp.c_str(), "R%d", &reg_srcs[i]);
	}

	ss>>mem_width;
	if(mem_width > 0)  //then it is a memory inst
	{
		for (int s = 0; s < warp_size(); s++) {
			if(mask_bits.test(s))
				ss>>std::hex>>mem_addresses[s];
			else
				mem_addresses[s]=0;
		}
	}

	//After parsing, fill the inst_t and warp_inst_t params
	active_mask_t active_mask = mask_bits;
	set_active( active_mask );

	for(unsigned i=0; i<warp_size(); ++i)
		set_addr(i, mem_addresses[i]);

	std::string opcode1 = opcode.substr(0, opcode.find("."));

	m_decoded = true;
	pc = m_pc;
	isize = 16;   //TO DO, change this
	for(unsigned i=0; i<MAX_OUTPUT_VALUES; i++) {
		out[i] = 0;
	}
	for(unsigned i=0; i<MAX_INPUT_VALUES; i++) {
		in[i] = 0;
	}
	incount=0;
	outcount=0;
	is_vectorin = 0;
	is_vectorout = 0;

	/*
	switch(opcode1){
	case "MOV":
		incount=1;
		break;
	default:
		std::cout<<"unknown instruction: "<<opcode;
		break;
	}
	*/

	return true;
}


