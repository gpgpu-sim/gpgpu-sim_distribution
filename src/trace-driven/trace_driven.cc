//developed by Mahmoud Khairy, Purdue Univ
//abdallm@purdue.edu

#include <time.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <bits/stdc++.h>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../../libcuda/gpgpu_context.h"
#include "trace_driven.h"
#include "trace_opcode.h"
#include "volta_opcode.h"
#include "turing_opcode.h"
#include "pascal_opcode.h"
#include "../gpgpusim_entrypoint.h"

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
	unsigned shmem=0, nregs=0, cuda_stream_id=0, kernel_id=0, binary_verion=0;
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
			else if (string1 == "binary" && string2 == "version") {
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
	trace_function_info* function_info = new trace_function_info(info, m_gpgpu_context);
	function_info->set_name(kernel_name.c_str());
	trace_kernel_info_t* kernel_info =  new trace_kernel_info_t(gridDim, blockDim, binary_verion, function_info, &ifs, m_gpgpu_sim, m_gpgpu_context);

	return kernel_info;
}


void trace_parser::kernel_finalizer(trace_kernel_info_t* kernel_info){
	if (ifs.is_open())
		ifs.close();

	delete kernel_info->entry();
	delete kernel_info;
}

const trace_warp_inst_t* trace_shd_warp_t::get_next_inst(){
	if(trace_pc < warp_traces.size())
	{
		return &warp_traces[trace_pc++];
	}
	else
		return NULL;
}

void trace_shd_warp_t::clear() {
	trace_pc=0;
	warp_traces.clear();
}

//functional_done
bool trace_shd_warp_t::trace_done() {
	return trace_pc==(warp_traces.size());
}

address_type trace_shd_warp_t::get_start_pc(){
	assert(warp_traces.size() > 0);
	return warp_traces[0].pc;
}

address_type trace_shd_warp_t::get_pc(){
	assert(warp_traces.size() > 0 );
	assert(trace_pc < warp_traces.size());
	return warp_traces[trace_pc].pc;
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim, unsigned m_binary_verion, trace_function_info* m_function_info, std::ifstream* inputstream, gpgpu_sim * gpgpu_sim, gpgpu_context* gpgpu_context):kernel_info_t(gridDim, blockDim, m_function_info) {
	ifs = inputstream;
	m_gpgpu_sim = gpgpu_sim;
	m_gpgpu_context = gpgpu_context;
	binary_verion = m_binary_verion;

	//resolve the binary version
	if(m_binary_verion == VOLTA_BINART_VERSION)
		OpcodeMap = &Volta_OpcodeMap;
	else if(m_binary_verion == PASCAL_TITANX_BINART_VERSION || m_binary_verion == PASCAL_P100_BINART_VERSION)
		OpcodeMap = &Pascal_OpcodeMap;
	else
		assert(0 && "unsupported binary version");
}

bool trace_kernel_info_t::get_next_threadblock_traces(std::vector<std::vector<trace_warp_inst_t>*> threadblock_traces) {

	for(unsigned i=0; i<threadblock_traces.size(); ++i) {
		threadblock_traces[i]->clear();
	}

	unsigned block_id_x=0, block_id_y=0, block_id_z=0;
	unsigned warp_id=0;
	unsigned insts_num=0;


	bool start_of_tb_stream_found = false;

	while(!ifs->eof()) {
		std::string line;
		std::stringstream ss;
		std::string string1, string2;

		getline(*ifs, line);

		if (line.length() == 0) {
			continue;
		}
		else {
			ss.str(line);
			ss>>string1>>string2;
			if (string1 == "#BEGIN_TB") {
				if(!start_of_tb_stream_found)
				{
					start_of_tb_stream_found=true;
				}
				else
					assert(0 && "Parsing error: thread block start before the previous one finish");
			}
			else if (string1 == "#END_TB") {
				assert(start_of_tb_stream_found);
				break; //end of TB stream
			}
			else if(string1 == "thread" && string2 == "block") {
				assert(start_of_tb_stream_found);
				sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x, &block_id_y, &block_id_z);
				std::cout << line << std::endl;
			}
			else if (string1 == "warp") {
				//the start of new warp stream
				assert(start_of_tb_stream_found);
				sscanf(line.c_str(), "warp = %d", &warp_id);
			}
			else if (string1 == "insts") {
				assert(start_of_tb_stream_found);
				sscanf(line.c_str(), "insts = %d", &insts_num);
				threadblock_traces[warp_id]->reserve(insts_num);
			}
			else {
				assert(start_of_tb_stream_found);
				trace_warp_inst_t inst(m_gpgpu_sim->getShaderCoreConfig(), m_gpgpu_context);
				inst.parse_from_string(line, OpcodeMap);
				threadblock_traces[warp_id]->push_back(inst);
			}
		}
	}

	return true;
}

bool trace_warp_inst_t::check_opcode_contain(const std::vector<std::string>& opcode, std::string param)
{
	for(unsigned i=0; i<opcode.size(); ++i)
		if(opcode[i] == param)
			return true;

	return false;
}

bool is_number(const std::string& s)
{
	std::string::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}

unsigned trace_warp_inst_t::get_datawidth_from_opcode(const std::vector<std::string>& opcode)
{
	for(unsigned i=0; i<opcode.size(); ++i) {
		if(is_number(opcode[i])){
			return (std::stoi(opcode[i],NULL)/8);
		}
		else if(opcode[i][0] == 'U' && is_number(opcode[i].substr(1))){
			//handle the U* case
			unsigned bits;
			sscanf(opcode[i].c_str(), "U%u",&bits);
			return bits/8;
		}
	}

	return 4;  //default is 4 bytes
}

bool trace_warp_inst_t::parse_from_string(std::string trace, const std::unordered_map<std::string,OpcodeChar>* OpcodeMap){

	std::stringstream ss;
	ss.str(trace);

	std::string temp;
	unsigned threadblock_x=0, threadblock_y=0, threadblock_z=0, warpid_tb=0, sm_id=0, warpid_sm=0;
	unsigned long long m_pc=0;
	unsigned mask=0;
	unsigned reg_dest[4];
	std::string opcode;
	unsigned reg_dsts_num=0;
	unsigned reg_srcs_num=0;
	unsigned reg_srcs[4];
	unsigned mem_width=0;
	unsigned long long mem_addresses[warp_size()];
	unsigned address_mode=0;
	unsigned long long base_address=0;
	int stride=0;

	//Start Parsing
	ss>>std::dec>>threadblock_x>>threadblock_y>>threadblock_z>>warpid_tb;

	//ignore core id
	//ss>>std::dec>>sm_id>>warpid_sm;

	ss>>std::hex>>m_pc;
	ss>>std::hex>>mask;

	std::bitset<MAX_WARP_SIZE> mask_bits(mask);

	ss>>std::dec>>reg_dsts_num;
	for(unsigned i=0; i<reg_dsts_num; ++i) {
		ss>>std::dec>>temp;
		sscanf(temp.c_str(), "R%d", &reg_dest[i]);
	}

	ss>>opcode;

	ss>>reg_srcs_num;
	for(unsigned i=0; i<reg_srcs_num; ++i) {
		ss>>temp;
		sscanf(temp.c_str(), "R%d", &reg_srcs[i]);

	}

	ss>>mem_width;

	if(mem_width > 0)  //then it is a memory inst
	{
		ss>>std::dec>>address_mode;
		if(address_mode==0){
			//read addresses one by one from the file
			for (int s = 0; s < warp_size(); s++) {
				if(mask_bits.test(s))
					ss>>std::hex>>mem_addresses[s];
				else
					mem_addresses[s]=0;
			}
		}
		else if(address_mode==1){
			//read addresses as base address and stride
			ss>>std::hex>>base_address;
			ss>>std::dec>>stride;
			bool first_bit1_found=false;
			bool last_bit1_found=false;
			unsigned long long addra=base_address;
			for (int s = 0; s < warp_size(); s++) {
				if(mask_bits.test(s) && !first_bit1_found){
					first_bit1_found=true;
					mem_addresses[s]=base_address;
				} else if(first_bit1_found && !last_bit1_found) {
					if(mask_bits.test(s)) {
						addra += stride;
						mem_addresses[s] = addra;
					} else
						last_bit1_found=true;
				}
				else
					mem_addresses[s]=0;
			}
		}
	}
	//Finish Parsing
	//After parsing, fill the inst_t and warp_inst_t params

	//fill active mask
	active_mask_t active_mask = mask_bits;
	set_active( active_mask );

	//get the opcode
	std::istringstream iss(opcode);
	std::vector<std::string> opcode_tokens;
	std::string token;
	while (std::getline(iss, token, '.')) {
		if (!token.empty())
			opcode_tokens.push_back(token);
	}

	std::string opcode1 = opcode_tokens[0];

	//fill and initialize common params
	m_decoded = true;
	pc = (address_type)m_pc;   //we will lose the high 32 bits from casting long to unsigned, it should be okay!

	isize = 16;   //starting from MAXWELL isize=16 bytes (including the control bytes)
	for(unsigned i=0; i<MAX_OUTPUT_VALUES; i++) {
		out[i] = 0;
	}
	for(unsigned i=0; i<MAX_INPUT_VALUES; i++) {
		in[i] = 0;
	}

	is_vectorin = 0;
	is_vectorout = 0;
	ar1 = 0;
	ar2 = 0;
	memory_op = no_memory_op;
	data_size = 0;
	op = ALU_OP;
	mem_op= NOT_TEX;

	std::unordered_map<std::string,OpcodeChar>::const_iterator it= OpcodeMap->find(opcode1);
	if (it != OpcodeMap->end()) {
		m_opcode = it->second.opcode;
		op = (op_type)(it->second.opcode_category);
	}
	else {
		std::cout<<"ERROR:  undefined instruction : "<<opcode<<" Opcode: "<<opcode1<<std::endl;
		assert(0 && "undefined instruction");
	}

	//fill regs information
	num_regs = reg_srcs_num+reg_dsts_num;
	num_operands = num_regs;
	outcount=reg_dsts_num;
	for(unsigned m=0; m<reg_dsts_num; ++m){
		out[m]=reg_dest[m]+1;         //Increment by one because GPGPU-sim starts from R1, while SASS starts from R0
		arch_reg.dst[m]=reg_dest[m]+1;
	}

	incount=reg_srcs_num;
	for(unsigned m=0; m<reg_srcs_num; ++m){
		in[m]=reg_srcs[m]+1;	     //Increment by one because GPGPU-sim starts from R1, while SASS starts from R0
		arch_reg.src[m]=reg_srcs[m]+1;
	}
	//TO DO: handle: vector, store insts have no output, double inst and hmma, and 64 bit address
	//remove redundant registers

	//fill latency and initl
	set_latency(op);

	//fill addresses
	if(mem_width > 0) {
		for(unsigned i=0; i<warp_size(); ++i)
			set_addr(i, mem_addresses[i]);
	}


	//handle special cases and fill memory space
	switch(m_opcode){
	case OP_LDG:
	case OP_LDL:
		assert(mem_width>0);
		//Nvbit reports incorrect data width, and we have to parse the opcode to get the correct data width
        data_size = get_datawidth_from_opcode(opcode_tokens);
		memory_op = memory_load;
		cache_op = CACHE_ALL;
		if(m_opcode == OP_LDL)
			space.set_type(local_space);
		else
			space.set_type(global_space);
		//check the cache scope, if its strong GPU, then bypass L1
		if(check_opcode_contain( opcode_tokens , "STRONG") && check_opcode_contain( opcode_tokens , "GPU")) {
			cache_op = CACHE_GLOBAL;
		}
		break;
	case OP_STG:
	case OP_STL:
	case OP_ATOMG:
	case OP_RED:
	case OP_ATOM:
		assert(mem_width>0);
        data_size = get_datawidth_from_opcode(opcode_tokens);
		memory_op = memory_store;
		cache_op = CACHE_ALL;
		if(m_opcode == OP_STL)
			space.set_type(local_space);
		else
			space.set_type(global_space);

		if(m_opcode == OP_ATOMG || m_opcode == OP_ATOM || m_opcode == OP_RED){
			m_isatomic = true;
			memory_op = memory_load;
			op=LOAD_OP;
			cache_op = CACHE_GLOBAL;

			//ATOMIC writes to the first operand, we missed that in the trace so we fixed it here. TO be fixed in tracer
			outcount=reg_dsts_num+1;
			out[0]=in[0];         //Increment by one because GPGPU-sim starts from R1, while SASS starts from R0
			arch_reg.dst[0]=reg_srcs[0];
			num_regs = reg_srcs_num+reg_dsts_num+1;
			num_operands = num_regs;

		}

		break;
	case OP_LDS:
	case OP_STS:
	case OP_ATOMS:
		assert(mem_width>0);
		data_size = mem_width;
		space.set_type(shared_space);
		if(m_opcode == OP_ATOMS || m_opcode == OP_LDS) {
			//m_isatomic = true;
			op=LOAD_OP;
			memory_op = memory_load;
		}
		break;
	case OP_ST:
	case OP_LD:
		//TO DO: set generic load based on the address
		//right now, we consider all loads are shared.
		assert(mem_width>0);
		data_size = get_datawidth_from_opcode(opcode_tokens);
		space.set_type(shared_space);
		if(m_opcode == OP_LD)
			memory_op = memory_load;
		else
			memory_op = memory_store;
		break;
	case OP_BAR:
		//TO DO: fill this correctly
		bar_id = 0;
		bar_count = (unsigned)-1;
		bar_type = SYNC;
		//TO DO
		//if bar_type = RED;
		//set bar_type
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
		initiation_interval = initiation_interval/2;   //FP16 has 2X throughput than FP32
		break;
	default:
		break;
	}

	return true;
}

void trace_warp_inst_t::set_latency(unsigned category)
{
	unsigned int_latency[5];
	unsigned fp_latency[5];
	unsigned dp_latency[5];
	unsigned sfu_latency;
	unsigned tensor_latency;
	unsigned int_init[5];
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
	 */
	sscanf(m_gpgpu_context->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u",
			&int_latency[0],&int_latency[1],&int_latency[2],
			&int_latency[3],&int_latency[4]);
	sscanf(m_gpgpu_context->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
			&fp_latency[0],&fp_latency[1],&fp_latency[2],
			&fp_latency[3],&fp_latency[4]);
	sscanf(m_gpgpu_context->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
			&dp_latency[0],&dp_latency[1],&dp_latency[2],
			&dp_latency[3],&dp_latency[4]);
	sscanf(m_gpgpu_context->func_sim->opcode_latency_sfu, "%u",
			&sfu_latency);
	sscanf(m_gpgpu_context->func_sim->opcode_latency_tensor, "%u",
			&tensor_latency);
	sscanf(m_gpgpu_context->func_sim->opcode_initiation_int, "%u,%u,%u,%u,%u",
			&int_init[0],&int_init[1],&int_init[2],
			&int_init[3],&int_init[4]);
	sscanf(m_gpgpu_context->func_sim->opcode_initiation_fp, "%u,%u,%u,%u,%u",
			&fp_init[0],&fp_init[1],&fp_init[2],
			&fp_init[3],&fp_init[4]);
	sscanf(m_gpgpu_context->func_sim->opcode_initiation_dp, "%u,%u,%u,%u,%u",
			&dp_init[0],&dp_init[1],&dp_init[2],
			&dp_init[3],&dp_init[4]);
	sscanf(m_gpgpu_context->func_sim->opcode_initiation_sfu, "%u",
			&sfu_init);
	sscanf(m_gpgpu_context->func_sim->opcode_initiation_tensor, "%u",
			&tensor_init);
	sscanf(m_gpgpu_context->func_sim->cdp_latency_str, "%u,%u,%u,%u,%u",
			&m_gpgpu_context->func_sim->cdp_latency[0],
			&m_gpgpu_context->func_sim->cdp_latency[1],
			&m_gpgpu_context->func_sim->cdp_latency[2],
			&m_gpgpu_context->func_sim->cdp_latency[3],
			&m_gpgpu_context->func_sim->cdp_latency[4]);

	initiation_interval = latency = 1;

	switch(category){
	case ALU_OP:
	case INTP_OP:
	case BRANCH_OP:
	case CALL_OPS:
	case RET_OPS:
		latency = int_latency[0];
		initiation_interval = int_init[0];
		break;
	case SP_OP:
		latency = fp_latency[0];
		initiation_interval = fp_init[0];
		break;
	case DP_OP:
		latency = dp_latency[0];
		initiation_interval = dp_init[0];
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

}

unsigned trace_shader_core_ctx::trace_sim_inc_thread( kernel_info_t &kernel)
{

	if ( kernel.no_more_ctas_to_run() ) {
		return 0; //finished!
	}

	if( kernel.more_threads_in_cta() ) {
		kernel.increment_thread_id();
	}

	if( !kernel.more_threads_in_cta() )
		kernel.increment_cta_id();

	return 1;
}

void trace_shader_core_ctx::init_traces( unsigned start_warp, unsigned end_warp, kernel_info_t &kernel ) {

	std::vector<std::vector<trace_warp_inst_t>*> threadblock_traces;
	for (unsigned i = start_warp; i < end_warp; ++i) {
		m_trace_warp[i].clear();
		threadblock_traces.push_back(&(m_trace_warp[i].warp_traces));
	}
	trace_kernel_info_t& trace_kernel = static_cast<trace_kernel_info_t&> (kernel);
	trace_kernel.get_next_threadblock_traces(threadblock_traces);

	//set pc
	for (unsigned i = start_warp; i < end_warp; ++i) {
		m_warp[i].set_next_pc(m_trace_warp[i].get_start_pc());
	}
}


void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)
{
	if(inst.isatomic())
		m_warp[inst.warp_id()].inc_n_atomic();

	if ( inst.op == EXIT_OPS ) {
		m_warp[inst.warp_id()].set_completed(t);
	}

}

void trace_shader_core_ctx::func_exec_inst( warp_inst_t &inst )
{
	//here, we generate memory acessess and set the status if thread (done?)
	if( inst.is_load() || inst.is_store() )
	{
		inst.generate_mem_accesses();
	}
	for ( unsigned t=0; t < m_warp_size; t++ ) {
		if( inst.active(t) ) {
			unsigned warpId = inst.warp_id();
			unsigned tid=m_warp_size*warpId+t;

			//virtual function
			checkExecutionStatusAndUpdate(inst,t,tid);
		}
	}
	if(m_trace_warp[inst.warp_id()].trace_done() && m_warp[inst.warp_id()].functional_done()) {
		m_warp[inst.warp_id()].ibuffer_flush();
		m_barriers.warp_exit( inst.warp_id() );
	}
}

