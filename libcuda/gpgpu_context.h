#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include "cuda_api_object.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpusim_entrypoint.h"
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/cuda_device_runtime.h"

class gpgpu_context {
    public:
	gpgpu_context() {
	    g_global_allfiles_symbol_table = NULL;
	    sm_next_access_uid=0;
	    warp_inst_sm_next_uid=0;
	    api = new cuda_runtime_api(this);
	    ptxinfo = new ptxinfo_data(this);
	    ptx_parser = new ptx_recognizer(this);
	    the_gpgpusim = new GPGPUsim_ctx(this);
	    func_sim = new cuda_sim(this);
	    device_runtime = new cuda_device_runtime(this);
	}
	// global list
	symbol_table *g_global_allfiles_symbol_table;
	const char *g_filename;
	unsigned sm_next_access_uid;
	unsigned warp_inst_sm_next_uid;
	// objects pointers for each file
	cuda_runtime_api* api;
	ptxinfo_data* ptxinfo;
	ptx_recognizer* ptx_parser;
	GPGPUsim_ctx* the_gpgpusim;
	cuda_sim* func_sim;
	cuda_device_runtime* device_runtime;
	// member function list
	void cuobjdumpParseBinary(unsigned int handle);
	class symbol_table *gpgpu_ptx_sim_load_ptx_from_string( const char *p, unsigned source_num );
	class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename( const char *filename );
	void gpgpu_ptx_info_load_from_filename( const char *filename, unsigned sm_version);
	void gpgpu_ptxinfo_load_from_string( const char *p_for_info, unsigned source_num, unsigned sm_version=20, int no_of_ptx=0 );
	void print_ptx_file( const char *p, unsigned source_num, const char *filename );
	class symbol_table* init_parser(const char*);
	class gpgpu_sim *gpgpu_ptx_sim_init_perf();
	struct _cuda_device_id *GPGPUSim_Init();
	void ptx_reg_options(option_parser_t opp);
};
gpgpu_context* GPGPU_Context();

#endif /* __gpgpu_context_h__ */
