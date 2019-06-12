#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include "cuda_api_object.h"

class gpgpu_context {
    public:
	gpgpu_context() {
	    api = new cuda_runtime_api();
	}
	// global list
	// objects pointers for each file
	cuda_runtime_api* api;
	// member function list
	void cuobjdumpParseBinary(unsigned int handle);
	class symbol_table *gpgpu_ptx_sim_load_ptx_from_string( const char *p, unsigned source_num );
	class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename( const char *filename );
};
gpgpu_context* GPGPU_Context();

#endif /* __gpgpu_context_h__ */
