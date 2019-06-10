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
};
#endif /* __gpgpu_context_h__ */
