#ifndef __cuda_api_object_h__
#define __cuda_api_object_h__
class cuobjdumpSection;
class kernel_config;


class cuda_runtime_api {
    public:
	// global list
	std::list<cuobjdumpSection*> libSectionList;
	std::list<kernel_config> g_cuda_launch_stack;
	// member function list
};
#endif /* __cuda_api_object_h__ */
