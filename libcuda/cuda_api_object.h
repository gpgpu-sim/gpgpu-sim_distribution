#ifndef __cuda_api_object_h__
#define __cuda_api_object_h__
class cuobjdumpSection;

class cuda_runtime_api {
    public:
	// global list
	std::list<cuobjdumpSection*> libSectionList;
	// member function list
};
#endif /* __cuda_api_object_h__ */
