#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include <list>
#include <map>
#include <set>
#include "cuda_api_object.h"

class cuobjdumpSection;
class cuobjdumpELFSection;
class cuobjdumpPTXSection;

class gpgpu_context {
    public:
	gpgpu_context() {
	    api = new cuda_runtime_api();
	}
	// global list
	std::list<cuobjdumpSection*> cuobjdumpSectionList;
	//maps sm version number to set of filenames
	std::map<unsigned, std::set<std::string> > version_filename;
	cuda_runtime_api* api;
	// member function list
	void cuobjdumpInit();
	void cuobjdumpParseBinary(unsigned int handle);
	void extract_code_using_cuobjdump();
	std::list<cuobjdumpSection*> pruneSectionList(CUctx_st *context);
	std::list<cuobjdumpSection*> mergeMatchingSections(std::string identifier);
	std::list<cuobjdumpSection*> mergeSections();
	cuobjdumpELFSection* findELFSection(const std::string identifier);
	cuobjdumpPTXSection* findPTXSection(const std::string identifier);
	void extract_ptx_files_using_cuobjdump(CUctx_st *context);
};
#endif /* __gpgpu_context_h__ */
