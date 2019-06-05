#include <list>

class cuobjdumpSection;
class cuobjdumpELFSection;
class cuobjdumpPTXSection;

class gpgpu_context {
    public:
	std::list<cuobjdumpSection*> cuobjdumpSectionList;
	void cuobjdumpInit();
	void cuobjdumpParseBinary(unsigned int handle);
	void extract_code_using_cuobjdump();
	std::list<cuobjdumpSection*> pruneSectionList(CUctx_st *context);
	std::list<cuobjdumpSection*> mergeMatchingSections(std::string identifier);
	std::list<cuobjdumpSection*> mergeSections();
	cuobjdumpELFSection* findELFSection(const std::string identifier, std::list<cuobjdumpSection*> &libSectionList);
	cuobjdumpPTXSection* findPTXSection(const std::string identifier, std::list<cuobjdumpSection*> &libSectionList);
};
