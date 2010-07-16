#include <stdlib.h>
#include "common.cu"

extern "C" {
struct QuerySet
{
	  int qfile;

	  char * h_tex_array;
	  char* d_tex_array;	
	  int* d_addrs_tex_array; 
	  int* h_addrs_tex_array;
	  int* h_lengths_array;
	  int* d_lengths_array;
	  char** h_names;

	  unsigned int count; 
	  size_t texlen;

	  // total device memory occupied by this query set
	  size_t bytes_on_board;
};


struct AuxiliaryNodeData
{
	  int length;
	  int depth;
	  int leafid;
      char leafchar;
      TextureAddress parent;
};


struct Reference
{
	  /* Reference string */
	  char* str;
	  size_t len;

	  unsigned int pitch;
  	  void* d_ref_tex_array;  //cudaArray*
 	  char* h_ref_tex_array;
 
	  /* Suffix tree for reference */
	  void* d_node_tex_array;  //really a cudaArray*
	  void* h_node_tex_array;  //really a PixelOfNode*	  
	  
	  void* d_children_tex_array; //cudaArray*
	  void* h_children_tex_array; //PixelOfChildren*	  

	  unsigned int tex_height;
	  unsigned int tex_width;

	  // total device memory occupied by this query set
	  size_t bytes_on_board;

	  AuxiliaryNodeData* aux_data;
	  int num_nodes;

};


// Matches are reported as a node in the suffix tree,
// plus a distance up the node's parent link for partial 
// matches on the patch from the root to the node

struct MatchCoord
{
	  unsigned int node; // match node
	  short edge_match_length;  // number of missing characters UP the parent edge
};

struct MatchResults
{	  
	  // Each MatchCoord in the buffers below corresponds to the first character
	  // of some substring of one of the queries
	  MatchCoord* d_match_coords;
	  MatchCoord* h_match_coords;

	  size_t numCoords;

	  // total device memory occupied by this query set
	  size_t bytes_on_board;
};

typedef unsigned int MUMMERGPU_OPTIONS;

enum MUMMerGPUOption {
   DEFAULT,
   ON_CPU = (1<<0)
};

//All times in milliseconds
struct Statistics
{
	  float t_kernel;
	  float t_output;
      float t_to_board;
      float t_from_board;
	  float t_moving_tree_pages;
	  float t_query_read;
	  float t_total;
	  float t_construction;

	  float bp_avg_query_length;
};

struct MatchContext
{
	  char* full_ref;
	  size_t full_ref_len;

	  Reference* ref;
	  QuerySet* queries;
	  MatchResults results;

	  bool on_cpu;

	  int min_match_length;

      bool reverse;
      bool forwardreverse;
      bool forwardcoordinates;
      bool show_query_length;
      bool maxmatch;

	  char* stats_file;

	  Statistics statistics;
};


struct ReferencePage
{
	  int begin;
	  int end;
	  int shadow_left;
	  int shadow_right;
	  MatchResults results;
	  unsigned int id;
	  Reference ref;
};


int createReference(const char* fromFile, Reference* ref);
int destroyReference(Reference* ref);

int createQuerySet(const char* fromFile, QuerySet* queries);
int destroyQuerySet(QuerySet* queries);

int createMatchContext(Reference* ref,
					   QuerySet* queries,
					   MatchResults* matches,
					   MUMMERGPU_OPTIONS options,
					   int min_match_length,
					   char* stats_file,
                       bool reverse,
                       bool forwardreverse,
                       bool forwardcoordinates,
                       bool showQueryLength,
					   MatchContext* ctx);

int destroyMatchContext(MatchContext* ctx);


int matchQueries(MatchContext* ctx);
				 
void printStringForError(int err);
}
