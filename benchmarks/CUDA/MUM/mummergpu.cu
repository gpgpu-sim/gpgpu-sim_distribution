// Includes, system
#define ulong4 uint4
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>

// includes, kernels
#include <common.cu>

#include <mummergpu.h>
#include <mummergpu_kernel.cu>

#define BLOCKSIZE 256

#define CUDA_SAFE_CALL( call) do {                                           \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold(MatchResults* results, 
				 char* refstr, 
				 char* queries, 
				 int* queryAddrs,
				 int* queryLengths,
				 PixelOfNode* nodeTexture,
				 PixelOfChildren* childrenTexture,
				 int numQueries,
				 int mismatch_length,
				 int rc); 
 
extern "C"
void getReferenceString(const char * filename, char** refstr, size_t* reflen);

extern "C"
void createTreeTexture(const char * filename,
                       PixelOfNode** nodeTexture, PixelOfChildren** childrenTexture,
                       unsigned int* width, unsigned int* height,
					   AuxiliaryNodeData** aux_data,
					   int* num_nodes,
					   const char * dotfilename,
                       const char * texfilename);

extern "C"
void getQueriesTexture(int qfile,
                       char** queryTexture, 
                       size_t* queryLength, 
                       int** queryAddrs, 
					   char*** queryNames,
                       int** queryLengths,
					   unsigned int* numQueries,
					   size_t device_memory_avail,
					   int min_match_length,
					   bool rc);

void printAlignments(char* ref, 
                     ReferencePage* page, 
                     char* query, 
                     int qrylen,
                     int nodeid, 
                     int qrypos, 
                     int edge_depth, 
                     int min_match, 
                     bool rc,
                     bool forwardcoordinates);

int  countLeafNodes(int nodeid);        

// Timer management
struct Timer_t
{
  struct timeval start_m;
  struct timeval end_m;
};

void createTimer(unsigned int * timer)
{
  unsigned int * ptr = (unsigned int *) malloc(sizeof(struct Timer_t));
  memset(ptr, 0, sizeof(struct Timer_t));

  *timer = (unsigned int)(unsigned long long) ptr;
}

void startTimer(unsigned int ptr)
{
  gettimeofday(&(((struct Timer_t *)ptr)->start_m), NULL);
}

void stopTimer(unsigned int ptr)
{
  gettimeofday(&(((struct Timer_t *)ptr)->end_m), NULL);
}

float getTimerValue(unsigned int ptr)
{
  Timer_t * timer = (Timer_t*) ptr;

  if (timer == NULL)
  {
    fprintf(stderr, "Uninitialized timer!!!\n");
    return 0.0;
  }

  if (timer->end_m.tv_sec == 0) { stopTimer(ptr); }

  return  (float) (1000.0 * (timer->end_m.tv_sec - timer->start_m.tv_sec) 
                + (0.001 *  (timer->end_m.tv_usec - timer->start_m.tv_usec)));
}

void deleteTimer(unsigned int ptr)
{
  free((Timer_t *)ptr);
}

extern "C"
int createReference(const char* fromFile, Reference* ref)
{
   if (!fromFile || !ref)
	  return -1;
   
   getReferenceString(fromFile, &(ref->str), &(ref->len));
  
   return 0;
}

extern "C"
int destroyReference(Reference* ref)
{
   free(ref->h_node_tex_array);
   free(ref->h_children_tex_array);
   free(ref->str);
   free(ref->h_ref_tex_array);
   free(ref->aux_data);
   ref->str = NULL;
   ref->len = 0;

   return 0;
}

extern "C"
int createQuerySet(const char* fromFile, QuerySet* queries)
{

   fprintf(stderr, "Opening %s...\n", fromFile);
   int qfile = open(fromFile, O_RDONLY);
   
   if (qfile == -1)
   {
	  fprintf(stderr, "Can't open %s: %d\n", fromFile, errno);
	  exit (1);
   }

   queries->qfile = qfile;

   return 0;
}

extern "C"
int destroyQuerySet(QuerySet* queries)
{
 
   if (queries->qfile)
	  close(queries->qfile);

   return 0;
}

extern "C"
void printStringForError(int err)
{
   
}

extern "C"
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
					   MatchContext* ctx)
{
   
   ctx->queries = queries;
   ctx->ref = ref;
   ctx->full_ref = ref->str;
   ctx->full_ref_len = ref->len;

   // break out options here
   ctx->on_cpu = options & ON_CPU;
   ctx->min_match_length = min_match_length;
   ctx->stats_file = stats_file;
   ctx->reverse = reverse;
   ctx->forwardreverse = forwardreverse;
   ctx->forwardcoordinates = forwardcoordinates;
   ctx->show_query_length = showQueryLength;
   return 0;
}

extern "C"
int destroyMatchContext(MatchContext* ctx)
{
   free(ctx->full_ref);
   //destroyReference(ctx->ref);
   destroyQuerySet(ctx->queries);
   return 0;
}

void buildReferenceTexture(Reference* ref, char* full_ref, size_t begin, size_t end)
{
   fprintf(stderr, "Building reference texture...\n");
   


   PixelOfNode* nodeTexture = NULL;
   PixelOfChildren * childrenTexture = NULL;
   
   unsigned int height = 0;
   unsigned int width = 0;

   AuxiliaryNodeData* aux_data = NULL;
   int num_nodes;

   ref->len = end - begin + 3;
   ref->str = (char*)malloc(ref->len);
   ref->str[0] = 's';
   strncpy(ref->str + 1, full_ref + begin, ref->len - 3);
   strcpy(ref->str + ref->len - 2, "$");
   createTreeTexture(ref->str, 
					 &nodeTexture, 
					 &childrenTexture, 
					 &width, &height,
					 &aux_data,
					 &num_nodes,
					 NULL,
					 NULL);
   
   ref->h_node_tex_array = nodeTexture;
   ref->h_children_tex_array = childrenTexture;
   ref->tex_width = width;
   ref->tex_height = height;
  
   ref->aux_data = aux_data;
   ref->num_nodes = num_nodes;

   ref->bytes_on_board = width * height * (sizeof(PixelOfNode) + sizeof(PixelOfChildren));

   unsigned int refpitch = ref->pitch = 65536;
   int numrows = ceil(ref->len / ((float)refpitch));
   int blocksize = 4;
   numrows += blocksize;

   ref->h_ref_tex_array = (char *) malloc(numrows*refpitch);
   
   ref->bytes_on_board += numrows*refpitch;

   int z_max = numrows * refpitch;
   for (int z = 0; z < z_max; z++) { ref->h_ref_tex_array[z] = 'Z'; }

   int x, y;
   int maxx = 0, maxy = 0;
       
   size_t reflen = ref->len;
   char* refstr = ref->str;

   
   int block_dim = refpitch * blocksize;
   for (int i = 0; i < reflen; i++)
   {
	  int bigx = i % (block_dim);
	  int bigy = i / (block_dim);

	  y = bigy*blocksize+bigx%blocksize;
	  x = bigx / blocksize;

	  //   printf("%d: (%d,%d)=%c\n", i, x, y, refstr[i]);

	  assert(x < refpitch);
	  assert(y < numrows);

	  ref->h_ref_tex_array[y*refpitch+x] = refstr[i];

	  if (x > maxx) { maxx = x; }
	  if (y > maxy) { maxy = y; }
   }

   if ((maxx >= refpitch) || (maxy >= numrows))
   {
	  fprintf(stderr, "ERROR: maxx: %d refpitch: %d, maxy: %d numrows: %d\n",
			  maxx,    refpitch,     maxy,    numrows);

	  exit(1);
   }

}

void loadReferenceTexture(MatchContext* ctx)
{
   Reference* ref = ctx->ref;
   int numrows = ceil(ref->len / ((float)ref->pitch));
   int blocksize = 4;
   numrows += blocksize;

   cudaChannelFormatDesc refTextureDesc = 
	  cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned);
     
   if (!ctx->on_cpu)
   {
      unsigned int toboardtimer = 0;
      createTimer(&toboardtimer);
      startTimer(toboardtimer);

	  fprintf(stderr, "allocating reference texture\n");
	  CUDA_SAFE_CALL(cudaMallocArray( (cudaArray**)(&ref->d_ref_tex_array), 
									  &refTextureDesc, 
									  ref->pitch, 
									  numrows)); 
	
	  //ref->bytes_on_board += ref->pitch * numrows;

	  CUDA_SAFE_CALL(cudaMemcpyToArray( (cudaArray*)(ref->d_ref_tex_array), 
										0, 
										0, 
										ref->h_ref_tex_array,
										numrows*ref->pitch, 
										cudaMemcpyHostToDevice));

	  reftex.addressMode[0] = cudaAddressModeClamp;
	  reftex.addressMode[1] = cudaAddressModeClamp;
	  reftex.filterMode = cudaFilterModePoint;
	  reftex.normalized = false;
      
	  CUDA_SAFE_CALL(cudaBindTextureToArray( reftex, (cudaArray*)ref->d_ref_tex_array, refTextureDesc));

      stopTimer(toboardtimer);
      ctx->statistics.t_moving_tree_pages += getTimerValue(toboardtimer);
      deleteTimer(toboardtimer);
   }
   else
   {
	  ref->d_ref_tex_array = NULL;
   }

   fprintf(stderr,"done\n");

}

void unloadReferenceTexture(Reference* ref)
{
   CUDA_SAFE_CALL(cudaUnbindTexture( reftex ) );
   CUDA_SAFE_CALL(cudaFreeArray((cudaArray*)(ref->d_ref_tex_array)));
   ref->d_ref_tex_array = NULL;
}


//loads a tree and text for [begin, end) in the reference
void loadReference(MatchContext* ctx)
{
   
   Reference* ref = ctx->ref;

   //ref->bytes_on_board = 0;

   loadReferenceTexture(ctx);
  
   if (!ctx->on_cpu)
   {
      unsigned int toboardtimer = 0;
      createTimer(&toboardtimer);
      startTimer(toboardtimer);

	  cudaChannelFormatDesc nodeTextureDesc = 
		 cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
	  
	  CUDA_SAFE_CALL( cudaMallocArray( (cudaArray**)(&ref->d_node_tex_array), 
									   &nodeTextureDesc, 
									   ref->tex_width, 
									   ref->tex_height ));
 
	  //ref->bytes_on_board += ref->tex_width * ref->tex_height * (sizeof(PixelOfNode));
	  
	  CUDA_SAFE_CALL( cudaMemcpyToArray( (cudaArray*)(ref->d_node_tex_array), 
										 0, 
										 0, 
										 ref->h_node_tex_array,
										 ref->tex_width * ref->tex_height * sizeof(PixelOfNode), 
										 cudaMemcpyHostToDevice));

	  nodetex.addressMode[0] = cudaAddressModeClamp;
	  nodetex.addressMode[1] = cudaAddressModeClamp;
	  nodetex.filterMode = cudaFilterModePoint;
	  nodetex.normalized = false;    // access with normalized texture coordinates

	  CUDA_SAFE_CALL( cudaBindTextureToArray( nodetex, 
									   (cudaArray*)ref->d_node_tex_array, 
									   nodeTextureDesc));

	  cudaChannelFormatDesc childrenTextureDesc = 
		 cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);

	  CUDA_SAFE_CALL( cudaMallocArray( (cudaArray**)(&ref->d_children_tex_array), 
									   &childrenTextureDesc, 
									   ref->tex_width, 
									   ref->tex_height ));
 
	  //ref->bytes_on_board += ref->tex_width * ref->tex_height * sizeof(PixelOfNode);

	  CUDA_SAFE_CALL( cudaMemcpyToArray((cudaArray*)(ref->d_children_tex_array), 
										0, 
										0, 
										ref->h_children_tex_array, 
										ref->tex_width * ref->tex_height * sizeof(PixelOfChildren), 
										cudaMemcpyHostToDevice));

	  childrentex.addressMode[0] = cudaAddressModeClamp;
	  childrentex.addressMode[1] = cudaAddressModeClamp;
	  childrentex.filterMode = cudaFilterModePoint;
	  childrentex.normalized = false;    // access with normalized texture coordinates

	  CUDA_SAFE_CALL( cudaBindTextureToArray( childrentex, 
									   (cudaArray*)(ref->d_children_tex_array), 
									   childrenTextureDesc));
	  fprintf(stderr, "done\n");

      stopTimer(toboardtimer);
      ctx->statistics.t_moving_tree_pages += getTimerValue(toboardtimer);
      deleteTimer(toboardtimer);
   }
   else
   {
	  ref->d_node_tex_array = NULL;
	  ref->d_children_tex_array = NULL;
   }
}

void unloadReference(MatchContext* ctx)
{
   Reference* ref = ctx->ref;

   CUDA_SAFE_CALL(cudaUnbindTexture( nodetex ) );
   CUDA_SAFE_CALL(cudaFreeArray((cudaArray*)(ref->d_node_tex_array)));
   ref->d_node_tex_array = NULL;

   CUDA_SAFE_CALL(cudaUnbindTexture( childrentex ) );
   CUDA_SAFE_CALL(cudaFreeArray((cudaArray*)(ref->d_children_tex_array)));
   ref->d_children_tex_array = NULL;

   unloadReferenceTexture(ctx->ref);
}



void loadQueries(MatchContext* ctx)
{
   unsigned int toboardtimer = 0;
   createTimer(&toboardtimer);
   startTimer(toboardtimer);

   QuerySet* queries = ctx->queries;
   queries->bytes_on_board = 0;
   
   unsigned int numQueries = queries->count;
	  
   if (!ctx->on_cpu)
   {
	  fprintf(stderr, "loadQueries on GPU: Allocating device memory for queries...\n");
   
	  CUDA_SAFE_CALL( cudaMalloc((void**) &queries->d_tex_array, queries->texlen));

	  queries->bytes_on_board += queries->texlen;

	  CUDA_SAFE_CALL( cudaMemcpy((void*) queries->d_tex_array, 
								 queries->h_tex_array + queries->h_addrs_tex_array[0], 
								 queries->texlen, 
								 cudaMemcpyHostToDevice));

	  CUDA_SAFE_CALL( cudaMalloc((void**) &queries->d_addrs_tex_array, 
								 numQueries * sizeof(int)));

	  queries->bytes_on_board += numQueries * sizeof(int);

	  CUDA_SAFE_CALL( cudaMemcpy((void*) queries->d_addrs_tex_array, 
								 queries->h_addrs_tex_array, 
								 numQueries * sizeof(int), 
								 cudaMemcpyHostToDevice));

	  CUDA_SAFE_CALL( cudaMalloc((void**) &queries->d_lengths_array, 
								 numQueries * sizeof(int)));

	  queries->bytes_on_board += numQueries * sizeof(int);

	  CUDA_SAFE_CALL( cudaMemcpy((void*) queries->d_lengths_array, 
								 queries->h_lengths_array, 
								 numQueries * sizeof(int), 
								 cudaMemcpyHostToDevice));

	  fprintf(stderr, "loadQueries on GPU: allocated %ld bytes done\n", 2 * numQueries*sizeof(int) + queries->texlen);
   }
   else
   {
	  queries->d_addrs_tex_array = NULL;
	  queries->d_tex_array = NULL;
	  queries->d_lengths_array = NULL;
	  fprintf(stderr, "loadQueries on CPU: allocated %ld bytes done\n", numQueries*sizeof(int) + queries->texlen);
   }

   stopTimer(toboardtimer);
   ctx->statistics.t_to_board += getTimerValue(toboardtimer);
   deleteTimer(toboardtimer);
}

void unloadQueries(MatchContext* ctx)
{
   QuerySet* queries = ctx->queries;

   CUDA_SAFE_CALL(cudaFree(queries->d_tex_array));
   queries->d_tex_array = NULL;

   CUDA_SAFE_CALL(cudaFree(queries->d_addrs_tex_array));
   queries->d_addrs_tex_array = NULL;

   CUDA_SAFE_CALL(cudaFree(queries->d_lengths_array));
   queries->d_lengths_array = NULL;

   queries->bytes_on_board = 0;
}

void loadResultBuffer(MatchContext* ctx)
{
   unsigned int numQueries = ctx->queries->count; 
  
   assert (numQueries);

   int match_length = ctx->min_match_length;

   unsigned int numCoords = 0;

   numCoords = ctx->queries->texlen - numQueries * (match_length + 1);
   
   ctx->results.numCoords = numCoords;

   fprintf(stderr, "Allocating result array for %d queries (%d bytes) ...",numQueries, numCoords*sizeof(MatchCoord) );
   ctx->results.h_match_coords = (MatchCoord*) calloc( numCoords, sizeof(MatchCoord));
   
   if (!ctx->on_cpu)
   {
      unsigned int toboardtimer = 0;
      createTimer(&toboardtimer);
      startTimer(toboardtimer);

	  ctx->results.bytes_on_board = 0;

	  CUDA_SAFE_CALL( cudaMalloc( (void**) &ctx->results.d_match_coords, 
								  numCoords * sizeof(MatchCoord)));
	  ctx->results.bytes_on_board += numCoords * sizeof(MatchCoord);

	  CUDA_SAFE_CALL( cudaMemset( (void*)ctx->results.d_match_coords, 0, 
								  numCoords * sizeof(MatchCoord)));

      stopTimer(toboardtimer);
      ctx->statistics.t_to_board += getTimerValue(toboardtimer);
      deleteTimer(toboardtimer);
   }
   else
   {
	  ctx->results.d_match_coords = NULL;
   }
   
   fprintf(stderr, "done\n");
}

void unloadResultBuffer(MatchContext* ctx)
{
   CUDA_SAFE_CALL(cudaFree(ctx->results.d_match_coords));
   
   ctx->results.bytes_on_board = 0;
}


void freeResults(MatchContext* ctx, ReferencePage pages[], unsigned int num_pages)
{
   for (int i = 0; i < num_pages; ++i)
   {
	  free(pages[i].results.h_match_coords);
   }
}

void transferResultsFromDevice(MatchContext* ctx)
{
   if (!ctx->on_cpu)
   {
      unsigned int fromboardtimer = 0;
      createTimer(&fromboardtimer);
      startTimer(fromboardtimer);
	       
	  CUDA_SAFE_CALL(cudaMemcpy(ctx->results.h_match_coords, 
								ctx->results.d_match_coords, 
								ctx->results.numCoords * sizeof(MatchCoord), 
								cudaMemcpyDeviceToHost) );

      stopTimer(fromboardtimer);
      ctx->statistics.t_from_board += getTimerValue(fromboardtimer);
      deleteTimer(fromboardtimer);
   }
  
}


int flushOutput();
int addToBuffer(char* string);

inline int match_coord_addrs(int qryid, int qry_addrs, int match_length)
{
   return qry_addrs - qryid * (match_length + 1);
}

#define MAX_QUERY_LEN 8192 

struct packed_slot
{
	  unsigned short page;
	  unsigned short qpos;
	  MatchCoord coord;
};

struct packed_slot_array
{
	  packed_slot* slots;
	  unsigned int num_slots;
};

void addPackedOutput(MatchContext* ctx, packed_slot_array** curr_output, packed_slot_array slot_array[])
{
   unsigned int numQueries = ctx->queries->count;
   if (*curr_output == NULL)
   {
	  *curr_output = slot_array;
   }
   else
   {
	  for (int i = 0; i < numQueries; i++)
	  {
		 if (slot_array[i].num_slots)
		 {
			//packed_slot_array* slots = &(slot_array[i]);
			(*curr_output)[i].slots = (packed_slot*)realloc((*curr_output)[i].slots,
															((*curr_output)[i].num_slots + slot_array[i].num_slots) * sizeof(packed_slot));
			memcpy((*curr_output)[i].slots + (*curr_output)[i].num_slots,
				   slot_array[i].slots, 
				   slot_array[i].num_slots * sizeof(packed_slot)); 
			(*curr_output)[i].num_slots += slot_array[i].num_slots;
			free(slot_array[i].slots);
		 }
	  }
	  free(slot_array);
   }
}

char numbuffer[32];

void printRCSlots(MatchContext * ctx, ReferencePage pages[], int qry, packed_slot_array * slots)
{
  char* h_tex_array = ctx->queries->h_tex_array;
  int*  h_addrs_tex_array = ctx->queries->h_addrs_tex_array;
  int   qrylen = ctx->queries->h_lengths_array[qry];

  addToBuffer("> ");
  addToBuffer(*(ctx->queries->h_names + qry));
  addToBuffer(" Reverse");

  if (ctx->show_query_length)
  {
    addToBuffer("  Len = ");
    sprintf(numbuffer, "%d", qrylen);
    addToBuffer(numbuffer);
  }

  addToBuffer("\n");

  for (int j = 0; j < slots->num_slots; ++j)
  {
     packed_slot slot = slots->slots[j];

     if (slot.coord.edge_match_length & FRMASK)
     {
       printAlignments(ctx->full_ref,
                       &(pages[slot.page]),
                       h_tex_array + h_addrs_tex_array[qry],
                       qrylen,
                       slot.coord.node, 
                       slot.qpos,
                       (slot.coord.edge_match_length & FRUMASK), 
                       ctx->min_match_length,
                       1,
                       ctx->forwardcoordinates);
     }
  }
}
int FOO;
void printForwardSlots(MatchContext * ctx, ReferencePage pages[], int qry, packed_slot_array * slots)
{
  char* h_tex_array = ctx->queries->h_tex_array;
  int* h_addrs_tex_array = ctx->queries->h_addrs_tex_array;
  int qrylen = ctx->queries->h_lengths_array[qry];

  addToBuffer("> ");
  addToBuffer(*(ctx->queries->h_names + qry));

  if (ctx->show_query_length)
  {
    addToBuffer("  Len = ");
    sprintf(numbuffer, "%d", qrylen);
    addToBuffer(numbuffer);
  }

  addToBuffer("\n");

  for (int j = 0; j < slots->num_slots; ++j)
  {
     packed_slot slot = slots->slots[j];
     if (!(slot.coord.edge_match_length & FRMASK))
     {
       printAlignments(ctx->full_ref,
                       &(pages[slot.page]),
                       h_tex_array + h_addrs_tex_array[qry],
                       qrylen,
                       slot.coord.node, 
                       slot.qpos,
                       slot.coord.edge_match_length, 
                       ctx->min_match_length,
                       0,
                       ctx->forwardcoordinates);
     }
  }
  FOO += slots->num_slots;
}

void printPackedResults(MatchContext* ctx, ReferencePage pages[],  packed_slot_array slot_array[])
{
   unsigned int numQueries = ctx->queries->count;
   FOO = 0;
   for (int qry = 0; qry < numQueries; qry++)
   {
      packed_slot_array* slots = &(slot_array[qry]);

      if (ctx->reverse)
      {
        printRCSlots(ctx, pages, qry, slots);
      }
      else
      {
        printForwardSlots(ctx, pages, qry, slots);

        if (ctx->forwardreverse)
        {
          printRCSlots(ctx, pages, qry, slots);
        }
      }
   }
   printf("FOO = %d\n", FOO);
   flushOutput();
}

void packSlots(MatchContext* ctx, MatchResults* results, unsigned int page_num, packed_slot_array** slot_arrays, bool rc) 
{
   unsigned int numQueries = ctx->queries->count;
   
   int* h_addrs_tex_array = ctx->queries->h_addrs_tex_array;
	 
   int match_length = ctx->min_match_length;

   *slot_arrays  = (packed_slot_array*)calloc(numQueries, sizeof(packed_slot_array));

   for (int i = 0; i < numQueries; i++)
   {
	  int qlen;
	  
	  if (i == numQueries - 1)
		 qlen = ctx->queries->texlen - h_addrs_tex_array[i] - match_length;
	  else
		 qlen = h_addrs_tex_array[i + 1] - h_addrs_tex_array[i] - match_length; 
	 
	  packed_slot* qslots = (packed_slot*)calloc(qlen, sizeof(packed_slot));
	  int filled = 0;
	  for (int p = 0; p < qlen; ++p)
	  {
		 MatchCoord* coords = results->h_match_coords;
		 
		 int query_coord_begin = match_coord_addrs(i, h_addrs_tex_array[i], match_length);
		 int query_coord_end = i < numQueries - 1 ? 
			match_coord_addrs(i + 1, h_addrs_tex_array[i + 1], match_length) : results->numCoords;
		 
		 int query_coord = query_coord_begin + p;
		
		 if ((query_coord < query_coord_end) && 
             (coords[query_coord].node > 1)  &&
		     (!(coords[query_coord].edge_match_length & FRMASK) == !rc))
		 {
			packed_slot s;
			s.page = page_num;
			s.qpos = p;
			s.coord = coords[query_coord];
			qslots[filled++] = s; 
		 }
	  }

	  if (filled)
	  {
		 packed_slot* pslots = (packed_slot*)calloc(filled, sizeof(packed_slot));
		 memcpy(pslots, qslots, (filled)*sizeof(packed_slot));
		 
		 (*slot_arrays)[i].slots = pslots;
		 (*slot_arrays)[i].num_slots = filled;
	  }
	  else
	  {
		 (*slot_arrays)[i].slots = NULL;
		 (*slot_arrays)[i].num_slots = 0;
	  }

	  free(qslots);
   }
}

int getQueryBlock(MatchContext* ctx, size_t device_mem_avail)
{
   QuerySet* queries = ctx->queries;
   char * queryTex = NULL;
   int* queryAddrs = NULL;
   int* queryLengths = NULL;
   unsigned int numQueries;
   size_t queryLen;
   char** names;

   unsigned int queryreadtimer = 0;
   createTimer(&queryreadtimer);
   startTimer(queryreadtimer);

   getQueriesTexture(queries->qfile, 
					 &queryTex, 
					 &queryLen, 
					 &queryAddrs, 
					 &names, 
					 &queryLengths, 
					 &numQueries, 
					 device_mem_avail,
					 ctx->min_match_length,
					 ctx->reverse || ctx->forwardreverse);

   stopTimer(queryreadtimer);
   ctx->statistics.t_query_read += getTimerValue(queryreadtimer);
   deleteTimer(queryreadtimer);

   queries->h_tex_array = queryTex;
   queries->count = numQueries;
   queries->h_addrs_tex_array = queryAddrs;
   queries->texlen = queryLen;
   queries->h_names = names;
   queries->h_lengths_array = queryLengths;

   return numQueries;
}


void destroyQueryBlock(QuerySet* queries)
{
   free(queries->h_tex_array);
   queries->h_tex_array = NULL;

   for (int i = 0; i < queries->count; ++i)
	  free(queries->h_names[i]);

   free(queries->h_names);

   queries->count = 0;
   queries->texlen = 0;

   free(queries->h_addrs_tex_array);
   queries->h_addrs_tex_array = NULL;   

   free(queries->h_lengths_array);
   queries->h_lengths_array = NULL;
}


void writeStatisticsFile(MatchContext* ctx, char* stats_filename)
{
   if (!stats_filename)
	  return;

   FILE* f = fopen(stats_filename, "w");
   
   if (!f)
   {
	  fprintf(stderr, "WARNING: could not open %s for writing\n", stats_filename);
	  return;
   }

   fprintf(f, "Total,%f\n",             ctx->statistics.t_total);
   fprintf(f, "Kernel,%f\n",            ctx->statistics.t_kernel);
   fprintf(f, "Print matches,%f\n",            ctx->statistics.t_output);
   fprintf(f, "Copy queries to GPU,%f\n",          ctx->statistics.t_to_board);
   fprintf(f, "Copy output from GPU,%f\n",        ctx->statistics.t_from_board);
   fprintf(f, "Copy suffix tree to GPU,%f\n", ctx->statistics.t_moving_tree_pages);
   fprintf(f, "Read queries from disk,%f\n",        ctx->statistics.t_query_read);
   fprintf(f, "Suffix tree constructions,%f\n",        ctx->statistics.t_construction);

   fprintf(f, "Minimum substring length, %d\n", ctx->min_match_length);
   fprintf(f, "Average query length, %f\n", ctx->statistics.bp_avg_query_length);
   fclose(f);
}

int matchSubset(MatchContext* ctx, 
				int query_block_offset,
				ReferencePage pages[],
				unsigned int num_pages)
{
   
   loadQueries(ctx);
   packed_slot_array* packed_slots = NULL;
   for (unsigned int i = 0; i < num_pages; ++i)
   { 
	  ctx->ref = &(pages[i].ref);
	  loadReference(ctx);
	  loadResultBuffer(ctx);
	  
      unsigned int ktimer = 0;
      createTimer(&ktimer);
	  
	  unsigned int numQueries = ctx->queries->count;
	  int blocksize = (numQueries > BLOCKSIZE) ? BLOCKSIZE : numQueries;
	  
	  dim3 dimBlock(blocksize,1,1);
	  dim3 dimGrid(ceil(numQueries/(float)BLOCKSIZE), 1, 1);
	  
	  if (!ctx->on_cpu)
	  {
		 fprintf(stderr,"Using blocks of %d x %d x %d threads\n", 
				 dimBlock.x, dimBlock.y, dimBlock.z);
		 fprintf(stderr,"Using a grid of %d x %d x %d blocks\n", 
				 dimGrid.x, dimGrid.y, dimBlock.z); 
		 fprintf(stderr,"Memory footprint is:\n\tqueries: %d\n\tref: %d\n\tresults: %d\n",
				 ctx->queries->bytes_on_board,
				 ctx->ref->bytes_on_board,
				 ctx->results.bytes_on_board);
	  }
	  
	  startTimer(ktimer);

      bool alignRC = ctx->reverse;

	  if (ctx->on_cpu)
	  { 
		  if (alignRC)
		  {
			 computeGold(&ctx->results,
						 ctx->ref->str,
						 ctx->queries->h_tex_array, 
						 ctx->queries->h_addrs_tex_array, 
						 ctx->queries->h_lengths_array,
						 (PixelOfNode*)(ctx->ref->h_node_tex_array), 
						 (PixelOfChildren*)(ctx->ref->h_children_tex_array), 
						 ctx->queries->count,
						 ctx->min_match_length, 
						 REVERSE);		
		  }
		  else
		  {
			 computeGold(&ctx->results,
						 ctx->ref->str,
						 ctx->queries->h_tex_array, 
						 ctx->queries->h_addrs_tex_array, 
						 ctx->queries->h_lengths_array,
						 (PixelOfNode*)(ctx->ref->h_node_tex_array), 
						 (PixelOfChildren*)(ctx->ref->h_children_tex_array), 
						 ctx->queries->count,
						 ctx->min_match_length, 
						 FORWARD);		
		  }
	  }
	  else
	  {
		 
		 if (alignRC)
		 {
			mummergpuRCKernel<<< dimGrid, dimBlock, 0 >>>(ctx->results.d_match_coords,
													   ctx->queries->d_tex_array,
													   ctx->queries->d_addrs_tex_array,
													   ctx->queries->d_lengths_array,
													   numQueries,
													   ctx->min_match_length);
		 }
		 else
		 {
			mummergpuKernel<<< dimGrid, dimBlock, 0 >>>(ctx->results.d_match_coords,
													 ctx->queries->d_tex_array,
													 ctx->queries->d_addrs_tex_array,
													 ctx->queries->d_lengths_array,
													 numQueries,
													 ctx->min_match_length);
		 }
		
	  }

      cudaThreadSynchronize();
	  
	  // check if kernel execution generated an error
      cudaError_t err = cudaGetLastError();
      if( cudaSuccess != err) 
      {
          fprintf(stderr, "Kernel execution failed: %s.\n",
                   cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

	  stopTimer(ktimer);

	  float ktime = getTimerValue(ktimer);
	  ctx->statistics.t_kernel += ktime;
	  fprintf(stderr,"kernel time= %f\n", ktime);
	  deleteTimer(ktimer);
	 
	  transferResultsFromDevice(ctx);
	  pages[i].results = ctx->results;

	  packed_slot_array* packed;
	  packSlots(ctx, &(pages[i].results), i, &packed, ctx->reverse);
	  addPackedOutput(ctx, &packed_slots, packed);

	  // now compute the reverse matches.
	  if (ctx->forwardreverse)
	  {
		 unsigned int rctimer = 0;
		 createTimer(&rctimer);
		 startTimer(rctimer);

		 if (ctx->on_cpu)
		 { 
			computeGold(&ctx->results,
						ctx->ref->str,
						ctx->queries->h_tex_array, 
						ctx->queries->h_addrs_tex_array, 
						ctx->queries->h_lengths_array,
						(PixelOfNode*)(ctx->ref->h_node_tex_array), 
						(PixelOfChildren*)(ctx->ref->h_children_tex_array), 
						ctx->queries->count,
						ctx->min_match_length, 
						REVERSE);		
		 }
		 else
		 {
		
			mummergpuRCKernel<<< dimGrid, dimBlock, 0 >>>(ctx->results.d_match_coords,
													   ctx->queries->d_tex_array,
													   ctx->queries->d_addrs_tex_array,
													   ctx->queries->d_lengths_array,
													   numQueries,
													   ctx->min_match_length);
			cudaThreadSynchronize();
		 }
		 
		 stopTimer(rctimer);

		 float rctime = getTimerValue(rctimer);
		 ctx->statistics.t_kernel += rctime;
		 fprintf(stderr,"rc kernel time= %f\n", rctime);
		 deleteTimer(rctimer);

		 transferResultsFromDevice(ctx);
		 pages[i].results = ctx->results;

		 packed_slot_array* packed;
		 packSlots(ctx, &(pages[i].results), i, &packed, 1);
		 addPackedOutput(ctx, &packed_slots, packed);
	  }
		 
	  free(pages[i].results.h_match_coords);
	  pages[i].results.h_match_coords = NULL;

	  unloadReference(ctx);
	  unloadResultBuffer(ctx);
   }


   unsigned int otimer = 0;
   createTimer(&otimer);
   startTimer(otimer);

   printPackedResults(ctx, pages, packed_slots);  
   
   stopTimer(otimer);
   ctx->statistics.t_output += getTimerValue(otimer);
   deleteTimer(otimer);
  
   for (int i = 0; i < ctx->queries->count; ++i)
   {
	  free(packed_slots[i].slots);
   }
   free(packed_slots);

   unloadQueries(ctx);
   return 0;
}

#define BREATHING_ROOM (64 * 1024 * 1024)
#define BASES_PER_TREE_PAGE 7500000
#define CHUMP_CHANGE 1500000

extern "C"
int matchQueries(MatchContext* ctx)
{	   
   assert(sizeof(struct PixelOfNode) == sizeof(ulong4));
   assert(sizeof(struct PixelOfChildren) == sizeof(ulong4));

   ctx->statistics.t_kernel = 0.0;
   ctx->statistics.t_output = 0.0;
   ctx->statistics.t_to_board = 0.0;
   ctx->statistics.t_from_board = 0.0;
   ctx->statistics.t_moving_tree_pages = 0.0;
   ctx->statistics.t_query_read = 0.0;
   ctx->statistics.t_total = 0.0;
   ctx->statistics.t_construction = 0.0;
   ctx->statistics.bp_avg_query_length = 0.0;

   unsigned int ttimer = 0;
   createTimer(&ttimer);
   startTimer(ttimer);

   unsigned int ctimer = 0;
   createTimer(&ctimer);
   startTimer(ctimer);


   unsigned int bases_in_ref = ctx->full_ref_len - 3;

   unsigned int page_size = BASES_PER_TREE_PAGE < bases_in_ref ? BASES_PER_TREE_PAGE : bases_in_ref;
   unsigned int num_reference_pages = bases_in_ref / page_size;

   ReferencePage* pages = (ReferencePage*)calloc(num_reference_pages, sizeof(ReferencePage));
   
   unsigned int page_overlap = MAX_QUERY_LEN + 1;


   pages[0].begin = 1;
   pages[0].end = pages[0].begin + 
	  page_size  +  
	  ceil(page_overlap / 2.0) + 1; //the 1 is for the 's' at the beginning
   pages[0].shadow_left = -1;
   pages[0].id = 0;
   
   buildReferenceTexture(&(pages[0].ref), ctx->full_ref, pages[0].begin, pages[0].end);

   for (int i = 1; i < num_reference_pages - 1; ++i)
   {
	  pages[i].begin = pages[i - 1].end - page_overlap;
	  pages[i].end = pages[i].begin + page_size +  page_overlap;
	  
	  pages[i - 1].shadow_right = pages[i].begin;
	  pages[i].shadow_left = pages[i-1].end;
	  pages[i].id = i;
	  buildReferenceTexture(&(pages[i].ref), ctx->full_ref, pages[i].begin, pages[i].end);
   }

   if (num_reference_pages > 1)
   {
	  int last_page = num_reference_pages - 1;
	  pages[last_page].begin = pages[last_page - 1].end - page_overlap;
	  pages[last_page].end = ctx->full_ref_len - 1;
	  pages[last_page - 1].shadow_right = pages[last_page].begin;
	  pages[last_page].shadow_right = -1;
	  pages[last_page].shadow_left = pages[last_page - 1].end;
	  pages[last_page].id = last_page;
	  buildReferenceTexture(&(pages[last_page].ref), 
							ctx->full_ref, 
							pages[last_page].begin, 
							pages[last_page].end);
   }

   stopTimer(ctimer);
   ctx->statistics.t_construction += getTimerValue(ctimer);
   deleteTimer(ctimer);

   cudaDeviceProp props;
   if (!ctx->on_cpu)
   {
	  int deviceCount = 0;
	  cudaGetDeviceCount(&deviceCount);
	  
	  if (deviceCount != 1)
	  {
		 //fprintf(stderr, "Fatal error: no CUDA-capable device found, exiting\n");
		 //return -1;
	  }

	  cudaGetDeviceProperties(&props, 0);
	  fprintf(stderr, "Running under CUDA %d.%d\n", props.major, props.minor);
	  fprintf(stderr, "CUDA device has %d bytes of memory\n", props.totalGlobalMem);	  
   }
   else
   {
	  props.totalGlobalMem = 804585472; // pretend we are on a 8800 GTX
   }
   
   size_t mem_avail = 0;
   for (int i = 0; i < num_reference_pages; ++i)
   {
	  mem_avail = max((unsigned int)pages[i].ref.bytes_on_board, 
					  (unsigned int)mem_avail);
   }

   mem_avail = props.totalGlobalMem - mem_avail;

   fprintf(stderr, "There are %d bytes left on the board\n", mem_avail);   

   mem_avail -= BREATHING_ROOM;

   while (getQueryBlock(ctx, mem_avail))
   {
	  matchSubset(ctx, 0, pages, num_reference_pages);
	  ctx->statistics.bp_avg_query_length = ctx->queries->texlen/(float)(ctx->queries->count) - 2; 
	  destroyQueryBlock(ctx->queries);
	  cudaThreadExit();
   }

   for (int i = 0; i < num_reference_pages; ++i)
   {
	  destroyReference(&(pages[i].ref));
   }
   free(pages);

   stopTimer(ttimer);
   ctx->statistics.t_total += getTimerValue(ttimer);
   deleteTimer(ttimer);

   writeStatisticsFile(ctx, ctx->stats_file);

   return 0;
}

