#include "PoolMalloc.hh"
//#include "CUtilities.h"
#include <iostream>
#include <cstring>
using namespace std;

static const size_t POOLBLOCKSIZE = 10*1024*1024;

/// Print current line and exit
#define DIE()                                                           \
  do {                                                                  \
    fprintf(stderr,"Aborted. File: %s, Line: %d\n",__FILE__,__LINE__);  \
    exit(EXIT_FAILURE);                                                 \
  } while(0)

/// Print a message and then DIE()
#define DIEM(m) do { fprintf(stderr,m"\n"); DIE(); } while(0)

/// DIE() if the pointer is NULL
#define DIENULL(p) do { if ( !(p) ) { DIEM("Out of memory"); } } while(0)


/// Malloc memory, DIE() if NULL
#define MALLOC(p,t,s) DIENULL( (p) = (t) malloc(s) )

/// Malloc memory for objects of type t, DIE() if NULL
#define NEW(p,t) MALLOC(p, t*, sizeof(t))


#include <cstdlib>

struct PoolNode_t
{
  char *block;
  size_t offset;
  size_t remaining;
  size_t size;
  
  PoolNode_t *next;
};



//============================================================ PoolMalloc_t ====
//------------------------------------------------------------ PoolMalloc_t ----
PoolMalloc_t::PoolMalloc_t()
{
  head_m = NULL;
}


//----------------------------------------------------------- ~PoolMalloc_t ----
PoolMalloc_t::~PoolMalloc_t()
{
  pfree();
}


//-------------------------------------------------------------------- free ----
void PoolMalloc_t::pfree()
{
  PoolNode_t * next;

  while ( head_m )
    {
      next = head_m->next;
      free(head_m->block);
      free(head_m);
      head_m = next;
    }

  head_m = NULL;
}


//------------------------------------------------------------------ malloc ----
///
/// Parcel memory from big byte buckets. Return all memory with 8 byte
/// alignment to silence the bus errors on Alpha and Solaris.
///
void * PoolMalloc_t::pmalloc(size_t size)
{
  size_t remainder = size % 8;

  //-- Make sure the next block is 8 byte aligned
  if ( remainder ) size += 8 - remainder;

  if ( head_m == NULL || size > head_m->remaining )
    {
      size_t blockSize = POOLBLOCKSIZE;
      if ( size > blockSize ) blockSize = size;

      PoolNode_t *newHead;
      NEW(newHead, PoolNode_t);
      MALLOC(newHead->block, char*, blockSize);
      newHead->size = blockSize;
      newHead->remaining = blockSize;
      newHead->offset = 0;
      newHead->next = head_m;
      head_m = newHead;
    }

  void *retval = head_m->block + head_m->offset;
  head_m->offset += size;
  head_m->remaining -= size;

  return retval;
}


//------------------------------------------------------------------ strdup ----
char * PoolMalloc_t::pstrdup(const char * s)
{
  size_t size = strlen(s) + 1;
  char *retval = (char *) pmalloc(size);
  memcpy(retval, s, size);
  return retval;
}
