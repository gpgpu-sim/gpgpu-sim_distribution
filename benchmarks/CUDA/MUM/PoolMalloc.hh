////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Adam M Phillippy
//! \date 09/22/2006
//!
//! \brief PoolMalloc_t header file
//!
////////////////////////////////////////////////////////////////////////////////

#ifndef POOLMALLOC_HH
#define POOLMALLOC_HH

#include <cstdlib>

struct PoolNode_t;

//============================================================ PoolMalloc_t ====
/** \brief Class for allocating memory from a pool
 ** Manage memory in pools to speed up allocation of many short strings or
 ** arrays.
 **
 ** All of the arrays/strings allocated from a given PoolMalloc_t object
 ** should share the same lifetime, since they will all be free'd when the
 ** object is destroyed.
 **/

class PoolMalloc_t
{

public:
  /// Initialize the pool
  PoolMalloc_t();

  /// Free all of the memory associated with this object
  ~PoolMalloc_t();

  /// Explicitly free all of the memory associated with this object
  void pfree();

  /// Allocate an array from the pool
  void * pmalloc(size_t size);

  /// Copy a string using memory from the pool
  char * pstrdup(const char *s);

private:

  PoolNode_t *head_m;

};


#endif // POOLMALLOC_HH
