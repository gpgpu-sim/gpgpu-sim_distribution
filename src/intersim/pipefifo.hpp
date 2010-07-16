#ifndef _PIPEFIFO_HPP_
#define _PIPEFIFO_HPP_

#include "module.hpp"

template<class T> class PipelineFIFO : public Module {
   int _lanes;
   int _depth;

   int _pipe_len;
   int _pipe_ptr;

   T ***_data;

public:
   PipelineFIFO( Module *parent, const string& name, int lanes, int depth );
   ~PipelineFIFO( );

   void Write( T* val, int lane = 0 );
   void WriteAll( T* val );

   T*   Read( int lane = 0 );

   void Advance( );
};

template<class T> PipelineFIFO<T>::PipelineFIFO( Module *parent, 
                                                 const string& name, 
                                                 int lanes, int depth ) :
Module( parent, name ),
_lanes( lanes ), _depth( depth )
{
   _pipe_len = depth + 1;
   _pipe_ptr = 0;

   _data = new T ** [_lanes];
   for ( int l = 0; l < _lanes; ++l ) {
      _data[l] = new T * [_pipe_len];

      for ( int d = 0; d < _pipe_len; ++d ) {
         _data[l][d] = 0;
      }
   }
}

template<class T> PipelineFIFO<T>::~PipelineFIFO( ) 
{
   for ( int l = 0; l < _lanes; ++l ) {
      delete [] _data[l];
   }
   delete [] _data;
}

template<class T> void PipelineFIFO<T>::Write( T* val, int lane )
{
   _data[lane][_pipe_ptr] = val;
}

template<class T> void PipelineFIFO<T>::WriteAll( T* val )
{
   for ( int l = 0; l < _lanes; ++l ) {
      _data[l][_pipe_ptr] = val;
   }
}

template<class T> T* PipelineFIFO<T>::Read( int lane )
{
   return _data[lane][_pipe_ptr];
}

template<class T> void PipelineFIFO<T>::Advance( )
{
   _pipe_ptr = ( _pipe_ptr + 1 ) % _pipe_len;
}

#endif 
