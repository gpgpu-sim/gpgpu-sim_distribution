#ifndef _OUTPUTSET_HPP_
#define _OUTPUTSET_HPP_

#include <queue>
#include <list>

class OutputSet {
   int _num_outputs;

   struct sSetElement {
      int vc_start;
      int vc_end;
      int pri;
   };

   list<sSetElement> *_outputs;

public:
   OutputSet( int num_outputs );
   ~OutputSet( );

   void Clear( );
   void Add( int output_port, int vc, int pri = 0 );
   void AddRange( int output_port, int vc_start, int vc_end, int pri = 0 );

   int Size( ) const;
   bool OutputEmpty( int output_port ) const;
   int NumVCs( int output_port ) const;

   int  GetVC( int output_port,  int vc_index, int *pri = 0 ) const;
   bool GetPortVC( int *out_port, int *out_vc ) const;
};

#endif


