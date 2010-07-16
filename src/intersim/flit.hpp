#ifndef _FLIT_HPP_
#define _FLIT_HPP_

#include "booksim.hpp"

#include <iostream>

struct Flit {
   void* data;
   int net_num; // which network is this flit in (we might have several icnt networks)

   int vc;

   bool head;
   bool tail;
   bool true_tail;

   int  time;

   int  sn;
   int  rob_time;

   int  id;
   bool record;

   int  src;
   int  dest;

   int  pri;

   int  hops;
   bool watch;

   // Fields for multi-phase algorithms
   mutable int intm;
   mutable int ph;

   mutable int dr;

   // Which VC parition to use for deadlock avoidance in a ring
   mutable int ring_par;
};

ostream& operator<<( ostream& os, const Flit& f );

#endif
