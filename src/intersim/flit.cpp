#include "booksim.hpp"
#include "flit.hpp"

ostream& operator<<( ostream& os, const Flit& f )
{
   os << "  Flit ID: " << f.id << " (" << &f << ")" 
   << " Head: " << f.head << " Tail: " << f.tail << endl;
   os << "  Source : " << f.src << "  Dest : " << f.dest << endl;
   os << "  Injection time : " << f.time << endl;

   return os;
}
