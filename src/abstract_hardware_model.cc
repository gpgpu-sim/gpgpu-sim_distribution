#include "abstract_hardware_model.h"
#include "cuda-sim/memory.h"
#include <algorithm>

unsigned mem_access_t::next_access_uid = 0;   

void move_warp( warp_inst_t *&dst, warp_inst_t *&src )
{
   assert( dst->empty() );
   warp_inst_t* temp = dst;
   dst = src;
   src = temp;
   src->clear();
}

gpgpu_t::gpgpu_t()
{
   g_global_mem = new memory_space_impl<8192>("global",64*1024);
   g_param_mem = new memory_space_impl<8192>("param",64*1024);
   g_tex_mem = new memory_space_impl<8192>("tex",64*1024);
   g_surf_mem = new memory_space_impl<8192>("surf",64*1024);

   g_dev_malloc=GLOBAL_HEAP_START; 
}

void warp_inst_t::sort_accessq( unsigned qbegin )
{
    std::stable_sort( m_accessq.begin()+qbegin,m_accessq.end());
}
