// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh, Timothy Rogers,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "abstract_hardware_model.h"
#include "cuda-sim/memory.h"
#include "option_parser.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx-stats.h"
#include <algorithm>

unsigned mem_access_t::sm_next_access_uid = 0;   
unsigned warp_inst_t::sm_next_uid = 0;

void move_warp( warp_inst_t *&dst, warp_inst_t *&src )
{
   assert( dst->empty() );
   warp_inst_t* temp = dst;
   dst = src;
   src = temp;
   src->clear();
}


void gpgpu_functional_sim_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL,
                 &m_ptx_convert_to_ptxplus,
                 "Convert embedded ptx to ptxplus",
                 "0");
    option_parser_register(opp, "-gpgpu_ptx_save_converted_ptxplus", OPT_BOOL,
                 &m_ptx_save_converted_ptxplus,
                 "Saved converted ptxplus to a file",
                 "0");
    option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                 &m_ptx_force_max_capability,
                 "Force maximum compute capability",
                 "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, 
                &g_ptx_inst_debug_to_file, 
                "Dump executed instructions' debug information to file", 
                "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file, 
                  "Executed instructions' debug output file",
                  "inst_debug.txt");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32, &g_ptx_inst_debug_thread_uid, 
               "Thread UID for executed instructions' debug output", 
               "1");
}

void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(unsigned linesize)
{
   m_texcache_linesize = linesize;
}

gpgpu_t::gpgpu_t( const gpgpu_functional_sim_config &config )
    : m_function_model_config(config)
{
   m_global_mem = new memory_space_impl<8192>("global",64*1024);
   m_tex_mem = new memory_space_impl<8192>("tex",64*1024);
   m_surf_mem = new memory_space_impl<8192>("surf",64*1024);

   m_dev_malloc=GLOBAL_HEAP_START; 

   if(m_function_model_config.get_ptx_inst_debug_to_file() != 0) 
      ptx_inst_debug_file = fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");
}

address_type line_size_based_tag_func(new_addr_type address, new_addr_type line_size)
{
   //gives the tag for an address based on a given line size
   return address & ~(line_size-1);
}

struct transaction_info {
    std::bitset<4> chunks; // bitmask: 32-byte chunks accessed
    mem_access_byte_mask_t bytes;
    active_mask_t active; // threads in this transaction
};

void warp_inst_t::clear_active( const active_mask_t &inactive ) {
    active_mask_t test = m_warp_active_mask;
    test &= inactive;
    assert( test == inactive ); // verify threads being disabled were active
    m_warp_active_mask &= ~inactive;
}

void warp_inst_t::set_not_active( unsigned lane_id ) {
    m_warp_active_mask.reset(lane_id);
}

void warp_inst_t::set_active( const active_mask_t &active ) {
   m_warp_active_mask = active;
   if( m_isatomic ) {
      for( unsigned i=0; i < m_config->warp_size; i++ ) {
         if( !m_warp_active_mask.test(i) ) {
             m_per_scalar_thread[i].callback.function = NULL;
             m_per_scalar_thread[i].callback.instruction = NULL;
             m_per_scalar_thread[i].callback.thread = NULL;
         }
      }
   }
}

void warp_inst_t::do_atomic() {
    do_atomic( m_warp_active_mask );
}

void warp_inst_t::do_atomic( const active_mask_t& access_mask ) {
    assert( m_isatomic && !m_empty );
    for( unsigned i=0; i < m_config->warp_size; i++ )
    {
        if( access_mask.test(i) )
        {
            dram_callback_t &cb = m_per_scalar_thread[i].callback;
            if( cb.thread )
                cb.function(cb.instruction, cb.thread);
        }
    }
}

void warp_inst_t::generate_mem_accesses()
{
    if( empty() || op == MEMORY_BARRIER_OP || m_mem_accesses_created ) 
        return;
    if ( !((op == LOAD_OP) || (op == STORE_OP)) )
        return; 
    if( m_warp_active_mask.count() == 0 ) 
        return; // predicated off

    const size_t starting_queue_size = m_accessq.size();

    assert( is_load() || is_store() );
    assert( m_per_scalar_thread_valid ); // need address information per thread

    bool is_write = is_store();

    mem_access_type access_type;
    switch (space.get_type()) {
    case const_space:
    case param_space_kernel: 
        access_type = CONST_ACC_R; 
        break;
    case tex_space: 
        access_type = TEXTURE_ACC_R;   
        break;
    case global_space:       
        access_type = is_write? GLOBAL_ACC_W: GLOBAL_ACC_R;   
        break;
    case local_space:
    case param_space_local:  
        access_type = is_write? LOCAL_ACC_W: LOCAL_ACC_R;   
        break;
    case shared_space: break;
    default: assert(0); break; 
    }

    // Calculate memory accesses generated by this warp
    new_addr_type cache_block_size = 0; // in bytes 

    switch( space.get_type() ) {
    case shared_space: {
        unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
        unsigned total_accesses=0;
        for( unsigned subwarp=0; subwarp <  m_config->mem_warp_parts; subwarp++ ) {

            // data structures used per part warp 
            std::map<unsigned,std::map<new_addr_type,unsigned> > bank_accs; // bank -> word address -> access count

            // step 1: compute accesses to words in banks
            for( unsigned thread=subwarp*subwarp_size; thread < (subwarp+1)*subwarp_size; thread++ ) {
                if( !active(thread) ) 
                    continue;
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
                //FIXME: deferred allocation of shared memory should not accumulate across kernel launches
                //assert( addr < m_config->gpgpu_shmem_size ); 
                unsigned bank = m_config->shmem_bank_func(addr);
                new_addr_type word = line_size_based_tag_func(addr,m_config->WORD_SIZE);
                bank_accs[bank][word]++;
            }

            // step 2: look for and select a broadcast bank/word if one occurs
            bool broadcast_detected = false;
            new_addr_type broadcast_word=(new_addr_type)-1;
            unsigned broadcast_bank=(unsigned)-1;
            std::map<unsigned,std::map<new_addr_type,unsigned> >::iterator b;
            for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                unsigned bank = b->first;
                std::map<new_addr_type,unsigned> &access_set = b->second;
                std::map<new_addr_type,unsigned>::iterator w;
                for( w=access_set.begin(); w != access_set.end(); ++w ) {
                    if( w->second > 1 ) {
                        // found a broadcast
                        broadcast_detected=true;
                        broadcast_bank=bank;
                        broadcast_word=w->first;
                        break;
                    }
                }
                if( broadcast_detected ) 
                    break;
            }

            // step 3: figure out max bank accesses performed, taking account of broadcast case
            unsigned max_bank_accesses=0;
            for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                unsigned bank_accesses=0;
                std::map<new_addr_type,unsigned> &access_set = b->second;
                std::map<new_addr_type,unsigned>::iterator w;
                for( w=access_set.begin(); w != access_set.end(); ++w ) 
                    bank_accesses += w->second;
                if( broadcast_detected && broadcast_bank == b->first ) {
                    for( w=access_set.begin(); w != access_set.end(); ++w ) {
                        if( w->first == broadcast_word ) {
                            unsigned n = w->second;
                            assert(n > 1); // or this wasn't a broadcast
                            assert(bank_accesses >= (n-1));
                            bank_accesses -= (n-1);
                            break;
                        }
                    }
                }
                if( bank_accesses > max_bank_accesses ) 
                    max_bank_accesses = bank_accesses;
            }

            // step 4: accumulate
            total_accesses+= max_bank_accesses;
        }
        assert( total_accesses > 0 && total_accesses <= m_config->warp_size );
        cycles = total_accesses; // shared memory conflicts modeled as larger initiation interval 
        break;
    }

    case tex_space: 
        cache_block_size = m_config->gpgpu_cache_texl1_linesize;
        break;
    case const_space:  case param_space_kernel:
        cache_block_size = m_config->gpgpu_cache_constl1_linesize; 
        break;

    case global_space: case local_space: case param_space_local:
         
        if( m_config->gpgpu_coalesce_arch == 13 ) {
            // see the CUDA manual where it discusses coalescing rules before reading this
            unsigned segment_size = 0;
            unsigned warp_parts = 2;
            switch( data_size ) {
            case 1: segment_size = 32; break;
            case 2: segment_size = 64; break;
            case 4: case 8: case 16: segment_size = 128; break;
            }
            unsigned subwarp_size = m_config->warp_size / warp_parts;

            for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
                std::map<new_addr_type,transaction_info> subwarp_transactions;

                // step 1: find all transactions generated by this subwarp
                for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
                    if( !active(thread) ) 
                        continue;

                    // local memory can only be accessed in 4B chunks by one thread
                    unsigned data_size_coales = (space.get_type() == local_space || space.get_type() == param_space_local ) ? 4 : data_size;
                    unsigned num_accesses = (space.get_type() == local_space || space.get_type() == param_space_local ) ? data_size/4 : 1;

                    for(unsigned access=0; access<num_accesses; access++) {
                        new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[access];
                        unsigned block_address = line_size_based_tag_func(addr,segment_size);
                        unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?
                        transaction_info &info = subwarp_transactions[block_address];

                        // can only write to one segment
                        assert(block_address == line_size_based_tag_func(addr+data_size_coales-1,segment_size));

                        info.chunks.set(chunk);
                        info.active.set(thread);
                        unsigned idx = (addr&127);
                        for( unsigned i=0; i < data_size_coales; i++ )
                            info.bytes.set(idx+i);
                    }
                }

                // step 2: reduce each transaction size, if possible
                std::map< new_addr_type, transaction_info >::iterator t;
                for( t=subwarp_transactions.begin(); t !=subwarp_transactions.end(); t++ ) {
                    new_addr_type addr = t->first;
                    assert( (addr & (segment_size-1)) == 0 );
                    const transaction_info &info = t->second;
                    const std::bitset<4> &q = info.chunks;
                    assert( q.count() >= 1 );
                    std::bitset<2> h; // halves (used to check if 64 byte segment can be compressed into a single 32 byte segment)

                    unsigned size=segment_size;
                    if( segment_size == 128 ) {
                        bool lower_half_used = q[0] || q[1];
                        bool upper_half_used = q[2] || q[3]; 
                        if( lower_half_used && !upper_half_used ) {
                            // only lower 64 bytes used
                            size = 64;
                            if(q[0]) h.set(0);
                            if(q[1]) h.set(1);
                        } else if ( (!lower_half_used) && upper_half_used ) {
                            // only upper 64 bytes used
                            addr = addr+64;
                            size = 64;
                            if(q[2]) h.set(0);
                            if(q[3]) h.set(1);
                        } else {
                            assert(lower_half_used && upper_half_used);
                        }
                    } else if( segment_size == 64 ) {
                        // need to set halves
                        if( (addr % 128) == 0 ) {
                            if(q[0]) h.set(0);
                            if(q[1]) h.set(1);
                        } else {
                            assert( (addr % 128) == 64 );
                            if(q[2]) h.set(0);
                            if(q[3]) h.set(1);
                        }
                    }
                    if( size == 64 ) {
                        bool lower_half_used = h[0];
                        bool upper_half_used = h[1]; 
                        if( lower_half_used && !upper_half_used ) {
                            size = 32;
                        } else if ( (!lower_half_used) && upper_half_used ) {
                            addr = addr+32;
                            size = 32;
                        } else {
                            assert(lower_half_used && upper_half_used);
                        }
                    }
                    m_accessq.push_back( mem_access_t(access_type,addr,size,is_write,info.active,info.bytes) );
                }
            }
            
        } else abort();

        break;

    default:
        abort();
    }

    if( cache_block_size ) {
        assert( m_accessq.empty() );
        mem_access_byte_mask_t byte_mask; 
        std::map<new_addr_type,active_mask_t> accesses; // block address -> set of thread offsets in warp
        std::map<new_addr_type,active_mask_t>::iterator a;
        for( unsigned thread=0; thread < m_config->warp_size; thread++ ) {
            if( !active(thread) ) 
                continue;
            new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
            unsigned block_address = line_size_based_tag_func(addr,cache_block_size);
            accesses[block_address].set(thread);
            unsigned idx = addr-block_address; 
            for( unsigned i=0; i < data_size; i++ ) 
                byte_mask.set(idx+i);
        }
        for( a=accesses.begin(); a != accesses.end(); ++a ) 
            m_accessq.push_back( mem_access_t(access_type,a->first,cache_block_size,is_write,a->second,byte_mask) );
    }

    ptx_file_line_stats_add_uncoalesced_gmem( pc, m_accessq.size() - starting_queue_size );
    m_mem_accesses_created=true;
} 

unsigned kernel_info_t::m_next_uid = 1;

kernel_info_t::kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry )
{
    m_kernel_entry=entry;
    m_grid_dim=gridDim;
    m_block_dim=blockDim;
    m_next_cta.x=0;
    m_next_cta.y=0;
    m_next_cta.z=0;
    m_next_tid=m_next_cta;
    m_num_cores_running=0;
    m_uid = m_next_uid++;
    m_param_mem = new memory_space_impl<8192>("param",64*1024);
}

kernel_info_t::~kernel_info_t()
{
    assert( m_active_threads.empty() );
    delete m_param_mem;
}

std::string kernel_info_t::name() const
{
    return m_kernel_entry->get_name();
}
