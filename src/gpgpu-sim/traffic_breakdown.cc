#include "traffic_breakdown.h" 
#include "mem_fetch.h" 

void traffic_breakdown::print(FILE* fout)
{
   for (traffic_stat_t::const_iterator i_stat = m_stats.begin(); i_stat != m_stats.end(); i_stat++) {
      unsigned int byte_transferred = 0; 
      for (traffic_class_t::const_iterator i_class = i_stat->second.begin(); i_class != i_stat->second.end(); i_class++) {
         byte_transferred += i_class->first * i_class->second;  // byte/packet x #packets
      }
      fprintf(fout, "traffic_breakdown_%s[%s] = %u {", m_network_name.c_str(), i_stat->first.c_str(), byte_transferred);  
      for (traffic_class_t::const_iterator i_class = i_stat->second.begin(); i_class != i_stat->second.end(); i_class++) {
         fprintf(fout, "%u:%u,", i_class->first, i_class->second); 
      }
      fprintf(fout, "}\n"); 
   }
}

void traffic_breakdown::record_traffic(class mem_fetch * mf, unsigned int size) 
{
   m_stats[classify_memfetch(mf)][size] += 1; 
}

std::string traffic_breakdown::classify_memfetch(class mem_fetch * mf)
{
   std::string traffic_name; 

   enum mem_access_type access_type = mf->get_access_type(); 

   switch (access_type) {
   case CONST_ACC_R:    
   case TEXTURE_ACC_R:   
   case GLOBAL_ACC_W:   
   case LOCAL_ACC_R:    
   case LOCAL_ACC_W:    
   case INST_ACC_R:     
   case L1_WRBK_ACC:    
   case L2_WRBK_ACC:    
   case L1_WR_ALLOC_R:  
   case L2_WR_ALLOC_R:  
      traffic_name = mem_access_type_str(access_type); 
      break; 
   case GLOBAL_ACC_R:   
      // check for global atomic operation 
      traffic_name = (mf->isatomic())? "GLOBAL_ATOMIC" : mem_access_type_str(GLOBAL_ACC_R); 
      break; 
   default: assert(0 && "Unknown traffic type"); 
   }
   return traffic_name; 
}

