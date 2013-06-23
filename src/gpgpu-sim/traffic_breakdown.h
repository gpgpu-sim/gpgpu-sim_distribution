#pragma once 

#include <stdio.h>
#include <map>
#include <string> 

// Breakdown traffic through the network according to category
class traffic_breakdown
{
public: 
   traffic_breakdown(const std::string &network_name) 
   : m_network_name(network_name) 
   { }

   // print the stats 
   void print(FILE* fout); 

   // record the amount and type of traffic introduced by this mem_fetch object 
   void record_traffic(class mem_fetch * mf, unsigned int size); 

protected:

   std::string m_network_name; 

   /// helper functions to identify the type of traffic sent 
   std::string classify_memfetch(class mem_fetch * mf); 

   /// helper functions to identify the size of traffic sent 
   unsigned int packet_size(class mem_fetch * mf); 

   typedef std::string mf_packet_type;  // use string so that it remains extensible 
   typedef unsigned int mf_packet_size; 
   typedef std::map < mf_packet_size, unsigned int > traffic_class_t; 
   typedef std::map < mf_packet_type, traffic_class_t > traffic_stat_t; 

   traffic_stat_t m_stats; 
}; 
