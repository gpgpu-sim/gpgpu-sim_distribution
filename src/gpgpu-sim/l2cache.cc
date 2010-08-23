#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <set>
#include "../tr1_hash_map.h" // for unordered_map failback

#include "../option_parser.h"
#include "mem_fetch.h"
#include "dram.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../intersim/statwraper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"

class L2c_mshr;
class L2c_miss_tracker;
class L2c_access_locality;

mem_fetch_t* g_debug_mf = NULL;

// L2 cache block (include the cache model + flow controls)
struct L2cacheblk 
{
   shd_cache_t *L2cache;

   delay_queue *cbtoL2queue; //latency 10
   delay_queue *cbtoL2writequeue;
   delay_queue *dramtoL2queue; //latency 10
   delay_queue *dramtoL2writequeue;
   delay_queue *L2todramqueue; //latency 0
   delay_queue *L2todram_wbqueue; 
   delay_queue *L2tocbqueue; //latency 0

   mem_fetch_t *L2request; //request currently being serviced by the L2 Cache

   L2c_mshr *m_mshr; // mshr model 
   L2c_miss_tracker *m_missTracker; // tracker observing for redundant misses
   L2c_access_locality *m_accessLocality; // tracking true locality of L2 Cache access 

   L2cacheblk(size_t linesize);
   ~L2cacheblk(); 
};

// external dependencies
extern unsigned long long int addrdec_mask[5];
extern int gpgpu_dram_sched_queue_size; 
extern unsigned made_write_mfs;
extern unsigned freed_L1write_mfs;
extern unsigned freed_L2write_mfs;

void memlatstat_icnt2sh_push(mem_fetch_t *mf);
void memlatstat_dram_access(mem_fetch_t *mf, unsigned dram_id, unsigned bank);
void memlatstat_start(mem_fetch_t *mf);
unsigned memlatstat_done(mem_fetch_t *mf);

// option
char *gpgpu_L2_queue_config;
bool gpgpu_l2_readoverwrite;
bool l2_ideal;

void L2c_options(option_parser_t opp)
{
   option_parser_register(opp, "-gpgpu_L2_queue", OPT_CSTR, &gpgpu_L2_queue_config, 
                  "L2 data cache queue length and latency config",
                  "0:0:0:0:0:0:10:10");

   option_parser_register(opp, "-gpgpu_l2_readoverwrite", OPT_BOOL, &gpgpu_l2_readoverwrite, 
                "Prioritize read requests over write requests for L2",
                "0");

   option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal, 
                "Use a ideal L2 cache that always hit",
                "0");
}

// stats
unsigned L2_write_miss = 0;
unsigned L2_write_hit = 0;
unsigned L2_read_hit = 0;
unsigned L2_read_miss = 0;
unsigned int *L2_cbtoL2length;
unsigned int *L2_cbtoL2writelength;
unsigned int *L2_L2tocblength;
unsigned int *L2_dramtoL2length;
unsigned int *L2_dramtoL2writelength;
unsigned int *L2_L2todramlength;

////////////////////////////////////////////////
// L2 MSHR model

class L2c_mshr 
{
private:
   typedef std::list<const mem_fetch_t*> mem_fetch_list;
   typedef tr1_hash_map<address_type, mem_fetch_list> L2missGroup;
   L2missGroup m_L2missgroup; // structure tracking redundant dram access

   struct active_chain {
      address_type cacheTag;
      mem_fetch_list *list;
      active_chain() : cacheTag(0xDEADBEEF), list(NULL) { }
   };
   active_chain m_active_mshr_chain; 
   size_t m_linesize; // L2 cache line size

   const size_t m_n_entries; // total number of entries available
   size_t m_entries_used; // number of entries in use

   int m_n_miss; 
   int m_n_miss_serviced_by_dram;
   int m_n_mshr_hits;
   size_t m_max_entries_used; 
   
   address_type cache_tag(const mem_fetch_t *mf) const 
   {
      // return mf->addr;
      return (mf->addr & ~(m_linesize - 1));
   }

public:
   L2c_mshr(size_t linesize, size_t n_entries = 64) 
   : m_linesize(linesize), m_n_entries(n_entries), m_entries_used(0), 
     m_n_miss(0), m_n_miss_serviced_by_dram(0), m_n_mshr_hits(0), m_max_entries_used(0) { }
  
   // add a cache miss to MSHR, return true if this access is hit another existing entry and merges with it
   bool new_miss(const mem_fetch_t *mf);

   // notify MSHR that a new cache line has been fetched, activate the associated MSHR chain
   void miss_serviced(const mem_fetch_t *mf);

   // probe if there are pending hits left in this MSHR chain
   bool mshr_chain_empty();

   // peek the first entry in the active MSHR chain
   mem_fetch_t *mshr_chain_top();

   // pop the first entry in the active MSHR chain
   void mshr_chain_pop(); 

   void print(FILE *fout = stdout); 
   void print_stat(FILE *fout = stdout); 
};

bool L2c_mshr::new_miss(const mem_fetch_t *mf)
{
   address_type cacheTag = cache_tag(mf);
   mem_fetch_list &missGroup = m_L2missgroup[cacheTag];

   bool mshr_hit = not missGroup.empty();

   missGroup.push_front(mf);
   
   m_n_miss += 1;
   if (mshr_hit) 
      m_n_mshr_hits += 1;
   m_entries_used += 1;
   m_max_entries_used = std::max(m_max_entries_used, m_entries_used);

   return mshr_hit;
}

void L2c_mshr::miss_serviced(const mem_fetch_t *mf) 
{
   assert(m_active_mshr_chain.list == NULL);
   address_type cacheTag = cache_tag(mf);
   L2missGroup::iterator missGroup = m_L2missgroup.find(cacheTag);
   if (missGroup == m_L2missgroup.end() || mf->type == L2_WTBK_DATA) {
      assert(mf->type == L2_WTBK_DATA); // only this returning mem req can be missed by the MSHR
      return; 
   } 
   assert(missGroup->first == cacheTag);

   m_active_mshr_chain.cacheTag = cacheTag;
   m_active_mshr_chain.list = &(missGroup->second);

   m_n_miss_serviced_by_dram += 1;
}

bool L2c_mshr::mshr_chain_empty()
{
   return (m_active_mshr_chain.list == NULL);
}

mem_fetch_t *L2c_mshr::mshr_chain_top()
{
   const mem_fetch_t *mf = m_active_mshr_chain.list->back();
   assert(cache_tag(mf) == m_active_mshr_chain.cacheTag);

   return const_cast<mem_fetch_t*>(mf);
}

void L2c_mshr::mshr_chain_pop()
{
   m_entries_used -= 1;
   m_active_mshr_chain.list->pop_back();
   if (m_active_mshr_chain.list->empty()) {
      address_type cacheTag = m_active_mshr_chain.cacheTag;
      m_L2missgroup.erase(cacheTag);
      m_active_mshr_chain.list = NULL;
   }
}

void L2c_mshr::print(FILE *fout)
{
   fprintf(fout, "L2c MSHR: n_entries_used = %zu\n", m_entries_used);
   L2missGroup::iterator missGroup;
   for (missGroup = m_L2missgroup.begin(); missGroup != m_L2missgroup.end(); ++missGroup) {
      fprintf(fout, "%#08x: ", missGroup->first); 
      mem_fetch_list &mf_list = missGroup->second; 
      for (mem_fetch_list::iterator imf = mf_list.begin(); imf != mf_list.end(); ++imf) {
         fprintf(fout, "%p:%d ", *imf, (*imf)->request_uid);
      }
      fprintf(fout, "\n");
   }
}

void L2c_mshr::print_stat(FILE *fout)
{
   fprintf(fout, "L2c MSHR: max_entry = %zu, n_miss = %d, n_mshr_hits = %d, n_serviced_by_dram %d\n", 
           m_max_entries_used, m_n_miss, m_n_mshr_hits, m_n_miss_serviced_by_dram);
}

////////////////////////////////////////////////
// track redundant dram access generated by L2 cache
class L2c_miss_tracker
{
private:
   typedef std::set<mem_fetch_t*> mem_fetch_set;
   typedef tr1_hash_map<address_type, mem_fetch_set> L2missGroup;
   L2missGroup m_L2missgroup; // structure tracking redundant dram access
   size_t m_linesize; // L2 cache line size

   typedef tr1_hash_map<address_type, int> L2redundantCnt; 
   L2redundantCnt m_L2redundantCnt; 

   int m_totalL2redundantAcc;

   address_type cache_tag(const mem_fetch_t *mf) const 
   {
      // return mf->addr;
      return (mf->addr & ~(m_linesize - 1));
   }

public:
   L2c_miss_tracker(size_t linesize) : m_linesize(linesize), m_totalL2redundantAcc(0) { }
   void new_miss(mem_fetch_t *mf);
   void miss_serviced(mem_fetch_t *mf);

   void print(FILE *fout, bool brief = true);
   void print_stat(FILE *fout, bool brief = true);

};

void L2c_miss_tracker::new_miss(mem_fetch_t *mf)
{
   address_type cacheTag = cache_tag(mf);
   mem_fetch_set &missGroup = m_L2missgroup[cacheTag];

   if (missGroup.size() != 0) {
      m_L2redundantCnt[cacheTag] += 1;
      m_totalL2redundantAcc += 1;
   }

   missGroup.insert(mf);
}

void L2c_miss_tracker::miss_serviced(mem_fetch_t *mf)
{
   address_type cacheTag = cache_tag(mf);
   L2missGroup::iterator iMissGroup = m_L2missgroup.find(cacheTag);
   if (iMissGroup == m_L2missgroup.end()) return; // this is possible for write miss 
   mem_fetch_set &missGroup = iMissGroup->second;

   missGroup.erase(mf);

   // remove the miss group if it goes empty
   if (missGroup.empty()) {
      m_L2missgroup.erase(iMissGroup);
   }
}

void L2c_miss_tracker::print(FILE *fout, bool brief)
{
   L2missGroup::iterator iMissGroup;
   for (iMissGroup = m_L2missgroup.begin(); iMissGroup != m_L2missgroup.end(); ++iMissGroup) {
      fprintf(fout, "%#08x: ", iMissGroup->first); 
      for (mem_fetch_set::iterator iMemSet = iMissGroup->second.begin(); iMemSet != iMissGroup->second.end(); ++iMemSet) { 
         fprintf(fout, "%p ", *iMemSet);
      }
      fprintf(fout, "\n");
   }
}

void L2c_miss_tracker::print_stat(FILE *fout, bool brief)
{
   fprintf(fout, "RedundantMiss = %d\n", m_totalL2redundantAcc);
   if (brief == true) return;
   fprintf(fout, "  Detail:");
   for (L2redundantCnt::iterator iL2rc = m_L2redundantCnt.begin(); iL2rc != m_L2redundantCnt.end(); ++iL2rc) {
      fprintf(fout, "%#08x:%d ", iL2rc->first, iL2rc->second);
   }
   fprintf(fout, "\n");
}

////////////////////////////////////////////////
// track all locality of L2 cache access
class L2c_access_locality
{
private:
   size_t m_linesize; // L2 cache line size

   typedef tr1_hash_map<address_type, int> L2accCnt; 
   L2accCnt m_L2accCnt; 

   int m_totalL2accAcc;

   address_type cache_tag(const mem_fetch_t *mf) const 
   {
      // return mf->addr;
      return (mf->addr & ~(m_linesize - 1));
   }

public:
   L2c_access_locality(size_t linesize) : m_linesize(linesize), m_totalL2accAcc(0) { }
   void access(mem_fetch_t *mf);

   void print_stat(FILE *fout, bool brief = true);

};

void L2c_access_locality::access(mem_fetch_t *mf)
{
   address_type cacheTag = cache_tag(mf);
   m_L2accCnt[cacheTag] += 1;
   m_totalL2accAcc += 1;
}

void L2c_access_locality::print_stat(FILE *fout, bool brief)
{
   float access_locality = (float) m_totalL2accAcc / m_L2accCnt.size();
   fprintf(fout, "Access Locality = %d / %zu (%f) \n", m_totalL2accAcc, m_L2accCnt.size(), access_locality);
   if (brief == true) return;
   fprintf(fout, "  Detail:");
   pow2_histogram locality_histo(" Hits");
   for (L2accCnt::iterator iL2rc = m_L2accCnt.begin(); iL2rc != m_L2accCnt.end(); ++iL2rc) {
      locality_histo.add2bin(iL2rc->second);
      // fprintf(fout, "%#08x:%d\n", iL2rc->first, iL2rc->second);
   }
   locality_histo.fprint(fout);
   fprintf(fout, "\n");
}

L2cacheblk::L2cacheblk(size_t linesize)
: m_mshr(new L2c_mshr(linesize)), 
  m_missTracker(new L2c_miss_tracker(linesize)), 
  m_accessLocality(new L2c_access_locality(linesize)) 
{ }

L2cacheblk::~L2cacheblk()
{
   delete m_mshr;
   delete m_missTracker;
   delete m_accessLocality; 
}


//////////////////////////////////////////////// 
// L2 access functions

// L2 Cache Creation 
void L2c_create ( dram_t* dram_p, const char* cache_opt )
{
   unsigned int shd_n_set;
   unsigned int shd_linesize;
   unsigned int shd_n_assoc;
   unsigned char shd_policy;

   unsigned int L2c_cb_L2_length;
   unsigned int L2c_cb_L2w_length;
   unsigned int L2c_L2_dm_length;
   unsigned int L2c_dm_L2_length;
   unsigned int L2c_dm_L2w_length;
   unsigned int L2c_L2_cb_length;
   unsigned int L2c_L2_cb_minlength;
   unsigned int L2c_L2_dm_minlength;

   sscanf(cache_opt,"%d:%d:%d:%c", 
          &shd_n_set, &shd_linesize, &shd_n_assoc, &shd_policy);

   L2cacheblk *p_L2c = new L2cacheblk(shd_linesize);

   char L2c_name[32];
   snprintf(L2c_name, 32, "L2c_%03d", dram_p->id);
   p_L2c->L2cache = shd_cache_create(L2c_name, 
                                      shd_n_set, shd_n_assoc, shd_linesize, 
				     shd_policy, 16, ~addrdec_mask[CHIP], 
				     write_through); //write_through maintains old behavior for now

   sscanf(gpgpu_L2_queue_config,"%d:%d:%d:%d:%d:%d:%d:%d", 
          &L2c_cb_L2_length, &L2c_cb_L2w_length, &L2c_L2_dm_length, 
          &L2c_dm_L2_length, &L2c_dm_L2w_length, &L2c_L2_cb_length,
          &L2c_L2_cb_minlength, &L2c_L2_dm_minlength );
   //(<name>,<latency>,<min_length>,<max_length>)
   p_L2c->cbtoL2queue        = dq_create("cbtoL2queue",       0,0,L2c_cb_L2_length); 
   p_L2c->cbtoL2writequeue   = dq_create("cbtoL2writequeue",  0,0,L2c_cb_L2w_length); 
   p_L2c->L2todramqueue      = dq_create("L2todramqueue",     0,L2c_L2_dm_minlength,L2c_L2_dm_length);
   p_L2c->dramtoL2queue      = dq_create("dramtoL2queue",     0,0,L2c_dm_L2_length);
   p_L2c->dramtoL2writequeue = dq_create("dramtoL2writequeue",0,0,L2c_dm_L2w_length);
   p_L2c->L2tocbqueue        = dq_create("L2tocbqueue",       0,L2c_L2_cb_minlength,L2c_L2_cb_length);

   p_L2c->L2todram_wbqueue   = dq_create("L2todram_wbqueue",  0,L2c_L2_dm_minlength,
                                          L2c_L2_dm_minlength + gpgpu_dram_sched_queue_size + L2c_dm_L2_length);

   p_L2c->L2request = NULL; 

   assert(dram_p->m_L2cache == NULL);
   dram_p->m_L2cache = reinterpret_cast<void*>(p_L2c);
}

unsigned L2c_get_linesize( dram_t *dram_p )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);
   return p_L2c->L2cache->line_sz; 
}

int L2c_full( dram_t *dram_p )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);
   return(dq_full(p_L2c->cbtoL2queue) || dq_full(p_L2c->cbtoL2writequeue));
}

void L2c_push( dram_t *dram_p, mem_fetch_t *mf )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   if (gpgpu_l2_readoverwrite && mf->write)
      dq_push(p_L2c->cbtoL2writequeue, mf);
   else
      dq_push(p_L2c->cbtoL2queue, mf);
   p_L2c->m_accessLocality->access(mf); 
   if (mf->mshr) mshr_update_status(mf->mshr, IN_CBTOL2QUEUE);
}

mem_fetch_t* L2c_pop( dram_t *dram_p )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   mem_fetch_t *mf;
   mf = (mem_fetch_t*)dq_pop(p_L2c->L2tocbqueue);

   return mf;
}

mem_fetch_t* L2c_top( dram_t *dram_p )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   return (mem_fetch_t*)dq_top(p_L2c->L2tocbqueue);
}

void L2c_qlen ( dram_t *dram_p )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   printf("\n");
   printf("cb->L2{%d}\tcb->L2w{%d}\tL2->cb{%d}\n", 
          p_L2c->cbtoL2queue->length, 
          p_L2c->cbtoL2writequeue->length, 
          p_L2c->L2tocbqueue->length);
   printf("dm->L2{%d}\tdm->L2w{%d}\tL2->dm{%d}\tL2->wb_dm{%d}\n", 
          p_L2c->dramtoL2queue->length, 
          p_L2c->dramtoL2writequeue->length, 
          p_L2c->L2todramqueue->length,
          p_L2c->L2todram_wbqueue->length);
}

// service memory request in icnt-to-L2 queue, writing to L2 as necessary
// (if L2 writeback miss, writeback to memory) 
void L2c_service_mem_req ( dram_t* dram_p, int dm_id )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   mem_fetch_t* mf;

   if (!p_L2c->L2request) {
      //if not servicing L2 cache request..
      p_L2c->L2request = (mem_fetch_t*) dq_pop(p_L2c->cbtoL2queue); //..then get one
      if (!p_L2c->L2request) {
         p_L2c->L2request = (mem_fetch_t*) dq_pop(p_L2c->cbtoL2writequeue);
      }
   }

   mf = p_L2c->L2request;

   if (!mf) return;

   switch (mf->type) {
   case RD_REQ:
   case WT_REQ: {
         shd_cache_line_t *hit_cacheline = shd_cache_access(p_L2c->L2cache,
                                                            mf->addr,
                                                            4, mf->write,
                                                            gpu_sim_cycle);

         if (hit_cacheline || l2_ideal) { //L2 Cache Hit; reads are sent as a single command and need to be stored
            if (!mf->write) { //L2 Cache Read
               if ( dq_full(p_L2c->L2tocbqueue) ) {
                  p_L2c->L2cache->access--;
               } else {
                  mf->type = REPLY_DATA;
                  dq_push(p_L2c->L2tocbqueue, mf);
                  // at this point, should first check if earlier L2 miss is ready to be serviced
                  // if so, service earlier L2 miss first
                  p_L2c->L2request = NULL; //finished servicing
                  L2_read_hit++;
                  memlatstat_icnt2sh_push(mf);
                  if (mf->mshr) mshr_update_status(mf->mshr, IN_L2TOCBQUEUE_HIT);
               }
            } else { //L2 Cache Write aka servicing L1 Writeback
               p_L2c->L2request = NULL;    
               L2_write_hit++;
               freed_L1write_mfs++;
               free(mf); //writeback from L1 successful
               gpgpu_n_processed_writes++;
            }
         } else {
            // L2 Cache Miss; issue commands accordingly
            if ( dq_full(p_L2c->L2todramqueue) ) {
               p_L2c->L2cache->miss--;
               p_L2c->L2cache->access--;
            } else {
               // if a miss hit the mshr, that means there is another inflight request for the same data
               // this miss just need to access the cache later when this request is serviced
               bool mshr_hit = p_L2c->m_mshr->new_miss(mf);
               if (not mshr_hit) {
                  if (!mf->write) {
                     dq_push(p_L2c->L2todramqueue, mf);
                  } else {
                     // if request is writeback from L1 and misses, 
                     // then redirect mf writes to dram (no write allocate)
                     mf->nbytes_L2 = mf->nbytes_L1 - READ_PACKET_SIZE;
                     dq_push(p_L2c->L2todramqueue, mf);
                  }
               }
               if (mf->mshr) mshr_update_status(mf->mshr, IN_L2TODRAMQUEUE);
               p_L2c->L2request = NULL;
            }
         }
      }
      break;
   default: assert(0);
   }
}

// service memory request in L2todramqueue, pushing to dram 
void L2c_push_miss_to_dram ( dram_t* dram_p )
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   mem_fetch_t* mf;

   if ( gpgpu_dram_sched_queue_size && dram_full(dram_p) ) return;

   mf = (mem_fetch_t*) dq_pop(p_L2c->L2todram_wbqueue); //prioritize writeback
   if (!mf) mf = (mem_fetch_t*) dq_pop(p_L2c->L2todramqueue);
   if (mf) {
      if (mf->write) {
         L2_write_miss++;
      } else {
         L2_read_miss++;
      }
      p_L2c->m_missTracker->new_miss(mf);
      memlatstat_dram_access(mf, dram_p->id, mf->tlx.bk);
      dram_push(dram_p,
                mf->tlx.bk, mf->tlx.row, mf->tlx.col,
                mf->nbytes_L2, mf->write,
                mf->wid, mf->sid, mf->cache_hits_waiting, mf->addr, mf);
      if (mf->mshr) mshr_update_status(mf->mshr, IN_DRAM_REQ_QUEUE);
   }
}

//Service writes that are finished in Dram 
//only updates the stats and frees the mf
void dramtoL2_service_write(mem_fetch_t * mf) {
   freed_L2write_mfs++;
   free(mf);
   gpgpu_n_processed_writes++;
}

// pop completed memory request from dram and push it to dram-to-L2 queue 
void L2c_get_dram_output ( dram_t* dram_p ) 
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   mem_fetch_t* mf;
   mem_fetch_t* mf_top;
   if ( dq_full(p_L2c->dramtoL2queue) || dq_full(p_L2c->dramtoL2writequeue) ) return;
   mf_top = (mem_fetch_t*) dram_top(dram_p); //test
   mf = (mem_fetch_t*) dram_pop(dram_p);
   assert (mf_top==mf );
   if (mf) {
      if (gpgpu_l2_readoverwrite && mf->write)
         dq_push(p_L2c->dramtoL2writequeue, mf);
      else
         dq_push(p_L2c->dramtoL2queue, mf);
      if (mf->mshr) mshr_update_status(mf->mshr, IN_DRAMTOL2QUEUE);
   }
}

// service memory request in dramtoL2queue, writing to L2 as necessary
// (may cause cache eviction and subsequent writeback) 
void L2c_process_dram_output ( dram_t* dram_p, int dm_id ) 
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   static mem_fetch_t **L2dramout = NULL; 
   static unsigned long long int *wb_addr = NULL;
   if (!L2dramout) L2dramout = (mem_fetch_t**)calloc(gpu_n_mem, sizeof(mem_fetch_t*));
   if (!wb_addr) {
      wb_addr = (unsigned long long int*)calloc(gpu_n_mem, sizeof(unsigned long long int));
      for (unsigned i = 0; i < gpu_n_mem; i++) wb_addr[i] = -1;
   }

   if (L2dramout[dm_id] == NULL) {
      // pop from mshr chain if it is not empty, otherwise, pop a new cacheline from dram output queue
      if (p_L2c->m_mshr->mshr_chain_empty() == false) {
         L2dramout[dm_id] = p_L2c->m_mshr->mshr_chain_top();
         p_L2c->m_mshr->mshr_chain_pop();
      } else {
         L2dramout[dm_id] = (mem_fetch_t*) dq_pop(p_L2c->dramtoL2queue);
         if (!L2dramout[dm_id]) L2dramout[dm_id] = (mem_fetch_t*) dq_pop(p_L2c->dramtoL2writequeue);

         if (L2dramout[dm_id] != NULL) {
            p_L2c->m_mshr->miss_serviced(L2dramout[dm_id]);

            if (p_L2c->m_mshr->mshr_chain_empty() == false) { // possible if this is a L2 writeback
               L2dramout[dm_id] = p_L2c->m_mshr->mshr_chain_top();
               p_L2c->m_mshr->mshr_chain_pop();
            }
         }
      }
   }

   mem_fetch_t* mf = L2dramout[dm_id];
   if (mf) {
      if (!mf->write) { //service L2 read miss

         // it is a pre-fill dramout mf
         if (wb_addr[dm_id] == (unsigned long long int)-1) {
            if ( dq_full(p_L2c->L2tocbqueue) ) goto RETURN;

            if (mf->mshr) mshr_update_status(mf->mshr, IN_L2TOCBQUEUE_MISS);

            //only transfer across icnt once the whole line has been received by L2 cache
            mf->type = REPLY_DATA;
            dq_push(p_L2c->L2tocbqueue, mf);

            assert(mf->sid <= (int)gpu_n_shader);           
            shd_cache_line_t *fetch_line_exist = shd_cache_probe(p_L2c->L2cache, mf->addr);
            if (fetch_line_exist == NULL) {
               wb_addr[dm_id] = L2_shd_cache_fill(p_L2c->L2cache, mf->addr, gpu_sim_cycle );
            }
         }
         // only perform a write on cache eviction (write-back policy)
         // it is the 1st or nth time trial to writeback
         if (wb_addr[dm_id] != (unsigned long long int)-1) {
            // performing L2 writeback (no false sharing for memory-side cache)
            int wb_succeed = L2c_write_back(wb_addr[dm_id], p_L2c->L2cache->line_sz, dm_id ); 
            if (!wb_succeed) goto RETURN; //try again next cycle
         }

         p_L2c->m_missTracker->miss_serviced(mf);
         L2dramout[dm_id] = NULL;
         wb_addr[dm_id] = -1;
      } else { //service L2 write miss
         p_L2c->m_missTracker->miss_serviced(mf);
         dramtoL2_service_write(mf);
         L2dramout[dm_id] = NULL;
         wb_addr[dm_id] = -1;
      }
   }
   RETURN:   
   assert (L2dramout[dm_id] || wb_addr[dm_id] == (unsigned long long int)-1);
}

// Writeback from L2 to DRAM: 
// - Takes in memory address and their parameters and pushes to dram request queue
// - This is used only for L2 writeback 
unsigned char L2c_write_back(unsigned long long int addr, int bsize, int dram_id ) 
{
   addrdec_t tlx;
   addrdec_tlx(addr,&tlx);

   assert(dram[dram_id]->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[dram_id]->m_L2cache);

   if ( dq_full(p_L2c->L2todram_wbqueue) ) return 0;

   mem_fetch_t *mf;

   mf = (mem_fetch_t*) malloc(sizeof(mem_fetch_t));
   made_write_mfs++;
   mf->request_uid = g_next_mf_request_uid++;
   mf->addr = addr;
   mf->nbytes_L1 = bsize + READ_PACKET_SIZE;
   mf->txbytes_L1 = 0;
   mf->rxbytes_L1 = 0;  
   mf->nbytes_L2 = bsize;
   mf->sid = gpu_n_shader; // (gpu_n_shader+1);
   mf->wid = 0;
   mf->txbytes_L2 = 0;
   mf->rxbytes_L2 = 0;
   mf->mshr = NULL;
   mf->pc = -1; // disable ptx_file_line_stats
   mf->write = 1; // it is writeback
   mf->mem_acc = L2_WRBK_ACC; 
   memlatstat_start(mf);
   mf->tlx = tlx;
   mf->bank = mf->tlx.bk;
   mf->chip = mf->tlx.chip;


   //writeback
   mf->type = L2_WTBK_DATA;
   if (!dq_push(p_L2c->L2todram_wbqueue, mf)) assert(0);
   gpgpu_n_sent_writes++;
   return 1;
}

unsigned int L2c_cache_flush ( dram_t* dram_p) {
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   shd_cache_t *cp = p_L2c->L2cache; 
   int dirty_lines_flushed = 0 ;
   for (unsigned i = 0; i < cp->nset * cp->assoc ; i++) {
      if ( (cp->lines[i].status & (DIRTY|VALID)) == (DIRTY|VALID) ) {
         dirty_lines_flushed++;
      }
      cp->lines[i].status &= ~VALID;
      cp->lines[i].status &= ~DIRTY;
   }
   return dirty_lines_flushed;
}

void L2c_init_stat()
{
   L2_cbtoL2length = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_cbtoL2writelength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_L2tocblength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_dramtoL2length = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_dramtoL2writelength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
   L2_L2todramlength = (unsigned int*) calloc(gpu_n_mem, sizeof(unsigned int));
}

void L2c_update_stat( dram_t* dram_p)
{
   assert(dram_p->m_L2cache != NULL);
   L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram_p->m_L2cache);

   int i = dram_p->id;

   if (p_L2c->cbtoL2queue->length > L2_cbtoL2length[i])
      L2_cbtoL2length[i] = p_L2c->cbtoL2queue->length;
   if (p_L2c->cbtoL2writequeue->length > L2_cbtoL2writelength[i])
      L2_cbtoL2writelength[i] = p_L2c->cbtoL2writequeue->length;
   if (p_L2c->L2tocbqueue->length > L2_L2tocblength[i])
      L2_L2tocblength[i] = p_L2c->L2tocbqueue->length;
   if (p_L2c->dramtoL2queue->length > L2_dramtoL2length[i])
      L2_dramtoL2length[i] = p_L2c->dramtoL2queue->length;
   if (p_L2c->dramtoL2writequeue->length > L2_dramtoL2writelength[i])
      L2_dramtoL2writelength[i] = p_L2c->dramtoL2writequeue->length;
   if (p_L2c->L2todramqueue->length > L2_L2todramlength[i])
      L2_L2todramlength[i] = p_L2c->L2todramqueue->length;
}

void L2c_print_stat( )
{
   unsigned i;

   printf("                                     ");
   for (i=0;i<gpu_n_mem;i++) {
      printf(" dram[%d]", i);
   }
   printf("\n");

   printf("cbtoL2 queue maximum length         ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_cbtoL2length[i]);
   }
   printf("\n");

   printf("cbtoL2 write queue maximum length   ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_cbtoL2writelength[i]);
   }
   printf("\n");

   printf("L2tocb queue maximum length         =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_L2tocblength[i]);
   }
   printf("\n");

   printf("dramtoL2 queue maximum length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_dramtoL2length[i]);
   }
   printf("\n");

   printf("dramtoL2 write queue maximum length ="); 
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_dramtoL2writelength[i]);
   }
   printf("\n");

   printf("L2todram queue maximum length       =");
   for (i=0;i<gpu_n_mem;i++) {
      printf("%8d", L2_L2todramlength[i]);
   }
   printf("\n");
}

void L2c_print_cache_stat()
{
   unsigned i;
   int j, k;
   for (i=0,j=0,k=0;i<gpu_n_mem;i++) {
      assert(dram[i]->m_L2cache != NULL);
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);

      shd_cache_print(p_L2c->L2cache,stdout);
      j += p_L2c->L2cache->miss;
      k += p_L2c->L2cache->access;
      p_L2c->m_mshr->print_stat(stdout); 
      p_L2c->m_missTracker->print_stat(stdout);
      p_L2c->m_accessLocality->print_stat(stdout, false);
   }
   printf("L2 Cache Total Miss Rate = %0.3f\n", (float)j/k);
}

void L2c_print_debug( )
{
   unsigned i;

   printf("                                     ");
   for (i=0;i<gpu_n_mem;i++) {
      printf(" dram[%d]", i);
   }
   printf("\n");

   printf("cbtoL2 queue length         ="); 
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->cbtoL2queue->length);
   }
   printf("\n");

   printf("cbtoL2 write queue length   ="); 
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->cbtoL2writequeue->length);
   }
   printf("\n");

   printf("L2tocb queue length         =");
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->L2tocbqueue->length);
   }
   printf("\n");

   printf("dramtoL2 queue length       =");
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->dramtoL2queue->length);
   }
   printf("\n");

   printf("dramtoL2 write queue length ="); 
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->dramtoL2writequeue->length);
   }
   printf("\n");

   printf("L2todram queue length       =");
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->L2todramqueue->length);
   }
   printf("\n");

   printf("L2todram writeback queue length       =");
   for (i=0;i<gpu_n_mem;i++) {
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);
      printf("%8d", p_L2c->L2todram_wbqueue->length);
   }
   printf("\n");
}

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

void L2c_log(int task)
{
   unsigned i;
   static void ** cbtol2_Dist   ;  
   static void ** cbtoL2wr_Dist  ;  
   static void ** L2tocb_Dist     ; 
   static void ** dramtoL2_Dist   ;
   static void ** dramtoL2wr_Dist  ;
   static void ** L2todram_Dist    ;
   static void ** L2todram_wb_Dist ;
   if (task == CREATELOG) {
      cbtol2_Dist = (void **)     calloc(gpu_n_mem,sizeof(void*));
      cbtoL2wr_Dist = (void **)    calloc(gpu_n_mem,sizeof(void*));
      L2tocb_Dist =   (void **)   calloc(gpu_n_mem,sizeof(void*));
      dramtoL2_Dist =   (void **)calloc(gpu_n_mem,sizeof(void*));
      dramtoL2wr_Dist  =  (void **)calloc(gpu_n_mem,sizeof(void*));
      L2todram_Dist    = (void **)calloc(gpu_n_mem,sizeof(void*));
      L2todram_wb_Dist = (void **)calloc(gpu_n_mem,sizeof(void*));

      for (i=0;i<gpu_n_mem;i++) {
         assert(dram[i]->m_L2cache != NULL);
         L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);

         cbtol2_Dist[i]      = StatCreate("cbtoL2",1,p_L2c->cbtoL2queue->max_len);
         cbtoL2wr_Dist[i]    = StatCreate("cbtoL2write",1,p_L2c->cbtoL2writequeue->max_len);
         L2tocb_Dist[i]      = StatCreate("L2tocb",1,p_L2c->L2tocbqueue->max_len);
         dramtoL2_Dist[i]    = StatCreate("dramtoL2",1,p_L2c->dramtoL2queue->max_len);
         dramtoL2wr_Dist[i]  = StatCreate("dramtoL2write",1,p_L2c->dramtoL2writequeue->max_len);
         L2todram_Dist[i]    = StatCreate("L2todram",1,p_L2c->L2todramqueue->max_len);
         L2todram_wb_Dist[i] = StatCreate("L2todram_wb",1,p_L2c->L2todram_wbqueue->max_len);
      }
   } else if (task == SAMPLELOG) {
      for (i=0;i<gpu_n_mem;i++) {
         assert(dram[i]->m_L2cache != NULL);
         L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);

         StatAddSample(cbtol2_Dist[i],       p_L2c->cbtoL2queue->length);
         StatAddSample(cbtoL2wr_Dist[i],     p_L2c->cbtoL2writequeue->length);
         StatAddSample(L2tocb_Dist[i],       p_L2c->L2tocbqueue->length);
         StatAddSample(dramtoL2_Dist[i],     p_L2c->dramtoL2queue->length);
         StatAddSample(dramtoL2wr_Dist[i],   p_L2c->dramtoL2writequeue->length);
         StatAddSample(L2todram_Dist[i],     p_L2c->L2todramqueue->length);
         StatAddSample(L2todram_wb_Dist[i],  p_L2c->L2todram_wbqueue->length);
      }
   } else if (task == DUMPLOG) {
      for (i=0;i<gpu_n_mem;i++) {
         printf ("Queue Length DRAM[%d] ",i); StatDisp(cbtol2_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i); StatDisp(cbtoL2wr_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i); StatDisp(L2tocb_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i); StatDisp(dramtoL2_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i); StatDisp(dramtoL2wr_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i); StatDisp(L2todram_Dist[i]);
         printf ("Queue Length DRAM[%d] ",i); StatDisp(L2todram_wb_Dist[i]);
      } 
   }
}

void L2c_latency_log_dump()
{
   unsigned i;
   for (i=0;i<gpu_n_mem;i++) {
      assert(dram[i]->m_L2cache != NULL);
      L2cacheblk *p_L2c = reinterpret_cast<L2cacheblk*>(dram[i]->m_L2cache);

      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->cbtoL2queue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->cbtoL2writequeue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->L2tocbqueue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->dramtoL2queue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->dramtoL2writequeue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->L2todramqueue->lat_stat);
      printf ("(LOGB2)Latency DRAM[%d] ",i); StatDisp(p_L2c->L2todram_wbqueue->lat_stat);
   }
}


