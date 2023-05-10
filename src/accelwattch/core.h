/*****************************************************************************
 *                                McPAT
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/
/********************************************************************
 *      Modified by:
 ** Jingwen Leng, Univeristy of Texas, Austin                   * Syed Gilani,
 *University of Wisconsin–Madison                * Tayler Hetherington,
 *University of British Columbia         * Ahmed ElTantawy, University of
 *British Columbia             *
 ********************************************************************/

#ifndef CORE_H_
#define CORE_H_

#include "XML_Parse.h"
#include "array.h"
#include "basic_components.h"
#include "cacti/arbiter.h"
#include "cacti/crossbar.h"
#include "cacti/parameter.h"
#include "interconnect.h"
#include "logic.h"
#include "noc.h"
#include "sharedcache.h"

class BranchPredictor : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  CoreDynParam coredynp;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  ArrayST *globalBPT;
  ArrayST *localBPT;
  ArrayST *L1_localBPT;
  ArrayST *L2_localBPT;
  ArrayST *chooser;
  ArrayST *RAS;
  bool exist;

  BranchPredictor(ParseXML *XML_interface, int ithCore_,
                  InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
                  bool exsit = true);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~BranchPredictor();
};

class InstFetchU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  CoreDynParam coredynp;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  enum Cache_policy cache_p;
  InstCache icache;
  ArrayST *IB;
  ArrayST *BTB;
  BranchPredictor *BPT;
  inst_decoder *ID_inst;
  inst_decoder *ID_operand;
  inst_decoder *ID_misc;
  bool exist;

  InstFetchU(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exsit = true);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~InstFetchU();
};

class SchedulerU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  CoreDynParam coredynp;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  double Iw_height, fp_Iw_height, ROB_height;
  ArrayST *int_inst_window;
  ArrayST *fp_inst_window;
  ArrayST *ROB;
  selection_logic *instruction_selection;
  bool exist;

  SchedulerU(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exist_ = true);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~SchedulerU();
};

class RENAMINGU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  double clockRate, executionTime;
  CoreDynParam coredynp;
  ArrayST *iFRAT;
  ArrayST *fFRAT;
  ArrayST *iRRAT;
  ArrayST *fRRAT;
  ArrayST *ifreeL;
  ArrayST *ffreeL;
  dep_resource_conflict_check *idcl;
  dep_resource_conflict_check *fdcl;
  ArrayST *RAHT;  // register alias history table Used to store GC
  bool exist;

  RENAMINGU(ParseXML *XML_interface, int ithCore_,
            InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
            bool exist_ = true);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~RENAMINGU();
};

class LoadStoreU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  CoreDynParam coredynp;
  enum Cache_policy cache_p;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  double lsq_height;
  DataCache dcache;
  DataCache ccache;
  DataCache tcache;
  DataCache sharedmemory;
  ArrayST *LSQ;  // it is actually the store queue but for inorder processors it
                 // serves as both loadQ and StoreQ
  ArrayST *LoadQ;
  vector<NoC *> nocs;
  bool exist;
  Crossbar *xbar_shared;
  Component noc;
  LoadStoreU(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exist_ = true);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  void displayDeviceType(int device_type_,
                         uint32_t indent);  // Added by Syed Gilani

  ~LoadStoreU();
};

class MemManU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  CoreDynParam coredynp;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  ArrayST *itlb;
  ArrayST *dtlb;
  bool exist;

  MemManU(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_,
          const CoreDynParam &dyn_p_, bool exist_ = false);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~MemManU();
};

class RegFU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  CoreDynParam coredynp;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  double int_regfile_height, fp_regfile_height;
  ArrayST *IRF;
  ArrayST *FRF;
  ArrayST *RFWIN;
  ArrayST *OPC;  // Operand collectors
  bool exist;
  double exClockRate;
  // OC Modelling (Syed)
  Crossbar *xbar_rfu;
  MCPAT_Arbiter *arbiter_rfu;
  RegFU(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_,
        const CoreDynParam &dyn_p_, double exClockRate, bool exist_ = true);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~RegFU();
};

class EXECU : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  double clockRate, executionTime;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  double lsq_height;
  CoreDynParam coredynp;
  RegFU *rfu;
  SchedulerU *scheu;
  FunctionalUnit *fp_u;
  FunctionalUnit *exeu;
  FunctionalUnit *mul;
  interconnect *int_bypass;
  interconnect *intTagBypass;
  interconnect *int_mul_bypass;
  interconnect *intTag_mul_Bypass;
  interconnect *fp_bypass;
  interconnect *fpTagBypass;
  bool exist;
  double rf_fu_clockRate;
  Component bypass;

  EXECU(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_,
        double lsq_height_, const CoreDynParam &dyn_p_, double exClockRate,
        bool exist_);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~EXECU();
};

class Core : public Component {
 public:
  ParseXML *XML;
  int ithCore;
  InputParameter interface_ip;
  double clockRate, executionTime;
  double exClockRate;
  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  InstFetchU *ifu;
  LoadStoreU *lsu;
  MemManU *mmu;
  EXECU *exu;
  RENAMINGU *rnu;
  double IdleCoreEnergy;
  double IdlePower_PerCore;
  Pipeline *corepipe;
  UndiffCore *undiffCore;
  SharedCache *l2cache;
  CoreDynParam coredynp;
  double Pipeline_energy;
  // full_decoder 	inst_decoder;
  // clock_network	clockNetwork;
  Core(ParseXML *XML_interface, int ithCore_, InputParameter *interface_ip_);
  void set_core_param();
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);

  float get_coefficient_icache_hits() {
    // return 1.5*ifu->icache.caches->local_result.power.readOp.dynamic;
    return ifu->icache.caches->local_result.power.readOp.dynamic;
  }

  float get_coefficient_icache_misses() {
    float value = 0;
    value += ifu->icache.caches->local_result.power.writeOp.dynamic;
    value += ifu->icache.caches->local_result.power.readOp.dynamic;
    value += ifu->icache.missb->local_result.power.searchOp.dynamic;
    value += ifu->icache.missb->local_result.power.writeOp.dynamic;
    value += ifu->icache.ifb->local_result.power.searchOp.dynamic;
    value += ifu->icache.ifb->local_result.power.writeOp.dynamic;
    value += ifu->icache.prefetchb->local_result.power.searchOp.dynamic;
    value += ifu->icache.prefetchb->local_result.power.writeOp.dynamic;
    return value;
  }

  float get_coefficient_tot_insts() {
    float value = 0;
    value += ifu->IB->local_result.power.readOp.dynamic;
    value += ifu->IB->local_result.power.writeOp.dynamic;
    value += ifu->ID_inst->power_t.readOp.dynamic;
    value += ifu->ID_operand->power_t.readOp.dynamic;
    value += ifu->ID_misc->power_t.readOp.dynamic;
    return value;
  }

  float get_coefficient_fpint_insts() {
    float value = 0;
    value += exu->scheu->int_inst_window->local_result.power.readOp.dynamic;
    value +=
        2 * exu->scheu->int_inst_window->local_result.power.searchOp.dynamic;
    value += exu->scheu->int_inst_window->local_result.power.writeOp.dynamic;
    value += exu->scheu->instruction_selection->power.readOp.dynamic;
    return value;
  }

  float get_coefficient_dcache_readhits() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.readOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    // return 0.5*value;
    return value;
  }
  float get_coefficient_dcache_readmisses() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->dcache.prefetchb->local_result.power.writeOp.dynamic;

    // return 0.5*value;
    return value;
  }
  float get_coefficient_dcache_writehits() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.writeOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    return value;
  }
  float get_coefficient_dcache_writemisses() {
    float value = 0;
    value += lsu->dcache.caches->local_result.power.writeOp.dynamic;
    value += lsu->dcache.caches->local_result.tag_array2->power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.caches->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.missb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.ifb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.prefetchb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.wbb->local_result.power.searchOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.missb->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.ifb->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.prefetchb->local_result.power.writeOp.dynamic
                 : 0;
    value += (lsu->cache_p == Write_back)
                 ? lsu->dcache.wbb->local_result.power.writeOp.dynamic
                 : 0;
    // return 1.6*value;
    return value;
  }

  float get_coefficient_tcache_readhits() {
    float value = 0;
    value += lsu->tcache.caches->local_result.power.readOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    // return 0.2*value;
    return value;
  }
  float get_coefficient_tcache_readmisses() {
    float value = 0;
    value += lsu->tcache.caches->local_result.power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.writeOp.dynamic;

    // return 0.2*value;
    return value;
  }
  float get_coefficient_tcache_readmisses1() {
    return lsu->tcache.caches->local_result.power.readOp.dynamic;
  }
  float get_coefficient_tcache_readmisses2() {
    float value = 0;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->tcache.prefetchb->local_result.power.writeOp.dynamic;
    return value;
  }

  float get_coefficient_ccache_readhits() {
    // return 1.2*lsu->ccache.caches->local_result.power.readOp.dynamic+lsu->xbar_shared->power.readOp.dynamic;
    // return 1.2*lsu->ccache.caches->local_result.power.readOp.dynamic+lsu->xbar_shared->power.readOp.dynamic;
    return lsu->ccache.caches->local_result.power.readOp.dynamic +
           lsu->xbar_shared->power.readOp.dynamic;
  }
  float get_coefficient_ccache_readmisses() {
    float value = 0;
    value += lsu->ccache.caches->local_result.power.readOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.missb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.ifb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.prefetchb->local_result.power.searchOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.missb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.ifb->local_result.power.writeOp.dynamic;
    value += (lsu->cache_p == Write_back)
                 ? 0
                 : lsu->ccache.prefetchb->local_result.power.writeOp.dynamic;
    return value;
  }

  float get_coefficient_sharedmemory_readhits() {
    float value = 0;
    value += lsu->sharedmemory.caches->local_result.power.readOp.dynamic;
    value += lsu->xbar_shared->power.readOp.dynamic;
    // return 3*value;
    return value;
  }

  float get_coefficient_lsq_accesses() {
    float value = 0;
    // Changed by Syed -- We have removed LSQ
    // value+=2*lsu->LSQ->local_result.power.searchOp.dynamic;
    // value+=2*lsu->LSQ->local_result.power.readOp.dynamic;
    // value+=2*lsu->LSQ->local_result.power.writeOp.dynamic;
    return value;
  }

  float get_coefficient_regreads_accesses() {
    float value = 0;
    value += ((exu->rfu->IRF->local_result.power.readOp.dynamic / 32) *
              (4 * 2) /*/1.5*/);
    value += exu->rfu->xbar_rfu->power.readOp.dynamic / (32 /**1.5*/);
    value += (exu->rfu->arbiter_rfu->power.readOp.dynamic / 32 /**1.5)*/);
    value += exu->rfu->OPC->local_result.power.readOp.dynamic /*/1.5*/;
    return value;
  }

  float get_coefficient_regwrites_accesses() {
    return ((exu->rfu->IRF->local_result.power.writeOp.dynamic / 32) *
            (4 * 2) /*/1.5*/);
  }

  float get_coefficient_noregfileops_accesses() {
    return ((exu->rfu->xbar_rfu->power.readOp.dynamic / (32 /**1.5*/)) +
            (exu->rfu->arbiter_rfu->power.readOp.dynamic / (32 /**1.5*/)) +
            (exu->rfu->OPC->local_result.power.readOp.dynamic /*/(1.5)*/));
  }

  float get_coefficient_ialu_accesses() {
    // return 10*exu->exeu->per_access_energy*g_tp.sckt_co_eff;
    return exu->exeu->per_access_energy * g_tp.sckt_co_eff;
  }

  float get_coefficient_sfu_accesses() {
    return exu->mul->per_access_energy * g_tp.sckt_co_eff;
    // return 2.6*exu->mul->per_access_energy*g_tp.sckt_co_eff;
  }

  float get_coefficient_fpu_accesses() {
    // return 3.2*exu->fp_u->per_access_energy*g_tp.sckt_co_eff;
    return exu->fp_u->per_access_energy * g_tp.sckt_co_eff;
  }

  float get_coefficient_duty_cycle() {
    float value = 0;
    float num_units = 4.0;
    value = XML->sys.total_cycles * XML->sys.number_of_cores;
    value *= coredynp.num_pipelines;
    value /= num_units;
    value *= corepipe->power.readOp.dynamic;
    value *= 3;
    return value;
    // return 1.5*value;
  }

  void compute();
  ~Core();
};

#endif /* CORE_H_ */
