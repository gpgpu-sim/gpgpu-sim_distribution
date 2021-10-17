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
#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include <vector>
#include "../gpgpu-sim/visualizer.h"
#include "XML_Parse.h"
#include "array.h"
#include "basic_components.h"
#include "cacti/arbiter.h"
#include "cacti/area.h"
#include "cacti/decoder.h"
#include "cacti/parameter.h"
#include "cacti/router.h"
#include "core.h"
#include "iocontrollers.h"
#include "memoryctrl.h"
#include "noc.h"
#include "sharedcache.h"

class Processor : public Component {
 public:
  ParseXML *XML;
  vector<Core *> cores;
  vector<SharedCache *> l2array;
  vector<SharedCache *> l3array;
  vector<SharedCache *> l1dirarray;
  vector<SharedCache *> l2dirarray;
  vector<NoC *> nocs;
  MemoryController *mc;
  NIUController *niu;
  PCIeController *pcie;
  FlashController *flashcontroller;
  InputParameter interface_ip;
  double exClockRate;
  ProcParam procdynp;
  // for debugging nonlinear model
  double dyn_power_before_scaling;

  // wire	globalInterconnect;
  // clock_network globalClock;
  Component core, l2, l3, l1dir, l2dir, noc, mcs, cc, nius, pcies,
      flashcontrollers;
  int numCore, numL2, numL3, numNOC, numL1Dir, numL2Dir;
  Processor(ParseXML *XML_interface);
  void compute();
  void set_proc_param();
  void visualizer_print(gzFile visualizer_file);
  void displayEnergy(uint32_t indent = 0, int plevel = 100,
                     bool is_tdp_parm = true);
  void displayDeviceType(int device_type_, uint32_t indent = 0);
  void displayInterconnectType(int interconnect_type_, uint32_t indent = 0);
  double l2_power;
  double idle_core_power;

  double get_const_dynamic_power() {
    double constpart = 0;
    constpart += (mc->frontend->power.readOp.dynamic * 0.1 *
                  mc->frontend->mcp.clockRate * mc->frontend->mcp.num_mcs *
                  mc->frontend->mcp.executionTime);
    constpart +=
        (mc->transecEngine->power.readOp.dynamic * 0.1 *
         mc->transecEngine->mcp.clockRate * mc->transecEngine->mcp.num_mcs *
         mc->transecEngine->mcp.executionTime);
    constpart += (mc->PHY->power.readOp.dynamic * 0.1 * mc->PHY->mcp.clockRate *
                  mc->PHY->mcp.num_mcs * mc->PHY->mcp.executionTime);
    constpart +=
        (cores[0]->exu->exeu->base_energy / cores[0]->exu->exeu->clockRate) *
        (cores[0]->exu->rf_fu_clockRate / cores[0]->exu->clockRate);
    constpart +=
        (cores[0]->exu->mul->base_energy / cores[0]->exu->mul->clockRate);
    constpart +=
        (cores[0]->exu->fp_u->base_energy / cores[0]->exu->fp_u->clockRate);
    return constpart;
  }
#define COALESCE_SCALE 1
  double get_coefficient_readcoalescing() {
    double value = 0;
    double perAccessCoalescingEnergy =
        COALESCE_SCALE *
        ((0.443e-3) * (0.5e-9) * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd) /
        (1 * 1);
    value += mc->frontend->PRT->local_result.power.readOp.dynamic;
    value += mc->frontend->threadMasks->local_result.power.readOp.dynamic;
    value += mc->frontend->PRC->local_result.power.readOp.dynamic;
    value += perAccessCoalescingEnergy;
    return value;
  }
  double get_coefficient_writecoalescing() {
    double value = 0;
    double perAccessCoalescingEnergy =
        COALESCE_SCALE *
        ((0.443e-3) * (0.5e-9) * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd) /
        (1 * 1);
    value += (mc->frontend->PRT->local_result.power.writeOp.dynamic);
    value += mc->frontend->threadMasks->local_result.power.writeOp.dynamic;
    value += mc->frontend->PRC->local_result.power.writeOp.dynamic;
    value += perAccessCoalescingEnergy;
    return value;
  }

  double get_coefficient_noc_accesses() {
    double read_coef = 0;
    // the 32/4 is applied to the NoC access counters (32/4*L2 cache access)
    read_coef += nocs[0]->router->buffer.power.readOp.dynamic;
    read_coef += nocs[0]->router->buffer.power.writeOp.dynamic;
    read_coef += nocs[0]->router->crossbar.power.readOp.dynamic;
    read_coef += nocs[0]->router->arbiter.power.readOp.dynamic;
    return read_coef;
  }

  double get_coefficient_l2_read_hits() {
    double read_coef = 0;
    if (XML->sys.number_of_L2s > 0)
      read_coef =
          l2array[0]->unicache.caches->local_result.power.readOp.dynamic;
    return read_coef;
  }

  double get_coefficient_l2_read_misses() {
    double read_coef = 0;
    if (XML->sys.number_of_L2s > 0)
      read_coef =
          l2array[0]
              ->unicache.caches->local_result.tag_array2->power.readOp.dynamic;
    return read_coef;
  }

  double get_coefficient_l2_write_hits() {
    double read_coef = 0;
    if (XML->sys.number_of_L2s > 0)
      read_coef =
          l2array[0]->unicache.caches->local_result.power.writeOp.dynamic;
    return read_coef;
  }
  double get_coefficient_l2_write_misses() {
    double read_coef = 0;
    if (XML->sys.number_of_L2s > 0) {
      read_coef = l2array[0]
                      ->unicache.caches->local_result.tag_array2->power.writeOp
                      .dynamic;  //*(32/4); // removed by Jingwen, the scaling
                                 // of 32/4 is not used in the mcpat
      read_coef +=
          l2array[0]->unicache.caches->local_result.power.writeOp.dynamic;
      read_coef +=
          l2array[0]->unicache.missb->local_result.power.searchOp.dynamic;
      read_coef +=
          l2array[0]->unicache.missb->local_result.power.writeOp.dynamic;
      read_coef +=
          l2array[0]->unicache.ifb->local_result.power.searchOp.dynamic;
      read_coef += l2array[0]->unicache.ifb->local_result.power.writeOp.dynamic;
      read_coef +=
          l2array[0]->unicache.prefetchb->local_result.power.searchOp.dynamic;
      read_coef +=
          l2array[0]->unicache.prefetchb->local_result.power.writeOp.dynamic;
      read_coef +=
          l2array[0]->unicache.wbb->local_result.power.searchOp.dynamic;
      read_coef += l2array[0]->unicache.wbb->local_result.power.writeOp.dynamic;
    }

    return read_coef;
  }

  double get_coefficient_mem_reads() {
    double value = 0;
    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.searchOp.dynamic);

    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.readOp.dynamic);

    // TODO: Jingwen this should only compute for one time?
    // value+=(mc->frontend->mcp.llcBlockSize*8.0/mc->frontend->mcp.dataBusWidth*mc->frontend->mcp.dataBusWidth/72)
    //*(mc->frontend->frontendBuffer->local_result.power.readOp.dynamic);

    value += (mc->frontend->mcp.llcBlockSize * 8.0 / mc->mcp.dataBusWidth) *
             (mc->frontend->readBuffer->local_result.power.readOp.dynamic);

    value += (mc->frontend->mcp.llcBlockSize * 8.0 / mc->mcp.dataBusWidth) *
             (mc->frontend->readBuffer->local_result.power.writeOp.dynamic);

    value += mc->dram->dramp.rd_coeff;
    /*
            value+=mc->frontend->PRT->local_result.power.readOp.dynamic;
            value+=mc->frontend->threadMasks->local_result.power.readOp.dynamic;
            value+=mc->frontend->PRC->local_result.power.readOp.dynamic;
            value+=perAccessCoalescingEnergy;
            */
    value += (mc->transecEngine->mcp.llcBlockSize * 8.0 /
              mc->transecEngine->mcp.dataBusWidth *
              mc->transecEngine->power_t.readOp.dynamic);

    // if mcp.type ==1 TODO: add this check here
    value += (mc->PHY->power_t.readOp.dynamic) * (mc->PHY->mcp.llcBlockSize) *
             8 / 1e9 / mc->PHY->mcp.executionTime *
             (mc->PHY->mcp.executionTime);
    // printf("MC PHY read power coeff:
    // %f\n",(mc->PHY->power_t.readOp.dynamic)*(mc->PHY->mcp.llcBlockSize)*8/1e9/mc->PHY->mcp.executionTime*(mc->PHY->mcp.executionTime));
    // printf("MC trans read power coeff:
    // %f\n",(mc->transecEngine->mcp.llcBlockSize*8.0/mc->transecEngine->mcp.dataBusWidth*mc->transecEngine->power_t.readOp.dynamic));

    // TODO: Jingwen nocs stats should not be here
    //		value+= nocs[0]->router->buffer.power.readOp.dynamic*(32/4);
    //		value+= nocs[0]->router->buffer.power.writeOp.dynamic*(32/4);
    //		value+= nocs[0]->router->crossbar.power.readOp.dynamic*(32/4);
    //		value+= nocs[0]->router->arbiter.power.readOp.dynamic*(32/4);

    // return 0.4*value;
    return value;
  }

  double get_coefficient_mem_writes() {
    double value = 0;

    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.searchOp.dynamic);

    value +=
        (mc->frontend->mcp.llcBlockSize * 8.0 / mc->frontend->mcp.dataBusWidth *
         mc->frontend->mcp.dataBusWidth / 72) *
        (mc->frontend->frontendBuffer->local_result.power.writeOp.dynamic);

    // value+=(mc->frontend->mcp.llcBlockSize*8.0/mc->frontend->mcp.dataBusWidth*mc->frontend->mcp.dataBusWidth/72)*
    // (mc->frontend->frontendBuffer->local_result.power.writeOp.dynamic);

    value += (mc->frontend->mcp.llcBlockSize * 8.0 /
              mc->frontend->mcp.dataBusWidth) *
             (mc->frontend->writeBuffer->local_result.power.readOp.dynamic);

    value += (mc->frontend->mcp.llcBlockSize * 8.0 /
              mc->frontend->mcp.dataBusWidth) *
             (mc->frontend->writeBuffer->local_result.power.writeOp.dynamic);

    value += mc->dram->dramp.wr_coeff;
    /*
            value+=(mc->frontend->PRT->local_result.power.writeOp.dynamic);

            value+=mc->frontend->threadMasks->local_result.power.writeOp.dynamic;

            value+=mc->frontend->PRC->local_result.power.writeOp.dynamic;

            value+=perAccessCoalescingEnergy;
            */

    value += (mc->transecEngine->mcp.llcBlockSize * 8.0 /
              mc->transecEngine->mcp.dataBusWidth *
              mc->transecEngine->power_t.readOp.dynamic);

    // if mcp.type ==1 TODO: add this check here
    value += (mc->PHY->power_t.readOp.dynamic) * (mc->PHY->mcp.llcBlockSize) *
             8 / 1e9 / mc->PHY->mcp.executionTime *
             (mc->PHY->mcp.executionTime);

    // TODO: Jingwen nocs stats should not be here
    //		value+= nocs[0]->router->buffer.power.readOp.dynamic*(32/4);
    //
    //		value+= nocs[0]->router->buffer.power.writeOp.dynamic*(32/4);
    //
    //		value+= nocs[0]->router->crossbar.power.readOp.dynamic*(32/4);
    //
    //		value+= nocs[0]->router->arbiter.power.readOp.dynamic*(32/4);
    //
    // return 0.4*value;
    return value;
  }

  double get_coefficient_mem_pre() {
    double value = 0;
    value += mc->dram->dramp.pre_coeff;
    // return 0.4*value;
    return value;
  }

  // nonlinear scale
  void nonlinear_scale(int, double, int);
  void coefficient_scale();
  void iterative_lse(double *, double *);

  ~Processor();
};

#endif /* PROCESSOR_H_ */
