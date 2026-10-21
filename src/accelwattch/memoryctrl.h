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

#ifndef MEMORYCTRL_H_
#define MEMORYCTRL_H_

#include "XML_Parse.h"
#include "cacti/parameter.h"
// #include "io.h"
#include "array.h"
// #include "Undifferentiated_Core_Area.h"
#include <vector>
#include "basic_components.h"

class MCBackend : public Component {
 public:
  InputParameter l_ip;
  uca_org_t local_result;
  enum MemoryCtrl_type mc_type;
  MCParam mcp;
  statsDef tdp_stats;
  statsDef rtp_stats;
  statsDef stats_t;
  powerDef power_t;
  MCBackend(InputParameter *interface_ip_, const MCParam &mcp_,
            enum MemoryCtrl_type mc_type_);
  void compute();
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~MCBackend(){};
};

class MCPHY : public Component {
 public:
  InputParameter l_ip;
  uca_org_t local_result;
  enum MemoryCtrl_type mc_type;
  MCParam mcp;
  statsDef tdp_stats;
  statsDef rtp_stats;
  statsDef stats_t;
  powerDef power_t;
  MCPHY(InputParameter *interface_ip_, const MCParam &mcp_,
        enum MemoryCtrl_type mc_type_);
  void compute();
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~MCPHY(){};
};

class MCFrontEnd : public Component {
 public:
  ParseXML *XML;
  InputParameter interface_ip;
  enum MemoryCtrl_type mc_type;
  MCParam mcp;
  selection_logic *MC_arb;
  ArrayST *frontendBuffer;
  ArrayST *readBuffer;
  ArrayST *writeBuffer;

  ArrayST *PRT;
  ArrayST *threadMasks;
  ArrayST *PRC;
  double coalesce_scale;

  MCFrontEnd(ParseXML *XML_interface, InputParameter *interface_ip_,
             const MCParam &mcp_, enum MemoryCtrl_type mc_type_);
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~MCFrontEnd();
};

class DRAM : public Component {
 public:
  ParseXML *XML;
  InputParameter interface_ip;
  enum Dram_type dram_type;
  DRAMParam dramp;
  powerDef power_t;
  DRAM(ParseXML *XML_interface, InputParameter *interface_ip_,
       enum Dram_type dram_type_);
  void set_dram_param();
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~DRAM();
};

class MemoryController : public Component {
 public:
  ParseXML *XML;
  InputParameter interface_ip;
  enum MemoryCtrl_type mc_type;
  MCParam mcp;
  DRAM *dram;
  MCFrontEnd *frontend;
  MCBackend *transecEngine;
  MCPHY *PHY;
  Pipeline *pipeLogic;

  // Add coalescing logic related modules with each memory controller --Syed
  // Gilani

  // clock_network clockNetwork;
  MemoryController(ParseXML *XML_interface, InputParameter *interface_ip_,
                   enum MemoryCtrl_type mc_type_, enum Dram_type dram_type_);
  void set_mc_param();
  void computeEnergy(bool is_tdp = true);
  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  ~MemoryController();
};
#endif /* MEMORYCTRL_H_ */
