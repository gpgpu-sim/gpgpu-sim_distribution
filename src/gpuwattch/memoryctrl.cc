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
*      Modified by:												   *
*      Jingwen Leng, Univeristy of Texas, Austin                   *
*      Syed Gilani, University of Wisconsin–Madison                *
*      Tayler Hetherington, University of British Columbia         *
*      Ahmed ElTantawy, University of British Columbia             *
********************************************************************/
#include "io.h"
#include "parameter.h"
#include "const.h"
#include "logic.h"
#include "cacti/basic_circuit.h"
#include <iostream>
#include <algorithm>
#include "XML_Parse.h"
#include <string>
#include <cmath>
#include <assert.h>
#include "memoryctrl.h"
#include "basic_components.h"

/* overview of MC models:
 * McPAT memory controllers are modeled according to large number of industrial data points.
 * The Basic memory controller architecture is base on the Synopsis designs
 * (DesignWare DDR2/DDR3-Lite memory controllers and DDR2/DDR3-Lite protocol controllers)
 * as in Cadence ChipEstimator Tool
 *
 * An MC has 3 parts as shown in this design. McPAT models both high performance MC
 * based on Niagara processor designs and curving and low power MC based on data points in
 * Cadence ChipEstimator Tool.
 *
 * The frontend is modeled analytically, the backend is modeled empirically according to
 * DDR2/DDR3-Lite protocol controllers in Cadence ChipEstimator Tool
 * The PHY is modeled based on
 * "A 100mW 9.6Gb/s Transceiver in 90nm CMOS for next-generation memory interfaces ," ISSCC 2006,
 * and A 14mW 6.25Gb/s Transceiver in 90nm CMOS for Serial Chip-to-Chip Communication," ISSCC 2007
 *
 * In Cadence ChipEstimator Tool there are two types of memory controllers: the full memory controllers
 * that includes the frontend as the DesignWare DDR2/DDR3-Lite memory controllers and the backend only
 * memory controllers as the DDR2/DDR3-Lite protocol controllers (except DesignWare DDR2/DDR3-Lite memory
 * controllers, all memory controller IP in Cadence ChipEstimator Tool are backend memory controllers such as
 * DDRC 1600A and DDRC 800A). Thus,to some extend the area and power difference between DesignWare
 * DDR2/DDR3-Lite memory controllers and DDR2/DDR3-Lite protocol controllers can be an estimation to the
 * frontend power and area, which is very close the analitically modeled results of the frontend for Niagara2@65nm
 *
 */

MCBackend::MCBackend(InputParameter* interface_ip_, const MCParam & mcp_, enum MemoryCtrl_type mc_type_)
:l_ip(*interface_ip_),
 mc_type(mc_type_),
 mcp(mcp_)
{

  local_result = init_interface(&l_ip);
  compute();

}


void MCBackend::compute()
{
  //double max_row_addr_width = 20.0;//Current address 12~18bits
  double C_MCB, mc_power, backend_dyn, backend_gates;//, refresh_period,refresh_freq;//Equivalent per bit Cap for backend,
  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio();
  double NMOS_sizing, PMOS_sizing;

  if (mc_type == MC)
  {
	  if (mcp.type == 0)
	  {
		  //area = (2.2927*log(peakDataTransferRate)-14.504)*memDataWidth/144.0*(l_ip.F_sz_um/0.09);
		  area.set_area((2.7927*log(mcp.peakDataTransferRate*2)-19.862)/2.0*mcp.dataBusWidth/128.0*(l_ip.F_sz_um/0.09)*mcp.num_channels*1e6);//um^2
		  //assuming the approximately same scaling factor as seen in processors.
		  //C_MCB=0.2/1.3/1.3/266/64/0.09*g_ip.F_sz_um;//based on AMD Geode processor which has a very basic mc on chip.
		  //C_MCB = 1.6/200/1e6/144/1.2/1.2*g_ip.F_sz_um/0.19;//Based on Niagara power numbers.The base power (W) is divided by device frequency and vdd and scale to target process.
		  //mc_power = 0.0291*2;//29.1mW@200MHz @130nm From Power Analysis of SystemLevel OnChip Communication Architectures by Lahiri et
		  mc_power = 4.32*0.1;//4.32W@1GhzMHz @65nm Cadence ChipEstimator 10% for backend
		  C_MCB = mc_power/1e9/72/1.1/1.1*l_ip.F_sz_um/0.065;
		  power_t.readOp.dynamic = C_MCB*g_tp.peri_global.Vdd*g_tp.peri_global.Vdd*(mcp.dataBusWidth/*+mcp.addressBusWidth*/);//per access energy in memory controller
		  power_t.readOp.leakage = area.get_area()/2 *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(g_tp.min_w_nmos_, g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1, inv)*g_tp.peri_global.Vdd;//unit W
		  power_t.readOp.gate_leakage = area.get_area()/2 *(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(g_tp.min_w_nmos_, g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1, inv)*g_tp.peri_global.Vdd;//unit W

	  }
	  else
	  {   NMOS_sizing 	  = g_tp.min_w_nmos_;
		  PMOS_sizing	  = g_tp.min_w_nmos_*pmos_to_nmos_sizing_r;
		  area.set_area(0.15*mcp.dataBusWidth/72.0*(l_ip.F_sz_um/0.065)* (l_ip.F_sz_um/0.065)*mcp.num_channels*1e6);//um^2
		  backend_dyn = 0.9e-9/800e6*mcp.clockRate/12800*mcp.peakDataTransferRate*mcp.dataBusWidth/72.0*g_tp.peri_global.Vdd/1.1*g_tp.peri_global.Vdd/1.1*(l_ip.F_sz_nm/65.0);//Average on DDR2/3 protocol controller and DDRC 1600/800A in Cadence ChipEstimate
		  //Scaling to technology and DIMM feature. The base IP support DDR3-1600(PC3 12800)
		  backend_gates = 50000*mcp.dataBusWidth/64.0;//5000 is from Cadence ChipEstimator

		  power_t.readOp.dynamic = backend_dyn;
		  power_t.readOp.leakage = (backend_gates)*cmos_Isub_leakage(NMOS_sizing, PMOS_sizing, 2, nand)*g_tp.peri_global.Vdd;//unit W
		  power_t.readOp.gate_leakage = (backend_gates)*cmos_Ig_leakage(NMOS_sizing, PMOS_sizing, 2, nand)*g_tp.peri_global.Vdd;//unit W

	  }
  }
  else
  {//skip old model
	  cout<<"Unknown memory controllers"<<endl;exit(0);
	  area.set_area(0.243*mcp.dataBusWidth/8);//area based on Cadence ChipEstimator for 8bit bus
	  //mc_power = 4.32*0.1;//4.32W@1GhzMHz @65nm Cadence ChipEstimator 10% for backend
	  C_MCB = mc_power/1e9/72/1.1/1.1*l_ip.F_sz_um/0.065;
	  power_t.readOp.leakage = area.get_area()/2 *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(g_tp.min_w_nmos_, g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1, inv)*g_tp.peri_global.Vdd;//unit W
	  power_t.readOp.gate_leakage = area.get_area()/2 *(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(g_tp.min_w_nmos_, g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1, inv)*g_tp.peri_global.Vdd;//unit W
	  power_t.readOp.dynamic *= 1.2;
	  power_t.readOp.leakage *= 1.2;
	  power_t.readOp.gate_leakage *= 1.2;
	  //flash controller has about 20% more backend power since BCH ECC in flash is complex and power hungry
  }
  double long_channel_device_reduction = longer_channel_device_reduction(Uncore_device);
  power_t.readOp.longer_channel_leakage = power_t.readOp.leakage * long_channel_device_reduction;
}

void MCBackend::computeEnergy(bool is_tdp)
{
	//backend uses internal data buswidth
	if (is_tdp)
	{
    power.reset(); //Jingwen
		//init stats for Peak
		stats_t.readAc.access   = 0.5*mcp.num_channels;
		stats_t.writeAc.access  = 0.5*mcp.num_channels;
		tdp_stats = stats_t;
	}
	else
	{
    rt_power.reset();//Jingwen
		//init stats for runtime power (RTP)
    //Jingwen: should use stats from XML object, modified in MemoryController::computeEnergy
		stats_t.readAc.access   = mcp.reads;
		stats_t.writeAc.access  = mcp.writes;
		tdp_stats = stats_t;
	}
	if (is_tdp)
    {
		power = power_t;
		power.readOp.dynamic	= (stats_t.readAc.access + stats_t.writeAc.access)*power_t.readOp.dynamic;

    }
    else
    {
    	rt_power.readOp.dynamic	= (stats_t.readAc.access + stats_t.writeAc.access)*mcp.llcBlockSize*8.0/mcp.dataBusWidth*power_t.readOp.dynamic;
    	rt_power = rt_power + power_t*pppm_lkg;
    	rt_power.readOp.dynamic = rt_power.readOp.dynamic + power.readOp.dynamic*0.1*mcp.clockRate*mcp.num_mcs*mcp.executionTime;
    	//Assume 10% of peak power is consumed by routine job including memory refreshing and scrubbing
    }
}


MCPHY::MCPHY(InputParameter* interface_ip_, const MCParam & mcp_, enum MemoryCtrl_type mc_type_)
:l_ip(*interface_ip_),
 mc_type(mc_type_),
 mcp(mcp_)
{

  local_result = init_interface(&l_ip);
  compute();
}

void MCPHY::compute()
{
  //PHY uses internal data buswidth but the actuall off-chip datawidth is 64bits + ecc
  double pmos_to_nmos_sizing_r = pmos_to_nmos_sz_ratio() ;
  /*
   * according to "A 100mW 9.6Gb/s Transceiver in 90nm CMOS for next-generation memory interfaces ," ISSCC 2006;
   * From Cadence ChipEstimator for normal I/O around 0.4~0.8 mW/Gb/s
   */
  double power_per_gb_per_s,phy_gates, NMOS_sizing, PMOS_sizing;

  if (mc_type == MC)
  {
	  if (mcp.type == 0)
	  {
		  power_per_gb_per_s = mcp.LVDS? 0.01:0.04;
		  //Based on die photos from Niagara 1 and 2.
		  //TODO merge this into undifferentiated core.PHY only achieves square root of the ideal scaling.
		  //area = (6.4323*log(peakDataTransferRate)-34.76)*memDataWidth/128.0*(l_ip.F_sz_um/0.09);
		  area.set_area((6.4323*log(mcp.peakDataTransferRate*2)-48.134)*mcp.dataBusWidth/128.0*(l_ip.F_sz_um/0.09)*mcp.num_channels*1e6/2);//TODO:/2
		  //This is from curve fitting based on Niagara 1 and 2's PHY die photo.
		  //This is power not energy, 10mw/Gb/s @90nm for each channel and scaling down
		  //power.readOp.dynamic = 0.02*memAccesses*llcBlocksize*8;//change from Bytes to bits.
		  power_t.readOp.dynamic = power_per_gb_per_s*sqrt(l_ip.F_sz_um/0.09)*g_tp.peri_global.Vdd/1.2*g_tp.peri_global.Vdd/1.2;
		  power_t.readOp.leakage = area.get_area()/2 *(g_tp.scaling_factor.core_tx_density)*cmos_Isub_leakage(g_tp.min_w_nmos_, g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1, inv)*g_tp.peri_global.Vdd;//unit W
		  power_t.readOp.gate_leakage = area.get_area()/2 *(g_tp.scaling_factor.core_tx_density)*cmos_Ig_leakage(g_tp.min_w_nmos_, g_tp.min_w_nmos_*pmos_to_nmos_sizing_r, 1, inv)*g_tp.peri_global.Vdd;//unit W

	  }
	  else
	  {
		  NMOS_sizing 	  = g_tp.min_w_nmos_;
		  PMOS_sizing	  = g_tp.min_w_nmos_*pmos_to_nmos_sizing_r;
		  //Designware/synopsis 16bit DDR3 PHY is 1.3mm (WITH IOs) at 40nm for upto DDR3 2133 (PC3 17066)
		  double non_IO_percentage = 0.2;
		  area.set_area(1.3*non_IO_percentage/2133.0e6*mcp.clockRate/17066*mcp.peakDataTransferRate*mcp.dataBusWidth/16.0*(l_ip.F_sz_um/0.040)* (l_ip.F_sz_um/0.040)*mcp.num_channels*1e6);//um^2
		  phy_gates = 200000*mcp.dataBusWidth/64.0;
		  power_per_gb_per_s = 0.01;
		  //This is power not energy, 10mw/Gb/s @90nm for each channel and scaling down
		  power_t.readOp.dynamic = power_per_gb_per_s*(l_ip.F_sz_um/0.09)*g_tp.peri_global.Vdd/1.2*g_tp.peri_global.Vdd/1.2;
		  power_t.readOp.leakage = (mcp.withPHY? phy_gates:0)*cmos_Isub_leakage(NMOS_sizing, PMOS_sizing, 2, nand)*g_tp.peri_global.Vdd;//unit W
		  power_t.readOp.gate_leakage = (mcp.withPHY? phy_gates:0)*cmos_Ig_leakage(NMOS_sizing, PMOS_sizing, 2, nand)*g_tp.peri_global.Vdd;//unit W
	  }

  }
  else
  {
	  area.set_area(0.4e6/2*mcp.dataBusWidth/8);//area based on Cadence ChipEstimator for 8bit bus
  }

//  double phy_factor = (int)ceil(mcp.dataBusWidth/72.0);//Previous phy power numbers are based on 72 bit DIMM interface
//  power_t.readOp.dynamic *= phy_factor;
//  power_t.readOp.leakage *= phy_factor;
//  power_t.readOp.gate_leakage *= phy_factor;

  double long_channel_device_reduction = longer_channel_device_reduction(Uncore_device);
  power_t.readOp.longer_channel_leakage = power_t.readOp.leakage * long_channel_device_reduction;
}


void MCPHY::computeEnergy(bool is_tdp)
{
	if (is_tdp)
	{
    power.reset(); //Jingwen
		//init stats for Peak
		stats_t.readAc.access   = 0.5*mcp.num_channels; //time share on buses
		stats_t.writeAc.access  = 0.5*mcp.num_channels;
		tdp_stats = stats_t;
	}
	else
	{
    rt_power.reset(); //Jingwen
		//init stats for runtime power (RTP)
    //Jingwen: should use stats from XML object, modified in MemoryController::computeEnergy
		stats_t.readAc.access   = mcp.reads;
		stats_t.writeAc.access  = mcp.writes;
		tdp_stats = stats_t;
	}

	if (is_tdp)
    {
		double data_transfer_unit = (mc_type == MC)? 72:16;/*DIMM data width*/
		power = power_t;
		power.readOp.dynamic	= power.readOp.dynamic * (mcp.peakDataTransferRate*8*1e6/1e9/*change to Gbs*/)*mcp.dataBusWidth/data_transfer_unit*mcp.num_channels/mcp.clockRate;
		// divide by clock rate is for match the final computation where *clock is used
		//(stats_t.readAc.access*power_t.readOp.dynamic+
//					stats_t.writeAc.access*power_t.readOp.dynamic);

    }
    else
    {
    	rt_power = power_t;
//    	rt_power.readOp.dynamic	= (stats_t.readAc.access*power_t.readOp.dynamic+
//    						stats_t.writeAc.access*power_t.readOp.dynamic);

    	rt_power.readOp.dynamic=power_t.readOp.dynamic*(stats_t.readAc.access + stats_t.writeAc.access)*(mcp.llcBlockSize)*8/1e9/mcp.executionTime*(mcp.executionTime);
    	rt_power.readOp.dynamic = rt_power.readOp.dynamic + power.readOp.dynamic*0.1*mcp.clockRate*mcp.num_mcs*mcp.executionTime;
    }
}

MCFrontEnd::MCFrontEnd(ParseXML *XML_interface,InputParameter* interface_ip_, const MCParam & mcp_, enum MemoryCtrl_type mc_type_)
:XML(XML_interface),
 interface_ip(*interface_ip_),
 mc_type(mc_type_),
 mcp(mcp_),
 MC_arb(0),
 frontendBuffer(0),
 readBuffer(0),
 writeBuffer(0),
 coalesce_scale(1.0)
{
  /* All computations are for a single MC
   *
   */

  int tag, data;
  bool is_default =true;//indication for default setup

  /* MC frontend engine channels share the same engines but logically partitioned
   * For all hardware inside MC. different channels do not share resources.
   * TODO: add docodeing/mux stage to steer memory requests to different channels.
   */

  //memory request reorder buffer
  tag							   = mcp.addressBusWidth  + EXTRA_TAG_BITS + mcp.opcodeW;
  data    					 	   = int(ceil((XML->sys.physical_address_width + mcp.opcodeW)/8.0));
  interface_ip.cache_sz            = data*XML->sys.mc.req_window_size_per_channel;
  interface_ip.line_sz             = data;
  interface_ip.assoc               = 0;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.specific_tag        = 1;
  interface_ip.tag_w               = tag;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          = 1.0/mcp.clockRate;
  interface_ip.latency             = 1.0/mcp.clockRate;
  interface_ip.is_cache			   = true;
  interface_ip.pure_cam            = false;
  interface_ip.pure_ram            = false;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports        = 0;
  interface_ip.num_rd_ports        = XML->sys.mc.memory_channels_per_mc;
  interface_ip.num_wr_ports        = interface_ip.num_rd_ports;
  interface_ip.num_se_rd_ports     = 0;
  interface_ip.num_search_ports     = XML->sys.mc.memory_channels_per_mc;
  frontendBuffer = new ArrayST(&interface_ip, "MC ReorderBuffer", Uncore_device);
  frontendBuffer->area.set_area(frontendBuffer->area.get_area()+ frontendBuffer->local_result.area*XML->sys.mc.memory_channels_per_mc);
  area.set_area(area.get_area()+ frontendBuffer->local_result.area*XML->sys.mc.memory_channels_per_mc);

  //selection and arbitration logic
  MC_arb = new selection_logic(is_default, XML->sys.mc.req_window_size_per_channel,1,&interface_ip, Uncore_device);

  //read buffers.
  data    					 	   = (int)ceil(mcp.dataBusWidth/8.0);//Support key words first operation //8 means converting bit to Byte
  interface_ip.cache_sz            = data*XML->sys.mc.IO_buffer_size_per_channel;//*llcBlockSize;
  interface_ip.line_sz             = data;
  interface_ip.assoc               = 1;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 1;
  interface_ip.throughput          = 1.0/mcp.clockRate;
  interface_ip.latency             = 2.0/mcp.clockRate;
  interface_ip.is_cache			   = false;
  interface_ip.pure_cam            = false;
  interface_ip.pure_ram            = true;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports        = 0;//XML->sys.mc.memory_channels_per_mc*2>2?2:XML->sys.mc.memory_channels_per_mc*2;
  interface_ip.num_rd_ports        = XML->sys.mc.memory_channels_per_mc;
  interface_ip.num_wr_ports        = interface_ip.num_rd_ports;
  interface_ip.num_se_rd_ports     = 0;
  readBuffer = new ArrayST(&interface_ip, "MC ReadBuffer", Uncore_device);
  readBuffer->area.set_area(readBuffer->area.get_area()+ readBuffer->local_result.area*XML->sys.mc.memory_channels_per_mc);
  area.set_area(area.get_area()+ readBuffer->local_result.area*XML->sys.mc.memory_channels_per_mc);

  //write buffer
  data    					 	   = (int)ceil(mcp.dataBusWidth/8.0);//Support key words first operation //8 means converting bit to Byte
  interface_ip.cache_sz            = data*XML->sys.mc.IO_buffer_size_per_channel;//*llcBlockSize;
  interface_ip.line_sz             = data;
  interface_ip.assoc               = 1;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          = 1.0/mcp.clockRate;
  interface_ip.latency             = 2.0/mcp.clockRate;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports        = 0;
  interface_ip.num_rd_ports        = XML->sys.mc.memory_channels_per_mc;
  interface_ip.num_wr_ports        = interface_ip.num_rd_ports;
  interface_ip.num_se_rd_ports     = 0;
  writeBuffer = new ArrayST(&interface_ip, "MC writeBuffer", Uncore_device);
  writeBuffer->area.set_area(writeBuffer->area.get_area()+ writeBuffer->local_result.area*XML->sys.mc.memory_channels_per_mc);
  area.set_area(area.get_area()+ writeBuffer->local_result.area*XML->sys.mc.memory_channels_per_mc);

  //SRAM structures for memory coalescing --Syed Gilani
  //Pending Request Table (base addresses, offset addresses, threads IDs), Thread Masks
  //***PRT
  //We assume 24 bits of base address and 8 bits of offset address.
  //THese values are used for coalesing memory requests to the same base address block.
  //TIDs are assumed to be 8 bits
	  /*Contents of each PRT entry
		*
		*  Warp ID (6 bits) | Memory address (32 bits) per thread | Request Size (2-bits) per thread |   
		*  line size= 6+ 32*16 + 2*16  ~ 64 bytes
		*
		*
		*/  
  data    					 	   = 64;//Support key words first operation //8 means converting bit to Byte
  interface_ip.cache_sz            = data*XML->sys.mc.PRT_entries;//PRT table;
  interface_ip.line_sz             = data;
  interface_ip.assoc               = 1;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          = 1.0/XML->sys.target_core_clockrate;
  interface_ip.latency             = 2.0/XML->sys.target_core_clockrate;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports        = 0;
  interface_ip.num_rd_ports        = 1;
  interface_ip.num_wr_ports        = 1;
  interface_ip.num_se_rd_ports     = 0;
  PRT = new ArrayST(&interface_ip, "MC PRT", Uncore_device);
  PRT->area.set_area(PRT->area.get_area()+ PRT->local_result.area*XML->sys.mc.memory_channels_per_mc);
  area.set_area(area.get_area()+ PRT->local_result.area*XML->sys.mc.memory_channels_per_mc);


  //***ThreadMasks storage (coalesced threads whose memory requests are satisfied by each memory access)
	  /* contents of the thread masks Array
		*  16-bit bit masks for up to 16 memory requests of a warp | Number of pending memory requests (5 bits)
		*  
		*  16*PRT_entry thread Mask, each entry has 16 mask bits.
		*
		*/  
  data    					 	   = 2;
  interface_ip.cache_sz            = data*XML->sys.mc.PRT_entries*16;//PRT table;
  interface_ip.line_sz             = data;
  interface_ip.assoc               = 1;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          = 1.0/XML->sys.target_core_clockrate;
  interface_ip.latency             = 2.0/XML->sys.target_core_clockrate;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports        = 0;
  interface_ip.num_rd_ports        = 1;
  interface_ip.num_wr_ports        = 1;
  interface_ip.num_se_rd_ports     = 0;
  threadMasks = new ArrayST(&interface_ip, "MC ThreadMasks", Uncore_device);
  threadMasks->area.set_area(threadMasks->area.get_area()+ threadMasks->local_result.area*XML->sys.mc.memory_channels_per_mc);
  area.set_area(area.get_area()+ threadMasks->local_result.area*XML->sys.mc.memory_channels_per_mc);

  //***Numer of pending requests per PRT entry
	  /* 
		* 1-byte data, PRT entries deep
		*/  
  data    					 	   = 1;
  interface_ip.cache_sz            = data*XML->sys.mc.PRT_entries;//PRT table;
  interface_ip.line_sz             = data;
  interface_ip.assoc               = 1;
  interface_ip.nbanks              = 1;
  interface_ip.out_w               = interface_ip.line_sz*8;
  interface_ip.access_mode         = 0;
  interface_ip.throughput          = 1.0/XML->sys.target_core_clockrate;
  interface_ip.latency             = 2.0/XML->sys.target_core_clockrate;
  interface_ip.obj_func_dyn_energy = 0;
  interface_ip.obj_func_dyn_power  = 0;
  interface_ip.obj_func_leak_power = 0;
  interface_ip.obj_func_cycle_t    = 1;
  interface_ip.num_rw_ports        = 0;
  interface_ip.num_rd_ports        = 1;
  interface_ip.num_wr_ports        = 1;
  interface_ip.num_se_rd_ports     = 0;
  PRC = new ArrayST(&interface_ip, "MC PendingRequestCount", Uncore_device);
  PRC->area.set_area(PRC->area.get_area()+ PRC->local_result.area*XML->sys.mc.memory_channels_per_mc);
  area.set_area(area.get_area()+ PRC->local_result.area*XML->sys.mc.memory_channels_per_mc);

}


void DRAM::computeEnergy(bool is_tdp)
{
	if (is_tdp){
		power.reset();
		return; /// not supporting TDP calculation for DRAM
	}
	rt_power.reset();
	dramp.executionTime   = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);

	power_t.reset();
	power_t.readOp.dynamic+=XML->sys.mc.memory_reads*dramp.rd_coeff;
	power_t.readOp.dynamic+=XML->sys.mc.memory_writes*dramp.wr_coeff;
	power_t.readOp.dynamic+=XML->sys.mc.dram_pre*dramp.pre_coeff;

	rt_power = rt_power + power_t ;
}


void MCFrontEnd::computeEnergy(bool is_tdp)
{
	if (is_tdp)
	    {
        power.reset();
	    	//init stats for Peak
	    	frontendBuffer->stats_t.readAc.access  = frontendBuffer->l_ip.num_search_ports;
	    	frontendBuffer->stats_t.writeAc.access = frontendBuffer->l_ip.num_wr_ports;
	    	frontendBuffer->tdp_stats = frontendBuffer->stats_t;

	    	readBuffer->stats_t.readAc.access  = readBuffer->l_ip.num_rd_ports*mcp.frontend_duty_cycle;
	    	readBuffer->stats_t.writeAc.access = readBuffer->l_ip.num_wr_ports*mcp.frontend_duty_cycle;
	    	readBuffer->tdp_stats = readBuffer->stats_t;

	    	writeBuffer->stats_t.readAc.access  = writeBuffer->l_ip.num_rd_ports*mcp.frontend_duty_cycle;
	    	writeBuffer->stats_t.writeAc.access = writeBuffer->l_ip.num_wr_ports*mcp.frontend_duty_cycle;
	    	writeBuffer->tdp_stats = writeBuffer->stats_t;

			PRT->stats_t.readAc.access = PRT->l_ip.num_rd_ports*mcp.frontend_duty_cycle;
	    	PRT->stats_t.writeAc.access = PRT->l_ip.num_wr_ports*mcp.frontend_duty_cycle;
	    	PRT->tdp_stats = PRT->stats_t;
			
			threadMasks->stats_t.readAc.access = threadMasks->l_ip.num_rd_ports*mcp.frontend_duty_cycle;
	    	threadMasks->stats_t.writeAc.access = threadMasks->l_ip.num_wr_ports*mcp.frontend_duty_cycle;
	    	threadMasks->tdp_stats = threadMasks->stats_t;
			
			PRC->stats_t.readAc.access = threadMasks->l_ip.num_rd_ports*mcp.frontend_duty_cycle;
	    	PRC->stats_t.writeAc.access = threadMasks->l_ip.num_wr_ports*mcp.frontend_duty_cycle;
	    	PRC->tdp_stats = threadMasks->stats_t;

	    }
	    else
	    {
        rt_power.reset(); //Jingwen
	    	//init stats for runtime power (RTP)
	    	frontendBuffer->stats_t.readAc.access  = XML->sys.mc.memory_reads *mcp.llcBlockSize*8.0/mcp.dataBusWidth*mcp.dataBusWidth/72;
	    	//For each channel, each memory word need to check the address data to achieve best scheduling results.
	    	//and this need to be done on all physical DIMMs in each logical memory DIMM *mcp.dataBusWidth/72
	    	frontendBuffer->stats_t.writeAc.access = XML->sys.mc.memory_writes*mcp.llcBlockSize*8.0/mcp.dataBusWidth*mcp.dataBusWidth/72;
	    	frontendBuffer->rtp_stats = frontendBuffer->stats_t;

	    	readBuffer->stats_t.readAc.access  = XML->sys.mc.memory_reads*mcp.llcBlockSize*8.0/mcp.dataBusWidth;//support key word first
	    	readBuffer->stats_t.writeAc.access = XML->sys.mc.memory_reads*mcp.llcBlockSize*8.0/mcp.dataBusWidth;//support key word first
	    	readBuffer->rtp_stats = readBuffer->stats_t;

	    	writeBuffer->stats_t.readAc.access  = XML->sys.mc.memory_writes*mcp.llcBlockSize*8.0/mcp.dataBusWidth;
	    	writeBuffer->stats_t.writeAc.access = XML->sys.mc.memory_writes*mcp.llcBlockSize*8.0/mcp.dataBusWidth;
	    	writeBuffer->rtp_stats = writeBuffer->stats_t;
	    	
			//Pending request table
			//Co-alesce all misses in caches and add an entry for them in PRT
			//TODO: Change 0 to ithCore and move to LSU (Syed)
			//TODO: Do these accesses represent coalesced accesses?
			PRT->stats_t.readAc.access  = XML->sys.core[0].dcache.read_accesses + XML->sys.core[0].ccache.read_accesses +
			                               XML->sys.core[0].tcache.read_accesses;
	    	PRT->stats_t.writeAc.access = XML->sys.core[0].dcache.write_accesses + XML->sys.core[0].ccache.write_accesses +
			                               XML->sys.core[0].tcache.write_accesses;
	    	PRT->rtp_stats = PRT->stats_t;
			
			threadMasks->stats_t.readAc.access  = XML->sys.core[0].dcache.read_accesses + XML->sys.core[0].ccache.read_accesses +
			                               XML->sys.core[0].tcache.read_accesses;
	    	threadMasks->stats_t.writeAc.access = XML->sys.core[0].dcache.write_accesses + XML->sys.core[0].ccache.write_accesses +
			                               XML->sys.core[0].tcache.write_accesses;
	    	threadMasks->rtp_stats = threadMasks->stats_t;

			PRC->stats_t.readAc.access  = XML->sys.core[0].dcache.read_accesses + XML->sys.core[0].ccache.read_accesses +
			                               XML->sys.core[0].tcache.read_accesses;
	    	PRC->stats_t.writeAc.access = XML->sys.core[0].dcache.write_accesses + XML->sys.core[0].ccache.write_accesses +
			                               XML->sys.core[0].tcache.write_accesses;
	    	PRC->rtp_stats = threadMasks->stats_t;

	    }

	frontendBuffer->power_t.reset();
	readBuffer->power_t.reset();
	writeBuffer->power_t.reset();
    threadMasks->power_t.reset();
    PRT->power_t.reset();
	PRC->power_t.reset();

//	frontendBuffer->power_t.readOp.dynamic	+= (frontendBuffer->stats_t.readAc.access*
//			(frontendBuffer->local_result.power.searchOp.dynamic+frontendBuffer->local_result.power.readOp.dynamic)+
//    		frontendBuffer->stats_t.writeAc.access*frontendBuffer->local_result.power.writeOp.dynamic);

	frontendBuffer->power_t.readOp.dynamic	+= (frontendBuffer->stats_t.readAc.access +
			frontendBuffer->stats_t.writeAc.access)*frontendBuffer->local_result.power.searchOp.dynamic+
		    frontendBuffer->stats_t.readAc.access * frontendBuffer->local_result.power.readOp.dynamic+
		    frontendBuffer->stats_t.writeAc.access*frontendBuffer->local_result.power.writeOp.dynamic;

	readBuffer->power_t.readOp.dynamic	+= (readBuffer->stats_t.readAc.access*
			readBuffer->local_result.power.readOp.dynamic+
    		readBuffer->stats_t.writeAc.access*readBuffer->local_result.power.writeOp.dynamic);
	writeBuffer->power_t.readOp.dynamic	+= (writeBuffer->stats_t.readAc.access*
			writeBuffer->local_result.power.readOp.dynamic+
    		writeBuffer->stats_t.writeAc.access*writeBuffer->local_result.power.writeOp.dynamic);

	PRT->power_t.readOp.dynamic	+= (PRT->stats_t.readAc.access*
			PRT->local_result.power.readOp.dynamic+
    		PRT->stats_t.writeAc.access*PRT->local_result.power.writeOp.dynamic);

	threadMasks->power_t.readOp.dynamic	+= (threadMasks->stats_t.readAc.access*
			threadMasks->local_result.power.readOp.dynamic+
    		threadMasks->stats_t.writeAc.access*threadMasks->local_result.power.writeOp.dynamic);

	PRC->power_t.readOp.dynamic	+= (PRC->stats_t.readAc.access*
			PRC->local_result.power.readOp.dynamic+
    		PRC->stats_t.writeAc.access*PRC->local_result.power.writeOp.dynamic);

		//Add coalescing logic power (Estimated from Verilog HDL description and Synopsys PowerCompiler)--Syed
 #define COALESCE_SCALE 1
	double perAccessCoalescingEnergy=coalesce_scale * ((0.443e-3)*(0.5e-9)*g_tp.peri_global.Vdd*g_tp.peri_global.Vdd)/(1*1);
	threadMasks->power_t.readOp.dynamic += (threadMasks->stats_t.readAc.access+threadMasks->stats_t.writeAc.access)*perAccessCoalescingEnergy;

	//printf("***PRT: %10.30f, threadMasks: %10.30f, PRC: %10.30f\n",PRT->power_t.readOp.dynamic,threadMasks->power_t.readOp.dynamic,PRC->power_t.readOp.dynamic);
	//printf("***Accesses: read:%lf write:%lf\n",threadMasks->stats_t.readAc.access, threadMasks->stats_t.writeAc.access);
	if (is_tdp)
    {
    	power = power + frontendBuffer->power_t + readBuffer->power_t + writeBuffer->power_t + PRT->power_t  + threadMasks->power_t +
               PRC->power_t +
		        (frontendBuffer->local_result.power +
    	        		readBuffer->local_result.power +
    	        		writeBuffer->local_result.power+PRT->local_result.power+
						threadMasks->local_result.power+PRC->local_result.power)*pppm_lkg;
		

    }
    else
    {
    	rt_power = rt_power + frontendBuffer->power_t + readBuffer->power_t + writeBuffer->power_t + PRT->power_t + threadMasks->power_t +
		         PRC->power_t+
    	         (frontendBuffer->local_result.power +
    	        		readBuffer->local_result.power +
    	        		writeBuffer->local_result.power + PRT->local_result.power + threadMasks->local_result.power+
						PRC->local_result.power)*pppm_lkg;
    	rt_power.readOp.dynamic = rt_power.readOp.dynamic + power.readOp.dynamic*0.1*mcp.clockRate*mcp.num_mcs*mcp.executionTime;
    }
}

void MCFrontEnd::displayEnergy(uint32_t indent,int plevel,bool is_tdp)
{
	string indent_str(indent, ' ');
	string indent_str_next(indent+2, ' ');

	if (is_tdp)
	{
		cout << indent_str << "Front End ROB:" << endl;
		cout << indent_str_next << "Area = " << frontendBuffer->area.get_area()*1e-6<< " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << frontendBuffer->power.readOp.dynamic*mcp.clockRate << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << frontendBuffer->power.readOp.leakage <<" W" << endl;
		cout << indent_str_next << "Gate Leakage = " << frontendBuffer->power.readOp.gate_leakage << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << frontendBuffer->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;

		cout <<endl;
		cout << indent_str<< "Read Buffer:" << endl;
		cout << indent_str_next << "Area = " << readBuffer->area.get_area()*1e-6  << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << readBuffer->power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << readBuffer->power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << readBuffer->power.readOp.gate_leakage  << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << readBuffer->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout <<endl;
		cout << indent_str << "Write Buffer:" << endl;
		cout << indent_str_next << "Area = " << writeBuffer->area.get_area() *1e-6 << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << writeBuffer->power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << writeBuffer->power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << writeBuffer->power.readOp.gate_leakage  << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << writeBuffer->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout <<endl;
		cout << indent_str << "PRT:" << endl;
		cout << indent_str_next << "Area = " << PRT->area.get_area() *1e-6 << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << PRT->power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << PRT->power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << PRT->power.readOp.gate_leakage  << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << PRT->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout <<endl;
		cout << indent_str << "Thread Masks and coalescing logic:" << endl;
		cout << indent_str_next << "Area = " << threadMasks->area.get_area() *1e-6 << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << threadMasks->power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << threadMasks->power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << threadMasks->power.readOp.gate_leakage  << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << threadMasks->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout <<endl;


	}
	else
	{
		cout << indent_str << "Front End ROB:" << endl;
		cout << indent_str_next << "Area = " << frontendBuffer->area.get_area()*1e-6<< " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << frontendBuffer->rt_power.readOp.dynamic*mcp.clockRate << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << frontendBuffer->rt_power.readOp.leakage <<" W" << endl;
		cout << indent_str_next << "Gate Leakage = " << frontendBuffer->rt_power.readOp.gate_leakage << " W" << endl;
		cout <<endl;
		cout << indent_str<< "Read Buffer:" << endl;
		cout << indent_str_next << "Area = " << readBuffer->area.get_area()*1e-6  << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << readBuffer->rt_power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << readBuffer->rt_power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << readBuffer->rt_power.readOp.gate_leakage  << " W" << endl;
		cout <<endl;
		cout << indent_str << "Write Buffer:" << endl;
		cout << indent_str_next << "Area = " << writeBuffer->area.get_area() *1e-6 << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << writeBuffer->rt_power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << writeBuffer->rt_power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << writeBuffer->rt_power.readOp.gate_leakage  << " W" << endl;
		cout <<endl;
		cout << indent_str << "PRT:" << endl;
		cout << indent_str_next << "Area = " << PRT->area.get_area() *1e-6 << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << PRT->rt_power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << PRT->rt_power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << PRT->rt_power.readOp.gate_leakage  << " W" << endl;
		cout <<endl;
		cout << indent_str << "Thread masks and coalescing logic:" << endl;
		cout << indent_str_next << "Area = " << threadMasks->area.get_area() *1e-6 << " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << threadMasks->rt_power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << threadMasks->rt_power.readOp.leakage  << " W" << endl;
		cout << indent_str_next << "Gate Leakage = " << threadMasks->rt_power.readOp.gate_leakage  << " W" << endl;


	}

}

DRAM::DRAM(ParseXML *XML_interface,InputParameter* interface_ip_, enum Dram_type dram_type_)
:XML(XML_interface),
 interface_ip(*interface_ip_),
 dram_type(dram_type_)
{

		set_dram_param();

}
MemoryController::MemoryController(ParseXML *XML_interface,InputParameter* interface_ip_, enum MemoryCtrl_type mc_type_,enum Dram_type dram_type_)
:XML(XML_interface),
 interface_ip(*interface_ip_),
 mc_type(mc_type_),
 frontend(0),
 transecEngine(0),
 PHY(0),
 pipeLogic(0)
{
  /* All computations are for a single MC
   *
   */
  interface_ip.wire_is_mat_type = 2;
  interface_ip.wire_os_mat_type = 2;
  interface_ip.wt               =Global;
  set_mc_param();
  frontend = new MCFrontEnd(XML, &interface_ip, mcp, mc_type);
  dram = new DRAM(XML, &interface_ip, dram_type_);
  area.set_area(area.get_area()+ frontend->area.get_area());
  transecEngine = new MCBackend(&interface_ip, mcp, mc_type);
  area.set_area(area.get_area()+ transecEngine->area.get_area());
  if (mcp.type==0 || (mcp.type==1&&mcp.withPHY))
  {
	  PHY = new MCPHY(&interface_ip, mcp, mc_type);
	  area.set_area(area.get_area()+ PHY->area.get_area());
  }
  //+++++++++Transaction engine +++++++++++++++++ ////TODO needs better numbers, Run the RTL code from OpenSparc.
//  transecEngine.initialize(&interface_ip);
//  transecEngine.peakDataTransferRate = XML->sys.mem.peak_transfer_rate;
//  transecEngine.memDataWidth = dataBusWidth;
//  transecEngine.memRank = XML->sys.mem.number_ranks;
//  //transecEngine.memAccesses=XML->sys.mc.memory_accesses;
//  //transecEngine.llcBlocksize=llcBlockSize;
//  transecEngine.compute();
//  transecEngine.area.set_area(XML->sys.mc.memory_channels_per_mc*transecEngine.area.get_area()) ;
//  area.set_area(area.get_area()+ transecEngine.area.get_area());
//  ///cout<<"area="<<area<<endl;
////
//  //++++++++++++++PHY ++++++++++++++++++++++++++ //TODO needs better numbers
//  PHY.initialize(&interface_ip);
//  PHY.peakDataTransferRate = XML->sys.mem.peak_transfer_rate;
//  PHY.memDataWidth = dataBusWidth;
//  //PHY.memAccesses=PHY.peakDataTransferRate;//this is the max power
//  //PHY.llcBlocksize=llcBlockSize;
//  PHY.compute();
//  PHY.area.set_area(XML->sys.mc.memory_channels_per_mc*PHY.area.get_area()) ;
//  area.set_area(area.get_area()+ PHY.area.get_area());
  ///cout<<"area="<<area<<endl;
//
//  interface_ip.pipeline_stages = 5;//normal memory controller has five stages in the pipeline.
//  interface_ip.per_stage_vector = addressBusWidth + XML->sys.core[0].opcode_width + dataBusWidth;
//  pipeLogic = new pipeline(is_default, &interface_ip);
//  //pipeLogic.init_pipeline(is_default, &interface_ip);
//  pipeLogic->compute_pipeline();
//  area.set_area(area.get_area()+ pipeLogic->area.get_area()*1e-6);
//  area.set_area((area.get_area()+mc_area*1e-6)*1.1);//placement and routing overhead
//
//
////  //clock
////  clockNetwork.init_wire_external(is_default, &interface_ip);
////  clockNetwork.clk_area           =area*1.1;//10% of placement overhead. rule of thumb
////  clockNetwork.end_wiring_level   =5;//toplevel metal
////  clockNetwork.start_wiring_level =5;//toplevel metal
////  clockNetwork.num_regs           = pipeLogic.tot_stage_vector;
////  clockNetwork.optimize_wire();


}
void MemoryController::computeEnergy(bool is_tdp)
{

  rt_power.reset(); //Jingwen
  frontend->rt_power.reset();
  transecEngine->rt_power.reset();
  dram->rt_power.reset();
  mcp.executionTime   = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6); //Jingwen
  frontend->mcp.executionTime   = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6); //Jingwen
  transecEngine->mcp.executionTime  = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6); //Jingwen
  
  /*Jingwen: give stats for backend and phy */
  transecEngine->mcp.reads =XML->sys.mc.memory_reads;
  transecEngine->mcp.writes =XML->sys.mc.memory_writes;

  //set_mc_param();

	frontend->computeEnergy(is_tdp);
	transecEngine->computeEnergy(is_tdp);
	dram->computeEnergy(is_tdp);
	if (mcp.type==0 || (mcp.type==1&&mcp.withPHY))
	{
    if(!is_tdp)
      PHY->rt_power.reset();//Jingwen
    PHY->mcp.reads =XML->sys.mc.memory_reads;
    PHY->mcp.writes =XML->sys.mc.memory_writes;
    PHY->mcp.executionTime   = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6); //Jingwen
		PHY->computeEnergy(is_tdp);
	}
	if (is_tdp)
	{
		power = power + frontend->power + transecEngine->power;
		if (mcp.type==0 || (mcp.type==1&&mcp.withPHY))
		{
			power = power + PHY->power;
		}
	}
	else
	{
		rt_power = rt_power + frontend->rt_power + transecEngine->rt_power+dram->rt_power;
		if (mcp.type==0 || (mcp.type==1&&mcp.withPHY))
		{
			rt_power = rt_power + PHY->rt_power;
		}
	}
}

void MemoryController::displayEnergy(uint32_t indent,int plevel,bool is_tdp)
{
	string indent_str(indent, ' ');
	string indent_str_next(indent+2, ' ');
	bool long_channel = XML->sys.longer_channel_device;

	if (is_tdp)
	{
		cout << "Memory Controller:" << endl;
		cout << indent_str<< "Area = " << area.get_area()*1e-6<< " mm^2" << endl;
		cout << indent_str << "Peak Dynamic = " << power.readOp.dynamic*mcp.clockRate  << " W" << endl;
		cout << indent_str<< "Subthreshold Leakage = "
			<< (long_channel? power.readOp.longer_channel_leakage:power.readOp.leakage) <<" W" << endl;
		//cout << indent_str<< "Subthreshold Leakage = " << power.readOp.longer_channel_leakage <<" W" << endl;
		cout << indent_str<< "Gate Leakage = " << power.readOp.gate_leakage << " W" << endl;
		cout << indent_str << "Runtime Dynamic = " << rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout<<endl;
		cout << indent_str << "Front End Engine:" << endl;
		cout << indent_str_next << "Area = " << frontend->area.get_area()*1e-6<< " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << frontend->power.readOp.dynamic*mcp.clockRate << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = "
			<< (long_channel? frontend->power.readOp.longer_channel_leakage:frontend->power.readOp.leakage) <<" W" << endl;
		cout << indent_str_next << "Gate Leakage = " << frontend->power.readOp.gate_leakage << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << frontend->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout <<endl;
		//if (plevel >2){
			frontend->displayEnergy(indent+4,is_tdp);
		//}
		cout << indent_str << "Transaction Engine:" << endl;
		cout << indent_str_next << "Area = " << transecEngine->area.get_area()*1e-6<< " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << transecEngine->power.readOp.dynamic*mcp.clockRate << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = "
			<< (long_channel? transecEngine->power.readOp.longer_channel_leakage:transecEngine->power.readOp.leakage) <<" W" << endl;
		cout << indent_str_next << "Gate Leakage = " << transecEngine->power.readOp.gate_leakage << " W" << endl;
		cout << indent_str_next << "Runtime Dynamic = " << transecEngine->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
		cout <<endl;
		if (mcp.type==0 || (mcp.type==1&&mcp.withPHY))
		{
			cout << indent_str << "PHY:" << endl;
			cout << indent_str_next << "Area = " << PHY->area.get_area()*1e-6<< " mm^2" << endl;
			cout << indent_str_next << "Peak Dynamic = " << PHY->power.readOp.dynamic*mcp.clockRate << " W" << endl;
			cout << indent_str_next << "Subthreshold Leakage = "
			<< (long_channel? PHY->power.readOp.longer_channel_leakage:PHY->power.readOp.leakage) <<" W" << endl;
			cout << indent_str_next << "Gate Leakage = " << PHY->power.readOp.gate_leakage << " W" << endl;
			cout << indent_str_next << "Runtime Dynamic = " << PHY->rt_power.readOp.dynamic/mcp.executionTime << " W" << endl;
			cout <<endl;
		}
	}
	else
	{
		cout << "Memory Controller:" << endl;
		cout << indent_str_next << "Area = " << area.get_area()*1e-6<< " mm^2" << endl;
		cout << indent_str_next << "Peak Dynamic = " << power.readOp.dynamic*mcp.clockRate << " W" << endl;
		cout << indent_str_next << "Subthreshold Leakage = " << power.readOp.leakage <<" W" << endl;
		cout << indent_str_next << "Gate Leakage = " << power.readOp.gate_leakage << " W" << endl;
		cout<<endl;
	}

}

void DRAM::set_dram_param()
{
	dramp.cmd_coeff 		= XML->sys.mc.dram_rd_coeff;
	dramp.act_coeff 		= XML->sys.mc.dram_act_coeff;
	dramp.nop_coeff			= XML->sys.mc.dram_nop_coeff;
	dramp.activity_coeff 	= XML->sys.mc.dram_activity_coeff;
	dramp.pre_coeff			= XML->sys.mc.dram_pre_coeff;
	dramp.rd_coeff			= XML->sys.mc.dram_rd_coeff;
	dramp.wr_coeff			= XML->sys.mc.dram_wr_coeff;
	dramp.req_coeff			= XML->sys.mc.dram_req_coeff;
	dramp.const_coeff		= XML->sys.mc.dram_const_coeff;
}

void MemoryController::set_mc_param()
{

	if (mc_type==MC)
	{
	  mcp.clockRate       =XML->sys.mc.mc_clock*2;//DDR double pumped
	  mcp.clockRate       *= 1e6;
	  mcp.executionTime   = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);

	  mcp.llcBlockSize    =int(ceil(XML->sys.mc.llc_line_length/8.0))+XML->sys.mc.llc_line_length;//ecc overhead
	  mcp.dataBusWidth    =int(ceil(XML->sys.mc.databus_width/8.0)) + XML->sys.mc.databus_width;
	  mcp.addressBusWidth =int(ceil(XML->sys.mc.addressbus_width));//XML->sys.physical_address_width;
	  mcp.opcodeW         =16;
	  mcp.num_mcs         = XML->sys.mc.number_mcs;
	  mcp.num_channels    = XML->sys.mc.memory_channels_per_mc;
	  mcp.reads  = XML->sys.mc.memory_reads;
	  mcp.writes = XML->sys.mc.memory_writes;
	  //+++++++++Transaction engine +++++++++++++++++ ////TODO needs better numbers, Run the RTL code from OpenSparc.
	  mcp.peakDataTransferRate = XML->sys.mc.peak_transfer_rate;
	  mcp.memRank = XML->sys.mc.number_ranks;
	  //++++++++++++++PHY ++++++++++++++++++++++++++ //TODO needs better numbers
	  //PHY.memAccesses=PHY.peakDataTransferRate;//this is the max power
	  //PHY.llcBlocksize=llcBlockSize;
	  mcp.frontend_duty_cycle = 0.5;//for max power, the actual off-chip links is bidirectional but time shared
	  mcp.LVDS = XML->sys.mc.LVDS;
	  mcp.type = XML->sys.mc.type;
	  mcp.withPHY = XML->sys.mc.withPHY;
	}
//	else if (mc_type==FLASHC)
//	{
//		mcp.clockRate       =XML->sys.flashc.mc_clock*2;//DDR double pumped
//		mcp.clockRate       *= 1e6;
//		mcp.executionTime   = XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);
//
//		mcp.llcBlockSize    =int(ceil(XML->sys.flashc.llc_line_length/8.0))+XML->sys.flashc.llc_line_length;//ecc overhead
//		mcp.dataBusWidth    =int(ceil(XML->sys.flashc.databus_width/8.0)) + XML->sys.flashc.databus_width;
//		mcp.addressBusWidth =int(ceil(XML->sys.flashc.addressbus_width));//XML->sys.physical_address_width;
//		mcp.opcodeW         =16;
//		mcp.num_mcs         = XML->sys.flashc.number_mcs;
//		mcp.num_channels    = XML->sys.flashc.memory_channels_per_mc;
//		mcp.reads  = XML->sys.flashc.memory_reads;
//		mcp.writes = XML->sys.flashc.memory_writes;
//		//+++++++++Transaction engine +++++++++++++++++ ////TODO needs better numbers, Run the RTL code from OpenSparc.
//		mcp.peakDataTransferRate = XML->sys.flashc.peak_transfer_rate;
//		mcp.memRank = XML->sys.flashc.number_ranks;
//		//++++++++++++++PHY ++++++++++++++++++++++++++ //TODO needs better numbers
//		//PHY.memAccesses=PHY.peakDataTransferRate;//this is the max power
//		//PHY.llcBlocksize=llcBlockSize;
//		mcp.frontend_duty_cycle = 0.5;//for max power, the actual off-chip links is bidirectional but time shared
//		mcp.LVDS = XML->sys.flashc.LVDS;
//		mcp.type = XML->sys.flashc.type;
//	}
	else
	{
		cout<<"Unknown memory controller type: neither DRAM controller nor Flash controller" <<endl;
		exit(0);
	}
}

MCFrontEnd ::~MCFrontEnd(){

	if(MC_arb) 	               {delete MC_arb; MC_arb = 0;}
	if(frontendBuffer) 	       {delete frontendBuffer; frontendBuffer = 0;}
	if(readBuffer) 	           {delete readBuffer; readBuffer = 0;}
	if(writeBuffer) 	       {delete writeBuffer; writeBuffer = 0;}
}

MemoryController ::~MemoryController(){

	if(frontend) 	               {delete frontend; frontend = 0;}
	if(transecEngine) 	           {delete transecEngine; transecEngine = 0;}
	if(PHY) 	                   {delete PHY; PHY = 0;}
	if(pipeLogic) 	               {delete pipeLogic; pipeLogic = 0;}
}

