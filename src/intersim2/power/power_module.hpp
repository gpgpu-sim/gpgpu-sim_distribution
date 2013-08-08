// $Id: power_module.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _POWER_MODULE_HPP_
#define _POWER_MODULE_HPP_

#include <map>

#include "module.hpp"
#include "network.hpp"
#include "config_utils.hpp"
#include "flitchannel.hpp"
#include "switch_monitor.hpp"
#include "buffer_monitor.hpp"

struct wire{
  double L;
  double K;
  double M;
  double N;
};

class Power_Module : public Module {

protected:
  //network undersimulation
  Network * net;
  int classes;
  //all channels are this width
  double channel_width;
  //resimulate all with channel_width decremented by channel_sweep until 0
  double  channel_sweep; 
  //write result to a tabbed format to file
  string output_file_name;

  //buffer depth
  double depthVC;
  //vcs
  double numVC;

  //store the property of wires based on length
  map<double, wire> wire_map;

  //////////////////////////////////Constants/////////////////////////////
  //wire length in (mm)
  double wire_length;
  //////////Metal Parameters////////////
  // Wire left/right coupling capacitance [ F/mm ]
  double Cw_cpl ; 
  // Wire up/down groudn capacitance      [ F/mm ]
  double Cw_gnd  ;
  double Cw ;
  double Rw ;
  // metal pitch [mm]
  double MetalPitch ; 


  //////////Device Parameters////////////
  
  double LAMBDA  ;       // [um/LAMBDA]
  double Cd   ;           // [F/um] (for Delay)
  double Cg  ;           // [F/um] (for Delay)
  double Cgdl  ;           // [F/um] (for Delay)
  
  double Cd_pwr;           // [F/um] (for Power)
  double Cg_pwr  ;           // [F/um] (for Power)
			       
  double IoffN  ;            // [A/um]
  double IoffP  ;            // [A/um]
  // Leakage from bitlines, two-port cell  [A]
  double IoffSRAM;  
  // [Ohm] ( D1=1um Inverter)
  double R     ;                         
  // [F]   ( D1=1um Inverter - for Power )
  double Ci_delay;   
  // [F]   ( D1=1um Inverter - for Power )
  double Co_delay ;              

  double Ci ;
  double Co ;
  double Vdd  ;
  double FO4   ;		     
  double tCLK ;
  double fCLK ;              

  double H_INVD2;
  double W_INVD2;
  double H_DFQD1;
  double W_DFQD1;
  double H_ND2D1;
  double W_ND2D1;
  double H_SRAM;
  double W_SRAM;
  double  ChannelPitch ;
  double   CrossbarPitch;
  ////////////////////////////////End of Constants/////////////////////////////

  /////////////results///////////////////
  double totalTime;
  double channelWirePower;
  double channelClkPower;
  double channelDFFPower;
  double channelLeakPower;
  double inputReadPower;
  double inputWritePower;
  double inputLeakagePower;
  double switchPower;
  double switchPowerCtrl;
  double switchPowerLeak;
  double outputPower;
  double outputPowerClk;
  double outputCtrlPower;
  double channelArea;
  double switchArea;
  double inputArea;
  double outputArea;
  double maxInputPort;
  double maxOutputPort;


  ////////////////////////

  //channels
  void calcChannel(const FlitChannel * f);
  wire const & wireOptimize(double l);
  double powerRepeatedWire(double L, double K, double M, double N);
  double powerRepeatedWireLeak (double K, double M, double N);
  double powerWireClk (double M, double W);
  double powerWireDFF(double M, double W, double alpha);
  
  //memory
  void calcBuffer(const BufferMonitor *bm);
  double powerWordLine(double memoryWidth, double memoryDepth);
  double powerMemoryBitRead(double memoryDepth);
  double powerMemoryBitWrite(double memoryDepth);
  double powerMemoryBitLeak(double memoryDepth );

  //switch
  void calcSwitch(const SwitchMonitor *sm);
  double powerCrossbar(double width, double inputs, double outputs, double from, double to);
  double powerCrossbarCtrl(double width, double inputs, double outputs);
  double powerCrossbarLeak (double width, double inputs, double outputs);
  
  //output
  double powerOutputCtrl(double width);

  //area

  double areaChannel (double K, double N, double M);
  double areaCrossbar(double Inputs, double Outputs) ;
  double areaInputModule(double Words) ;
  double areaOutputModule(double Outputs);

public:
  Power_Module(Network * net, const Configuration &config);
  ~Power_Module();

  void run();


};
#endif
