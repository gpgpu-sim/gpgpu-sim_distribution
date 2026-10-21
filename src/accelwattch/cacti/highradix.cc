/*------------------------------------------------------------
 *                              CACTI 6.5
 *         Copyright 2008 Hewlett-Packard Development Corporation
 *                         All Rights Reserved
 *
 * Permission to use, copy, and modify this software and its documentation is
 * hereby granted only under the following terms and conditions.  Both the
 * above copyright notice and this permission notice must appear in all copies
 * of the software, derivative works or modified versions, and any portions
 * thereof, and both notices must appear in supporting documentation.
 *
 * Users of this software agree to the terms and conditions set forth herein, and
 * hereby grant back to Hewlett-Packard Company and its affiliated companies ("HP")
 * a non-exclusive, unrestricted, royalty-free right and license under any changes, 
 * enhancements or extensions  made to the core functions of the software, including 
 * but not limited to those affording compatibility with other hardware or software
 * environments, but excluding applications which incorporate this software.
 * Users further agree to use their best efforts to return to HP any such changes,
 * enhancements or extensions that they make and inform HP of noteworthy uses of
 * this software.  Correspondence should be provided to HP at:
 *
 *                       Director of Intellectual Property Licensing
 *                       Office of Strategy and Technology
 *                       Hewlett-Packard Company
 *                       1501 Page Mill Road
 *                       Palo Alto, California  94304
 *
 * This software may be distributed (but not offered for sale or transferred
 * for compensation) to third parties, provided such third parties agree to
 * abide by the terms and conditions of this notice.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND HP DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL HP 
 * CORPORATION BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *------------------------------------------------------------*/

#include "highradix.h"
#include <iomanip>
using namespace std;

#define MAX_WIRE_SCALE 1

HighRadix::HighRadix(
    double SUB_SWITCH_SZ_,
    double ROWS_,
    double FREQUENCY_, // GHz
    double RADIX_,
    double VC_COUNT_,
    double FLIT_SZ_,
    double AF_,// activity factor
    double DIE_LEN_,//u
    double DIE_HT_,//u
    double INP_BUFF_ENT_, 
    double ROW_BUFF_ENT_, 
    double COL_BUFF_ENT_,
        TechnologyParameter::DeviceType *dt
    ):SUB_SWITCH_SZ(SUB_SWITCH_SZ_), ROWS(ROWS_), FREQUENCY(FREQUENCY_), 
    RADIX(RADIX_), VC_COUNT(VC_COUNT_), FLIT_SZ(FLIT_SZ_), AF(AF_), 
    DIE_LEN(DIE_LEN_), DIE_HT(DIE_HT_), INP_BUFF_ENT(INP_BUFF_ENT_),
    ROW_BUFF_ENT(ROW_BUFF_ENT_), COL_BUFF_ENT(COL_BUFF_ENT_), deviceType(dt)
{
  double area_scale=1;
  double tech_init = 90;
  if (g_ip->F_sz_nm == 65) {
    area_scale*=1;
  }
  else if(g_ip->F_sz_nm == 45) {
    area_scale*=1;
  }
  else if(g_ip->F_sz_nm == 32) {
    area_scale*=2;
  }

  DIE_LEN = sqrt(DIE_LEN_*DIE_HT_/area_scale);
  DIE_HT = DIE_LEN;

  COLUMNS = pow(RADIX/SUB_SWITCH_SZ, 2)/ROWS;
  INP_BUFF_SZ = FLIT_SZ * INP_BUFF_ENT;
  ROW_BUFF_SZ = ROW_BUFF_ENT * FLIT_SZ;
  COL_BUFF_SZ = COL_BUFF_ENT * FLIT_SZ;
  area.set_area(0);
}

void
HighRadix::compute_power()
{
  num_sub = ROWS * COLUMNS;
  //FIXME change cb power to per input
 
  double scale = 1;
  while (true) {
    Wire winit(scale, scale);
    cb = new Crossbar(SUB_SWITCH_SZ, SUB_SWITCH_SZ, FLIT_SZ);
    cb->compute_power();
    if (cb->delay*1e12 < (1/FREQUENCY)*(1e3))
      break;
    else {
      scale+=0.2;
      if (scale > MAX_WIRE_SCALE) break;
      cout << "scale = " << scale << endl;
    }
  }
  cb->power.readOp.dynamic /= SUB_SWITCH_SZ; // crossbar power per message
  scale = 1;

  while (true) {
    Wire winit(scale, scale);
    out_cb = new Crossbar(1, SUB_SWITCH_SZ, FLIT_SZ);
    out_cb->compute_power();
    if (out_cb->delay*1e12 < (1/FREQUENCY)*(1e3))
      break;
    else {
      scale+=0.2;
      if (scale > MAX_WIRE_SCALE) break;
      cout << "scale = " << scale << endl;
    }
  }
  Wire winit;
  out_cb->power.readOp.dynamic /= SUB_SWITCH_SZ; // power per message

  //arbiter initialization
  vc_arb = new MCPAT_Arbiter(VC_COUNT, FLIT_SZ, cb->area.w);
  vc_arb->compute_power();
  c_arb = new MCPAT_Arbiter(COLUMNS, FLIT_SZ, cb->area.w);
  c_arb->compute_power();
  cb_arb = new MCPAT_Arbiter(RADIX/ROWS, FLIT_SZ, cb->area.w);
  cb_arb->compute_power();

  // input buffer, row/column buffer initialization
  inp_buff =  buffer_(FLIT_SZ, INP_BUFF_SZ);
  c_buff = buffer_(FLIT_SZ, COL_BUFF_SZ*2);
  r_buff = buffer_(FLIT_SZ, ROW_BUFF_SZ*2);
  

  // repeated wire initialization
  hor_bus = new Wire(g_ip->wt, DIE_LEN);
  // effective ht of vertical bus (connecting cb to column buffer) in each sub-switch
  double eff_ht = (ROWS * (ROWS +1)/2) * (DIE_HT/ROWS);
  ver_bus = new Wire(g_ip->wt, eff_ht);

  // sub switch includes row buffers, column buffers, vc/crossbar/column arbitration and a 2 stage crossbar traversal
  sub_switch_power();
  power.readOp.dynamic += sub_sw.power.readOp.dynamic * num_sub;
  power.readOp.leakage += sub_sw.power.readOp.leakage * num_sub;

  // input buffer power
  power.readOp.dynamic += 2 /*r&w*/ * inp_buff->power.readOp.dynamic * RADIX;
  power.readOp.leakage += inp_buff->power.readOp.leakage * RADIX;

  // buses
  power.readOp.dynamic += hor_bus->power.readOp.dynamic * FLIT_SZ * SUB_SWITCH_SZ * ROWS;
  power.readOp.leakage += hor_bus->power.readOp.leakage * FLIT_SZ * SUB_SWITCH_SZ * ROWS;
  power.readOp.dynamic += ver_bus->power.readOp.dynamic * FLIT_SZ * COLUMNS * SUB_SWITCH_SZ;
  power.readOp.leakage += ver_bus->power.readOp.leakage * FLIT_SZ * ROWS * COLUMNS * SUB_SWITCH_SZ;

  // To calculate contribution of each component to the total power
  compute_crossbar_power();
  compute_bus_power();
  compute_arb_power();
  compute_buff_power();

  //area 
  sub_sw.area.set_area(sub_sw.area.get_area() + cb->area.get_area());
  sub_sw.area.set_area(sub_sw.area.get_area() + out_cb->area.get_area());  
  sub_sw.area.set_area(sub_sw.area.get_area() + r_buff->area.get_area() * VC_COUNT * SUB_SWITCH_SZ);
  sub_sw.area.set_area(sub_sw.area.get_area() + c_buff->area.get_area() * VC_COUNT * SUB_SWITCH_SZ);

  buff_tot.area.set_area(buff_tot.area.get_area() + inp_buff->area.get_area() * RADIX);
  buff_tot.area.set_area(buff_tot.area.get_area() + VC_COUNT * r_buff->area.get_area() * SUB_SWITCH_SZ * num_sub);
  buff_tot.area.set_area(buff_tot.area.get_area() + VC_COUNT * c_buff->area.get_area() * SUB_SWITCH_SZ * num_sub);

  crossbar_tot.area.set_area(crossbar_tot.area.get_area() + cb->area.get_area() * num_sub);
  crossbar_tot.area.set_area(crossbar_tot.area.get_area() + out_cb->area.get_area() * num_sub);

  wire_tot.area.set_area(hor_bus->area.get_area() * FLIT_SZ * SUB_SWITCH_SZ * ROWS);
  wire_tot.area.set_area(ver_bus->area.get_area() * FLIT_SZ * ROWS * COLUMNS);
}

void HighRadix::compute_crossbar_power()
{
  crossbar_tot.power = cb->power;
  crossbar_tot.power = crossbar_tot.power + out_cb->power;
  crossbar_tot.power.readOp.dynamic *= num_sub;
  crossbar_tot.power.readOp.leakage *= num_sub;
}

void HighRadix::compute_bus_power()
{
  wire_tot.power.readOp.dynamic = hor_bus->power.readOp.dynamic * FLIT_SZ * SUB_SWITCH_SZ * ROWS;
  wire_tot.power.readOp.leakage = hor_bus->power.readOp.leakage * FLIT_SZ * SUB_SWITCH_SZ * ROWS;
  wire_tot.power.readOp.dynamic += ver_bus->power.readOp.dynamic * FLIT_SZ * COLUMNS * SUB_SWITCH_SZ;
  wire_tot.power.readOp.leakage += ver_bus->power.readOp.leakage * FLIT_SZ * ROWS * COLUMNS * SUB_SWITCH_SZ;
}

void HighRadix::compute_arb_power()
{
  arb_tot.power = cb_arb->power;
  arb_tot.power = arb_tot.power + vc_arb->power; // for CB traversal
  arb_tot.power = arb_tot.power + c_arb->power;
  arb_tot.power = arb_tot.power + vc_arb->power; // to the o/p port

  arb_tot.power.readOp.dynamic *= num_sub;
  arb_tot.power.readOp.leakage *= num_sub;
}

void HighRadix::compute_buff_power()
{
  //input buffer read/write
  buff_tot.power.readOp.dynamic = 2 * inp_buff->power.readOp.dynamic * RADIX;
  buff_tot.power.readOp.leakage = inp_buff->power.readOp.leakage * RADIX;

  //row buffer read/write
  buff_tot.power.readOp.dynamic += r_buff->power.readOp.dynamic * 2 * num_sub; 
  buff_tot.power.readOp.leakage += r_buff->power.readOp.leakage * num_sub;

  //column buffer read/write
  buff_tot.power.readOp.dynamic += c_buff->power.readOp.dynamic * 2 * num_sub; 
  buff_tot.power.readOp.leakage += c_buff->power.readOp.leakage * num_sub; 
}

void
HighRadix::sub_switch_power()
{
  // each sub-switch power
  sub_sw.power.readOp.dynamic = sub_sw.power.readOp.dynamic + 
          r_buff->power.readOp.dynamic * 2 /* one read and one write */ * VC_COUNT; 
  sub_sw.power.readOp.leakage = sub_sw.power.readOp.leakage + 
          r_buff->power.readOp.leakage * VC_COUNT; 
  sub_sw.power = sub_sw.power + cb->power;

  sub_sw.power.readOp.dynamic = sub_sw.power.readOp.dynamic + 
          2 * c_buff->power.readOp.dynamic /* one read and one write */ * VC_COUNT; 
  sub_sw.power.readOp.leakage = sub_sw.power.readOp.leakage + 
          c_buff->power.readOp.leakage * VC_COUNT; 
  sub_sw.power = sub_sw.power + out_cb->power;

  // arbiter power
  sub_sw.power = sub_sw.power + cb_arb->power;
  sub_sw.power = sub_sw.power + vc_arb->power; // for CB traversal
  sub_sw.power = sub_sw.power + c_arb->power;
  sub_sw.power = sub_sw.power + vc_arb->power; // to the o/p port
}
  

HighRadix::~HighRadix()
{
  delete inp_buff;
  delete r_buff;
  delete c_buff;
  delete c_arb;
  delete cb_arb;
  delete vc_arb;
  delete out_cb;
}

Mat * HighRadix::buffer_(double block_sz, double sz)
{
  DynamicParameter dyn_p;
  dyn_p.is_tag = false;
  dyn_p.num_subarrays = 1;
  dyn_p.num_mats = 1;
  dyn_p.Ndbl = 1;
  dyn_p.Ndwl = 1;
  dyn_p.Nspd = 1;
  dyn_p.deg_bl_muxing = 1;
  dyn_p.deg_senseamp_muxing_non_associativity = 1;
  dyn_p.Ndsam_lev_1 = 1;
  dyn_p.Ndsam_lev_2 = 1;
  dyn_p.number_addr_bits_mat = 8;
  dyn_p.number_way_select_signals_mat = 1;
  dyn_p.num_act_mats_hor_dir = 1;
  dyn_p.is_dram = false;
  dyn_p.V_b_sense = deviceType->Vdd; // FIXME check power calc.
  dyn_p.ram_cell_tech_type = 
  dyn_p.num_r_subarray = (int) (sz/block_sz);
  dyn_p.num_c_subarray = (int) block_sz;
  dyn_p.num_mats_h_dir = 1;
  dyn_p.num_mats_v_dir = 1;
  dyn_p.num_do_b_subbank = (int)block_sz;
  dyn_p.num_do_b_mat = (int) block_sz;
  dyn_p.num_di_b_mat = (int) block_sz;

  dyn_p.use_inp_params = 1;
  dyn_p.num_wr_ports = 1;
  dyn_p.num_rd_ports = 1;
  dyn_p.num_rw_ports = 0;
  dyn_p.num_se_rd_ports =0;
  dyn_p.out_w = (int) block_sz;


  dyn_p.cell.h = g_tp.sram.b_h + 2 * g_tp.wire_outside_mat.pitch * (dyn_p.num_wr_ports + 
      dyn_p.num_rw_ports - 1 + dyn_p.num_rd_ports);
  dyn_p.cell.w = g_tp.sram.b_w + 2 * g_tp.wire_outside_mat.pitch * (dyn_p.num_rw_ports - 1 + 
      (dyn_p.num_rd_ports - dyn_p.num_se_rd_ports) + 
      dyn_p.num_wr_ports) + g_tp.wire_outside_mat.pitch * dyn_p.num_se_rd_ports;

  Mat *buff = new Mat(dyn_p);
  buff->compute_delays(0);
  buff->compute_power_energy();
  return buff;
}

void HighRadix::print_buffer(Component *c)
{
//  cout << "\tDelay         - " << c->delay * 1e6 << " ns" << endl;
  cout << "\tDynamic Power - " << c->power.readOp.dynamic*1e9 << " nJ" << endl;
  cout << "\tLeakage Power - " << c->power.readOp.leakage*1e3 << " mW" << endl;
  cout << "\tWidth         - " << c->area.w << " u" << endl;
  cout << "\tLength        - " << c->area.h << " u" << endl;
}


void HighRadix::print_router()
{
  cout << "\n\nMCPAT_Router stats:\n";
  cout << "\tNetwork frequency - " << FREQUENCY <<" GHz\n";
  cout << "\tNo. of Virtual channels - " << VC_COUNT << "\n";
  cout << "\tSub-switch size - " << (int)SUB_SWITCH_SZ << endl;
  cout << "\tNo. of rows - " << (int)ROWS << endl;
  cb->print_crossbar();
  out_cb->print_crossbar();
  vc_arb->print_arbiter();
  c_arb->print_arbiter();
  cb_arb->print_arbiter();
//  hor_bus->print_wire();
  cout << "\n\nBuffer stats:\n";
  cout << "\nInput Buffer stats:\n";
  print_buffer (inp_buff);
  cout << "\nRow Buffer stats:\n";
  print_buffer (r_buff);
  cout << "\nColumn Buffer stats:\n";
  print_buffer (c_buff);

  
  cout << "\n\n MCPAT_Router dynamic power (max) = " << power.readOp.dynamic * FREQUENCY * 1e9 << " W\n";
  cout << " MCPAT_Router dynamic power (load - " << AF << ") = " << power.readOp.dynamic * FREQUENCY * 1e9 * AF << " W\n";
  cout << "\n\nDetailed Stats\n";
  cout << "--------------\n";
  cout << "Power dissipated in buses/wires - " << setprecision(3) << 
    wire_tot.power.readOp.dynamic * FREQUENCY * 1e9 << " W";
  cout << "\t" <<setiosflags(ios::fixed) << setprecision(2) <<
          (wire_tot.power.readOp.dynamic/power.readOp.dynamic)*100 << " %\n";
  cout << "Buffer power                    - " << buff_tot.power.readOp.dynamic * 
          FREQUENCY * 1e9 << " W";
  cout << "\t" << 
          (buff_tot.power.readOp.dynamic/power.readOp.dynamic)*100 << " %\n";
  cout << "Crossbar power                  - " << crossbar_tot.power.readOp.dynamic * 
          FREQUENCY * 1e9 << " W";
  cout << "\t" << 
          (crossbar_tot.power.readOp.dynamic/power.readOp.dynamic)*100 << " %\n";
  cout << "Arbiter power                   - " << arb_tot.power.readOp.dynamic * 
          FREQUENCY * 1e9 << " W";
  cout << "\t" << 
          (arb_tot.power.readOp.dynamic/power.readOp.dynamic)*100 << " %\n";
  cout << "Sub-switch power (dynamic)      - " << sub_sw.power.readOp.dynamic * num_sub *
          FREQUENCY * 1e9 << " W";
  cout << "\t" << 
          (sub_sw.power.readOp.dynamic * num_sub/power.readOp.dynamic)*100 << " %\n";
  cout << "Input buffer power (dynamic)    - " << 2 * inp_buff->power.readOp.dynamic * 
          RADIX * FREQUENCY * 1e9 << " W";
  cout << "\t" << 
          (2 * inp_buff->power.readOp.dynamic * RADIX/power.readOp.dynamic)*100 << " %\n";
  cout << "\nLeakage power\n";
  cout << "MCPAT_Router power                    - " << power.readOp.leakage << " W\n";
  cout << "Bus power                       - " <<setprecision(4) <<  wire_tot.power.readOp.leakage << " W\n";
  cout << "Buffer power                    - " << buff_tot.power.readOp.leakage << " W\n";
  cout << "Crossbar power                  - " << crossbar_tot.power.readOp.leakage << " W\n";
  cout << "Arbiter power                   - " << arb_tot.power.readOp.leakage << " W\n";
  cout << "Sub-switch power                - " << sub_sw.power.readOp.leakage << " W" <<endl;

  cout << "\n\nArea Stats\n";
  cout << "Input buffer dimension (mm x mm)- " << inp_buff->area.get_h()*1e-3 << " x " << inp_buff->area.get_w()*1e-3 << endl;
  cout << "Row buffer (mm x mm)            - " << r_buff->area.w*1e-3 << " x " << r_buff->area.h*1e-3 << endl;
  cout << "Col buffer (mm x mm)            - " << c_buff->area.w*1e-3 << " x " << c_buff->area.h*1e-3 << endl;
  cout << "Crossbar area  (mm x mm)        - " << cb->area.w*1e-3 << " x " << cb->area.h*1e-3 << endl;
//  cout << "Wire hor area  (nm x nm)        - " << hor_bus->area.w*1e3 << " x " << hor_bus->area.h*1e3 << endl;
//  cout << "Wire ver area  (nm x nm)        - " << ver_bus->area.w*1e3 << " x " << ver_bus->area.h*1e3 << endl;
  cout << "Wire total                      - " << wire_tot.area.get_area()*1e-6 << " mm2\n";
  cout << "Crossbar total                  - " << crossbar_tot.area.get_area()*1e-6 << " mm2\n";
  cout << "Buff total                      - " << buff_tot.area.get_area()*1e-6 << " mm2\n";
  cout << "Subswitch                       - " << sub_sw.area.get_area()*1e-6 << " mm2\n";
  cout << "Subswitch total                 - " << sub_sw.area.get_area()*num_sub*1e-6 << " mm2\n";

  cout << "Total area                      - " << (wire_tot.area.get_area() + crossbar_tot.area.get_area() +
                                                  buff_tot.area.get_area())*1e-6 << endl;
}

