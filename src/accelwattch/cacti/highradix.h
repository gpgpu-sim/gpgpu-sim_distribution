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

#ifndef __HIGHRADIX__
#define __HIGHRADIX__

#include <iostream>
#include "basic_circuit.h"
#include "component.h"
#include "parameter.h"
#include "assert.h"
#include "cacti_interface.h"
#include "wire.h"
#include "mat.h"
#include "crossbar.h"
#include "arbiter.h"
#include "ROUTER.def"

#define FLIP_FLOP_L 0 //W leakage
#define FLIP_FLOP_D 0 //J dynamic
#define ROUTE_LOGIC_D 0 //J
#define ROUTE_LOGIC_L 0 //W

class HighRadix : public Component
{
  public:
    HighRadix(
    double SUB_SWITCH_SZ_ = DEF_SUB_SWITCH_SZ,
    double ROWS_ = DEF_ROWS,
    double FREQUENCY_ = DEF_FREQUENCY, // GHz
    double RADIX_ = DEF_RADIX,
    double VC_COUNT_ = DEF_VC_COUNT,
    double FLIT_SZ_ = DEF_FLIT_SZ,
    double AF_ = DEF_AF,// activity factor
    double DIE_LEN_ = DEF_DIE_LEN,//u
    double DIE_HT_ = DEF_DIE_HT,//u
    double INP_BUFF_ENT_ = DEF_INP_BUFF_ENT, 
    double ROW_BUFF_ENT_ = DEF_ROW_BUFF_ENT, 
    double COL_BUFF_ENT_ = DEF_COL_BUFF_ENT,
    TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~HighRadix();


// Params
    double SUB_SWITCH_SZ;
    double ROWS;
    double FREQUENCY;// GHz
    double RADIX;
    double VC_COUNT;
    double FLIT_SZ;
    double AF;// activity factor
    double DIE_LEN;//u
    double DIE_HT;//u
    double INP_BUFF_ENT;
    double ROW_BUFF_ENT;
    double COL_BUFF_ENT;



    void print_router();

    double INP_BUFF_SZ;
    double COLUMNS;
    double ROW_BUFF_SZ;
    double COL_BUFF_SZ;
    void compute_power();
    void compute_arb_power();
    void compute_crossbar_power();
    void compute_buff_power();
    void compute_bus_power();
    void print_buffer(Component *r);
    void sub_switch_power();
    Mat * buffer_(double block_sz, double sz);

    Crossbar *cb, *out_cb;
    MCPAT_Arbiter *cb_arb, *vc_arb, *c_arb;
    Mat *inp_buff, *r_buff, *c_buff;
    Component sub_sw;
    Component wire_tot, buff_tot, crossbar_tot, arb_tot;
    Wire *hor_bus, *ver_bus;

  private:
    double min_w_pmos;
    TechnologyParameter::DeviceType *deviceType;
    double num_sub;

};

class Waveguide : public Component
{
  public:
    Waveguide(TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~Waveguide();
};

#endif
