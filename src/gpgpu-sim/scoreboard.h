/*
 * scoreboard.h
 *
 *  Created on: Aug 10, 2010
 *      Author: inder
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include "assert.h"

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

class Scoreboard {
public:
    Scoreboard( unsigned sid, unsigned n_warps );

    void printContents();

    void reserveRegisters(unsigned wid, const inst_t *inst);
    void releaseRegisters(const warp_inst_t *inst);

    bool checkCollision(unsigned wid, const inst_t *inst);
    bool pendingWrites(unsigned wid) const;
private:
    void reserveRegister(unsigned wid, unsigned regnum);
    void releaseRegister(unsigned wid, unsigned regnum);

    unsigned m_sid;

    // keeps track of pending writes to registers
    // indexed by warp id
    std::vector< std::set<int> > reg_table;
};


#endif /* SCOREBOARD_H_ */
