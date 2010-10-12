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

    void reserveRegisters(const warp_inst_t *inst);
    void releaseRegisters(const warp_inst_t *inst);
    void releaseRegister(unsigned wid, unsigned regnum);

    bool checkCollision(unsigned wid, const inst_t *inst) const;
    bool pendingWrites(unsigned wid) const;
    void printContents() const;
private:
    void reserveRegister(unsigned wid, unsigned regnum);

    unsigned m_sid;

    // keeps track of pending writes to registers
    // indexed by warp id, reg_id => pending write count
    std::vector< std::set<unsigned> > reg_table;
};


#endif /* SCOREBOARD_H_ */
