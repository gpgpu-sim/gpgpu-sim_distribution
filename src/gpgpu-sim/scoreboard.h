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

typedef unsigned op_type;

class Scoreboard
{
	private:
		int sid; // Shader id
		// Table to keep track of write-pending registers
		// Indexed by warp id (wid)
		std::vector< std::set<int> > reg_table;

		void reserveRegister(int wid, int regnum);
		void releaseRegister(int wid, int regnum);

	public:
		Scoreboard( int sid, int n_warps );

		void printContents();

		void reserveRegisters(int wid, void *inst_void);
		void releaseRegisters(int wid, void *inst_void);

		bool checkCollision(int wid, void *inst_void);

};


#endif /* SCOREBOARD_H_ */
