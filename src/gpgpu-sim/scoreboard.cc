/*
 * scoreboard.cc
 *
 *  Created on: Aug 10, 2010
 *      Author: inder
 */

#include "scoreboard.h"
#include "shader.h"
#include "../cuda-sim/ptx_sim.h"


//Constructor
Scoreboard::Scoreboard( unsigned sid, unsigned n_warps )
{
	m_sid = sid;
	//Initialize size of table
	reg_table.resize(n_warps);
}

// Print scoreboard contents
void Scoreboard::printContents() 
{
	printf("scoreboard contents (sid=%d): \n", m_sid);
	for(unsigned i=0; i<reg_table.size(); i++) {
		if(reg_table[i].size() == 0 ) continue;
		printf("  wid = %d: ", i);
		std::set<int>::iterator it;
		for ( it=reg_table[i].begin() ; it != reg_table[i].end(); it++ )
			printf("%d ", *it);
		printf("\n");
	}
}


// Mark register as write-pending
void Scoreboard::reserveRegister(unsigned wid, unsigned regnum) 
{
	if( !(reg_table[wid].find(regnum) == reg_table[wid].end()) ){
		printf("Error: trying to reserve an already reserved register (sid=%d, wid=%d, regnum=%d).", m_sid, wid, regnum);
		assert(reg_table[wid].find(regnum) == reg_table[wid].end());
	}

	reg_table[wid].insert(regnum);
}


// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) 
{
	if( !(reg_table[wid].find(regnum) != reg_table[wid].end()) ) {
		printf("Error: trying to release an unreserved register (sid=%d, wid=%d, regnum=%d).", m_sid, wid, regnum);
		assert(reg_table[wid].find(regnum) != reg_table[wid].end());
	}
	reg_table[wid].erase(regnum);
}


// Reserve registers for an instruction
void Scoreboard::reserveRegisters(unsigned wid, const class inst_t* inst) 
{
	// Reserve registers
	if(inst->out[0] > 0) reserveRegister(wid, inst->out[0]);
	if(inst->out[1] > 0) reserveRegister(wid, inst->out[1]);
	if(inst->out[2] > 0) reserveRegister(wid, inst->out[2]);
	if(inst->out[3] > 0) reserveRegister(wid, inst->out[3]);
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t *inst) 
{
	unsigned wid = inst->warp_id();
	if(inst->out[0] > 0) releaseRegister(wid, inst->out[0]);
	if(inst->out[1] > 0) releaseRegister(wid, inst->out[1]);
	if(inst->out[2] > 0) releaseRegister(wid, inst->out[2]);
	if(inst->out[3] > 0) releaseRegister(wid, inst->out[3]);
}

/** 
 * Checks to see if registers used by an instruction are reserved in the scoreboard
 *  
 * @return 
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/ 
bool Scoreboard::checkCollision( unsigned wid, const class inst_t *inst ) 
{
	// Get list of all input and output registers
	std::set<int> inst_regs;

	if(inst->out[0] > 0) inst_regs.insert(inst->out[0]);
	if(inst->out[1] > 0) inst_regs.insert(inst->out[1]);
	if(inst->out[2] > 0) inst_regs.insert(inst->out[2]);
	if(inst->out[3] > 0) inst_regs.insert(inst->out[3]);
	if(inst->in[0] > 0) inst_regs.insert(inst->in[0]);
	if(inst->in[1] > 0) inst_regs.insert(inst->in[1]);
	if(inst->in[2] > 0) inst_regs.insert(inst->in[2]);
	if(inst->in[3] > 0) inst_regs.insert(inst->in[3]);
	if(inst->pred > 0) inst_regs.insert(inst->pred);
	if(inst->ar1 > 0) inst_regs.insert(inst->ar1);
	if(inst->ar2 > 0) inst_regs.insert(inst->ar2);

	// Check for collision, get the intersection of reserved registers and instruction registers
	std::set<int>::iterator it2;
	for ( it2=inst_regs.begin() ; it2 != inst_regs.end(); it2++ )
		if(reg_table[wid].find(*it2) != reg_table[wid].end()) {
			return true;
		}
	return false;
}

bool Scoreboard::pendingWrites(unsigned wid) const
{
	return !reg_table[wid].empty();
}
