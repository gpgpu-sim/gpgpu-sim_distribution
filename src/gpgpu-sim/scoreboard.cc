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
Scoreboard::Scoreboard( int sid, int n_warps )
{
	this->sid = sid;
	//Initialize size of table
	reg_table.resize(n_warps);
}

// Print scoreboard contents
void Scoreboard::printContents() {
	printf("scoreboard contents (sid=%d): \n", sid);
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
void Scoreboard::reserveRegister(int wid, int regnum) {
	if( !(reg_table[wid].find(regnum) == reg_table[wid].end()) ){
		printf("Error: trying to reserve an already reserved register (sid=%d, wid=%d, regnum=%d).", sid, wid, regnum);
		assert(reg_table[wid].find(regnum) == reg_table[wid].end());
	}

	reg_table[wid].insert(regnum);
}


// Unmark register as write-pending
void Scoreboard::releaseRegister(int wid, int regnum) {
	if( !(reg_table[wid].find(regnum) != reg_table[wid].end()) ) {
		printf("Error: trying to release an unreserved register (sid=%d, wid=%d, regnum=%d).", sid, wid, regnum);
		assert(reg_table[wid].find(regnum) != reg_table[wid].end());
	}

	reg_table[wid].erase(regnum);
}


// Reserve registers for an instruction
void Scoreboard::reserveRegisters(int wid, void* inst_void) {
	inst_t *inst = (inst_t *) inst_void;

	// Reserve registers
	if(inst->out[0] > 0) reserveRegister(wid, inst->out[0]);
	if(inst->out[1] > 0) reserveRegister(wid, inst->out[1]);
	if(inst->out[2] > 0) reserveRegister(wid, inst->out[2]);
	if(inst->out[3] > 0) reserveRegister(wid, inst->out[3]);
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(int wid, void *inst_void) {
	inst_t *inst = (inst_t *) inst_void;

	if(inst->out[0] > 0) releaseRegister(wid, inst->out[0]);
	if(inst->out[1] > 0) releaseRegister(wid, inst->out[1]);
	if(inst->out[2] > 0) releaseRegister(wid, inst->out[2]);
	if(inst->out[3] > 0) releaseRegister(wid, inst->out[3]);
}

// Checks to see if registers used by an instruction are reserved in the scoreboard
bool Scoreboard::checkCollision(int wid, void *inst_void) {
	inst_t *inst = (inst_t *) inst_void;

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

	/*
	printf("Inst registers: ");
	std::set<int>::iterator it;
	for ( it=inst_regs.begin() ; it != inst_regs.end(); it++ )
		printf("%d ", *it);
	printf("\n");
	*/

	// Check for collision, get the intersection of reserved registers and instruction registers
	//std::set<int> reg_intr;
	std::set<int>::iterator it2;
	for ( it2=inst_regs.begin() ; it2 != inst_regs.end(); it2++ )
		if(reg_table[wid].find(*it2) != reg_table[wid].end()) {
			//reg_intr.insert(*it2);
			return true;
		}

	return false;

	/*
	printf("Intersection registers: ");
	std::set<int>::iterator it3;
	for ( it3=reg_intr.begin() ; it3 != reg_intr.end(); it3++ )
		printf("%d ", *it3);
	printf("\n");
	*/


}
