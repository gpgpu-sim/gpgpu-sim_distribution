

#ifndef TRACE_OPCODE_H
#define TRACE_OPCODE_H

#include "../abstract_hardware_model.h"
#include <unordered_map>
#include <string>


enum TraceInstrOpcode {
	OP_FADD = 1, OP_FADD32I, OP_FCHK, OP_FFMA32I, OP_FFMA, OP_FMNMX, OP_FMUL, OP_FMUL32I, OP_FSEL, OP_FSET, OP_FSETP,
	OP_FSWZADD, OP_MUFU, OP_HADD2, OP_HADD2_32I, OP_HFMA2, OP_HFMA2_32I, OP_HMUL2, OP_HMUL2_32I, OP_HSET2, OP_HSETP2,
	OP_HMMA, OP_DADD, OP_DFMA, OP_DMUL, OP_DSETP,
	OP_BMSK, OP_BREV, OP_FLO, OP_IABS, OP_IADD, OP_IADD3, OP_IADD32I, OP_IDP, OP_IDP4A, OP_IMAD, OP_IMMA, OP_IMNMX,
	OP_IMUL, OP_IMUL32I, OP_ISCADD, OP_ISCADD32I, OP_ISETP, OP_LEA, OP_LOP, OP_LOP3, OP_LOP32I, OP_POPC, OP_SHF, OP_SHR,
	OP_VABSDIFF, OP_VABSDIFF4,
	OP_F2F, OP_F2I, OP_I2F, OP_I2I, OP_I2IP, OP_FRND, OP_MOV, OP_MOV32I, OP_PRMT, OP_SEL, OP_SGXT, OP_SHFL, OP_PLOP3,
	OP_PSETP, OP_P2R, OP_R2P, OP_LD, OP_LDC, OP_LDG, OP_LDL, OP_LDS, OP_ST, OP_STG, OP_STL, OP_STS, OP_MATCH, OP_QSPC,
	OP_ATOM, OP_ATOMS, OP_ATOMG, OP_RED, OP_CCTL, OP_CCTLL, OP_ERRBAR, OP_MEMBAR, OP_CCTLT,
	OP_TEX, OP_TLD, OP_TLD4,
	OP_TMML, OP_TXD, OP_TXQ, OP_BMOV, OP_BPT, OP_BRA, OP_BREAK, OP_BRX, OP_BSSY, OP_BSYNC, OP_CALL, OP_EXIT, OP_JMP, OP_JMX,
	OP_KILL, OP_NANOSLEEP, OP_RET, OP_RPCMOV, OP_RTT, OP_WARPSYNC, OP_YIELD, OP_B2R, OP_BAR, OP_CS2R, OP_CSMTEST, OP_DEPBAR,
	OP_GETLMEMBASE, OP_LEPC, OP_NOP, OP_PMTRIG, OP_R2B, OP_S2R, OP_SETCTAID,  OP_SETLMEMBASE, OP_VOTE, OP_VOTE_VTG,
	SASS_NUM_OPCODES /* The total number of opcodes. */
};
typedef enum TraceInstrOpcode sass_op_type;

/*
enum uarch_op_t {
   NO_OP=-1,
   ALU_OP=1,
   SFU_OP,
   TENSOR_CORE_OP,
   DP_OP,
   SP_OP,
   INTP_OP,
   ALU_SFU_OP,
   LOAD_OP,
   TENSOR_CORE_LOAD_OP,
   TENSOR_CORE_STORE_OP,
   STORE_OP,
   BRANCH_OP,
   BARRIER_OP,
   MEMORY_BARRIER_OP,
   CALL_OPS,
   RET_OPS
};
typedef enum uarch_op_t op_type;
 */

struct OpcodeChar
{
	OpcodeChar(unsigned m_opcode, unsigned m_opcode_category) {
		opcode = m_opcode;
		opcode_category = m_opcode_category;
	}
	unsigned opcode;
	unsigned opcode_category;
};

///Volta SM_70 ISA
//see: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
static const std::unordered_map<std::string,OpcodeChar> OpcodeMap = {
		//Floating Point 32 Instructions
		{"FADD", OpcodeChar(OP_FADD, SP_OP)},
		{"FADD32I", OpcodeChar(OP_FADD32I, SP_OP)},
		{"FCHK", OpcodeChar(OP_FCHK, SP_OP)},
		{"FFMA32I", OpcodeChar(OP_FFMA32I, SP_OP)},
		{"FFMA", OpcodeChar(OP_FFMA, SP_OP)},
		{"FMNMX", OpcodeChar(OP_FMNMX, SP_OP)},
		{"FMUL", OpcodeChar(OP_FMUL, SP_OP)},
		{"FMUL32I", OpcodeChar(OP_FMUL32I, SP_OP)},
		{"FSEL", OpcodeChar(OP_FSEL, SP_OP)},
		{"FSET", OpcodeChar(OP_FSET, SP_OP)},
		{"FSETP", OpcodeChar(OP_FSETP, SP_OP)},
		{"FSWZADD", OpcodeChar(OP_FSWZADD, SP_OP)},
		{"MUFU", OpcodeChar(OP_MUFU, SP_OP)},

		//Floating Point 16 Instructions
		{"HADD2", OpcodeChar(OP_HADD2, SP_OP)},
		{"HADD2_32I", OpcodeChar(OP_HADD2_32I, SP_OP)},
		{"HFMA2", OpcodeChar(OP_HFMA2, SP_OP)},
		{"HFMA2_32I", OpcodeChar(OP_HFMA2_32I, SP_OP)},
		{"HMUL2", OpcodeChar(OP_HMUL2, SP_OP)},
		{"HMUL2_32I", OpcodeChar(OP_HMUL2_32I, SP_OP)},
		{"HSET2", OpcodeChar(OP_HSET2, SP_OP)},
		{"HSETP2", OpcodeChar(OP_HSETP2, SP_OP)},

		//Tensor Core Instructions
		{"HMMA", OpcodeChar(OP_HMMA, TENSOR_CORE_OP)},

		//Double Point Instructions
		{"DADD", OpcodeChar(OP_DADD, DP_OP)},
		{"DFMA", OpcodeChar(OP_DFMA, DP_OP)},
		{"DMUL", OpcodeChar(OP_DMUL, DP_OP)},
		{"DSETP", OpcodeChar(OP_DSETP, DP_OP)},

		//Integer Instructions
		{"BMSK", OpcodeChar(OP_BMSK, INTP_OP)},
		{"BREV", OpcodeChar(OP_BREV, INTP_OP)},
		{"FLO", OpcodeChar(OP_FLO, INTP_OP)},
		{"IABS", OpcodeChar(OP_IABS, INTP_OP)},
		{"IADD", OpcodeChar(OP_IADD, INTP_OP)},
		{"IADD3", OpcodeChar(OP_IADD3, INTP_OP)},
		{"IADD32I", OpcodeChar(OP_IADD32I, INTP_OP)},
		{"IDP", OpcodeChar(OP_IDP, INTP_OP)},
		{"IDP4A", OpcodeChar(OP_IDP4A, INTP_OP)},
		{"IMAD", OpcodeChar(OP_IMAD, INTP_OP)},
		{"IMMA", OpcodeChar(OP_IMMA, INTP_OP)},
		{"IMNMX", OpcodeChar(OP_IMNMX, INTP_OP)},
		{"IMUL", OpcodeChar(OP_IMUL, INTP_OP)},
		{"IMUL32I", OpcodeChar(OP_IMUL32I, INTP_OP)},
		{"ISCADD", OpcodeChar(OP_ISCADD, INTP_OP)},
		{"ISCADD32I", OpcodeChar(OP_ISCADD32I, INTP_OP)},
		{"ISETP", OpcodeChar(OP_ISETP, INTP_OP)},
		{"LEA", OpcodeChar(OP_LEA, INTP_OP)},
		{"LOP", OpcodeChar(OP_LOP, INTP_OP)},
		{"LOP3", OpcodeChar(OP_LOP3, INTP_OP)},
		{"LOP32I", OpcodeChar(OP_LOP32I, INTP_OP)},
		{"POPC", OpcodeChar(OP_POPC, INTP_OP)},
		{"SHF", OpcodeChar(OP_SHF, INTP_OP)},
		{"SHR", OpcodeChar(OP_SHR, INTP_OP)},
		{"VABSDIFF", OpcodeChar(OP_VABSDIFF, INTP_OP)},
		{"VABSDIFF4", OpcodeChar(OP_VABSDIFF4, INTP_OP)},

		//Conversion Instructions
		{"F2F", OpcodeChar(OP_F2F, ALU_OP)},
		{"F2I", OpcodeChar(OP_F2I, ALU_OP)},
		{"I2F", OpcodeChar(OP_I2F, ALU_OP)},
		{"I2I", OpcodeChar(OP_I2I, ALU_OP)},
		{"I2IP", OpcodeChar(OP_I2IP, ALU_OP)},
		{"FRND", OpcodeChar(OP_FRND, ALU_OP)},

		//Movement Instructions
		{"MOV", OpcodeChar(OP_MOV, ALU_OP)},
		{"MOV32I", OpcodeChar(OP_MOV32I, ALU_OP)},
		{"PRMT", OpcodeChar(OP_PRMT, ALU_OP)},
		{"SEL", OpcodeChar(OP_SEL, ALU_OP)},
		{"SGXT", OpcodeChar(OP_SGXT, ALU_OP)},
		{"SHFL", OpcodeChar(OP_SHFL, ALU_OP)},

		//Predicate Instructions
		{"PLOP3", OpcodeChar(OP_PLOP3, ALU_OP)},
		{"PSETP", OpcodeChar(OP_PSETP, ALU_OP)},
		{"P2R", OpcodeChar(OP_P2R, ALU_OP)},
		{"R2P", OpcodeChar(OP_R2P, ALU_OP)},

		//Load/Store Instructions
		{"LD", OpcodeChar(OP_LD, LOAD_OP)},
		{"LDC", OpcodeChar(OP_LDC, LOAD_OP)},
		{"LDG", OpcodeChar(OP_LDG, LOAD_OP)},
		{"LDL", OpcodeChar(OP_LDL, LOAD_OP)},
		{"LDS", OpcodeChar(OP_LDS, LOAD_OP)},
		{"ST", OpcodeChar(OP_ST, STORE_OP)},
		{"STG", OpcodeChar(OP_STG, STORE_OP)},
		{"STL", OpcodeChar(OP_STL, STORE_OP)},
		{"STS", OpcodeChar(OP_STS, STORE_OP)},
		{"MATCH", OpcodeChar(OP_MATCH, ALU_OP)},
		{"QSPC", OpcodeChar(OP_QSPC, ALU_OP)},
		{"ATOM", OpcodeChar(OP_ATOM, STORE_OP)},
		{"ATOMS", OpcodeChar(OP_ATOMS, STORE_OP)},
		{"ATOMG", OpcodeChar(OP_ATOMG, STORE_OP)},
		{"RED", OpcodeChar(OP_RED, STORE_OP)},
		{"CCTL", OpcodeChar(OP_CCTL, ALU_OP)},
		{"CCTLL", OpcodeChar(OP_CCTLL, ALU_OP)},
		{"ERRBAR", OpcodeChar(OP_ERRBAR, ALU_OP)},
		{"MEMBAR", OpcodeChar(OP_MEMBAR, MEMORY_BARRIER_OP)},
		{"CCTLT", OpcodeChar(OP_CCTLT, ALU_OP)},

		//Texture Instructions
		{"TEX", OpcodeChar(OP_TEX, ALU_OP)},
		{"TLD", OpcodeChar(OP_TLD, ALU_OP)},
		{"TLD4", OpcodeChar(OP_TLD4, ALU_OP)},
		{"TMML", OpcodeChar(OP_TMML, ALU_OP)},
		{"TXD", OpcodeChar(OP_TXD, ALU_OP)},
		{"TXQ", OpcodeChar(OP_TXQ, ALU_OP)},

		//Control Instructions
		{"BMOV", OpcodeChar(OP_BMOV, BRANCH_OP)},
		{"BPT", OpcodeChar(OP_BPT, BRANCH_OP)},
		{"BRA", OpcodeChar(OP_BRA, BRANCH_OP)},
		{"BREAK", OpcodeChar(OP_BREAK, BRANCH_OP)},
		{"BRX", OpcodeChar(OP_BRX, BRANCH_OP)},
		{"BSSY", OpcodeChar(OP_BSSY, BRANCH_OP)},
		{"BSYNC", OpcodeChar(OP_BSYNC, BRANCH_OP)},
		{"CALL", OpcodeChar(OP_CALL, CALL_OPS)},
		{"EXIT", OpcodeChar(OP_EXIT, BRANCH_OP)},
		{"JMP", OpcodeChar(OP_JMP, BRANCH_OP)},
		{"JMX", OpcodeChar(OP_JMX, BRANCH_OP)},
		{"KILL", OpcodeChar(OP_KILL, BRANCH_OP)},
		{"NANOSLEEP", OpcodeChar(OP_NANOSLEEP, BRANCH_OP)},
		{"RET", OpcodeChar(OP_RET, RET_OPS)},
		{"RPCMOV", OpcodeChar(OP_RPCMOV, BRANCH_OP)},
		{"RTT", OpcodeChar(OP_RTT, RET_OPS)},
		{"WARPSYNC", OpcodeChar(OP_WARPSYNC, BRANCH_OP)},
		{"YIELD", OpcodeChar(OP_YIELD, BRANCH_OP)},

		//Miscellaneous Instructions
		{"B2R", OpcodeChar(OP_B2R, ALU_OP)},
		{"BAR", OpcodeChar(OP_BAR, BARRIER_OP)},
		{"CS2R", OpcodeChar(OP_CS2R, ALU_OP)},
		{"CSMTEST", OpcodeChar(OP_CSMTEST, ALU_OP)},
		{"DEPBAR", OpcodeChar(OP_DEPBAR, ALU_OP)},
		{"GETLMEMBASE", OpcodeChar(OP_GETLMEMBASE, ALU_OP)},
		{"LEPC", OpcodeChar(OP_LEPC ,ALU_OP)},
		{"NOP", OpcodeChar(OP_NOP ,ALU_OP)},
		{"PMTRIG", OpcodeChar(OP_PMTRIG, ALU_OP)},
		{"R2B", OpcodeChar(OP_R2B, ALU_OP)},
		{"S2R", OpcodeChar(OP_S2R, ALU_OP)},
		{"SETCTAID", OpcodeChar(OP_SETCTAID, ALU_OP)},
		{"SETLMEMBASE", OpcodeChar(OP_SETLMEMBASE, ALU_OP)},
		{"VOTE", OpcodeChar(OP_VOTE, ALU_OP)},
		{"VOTE_VTG", OpcodeChar(OP_VOTE_VTG, ALU_OP)},

};

#endif
