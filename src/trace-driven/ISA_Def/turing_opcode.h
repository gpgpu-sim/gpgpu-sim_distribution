// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#ifndef TURING_OPCODE_H
#define TURING_OPCODE_H

#include <string>
#include <unordered_map>
#include "trace_opcode.h"

#define TURING_BINART_VERSION 75
#define TURING_SHARED_MEMORY_VIRTIAL_ADDRESS_START 0x00007f2c60000000

// TO DO: moving this to a yml or def files

/// Volta SM_70 ISA
// see: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
static const std::unordered_map<std::string, OpcodeChar> Turing_OpcodeMap = {
    // Floating Point 32 Instructions
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
    // SFU
    {"MUFU", OpcodeChar(OP_MUFU, SFU_OP)},

    // Floating Point 16 Instructions
    {"HADD2", OpcodeChar(OP_HADD2, SP_OP)},
    {"HADD2_32I", OpcodeChar(OP_HADD2_32I, SP_OP)},
    {"HFMA2", OpcodeChar(OP_HFMA2, SP_OP)},
    {"HFMA2_32I", OpcodeChar(OP_HFMA2_32I, SP_OP)},
    {"HMUL2", OpcodeChar(OP_HMUL2, SP_OP)},
    {"HMUL2_32I", OpcodeChar(OP_HMUL2_32I, SP_OP)},
    {"HSET2", OpcodeChar(OP_HSET2, SP_OP)},
    {"HSETP2", OpcodeChar(OP_HSETP2, SP_OP)},

    // Tensor Core Instructions
    // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
    {"HMMA", OpcodeChar(OP_HMMA, SPECIALIZED_UNIT_3_OP)},

    // Double Point Instructions
    {"DADD", OpcodeChar(OP_DADD, DP_OP)},
    {"DFMA", OpcodeChar(OP_DFMA, DP_OP)},
    {"DMUL", OpcodeChar(OP_DMUL, DP_OP)},
    {"DSETP", OpcodeChar(OP_DSETP, DP_OP)},

    // Integer Instructions
    {"BMMA", OpcodeChar(OP_BMMA, INTP_OP)},  ////////
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
    {"SHL", OpcodeChar(OP_SHL, INTP_OP)},  //////////
    {"SHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"VABSDIFF", OpcodeChar(OP_VABSDIFF, INTP_OP)},
    {"VABSDIFF4", OpcodeChar(OP_VABSDIFF4, INTP_OP)},

    // Conversion Instructions
    {"F2F", OpcodeChar(OP_F2F, ALU_OP)},
    {"F2I", OpcodeChar(OP_F2I, ALU_OP)},
    {"I2F", OpcodeChar(OP_I2F, ALU_OP)},
    {"I2I", OpcodeChar(OP_I2I, ALU_OP)},
    {"I2IP", OpcodeChar(OP_I2IP, ALU_OP)},
    {"FRND", OpcodeChar(OP_FRND, ALU_OP)},

    // Movement Instructions
    {"MOV", OpcodeChar(OP_MOV, ALU_OP)},
    {"MOV32I", OpcodeChar(OP_MOV32I, ALU_OP)},
    {"MOVM", OpcodeChar(OP_MOVM, ALU_OP)},  // move matrix
    {"PRMT", OpcodeChar(OP_PRMT, ALU_OP)},
    {"SEL", OpcodeChar(OP_SEL, ALU_OP)},
    {"SGXT", OpcodeChar(OP_SGXT, ALU_OP)},
    {"SHFL", OpcodeChar(OP_SHFL, ALU_OP)},

    // Predicate Instructions
    {"PLOP3", OpcodeChar(OP_PLOP3, ALU_OP)},
    {"PSETP", OpcodeChar(OP_PSETP, ALU_OP)},
    {"P2R", OpcodeChar(OP_P2R, ALU_OP)},
    {"R2P", OpcodeChar(OP_R2P, ALU_OP)},

    // Load/Store Instructions
    {"LD", OpcodeChar(OP_LD, LOAD_OP)},
    // For now, we ignore constant loads, consider it as ALU_OP, TO DO
    {"LDC", OpcodeChar(OP_LDC, ALU_OP)},
    {"LDG", OpcodeChar(OP_LDG, LOAD_OP)},
    {"LDL", OpcodeChar(OP_LDL, LOAD_OP)},
    {"LDS", OpcodeChar(OP_LDS, LOAD_OP)},
    {"LDSM", OpcodeChar(OP_LDSM, LOAD_OP)},  //
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

    // Uniform Datapath Instruction
    // UDP unit
    // for more info about UDP, see
    // https://www.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf
    {"R2UR", OpcodeChar(OP_R2UR, SPECIALIZED_UNIT_4_OP)},
    {"S2UR", OpcodeChar(OP_S2UR, SPECIALIZED_UNIT_4_OP)},
    {"UBMSK", OpcodeChar(OP_UBMSK, SPECIALIZED_UNIT_4_OP)},
    {"UBREV", OpcodeChar(OP_UBREV, SPECIALIZED_UNIT_4_OP)},
    {"UCLEA", OpcodeChar(OP_UCLEA, SPECIALIZED_UNIT_4_OP)},
    {"UFLO", OpcodeChar(OP_UFLO, SPECIALIZED_UNIT_4_OP)},
    {"UIADD3", OpcodeChar(OP_UIADD3, SPECIALIZED_UNIT_4_OP)},
    {"UIMAD", OpcodeChar(OP_UIMAD, SPECIALIZED_UNIT_4_OP)},
    {"UISETP", OpcodeChar(OP_UISETP, SPECIALIZED_UNIT_4_OP)},
    {"ULDC", OpcodeChar(OP_ULDC, SPECIALIZED_UNIT_4_OP)},
    {"ULEA", OpcodeChar(OP_ULEA, SPECIALIZED_UNIT_4_OP)},
    {"ULOP", OpcodeChar(OP_ULOP, SPECIALIZED_UNIT_4_OP)},
    {"ULOP3", OpcodeChar(OP_ULOP3, SPECIALIZED_UNIT_4_OP)},
    {"ULOP32I", OpcodeChar(OP_ULOP32I, SPECIALIZED_UNIT_4_OP)},
    {"UMOV", OpcodeChar(OP_UMOV, SPECIALIZED_UNIT_4_OP)},
    {"UP2UR", OpcodeChar(OP_UP2UR, SPECIALIZED_UNIT_4_OP)},
    {"UPLOP3", OpcodeChar(OP_UPLOP3, SPECIALIZED_UNIT_4_OP)},
    {"UPOPC", OpcodeChar(OP_UPOPC, SPECIALIZED_UNIT_4_OP)},
    {"UPRMT", OpcodeChar(OP_UPRMT, SPECIALIZED_UNIT_4_OP)},
    {"UPSETP", OpcodeChar(OP_UPSETP, SPECIALIZED_UNIT_4_OP)},
    {"UR2UP", OpcodeChar(OP_UR2UP, SPECIALIZED_UNIT_4_OP)},
    {"USEL", OpcodeChar(OP_USEL, SPECIALIZED_UNIT_4_OP)},
    {"USGXT", OpcodeChar(OP_USGXT, SPECIALIZED_UNIT_4_OP)},
    {"USHF", OpcodeChar(OP_USHF, SPECIALIZED_UNIT_4_OP)},
    {"USHL", OpcodeChar(OP_USHL, SPECIALIZED_UNIT_4_OP)},
    {"USHR", OpcodeChar(OP_USHR, SPECIALIZED_UNIT_4_OP)},
    {"VOTEU", OpcodeChar(OP_VOTEU, SPECIALIZED_UNIT_4_OP)},

    // Texture Instructions
    // For now, we ignore texture loads, consider it as ALU_OP
    {"TEX", OpcodeChar(OP_TEX, SPECIALIZED_UNIT_2_OP)},
    {"TLD", OpcodeChar(OP_TLD, SPECIALIZED_UNIT_2_OP)},
    {"TLD4", OpcodeChar(OP_TLD4, SPECIALIZED_UNIT_2_OP)},
    {"TMML", OpcodeChar(OP_TMML, SPECIALIZED_UNIT_2_OP)},
    {"TXD", OpcodeChar(OP_TXD, SPECIALIZED_UNIT_2_OP)},
    {"TXQ", OpcodeChar(OP_TXQ, SPECIALIZED_UNIT_2_OP)},

    // Surface Instructions //
    {"SUATOM", OpcodeChar(OP_SUATOM, ALU_OP)},
    {"SULD", OpcodeChar(OP_SULD, ALU_OP)},
    {"SURED", OpcodeChar(OP_SURED, ALU_OP)},
    {"SUST", OpcodeChar(OP_SUST, ALU_OP)},

    // Control Instructions
    // execute branch insts on a dedicated branch unit (SPECIALIZED_UNIT_1)
    {"BMOV", OpcodeChar(OP_BMOV, SPECIALIZED_UNIT_1_OP)},
    {"BPT", OpcodeChar(OP_BPT, SPECIALIZED_UNIT_1_OP)},
    {"BRA", OpcodeChar(OP_BRA, SPECIALIZED_UNIT_1_OP)},
    {"BREAK", OpcodeChar(OP_BREAK, SPECIALIZED_UNIT_1_OP)},
    {"BRX", OpcodeChar(OP_BRX, SPECIALIZED_UNIT_1_OP)},
    {"BRXU", OpcodeChar(OP_BRXU, SPECIALIZED_UNIT_1_OP)},  //
    {"BSSY", OpcodeChar(OP_BSSY, SPECIALIZED_UNIT_1_OP)},
    {"BSYNC", OpcodeChar(OP_BSYNC, SPECIALIZED_UNIT_1_OP)},
    {"CALL", OpcodeChar(OP_CALL, SPECIALIZED_UNIT_1_OP)},
    {"EXIT", OpcodeChar(OP_EXIT, EXIT_OPS)},
    {"JMP", OpcodeChar(OP_JMP, SPECIALIZED_UNIT_1_OP)},
    {"JMX", OpcodeChar(OP_JMX, SPECIALIZED_UNIT_1_OP)},
    {"JMXU", OpcodeChar(OP_JMXU, SPECIALIZED_UNIT_1_OP)},  ///
    {"KILL", OpcodeChar(OP_KILL, SPECIALIZED_UNIT_3_OP)},
    {"NANOSLEEP", OpcodeChar(OP_NANOSLEEP, SPECIALIZED_UNIT_1_OP)},
    {"RET", OpcodeChar(OP_RET, SPECIALIZED_UNIT_1_OP)},
    {"RPCMOV", OpcodeChar(OP_RPCMOV, SPECIALIZED_UNIT_1_OP)},
    {"RTT", OpcodeChar(OP_RTT, SPECIALIZED_UNIT_1_OP)},
    {"WARPSYNC", OpcodeChar(OP_WARPSYNC, SPECIALIZED_UNIT_1_OP)},
    {"YIELD", OpcodeChar(OP_YIELD, SPECIALIZED_UNIT_1_OP)},

    // Miscellaneous Instructions
    {"B2R", OpcodeChar(OP_B2R, ALU_OP)},
    {"BAR", OpcodeChar(OP_BAR, BARRIER_OP)},
    {"CS2R", OpcodeChar(OP_CS2R, ALU_OP)},
    {"CSMTEST", OpcodeChar(OP_CSMTEST, ALU_OP)},
    {"DEPBAR", OpcodeChar(OP_DEPBAR, ALU_OP)},
    {"GETLMEMBASE", OpcodeChar(OP_GETLMEMBASE, ALU_OP)},
    {"LEPC", OpcodeChar(OP_LEPC, ALU_OP)},
    {"NOP", OpcodeChar(OP_NOP, ALU_OP)},
    {"PMTRIG", OpcodeChar(OP_PMTRIG, ALU_OP)},
    {"R2B", OpcodeChar(OP_R2B, ALU_OP)},
    {"S2R", OpcodeChar(OP_S2R, ALU_OP)},
    {"SETCTAID", OpcodeChar(OP_SETCTAID, ALU_OP)},
    {"SETLMEMBASE", OpcodeChar(OP_SETLMEMBASE, ALU_OP)},
    {"VOTE", OpcodeChar(OP_VOTE, ALU_OP)},
    {"VOTE_VTG", OpcodeChar(OP_VOTE_VTG, ALU_OP)},

};

#endif
