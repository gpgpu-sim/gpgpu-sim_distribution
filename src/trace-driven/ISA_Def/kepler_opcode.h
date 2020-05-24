// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#ifndef KEPLER_OPCODE_H
#define KEPLER_OPCODE_H

#include <string>
#include <unordered_map>
#include "trace_opcode.h"

#define KEPLER_BINART_VERSION 35
#define KEPLER_SHARED_MEMORY_VIRTIAL_ADDRESS_START 0x00007f2c60000000

// TO DO: moving this to a yml or def files

/// Kepler ISA
// see: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
static const std::unordered_map<std::string, OpcodeChar> Kepler_OpcodeMap = {
    // Floating Point 32 Instructions
    {"FFMA", OpcodeChar(OP_FFMA, SP_OP)},
    {"FFMA32I", OpcodeChar(OP_FFMA32I, SP_OP)},
    {"FADD", OpcodeChar(OP_FADD, SP_OP)},
    {"FADD32I", OpcodeChar(OP_FADD32I, SP_OP)},
    {"FCMP", OpcodeChar(OP_FCMP, SP_OP)},
    {"FMUL", OpcodeChar(OP_FMUL, SP_OP)},
    {"FMUL32I", OpcodeChar(OP_FMUL32I, SP_OP)},
    {"FMNMX", OpcodeChar(OP_FMNMX, SP_OP)},
    {"FSWZ", OpcodeChar(OP_FSWZ, SP_OP)},
    {"FSET", OpcodeChar(OP_FSET, SP_OP)},
    {"FSETP", OpcodeChar(OP_FSETP, SP_OP)},
    {"FCHK", OpcodeChar(OP_FCHK, SP_OP)},
    {"RRO", OpcodeChar(OP_RRO, SP_OP)},
    // SFU
    {"MUFU", OpcodeChar(OP_MUFU, SFU_OP)},

    // Double Point Instructions
    {"DFMA", OpcodeChar(OP_DFMA, DP_OP)},
    {"DADD", OpcodeChar(OP_DADD, DP_OP)},
    {"DMUL", OpcodeChar(OP_DMUL, DP_OP)},
    {"DMNMX", OpcodeChar(OP_DMNMX, DP_OP)},
    {"DSET", OpcodeChar(OP_DSET, DP_OP)},
    {"DSETP", OpcodeChar(OP_DSETP, DP_OP)},

    // Integer Instructions
    {"IMAD", OpcodeChar(OP_IMAD, INTP_OP)},
    {"IMADSP", OpcodeChar(OP_IMADSP, INTP_OP)},
    {"IMUL", OpcodeChar(OP_IMUL, INTP_OP)},
    {"IMUL32I", OpcodeChar(OP_IMUL32I, INTP_OP)},
    {"IADD", OpcodeChar(OP_IADD, INTP_OP)},
    {"IADD32I", OpcodeChar(OP_IADD32I, INTP_OP)},
    {"ISUB", OpcodeChar(OP_ISUB, INTP_OP)},
    {"ISCADD", OpcodeChar(OP_ISCADD, INTP_OP)},
    {"ISCADD32I", OpcodeChar(OP_ISCADD32I, INTP_OP)},
    {"ISAD", OpcodeChar(OP_ISAD, INTP_OP)},
    {"IMNMX", OpcodeChar(OP_IMNMX, INTP_OP)},
    {"BFE", OpcodeChar(OP_BFE, INTP_OP)},
    {"BFI", OpcodeChar(OP_BFI, INTP_OP)},
    {"SHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"SHL", OpcodeChar(OP_SHL, INTP_OP)},
    {"SHF", OpcodeChar(OP_SHF, INTP_OP)},
    {"LOP", OpcodeChar(OP_LOP, INTP_OP)},
    {"LOP32I", OpcodeChar(OP_LOP32I, INTP_OP)},
    {"FLO", OpcodeChar(OP_FLO, INTP_OP)},
    {"ISET", OpcodeChar(OP_ISET, INTP_OP)},
    {"ISETP", OpcodeChar(OP_ISETP, INTP_OP)},
    {"ICMP", OpcodeChar(OP_ICMP, INTP_OP)},
    {"POPC", OpcodeChar(OP_POPC, INTP_OP)},

    // Conversion Instructions
    {"F2F", OpcodeChar(OP_F2F, ALU_OP)},
    {"F2I", OpcodeChar(OP_F2I, ALU_OP)},
    {"I2F", OpcodeChar(OP_I2F, ALU_OP)},
    {"I2I", OpcodeChar(OP_I2I, ALU_OP)},

    // Movement Instructions
    {"MOV", OpcodeChar(OP_MOV, ALU_OP)},
    {"MOV32I", OpcodeChar(OP_MOV32I, ALU_OP)},
    {"SEL", OpcodeChar(OP_SEL, ALU_OP)},
    {"PRMT", OpcodeChar(OP_PRMT, ALU_OP)},
    {"SHFL", OpcodeChar(OP_SHFL, ALU_OP)},

    // Predicate Instructions
    {"P2R", OpcodeChar(OP_P2R, ALU_OP)},
    {"R2P", OpcodeChar(OP_R2P, ALU_OP)},
    {"CSET", OpcodeChar(OP_CSET, ALU_OP)},
    {"CSETP", OpcodeChar(OP_CSETP, ALU_OP)},
    {"PSET", OpcodeChar(OP_PSET, ALU_OP)},
    {"PSETP", OpcodeChar(OP_PSETP, ALU_OP)},

    // Texture Instructions
    // For now, we ignore texture loads, consider it as ALU_OP
    {"TEX", OpcodeChar(OP_TEX, ALU_OP)},
    {"TLD", OpcodeChar(OP_TLD, ALU_OP)},
    {"TLD4", OpcodeChar(OP_TLD4, ALU_OP)},
    {"TXQ", OpcodeChar(OP_TXQ, ALU_OP)},

    // Load/Store Instructions
    // For now, we ignore constant loads, consider it as ALU_OP, TO DO
    {"LDC", OpcodeChar(OP_LDC, ALU_OP)},
    // in Kepler, LD is load global so set it to LDG
    {"LD", OpcodeChar(OP_LDG, LOAD_OP)},
    {"LDG", OpcodeChar(OP_LDG, LOAD_OP)},
    {"LDL", OpcodeChar(OP_LDL, LOAD_OP)},
    {"LDS", OpcodeChar(OP_LDS, LOAD_OP)},
    {"LDSLK", OpcodeChar(OP_LDSLK, LOAD_OP)},
    {"ST", OpcodeChar(OP_STG, STORE_OP)},
    {"STL", OpcodeChar(OP_STL, STORE_OP)},
    {"STS", OpcodeChar(OP_STS, STORE_OP)},
    {"STSCUL", OpcodeChar(OP_STSCUL, STORE_OP)},
    {"ATOM", OpcodeChar(OP_ATOM, STORE_OP)},
    {"RED", OpcodeChar(OP_RED, STORE_OP)},
    {"CCTL", OpcodeChar(OP_CCTL, ALU_OP)},
    {"CCTLL", OpcodeChar(OP_CCTLL, ALU_OP)},
    {"MEMBAR", OpcodeChar(OP_MEMBAR, MEMORY_BARRIER_OP)},

    // surface memory instructions
    {"SUCLAMP", OpcodeChar(OP_SUCLAMP, LOAD_OP)},
    {"SUBFM", OpcodeChar(OP_SUBFM, LOAD_OP)},
    {"SUEAU", OpcodeChar(OP_SUEAU, LOAD_OP)},
    {"SULDGA", OpcodeChar(OP_SULDGA, LOAD_OP)},
    {"SUSTGA", OpcodeChar(OP_SUSTGA, STORE_OP)},

    // Control Instructions
    {"BRA", OpcodeChar(OP_BRA, BRANCH_OP)},
    {"BRX", OpcodeChar(OP_BRX, BRANCH_OP)},
    {"JMP", OpcodeChar(OP_JMP, BRANCH_OP)},
    {"JMX", OpcodeChar(OP_JMX, BRANCH_OP)},
    {"CAL", OpcodeChar(OP_CAL, CALL_OPS)},
    {"JCAL", OpcodeChar(OP_JCAL, CALL_OPS)},
    {"RET", OpcodeChar(OP_RET, RET_OPS)},
    {"BRK", OpcodeChar(OP_BRK, RET_OPS)},
    {"CONT", OpcodeChar(OP_CONT, RET_OPS)},
    {"SSY", OpcodeChar(OP_SSY, RET_OPS)},
    {"PBK", OpcodeChar(OP_PBK, RET_OPS)},
    {"PCNT", OpcodeChar(OP_PCNT, RET_OPS)},
    {"PRET", OpcodeChar(OP_PRET, RET_OPS)},
    {"BPT", OpcodeChar(OP_BPT, BRANCH_OP)},
    {"EXIT", OpcodeChar(OP_EXIT, EXIT_OPS)},

    // Miscellaneous Instructions
    {"NOP", OpcodeChar(OP_NOP, ALU_OP)},
    {"S2R", OpcodeChar(OP_S2R, ALU_OP)},
    {"B2R", OpcodeChar(OP_B2R, ALU_OP)},
    {"BAR", OpcodeChar(OP_BAR, BARRIER_OP)},
    {"VOTE", OpcodeChar(OP_VOTE, ALU_OP)},
};

#endif
