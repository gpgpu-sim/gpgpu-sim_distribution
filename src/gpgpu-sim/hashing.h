// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../option_parser.h"

#ifndef HASHING_H
#define HASHING_H

#include "../abstract_hardware_model.h"
#include "gpu-cache.h"

unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num);

unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num);

unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num);

#endif
