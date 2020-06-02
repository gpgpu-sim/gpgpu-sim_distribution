// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

#include <math.h>
#include <string.h>
#include "../abstract_hardware_model.h"
#include "gpu-cache.h"

unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num) {
  /*
   * Set Indexing function from "Pseudo-randomly interleaved memory."
   * Rau, B. R et al.
   * ISCA 1991
   * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
   *
   * equations are corresponding to IPOLY(37) and are adopted from:
   * "Sacat: streaming-aware conflict-avoiding thrashing-resistant gpgpu
   * cache management scheme." Khairy et al. IEEE TPDS 2017.
   *
   * equations for 16 banks are corresponding to IPOLY(5)
   * equations for 32 banks are corresponding to IPOLY(37)
   * equations for 64 banks are corresponding to IPOLY(67)
   * To see all the IPOLY equations for all the degrees, see
   * http://wireless-systems.ece.gatech.edu/6604/handouts/Peterson's%20Table.pdf
   *
   * We generate these equations using GF(2) arithmetic:
   * http://www.ee.unb.ca/cgi-bin/tervo/calc.pl?num=&den=&f=d&e=1&m=1
   *
   * We go through all the strides 128 (10000000), 256 (100000000),...  and
   * do modular arithmetic in GF(2) Then, we create the H-matrix and group
   * each bit together, for more info read the ISCA 1991 paper
   *
   * IPOLY hashing guarantees conflict-free for all 2^n strides which widely
   * exit in GPGPU applications and also show good performance for other
   * strides.
   */
  if (bank_set_num == 16) {
    std::bitset<64> a(higher_bits);
    std::bitset<4> b(index);
    std::bitset<4> new_index(index);

    new_index[0] =
        a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[6] ^ a[4] ^ a[3] ^ a[0] ^ b[0];
    new_index[1] =
        a[12] ^ a[8] ^ a[7] ^ a[6] ^ a[5] ^ a[3] ^ a[1] ^ a[0] ^ b[1];
    new_index[2] = a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[4] ^ a[2] ^ a[1] ^ b[2];
    new_index[3] = a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[5] ^ a[3] ^ a[2] ^ b[3];

    return new_index.to_ulong();

  } else if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits);
    std::bitset<5> b(index);
    std::bitset<5> new_index(index);

    new_index[0] =
        a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[6] ^ a[5] ^ a[3] ^ a[0] ^ b[0];
    new_index[1] = a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[6] ^ a[4] ^
                   a[1] ^ b[1];
    new_index[2] =
        a[14] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[6] ^ a[3] ^ a[2] ^ a[0] ^ b[2];
    new_index[3] =
        a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[7] ^ a[4] ^ a[3] ^ a[1] ^ b[3];
    new_index[4] =
        a[12] ^ a[11] ^ a[10] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ a[2] ^ b[4];
    return new_index.to_ulong();

  } else if (bank_set_num == 64) {
    std::bitset<64> a(higher_bits);
    std::bitset<6> b(index);
    std::bitset<6> new_index(index);

    new_index[0] = a[18] ^ a[17] ^ a[16] ^ a[15] ^ a[12] ^ a[10] ^ a[6] ^ a[5] ^
                   a[0] ^ b[0];
    new_index[1] = a[15] ^ a[13] ^ a[12] ^ a[11] ^ a[10] ^ a[7] ^ a[5] ^ a[1] ^
                   a[0] ^ b[1];
    new_index[2] = a[16] ^ a[14] ^ a[13] ^ a[12] ^ a[11] ^ a[8] ^ a[6] ^ a[2] ^
                   a[1] ^ b[2];
    new_index[3] = a[17] ^ a[15] ^ a[14] ^ a[13] ^ a[12] ^ a[9] ^ a[7] ^ a[3] ^
                   a[2] ^ b[3];
    new_index[4] = a[18] ^ a[16] ^ a[15] ^ a[14] ^ a[13] ^ a[10] ^ a[8] ^ a[4] ^
                   a[3] ^ b[4];
    new_index[5] =
        a[17] ^ a[16] ^ a[15] ^ a[14] ^ a[11] ^ a[9] ^ a[5] ^ a[4] ^ b[5];
    return new_index.to_ulong();
  } else { /* Else incorrect number of channels for the hashing function */
    assert(
        "\nmemory_partition_indexing error: The number of "
        "channels should be "
        "16, 32 or 64 for the hashing IPOLY index function. other banks "
        "numbers are not supported. Generate it by yourself! \n" &&
        0);

    return 0;
  }
}

unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num) {
  return (index) ^ (higher_bits & (bank_set_num - 1));
}

unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num) {
  // Page Address Entropy
  // random selected bits from the page and bank bits
  // similar to
  // Liu, Yuxi, et al. "Get Out of the Valley: Power-Efficient Address
  if (bank_set_num == 32) {
    std::bitset<64> a(higher_bits);
    std::bitset<5> b(index);
    std::bitset<5> new_index(index);
    new_index[0] = a[13] ^ a[10] ^ a[9] ^ a[5] ^ a[0] ^ b[3] ^ b[0] ^ b[0];
    new_index[1] = a[12] ^ a[11] ^ a[6] ^ a[1] ^ b[3] ^ b[2] ^ b[1] ^ b[1];
    new_index[2] = a[14] ^ a[9] ^ a[8] ^ a[7] ^ a[2] ^ b[1] ^ b[2];
    new_index[3] = a[11] ^ a[10] ^ a[8] ^ a[3] ^ b[2] ^ b[3] ^ b[3];
    new_index[4] = a[12] ^ a[9] ^ a[8] ^ a[5] ^ a[4] ^ b[1] ^ b[0] ^ b[4];

    return new_index.to_ulong();
  } else {
    assert(0);
    return 0;
  }
}
